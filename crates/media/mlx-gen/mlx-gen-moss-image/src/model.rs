//! Generator lifecycle, explicit immutable renderer policy and actual completed receipts.
use crate::{
    loader::{fp32_map, read_weights, Sources},
    memory_strategy,
    pipeline::{check, digest_f32, normal, Sampling},
    text_encoder::{supplied, Encoder},
    transformer::Transformer,
};
use gen_core_moss_image::moss_image::*;
use gen_core_moss_image::moss_image_edit::*;
use mlx_gen::gen_core::{
    self, LoadShape, MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, Modality,
    PerComponentBytes, SizeFloor,
};
use mlx_gen::{
    CancelFlag, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    ModelDescriptor, OffloadPolicy, Precision, Progress, Result,
};
use mlx_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use mlx_rs::Array;
use std::{collections::BTreeMap, sync::Mutex};

fn incremental_peak(peak: u64, owned: u64, committed: u64) -> u64 {
    peak.saturating_sub(owned.min(committed))
}

struct AttemptState<R, P> {
    renderer: Option<(MossImageAcceleration, R)>,
    report: Option<P>,
    apply_reports: Vec<gen_core::AdapterApplyReport>,
}
impl<R, P> AttemptState<R, P> {
    fn begin_request<E>(
        &mut self,
        pins: &[gen_core::PinnedWeightsFile],
        edit_report: &mut Option<E>,
    ) -> Result<()> {
        // Pin authentication can fail before execution enters its cleanup scope.
        *edit_report = None;
        self.begin(pins)
    }
    fn begin(&mut self, pins: &[gen_core::PinnedWeightsFile]) -> Result<()> {
        self.report = None;
        self.apply_reports.clear();
        for pin in pins {
            if let Err(error) = pin.verify_unchanged() {
                self.renderer = None;
                return Err(error.into());
            }
        }
        Ok(())
    }
    fn failed(&mut self) {
        self.report = None;
        self.apply_reports.clear();
        self.renderer = None;
    }
    fn release_for_request(&mut self, mode: MossImageAcceleration, edit: bool) -> bool {
        let release = self
            .renderer
            .as_ref()
            .is_some_and(|(current, _)| edit || *current != mode);
        if release {
            self.renderer = None;
        }
        release
    }
    fn select(
        &mut self,
        mode: MossImageAcceleration,
        build: impl FnOnce() -> Result<R>,
    ) -> Result<&R> {
        if self
            .renderer
            .as_ref()
            .is_none_or(|(current, _)| *current != mode)
        {
            self.renderer = None;
            self.renderer = Some((mode, build()?));
        }
        self.renderer
            .as_ref()
            .map(|(_, renderer)| renderer)
            .ok_or_else(|| Error::Msg("MOSS renderer missing".into()))
    }
}
type State = AttemptState<Transformer, MossImageExecutionReport>;
/// Native MOSS generator. Use this concrete type with the advanced tensor/latent entry point.
/// One lifecycle lock owns validation, mode selection, execution and completed-report publication.
pub struct MossImage {
    variant: MossImageVariant,
    custom_adapter_bytes: u64,
    custom_pins: Vec<gen_core::PinnedWeightsFile>,
    descriptor: ModelDescriptor,
    spec: LoadSpec,
    sources: Sources,
    encoder: Encoder,
    vae: Flux2Vae,
    memory: MemoryProviderContract,
    state: Mutex<State>,
    edit_report: Mutex<Option<MossImageEditExecutionReport>>,
}
fn descriptor(variant: MossImageVariant) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&gen_core::FLUX2_PACKED_LATENT_SPACE),
        control_kinds: None,
        required_components: if variant == MossImageVariant::Base {
            &[MOSS_SHARED_SNAPSHOT_COMPONENT]
        } else {
            &[]
        },
        id: variant.provider_id(),
        family: "moss_image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_lora: true,
            supports_lokr: true,
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_prompt_batch: true,
            samplers: vec!["euler"],
            schedulers: vec!["moss_training_grid"],
            supported_guidance_methods: vec!["cfg"],
            min_size: crate::SIZE_MULTIPLE,
            max_size: u32::MAX,
            max_count: 64,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: crate::SIZE_MULTIPLE,
            },
            mac_only: true,
            ..Default::default()
        },
    }
}
/// Frozen Base registry descriptor.
pub fn descriptor_base() -> ModelDescriptor {
    descriptor(MossImageVariant::Base)
}
/// Frozen SFT registry descriptor, acceleration OFF by default.
pub fn descriptor_sft() -> ModelDescriptor {
    descriptor(MossImageVariant::Sft)
}
/// Load the ordinary Base provider. Base requires explicitly provisioned shared SFT components.
pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(MossImage::load(MossImageVariant::Base, spec)?))
}
/// Load SFT with optional prepared DMD availability. Assets never opt a request into acceleration.
pub fn load_sft(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(MossImage::load(MossImageVariant::Sft, spec)?))
}
impl MossImage {
    pub(crate) fn validate_load_spec(spec: &LoadSpec) -> Result<()> {
        if spec.precision != Precision::Bf16
            || spec.offload_policy != OffloadPolicy::Resident
            || spec.load_shape != LoadShape::EagerMaterialization
        {
            return Err(Error::Unsupported("MOSS supports resident BF16 official loading; select full-FP32 DMD through explicit MOSS controls. Staging/deferred loading is not implemented".into()));
        }
        Ok(())
    }
    /// Load exact frozen components. Optional DMD policies remain explicit per request.
    pub fn load(variant: MossImageVariant, spec: &LoadSpec) -> Result<Self> {
        Self::validate_load_spec(spec)?;
        let sources = Sources::prepare(variant, spec)?;
        Self::from_sources(variant, spec, sources)
    }
    pub(crate) fn from_sources(
        variant: MossImageVariant,
        spec: &LoadSpec,
        sources: Sources,
    ) -> Result<Self> {
        let facts = memory_strategy::facts(&sources)?;
        let custom_adapter_bytes =
            crate::adapters::estimated_bytes(&spec.adapters, &sources.config)?;
        let (memory, load_peak) = memory_strategy::loaded_admission(
            variant,
            facts,
            custom_adapter_bytes,
            sources.load_staging_bytes()?,
        )?;
        let custom_pins = spec
            .adapters
            .iter()
            .map(|adapter| spec.file_pin_for(&adapter.path))
            .collect::<gen_core::Result<Vec<_>>>()?;
        if sources
            .working_set_budget()?
            .is_some_and(|cap| load_peak > cap)
        {
            return Err(Error::Unsupported(format!("MOSS resident load requires approximately {load_peak} bytes including source staging, beyond the admitted MLX working set")));
        }
        let cancel = CancelFlag::new();
        let encoder = Encoder::load(&sources, &cancel)?;
        let vae_weights = read_weights(
            sources
                .ordinary
                .iter()
                .filter(|a| a.relative == "vae/diffusion_pytorch_model.safetensors"),
            |k| {
                k.starts_with("decoder.")
                    || k.starts_with("post_quant_conv.")
                    || k == "bn.running_mean"
                    || k == "bn.running_var"
            },
            &cancel,
        )?;
        let vae = Flux2Vae::from_weights_decode_only(
            &vae_weights,
            &Flux2VaeDecoderConfig {
                block_out_channels: if sources.is_fixture() {
                    [32; 4]
                } else {
                    [96, 192, 384, 384]
                },
            },
        )?;
        vae_weights.materialize_accessed()?;
        if !vae_weights.unused_keys().is_empty() {
            return Err(Error::Unsupported(format!(
                "unconsumed MOSS VAE tensors {:?}",
                vae_weights.unused_keys()
            )));
        }
        sources.ensure_unchanged()?;
        Ok(Self {
            variant,
            custom_adapter_bytes,
            custom_pins,
            descriptor: descriptor(variant),
            spec: spec.clone(),
            sources,
            encoder,
            vae,
            memory,
            edit_report: Mutex::new(None),
            state: Mutex::new(State {
                renderer: None,
                report: None,
                apply_reports: vec![],
            }),
        })
    }
    fn selected_sources(&self, mode: MossImageAcceleration) -> Result<Sources> {
        if self.sources.is_fixture() {
            if mode != MossImageAcceleration::Off {
                return Err(Error::Unsupported(
                    "synthetic test profile has no published DMD assets".into(),
                ));
            }
            self.sources.ensure_unchanged()?;
            return Ok(self.sources.clone());
        }
        Sources::selected(self.variant, &self.spec, mode)
    }
    fn validate_inputs(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> Result<(MossImageResolved, usize)> {
        let (resolved, count) =
            Self::resolve_request_inputs(self.variant, &self.spec, request, inputs)?;
        // Fixture loads keep their already-prepared sources (a synthetic profile has no published
        // DMD selection); real loads re-select the requested acceleration closure.
        let selected = self.selected_sources(resolved.acceleration)?;
        selected.mode_assets(resolved.acceleration)?;
        let _ = selected.request_workspace_bytes(
            request.width,
            request.height,
            count,
            resolved.max_sequence_length,
            request_branches(&resolved),
        )?;
        Ok((resolved, count))
    }
    pub(crate) fn validate_request_inputs(
        variant: MossImageVariant,
        spec: &LoadSpec,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> Result<(MossImageResolved, usize)> {
        let (resolved, count) = Self::resolve_request_inputs(variant, spec, request, inputs)?;
        let selected = Sources::selected(variant, spec, resolved.acceleration)?;
        selected.mode_assets(resolved.acceleration)?;
        let _ = selected.request_workspace_bytes(
            request.width,
            request.height,
            count,
            resolved.max_sequence_length,
            request_branches(&resolved),
        )?;
        Ok((resolved, count))
    }
    fn resolve_request_inputs(
        variant: MossImageVariant,
        _spec: &LoadSpec,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> Result<(MossImageResolved, usize)> {
        let descriptor = descriptor(variant);
        descriptor
            .capabilities
            .validate_request(descriptor.id, request)?;
        let resolved = MossImageResolved::resolve(variant, request)?;
        let count = inputs.validate(request, &resolved)?;
        if count > 64 {
            return Err(Error::Unsupported("MOSS expanded prompt-batch × image-count exceeds the native 64-output resource limit".into()));
        }
        if request
            .memory
            .is_some_and(|m| m != gen_core::GenerationMemory::default())
        {
            return Err(Error::Unsupported("MOSS currently implements resident execution; staging, tiling, chunking, windows and alternate execution policies are unavailable".into()));
        }
        Ok((resolved, count))
    }
    /// Structural request peak for admission, including expanded batches, context, precision assets
    /// and CFG/SLG branch outputs. This is deliberately not a calibration measurement.
    pub fn estimated_request_peak(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
    ) -> Result<u64> {
        let (r, count) = self.validate_inputs(request, inputs)?;
        let branches = if r.acceleration == MossImageAcceleration::Off {
            1 + u32::from(r.guidance != 1.0)
                + u32::from(
                    r.slg
                        .as_ref()
                        .is_some_and(|s| s.scale > 0.0 && !s.layers.is_empty()),
                )
        } else {
            1
        };
        let assets = memory_strategy::selected_facts(
            &self.selected_sources(r.acceleration)?,
            r.acceleration,
        )?;
        self.sources
            .request_workspace_bytes(
                request.width,
                request.height,
                count,
                r.max_sequence_length,
                branches,
            )?
            .checked_add(assets.base_bytes)
            .and_then(|n| n.checked_add(assets.overlay_bytes))
            .and_then(|n| n.checked_add(self.custom_adapter_bytes))
            .ok_or_else(|| Error::Unsupported("MOSS peak overflow".into()))
    }
    /// Incremental live demand above a caller's committed snapshot. Credit is bounded by this
    /// provider's retained materialized buffers and by that snapshot; foreign models earn no credit.
    /// A mode switch releases the previous renderer before encoding, so its owned bytes may be
    /// credited as released capacity. Callers must serialize admission with this generator's use.
    pub fn estimated_incremental_peak_bytes(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        budget: gen_core::MemoryBudget,
    ) -> Result<u64> {
        let peak = self.estimated_request_peak(request, inputs)?;
        let state = self
            .state
            .lock()
            .map_err(|_| Error::Msg("MOSS lifecycle lock poisoned".into()))?;
        Ok(incremental_peak(
            peak,
            self.cached_owned_weight_bytes(&state),
            budget.committed_bytes,
        ))
    }
    fn cached_owned_weight_bytes(&self, state: &State) -> u64 {
        self.encoder
            .resident_bytes()
            .saturating_add(self.memory.asset_facts.decoder_bytes)
            .saturating_add(
                state
                    .renderer
                    .as_ref()
                    .map_or(0, |(_, renderer)| renderer.resident_bytes()),
            )
    }
    fn execute(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        progress: &mut dyn FnMut(Progress),
        edit: Option<MossImageEditInputs<'_>>,
    ) -> Result<MossImageAdvancedOutput> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::Msg("MOSS lifecycle lock poisoned".into()))?;
        let mut edit_report = self
            .edit_report
            .lock()
            .map_err(|_| Error::Msg("MOSS edit receipt lock poisoned".into()))?;
        state.begin_request(&self.custom_pins, &mut edit_report)?;
        let result = self.execute_attempt(
            &mut state,
            &mut edit_report,
            request,
            inputs,
            progress,
            edit,
        );
        if result.is_err() {
            *edit_report = None;
            state.failed();
        }
        if result.is_err() || edit.is_some() {
            self.sources.drain_cache()?;
        }
        result
    }

    /// One attempt under the lifecycle lock: admit, encode, select the renderer, sample, render
    /// the output and publish the completed receipt. Every value the attempt builds lives until it
    /// returns, as the inline attempt body held them.
    fn execute_attempt(
        &self,
        state: &mut State,
        edit_report: &mut Option<MossImageEditExecutionReport>,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        progress: &mut dyn FnMut(Progress),
        edit: Option<MossImageEditInputs<'_>>,
    ) -> Result<MossImageAdvancedOutput> {
        check(&request.cancel)?;
        self.sources.ensure_unchanged()?;
        let (resolved, count, sources, edit_assets) = self.admit(request, inputs, edit)?;
        // Only the selected immutable renderer may overlap conditioning activations.
        if state.release_for_request(resolved.acceleration, edit.is_some()) {
            self.sources.drain_cache()?;
        }
        if inputs.positive.is_none()
            || (resolved.acceleration == MossImageAcceleration::Off
                && resolved.guidance != 1.0
                && inputs.negative.is_none())
        {
            progress(Progress::Loading(
                mlx_gen::gen_core::LoadPhase::EncodingPrompt,
            ));
        }
        let (positive, negative, conditioning) =
            self.encode_conditioning(request, inputs, &resolved)?;
        check(&request.cancel)?;
        let (edit_renderer, edit_references) = self.prepare_renderer(
            state,
            &sources,
            &resolved,
            edit.zip(edit_assets.as_ref()),
            &request.cancel,
            progress,
        )?;
        let (noise, samples) = draw_noise(request, inputs, count)?;
        let assets = sources.mode_assets(resolved.acceleration)?;
        let dataspace = load_dataspace(&assets, &resolved, &request.cancel)?;
        let renderer = match edit_renderer.as_ref() {
            Some(renderer) => renderer,
            None => {
                &state
                    .renderer
                    .as_ref()
                    .ok_or_else(|| Error::Msg("MOSS renderer missing".into()))?
                    .1
            }
        };
        let latents = if let (Some(edit), Some((references, _))) = (edit, edit_references.as_ref())
        {
            crate::edit::EditSampling {
                transformer: renderer,
                resolved: &resolved,
                positive: &positive,
                negative: &negative,
                references,
                use_cache: edit.use_kv_cache,
            }
            .run(&noise, &request.cancel, progress)?
        } else {
            Sampling {
                transformer: renderer,
                resolved: &resolved,
                positive: &positive,
                negative: &negative,
                dataspace: dataspace
                    .as_ref()
                    .map(|(q, codes, _)| (q, codes.as_slice())),
            }
            .run(&noise, &request.cancel, progress)?
        };
        check(&request.cancel)?;
        mlx_rs::transforms::eval([&latents])?;
        let output = self.render_output(&latents, inputs.output, count, request, progress)?;
        let receipts = execution_receipts(&assets, &resolved, dataspace.as_ref(), renderer)?;
        sources.ensure_unchanged()?;
        check(&request.cancel)?;
        let ExecutionReceipts {
            asset_receipts,
            sampling,
            dtypes,
            mut apply_reports,
            adapter_reports,
        } = receipts;
        apply_reports.extend(adapter_reports.iter().cloned());
        #[cfg(feature = "test-support")]
        if self.sources.is_fixture() {
            check(&request.cancel)?;
            // No sealed author receipt is manufactured for generated synthetic source bytes.
            state.report = None;
            state.apply_reports = apply_reports;
            return Ok(output);
        }
        let report = MossImageExecutionReport::new(MossImageExecutionData {
            execution_device: MossImageExecutionDevice::Mlx,
            resolved,
            assets: asset_receipts,
            sampling,
            samples,
            conditioning,
            images_per_prompt: request.count,
            latent_shape: [
                count,
                128,
                request.height as usize / 16,
                request.width as usize / 16,
            ],
            output_kind: inputs.output,
            component_dtypes: dtypes,
            adapter_reports,
        })?;
        check(&request.cancel)?;
        if let (Some(edit), Some(assets), Some((_, references))) =
            (edit, edit_assets.as_ref(), edit_references)
        {
            publish_edit_report(
                edit_report,
                &mut apply_reports,
                report,
                edit,
                assets,
                references,
                &request.cancel,
            )?;
        } else {
            state.report = Some(report);
        }
        state.apply_reports = apply_reports;
        Ok(output)
    }

    /// Validate the request, select its immutable sources and check the structural peak (plus
    /// any edit's extra peak) against the admitted working set.
    #[allow(clippy::type_complexity)]
    fn admit(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        edit: Option<MossImageEditInputs<'_>>,
    ) -> Result<(
        MossImageResolved,
        usize,
        Sources,
        Option<crate::edit::EditAssets>,
    )> {
        let (resolved, count) = self.validate_inputs(request, inputs)?;
        let sources = self.selected_sources(resolved.acceleration)?;
        sources.ensure_unchanged()?;
        let peak = self
            .estimated_request_peak(request, inputs)?
            .checked_add(
                edit.map(|edit| crate::edit::extra_peak(edit, request))
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or_else(|| Error::Unsupported("MOSS edit peak overflow".into()))?;
        let edit_assets = edit
            .map(|_| crate::edit::EditAssets::prepare(&self.spec))
            .transpose()?;
        if let Some(cap) = self.sources.working_set_budget()? {
            if peak > cap {
                return Err(Error::Unsupported(format!("MOSS structural resident peak {peak} exceeds admitted MLX working set {cap}; choose a supported geometry/batch or a larger admitted device")));
            }
        }
        Ok((resolved, count, sources, edit_assets))
    }

    /// Encode (or take the supplied) positive and CFG-negative conditioning per prompt row,
    /// repeated `count` times per row, with each row's receipt.
    #[allow(clippy::type_complexity)]
    fn encode_conditioning(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        resolved: &MossImageResolved,
    ) -> Result<(Vec<Array>, Vec<Array>, Vec<MossImageConditioningReceipt>)> {
        let batch = inputs
            .positive
            .map_or(request.prompt_batch.len().max(1), |e| e.batch);
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        let mut conditioning = Vec::new();
        for row in 0..batch {
            check(&request.cancel)?;
            let (pos, pos_receipt) = if let Some(e) = inputs.positive {
                supplied(e, row)?
            } else {
                let text = if request.prompt_batch.is_empty() {
                    request.prompt.as_str()
                } else {
                    &request.prompt_batch[row]
                };
                self.encoder
                    .encode(text, resolved.max_sequence_length, &request.cancel)?
            };
            let neg = if resolved.acceleration == MossImageAcceleration::Off
                && resolved.guidance != 1.0
            {
                Some(if let Some(e) = inputs.negative {
                    supplied(e, row)?
                } else {
                    let text = inputs
                        .negative_prompts
                        .map(|p| p[row].as_str())
                        .unwrap_or(request.negative_prompt.as_deref().unwrap_or(""));
                    self.encoder
                        .encode(text, resolved.max_sequence_length, &request.cancel)?
                })
            } else {
                None
            };
            for _ in 0..request.count {
                positive.push(pos.clone());
                if let Some((array, _)) = &neg {
                    negative.push(array.clone());
                }
            }
            conditioning.push(MossImageConditioningReceipt {
                positive: pos_receipt,
                negative: neg.map(|(_, receipt)| receipt),
            });
        }
        Ok((positive, negative, conditioning))
    }

    /// An edit builds its own adapted renderer and encodes its references; an ordinary request
    /// selects (building on a mode change) the cached renderer for its acceleration mode.
    #[allow(clippy::type_complexity)]
    fn prepare_renderer(
        &self,
        state: &mut State,
        sources: &Sources,
        resolved: &MossImageResolved,
        edit: Option<(MossImageEditInputs<'_>, &crate::edit::EditAssets)>,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<(
        Option<Transformer>,
        Option<(Vec<Array>, Vec<MossImageEditReferenceReceipt>)>,
    )> {
        if let Some((edit, assets)) = edit {
            let mut renderer = sources.transformer(resolved.acceleration, cancel)?;
            let custom_reports = crate::adapters::apply_pinned(
                &mut renderer,
                self.variant,
                &self.spec.adapters,
                &self.custom_pins,
            )?;
            renderer.custom_reports = custom_reports;
            assets.install(&mut renderer, cancel)?;
            let references = crate::edit::encode_references(sources, edit, cancel)?;
            progress(Progress::ConditioningComplete);
            check(cancel)?;
            return Ok((Some(renderer), Some(references)));
        }
        state.select(resolved.acceleration, || {
            self.sources.drain_cache()?;
            let mut transformer = sources.transformer(resolved.acceleration, cancel)?;
            let custom_reports = crate::adapters::apply_pinned(
                &mut transformer,
                self.variant,
                &self.spec.adapters,
                &self.custom_pins,
            )?;
            transformer.custom_reports = custom_reports;
            Ok(transformer)
        })?;
        Ok((None, None))
    }

    /// The requested output: the finite latents read back, or each image decoded through the VAE.
    fn render_output(
        &self,
        latents: &Array,
        kind: MossImageOutputKind,
        count: usize,
        request: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<MossImageAdvancedOutput> {
        Ok(match kind {
            MossImageOutputKind::Latents => {
                let values = crate::pipeline::logical_f32_readback(latents)?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Msg("nonfinite MOSS latent output".into()));
                }
                MossImageAdvancedOutput::Latents {
                    values,
                    shape: [
                        count,
                        128,
                        request.height as usize / 16,
                        request.width as usize / 16,
                    ],
                }
            }
            MossImageOutputKind::Images => {
                progress(Progress::Decoding);
                let mut images = Vec::new();
                for i in 0..count {
                    check(&request.cancel)?;
                    let packed = mlx_gen::attention::slice_axis(latents, 0, i as i32, 1)?
                        .transpose_axes(&[0, 2, 3, 1])?;
                    let decoded = self.vae.decode_packed_latents(&packed)?;
                    mlx_rs::transforms::eval([&decoded])?;
                    check(&request.cancel)?;
                    let values = decoded
                        .try_as_slice::<f32>()
                        .map_err(|e| Error::Msg(e.to_string()))?;
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(Error::Msg("nonfinite MOSS VAE pixels".into()));
                    }
                    images.push(mlx_gen::image::decoded_to_image(
                        &decoded.transpose_axes(&[0, 3, 1, 2])?,
                    )?);
                    check(&request.cancel)?;
                }
                MossImageAdvancedOutput::Images(images)
            }
        })
    }
}

/// The request's initial noise — the supplied values, or one seeded draw per output — with each
/// output's seed + noise digest receipt.
fn draw_noise(
    request: &GenerationRequest,
    inputs: MossImageAdvancedInputs<'_>,
    count: usize,
) -> Result<(Array, Vec<MossImageSampleReceipt>)> {
    let shape = [
        count as i32,
        128,
        (request.height / 16) as i32,
        (request.width / 16) as i32,
    ];
    let per_image = 128usize
        .checked_mul(request.height as usize / 16)
        .and_then(|n| n.checked_mul(request.width as usize / 16))
        .ok_or_else(|| Error::Unsupported("MOSS noise shape overflow".into()))?;
    let total = per_image
        .checked_mul(count)
        .ok_or_else(|| Error::Unsupported("MOSS noise allocation overflow".into()))?;
    let mut noise_values = Vec::with_capacity(total);
    let mut samples = Vec::new();
    let first_seed = request.seed.unwrap_or_else(mlx_gen::default_seed);
    for index in 0..count {
        check(&request.cancel)?;
        let (values, seed) = if let Some(noise) = inputs.noise {
            (
                noise[index * per_image..(index + 1) * per_image].to_vec(),
                None,
            )
        } else {
            let seed = first_seed.wrapping_add(index as u64);
            (normal(seed, per_image), Some(seed))
        };
        samples.push(MossImageSampleReceipt {
            seed,
            noise_sha256: digest_f32(&values),
        });
        noise_values.extend(values);
    }
    let noise = Array::from_slice(&noise_values, &shape);
    drop(noise_values);
    Ok((noise, samples))
}

/// The DMD dataspace `(latent_mix.Q, step_codes, mix-matrix digest)` an accelerated request
/// samples through; `None` with acceleration off.
fn load_dataspace(
    assets: &[&crate::loader::Asset],
    resolved: &MossImageResolved,
    cancel: &CancelFlag,
) -> Result<Option<(Array, Vec<f32>, String)>> {
    if resolved.acceleration == MossImageAcceleration::Off {
        return Ok(None);
    }
    let weights = read_weights(
        assets
            .iter()
            .copied()
            .filter(|a| a.role == MossImageAssetRole::DmdDataspace),
        |_| true,
        cancel,
    )?;
    let values = fp32_map(&weights)?;
    let q = weights.require("latent_mix.Q")?.clone();
    let codes = values
        .get("step_codes")
        .ok_or_else(|| Error::MissingTensor("step_codes".into()))?
        .clone();
    let matrix = values
        .get("latent_mix.Q")
        .ok_or_else(|| Error::MissingTensor("latent_mix.Q".into()))?;
    Ok(Some((q, codes, digest_f32(matrix))))
}

/// The receipt parts of a completed attempt, before publication.
struct ExecutionReceipts {
    asset_receipts: Vec<MossImageAssetReceipt>,
    sampling: MossImageSamplingReceipt,
    dtypes: BTreeMap<String, MossImageDtype>,
    apply_reports: Vec<gen_core::AdapterApplyReport>,
    adapter_reports: Vec<gen_core::AdapterApplyReport>,
}

/// The assets', sampler's, component dtypes' and adapters' receipts for a completed attempt.
fn execution_receipts(
    assets: &[&crate::loader::Asset],
    resolved: &MossImageResolved,
    dataspace: Option<&(Array, Vec<f32>, String)>,
    renderer: &Transformer,
) -> Result<ExecutionReceipts> {
    let asset_receipts = assets
        .iter()
        .map(|a| a.receipt())
        .collect::<Result<Vec<_>>>()?;
    let sampling = if let Some((_, codes, digest)) = dataspace {
        MossImageSamplingReceipt::Dmd {
            step_codes: codes[..resolved.steps as usize].to_vec(),
            mix_matrix_sha256: digest.clone(),
        }
    } else {
        MossImageSamplingReceipt::Euler {
            sigmas: moss_image_schedule(resolved.steps, resolved.shift, resolved.training_shift)?,
        }
    };
    let mut dtypes: BTreeMap<String, MossImageDtype> = [
        ("transformer", MossImageDtype::Bf16),
        ("text_encoder", MossImageDtype::Bf16),
        ("vae", MossImageDtype::F32),
        ("latents", MossImageDtype::F32),
        ("time_embedding", MossImageDtype::Bf16),
        ("shared_block_modulation", MossImageDtype::Bf16),
        ("video_head_modulation", MossImageDtype::Bf16),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    let apply_reports = renderer.custom_reports.clone();
    // This sealed source receipt records the published DMD installation only.
    let mut adapter_reports = Vec::new();
    if resolved.acceleration != MossImageAcceleration::Off {
        for key in [
            "time_embedding",
            "shared_block_modulation",
            "video_head_modulation",
        ] {
            dtypes.insert(key.into(), MossImageDtype::F32);
        }
        if resolved.acceleration == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
            dtypes.insert("transformer".into(), MossImageDtype::F32);
        } else {
            dtypes.insert("lora_residual".into(), MossImageDtype::F32);
            let path = assets
                .iter()
                .find(|a| a.role == MossImageAssetRole::DmdLora)
                .ok_or_else(|| Error::Msg("adapter receipt source missing".into()))?
                .pin
                .loader_path()
                .to_path_buf();
            adapter_reports.push(gen_core::AdapterApplyReport {
                adapter_path: path,
                applied: 252,
                skipped: vec![],
            });
        }
    }
    Ok(ExecutionReceipts {
        asset_receipts,
        sampling,
        dtypes,
        apply_reports,
        adapter_reports,
    })
}

/// Publish (and validate) a completed edit's receipt, crediting its adapter in `apply_reports`.
fn publish_edit_report(
    edit_report: &mut Option<MossImageEditExecutionReport>,
    apply_reports: &mut Vec<gen_core::AdapterApplyReport>,
    report: MossImageExecutionReport,
    edit: MossImageEditInputs<'_>,
    assets: &crate::edit::EditAssets,
    references: Vec<MossImageEditReferenceReceipt>,
    cancel: &CancelFlag,
) -> Result<()> {
    assets.ensure_unchanged()?;
    check(cancel)?;
    let receipt = edit_report.insert(MossImageEditExecutionReport {
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
    });
    receipt.validate()?;
    apply_reports.push(gen_core::AdapterApplyReport {
        adapter_path: receipt.adapter_path.clone(),
        applied: receipt.applied_projections,
        skipped: vec![],
    });
    Ok(())
}
impl MossImageAdvancedGenerator for MossImage {
    fn generate_moss_advanced(
        &self,
        request: &GenerationRequest,
        inputs: MossImageAdvancedInputs<'_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        self.execute(request, inputs, on_progress, None)
            .map_err(Into::into)
    }
}
impl Generator for MossImage {
    fn generate_moss_image_edit(
        &self,
        request: &GenerationRequest,
        inputs: MossImageEditInputs<'_>,
        progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<MossImageAdvancedOutput> {
        // Clear both receipts even if edit validation rejects before execution starts.
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| gen_core::Error::Unsupported("MOSS lifecycle lock poisoned".into()))?;
            let mut edit_report = self.edit_report.lock().map_err(|_| {
                gen_core::Error::Unsupported("MOSS edit receipt lock poisoned".into())
            })?;
            state
                .begin_request(&self.custom_pins, &mut edit_report)
                .map_err(gen_core::Error::from)?;
        }
        inputs.validate()?;
        let request = edit_request(request)?;
        self.execute(&request, inputs.advanced, progress, Some(inputs))
            .map_err(Into::into)
    }
    fn moss_image_edit_request_peak(
        &self,
        request: &GenerationRequest,
        inputs: MossImageEditInputs<'_>,
    ) -> gen_core::Result<u64> {
        memory_strategy::edit_request_peak(self.variant, &self.spec, request, inputs)
    }
    fn moss_image_edit_execution_report(&self) -> Option<MossImageEditExecutionReport> {
        self.edit_report
            .lock()
            .ok()
            .and_then(|report| report.clone())
    }
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_inputs(request, MossImageAdvancedInputs::default())
            .map(|_| ())
            .map_err(Into::into)
    }
    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        match self.execute(
            request,
            MossImageAdvancedInputs::default(),
            on_progress,
            None,
        )? {
            MossImageAdvancedOutput::Images(images) => Ok(GenerationOutput::Images(images)),
            _ => Err(gen_core::Error::Unsupported(
                "generic MOSS generation requires image output".into(),
            )),
        }
    }
    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.memory)
    }
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        memory_strategy::safety(&self.spec, &self.memory, context)
    }
    fn moss_image_execution_report(&self) -> Option<MossImageExecutionReport> {
        self.state.lock().ok().and_then(|s| s.report.clone())
    }
    fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
        self.state
            .lock()
            .ok()
            .map(|state| self.cached_owned_weight_bytes(&state))
    }
    fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
        self.state
            .lock()
            .map_or_else(|_| Vec::new(), |state| state.apply_reports.clone())
    }
}
fn request_branches(resolved: &MossImageResolved) -> u32 {
    if matches!(resolved.acceleration, MossImageAcceleration::Off) {
        1 + u32::from(resolved.guidance != 1.0)
            + u32::from(
                resolved
                    .slg
                    .as_ref()
                    .is_some_and(|s| s.scale > 0.0 && !s.layers.is_empty()),
            )
    } else {
        1
    }
}
fn footprint(variant: MossImageVariant, spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    let s = Sources::prepare(variant, spec)?;
    let f = memory_strategy::facts(&s)?;
    Ok(PerComponentBytes {
        text_encoder: f.conditioning_bytes,
        dit: f.transformer_bytes
            + f.overlay_bytes
            + crate::adapters::estimated_bytes(&spec.adapters, &s.config)?,
        vae: f.decoder_bytes,
    })
}
fn footprint_base(s: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    footprint(MossImageVariant::Base, s)
}
fn footprint_sft(s: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    footprint(MossImageVariant::Sft, s)
}
mlx_gen::register_generators! {pub(crate) const BASE_REGISTRATION=descriptor_base=>load_base;footprint=footprint_base}
mlx_gen::register_generators! {pub(crate) const SFT_REGISTRATION=descriptor_sft=>load_sft;footprint=footprint_sft}
pub(crate) const BASE_MEMORY: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: "moss_image_base",
    contract: |s| memory_strategy::contract(MossImageVariant::Base, s),
    safety_check: memory_strategy::safety,
};
pub(crate) const SFT_MEMORY: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: "moss_image_sft",
    contract: |s| memory_strategy::contract(MossImageVariant::Sft, s),
    safety_check: memory_strategy::safety,
};

#[cfg(test)]
pub(crate) fn verify_numerical_renderer_transitions(
    mut build: impl FnMut(MossImageAcceleration) -> Transformer,
    mut predict: impl FnMut(&Transformer) -> Vec<f32>,
    mut edit: impl FnMut() -> Vec<f32>,
) {
    let mut state: AttemptState<Transformer, ()> = AttemptState {
        renderer: None,
        report: None,
        apply_reports: vec![],
    };
    let expected_edit = edit();
    // The same release/select boundary used by generate_inner: ordinary -> edit ->
    // full-F32 DMD -> edit -> ordinary, using real numerical tiny transformers.
    let mut ordinary = None;
    for mode in [
        MossImageAcceleration::Off,
        MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        MossImageAcceleration::Off,
    ] {
        state.begin(&[]).unwrap();
        state.release_for_request(mode, false);
        let actual = predict(state.select(mode, || Ok(build(mode))).unwrap());
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
        assert!(state.release_for_request(MossImageAcceleration::Off, true));
        assert!(state.renderer.is_none());
        assert_eq!(edit(), expected_edit);
        assert!(
            state.renderer.is_none(),
            "request-local edit must not become ordinary residency"
        );
    }
}

#[cfg(test)]
mod cpu_tests {
    use super::*;
    #[test]
    fn pin_failure_clears_ordinary_and_edit_receipts_before_execution() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("custom-adapter");
        std::fs::write(&path, b"original").unwrap();
        let pin = gen_core::PinnedWeightsFile::pin(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut state = AttemptState {
            renderer: Some((MossImageAcceleration::Off, ())),
            report: Some("completed ordinary"),
            apply_reports: vec![gen_core::AdapterApplyReport {
                adapter_path: "custom".into(),
                applied: 1,
                skipped: vec![],
            }],
        };
        let mut edit_report = Some("completed edit");
        assert!(state.begin_request(&[pin], &mut edit_report).is_err());
        assert!(state.report.is_none());
        assert!(state.renderer.is_none());
        assert!(state.apply_reports.is_empty());
        assert!(edit_report.is_none());
    }
    #[test]
    fn warm_admission_only_credits_owned_committed_buffers() {
        assert_eq!(incremental_peak(1000, 400, 0), 1000);
        assert_eq!(incremental_peak(1000, 400, 300), 700);
        assert_eq!(incremental_peak(1000, 400, 900), 600);
        assert_eq!(incremental_peak(1000, 0, 900), 1000);
    }
    #[test]
    fn immutable_off_on_off_cache_and_failed_attempt_clear_actual_report() {
        let mut state: AttemptState<(u32, u32), &str> = AttemptState {
            renderer: None,
            report: None,
            apply_reports: vec![],
        };
        let off = MossImageAcceleration::Off;
        let on = MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16);
        let mut loads = 0;
        assert_eq!(
            *state
                .select(off, || {
                    loads += 1;
                    Ok((0, 0))
                })
                .unwrap(),
            (0, 0)
        );
        state.report = Some("completed");
        state.begin(&[]).unwrap();
        assert!(state.report.is_none());
        assert_eq!(
            *state
                .select(on, || {
                    loads += 1;
                    Ok((128, 6))
                })
                .unwrap(),
            (128, 6)
        );
        assert_eq!(
            *state
                .select(off, || {
                    loads += 1;
                    Ok((0, 0))
                })
                .unwrap(),
            (0, 0)
        );
        assert_eq!(loads, 3);
        state.report = Some("prior completion");
        state.begin(&[]).unwrap();
        assert!(matches!(
            state.select(on, || Err(Error::Canceled)),
            Err(Error::Canceled)
        ));
        state.failed();
        assert!(state.report.is_none());
        assert!(state.renderer.is_none());
        assert_eq!(*state.select(off, || Ok((0, 0))).unwrap(), (0, 0));
    }
    #[test]
    fn descriptors_keep_native_batch_cfg_and_transformer_adapters_without_references() {
        for descriptor in [descriptor_base(), descriptor_sft()] {
            assert!(descriptor.capabilities.supports_prompt_batch);
            assert!(descriptor.capabilities.supports_guidance);
            assert!(descriptor.capabilities.supports_negative_prompt);
            assert!(descriptor.capabilities.conditioning.is_empty());
            assert!(descriptor.capabilities.supports_lora);
            assert!(descriptor.capabilities.supports_lokr);
            assert!(descriptor
                .capabilities
                .validate_request(
                    descriptor.id,
                    &GenerationRequest {
                        width: 16,
                        height: 32,
                        guidance: Some(0.0),
                        ..Default::default()
                    }
                )
                .is_ok());
        }
        let request = GenerationRequest {
            steps: Some(8),
            ..Default::default()
        };
        assert_eq!(
            MossImageResolved::resolve(MossImageVariant::Sft, &request)
                .unwrap()
                .acceleration,
            MossImageAcceleration::Off
        );
    }
}

#[cfg(test)]
mod review_lifecycle_tests {
    use super::*;
    #[test]
    fn cached_renderer_request_boundary_rejects_replaced_or_removed_adapter() {
        for remove in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("adapter.safetensors");
            std::fs::write(&path, b"original adapter source").unwrap();
            let pins = [gen_core::PinnedWeightsFile::pin(&path).unwrap()];
            let mut state: AttemptState<u8, ()> = AttemptState {
                renderer: Some((MossImageAcceleration::Off, 7)),
                report: Some(()),
                apply_reports: vec![],
            };
            state.begin(&pins).unwrap();
            assert_eq!(
                *state
                    .select(MossImageAcceleration::Off, || panic!(
                        "warm mode must reuse its renderer"
                    ))
                    .unwrap(),
                7
            );
            state.report = Some(());
            if remove {
                std::fs::remove_file(&path).unwrap();
            } else {
                std::fs::write(&path, b"replaced adapter source").unwrap();
            }
            assert!(
                state.begin(&pins).is_err(),
                "cached renderer accepted changed adapter, remove={remove}"
            );
            assert!(state.report.is_none());
            assert!(state.renderer.is_none());
        }
    }
}
