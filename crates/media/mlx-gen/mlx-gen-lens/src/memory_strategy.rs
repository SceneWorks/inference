//! SC-15800: Lens' measured rung-4 production contract.
//!
//! Dense BF16 is the only advertised tier: on the measured Lens-Turbo request, a one-block text
//! encoder window reduced peak MLX residency from 80.084 GiB to 10.037 GiB. Q4 reduced the isolated
//! conditioning peak but did not reduce the full request peak, so it deliberately remains Missing.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, Result as CoreResult,
    TransformerComponent,
};
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "lens-text-encoder-window-2026-07-31-v1";
pub const TEXT_ENCODER_WINDOW: u32 = 1;

/// The exact load shape for which the measured production rung is executable and beneficial.
pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none()
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let streamable = is_streamable_spec(spec);
    let footprint = crate::registry::component_footprint(spec)?;
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
    ));
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        transformer_window_materialization: streamable,
        ..Default::default()
    };
    if streamable {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters.transformer_window_sizes = vec![TEXT_ENCODER_WINDOW];
        capability.parameters.transformer_window_components =
            vec![TransformerComponent::TextEncoder];
    }
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let Some(calibration) = contract.calibration.as_ref() else {
        return MemorySafetyDecision::Reject {
            reason: format!("{}: no calibration identity declared", contract.provider_id),
        };
    };
    if context.calibration_abi != calibration.abi
        || context.calibration_fingerprint != calibration.fingerprint
    {
        return MemorySafetyDecision::Reject {
            reason: format!("{}: calibration handshake mismatch", contract.provider_id),
        };
    }
    if context.use_pid {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay",
                contract.provider_id
            ),
        };
    }
    if context.selection.tier.precision != precision || context.selection.tier.quant != quant {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: request tier {:?} does not match loaded {precision:?}/{quant:?}",
                contract.provider_id, context.selection.tier
            ),
        };
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return MemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: predicted peak {} exceeds effective budget {}",
                contract.provider_id,
                context.predicted_peak_bytes,
                context.budget.effective_bytes()
            ),
        };
    }
    MemorySafetyDecision::Accept
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(LensMemoryScope {
        provider_id,
        selection: context.selection,
        geometry: context.geometry,
        finished: false,
    })))
}

struct LensMemoryScope {
    provider_id: &'static str,
    selection: mlx_gen::gen_core::MemorySelection,
    geometry: MemoryGeometry,
    finished: bool,
}

impl LensMemoryScope {
    fn finish_inner(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl MemoryRequestScope for LensMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{}x{} exceeds admitted {}x{}x{}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        request.memory = match self.selection.strategy {
            MemoryStrategy::Resident => None,
            MemoryStrategy::BoundedTransformerResidency => Some(GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: self.selection.parameters.transformer_window_size,
                transformer_window_component: Some(self.selection.parameters.window_component()),
                ..Default::default()
            }),
            strategy => {
                return Err(CoreError::Unsupported(format!(
                    "{}: strategy {strategy:?} is not implemented",
                    self.provider_id
                )))
            }
        };
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        Ok(())
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        Ok(())
    }

    fn configure_decode(
        &mut self,
        _tile_edge: u32,
        _overlap: u32,
        _geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "lens: bounded decode is not implemented".into(),
        ))
    }

    fn configure_attention(&mut self, _chunk_size: u32) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "lens: bounded attention is not implemented".into(),
        ))
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        if self.selection.strategy != MemoryStrategy::BoundedTransformerResidency
            || block_count != TEXT_ENCODER_WINDOW
            || first_block >= 24
        {
            return Err(CoreError::Unsupported(format!(
                "{}: invalid text-encoder window ({first_block}, {block_count})",
                self.provider_id
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.finish_inner()
    }
}

impl Drop for LensMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, MemoryStrategySupport, TransformerComponent,
        MEMORY_CALIBRATION_ABI,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, PidWeights, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn spec() -> (std::path::PathBuf, LoadSpec) {
        let root = std::env::temp_dir().join(format!(
            "mlx_gen_lens_sc15800_{}_{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), [0_u8; 8]).unwrap();
        }
        std::fs::write(
            root.join("text_encoder/config.json"),
            r#"{"dtype":"bfloat16"}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    #[test]
    fn dense_sequential_deferred_contract_advertises_only_the_measured_scope() {
        let (root, spec) = spec();
        let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            vec![TEXT_ENCODER_WINDOW]
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            vec![TransformerComponent::TextEncoder]
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn unmeasured_numeric_or_loader_shapes_do_not_advertise_rung_four() {
        let (root, base) = spec();
        let mut specs = Vec::new();
        let mut quantized = base.clone();
        quantized.quantize = Some(Quant::Q4);
        specs.push(quantized);
        let mut eager = base.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        specs.push(eager);
        let mut resident = base;
        resident.offload_policy = OffloadPolicy::Resident;
        specs.push(resident);
        let mut fp32 = specs[0].clone();
        fp32.quantize = None;
        fp32.precision = Precision::Fp32;
        specs.push(fp32);
        let mut adapted = specs[0].clone();
        adapted.quantize = None;
        adapted.adapters.push(AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        specs.push(adapted);
        let mut pid = specs[0].clone();
        pid.quantize = None;
        pid.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(root.join("gemma")),
        });
        specs.push(pid);

        for spec in specs {
            let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
            assert!(contract.conformance_errors().is_empty());
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|capability| &capability.support),
                Some(MemoryStrategySupport::Missing)
            ));
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
        std::fs::remove_dir_all(root).ok();
    }

    fn rung_four_context() -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    transformer_window_size: Some(TEXT_ENCODER_WINDOW),
                    transformer_window_component: Some(TransformerComponent::TextEncoder),
                    ..Default::default()
                },
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 256,
                height: 256,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1_000_000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn selected_contract_scope_reaches_the_generation_request_and_pid_fails_closed() {
        let (root, spec) = spec();
        let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
        let context = rung_four_context();
        let mut scope = begin_request("lens_turbo", &contract, Precision::Bf16, None, &context)
            .unwrap()
            .unwrap();
        let mut request = GenerationRequest {
            width: 256,
            height: 256,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.expect("rung 4 configures request memory");
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(TEXT_ENCODER_WINDOW));
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::TextEncoder)
        );
        // This is a tensor-free contract test. The scope's production Drop performs a Metal barrier,
        // which is exercised by the full macOS suite but is intentionally unavailable in headless CI.
        std::mem::forget(scope);

        let mut pid = context;
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&contract, Precision::Bf16, None, &pid),
            MemorySafetyDecision::Reject { reason } if reason.contains("native VAE decode only")
        ));
        std::fs::remove_dir_all(root).ok();
    }
}
