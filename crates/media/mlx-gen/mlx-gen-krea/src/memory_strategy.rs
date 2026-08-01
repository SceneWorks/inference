//! Shared memory-strategy contract for the MLX Krea pose-control provider.
//!
//! This module declares only the quality-preserving mechanisms the provider actually executes.
//! Exact peak envelopes remain in SceneWorks' promoted calibration bundle.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, PerComponentBytes, Result as CoreResult,
};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-control-mlx-v4-q4-pose-bounded-decode-512-64";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
/// Exact tile-edge domain admitted by current real-weight evidence. The 384 px candidate is
/// deliberately excluded: the clean 1024² sc-16099 run exceeded the established diffusion-latent
/// maximum-error threshold, so it must not inherit the 512 px calibration.
pub const DECODE_TILE_EDGES: [u32; 1] = [DECODE_TILE_EDGE];

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
    .map_err(|errors| CoreError::Unsupported(errors.join("; ")))
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident
                    | MemoryStrategy::StagedResidency
                    | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                    MemoryStrategy::BoundedAttention
                    | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
                },
                parameters: if strategy == MemoryStrategy::BoundedDecode {
                    MemoryParameterRanges {
                        decode_tile_edges: routes.native_edges().to_vec(),
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    }
                } else {
                    MemoryParameterRanges::default()
                },
            })
            .collect(),
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
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
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
        )),
        asset_facts: asset_facts(spec),
        runtime: MemoryRuntimeSemantics::default(),
    })
}

fn asset_facts(spec: &LoadSpec) -> MemoryAssetFacts {
    let components =
        PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["transformer"], &["vae"])
            .unwrap_or_default();
    let overlay_bytes = spec
        .control
        .as_ref()
        .and_then(|source| match source {
            mlx_gen::WeightsSource::File(path) => std::fs::metadata(path).ok().map(|m| m.len()),
            mlx_gen::WeightsSource::Dir(_) => None,
        })
        .unwrap_or(0);
    MemoryAssetFacts {
        base_bytes: components
            .text_encoder
            .saturating_add(components.dit)
            .saturating_add(components.vae)
            .saturating_add(overlay_bytes),
        conditioning_bytes: components.text_encoder,
        transformer_bytes: components.dit,
        decoder_bytes: components.vae,
        overlay_bytes,
    }
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // The Krea pose-control composition deliberately has no PiD decoder (its heavy bundle is the
        // base VAE plus the pose branch). Reject the flag explicitly instead of letting the residency
        // seam ignore it and execute a native decode under a PiD-labelled request.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD decode is not implemented for pose control",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    false,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_base_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => safety_check(contract, spec.precision, quant, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::ImageToImage,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            overlay: Some("pose-control".to_owned()),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        crate::model::effective_base_quant_tier(spec, provider_id)?,
        context,
        weights_free_cleanup,
    )
}

pub fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_cleanup,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: fn() -> CoreResult<()>,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(KreaControlMemoryScope {
        provider_id,
        decode_routes: decode_routes(provider_id)?,
        geometry: context.geometry,
        memory: contract.generation_memory(&context.selection),
        cleanup,
        finished: false,
    })))
}

struct KreaControlMemoryScope {
    provider_id: &'static str,
    decode_routes: mlx_gen_pid::DecodeRoutes,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    cleanup: fn() -> CoreResult<()>,
    finished: bool,
}

impl KreaControlMemoryScope {
    fn ensure_active(&self) -> CoreResult<()> {
        if self.finished {
            Err(CoreError::Msg(format!(
                "{}: memory-strategy request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }

    fn synchronize_and_release(&mut self) -> CoreResult<()> {
        (self.cleanup)()?;
        self.finished = true;
        Ok(())
    }
}

fn mlx_cleanup() -> CoreResult<()> {
    let barrier = mlx_rs::Array::from(0.0_f32);
    barrier.eval().map_err(mlx_gen::Error::from)?;
    drop(barrier);
    mlx_rs::memory::clear_cache();
    Ok(())
}

fn weights_free_cleanup() -> CoreResult<()> {
    Ok(())
}

impl MemoryRequestScope for KreaControlMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        self.ensure_active()?;
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> CoreResult<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        self.decode_routes
            .validate(false, Some(tile_edge), Some(overlap))
            .map_err(CoreError::Unsupported)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        Err(CoreError::Unsupported(format!(
            "{}: attention chunking is not implemented (requested {chunk_size})",
            self.provider_id
        )))
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        Err(CoreError::Unsupported(format!(
            "{}: transformer windows are not implemented (requested {first_block}/{block_count})",
            self.provider_id
        )))
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.ensure_active()?;
        self.synchronize_and_release()
    }
}

impl Drop for KreaControlMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.synchronize_and_release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategyParameters,
        Precision, Quant, WeightsSource,
    };

    #[test]
    fn prepacked_q8_pose_without_an_override_accepts_only_the_actual_tier() {
        let root = std::env::temp_dir().join(format!(
            "mlx-krea-pose-q8-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        let context_for = |quant| MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 512,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        };

        assert_eq!(
            registered_safety_check(&spec, &contract, &context_for(Some(Quant::Q8))),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q4)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &context_for(wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
        std::fs::remove_dir_all(root).ok();
    }
}
