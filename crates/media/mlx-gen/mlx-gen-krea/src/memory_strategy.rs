//! Shared memory-strategy contract for the MLX Krea pose-control provider.
//!
//! This module declares only the quality-preserving mechanisms the provider actually executes.
//! Exact peak envelopes remain in SceneWorks' promoted calibration bundle.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, PerComponentBytes, Result as CoreResult,
};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-control-mlx-v3-q4-pose-bounded-decode-512-384-64";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const DECODE_TILE_EDGES: [u32; 2] = [DECODE_TILE_EDGE, 384];

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
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
                        decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    }
                } else {
                    MemoryParameterRanges::default()
                },
            })
            .collect(),
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

pub fn generation_memory(
    contract: &MemoryProviderContract,
    strategy: &mlx_gen::gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    if strategy.strategy == MemoryStrategy::Resident {
        return None;
    }
    let bounded_decode = contract.engages(strategy.strategy, MemoryStrategy::BoundedDecode);
    Some(GenerationMemory {
        tile_vae_decode: bounded_decode,
        decode_tile_edge: bounded_decode
            .then_some(strategy.parameters.decode_tile_edge)
            .flatten(),
        decode_overlap: bounded_decode
            .then_some(strategy.parameters.decode_overlap)
            .flatten(),
        ..Default::default()
    })
}

pub fn safety_check(
    contract: &MemoryProviderContract,
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

pub fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(KreaControlMemoryScope {
        provider_id,
        geometry: context.geometry,
        memory: generation_memory(contract, &context.selection),
        finished: false,
    })))
}

struct KreaControlMemoryScope {
    provider_id: &'static str,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
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
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
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
        if DECODE_TILE_EDGES.contains(&tile_edge) && overlap == DECODE_OVERLAP {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "{}: unsupported decode tile {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
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
