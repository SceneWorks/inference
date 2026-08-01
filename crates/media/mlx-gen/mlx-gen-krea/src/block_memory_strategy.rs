//! SC-16352: request-scoped memory contract for the four base Krea 2 providers.
//!
//! Pose control has its own contract and seven-block control branch. This module covers the shared
//! 28-block base DiT used by Turbo, Raw, and their edit surfaces.
//!
//! The production domain is one block. Real `krea_2_turbo` weights at 512²/1 step measured the full
//! request (max of conditioning, denoise, decode) against an otherwise-identical Sequential + deferred
//! resident-stack attribution control:
//!
//! | tier / overlay | resident request | window 1 request | reduction |
//! |---|---:|---:|---:|
//! | q4 base | 11.928 GiB | 5.555 GiB | 53.4% |
//! | q8 base | 17.748 GiB | 5.715 GiB | 67.8% |
//! | bf16 base | 28.660 GiB | 8.383 GiB | 70.8% |
//! | q4 LoRA | 12.141 GiB | 5.768 GiB | 52.5% |
//! | q4 LoKr | 13.383 GiB | 7.010 GiB | 47.6% |
//!
//! Every windowed image was byte-identical to its resident control. Low-rank adapters are captured
//! and replayed per materialized block. A dense `.diff`/`.diff_b` patch is excluded at contract build
//! time because it mutates the resident base and cannot be reconstructed from the pristine snapshot.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities,
    MemoryNumericTier, MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-request-peak-block-residency-2026-08-01-v1";

pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch(&spec.adapters)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let streamable = is_streamable_spec(spec);
    let footprint = crate::model::component_footprint(spec)?;
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
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        transformer_window_materialization: streamable,
        ..Default::default()
    };
    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy")
            .support = MemoryStrategySupport::Implemented;
    }
    if streamable {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
        capability.parameters.transformer_window_components = vec![TransformerComponent::Dit];
    }
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: Krea rung 4 is calibrated for native VAE decode, not the PiD overlay",
                contract.provider_id
            )));
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

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, spec.precision, spec.quantize, context)
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
    Ok(Some(Box::new(KreaMemoryScope {
        provider_id,
        selection: context.selection,
        geometry: context.geometry,
        finished: false,
    })))
}

struct KreaMemoryScope {
    provider_id: &'static str,
    selection: mlx_gen::gen_core::MemorySelection,
    geometry: MemoryGeometry,
    finished: bool,
}

impl KreaMemoryScope {
    fn finish_inner(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl MemoryRequestScope for KreaMemoryScope {
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
            MemoryStrategy::StagedResidency => Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            MemoryStrategy::BoundedTransformerResidency => Some(GenerationMemory {
                stage_residency: true,
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
            "krea: bounded decode is not implemented by the base provider".into(),
        ))
    }

    fn configure_attention(&mut self, _chunk_size: u32) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "krea: bounded attention is not implemented".into(),
        ))
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        if self.selection.strategy != MemoryStrategy::BoundedTransformerResidency
            || block_count != TRANSFORMER_WINDOW_SIZE
            || first_block >= 28
        {
            return Err(CoreError::Unsupported(format!(
                "{}: invalid Krea DiT window ({first_block}, {block_count})",
                self.provider_id
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.finish_inner()
    }
}

impl Drop for KreaMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish_inner();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategySupport};
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (std::path::PathBuf, LoadSpec) {
        let root = std::env::temp_dir().join(format!(
            "mlx_gen_krea_sc16352_{}_{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), [0_u8; 8]).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(Quant::Q4);
        (root, spec)
    }

    fn write_diff_patch(path: &std::path::Path) {
        let header = br#"{"transformer_blocks.0.attn.to_q.diff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::with_capacity(8 + header.len() + 4);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn sequential_deferred_snapshot_advertises_the_exact_dit_window() {
        let (root, spec) = fixture();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let staged = contract
            .capability(MemoryStrategy::StagedResidency)
            .unwrap();
        assert_eq!(staged.support, MemoryStrategySupport::Implemented);
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            [TRANSFORMER_WINDOW_SIZE]
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn low_rank_overlay_is_admissible_but_dense_diff_patch_is_not() {
        let (root, mut spec) = fixture();
        let low_rank = root.join("low-rank.safetensors");
        std::fs::write(&low_rank, [0_u8; 8]).unwrap();
        spec.adapters = vec![AdapterSpec::new(low_rank, 1.0, AdapterKind::Lora)];
        assert!(is_streamable_spec(&spec));

        let diff = root.join("dense-diff.safetensors");
        write_diff_patch(&diff);
        spec.adapters = vec![AdapterSpec::new(diff, 1.0, AdapterKind::Lora)];
        assert!(!is_streamable_spec(&spec));
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn non_reopenable_or_wrong_loader_shapes_do_not_advertise_rung_four() {
        let (root, base) = fixture();
        let mut resident = base.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut eager = base.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let file = LoadSpec::new(WeightsSource::File(root.join("single.safetensors")))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        for spec in [resident, eager, file] {
            let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
        std::fs::remove_dir_all(root).ok();
    }
}
