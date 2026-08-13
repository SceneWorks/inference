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
//! The full-ladder rung-4 key includes `engaged_composition=[Resident, StagedResidency,
//! BoundedDecode, BoundedAttention, BoundedTransformerResidency]`: this loader reopens components
//! between phases, so block streaming is valid only when staged residency is active in the same
//! request.
//!
//! SC-15517 re-ran the complete q4 ladder at 1024²/1 step on the exact Turbo cache revision
//! `d009674080cc1bccf2b629d834c34bf5eccdb723`:
//!
//! | engaged composition | conditioning | denoise | decode | request |
//! |---|---:|---:|---:|---:|
//! | staged | 3.105 GiB | 9.668 GiB | 15.674 GiB | 15.674 GiB |
//! | + 512/64 bounded decode | 3.044 GiB | 9.668 GiB | 12.013 GiB | 12.013 GiB |
//! | + 64 Mi-score attention | 3.380 GiB | 9.409 GiB | 12.013 GiB | 12.013 GiB |
//! | + DiT window 1 | 3.400 GiB | 3.316 GiB | 5.640 GiB | **5.640 GiB** |
//!
//! The attention and block-window arms were pixel-identical to the tiled-decode arm. The independent
//! real-Qwen-VAE 512/64 seam test measured max float delta `1.0857e-2`, mean `2.8614e-4` against the
//! untiled decoder. The final ladder therefore reduces request peak by 64.0% without changing denoise
//! numerics; the bounded-decode comparison retains the existing Qwen spatial blend tolerance.
//!
//! The optional Qwen PiD route uses the student's separately measured 2048/256 output-pixel tiling
//! domain. On exact PiD revision `39d7b0a9003a3fc934d36d8b5658b2d8ea9c1231`, Gemma revision
//! `684c553b5b41a1c835989d89f62f585e6269a7de`, and the same q4 Krea revision, a 768→3072 multitile
//! A/B measured staged+tiled request peak 21.848 GiB and full decode+attention+window peak 15.475 GiB
//! (29.2% lower), with max/mean pixel delta zero. Native and PiD decode domains remain disjoint and
//! are validated against `use_pid` at both admission and request-scope configuration.

use mlx_gen::asset_facts::{
    projected_safetensors_bytes, projected_tensor_headers_bytes, ResidentProjection,
};
#[cfg(test)]
use mlx_gen::gen_core::MemoryGeometry;
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-full-ladder-native-pid-attn64m-window1-2026-08-03-v3";

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(provider_id, [DECODE_TILE_EDGE], DECODE_OVERLAP)
}

#[cfg(test)]
pub(crate) fn is_streamable_spec(provider_id: &str, spec: &LoadSpec) -> CoreResult<bool> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch_for_spec(spec)?
        && plan.load_time_quant_bits.is_none();
    Ok(plan.streamable_transformer)
}

fn resolved_load_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<crate::model::ResolvedLoadPlan> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
        ));
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch_for_spec(spec)?
        && plan.load_time_quant_bits.is_none();
    Ok(plan)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if matches!(spec.weights, WeightsSource::File(_)) {
        crate::model::validate_native_krea_spec(spec, provider_id)
            .map_err(|error| CoreError::Msg(error.to_string()))?;
        let base = mlx_gen::require_base_snapshot(spec, provider_id)?;
        // The native loader is retained and smoke-tested as reopenable, but File has no promoted
        // rung-4 measurement. Keep authorization Missing until source-specific evidence exists.
        return native_memory_strategy_contract_from_spec(provider_id, spec, base, false);
    }
    Ok(memory_strategy_contract_with_plan(provider_id, spec)?.0)
}

pub(crate) fn memory_strategy_contract_with_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<(MemoryProviderContract, crate::model::ResolvedLoadPlan)> {
    let _ = crate::model::component_footprint_for(provider_id, spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
        ));
    };
    let plan = resolved_load_plan(provider_id, spec)?;
    let project = |path: &std::path::Path, select: &dyn Fn(&str) -> bool| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |tensor| {
            if let Some(quant) = spec.quantize.filter(|_| select(&tensor.name)) {
                ResidentProjection::GroupQuantized {
                    bits: quant.bits(),
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
    };
    let selected_text_encoder = crate::model::ENCODER_CONTRACT.source_for_load(spec, root)?;
    let language_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        plan.effective_quant.map(mlx_gen::Quant::bits),
        provider_id,
    )?;
    let mut vision_bytes = 0;
    if matches!(
        provider_id,
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID
    ) {
        let vision = crate::model::ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
        let vision_headers = vision.materialized_vision_tensor_headers(
            &crate::model::VISION_ENCODER_CONTRACT,
            &crate::model::ENCODER_CONTRACT,
        )?;
        vision_bytes =
            projected_tensor_headers_bytes(&vision_headers, |_| ResidentProjection::Stored)?;
    }
    let selected_text_encoder_bytes =
        language_bytes.checked_add(vision_bytes).ok_or_else(|| {
            CoreError::Msg(format!(
                "{provider_id}: selected language plus builtin vision resident byte overflow"
            ))
        })?;
    let components = mlx_gen::PerComponentBytes {
        text_encoder: selected_text_encoder_bytes,
        dit: project(&root.join("transformer"), &|name| {
            crate::convert::is_transformer_quant_target(name)
        })?,
        vae: project(&root.join("vae"), &|_| false)?,
    };
    Ok((
        memory_strategy_contract_with_components(
            provider_id,
            spec,
            components,
            plan.streamable_transformer,
        )?,
        plan,
    ))
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let plan = resolved_load_plan(provider_id, spec)?;
    memory_strategy_contract_with_components(
        provider_id,
        spec,
        Default::default(),
        plan.streamable_transformer,
    )
}

fn memory_strategy_contract_with_components(
    provider_id: &str,
    spec: &LoadSpec,
    components: mlx_gen::PerComponentBytes,
    streamable: bool,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
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
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.asset_facts.base_bytes = components
        .text_encoder
        .saturating_add(components.dit)
        .saturating_add(components.vae);
    contract.asset_facts.conditioning_bytes = components.text_encoder;
    contract.asset_facts.transformer_bytes = components.dit;
    contract.asset_facts.decoder_bytes = components.vae;
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
    let bounded_decode = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .expect("compatibility contract contains every strategy");
    bounded_decode.support = MemoryStrategySupport::Implemented;
    bounded_decode.parameters.decode_tile_edges = routes.published_edges();
    bounded_decode.parameters.decode_overlaps = routes.published_overlaps();

    let bounded_attention = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedAttention)
        .expect("compatibility contract contains every strategy");
    bounded_attention.support = MemoryStrategySupport::Implemented;
    bounded_attention.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
    contract.lifecycle.decode_tiling = true;
    contract.lifecycle.attention_chunking = true;
    contract.pid_decode_routes = Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
        native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: routes.native_edges().to_vec(),
            tile_overlap: DECODE_OVERLAP,
        },
        pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
            tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
            tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
        },
    });
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
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

/// Exact contract for the supported community single-file DiT composition. The native I8 format is
/// dequantized projection-by-projection to bf16, while its scale/descriptor tensors are consumed and
/// dropped; the text encoder and VAE remain sourced from the resident base snapshot.
///
/// Keeping the snapshot and imported forms on the same provider/calibration identity is intentional:
/// the implementation, phase model, and non-transformer components are the same. The promoted-memory
/// evidence matrix does not currently have a load-source axis, however. Consequently a published
/// snapshot (`Dir`) rung-4 cell must not be described as an imported-file measurement merely because
/// this contract can re-open a pinned `File`; the `File` route needs its own real-path measurement
/// before release evidence may claim that cell. The lower-level loader may still be reopened for its
/// story smoke; the public contract must pass `streamable = false` until that evidence exists.
pub(crate) fn native_memory_strategy_contract_from_spec(
    provider_id: &str,
    spec: &LoadSpec,
    base_snapshot_dir: &std::path::Path,
    streamable: bool,
) -> CoreResult<MemoryProviderContract> {
    let stored = |path: &std::path::Path, what: &str| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |_| ResidentProjection::Stored).map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native {what} asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let dit_file = match &spec.weights {
        WeightsSource::File(path) => path,
        WeightsSource::Dir(path) => {
            return Err(CoreError::Msg(format!(
                "{provider_id}: native memory facts require a single-file DiT, not directory {}",
                path.display()
            )))
        }
    };
    let selected_text_encoder =
        crate::model::ENCODER_CONTRACT.source_for_load(spec, base_snapshot_dir)?;
    let expected_language_bits =
        crate::model::native_text_encoder_expected_quant_bits(base_snapshot_dir)?;
    let language_bytes = crate::model::selected_language_resident_bytes(
        &selected_text_encoder,
        expected_language_bits,
        provider_id,
    )?;
    let mut vision_bytes = 0;
    if matches!(
        provider_id,
        crate::model::KREA_2_EDIT_ID | crate::model::KREA_2_TURBO_EDIT_ID
    ) {
        let builtin = crate::model::ENCODER_CONTRACT.validate_source_against_base(
            &WeightsSource::Dir(base_snapshot_dir.join("text_encoder")),
            base_snapshot_dir,
        )?;
        let headers = builtin.materialized_vision_tensor_headers(
            &crate::model::VISION_ENCODER_CONTRACT,
            &crate::model::ENCODER_CONTRACT,
        )?;
        vision_bytes = projected_tensor_headers_bytes(&headers, |_| ResidentProjection::Stored)?;
    }
    let selected_text_encoder_bytes =
        language_bytes.checked_add(vision_bytes).ok_or_else(|| {
            CoreError::Msg(format!(
                "{provider_id}: selected language plus builtin vision resident byte overflow"
            ))
        })?;
    let components = mlx_gen::PerComponentBytes {
        text_encoder: selected_text_encoder_bytes,
        dit: spec.read_file_unchanged_if_prepared(dit_file, |p| {
            native_dit_transformer_bytes(provider_id, p, spec.quantize)
        })?,
        vae: stored(&base_snapshot_dir.join("vae"), "base VAE")?,
    };
    memory_strategy_contract_with_components(provider_id, spec, components, streamable)
}

/// Compatibility shim for the pre-registry native loader. New call sites carry the base snapshot in
/// `LoadSpec::components` and use [`native_memory_strategy_contract_from_spec`].
#[cfg(test)]
pub(crate) fn native_memory_strategy_contract(
    provider_id: &str,
    dit_file: &std::path::Path,
    base_snapshot_dir: &std::path::Path,
) -> CoreResult<MemoryProviderContract> {
    let spec = LoadSpec::new(WeightsSource::File(dit_file.to_path_buf())).with_component(
        mlx_gen::BASE_SNAPSHOT_COMPONENT,
        WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
    );
    native_memory_strategy_contract_from_spec(provider_id, &spec, base_snapshot_dir, false)
}

/// Resident bytes of a community single-file native DiT: I8 projections materialize to bf16, their
/// scale/descriptor companion tensors are consumed and dropped, everything else is stored as-is.
/// The SINGLE projection both native contracts read (the t2i one above and the pose-control one in
/// `crate::memory_strategy`), so the two can never disagree about what a native file costs resident.
pub(crate) fn native_dit_transformer_bytes(
    provider_id: &str,
    dit_file: &std::path::Path,
    quant: Option<mlx_gen::Quant>,
) -> CoreResult<u64> {
    projected_safetensors_bytes(dit_file, |tensor| {
        if tensor.name.ends_with(".weight_scale") || tensor.name.ends_with(".comfy_quant") {
            ResidentProjection::Omit
        } else if let Some(quant) = quant.filter(|_| {
            crate::native_remap::native_dit_key_to_diffusers(&tensor.name)
                .is_some_and(|name| crate::convert::is_transformer_quant_target(&name))
        }) {
            ResidentProjection::GroupQuantized {
                bits: quant.bits(),
                group_size: crate::quant::GROUP_SIZE as usize,
            }
        } else if tensor.dtype == mlx_gen::gen_core::weightsmeta::Dtype::I8 {
            ResidentProjection::Bfloat16
        } else {
            ResidentProjection::Stored
        }
    })
    .map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            decode_routes(contract.provider_id.as_str())?
                .validate(
                    context.use_pid,
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

pub(crate) fn registered_safety_check(
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let is_edit = contract.provider_id.contains("edit");
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    };
    let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
        mode: if is_edit {
            mlx_gen::gen_core::MemoryMode::Edit
        } else {
            mlx_gen::gen_core::MemoryMode::TextToImage
        },
        reference_count: u32::from(is_edit),
        use_pid,
        has_phases: false,
        overlay: None,
    };
    let mut fixtures = vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
    )];
    if contract.pid_decode_routes.is_some()
        && contract.engages(strategy, MemoryStrategy::BoundedDecode)
    {
        fixtures.push(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route(true),
            )?,
        ));
    }
    Ok(fixtures)
}

pub(crate) fn registered_begin_request(
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
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::Krea2Config::turbo().num_layers,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategy,
        MemoryStrategyParameters, MemoryStrategySupport,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_native_i8_safetensors(path: &std::path::Path) {
        let mut header = br#"{
            "model.diffusion_model.proj.weight":{"dtype":"I8","shape":[2,64],"data_offsets":[0,128]},
            "model.diffusion_model.proj.weight_scale":{"dtype":"F32","shape":[2],"data_offsets":[128,136]},
            "model.diffusion_model.proj.comfy_quant":{"dtype":"U8","shape":[2],"data_offsets":[136,138]}
        }"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 138]);
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "mlx_gen_krea_sc16352_{}",
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::model::ENCODER_CONTRACT,
            crate::model::VISION_ENCODER_CONTRACT,
        )
        .expect("validation-complete text encoder fixture");
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
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
    fn identical_fingerprint_is_separated_by_typed_load_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, deferred_spec) = fixture(&tmp);
        let deferred = memory_strategy_contract("krea_2_turbo", &deferred_spec).unwrap();
        let mut eager_spec = deferred_spec;
        eager_spec.load_shape = LoadShape::EagerMaterialization;
        let eager = memory_strategy_contract("krea_2_turbo", &eager_spec).unwrap();
        assert_eq!(
            deferred.calibration.as_ref().unwrap().fingerprint,
            eager.calibration.as_ref().unwrap().fingerprint
        );
        assert_ne!(
            deferred.calibration.as_ref().unwrap().load_shape,
            eager.calibration.as_ref().unwrap().load_shape
        );
        std::fs::remove_dir_all(root).ok();
    }

    fn resident_context(
        contract: &MemoryProviderContract,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        MemoryRunContext {
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
            load_shape: calibration.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
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
        }
    }

    #[test]
    fn prepacked_q4_without_an_override_binds_registration_to_the_actual_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, mut spec) = fixture(&tmp);
        spec.quantize = None;
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            registered_safety_check(
                &spec,
                &contract,
                &resident_context(&contract, Some(Quant::Q4))
            ),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q8)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &resident_context(&contract, wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rung_four_declares_and_engages_staged_residency_in_the_same_request() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn base_family_declares_route_exact_decode_attention_domains() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        let routes = decode_routes("krea_2_turbo").unwrap();
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            decode.parameters.decode_tile_edges,
            routes.published_edges()
        );
        assert_eq!(
            decode.parameters.decode_overlaps,
            routes.published_overlaps()
        );
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );

        let pid_spec = spec.clone().with_pid(
            WeightsSource::File(root.join("pid.safetensors")),
            WeightsSource::Dir(root.clone()),
        );
        let pid_contract = memory_strategy_contract("krea_2_turbo", &pid_spec).unwrap();
        assert!(pid_contract.engages(
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedDecode
        ));

        let mut native_attention = resident_context(&pid_contract, Some(Quant::Q4));
        native_attention.selection.strategy = MemoryStrategy::BoundedAttention;
        native_attention.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        assert_eq!(
            safety_check(
                &pid_contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &native_attention
            ),
            MemorySafetyDecision::Accept,
            "loading PiD must not strip cumulative native Qwen bounded decode from use_pid=false"
        );

        let mut pid_attention = resident_context(&pid_contract, Some(Quant::Q4));
        pid_attention.use_pid = true;
        pid_attention.selection.strategy = MemoryStrategy::BoundedAttention;
        pid_attention.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(mlx_gen_pid::DecodeRoutes::pid_edges()[0]),
            decode_overlap: Some(mlx_gen_pid::DecodeRoutes::pid_overlap()),
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        assert_eq!(
            safety_check(
                &pid_contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &pid_attention
            ),
            MemorySafetyDecision::Accept,
            "PiD must combine its own measured decode domain with bounded DiT attention"
        );

        let mut pid_window = pid_attention.clone();
        pid_window.selection.strategy = MemoryStrategy::BoundedTransformerResidency;
        pid_window.selection.parameters.transformer_window_size = Some(TRANSFORMER_WINDOW_SIZE);
        pid_window.selection.parameters.transformer_window_component =
            Some(TransformerComponent::Dit);
        assert_eq!(
            safety_check(&pid_contract, Precision::Bf16, Some(Quant::Q4), &pid_window),
            MemorySafetyDecision::Accept,
            "PiD decode tiling, bounded DiT attention, and block residency are independently verified request-scoped mechanisms"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sequential_deferred_snapshot_advertises_the_exact_dit_window() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
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
    fn missing_required_component_directory_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        std::fs::remove_dir_all(root.join("text_encoder")).unwrap();
        assert!(memory_strategy_contract("krea_2_turbo", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("krea_2_turbo", &spec).is_ok());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn native_required_file_rejects_missing_empty_and_corrupt_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        let missing = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("native DiT asset facts"), "{missing}");
        std::fs::write(&native, []).unwrap();
        let empty = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(empty.contains("native DiT asset facts"), "{empty}");
        std::fs::write(&native, b"corrupt").unwrap();
        let corrupt = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(corrupt.contains("native DiT asset facts"), "{corrupt}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn imported_base_component_inventory_rejects_missing_empty_and_corrupt_files() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        write_native_i8_safetensors(&native);
        let spec = LoadSpec::new(WeightsSource::File(native)).with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(root.clone()),
        );

        for component in ["text_encoder", "vae"] {
            let file = root.join(component).join("model.safetensors");
            for (case, replacement) in [
                ("empty", Some(Vec::new())),
                ("corrupt", Some(b"corrupt".to_vec())),
                ("missing", None),
            ] {
                match replacement {
                    Some(bytes) => std::fs::write(&file, bytes).unwrap(),
                    None => std::fs::remove_file(&file).unwrap(),
                }
                let error = memory_strategy_contract("krea_2_turbo", &spec)
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(component) || error.contains("safetensors"),
                    "{component}/{case}: {error}"
                );
                if component == "text_encoder" {
                    gen_core_testkit::write_encoder_contract_fixture(
                        &root.join("text_encoder"),
                        crate::model::ENCODER_CONTRACT,
                    )
                    .expect("restore validation-complete text encoder fixture");
                } else {
                    write_minimal_safetensors(&file);
                }
            }
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn low_rank_overlay_is_admissible_but_dense_diff_patch_is_not() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, mut spec) = fixture(&tmp);
        let low_rank = root.join("low-rank.safetensors");
        std::fs::write(&low_rank, [0_u8; 8]).unwrap();
        spec.adapters = vec![AdapterSpec::new(low_rank, 1.0, AdapterKind::Lora)];
        assert!(is_streamable_spec("krea_2_turbo", &spec).unwrap());

        let diff = root.join("dense-diff.safetensors");
        write_diff_patch(&diff);
        spec.adapters = vec![AdapterSpec::new(diff, 1.0, AdapterKind::Lora)];
        assert!(!is_streamable_spec("krea_2_turbo", &spec).unwrap());
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
    fn rung_four_resolves_load_time_quantization_instead_of_the_override_presence() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);

        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let mut packed = spec.clone();
            packed.quantize = Some(quant);
            assert!(
                is_streamable_spec("krea_2_turbo", &packed).unwrap(),
                "a matching prepacked Q{bits} override is a no-op and must remain streamable"
            );
            let (contract, plan) =
                memory_strategy_contract_with_plan("krea_2_turbo", &packed).unwrap();
            assert!(contract.lifecycle.transformer_window_materialization);
            assert_eq!(plan.effective_quant, Some(quant));
            assert_eq!(plan.load_time_quant_bits, None);
            assert!(plan.streamable_transformer);
        }

        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        assert!(
            !is_streamable_spec("krea_2_turbo", &spec).unwrap(),
            "a dense snapshot requiring per-window Q4 packing must not be streamable"
        );
        let dense = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(!dense.lifecycle.transformer_window_materialization);
        assert_eq!(
            dense
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let mismatch = memory_strategy_contract("krea_2_turbo", &spec)
            .unwrap_err()
            .to_string();
        assert!(
            mismatch.contains("Q8") && mismatch.contains("Q4"),
            "{mismatch}"
        );

        let mut no_override = spec.clone();
        no_override.quantize = None;
        std::fs::write(root.join("transformer/config.json"), "{ malformed").unwrap();
        let eligibility_error = is_streamable_spec("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(
            eligibility_error.contains("packed quant"),
            "{eligibility_error}"
        );
        let contract_error = memory_strategy_contract("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(contract_error.contains("packed quant"), "{contract_error}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn file_contract_withholds_rung_four_but_lower_level_loader_remains_reopenable() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, base) = fixture(&tmp);
        let mut resident = base.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut eager = base.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let native = root.join("single.safetensors");
        write_native_i8_safetensors(&native);
        let file = LoadSpec::new(WeightsSource::File(native))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        for spec in [resident, eager] {
            let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
        let file_contract =
            native_memory_strategy_contract_from_spec("krea_2_turbo", &file, &root, true).unwrap();
        assert_eq!(
            file_contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "the registry File source is lstat-pinned and reopened for each transformer window"
        );
        let registered = memory_strategy_contract("krea_2_turbo", &file).unwrap();
        assert_eq!(
            registered
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "a reopenable implementation is not authorization without File-specific evidence"
        );
        assert!(!registered.lifecycle.transformer_window_materialization);
        assert!(
            crate::model::native_file_streamable(&file).unwrap(),
            "explicit Sequential + Deferred execution may use the pinned File stream seam even while automatic authorization stays Missing"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn imported_file_contract_matches_the_base_loader_for_every_typed_field() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("single.safetensors");
        write_native_i8_safetensors(&native);
        let valid = LoadSpec::new(WeightsSource::File(native)).with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(root.clone()),
        );

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        let mut control = valid.clone();
        control.control = Some(WeightsSource::File(root.join("control.safetensors")));
        let mut extra_control = valid.clone();
        extra_control
            .extra_controls
            .push(WeightsSource::File(root.join("extra-control.safetensors")));
        let mut ip_adapter = valid.clone();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir(root.join("ip-adapter")));
        let mut identity = valid.clone();
        identity.identity = Some(mlx_gen::gen_core::IdentityWeights::default());
        let mut text_encoder = valid.clone();
        let external_text_encoder = root.join("external-text");
        gen_core_testkit::write_encoder_contract_fixture(
            &external_text_encoder,
            crate::model::ENCODER_CONTRACT,
        )
        .expect("validation-complete selected text encoder fixture");
        text_encoder.text_encoder = Some(WeightsSource::Dir(external_text_encoder));
        let mut unknown_component = valid.clone();
        unknown_component.components.insert(
            "unknown".into(),
            WeightsSource::File(root.join("unknown.safetensors")),
        );
        let mut missing_base = valid.clone();
        missing_base.components.clear();
        let accepted_adapter = valid.clone().with_adapters(vec![AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        )]);
        let accepted_pid = valid.clone().with_pid(
            WeightsSource::File(root.join("pid.safetensors")),
            WeightsSource::Dir(root.join("gemma")),
        );
        let accepted_deferred = valid
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);

        for (case, spec, expected) in [
            ("valid", valid.clone(), true),
            ("adapter", accepted_adapter, true),
            ("pid", accepted_pid, true),
            ("deferred", accepted_deferred, true),
            ("precision", precision, false),
            ("quantize", valid.clone().with_quant(Quant::Q4), true),
            ("control", control, false),
            ("extra_control", extra_control, false),
            ("ip_adapter", ip_adapter, false),
            ("identity", identity, false),
            ("text_encoder", text_encoder, true),
            ("unknown_component", unknown_component, false),
            ("missing_base", missing_base, false),
        ] {
            let loader = crate::model::validate_native_krea_spec(&spec, "krea_2_turbo").is_ok();
            let contract = memory_strategy_contract("krea_2_turbo", &spec).is_ok();
            assert_eq!(loader, expected, "loader validation for {case}");
            assert_eq!(contract, loader, "contract/loader parity for {case}");
        }
    }

    #[test]
    fn native_i8_contract_counts_bf16_materialization_and_omits_source_companions() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, _) = fixture(&tmp);
        let native = root.join("native.safetensors");
        write_native_i8_safetensors(&native);
        let contract = native_memory_strategy_contract("krea_2_turbo", &native, &root).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 7_843_069_440);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert_eq!(contract.asset_facts.transformer_bytes, 2 * 64 * 2);
        assert_eq!(contract.asset_facts.base_bytes, 7_843_069_698);
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "native ConvRot materialization must retain the execution-only {strategy:?} rung"
            );
        }
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "the compatibility helper intentionally models the historical eager load; registry File specs carry the reopenable lifecycle"
            );
        }
        assert!(contract.conformance_errors().is_empty());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn edit_prices_the_materialized_vision_surface_while_t2i_excludes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = fixture(&tmp);
        let (t2i, _) = memory_strategy_contract_with_plan(crate::model::KREA_2_TURBO_ID, &spec)
            .expect("t2i contract");
        let (edit, _) = memory_strategy_contract_with_plan(crate::model::KREA_2_EDIT_ID, &spec)
            .expect("edit contract");
        let vision = crate::model::ENCODER_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), &root)
            .unwrap();
        let vision_bytes = projected_tensor_headers_bytes(
            &vision
                .materialized_vision_tensor_headers(
                    &crate::model::VISION_ENCODER_CONTRACT,
                    &crate::model::ENCODER_CONTRACT,
                )
                .unwrap(),
            |_| ResidentProjection::Stored,
        )
        .unwrap();
        assert_eq!(
            edit.asset_facts.conditioning_bytes - t2i.asset_facts.conditioning_bytes,
            vision_bytes
        );
    }
}
