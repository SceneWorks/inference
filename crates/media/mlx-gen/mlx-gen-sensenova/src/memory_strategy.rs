//! SenseNova-U1 MLX shared image-memory ladder.
//!
//! The checkpoint is one flat, fused dual-path Qwen3 model. There is no separately releasable text
//! encoder or VAE: conditioning and denoise use different weights interleaved in every resident
//! layer, while the final FM head already emits RGB patches. Consequently staged component
//! residency and bounded decode are structural N/A. Bounded attention is wired only through the
//! generation path used by denoise; understanding, VQA, interleave text, and think-token forwards
//! retain their historical unbounded attention path.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, Quant, WeightsSource};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Exact production parameters exercised by the serial real-Metal runner below.
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const QUALITY_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-quality-q8-mlx-shared-ladder-2026-08-03-v1";
pub const FAST_CALIBRATION_FINGERPRINT: &str =
    "sensenova-u1-fast-q8-mlx-shared-ladder-2026-08-03-v1";
const QUALITY_Q8_ARTIFACT: &str =
    "8da38dde4c39722259a98cfc47643c88e48cea205595625fdbd9fec097f9dc4f";
const FAST_Q8_ARTIFACT: &str = "a9f8968d44ec440bdd7bfb2937a61b847d6f80bb563ffe60ca56be0e395bcf50";

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ArtifactFileIdentity {
    canonical_path: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

fn file_identity(path: &Path) -> std::io::Result<ArtifactFileIdentity> {
    let canonical_path = std::fs::canonicalize(path)?;
    let metadata = std::fs::metadata(&canonical_path)?;
    Ok(ArtifactFileIdentity {
        canonical_path,
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn digest_cache() -> &'static Mutex<HashMap<ArtifactFileIdentity, String>> {
    static CACHE: OnceLock<Mutex<HashMap<ArtifactFileIdentity, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn content_sha256(path: &Path) -> Option<String> {
    let before = file_identity(path).ok()?;
    if let Some(digest) = digest_cache().lock().ok()?.get(&before).cloned() {
        return Some(digest);
    }

    let file = File::open(&before.canonical_path).ok()?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).ok()?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let after = file_identity(path).ok()?;
    if before != after {
        return None;
    }
    let digest = format!("{:x}", hasher.finalize());
    let mut cache = digest_cache().lock().ok()?;
    cache.retain(|identity, _| identity.canonical_path != before.canonical_path);
    cache.insert(before, digest.clone());
    Some(digest)
}

/// SHA-256 of the exact checkpoint bytes, cached by a mutation-sensitive filesystem identity.
///
/// The cache key includes device/inode/size plus mtime and ctime at nanosecond precision. A
/// before/after identity comparison prevents caching a digest when the file changes while it is
/// being read. This keeps repeated selector/contract calls cheap without trusting an HF blob
/// basename (which is attacker-controlled local path text).
pub fn verified_artifact_identity(spec: &LoadSpec) -> Option<String> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return None;
    };
    content_sha256(&root.join("model.safetensors"))
}

fn calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<&'static str> {
    if spec.precision != mlx_gen::Precision::Bf16
        || spec.quantize != Some(Quant::Q8)
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return None;
    }
    match (provider_id, verified_artifact_identity(spec)?.as_str()) {
        (crate::MODEL_ID, QUALITY_Q8_ARTIFACT) => Some(QUALITY_CALIBRATION_FINGERPRINT),
        (crate::MODEL_ID_FAST, FAST_Q8_ARTIFACT) if matches!(&spec.weights, WeightsSource::Dir(root) if root.join(crate::DISTILL_MERGED_MARKER).is_file()) => {
            Some(FAST_CALIBRATION_FINGERPRINT)
        }
        _ => None,
    }
}

pub(crate) fn can_stream_gen(provider_id: &str, spec: &LoadSpec) -> bool {
    if spec.load_shape != LoadShape::DeferredMaterialization
        || spec.precision != mlx_gen::Precision::Bf16
        || !spec.adapters.is_empty()
        || !spec.components.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !matches!(spec.weights, WeightsSource::Dir(_))
    {
        return false;
    }
    if provider_id == crate::MODEL_ID_FAST {
        let WeightsSource::Dir(root) = &spec.weights else {
            return false;
        };
        // Dense-base fast loads merge a curated LoRA into every Gen block at runtime. Streaming
        // those unmerged blocks would silently change the model; only a pre-merged turnkey is exact.
        return root.join(crate::DISTILL_MERGED_MARKER).is_file();
    }
    provider_id == crate::MODEL_ID
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let footprint = crate::model::component_footprint(spec)?;
    let streamable = can_stream_gen(provider_id, spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.calibration = calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![MemoryPhase::Conditioning, MemoryPhase::Denoise],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.asset_facts.base_bytes = footprint.dit;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: Vec::new(),
        synchronized_phase_release: false,
        decode_tiling: false,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "SenseNova is one flat fused dual-path checkpoint; no independently releasable conditioning component exists".to_owned(),
                }
            }
            MemoryStrategy::BoundedDecode => {
                MemoryStrategySupport::StructurallyNotApplicable {
                    reason: "SenseNova has no VAE/decoder phase; the FM head emits RGB patches and unpatchify only reshapes them".to_owned(),
                }
            }
            MemoryStrategy::BoundedAttention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes =
                    vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    quant: Option<Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if !matches!(
            (&context.mode, context.geometry.reference_count),
            (MemoryMode::TextToImage, 0) | (MemoryMode::Edit, 1)
        ) {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory routes are exactly TextToImage with zero references and Edit with one reference",
                contract.provider_id
            )));
        }
        if context.use_pid || context.overlay.is_some() {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory route has no PiD or overlay",
                contract.provider_id
            )));
        }
        if context.geometry.width != 1024
            || context.geometry.height != 1024
            || context.geometry.batch != 1
            || context.geometry.frames != 1
        {
            return Err(CoreError::Unsupported(format!(
                "{}: calibrated memory geometry is exactly 1024x1024, batch 1, and one frame",
                contract.provider_id
            )));
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
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
    safety_check(contract, spec.quantize, context)
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    quant: Option<Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, quant, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        42,
        move |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded decode is structurally not applicable"
            )))
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() || contract.calibration.is_none() {
        return Ok(Vec::new());
    }
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    [
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::Edit,
            reference_count: 1,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    ]
    .into_iter()
    .map(|route| {
        Ok(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, route)?,
        ))
    })
    .collect()
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request(
        provider_id,
        contract,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{MemoryBehaviorRoute, MemoryStrategySupport};
    use mlx_gen::{LoadShape, LoadSpec, Quant, WeightsSource};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_root(label: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "sensenova-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn fixture_spec() -> (std::path::PathBuf, LoadSpec) {
        let root = unique_root("memory-contract");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), [0_u8; 8]).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    #[test]
    fn contract_declares_only_the_current_truthful_surface() {
        let (root, spec) = fixture_spec();
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        assert!(contract.calibration.is_none());
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert!(matches!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::StructurallyNotApplicable { .. }
            ));
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn fast_runtime_lora_route_refuses_streaming_until_premerged() {
        let (root, spec) = fixture_spec();
        let contract = memory_strategy_contract(crate::MODEL_ID_FAST, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        std::fs::write(root.join(crate::DISTILL_MERGED_MARKER), b"{}\n").unwrap();
        let contract = memory_strategy_contract(crate::MODEL_ID_FAST, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn expected_digest_basename_with_arbitrary_bytes_does_not_calibrate() {
        let root = unique_root("calibration-contract");
        std::fs::create_dir_all(&root).unwrap();
        let blob = root.join(QUALITY_Q8_ARTIFACT);
        std::fs::write(&blob, [0_u8; 8]).unwrap();
        std::os::unix::fs::symlink(&blob, root.join("model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q8);
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let first = verified_artifact_identity(&spec).unwrap();
        assert_ne!(
            first, QUALITY_Q8_ARTIFACT,
            "the verifier must hash content instead of trusting the target basename"
        );
        assert!(contract.calibration.is_none());

        // Same path, inode and size: changing only the bytes must invalidate the cached result via
        // ctime/mtime and remain uncalibrated.
        std::fs::write(&blob, [1_u8; 8]).unwrap();
        let second = verified_artifact_identity(&spec).unwrap();
        assert_ne!(first, second);
        assert!(memory_strategy_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn safety_accepts_only_the_two_measured_mode_reference_pairs() {
        let (root, spec) = fixture_spec();
        let spec = spec.with_quant(Quant::Q8);
        let mut contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "sensenova-route-test",
            spec.load_shape,
        ));
        let context = |mode, reference_count| {
            mlx_gen::gen_core::standard_memory_behavior_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryNumericTier {
                    precision: mlx_gen::Precision::Bf16,
                    quant: Some(Quant::Q8),
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )
            .unwrap()
        };
        for accepted in [
            context(MemoryMode::TextToImage, 0),
            context(MemoryMode::Edit, 1),
        ] {
            assert_eq!(
                safety_check(&contract, Some(Quant::Q8), &accepted),
                MemorySafetyDecision::Accept
            );
        }
        for rejected in [
            context(MemoryMode::ImageToImage, 1),
            context(MemoryMode::TextToImage, 1),
            context(MemoryMode::Edit, 0),
            context(MemoryMode::Edit, 2),
        ] {
            assert!(matches!(
                safety_check(&contract, Some(Quant::Q8), &rejected),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("exactly TextToImage with zero references and Edit with one reference")
            ));
        }

        let context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedAttention,
            MemoryNumericTier {
                precision: mlx_gen::Precision::Bf16,
                quant: Some(Quant::Q8),
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(&contract, Some(Quant::Q8), &context),
            MemorySafetyDecision::Accept
        );
        let fixtures =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention).unwrap();
        assert_eq!(fixtures.len(), 2, "T2I and single-reference edit routes");
        let mut scope =
            registered_begin_request(crate::MODEL_ID, &spec, &contract, &fixtures[0].context)
                .unwrap()
                .unwrap();
        let mut request = mlx_gen::GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.chunk_attention);
        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));

        let deferred_spec = spec
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        let mut deferred = memory_strategy_contract(crate::MODEL_ID, &deferred_spec).unwrap();
        deferred.calibration = Some(MemoryCalibrationIdentity::new(
            "sensenova-route-test",
            deferred_spec.load_shape,
        ));
        let fixtures = registered_valid_fixture(
            &deferred_spec,
            &deferred,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        let mut scope = registered_begin_request(
            crate::MODEL_ID,
            &deferred_spec,
            &deferred,
            &fixtures[0].context,
        )
        .unwrap()
        .unwrap();
        let mut request = mlx_gen::GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.chunk_attention && memory.stream_transformer_blocks);
        assert_eq!(
            memory.transformer_window_size,
            Some(TRANSFORMER_WINDOW_SIZE)
        );
        std::fs::remove_dir_all(root).ok();
    }
}
