//! Iris-3B MLX request-memory contracts for the three inference tasks (sc-25690).
//!
//! Each task is its own route with its own resource closure, so each publishes its own contract,
//! behavior registration and calibration fingerprint:
//!
//! | rung | strategy | `iris_3b` (generation) | `iris_3b_depth` | `iris_3b_restore` |
//! |---|---|---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | implemented | implemented | implemented |
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — the Qwen3-VL tower is loaded, encodes, and is dropped before the backbone loads | not applicable — no text encoder | not applicable — no text encoder |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | not applicable — pixel space, no decoder | not applicable | not applicable |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | missing | missing | missing |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | missing | missing | missing |
//!
//! # What each declaration is true of
//!
//! * **Generation, rung 1.** [`crate::model::Iris3b`] holds a request-scoped
//!   [`mlx_gen::Residency`]: a request whose admitted `GenerationMemory::stage_residency` is set
//!   (written by the request scope this module opens), or any request of a generator loaded
//!   `OffloadPolicy::Sequential`, evicts the warm pair, loads the text encoder, encodes (the
//!   conditioning is evaluated inside the encode), drops the encoder, then loads the backbone. The
//!   staged weight floor is therefore `max(tower, backbone)`, never the sum.
//! * **Depth and restoration, rung 1.** Neither task loads a text encoder (epic E4): the
//!   conditioning is the export's shipped empty-prompt states, read once at load. There is one
//!   heavy component and nothing to stage, so the rung is structurally absent rather than missing.
//! * **Rung 2, every task.** Iris denoises in pixel space: the backbone's PiT head emits RGB (or
//!   the depth map) directly and no autoencoder exists on any route.
//! * **Restoration tiling is the resident baseline, not a rung.** The restorer runs one backbone
//!   forward per export tile (the release: 1024-px tiles on a 50 %-overlap stride,
//!   [`gen_core::iris::restoration`]) on every request larger than one tile, whatever is selected,
//!   and has no untiled path. Its accelerator transient is therefore bounded by one tile and stops
//!   growing with the output; what does grow with the output is the host-side fusion canvas. That
//!   is why the contract publishes [`MemoryStrategy::Resident`] only and a pixel-count formula: a
//!   tiling "rung" would be one the engine could not decline.
//! * **Rungs 3 and 4 are `Missing`, not "not applicable".** The backbone has attention and a block
//!   stack, so both rungs describe real work; neither is wired. See the two constants
//!   [`ATTENTION_NOT_ENABLED`] and [`TRANSFORMER_WINDOW_NOT_ENABLED`] for why.
//!
//! # Nothing here is a measurement
//!
//! The contracts carry the route shape and, for a materialized snapshot, the resident bytes of the
//! components each loader builds — priced from safetensors **headers** at the width the loader
//! materializes them. No activation number is published and this crate registers no
//! [`mlx_gen::gen_core::ActivationMemoryRegistration`]: that carrier is measurement-only, and the
//! epic's terminal campaign owns the anchors.

use std::path::Path;

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::iris::depth::DepthInputPlan;
use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::restoration::{RestorationPlan, RestorationResources};
use mlx_gen::gen_core::iris::{GenerationResources, IrisConfig, IrisTask};
use mlx_gen::gen_core::{
    self, adapter_stack_resident_bytes, AdapterResidencyMode, Error as CoreError, LoadShape,
    MemoryArchitectureFacts, MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture,
    MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency,
    MemoryContractSurfaceSpec, MemoryContractSurfaceTier, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, Precision, ResidentRequestMemory,
    Result as CoreResult,
};
use mlx_gen::gen_core::{GenerationRequest, MemoryGeometry, MemoryRunOutcome};
use mlx_gen::LoadSpec;

use crate::depth::DEPTH_MODEL_ID;
use crate::model::MODEL_ID;
use crate::restoration::MODEL_ID as RESTORE_MODEL_ID;

/// Content fingerprint of the generation contract. Load shape is a separate typed axis on
/// [`MemoryCalibrationIdentity`], so it is deliberately absent from the string.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "iris-3b-mlx-derived-2026-10-12-v1";
/// Content fingerprint of the depth contract.
pub const DEPTH_MEMORY_CALIBRATION_FINGERPRINT: &str = "iris-3b-depth-mlx-derived-2026-10-12-v1";
/// Content fingerprint of the restoration contract.
pub const RESTORE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "iris-3b-restore-mlx-derived-2026-10-12-v1";

/// [`MemoryMode::Other`] key of the depth route (`IrisTask::Depth.name()`).
pub const DEPTH_MEMORY_MODE: &str = "depth";
/// [`MemoryMode::Other`] key of the restoration route (`IrisTask::Restoration.name()`).
pub const RESTORE_MEMORY_MODE: &str = "restoration";

/// Why rung 3 is declared `Missing` on every Iris route (applicable, not enabled).
///
/// The shared MLX kernel (`mlx_gen::attention`) bounds no score matrix on this backend — the fused
/// SDPA streams the scores — so its saving is a graph cut whose size is a per-family measurement
/// (−1.7 % on the one family measured), and that module's own rule is that a family selects the
/// rung only on its own real-weight measurement, which this story's CPU-only lane cannot take.
///
/// The kernel's bit-exact axis slices complete heads of q, k and v together. That fits the pixel
/// stage and the text refiner, whose attention is ordinary multi-head, but not the trunk, which is
/// grouped-query in the release (20 query heads over 5 key/value heads, `model.num_kv_heads`): a
/// plain head slice would cut k/v out of step with q. A split aligned to the key/value groups
/// would be exact in principle, but it is new kernel code, not a call into the shared seam, and
/// its saving would still need a real-weight measurement. The remaining query-row axis is
/// tolerance-equivalent only, i.e. a change to the rendered pixels. Depth and restoration
/// additionally have no request-memory carrier (`TransformRequest` and the depth request have no
/// `GenerationMemory`), so no selection could reach their forward.
pub const ATTENTION_NOT_ENABLED: &str =
    "bounded attention is not wired through IrisDiT::forward: the shared MLX kernel's saving is a \
     per-family real-weight measurement; its bit-exact head axis fits the pixel stage and the \
     text refiner but not the grouped-query trunk, where an exact split needs a new \
     group-aligned kernel; and the query-row axis is not bit-exact";

/// Why rung 4 is declared `Missing` on every Iris route (applicable, not enabled).
///
/// Every loader here materializes the whole backbone and evaluates each parameter before it
/// returns ([`crate::model::load_backbone_with_adapters`]); there is no deferred-materialization
/// load shape, no per-block inference loader and no window hook in the inference forward. The
/// block-level constructors on `IrisDiT` (`load_trunk_block`, `load_pixel_block`) serve the
/// trainer's activation-checkpointing recompute from tensors already in memory, not a streamed
/// read. Building the streamed loader is a new load path, not a declaration.
pub const TRANSFORMER_WINDOW_NOT_ENABLED: &str =
    "the Iris loaders materialize and evaluate the whole backbone at load; there is no \
     deferred-materialization load shape or block-window inference loader";

/// The Iris task a registered provider id names.
fn task_of(provider_id: &str) -> CoreResult<IrisTask> {
    match provider_id {
        MODEL_ID => Ok(IrisTask::Generation),
        DEPTH_MODEL_ID => Ok(IrisTask::Depth),
        RESTORE_MODEL_ID => Ok(IrisTask::Restoration),
        other => Err(CoreError::Unsupported(format!(
            "{other}: not an Iris-3B MLX memory provider"
        ))),
    }
}

/// The registered provider id of `task`.
pub const fn provider_id(task: IrisTask) -> &'static str {
    match task {
        IrisTask::Generation => MODEL_ID,
        IrisTask::Depth => DEPTH_MODEL_ID,
        IrisTask::Restoration => RESTORE_MODEL_ID,
    }
}

/// The calibration fingerprint of `task`'s contract.
pub const fn calibration_fingerprint(task: IrisTask) -> &'static str {
    match task {
        IrisTask::Generation => MEMORY_CALIBRATION_FINGERPRINT,
        IrisTask::Depth => DEPTH_MEMORY_CALIBRATION_FINGERPRINT,
        IrisTask::Restoration => RESTORE_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// The request surface `task`'s evidence is keyed on. Optimized evidence never transfers between
/// modes, and neither downstream task is a text-to-image request.
pub fn memory_mode(task: IrisTask) -> MemoryMode {
    match task {
        IrisTask::Generation => MemoryMode::TextToImage,
        IrisTask::Depth => MemoryMode::Other(DEPTH_MEMORY_MODE.to_owned()),
        IrisTask::Restoration => MemoryMode::Other(RESTORE_MEMORY_MODE.to_owned()),
    }
}

/// The numeric tier a spec loads: the precision and the affine tier it names (only generation
/// accepts one, and it packs the text tower and the backbone at that same width — sc-26266). No
/// component is resident above the selected width.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The numeric tier of an already-loaded encoder-free model, which holds its precision but no
/// spec: the same tier [`loaded_tier`] names for the spec it was loaded from.
const fn dense_tier(precision: Precision) -> MemoryNumericTier {
    MemoryNumericTier {
        precision,
        quant: None,
        component_precision_floors: &[],
    }
}

/// Refuse a deferred-materialization load shape. No Iris loader has one: each materializes and
/// evaluates the whole backbone before it returns (see [`TRANSFORMER_WINDOW_NOT_ENABLED`]), so a
/// deferred spec would be accepted and silently loaded eagerly. The three loaders and the
/// contract factories share this one refusal.
pub(crate) fn refuse_deferred_load_shape(id: &str, spec: &LoadSpec) -> CoreResult<()> {
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(CoreError::Unsupported(format!(
            "{id}: LoadSpec::load_shape {:?} is not supported; the Iris-3B loaders have no \
             deferred/block-window load path, use EagerMaterialization",
            spec.load_shape
        )));
    }
    Ok(())
}

/// Refuse a quantized tier the generation loader cannot build (sc-26266): it packs the affine
/// `q8` / `q4` tiers only, and only under the bf16 compute policy (the packed scales are bf16, and
/// upstream's FP32 path is a dense reference, not a tier). The loader and the contract factories
/// share this one refusal.
pub(crate) fn refuse_unloadable_quant(id: &str, spec: &LoadSpec) -> CoreResult<()> {
    match spec.quantize {
        None | Some(gen_core::Quant::Q4 | gen_core::Quant::Q8) => {}
        Some(other) => {
            return Err(CoreError::Unsupported(format!(
                "{id}: LoadSpec::quantize {other:?} is not a tier of the MLX Iris-3B route; it \
                 packs the q8 and q4 affine tiers"
            )))
        }
    }
    if spec.quantize.is_some() && spec.precision != Precision::Bf16 {
        return Err(CoreError::Unsupported(format!(
            "{id}: a quantized tier runs under the bf16 compute policy; LoadSpec::precision {:?} \
             is the dense reference path only",
            spec.precision
        )));
    }
    Ok(())
}

/// Refuse the spec axes that would make a contract describe a load that cannot happen: a deferred
/// load shape on any route, a tier the generation loader cannot pack
/// ([`refuse_unloadable_quant`]), a quantized tier on the two encoder-free tasks (their loaders
/// ship one dense tier), and an adapter stack on those tasks (their loaders refuse
/// `LoadSpec::adapters`; only generation installs adapters, and prices them as overlays).
fn refuse_unloadable_spec(task: IrisTask, spec: &LoadSpec) -> CoreResult<()> {
    let id = provider_id(task);
    refuse_deferred_load_shape(id, spec)?;
    if task.uses_text_encoder() {
        refuse_unloadable_quant(id, spec)?;
    } else if spec.quantize.is_some() {
        return Err(CoreError::Unsupported(format!(
            "{id}: this Iris-3B task ships one dense tier; LoadSpec::quantize is refused at load, \
             so there is no quantized memory contract"
        )));
    }
    if !task.uses_text_encoder() && !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "{id}: LoadSpec::adapters is refused at load on this task, so there is no adapter \
             memory contract"
        )));
    }
    Ok(())
}

/// Bytes per element of the backbone's compute dtype.
const fn compute_width(precision: Precision) -> u32 {
    match precision {
        Precision::Bf16 => mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
        Precision::Fp32 => mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH,
    }
}

/// Architecture axes of the backbone.
///
/// With a parsed `config.yaml` these are the checkpoint's own values. Weights-free, the one axis
/// the route fixes at compile time is the patch size: [`gen_core::iris::SIZE_MULTIPLE`] is the
/// request grid every size is validated against, and the loaders refuse a checkpoint whose patch
/// does not divide it. Heads, width and depth live only in `config.yaml` and are left undeclared
/// rather than guessed. Pixel space has no latent and no autoencoder, so those axes are absent.
fn architecture_facts(
    config: Option<&IrisConfig>,
    precision: Precision,
) -> MemoryArchitectureFacts {
    use mlx_gen::architecture_facts as af;
    let model = config.map(|config| &config.model);
    MemoryArchitectureFacts {
        attention_heads: model.and_then(|m| af::axis(m.num_heads)),
        head_dim: model.and_then(|m| af::head_dim(m.hidden_size, m.num_heads)),
        transformer_blocks: model.and_then(|m| af::axis(m.depth)),
        patch_size: match model {
            Some(m) => af::axis(m.patch_size),
            None => af::axis(gen_core::iris::SIZE_MULTIPLE),
        },
        latent_channels: None,
        vae_spatial_scale: None,
        vae_temporal_scale: None,
        activation_dtype_width: Some(compute_width(precision)),
    }
}

/// The declaration every surface of `task` shares: route shape, calibration identity and strategy
/// classification, with no asset facts.
fn base_contract(
    task: IrisTask,
    precision: Precision,
    config: Option<&IrisConfig>,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id(task),
        MemoryBackendRealization::MlxMetal {
            // Unified memory: a released phase returns to the wired-residency budget.
            bounded_wired_residency: true,
            // Not declared: every loader casts each tensor to the compute dtype and evaluates it
            // before returning, so the resident arrays are materialized copies, never the mapped
            // file.
            lazy_or_mmap_materialization: false,
            // The conditioning is evaluated inside the encode and every solver step ends in an
            // `eval`, so a phase drop frees what it names.
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    // The only shape any Iris loader has; a deferred spec is refused before a contract is built.
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.architecture_facts = architecture_facts(config, precision);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        calibration_fingerprint(task),
        LoadShape::EagerMaterialization,
    ));
    // A `Resident` selection leaves the request memory block absent, which keeps exactly the
    // loaded policy: warm for a `Resident` load, staged for a `Sequential` one.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    let staged = task.uses_text_encoder();
    if staged {
        // After the conditioning release the backbone is the one heavy bundle, through the final
        // pixel conversion.
        contract.phase_facts = Some(gen_core::MemoryPhaseFacts::staged(
            gen_core::StagedWeightSchedule::TwoStage,
        ));
        contract.lifecycle = MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        };
        contract.formula = MemoryFormulaKind::PhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            // The text window is a fixed `text_len`, so the token count is not a request variable:
            // the transient moves with the pixel count and with the rows one solve batches (the
            // prompt batch, doubled under CFG).
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
            ],
        };
    } else {
        // One forward (depth) or one forward per tile (restoration), no phases to separate. Depth
        // scales with the model-input pixel count; restoration's accelerator transient is bounded
        // by one tile while its host fusion canvas scales with the output pixel count.
        contract.formula = MemoryFormulaKind::Affine {
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
            ],
        };
    }
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => MemoryStrategySupport::StructurallyNotApplicable {
                reason: format!(
                    "{} loads no text encoder: its conditioning is the export's shipped \
                     empty-prompt states, read once at load, so the backbone is the only heavy \
                     component and there is no phase to stage",
                    provider_id(task)
                ),
            },
            MemoryStrategy::BoundedDecode => MemoryStrategySupport::StructurallyNotApplicable {
                reason: match task {
                    IrisTask::Restoration => format!(
                        "{RESTORE_MODEL_ID} has no decoder (pixel space); its per-tile backbone \
                         forward is the unconditional resident baseline, not a selectable rung"
                    ),
                    _ => format!(
                        "{} runs in pixel space: the backbone's pixel head emits the output \
                         directly and no autoencoder exists on the route",
                        provider_id(task)
                    ),
                },
            },
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::Missing
            }
        };
    }
    contract
}

/// Declaration-equivalent contract with no asset facts — the registry conformance surface. It
/// carries the same shape, calibration identity and strategy classification as the loaded
/// contract.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let task = task_of(provider_id)?;
    refuse_unloadable_spec(task, spec)?;
    Ok(base_contract(task, spec.precision, None))
}

/// The executable contract for a real load: the declaration plus the resident bytes of what the
/// task's loader builds, read from the resources `spec` names.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let task = task_of(provider_id)?;
    refuse_unloadable_spec(task, spec)?;
    match task {
        IrisTask::Generation => {
            let resources = GenerationResources::from_spec(spec, MODEL_ID)?;
            let config = IrisConfig::from_dir(&resources.backbone_dir)?;
            generation_contract(spec, &resources, &config)
        }
        IrisTask::Depth => {
            let export = TaskExport::from_spec(spec, IrisTask::Depth, DEPTH_MODEL_ID)?;
            export_contract(IrisTask::Depth, spec.precision, &export)
        }
        IrisTask::Restoration => {
            let resources = RestorationResources::from_spec(spec, RESTORE_MODEL_ID)?;
            export_contract(IrisTask::Restoration, spec.precision, &resources.export)
        }
    }
}

/// The generation contract for already-resolved resources (what [`crate::model::load`] holds).
pub(crate) fn generation_contract(
    spec: &LoadSpec,
    resources: &GenerationResources,
    config: &IrisConfig,
) -> CoreResult<MemoryProviderContract> {
    refuse_unloadable_spec(IrisTask::Generation, spec)?;
    let precision = spec.precision;
    let mut contract = base_contract(IrisTask::Generation, precision, Some(config));
    let quant = crate::nn::BackboneQuant::for_load(spec.quantize)?;
    let transformer = backbone_bytes(
        &resources
            .backbone_dir
            .join(gen_core::iris::BACKBONE_WEIGHTS_FILE),
        precision,
        quant,
    )?;
    let conditioning = text_encoder_bytes(&resources.text_encoder_dir, spec.quantize)?;
    let overlays = adapter_overlay_components(spec, precision)?;
    let overlay_bytes = overlays
        .iter()
        .try_fold(0_u64, |total, component| {
            total.checked_add(component.resident_bytes)
        })
        .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: overlay byte sum overflow")))?;
    contract.asset_facts = MemoryAssetFacts {
        // The base model only: the shared validator requires `base_bytes` to equal the base
        // components and to exclude `overlay_bytes`, which reaches the resident total through the
        // typed auxiliary components below.
        base_bytes: conditioning.checked_add(transformer).ok_or_else(|| {
            CoreError::Msg(format!("{MODEL_ID}: base component byte sum overflow"))
        })?,
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: 0,
        overlay_bytes,
    };
    // An adapter load declares its overlay on the typed component axis; a clean load keeps the
    // plain envelope, because an empty component vector would claim an axis it does not use.
    if !overlays.is_empty() {
        let MemoryFormulaKind::PhaseEnvelope {
            phases,
            mut variables,
        } = contract.formula.clone()
        else {
            return Err(CoreError::Msg(format!(
                "{MODEL_ID}: the generation contract no longer declares a PhaseEnvelope"
            )));
        };
        variables.push(MemoryFormulaVariable::OverlayBytes);
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: overlays,
        };
    }
    Ok(contract)
}

/// The depth or restoration contract for an already-resolved task export, at the precision its
/// loader computes in. The load shape is not a parameter: the loaders refuse a deferred spec
/// ([`refuse_deferred_load_shape`]), so a loaded model is always eagerly materialized.
pub(crate) fn export_contract(
    task: IrisTask,
    precision: Precision,
    export: &TaskExport,
) -> CoreResult<MemoryProviderContract> {
    let mut contract = base_contract(task, precision, Some(&export.config));
    // The export's `model.safetensors`: the backbone (under `pixel.` plus the 1×1 reducer for
    // depth), every matrix cast to the compute dtype.
    let transformer = backbone_bytes(&export.weights_path(), precision, None)?;
    // The shipped empty-prompt states are the whole conditioning of these tasks. They are read to
    // the host and uploaded as stored (f32), so they price at their file width.
    let conditioning =
        projected_safetensors_bytes(export.empty_prompt_path(), |_| ResidentProjection::Stored)?;
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: conditioning.checked_add(transformer).ok_or_else(|| {
            CoreError::Msg(format!(
                "{}: base component byte sum overflow",
                provider_id(task)
            ))
        })?,
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: 0,
        overlay_bytes: 0,
    };
    Ok(contract)
}

/// Resident bytes of a backbone checkpoint, priced the way the loaders hold it: every projection
/// weight (a rank-2 `.weight`) as [`crate::nn::ProjectionLayout::of`] holds it under `quant` —
/// in the compute dtype on the dense tier, packed (over its zero-padded input) on a quantized one;
/// every other tensor of rank two or more (the text position table excepted) cast to the compute
/// dtype; vectors (norm gains, biases) and that table as stored. The rule is the one in
/// [`crate::model::load_backbone_quantized`] and [`crate::depth::load_depth_export`] (whose keys
/// carry a `pixel.` prefix, hence the suffix match).
fn backbone_bytes(
    weights: &Path,
    precision: Precision,
    quant: Option<crate::nn::BackboneQuant>,
) -> CoreResult<u64> {
    let width = u64::from(compute_width(precision));
    let mut total = 0_u64;
    for header in gen_core::weightsmeta::safetensors_path_tensor_headers(weights)? {
        let position_table =
            header.name == "y_pos_embedding" || header.name.ends_with(".y_pos_embedding");
        let elements = header.shape.iter().map(|&d| d as u64).product::<u64>();
        let bytes = match header.shape.as_slice() {
            [out, input] if header.name.ends_with(".weight") => crate::nn::ProjectionLayout::of(
                *input, quant,
            )
            .weight_bytes(*out as u64, *input as u64, width),
            shape if shape.len() >= 2 && !position_table => elements * width,
            _ => header.data_bytes,
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: backbone byte sum overflow")))?;
    }
    Ok(total)
}

/// Key prefix of the Qwen3-VL language tower inside the VLM checkpoint.
const LANGUAGE_TOWER_PREFIX: &str = "model.language_model.";
/// The untied output head, at the checkpoint root.
const LANGUAGE_HEAD_KEY: &str = "lm_head.weight";

/// Resident bytes of the text encoder as [`crate::text_encoder::IrisTextEncoder::load_quantized`]
/// builds it at `quant`.
///
/// The shared decoder (`mlx_llm::CausalLm`) reads the language tower (`model.language_model.*`)
/// and, when the head is untied, `lm_head.weight`, casting each to bf16 whatever the file stores.
/// On a quantized tier every attention / MLP projection (each rank-2 `.weight` of the tower but
/// the token embedding) is packed at the tower's group size
/// ([`crate::text_encoder::text_encoder_quant_group`]); the embedding, the head and the norms stay
/// bf16. The vision tower (`model.visual.*`) rides along on disk and is materialized by nothing on
/// this route. The file list is the loader's own ([`crate::text_encoder::weight_shards`]), so a
/// stray safetensors beside the shards is not priced.
fn text_encoder_bytes(dir: &Path, quant: Option<gen_core::Quant>) -> CoreResult<u64> {
    let config = mlx_llm::ModelConfig::from_dir(dir)
        .map_err(|e| CoreError::Msg(format!("{MODEL_ID}: reading the text encoder config: {e}")))?;
    let untied_head = !config.tie_word_embeddings;
    let packed = crate::text_encoder::text_encoder_quant_spec(&config, quant)?;
    let embedding = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
    let mut total = 0_u64;
    for shard in crate::text_encoder::weight_shards(dir)? {
        let bytes = projected_safetensors_bytes(&shard, |header| {
            if header.name.starts_with(LANGUAGE_TOWER_PREFIX) {
                match packed {
                    Some(spec)
                        if header.shape.len() == 2
                            && header.name.ends_with(".weight")
                            && header.name != embedding =>
                    {
                        ResidentProjection::GroupQuantized {
                            bits: spec.bits,
                            group_size: spec.group_size as usize,
                        }
                    }
                    _ => ResidentProjection::Bfloat16,
                }
            } else if untied_head && header.name == LANGUAGE_HEAD_KEY {
                ResidentProjection::Bfloat16
            } else {
                ResidentProjection::Omit
            }
        })?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: text encoder byte sum overflow")))?;
    }
    Ok(total)
}

/// Stable provider-local ids for the adapter overlays [`adapter_overlay_components`] declares.
const ADAPTER_RESIDUALS_COMPONENT_ID: &str = "iris_3b.adapters.forward_residuals";
const ADAPTER_DELTAS_COMPONENT_ID: &str = "iris_3b.adapters.materialized_deltas";

/// The adapter stack [`crate::adapters::install`] leaves beside the backbone, priced as held.
///
/// * **The factors.** Every file is installed as a forward-time residual and never folded into the
///   base, so the stack is priced through the shared [`adapter_stack_resident_bytes`] at
///   [`AdapterResidencyMode::Additive`]. A `None` from it means a requested source could not be
///   sized; this fails closed rather than declaring a zero.
/// * **The materialized deltas.** The Iris projections are dense, so the shared install
///   reconstructs every LoKr / LoHa module's full `[out, in]` delta in the compute dtype
///   ([`materialized_delta_elements`]). A LoRA file carries no such keys and prices zero here.
///
/// A materialized module's factors stay priced too: whether the install releases them is not a
/// contract this crate owns, and the error is in the conservative direction.
fn adapter_overlay_components(
    spec: &LoadSpec,
    precision: Precision,
) -> CoreResult<Vec<MemoryResidentComponent>> {
    let mut components = Vec::new();
    let residuals = adapter_stack_resident_bytes(&spec.adapters, AdapterResidencyMode::Additive)
        .ok_or_else(|| {
            CoreError::Unsupported(format!(
                "{MODEL_ID}: an adapter stack was requested but at least one source could not be \
                 sized; refusing to declare a zero the shared validator would wave through"
            ))
        })?;
    push_overlay(&mut components, ADAPTER_RESIDUALS_COMPONENT_ID, residuals);
    let mut elements = 0_u64;
    for adapter in &spec.adapters {
        elements = elements.saturating_add(materialized_delta_elements(&adapter.path)?);
    }
    push_overlay(
        &mut components,
        ADAPTER_DELTAS_COMPONENT_ID,
        elements.saturating_mul(u64::from(compute_width(precision))),
    );
    Ok(components)
}

/// Record one adapter overlay, skipping a zero: the shared validator refuses a declared component
/// with zero bytes.
fn push_overlay(into: &mut Vec<MemoryResidentComponent>, id: &str, resident_bytes: u64) {
    if resident_bytes == 0 {
        return;
    }
    into.push(MemoryResidentComponent {
        id: id.to_owned(),
        kind: MemoryComponentKind::AdapterStack,
        resident_bytes,
        // No rung bounds an adapter: it lives exactly as long as the backbone it is installed on.
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    });
}

/// Elements of the full `[out, in]` deltas the shared LyCORIS install materializes for the adapter
/// file at `path`, read from its safetensors headers (no tensor data): `|w1| · |w2|` per LoKr
/// module (`kron(w1, w2)`), and one delta per LoHa module sized by its first Hadamard pair. A
/// module whose factor set cannot describe a delta fails closed — the install refuses it too.
fn materialized_delta_elements(path: &Path) -> CoreResult<u64> {
    use gen_core::weightsmeta::{
        safetensors_path_tensor_headers, split_factor_key, LOHA_TP_SUFFIXES, LOKR_TP_SUFFIXES,
    };
    use std::collections::BTreeMap;

    type Factors<'a> = BTreeMap<&'a str, &'a [usize]>;
    let headers = safetensors_path_tensor_headers(path)?;
    let mut lokr: BTreeMap<&str, Factors<'_>> = BTreeMap::new();
    let mut loha: BTreeMap<&str, Factors<'_>> = BTreeMap::new();
    for header in &headers {
        if let Some((module, factor)) = split_factor_key(&header.name, &LOKR_TP_SUFFIXES) {
            lokr.entry(module)
                .or_default()
                .insert(factor, header.shape.as_slice());
        } else if let Some((module, factor)) = split_factor_key(&header.name, &LOHA_TP_SUFFIXES) {
            loha.entry(module)
                .or_default()
                .insert(factor, header.shape.as_slice());
        }
    }
    let refuse = |module: &str, what: &str| {
        CoreError::Unsupported(format!(
            "{MODEL_ID}: adapter {} module `{module}` {what}; its materialized delta cannot be \
             priced",
            path.display()
        ))
    };
    fn numel(shape: &[usize]) -> u64 {
        shape.iter().map(|&d| d as u64).product()
    }
    // A low-rank pair `a [p, r] · b [r, q, k…]` reconstructs `[p, q, k…]`.
    fn low_rank(a: &[usize], b: &[usize]) -> Option<u64> {
        Some((*a.first()? as u64).saturating_mul(numel(b.get(1..)?)))
    }
    // A tucker triple `einsum(t [i, j, k…], a [i, p], b [j, q]) → [p, q, k…]`.
    fn tucker(t: &[usize], a: &[usize], b: &[usize]) -> Option<u64> {
        Some(
            (*a.get(1)? as u64)
                .saturating_mul(*b.get(1)? as u64)
                .saturating_mul(numel(t.get(2..)?)),
        )
    }

    let mut elements = 0_u64;
    for (&module, f) in &lokr {
        let w1 = match (f.get("lokr_w1"), f.get("lokr_w1_a"), f.get("lokr_w1_b")) {
            (Some(w), _, _) => Some(numel(w)),
            (None, Some(a), Some(b)) => low_rank(a, b),
            _ => None,
        }
        .ok_or_else(|| refuse(module, "has no complete w1 factor"))?;
        let w2 = match (
            f.get("lokr_w2"),
            f.get("lokr_t2"),
            f.get("lokr_w2_a"),
            f.get("lokr_w2_b"),
        ) {
            (Some(w), _, _, _) => Some(numel(w)),
            (None, Some(t), Some(a), Some(b)) => tucker(t, a, b),
            (None, None, Some(a), Some(b)) => low_rank(a, b),
            _ => None,
        }
        .ok_or_else(|| refuse(module, "has no complete w2 factor"))?;
        elements = elements.saturating_add(w1.saturating_mul(w2));
    }
    for (&module, f) in &loha {
        let (Some(a), Some(b)) = (f.get("hada_w1_a"), f.get("hada_w1_b")) else {
            return Err(refuse(module, "has no complete hada_w1 pair"));
        };
        let delta = match f.get("hada_t1") {
            Some(t) => tucker(t, a, b),
            None => low_rank(a, b),
        }
        .ok_or_else(|| refuse(module, "has a malformed hada_w1 pair"))?;
        elements = elements.saturating_add(delta);
    }
    Ok(elements)
}

/// The provider safety check of all three routes: the shared handshake (calibration identity,
/// loaded tier, selection, budget), then the route's own gate. It can reject, never swap in a
/// different strategy or tier.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check_at(loaded_tier(spec), contract, context)
}

/// [`safety_check`] for an already-loaded encoder-free model (`IrisDepth`, `IrisRestorer`), which
/// holds its precision rather than a spec: the same check against the same tier.
pub(crate) fn loaded_safety_check(
    precision: Precision,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check_at(dense_tier(precision), contract, context)
}

fn safety_check_at(
    tier: MemoryNumericTier,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let task = task_of(&contract.provider_id)?;
        let id = provider_id(task);
        if context.mode != memory_mode(task) {
            return Err(CoreError::Unsupported(format!(
                "{id}: this route's request surface is {:?}; got {:?}",
                memory_mode(task).as_key(),
                context.mode.as_key()
            )));
        }
        if context.geometry.reference_count > 0 {
            return Err(CoreError::Unsupported(format!(
                "{id}: the route takes no reference images"
            )));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{id}: there is no PiD decoder on this route"
            )));
        }
        if context.geometry.frames > 1 {
            return Err(CoreError::Unsupported(format!(
                "{id}: a still-image route cannot admit {} frames",
                context.geometry.frames
            )));
        }
        // Depth estimates and restores one image per request; only generation batches.
        if !task.uses_text_encoder() && context.geometry.batch != 1 {
            return Err(CoreError::Unsupported(format!(
                "{id}: one image per request, got a batch of {}",
                context.geometry.batch
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(tier),
        Some(&route_gate),
    )
}

/// The geometry a depth request is admitted at: the grid the model actually runs on (the source
/// resized to the request's resolution cap and snapped to the patch), one image, no reference —
/// the source is the task's subject, not a conditioning reference. Same semantics as the Candle
/// twin's helper of the same name.
pub fn depth_memory_geometry(plan: &DepthInputPlan) -> MemoryGeometry {
    MemoryGeometry {
        width: plan.model_width,
        height: plan.model_height,
        batch: 1,
        frames: 1,
        reference_count: 0,
    }
}

/// The geometry a restoration request is admitted at: the size the model runs at
/// (`RestorationPlan::processing` — the output, or the output enlarged to one tile on its short
/// side), which is what the host fusion canvases scale with. One image, no reference.
pub fn restoration_memory_geometry(plan: &RestorationPlan) -> MemoryGeometry {
    MemoryGeometry {
        width: plan.processing.width,
        height: plan.processing.height,
        batch: 1,
        frames: 1,
        reference_count: 0,
    }
}

/// Pixels of the largest single backbone forward a restoration plan runs: the whole padded image
/// when it fits one tile, else one `tile × tile` tile — the bound the unconditional tiling puts on
/// the accelerator activation, whatever the output size.
pub fn restoration_forward_pixels(plan: &RestorationPlan) -> u64 {
    if plan.single_pass() {
        u64::from(plan.padded.width) * u64::from(plan.padded.height)
    } else {
        u64::from(plan.tile) * u64::from(plan.tile)
    }
}

/// One provider-owned valid context per implemented optimized rung. Depth and restoration
/// implement none, so they publish none.
pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let task = task_of(&contract.provider_id)?;
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        loaded_tier(spec),
        MemoryBehaviorRoute {
            mode: memory_mode(task),
            reference_count: 0,
            use_pid: false,
            has_phases: task.uses_text_encoder(),
            // The loaded adapter stack is part of the route identity, so a fixture for an adapted
            // load is keyed apart from a plain one (the key the Candle twin uses).
            overlay: (!spec.adapters.is_empty()).then(|| ADAPTER_OVERLAY_KEY.to_owned()),
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

/// The overlay identity of a behavior fixture built for a spec that loads an adapter stack.
pub const ADAPTER_OVERLAY_KEY: &str = "lora";

/// Images a generation request renders: one per prompt of a prompt batch, `count` times.
pub fn request_images(request: &GenerationRequest) -> u64 {
    request.prompt_batch.len().max(1) as u64 * u64::from(request.count)
}

/// The generation request scope: the shared MLX lifecycle, plus the one axis the shared geometry
/// check cannot see. A prompt batch denoises its rows **together** (one batched solve per `count`
/// repetition) and every repetition's samples stay live until the final conversion, so the
/// admitted `MemoryGeometry::batch` bounds the request's total images ([`request_images`]), not
/// `count` alone — an admitted single-image request cannot be widened into an N-row batch at N
/// times the activation.
struct GenerationRequestScope {
    core: mlx_gen::request_scope::MlxRequestScopeCore,
    admitted_images: u32,
}

impl MemoryRequestScope for GenerationRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        let images = request_images(request);
        if images > u64::from(self.admitted_images) {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: the request renders {images} images ({} prompts x count {}) but was \
                 admitted for {}; a prompt batch denoises its rows together, so each one is a \
                 batch row of the admitted geometry",
                request.prompt_batch.len().max(1),
                request.count,
                self.admitted_images
            )));
        }
        self.core.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> CoreResult<()> {
        self.core.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> CoreResult<()> {
        self.core.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> CoreResult<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }

    fn configure_attention(&mut self, size: u32) -> CoreResult<()> {
        self.core.configure_attention(size)
    }

    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> CoreResult<()> {
        self.core.materialize_transformer_window(first, count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.core.finish(outcome)
    }
}

/// The executable admission seam. Generation opens the shared MLX request scope, whose
/// `configure_request` bounds the request's images by the admitted batch and writes the admitted
/// `GenerationMemory` onto the request that [`crate::model::Iris3b`] then honours. Depth and restoration run their resident baseline only
/// and take no `GenerationRequest`, so an admitted request needs no scope.
pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    if !task_of(&contract.provider_id)?.uses_text_encoder() {
        return Ok(None);
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        contract
            .architecture_facts
            .transformer_blocks
            .unwrap_or_default() as usize,
        |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: pixel space has no decoder to tile"
            )))
        },
    )?;
    Ok(Some(Box::new(GenerationRequestScope {
        core: mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
            config,
            mlx_gen::request_scope::MlxScopeCleanup::Device,
        ),
        admitted_images: context.geometry.batch,
    })))
}

/// The registry-load surface the two encoder-free routes witness: the one dense tier, eagerly
/// materialized, under both offload policies (advisory there: those tasks are always resident).
/// Neither loader accepts a quantized tier or has a deferred-materialization load shape, so
/// neither is witnessed — a selector no load can take would be a witness-only surface.
pub fn memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.selector.tier == MemoryContractSurfaceTier::Bf16
                && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

/// The registry-load surface of generation (sc-26266): the dense tier and the two affine tiers
/// the loader packs, eagerly materialized (there is no deferred loader), under both offload
/// policies.
pub fn generation_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

/// Generation: the pre-load contract and admission check of `iris_3b`.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: |spec| memory_strategy_contract(MODEL_ID, spec),
    safety_check,
};
/// Depth: the pre-load contract and admission check of `iris_3b_depth`.
pub const DEPTH_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: DEPTH_MODEL_ID,
    contract: |spec| memory_strategy_contract(DEPTH_MODEL_ID, spec),
    safety_check,
};
/// Restoration: the pre-load contract and admission check of `iris_3b_restore`.
pub const RESTORE_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: RESTORE_MODEL_ID,
        contract: |spec| memory_strategy_contract(RESTORE_MODEL_ID, spec),
        safety_check,
    };

/// Generation: the weights-free contract fixture.
pub const MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(MODEL_ID, spec),
        surface_specs: generation_memory_contract_surface_specs,
    };
/// Depth: the weights-free contract fixture.
pub const DEPTH_MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: DEPTH_MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(DEPTH_MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };
/// Restoration: the weights-free contract fixture.
pub const RESTORE_MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: RESTORE_MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(RESTORE_MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

/// Generation: the provider-owned valid fixtures and the executable admission seam.
pub const MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };
/// Depth: the behavior seam (no optimized rung, so no fixtures and no request scope).
pub const DEPTH_MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: DEPTH_MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };
/// Restoration: the behavior seam (no optimized rung, so no fixtures and no request scope).
pub const RESTORE_MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: RESTORE_MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::iris::restoration::{plan, Dims, RestorationOptions, TileGeometry};
    use mlx_gen::gen_core::weightsmeta::{
        safetensors_path_tensor_headers, SafetensorsTensorHeader,
    };
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::gen_core::InputSizing;
    use mlx_gen::{AdapterKind, AdapterSpec, Quant, WeightsSource};
    use std::path::PathBuf;

    const TASKS: [IrisTask; 3] = [IrisTask::Generation, IrisTask::Depth, IrisTask::Restoration];

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    /// A spec no loader could open: the weights-free surface.
    fn sentinel() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()))
    }

    fn weights_free(task: IrisTask) -> MemoryProviderContract {
        weights_free_memory_strategy_contract(provider_id(task), &sentinel()).unwrap()
    }

    /// The committed miniature resources of `task`.
    fn tiny_spec(task: IrisTask) -> LoadSpec {
        match task {
            IrisTask::Generation => {
                let mut spec =
                    LoadSpec::new(WeightsSource::Dir(fixtures().join("tiny-snapshot/iris")));
                spec.components.insert(
                    gen_core::iris::TEXT_ENCODER_COMPONENT.into(),
                    WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
                );
                spec
            }
            IrisTask::Depth => LoadSpec::new(WeightsSource::Dir(fixtures().join("tiny-depth"))),
            IrisTask::Restoration => LoadSpec::new(WeightsSource::Dir(
                fixtures().join("tiny-snapshot/upscaler"),
            )),
        }
    }

    fn loaded(task: IrisTask, spec: &LoadSpec) -> MemoryProviderContract {
        memory_strategy_contract(provider_id(task), spec).unwrap()
    }

    fn support(contract: &MemoryProviderContract, strategy: MemoryStrategy) -> String {
        match &contract.capability(strategy).unwrap().support {
            MemoryStrategySupport::Implemented => "implemented",
            MemoryStrategySupport::StructurallyNotApplicable { .. } => "not-applicable",
            MemoryStrategySupport::Missing => "missing",
        }
        .to_owned()
    }

    /// A context a caller would present for `strategy` on `task`'s own route.
    fn context(
        task: IrisTask,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryRunContext {
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            loaded_tier(&sentinel()),
            MemoryBehaviorRoute {
                mode: memory_mode(task),
                reference_count: 0,
                use_pid: false,
                has_phases: task.uses_text_encoder(),
                overlay: None,
            },
        )
        .unwrap()
    }

    fn sum(path: &Path, bytes: impl Fn(&SafetensorsTensorHeader) -> u64) -> u64 {
        safetensors_path_tensor_headers(path)
            .unwrap()
            .iter()
            .map(bytes)
            .sum()
    }

    fn elements(header: &SafetensorsTensorHeader) -> u64 {
        header.shape.iter().map(|&d| d as u64).product()
    }

    /// The ladder each route publishes, on every surface it witnesses.
    ///
    /// *Mutations that red this:* declaring `StagedResidency` implemented on an encoder-free task
    /// (the shared conformance refuses a staged rung with no lifecycle phases); declaring it
    /// `Missing` on generation; promoting either bounded-DiT rung to `Implemented` (no lifecycle
    /// hook backs it).
    #[test]
    fn each_route_declares_exactly_the_rungs_its_engine_runs() {
        let dense = memory_contract_surface_specs();
        // *Mutation that reds this:* dropping the eager filter in
        // `memory_contract_surface_specs` (a deferred selector no loader can take reappears, and
        // its contract construction is refused).
        assert_eq!(
            dense.len(),
            2,
            "one dense tier, eager, x 2 offload policies"
        );
        let selectors: Vec<_> = dense.iter().map(|s| s.selector.id()).collect();
        assert_eq!(selectors, ["bf16:resident:eager", "bf16:sequential:eager"]);
        // sc-26266: generation witnesses the two affine tiers its loader packs as well.
        // *Mutation that reds this:* dropping the eager filter, or filtering generation to bf16.
        let generation = generation_memory_contract_surface_specs();
        let mut selectors: Vec<_> = generation.iter().map(|s| s.selector.id()).collect();
        selectors.sort();
        assert_eq!(
            selectors,
            [
                "bf16:resident:eager",
                "bf16:sequential:eager",
                "q4:resident:eager",
                "q4:sequential:eager",
                "q8:resident:eager",
                "q8:sequential:eager",
            ]
        );
        for task in TASKS {
            let surfaces = if task == IrisTask::Generation {
                &generation
            } else {
                &dense
            };
            for surface in surfaces {
                if task != IrisTask::Generation {
                    assert_eq!(surface.selector.tier, MemoryContractSurfaceTier::Bf16);
                }
                let contract =
                    weights_free_memory_strategy_contract(provider_id(task), &surface.spec)
                        .unwrap();
                gen_core_testkit::memory_strategy_conformance(&contract);
                assert_eq!(contract.provider_id, provider_id(task));
                assert_eq!(contract.load_shape, surface.spec.load_shape);
                let calibration = contract.calibration.as_ref().unwrap();
                assert_eq!(calibration.fingerprint, calibration_fingerprint(task));
                assert_eq!(calibration.load_shape, surface.spec.load_shape);
                assert_eq!(contract.asset_facts, MemoryAssetFacts::default());

                let staged = if task == IrisTask::Generation {
                    "implemented"
                } else {
                    "not-applicable"
                };
                let declared: Vec<String> = MemoryStrategy::ALL
                    .into_iter()
                    .map(|strategy| support(&contract, strategy))
                    .collect();
                assert_eq!(
                    declared,
                    [
                        "implemented",
                        staged,
                        "not-applicable",
                        "missing",
                        "missing"
                    ],
                    "{} [{}]",
                    provider_id(task),
                    surface.selector.id()
                );
                // No hook is declared for a rung that is not implemented.
                assert!(!contract.lifecycle.decode_tiling);
                assert!(!contract.lifecycle.attention_chunking);
                assert!(!contract.lifecycle.transformer_window_materialization);
                assert_eq!(
                    contract.lifecycle.synchronized_phase_release,
                    task == IrisTask::Generation
                );
                assert_eq!(contract.phase_facts.is_some(), task == IrisTask::Generation);
            }
        }
    }

    /// *Mutation that reds this:* giving two tasks one fingerprint, or a string the persisted
    /// evidence grammar refuses (two `vN` tokens, an upper-case token).
    #[test]
    fn fingerprints_are_valid_and_distinct_per_task() {
        let mut seen = std::collections::BTreeSet::new();
        for task in TASKS {
            let fingerprint = calibration_fingerprint(task);
            gen_core::validate_calibration_fingerprint(fingerprint).unwrap();
            assert!(seen.insert(fingerprint), "{fingerprint} is shared");
            assert!(seen.insert(provider_id(task)));
        }
        assert_eq!(memory_mode(IrisTask::Depth).as_key(), "depth");
        assert_eq!(memory_mode(IrisTask::Restoration).as_key(), "restoration");
        assert_eq!(memory_mode(IrisTask::Generation), MemoryMode::TextToImage);
        // The exact strings both backends' consumers key on (the Candle twin's differ only in the
        // backend token).
        assert_eq!(
            TASKS.map(calibration_fingerprint),
            [
                "iris-3b-mlx-derived-2026-10-12-v1",
                "iris-3b-depth-mlx-derived-2026-10-12-v1",
                "iris-3b-restore-mlx-derived-2026-10-12-v1",
            ]
        );
    }

    /// Only generation packs a quantized tier — Q4/Q8 under bf16 compute (sc-26266) — and only
    /// generation installs adapters, so every other spec has no contract, weights-free or loaded.
    ///
    /// *Mutations that red this:* dropping either arm of `refuse_unloadable_spec`; dropping either
    /// arm of `refuse_unloadable_quant`.
    #[test]
    fn a_spec_no_loader_opens_has_no_contract() {
        let refused =
            |id: &str, result: CoreResult<MemoryProviderContract>, field: &str| match result {
                Err(CoreError::Unsupported(message)) => {
                    assert!(message.contains(field), "{id}: {message}")
                }
                other => panic!("{id}: expected a typed refusal, got {other:?}"),
            };
        let scratch = tempfile::tempdir().unwrap();
        let aligned_tower = write_aligned_tiny_text_encoder(scratch.path());
        for task in TASKS {
            let id = provider_id(task);
            let mut resources = tiny_spec(task);
            if task == IrisTask::Generation {
                // The released miniature tower has no affine group (see
                // `a_quantized_generation_load_prices_the_packed_projections`).
                resources.components.insert(
                    gen_core::iris::TEXT_ENCODER_COMPONENT.into(),
                    WeightsSource::Dir(aligned_tower.clone()),
                );
            }
            for quant in [Quant::Q4, Quant::Q8] {
                let weights_free =
                    weights_free_memory_strategy_contract(id, &sentinel().with_quant(quant));
                let loaded = memory_strategy_contract(id, &resources.clone().with_quant(quant));
                if task == IrisTask::Generation {
                    gen_core_testkit::memory_strategy_conformance(&weights_free.unwrap());
                    gen_core_testkit::memory_strategy_conformance(&loaded.unwrap());
                } else {
                    refused(id, weights_free, "quantize");
                    refused(id, loaded, "quantize");
                }
            }
            refused(
                id,
                weights_free_memory_strategy_contract(id, &sentinel().with_quant(Quant::Nvfp4)),
                "quantize",
            );
            let mut fp32 = sentinel().with_quant(Quant::Q8);
            fp32.precision = Precision::Fp32;
            refused(
                id,
                weights_free_memory_strategy_contract(id, &fp32),
                if task == IrisTask::Generation {
                    "precision"
                } else {
                    "quantize"
                },
            );
            let mut adapted = sentinel();
            adapted.adapters = vec![AdapterSpec::new(
                fixtures().join("iris_lora.safetensors"),
                1.0,
                AdapterKind::Lora,
            )];
            let result = weights_free_memory_strategy_contract(id, &adapted);
            if task == IrisTask::Generation {
                result.unwrap();
            } else {
                match result {
                    Err(CoreError::Unsupported(message)) => {
                        assert!(message.contains("adapters"), "{message}")
                    }
                    other => panic!("{id}: expected a typed refusal, got {other:?}"),
                }
            }
        }
        assert!(weights_free_memory_strategy_contract("iris_3b_unknown", &sentinel()).is_err());
    }

    /// No Iris loader has a deferred-materialization load shape, so a deferred spec has no
    /// contract on any route — weights-free or against real resources.
    ///
    /// *Mutation that reds this:* removing `refuse_deferred_load_shape` from
    /// `refuse_unloadable_spec`.
    #[test]
    fn a_deferred_load_shape_has_no_contract() {
        for task in TASKS {
            let id = provider_id(task);
            for result in [
                weights_free_memory_strategy_contract(
                    id,
                    &sentinel().with_load_shape(LoadShape::DeferredMaterialization),
                ),
                memory_strategy_contract(
                    id,
                    &tiny_spec(task).with_load_shape(LoadShape::DeferredMaterialization),
                ),
            ] {
                match result {
                    Err(CoreError::Unsupported(message)) => {
                        assert!(message.contains("load_shape"), "{id}: {message}")
                    }
                    other => panic!("{id}: expected a typed refusal, got {other:?}"),
                }
            }
            let eager = weights_free(task);
            assert_eq!(eager.load_shape, LoadShape::EagerMaterialization);
            assert_eq!(
                eager.calibration.unwrap().load_shape,
                LoadShape::EagerMaterialization
            );
        }
    }

    /// A prompt batch denoises its rows together, so the admitted batch bounds
    /// `prompts x count`, not `count` alone.
    ///
    /// *Mutation that reds this:* opening the bare `MlxRequestScopeCore` (it checks only
    /// `count <= batch`, so a four-prompt request is admitted on a one-image admission).
    #[test]
    fn the_generation_scope_bounds_a_prompt_batch_by_the_admitted_batch() {
        let spec = sentinel();
        let contract = weights_free(IrisTask::Generation);
        let configure = |admitted: u32, prompts: usize, count: u32| {
            let mut context = context(
                IrisTask::Generation,
                &contract,
                MemoryStrategy::StagedResidency,
            );
            context.geometry.batch = admitted;
            let mut request = MemoryBehaviorFixture::new(context.clone()).request;
            request.count = count;
            request.prompt_batch = (0..prompts).map(|i| format!("prompt {i}")).collect();
            assert_eq!(
                request_images(&request),
                prompts.max(1) as u64 * u64::from(count)
            );
            let mut scope = registered_begin_request(&spec, &contract, &context)
                .unwrap()
                .unwrap();
            let result = scope.configure_request(&mut request);
            scope.finish(MemoryRunOutcome::Complete).unwrap();
            result.map(|()| request.memory)
        };
        // No prompt batch is one row per `count`.
        assert!(configure(1, 0, 1).unwrap().unwrap().stage_residency);
        assert!(configure(4, 0, 4).is_ok());
        assert!(configure(4, 4, 1).is_ok());
        assert!(configure(4, 2, 2).is_ok());
        for (admitted, prompts, count) in [(1, 4, 1), (4, 2, 3), (4, 5, 1), (3, 2, 2)] {
            match configure(admitted, prompts, count) {
                Err(CoreError::Unsupported(message)) => {
                    assert!(message.contains("prompt batch"), "{message}")
                }
                other => panic!(
                    "{prompts} prompts x count {count} on a batch of {admitted} must be \
                     refused, got {other:?}"
                ),
            }
        }
    }

    /// The overlay identity of a behavior fixture: `"lora"` when the spec loads an adapter stack,
    /// absent otherwise — the key the Candle twin uses, so one adapted route has one identity on
    /// both lanes.
    ///
    /// *Mutation that reds this:* publishing `overlay: None` for an adapted spec.
    #[test]
    fn an_adapted_load_keys_its_fixture_by_the_adapter_overlay() {
        let contract = weights_free(IrisTask::Generation);
        let overlay = |spec: &LoadSpec| {
            registered_valid_fixture(spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context
                .overlay
        };
        assert_eq!(overlay(&sentinel()), None);
        let mut adapted = sentinel();
        adapted.adapters = vec![AdapterSpec::new(
            fixtures().join("iris_lokr.safetensors"),
            1.0,
            AdapterKind::Lokr,
        )];
        assert_eq!(overlay(&adapted).as_deref(), Some("lora"));
        assert_eq!(ADAPTER_OVERLAY_KEY, "lora");
    }

    /// The admission geometry of the encoder-free routes, and the one-tile bound of a tiled
    /// restoration.
    ///
    /// *Mutations that red this:* admitting depth at the source size instead of the model grid;
    /// admitting restoration at the output instead of the processing size; pricing a tiled
    /// forward at the whole padded image.
    #[test]
    fn the_encoder_free_routes_are_admitted_at_the_size_the_model_runs() {
        let depth = depth_memory_geometry(&DepthInputPlan {
            source_width: 4000,
            source_height: 3000,
            model_width: 1024,
            model_height: 768,
        });
        assert_eq!(
            (
                depth.width,
                depth.height,
                depth.batch,
                depth.frames,
                depth.reference_count
            ),
            (1024, 768, 1, 1, 0)
        );

        let restore = |width, height, scale| {
            plan(
                Dims::new(width, height),
                &RestorationOptions {
                    scale,
                    input_sizing: InputSizing::Original,
                    color_fix: true,
                },
                TileGeometry::RELEASE,
                "iris_3b_restore",
            )
            .unwrap()
        };
        // 600 x 400 at 4x is 2400 x 1600: larger than one release tile on both sides, so tiled.
        let tiled = restore(600, 400, 4.0);
        assert!(!tiled.single_pass());
        let geometry = restoration_memory_geometry(&tiled);
        assert_eq!(
            (geometry.width, geometry.height),
            (tiled.processing.width, tiled.processing.height)
        );
        assert_eq!((geometry.width, geometry.height), (2400, 1600));
        assert_eq!(
            (geometry.batch, geometry.frames, geometry.reference_count),
            (1, 1, 0)
        );
        assert_eq!(restoration_forward_pixels(&tiled), 1024 * 1024);
        assert!(
            restoration_forward_pixels(&tiled)
                < u64::from(tiled.padded.width) * u64::from(tiled.padded.height)
        );

        // 100 x 50 at 2x is 200 x 100: enlarged so its short side is one tile, and the long side
        // then exceeds a tile, so the admitted size is the processing size, not the output.
        let enlarged = restore(100, 50, 2.0);
        assert!(enlarged.enlarged);
        let geometry = restoration_memory_geometry(&enlarged);
        assert_eq!(
            (geometry.width, geometry.height),
            (enlarged.processing.width, enlarged.processing.height)
        );
        assert_ne!(
            (geometry.width, geometry.height),
            (enlarged.output.width, enlarged.output.height)
        );

        // A square one-tile image runs as a single forward over the whole padded image. Under the
        // planner's law that image is always exactly one tile (the short side is enlarged to a
        // tile and a single pass needs both padded sides within one), so the single-pass arm and
        // the tiled arm of `restoration_forward_pixels` agree by construction: the function is
        // the one-tile bound on every plan. The arm is kept to state the bound in the plan's own
        // terms, as the Candle twin does; a mutation of it alone is therefore not observable.
        let single = restore(256, 256, 4.0);
        assert!(single.single_pass());
        assert_eq!((single.padded.width, single.padded.height), (1024, 1024));
        assert_eq!(
            restoration_forward_pixels(&single),
            u64::from(single.padded.width) * u64::from(single.padded.height)
        );

        // A loaded encoder-free model checks against the precision it computes in.
        for task in [IrisTask::Depth, IrisTask::Restoration] {
            let contract = weights_free(task);
            let own = context(task, &contract, MemoryStrategy::Resident);
            assert_eq!(
                loaded_safety_check(Precision::Bf16, &contract, &own),
                MemorySafetyDecision::Accept
            );
            assert!(matches!(
                loaded_safety_check(Precision::Fp32, &contract, &own),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    /// The route gate behind the shared handshake.
    ///
    /// *Mutations that red this:* removing the mode check (another task's context is admitted);
    /// passing `None` for the loaded tier (the precision flip is admitted); removing the batch
    /// gate on the encoder-free tasks.
    #[test]
    fn the_safety_check_admits_its_own_route_and_rejects_the_others() {
        let spec = sentinel();
        for task in TASKS {
            let contract = weights_free(task);
            let own = context(task, &contract, MemoryStrategy::Resident);
            assert_eq!(
                safety_check(&spec, &contract, &own),
                MemorySafetyDecision::Accept,
                "{}",
                provider_id(task)
            );
            let rejects = |context: &MemoryRunContext, what: &str| {
                assert!(
                    matches!(
                        safety_check(&spec, &contract, context),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{}: {what} must be rejected",
                    provider_id(task)
                );
            };
            for other in TASKS.into_iter().filter(|other| *other != task) {
                let mut foreign = own.clone();
                foreign.mode = memory_mode(other);
                rejects(&foreign, "another task's mode");
            }
            let mut precision = own.clone();
            precision.selection.tier.precision = Precision::Fp32;
            rejects(&precision, "a tier the spec did not load");
            let mut reference = own.clone();
            reference.geometry.reference_count = 1;
            reference.has_reference = true;
            rejects(&reference, "a reference image");
            let mut pid = own.clone();
            pid.use_pid = true;
            rejects(&pid, "a PiD decode");
            let mut clip = own.clone();
            clip.geometry.frames = 2;
            rejects(&clip, "a multi-frame geometry");

            let mut batch = own.clone();
            batch.geometry.batch = 2;
            if task == IrisTask::Generation {
                assert_eq!(
                    safety_check(&spec, &contract, &batch),
                    MemorySafetyDecision::Accept
                );
            } else {
                rejects(&batch, "a batch of two");
                // The staged rung does not exist on an encoder-free route.
                let mut staged = own.clone();
                staged.selection.strategy = MemoryStrategy::StagedResidency;
                rejects(&staged, "a staged selection");
            }
        }
    }

    /// The executable seam: generation's scope writes the staged selection onto the request that
    /// `Iris3b::generate` reads; the encoder-free routes admit their resident baseline with no
    /// scope and publish no optimized fixture.
    ///
    /// *Mutations that red this:* returning `Ok(None)` for generation; declaring
    /// `ResidentRequestMemory::ExplicitResident` (a resident selection would then write a block);
    /// opening a scope for depth or restoration.
    #[test]
    fn the_generation_scope_stages_the_request_and_the_encoder_free_routes_open_none() {
        let spec = sentinel();
        let contract = weights_free(IrisTask::Generation);
        for strategy in MemoryStrategy::ALL {
            let fixtures = registered_valid_fixture(&spec, &contract, strategy).unwrap();
            assert_eq!(
                fixtures.len(),
                usize::from(strategy == MemoryStrategy::StagedResidency),
                "{strategy:?}"
            );
        }
        let mut fixture =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0);
        assert_eq!(fixture.context.mode, MemoryMode::TextToImage);
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .expect("generation opens a request scope");
        assert_eq!(fixture.request.memory, None);
        scope.configure_request(&mut fixture.request).unwrap();
        assert_eq!(
            fixture.request.memory,
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
        assert!(crate::model::stage_residency_for(&spec, &fixture.request));
        // Pixel space: there is no decoder for a scope to tile.
        assert!(scope
            .configure_decode(1024, 64, fixture.context.geometry)
            .is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        // A resident selection leaves the request untouched: the load policy decides.
        let resident = context(IrisTask::Generation, &contract, MemoryStrategy::Resident);
        let mut request = MemoryBehaviorFixture::new(resident.clone()).request;
        let mut scope = registered_begin_request(&spec, &contract, &resident)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut request).unwrap();
        assert_eq!(request.memory, None);
        assert!(!crate::model::stage_residency_for(&spec, &request));
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        for task in [IrisTask::Depth, IrisTask::Restoration] {
            let contract = weights_free(task);
            for strategy in MemoryStrategy::ALL {
                assert!(registered_valid_fixture(&spec, &contract, strategy)
                    .unwrap()
                    .is_empty());
            }
            let own = context(task, &contract, MemoryStrategy::Resident);
            assert!(registered_begin_request(&spec, &contract, &own)
                .unwrap()
                .is_none());
            let mut foreign = own;
            foreign.mode = MemoryMode::TextToImage;
            assert!(registered_begin_request(&spec, &contract, &foreign).is_err());
        }
    }

    /// The explicit catalog carries all three routes through the shared seams and conforms.
    ///
    /// *Mutations that red this:* dropping any registration in `register_providers`; registering
    /// depth or restoration through `register_memory_strategy` (the builder refuses an id with no
    /// generator); withdrawing the weights-free patch axis (an MLX surface must declare one).
    #[test]
    fn the_catalog_registers_all_three_routes_and_conforms() {
        let registry = crate::provider_registry().unwrap();
        let ids: Vec<_> = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(ids, ["iris_3b", "iris_3b_depth", "iris_3b_restore"]);
        let fixtures: Vec<_> = registry
            .memory_contract_fixture_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(fixtures, ids);
        let behaviors: Vec<_> = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(behaviors, ids);
        assert_eq!(
            registry.resident_only_memory_contract_registrations().len(),
            0
        );

        let surfaces = registry.memory_contract_surfaces().unwrap();
        // Generation: three tiers x two policies (sc-26266); depth and restoration: bf16 x two.
        assert_eq!(surfaces.len(), 3 * 2 + 2 + 2);
        assert!(surfaces
            .iter()
            .all(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization));
        // Depth is a provider-specific API and restoration a transform: both are composed routes.
        for surface in &surfaces {
            assert_eq!(
                surface.composed,
                surface.contract.provider_id != "iris_3b",
                "{}",
                surface.contract.provider_id
            );
        }
        gen_core_testkit::memory_contract_surface_registry_conformance(&registry);
        let coverage =
            gen_core_testkit::memory_contract_surface_registry_facts_conformance(&registry, None);
        assert_eq!(coverage.surfaces_checked, 10);
        for load_shape in [
            LoadShape::EagerMaterialization,
            LoadShape::DeferredMaterialization,
        ] {
            gen_core_testkit::memory_strategy_registry_conformance(
                &registry,
                &sentinel().with_load_shape(load_shape),
            );
        }
        // Measurement-only carrier: nothing derived is filed as an activation anchor.
        assert_eq!(
            registry.activation_memory_bytes_1024("iris_3b").unwrap(),
            None
        );
    }

    /// A materialized snapshot prices what each loader holds, at the width it holds it.
    ///
    /// *Mutations that red this:* pricing the backbone `Stored` (the bf16 load would charge the
    /// f32 file); casting the text position table; pricing the vision tower; pricing the text
    /// encoder at the backbone's precision; folding the empty prompt into the backbone.
    #[test]
    fn a_materialized_snapshot_prices_what_each_loader_holds() {
        let matrix = |header: &SafetensorsTensorHeader| {
            header.shape.len() >= 2 && !header.name.ends_with("y_pos_embedding")
        };
        for task in TASKS {
            let spec = tiny_spec(task);
            let weights = match &spec.weights {
                WeightsSource::Dir(dir) => dir.join("model.safetensors"),
                WeightsSource::File(_) => unreachable!(),
            };
            let matrix_elements = sum(&weights, |h| if matrix(h) { elements(h) } else { 0 });
            let stored_rest = sum(&weights, |h| if matrix(h) { 0 } else { h.data_bytes });
            assert!(matrix_elements > 0 && stored_rest > 0);

            let bf16 = loaded(task, &spec);
            gen_core_testkit::memory_strategy_conformance(&bf16);
            gen_core_testkit::assert_memory_contract_facts_conform(&bf16);
            let facts = bf16.asset_facts;
            assert_eq!(facts.transformer_bytes, matrix_elements * 2 + stored_rest);
            assert_eq!(facts.decoder_bytes, 0, "pixel space");
            assert_eq!(facts.overlay_bytes, 0);
            assert_eq!(
                facts.base_bytes,
                facts.conditioning_bytes + facts.transformer_bytes
            );

            let mut fp32_spec = spec.clone();
            fp32_spec.precision = Precision::Fp32;
            let fp32 = loaded(task, &fp32_spec);
            assert_eq!(
                fp32.asset_facts.transformer_bytes,
                matrix_elements * 4 + stored_rest
            );
            // The conditioning never follows the backbone's precision: the text tower is bf16
            // whatever the backbone computes in, and the empty prompt is stored f32.
            assert_eq!(
                fp32.asset_facts.conditioning_bytes,
                facts.conditioning_bytes
            );
            assert_eq!(fp32.architecture_facts.activation_dtype_width, Some(4));
            assert_eq!(bf16.architecture_facts.activation_dtype_width, Some(2));

            match task {
                IrisTask::Generation => {
                    let tower = fixtures().join("tiny-snapshot/text_encoder/model.safetensors");
                    let language = sum(&tower, |h| {
                        if h.name.starts_with("model.language_model.") {
                            elements(h)
                        } else {
                            0
                        }
                    });
                    let everything = sum(&tower, elements);
                    assert!(language < everything, "the fixture carries a vision tower");
                    assert_eq!(facts.conditioning_bytes, language * 2);
                }
                _ => {
                    let prompt = weights.with_file_name("empty_prompt.safetensors");
                    assert_eq!(facts.conditioning_bytes, sum(&prompt, |h| h.data_bytes));
                }
            }

            // The architecture axes are the checkpoint's own config, not the weights-free grid.
            let config = IrisConfig::from_dir(weights.parent().unwrap()).unwrap();
            let axes = bf16.architecture_facts;
            assert_eq!(axes.attention_heads, Some(config.model.num_heads as u32));
            assert_eq!(axes.transformer_blocks, Some(config.model.depth as u32));
            assert_eq!(axes.patch_size, Some(config.model.patch_size as u32));
            assert_eq!(
                axes.head_dim,
                Some((config.model.hidden_size / config.model.num_heads) as u32)
            );
            assert_eq!(axes.latent_channels, None);
            assert_eq!(axes.vae_spatial_scale, None);
            assert_eq!(
                weights_free(task).architecture_facts.patch_size,
                Some(gen_core::iris::SIZE_MULTIPLE)
            );
            assert_eq!(weights_free(task).architecture_facts.attention_heads, None);

            // The loaded contract is the weights-free declaration plus facts, nothing else.
            let mut declaration = bf16.clone();
            declaration.asset_facts = MemoryAssetFacts::default();
            declaration.architecture_facts = weights_free(task).architecture_facts;
            assert_eq!(
                declaration,
                weights_free_memory_strategy_contract(provider_id(task), &spec).unwrap()
            );
        }
    }

    /// The miniature text tower with an MLP width every affine group divides (its released 48 does
    /// not, so the miniature has no quantized tier): `config.json` with `intermediate_size: 64` and
    /// a `model.safetensors` of the same keys with the MLP shapes widened, zero-filled. Pricing
    /// reads headers only. Returns the snapshot directory.
    fn write_aligned_tiny_text_encoder(root: &Path) -> PathBuf {
        let source = fixtures().join("tiny-snapshot/text_encoder");
        let dir = root.join("text_encoder");
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&source).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_owned();
            if name != "model.safetensors" && name != "config.json" {
                std::fs::copy(&path, dir.join(name)).unwrap();
            }
        }
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(config["text_config"]["intermediate_size"], 48);
        config["text_config"]["intermediate_size"] = 64.into();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        for tensor in safetensors_path_tensor_headers(source.join("model.safetensors")).unwrap() {
            let mut shape = tensor.shape.clone();
            if tensor.name.contains(".mlp.gate_proj.") || tensor.name.contains(".mlp.up_proj.") {
                shape[0] = 64;
            } else if tensor.name.contains(".mlp.down_proj.") {
                shape[1] = 64;
            }
            let bytes = shape.iter().product::<usize>() * 4;
            header.insert(
                tensor.name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let header = serde_json::Value::Object(header).to_string().into_bytes();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.resize(file.len() + offset, 0);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        dir
    }

    /// sc-26266: a quantized generation load prices what its loaders hold at the tier — every
    /// backbone projection packed (the unaligned ones over a zero-padded input), the text tower's
    /// attention/MLP projections packed at its group size, everything else as on the dense tier —
    /// and an adapted load prices the same packed backbone (the tier is one layout whatever the
    /// adapters; the stack is an overlay). Expectations are recomputed here from the headers, not
    /// through the provider's helpers.
    ///
    /// *Mutations that red this:* pricing a quantized load at bf16; quantizing the token
    /// embedding; dropping the padding (with or without adapters); a scale/bias table at the
    /// wrong width.
    #[test]
    fn a_quantized_generation_load_prices_the_packed_projections() {
        let scratch = tempfile::tempdir().unwrap();
        let tower = write_aligned_tiny_text_encoder(scratch.path());
        let mut spec = tiny_spec(IrisTask::Generation);
        spec.components.insert(
            gen_core::iris::TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(tower.clone()),
        );
        let packed = |out: u64, padded_in: u64, bits: u64, group: u64| {
            out * padded_in * bits / 8 + out * (padded_in / group) * 4
        };
        let backbone = fixtures().join("tiny-snapshot/iris/model.safetensors");
        let backbone_at = |bits: Option<u64>, pad: bool| {
            sum(&backbone, |h| match (h.shape.as_slice(), bits) {
                ([out, input], Some(bits))
                    if h.name.ends_with(".weight") && *input >= 64 && (input % 64 == 0 || pad) =>
                {
                    packed(*out as u64, input.div_ceil(64) as u64 * 64, bits, 64)
                }
                (shape, _) if shape.len() >= 2 && !h.name.ends_with("y_pos_embedding") => {
                    elements(h) * 2
                }
                _ => h.data_bytes,
            })
        };
        let tower_at = |bits: Option<u64>| {
            sum(&tower.join("model.safetensors"), |h| {
                if !h.name.starts_with("model.language_model.") {
                    return 0;
                }
                match (h.shape.as_slice(), bits) {
                    ([out, input], Some(bits))
                        if h.name.ends_with(".weight")
                            && !h.name.ends_with("embed_tokens.weight") =>
                    {
                        // The miniature's hidden width (32) is not a multiple of 64.
                        packed(*out as u64, *input as u64, bits, 32)
                    }
                    _ => elements(h) * 2,
                }
            })
        };
        // The miniature carries an unaligned projection (the 170-wide SwiGLU down projections),
        // so padding is observable.
        assert!(backbone_at(Some(4), true) < backbone_at(Some(4), false));

        let bf16 = loaded(IrisTask::Generation, &spec);
        assert_eq!(bf16.asset_facts.transformer_bytes, backbone_at(None, true));
        assert_eq!(bf16.asset_facts.conditioning_bytes, tower_at(None));
        let mut previous = bf16.asset_facts;
        for (quant, bits) in [(Quant::Q8, 8), (Quant::Q4, 4)] {
            let contract = loaded(IrisTask::Generation, &spec.clone().with_quant(quant));
            gen_core_testkit::memory_strategy_conformance(&contract);
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
            let facts = contract.asset_facts;
            assert_eq!(
                facts.transformer_bytes,
                backbone_at(Some(bits), true),
                "{quant:?}"
            );
            assert_eq!(facts.conditioning_bytes, tower_at(Some(bits)), "{quant:?}");
            assert_eq!(
                facts.base_bytes,
                facts.transformer_bytes + facts.conditioning_bytes
            );
            assert!(
                facts.transformer_bytes < previous.transformer_bytes,
                "{quant:?}"
            );
            assert!(
                facts.conditioning_bytes < previous.conditioning_bytes,
                "{quant:?}"
            );
            previous = facts;
        }

        let mut adapted = spec.clone().with_quant(Quant::Q4);
        adapted.adapters = vec![AdapterSpec::new(
            fixtures().join("iris_lora.safetensors"),
            1.0,
            AdapterKind::Lora,
        )];
        assert_eq!(
            loaded(IrisTask::Generation, &adapted)
                .asset_facts
                .transformer_bytes,
            backbone_at(Some(4), true)
        );

        // The released miniature tower has no affine group, so it has no quantized tier.
        match memory_strategy_contract(
            MODEL_ID,
            &tiny_spec(IrisTask::Generation).with_quant(Quant::Q4),
        ) {
            Err(CoreError::Unsupported(message)) => {
                assert!(message.contains("group size"), "{message}")
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }

    /// The loaded tier names the quantization (sc-26266), so a selection at another tier — or a
    /// dense load handed a quantized selection — is rejected by the shared handshake.
    ///
    /// *Mutation that reds this:* `loaded_tier` dropping `spec.quantize`.
    #[test]
    fn a_selection_must_name_the_tier_the_generator_loaded() {
        let route = || MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        };
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut spec = sentinel();
            spec.quantize = quant;
            let contract = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
            for selected in [None, Some(Quant::Q8), Some(Quant::Q4)] {
                let mut at = sentinel();
                at.quantize = selected;
                let context = gen_core::standard_memory_behavior_context(
                    &contract,
                    MemoryStrategy::Resident,
                    loaded_tier(&at),
                    route(),
                )
                .unwrap();
                let decision = safety_check(&spec, &contract, &context);
                if selected == quant {
                    assert_eq!(decision, MemorySafetyDecision::Accept, "{quant:?}");
                } else {
                    assert!(
                        matches!(decision, MemorySafetyDecision::Reject { .. }),
                        "{quant:?} load, {selected:?} selection"
                    );
                }
            }
        }
    }

    /// An adapter stack is an overlay, never base bytes: the factors at their file size, plus the
    /// full `[out, in]` delta of every LoKr module at the compute width.
    ///
    /// *Mutations that red this:* dropping the materialized-delta component; pricing the deltas at
    /// a fixed bf16 width; adding the overlay into `base_bytes`.
    #[test]
    fn an_adapter_stack_is_priced_as_typed_overlay_components() {
        let clean = loaded(IrisTask::Generation, &tiny_spec(IrisTask::Generation));
        assert!(matches!(
            clean.formula,
            MemoryFormulaKind::PhaseEnvelope { .. }
        ));
        let with = |file: &str, kind: AdapterKind, precision: Precision| {
            let mut spec = tiny_spec(IrisTask::Generation);
            spec.precision = precision;
            spec.adapters = vec![AdapterSpec::new(fixtures().join(file), 1.0, kind)];
            let contract = loaded(IrisTask::Generation, &spec);
            gen_core_testkit::memory_strategy_conformance(&contract);
            contract
        };
        let components = |contract: &MemoryProviderContract| match &contract.formula {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                variables,
                resident_components,
                ..
            } => {
                assert!(variables.contains(&MemoryFormulaVariable::OverlayBytes));
                resident_components
                    .iter()
                    .map(|component| (component.id.clone(), component.resident_bytes))
                    .collect::<Vec<_>>()
            }
            other => panic!("an adapter load must declare typed components, got {other:?}"),
        };

        let lora_file = mlx_gen::safetensors_path_bytes(fixtures().join("iris_lora.safetensors"));
        assert!(lora_file > 0);
        let lora = with("iris_lora.safetensors", AdapterKind::Lora, Precision::Bf16);
        assert_eq!(
            components(&lora),
            [(ADAPTER_RESIDUALS_COMPONENT_ID.to_owned(), lora_file)]
        );
        assert_eq!(lora.asset_facts.overlay_bytes, lora_file);
        assert_eq!(lora.asset_facts.base_bytes, clean.asset_facts.base_bytes);

        // The fixture's four LoKr modules target the same projections as the LoRA fixture:
        // `[64, 64]`, `[64, 64]`, `[512, 64]` and `[64, 32]` — every one dense, so every one is
        // reconstructed in full.
        let delta_elements = 64 * 64 + 64 * 64 + 512 * 64 + 64 * 32;
        let lokr_file = mlx_gen::safetensors_path_bytes(fixtures().join("iris_lokr.safetensors"));
        for (precision, width) in [(Precision::Bf16, 2), (Precision::Fp32, 4)] {
            let lokr = with("iris_lokr.safetensors", AdapterKind::Lokr, precision);
            assert_eq!(
                components(&lokr),
                [
                    (ADAPTER_RESIDUALS_COMPONENT_ID.to_owned(), lokr_file),
                    (
                        ADAPTER_DELTAS_COMPONENT_ID.to_owned(),
                        delta_elements * width
                    ),
                ]
            );
            assert_eq!(
                lokr.asset_facts.overlay_bytes,
                lokr_file + delta_elements * width
            );
        }
    }
}
