//! Iris-3B Candle request-memory contracts (sc-25690) — one per inference task, each published
//! through the shared seams the other Candle families use (`gen_core::MemoryRegistration`, its
//! weights-free contract fixture, and the executable behavior registration).
//!
//! | rung | strategy | `iris_3b` (generation) | `iris_3b_depth` | `iris_3b_restore` |
//! |---|---|---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | **implemented** | **implemented** | **implemented** |
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — `Residency::run_request_scoped` loads the Qwen3-VL tower, encodes, drops it, then loads the backbone | not applicable — no text encoder is loaded (the export ships its empty-prompt states); the backbone is the only component | not applicable — same |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | not applicable — pixel space, there is no decoder | not applicable — same | not applicable — same (the 50 %-overlap tiling is the forward itself, see below) |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | **implemented** — [`crate::nn::with_attention_budget`] bounds every attention's scores tensor | missing — `DepthRequest` carries no `GenerationMemory` to select a budget with | missing — `TransformRequest` carries none either |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | missing — no block-streaming loader | missing | missing |
//!
//! # What each task holds
//!
//! * **Generation** is two components with different lifetimes: the Qwen3-VL language tower
//!   (conditioning) and the backbone (denoise). Pixel space, so the "decode" phase is only the
//!   `[-1, 1]` → RGB8 conversion.
//! * **Depth** and **restoration** are one backbone forward on the export's shipped empty-prompt
//!   states. They are not gen-core [`Generator`](gen_core::Generator)s — depth is an
//!   `IrisDepthEstimator`, restoration a `Transform` — so their registrations use the composed-route
//!   seam, and no request scope exists for them: neither request type carries a memory block.
//! * **Restoration tiling is unconditional, not a selectable rung.** An output larger than one tile
//!   is restored as 50 %-overlap tiles of the export's `task.tile` side (1024 px in the release), one
//!   backbone forward per tile, fused on the host. The device activation therefore saturates at one
//!   tile ([`restoration_forward_pixels`]) while the host canvases grow with the output; nothing a
//!   request sends turns it on, off, or resizes it, which is why it is not declared as rung 2.
//!
//! # Derived, never measured
//!
//! Every byte figure here is read from safetensors **headers** at the width the loader holds each
//! tensor in — the backbone, the text encoder, the empty-prompt states and the adapter stack alike
//! ([`crate::adapters::adapter_footprint`] reads no adapter tensor). Nothing is an on-device
//! observation, and pricing a contract costs a few header reads however often a caller asks. The
//! published attention budget is the shared 64 Mi operating point, a derived candidate pending the
//! epic-end campaign, not a measurement on Iris.
//!
//! Every loader prices its contract before it reads a weight and propagates a pricing error, so a
//! spec with no contract — a deferred materialization shape, an incomplete text encoder, a refused
//! adapter — fails `load` with that reason on every offload policy.
//!
//! # What bounded attention does to the image
//!
//! Chunking the query rows (and, for the PiT pixel stage, the batch rows) computes the same
//! attention: rows are independent, so the result equals the single pass to GEMM-order f32
//! rounding. On the FP32 path a bounded render is the unbounded one to within one RGB8 level.
//! Under the bf16 policy — the only tier these contracts witness — every projection re-rounds its
//! output to bf16, so that f32 ulp can become one bf16 step and travel through the remaining
//! blocks and solver steps: a bounded bf16 render is **not** bit-identical to the unbounded one.
//! The declared bound, held by `tests/memory_strategy.rs` on the miniature snapshot (Candle CPU,
//! three seeds × three binding budgets): no channel moves by more than **8 RGB8 levels** and the
//! mean absolute change is at most **1.0 level** (observed on arm64 macOS: 2 levels and 0.20). It
//! is a bound on the miniature, not a measurement of the release weights on CUDA.

use std::collections::BTreeSet;
use std::path::Path;

use candle_gen::candle_core::DType;
use candle_gen::gen_core::iris::depth::DepthInputPlan;
use candle_gen::gen_core::iris::downstream::{TaskExport, EMPTY_PROMPT_FILE};
use candle_gen::gen_core::iris::restoration::{RestorationPlan, RestorationResources};
use candle_gen::gen_core::iris::{
    GenerationResources, IrisConfig, IrisTask, TextEncoderConfig, BACKBONE_WEIGHTS_FILE,
};
use candle_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity,
    MemoryContractSurfaceSpec, MemoryContractSurfaceTier, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, MemoryWindowMaterialization, ResidentRequestMemory,
};

use crate::depth::DEPTH_MODEL_ID;
use crate::model::{compute_dtype, MODEL_ID};
use crate::nn::{BackboneQuant, ProjectionLayout};
use crate::restoration::MODEL_ID as RESTORATION_MODEL_ID;

/// Content fingerprint of the generation contract. Distinct from the MLX twin's: the two backends
/// never share calibration evidence.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "iris-3b-candle-derived-2026-10-12-v1";
/// Content fingerprint of the depth contract.
pub const DEPTH_MEMORY_CALIBRATION_FINGERPRINT: &str = "iris-3b-depth-candle-derived-2026-10-12-v1";
/// Content fingerprint of the restoration contract.
pub const RESTORE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "iris-3b-restore-candle-derived-2026-10-12-v1";

/// The retained calibration fingerprint of `task`'s DENSE bf16 contract (the strings consumers
/// already key on). Every other tier has its own identity: [`calibration_fingerprint_for`].
pub const fn calibration_fingerprint(task: IrisTask) -> &'static str {
    match task {
        IrisTask::Generation => MEMORY_CALIBRATION_FINGERPRINT,
        IrisTask::Depth => DEPTH_MEMORY_CALIBRATION_FINGERPRINT,
        IrisTask::Restoration => RESTORE_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// The calibration identity of `task` at the tier the load computes and packs in, keyed on
/// **(task, tier)** — the MLX twin's rule (sc-22731). One string per task across tiers would let a
/// q8 record satisfy a q4 request (and vice versa), though the packed widths and so the footprints
/// differ. The dense bf16 cell keeps its retained string byte-for-byte; the fp32 dense reference
/// and the q8 / q4 packed tiers each get their own.
pub fn calibration_fingerprint_for(
    task: IrisTask,
    precision: gen_core::Precision,
    quant: Option<gen_core::Quant>,
) -> String {
    let tier = match (quant, precision) {
        (None, gen_core::Precision::Bf16) => return calibration_fingerprint(task).to_owned(),
        (None, gen_core::Precision::Fp32) => "fp32",
        (Some(gen_core::Quant::Q8), _) => "q8",
        (Some(gen_core::Quant::Q4), _) => "q4",
        // Refused before any contract is built (`refuse_unloadable_quant`); never published.
        (Some(_), _) => "unsupported",
    };
    let route = match task {
        IrisTask::Generation => "iris-3b",
        IrisTask::Depth => "iris-3b-depth",
        IrisTask::Restoration => "iris-3b-restore",
    };
    format!("{route}-{tier}-candle-derived-2026-10-12-v1")
}

/// The published attention score budget (elements of one `[rows, heads, queries, keys]` scores
/// tensor): the shared rung-3 operating point
/// ([`candle_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`], 64 Mi), like the sibling families.
/// A **derived candidate, pending the epic-end campaign** — nothing here measured it on Iris. The
/// floor one kernel launch can reach is a single query row of a single batch row, `heads · keys` —
/// 333,680 elements for the release trunk at 2048² (20 heads × (128² + 300) keys) — so the budget
/// is honoured at every admitted geometry.
pub const ATTENTION_CHUNK_SIZE: u32 = 67_108_864;

/// `MemoryRunContext::mode` key of the depth route ([`IrisTask::name`]).
pub const DEPTH_MEMORY_MODE: &str = "depth";
/// `MemoryRunContext::mode` key of the restoration route ([`IrisTask::name`]).
pub const RESTORE_MEMORY_MODE: &str = "restoration";

// ================================================================================================
// Pricing.
// ================================================================================================

/// Whether the build's production device ([`candle_gen::default_device`]) is the Candle CPU
/// backend. The contract prices the production device without opening it.
fn production_device_is_cpu() -> bool {
    !cfg!(any(feature = "cuda", feature = "metal"))
}

/// The dtype the Qwen3-VL tower is materialized in on the production device — the same rule as
/// [`crate::text_encoder::tower_dtype`]: the backbone's `text_encoder.dtype` on a GPU, f32 on the
/// Candle CPU backend.
pub fn production_tower_dtype(cfg: &TextEncoderConfig) -> gen_core::Result<DType> {
    let declared = crate::text_encoder::declared_tower_dtype(cfg)?;
    Ok(if production_device_is_cpu() {
        DType::F32
    } else {
        declared
    })
}

/// The dtype a GEMM over `compute` runs in on the production device (`nn::kernel_dtype`): itself,
/// except a half-precision operand on the Candle CPU backend, which accumulates in f32.
fn production_kernel_dtype(compute: DType) -> DType {
    if production_device_is_cpu() && matches!(compute, DType::BF16 | DType::F16) {
        DType::F32
    } else {
        compute
    }
}

fn tensor_bytes(header: &SafetensorsTensorHeader, float_width: u64) -> gen_core::Result<u64> {
    if header.is_float() {
        header.materialized_bytes(float_width)
    } else {
        Ok(header.data_bytes)
    }
}

fn checked_sum(
    model_id: &str,
    parts: impl Iterator<Item = gen_core::Result<u64>>,
) -> gen_core::Result<u64> {
    gen_core::checked_byte_total(
        model_id,
        "component",
        parts.collect::<gen_core::Result<Vec<_>>>()?,
    )
}

/// Candle's CUDA `MATRIX_ROW_PADDING` (elements): every GGML tensor candle quantizes onto a CUDA
/// device is allocated this many elements' worth of blocks past its payload.
const GGML_CUDA_ROW_PADDING: u64 = 512;

/// Device bytes of a packed `[out, in]` projection: its GGML blocks ([`ProjectionLayout`]) plus,
/// on a CUDA build, the row padding candle allocates past them.
fn packed_bytes(layout: ProjectionLayout, out: u64, in_features: u64) -> u64 {
    let blocks = layout.weight_bytes(out, in_features, 0);
    match layout {
        ProjectionLayout::Quantized { dtype, .. } if cfg!(feature = "cuda") => {
            blocks + GGML_CUDA_ROW_PADDING / dtype.block_size() as u64 * dtype.type_size() as u64
        }
        _ => blocks,
    }
}

/// Bytes an Iris backbone checkpoint occupies once [`IrisDiT`](crate::IrisDiT) is built from it at
/// a `compute_width`-byte compute dtype and the load-time quantization `quant` (`None` is the
/// dense tier).
///
/// The loader's rule ([`crate::nn::Loader`]): a projection's `‹path›.weight` (every tensor of rank
/// ≥ 2 under that name — the Linears, and the depth export's 1×1 `depth_reducer` kernel) and its
/// sibling `‹path›.bias` are held in the compute dtype — except that on a quantized tier an
/// `[out, in]` weight is held as [`ProjectionLayout::of`] says (packed over its zero-padded input,
/// sc-26266); every other tensor — RMSNorm gains, the shared-modulation bias tables, the text
/// position table — stays f32. Held to the loader's own recorded widths by
/// `the_backbone_is_priced_at_the_widths_the_loader_holds_each_tensor_in`, and to a quantized
/// load's measured bytes by `tests/quant_tiers.rs`.
pub fn backbone_bytes(
    file: &Path,
    compute_width: u64,
    quant: Option<BackboneQuant>,
) -> gen_core::Result<u64> {
    let headers = safetensors_path_tensor_headers(file)?;
    let projections: BTreeSet<&str> = headers
        .iter()
        .filter(|header| header.shape.len() >= 2)
        .filter_map(|header| header.name.strip_suffix(".weight"))
        .collect();
    checked_sum(
        "iris",
        headers.iter().map(|header| {
            if let ([out, input], Some(_)) =
                (header.shape.as_slice(), header.name.strip_suffix(".weight"))
            {
                let layout = ProjectionLayout::of(*input, quant);
                if layout != ProjectionLayout::Dense {
                    return Ok(packed_bytes(layout, *out as u64, *input as u64));
                }
            }
            let in_compute = match (
                header.name.strip_suffix(".weight"),
                header.name.strip_suffix(".bias"),
            ) {
                (Some(_), _) => header.shape.len() >= 2,
                (_, Some(stem)) => projections.contains(stem),
                _ => false,
            };
            tensor_bytes(header, if in_compute { compute_width } else { 4 })
        }),
    )
}

/// Whether [`IrisTextEncoder::load`](crate::IrisTextEncoder::load) keeps the checkpoint tensor
/// `name` resident: the language tower under `model.language_model.*`, plus the root
/// `lm_head.weight` when the head is untied (the shared `CausalLm` holds it; a tied head is a
/// handle to the embedding). The vision tower (`model.visual.*`) never reaches the device.
pub fn text_encoder_keeps(name: &str, tie_word_embeddings: bool) -> bool {
    name.starts_with("model.language_model.") || (name == "lm_head.weight" && !tie_word_embeddings)
}

/// Key prefix of the language tower's decoder layers — every rank-2 `.weight` under it is an
/// attention or MLP projection the shared decoder quantizes on a quantized tier.
const TEXT_ENCODER_LAYERS_PREFIX: &str = "model.language_model.layers.";

/// Bytes the Qwen3-VL language tower occupies once loaded at a `tower_width`-byte dtype and the
/// tier `quant`, over the exact shard files the loader opens.
///
/// On a quantized tier (sc-26266) every attention / MLP projection — each rank-2 `.weight` of a
/// decoder layer — is held in the GGML block form the shared decoder quantizes it to
/// ([`crate::text_encoder::text_encoder_quant_spec`], `QuantSpec::dtype_for_in_dim`); the token
/// embedding, an untied head and the norms stay at `tower_width`.
pub fn text_encoder_bytes(
    dir: &Path,
    tower_width: u64,
    quant: Option<gen_core::Quant>,
) -> gen_core::Result<u64> {
    let config = candle_llm::ModelConfig::from_dir(dir)
        .map_err(|e| gen_core::Error::Msg(format!("{MODEL_ID}: text encoder config: {e}")))?;
    let tied = config.tie_word_embeddings;
    let spec = crate::text_encoder::text_encoder_quant_spec(&config, quant)?;
    let mut total = 0_u64;
    for file in crate::text_encoder::shard_files(dir)? {
        let headers = safetensors_path_tensor_headers(&file)?;
        let part = checked_sum(
            MODEL_ID,
            headers
                .iter()
                .filter(|header| text_encoder_keeps(&header.name, tied))
                .map(|header| match (spec, header.shape.as_slice()) {
                    (Some(spec), [out, input])
                        if header.name.starts_with(TEXT_ENCODER_LAYERS_PREFIX)
                            && header.name.ends_with(".weight") =>
                    {
                        let dtype = spec.dtype_for_in_dim(*input);
                        Ok(packed_bytes(
                            ProjectionLayout::Quantized {
                                dtype,
                                padded_in: *input,
                            },
                            *out as u64,
                            *input as u64,
                        ))
                    }
                    _ => tensor_bytes(header, tower_width),
                }),
        )?;
        total = total.checked_add(part).ok_or_else(|| {
            gen_core::Error::Msg(format!("{MODEL_ID}: text encoder byte sum overflow"))
        })?;
    }
    Ok(total)
}

/// Device bytes of a downstream export's shipped empty-prompt conditioning: the `[1, T, L, D]`
/// `embeddings` tensor, uploaded as f32. Its mask stays on the host.
pub fn empty_prompt_bytes(file: &Path) -> gen_core::Result<u64> {
    checked_sum(
        "iris",
        safetensors_path_tensor_headers(file)?
            .iter()
            .filter(|header| header.name == "embeddings")
            .map(|header| header.materialized_bytes(4)),
    )
}

/// Component id of the adapter residuals the forward applies as factors (LoRA, structured LoKr,
/// low-rank LoHa). The MLX twin publishes the same id.
pub const ADAPTER_RESIDUALS_COMPONENT_ID: &str = "iris_3b.adapters.forward_residuals";
/// Component id of the adapter residuals held as a full `[out, in]` delta (a LoHa at or above its
/// break-even rank, a LoKr whose factors are not plain matrices). The MLX twin publishes the same
/// id.
pub const ADAPTER_DELTAS_COMPONENT_ID: &str = "iris_3b.adapters.materialized_deltas";

/// Device bytes of `spec`'s adapter stack beside the generation backbone, priced from each
/// adapter file's safetensors **header** ([`crate::adapters::adapter_footprint`]) — no adapter
/// tensor is read. Every refusal the loader can make from a header is made here, so an adapter is
/// never admitted as free.
pub fn adapter_overlay(
    spec: &LoadSpec,
    backbone_dir: &Path,
) -> gen_core::Result<crate::adapters::AdapterFootprint> {
    if spec.adapters.is_empty() {
        return Ok(crate::adapters::AdapterFootprint::default());
    }
    let compute = compute_dtype(spec);
    Ok(crate::adapters::adapter_footprint(
        &crate::nn::safetensors_shapes(&backbone_dir.join(BACKBONE_WEIGHTS_FILE))?,
        &spec.adapters,
        IrisTask::Generation,
        MODEL_ID,
        compute.size_in_bytes() as u64,
        production_kernel_dtype(compute).size_in_bytes() as u64,
    )?)
}

/// Architecture axes for one resolved load, read from the task's own `config.yaml`.
///
/// Gated on a materialized root: a weights-free contract surface names a sentinel path that is not
/// on disk, so nothing is resolved there and every axis stays `None`. `IrisConfig` takes each key
/// the file omits from the release preset, and the loader builds exactly that parsed struct, so the
/// axes are read back off it; a directory whose `config.yaml` does not parse has no loadable
/// geometry and declares nothing.
///
/// `transformer_blocks` is the trunk depth (the dual- and single-stream blocks the joint sequence
/// runs through), `attention_heads` / `head_dim` its attention geometry and `patch_size` the pixel
/// patch. There is no latent and no autoencoder on any Iris task, so the three VAE axes are
/// structurally absent.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    let Some(root) = af::snapshot_root(spec) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let Ok(config) = IrisConfig::from_dir(root) else {
        return gen_core::MemoryArchitectureFacts::default();
    };
    let model = &config.model;
    let heads = af::declared(model.num_heads);
    gen_core::MemoryArchitectureFacts {
        attention_heads: heads,
        head_dim: af::head_dim(af::declared(model.hidden_size), heads),
        transformer_blocks: af::declared(model.depth),
        patch_size: af::declared(model.patch_size),
        latent_channels: None,
        vae_spatial_scale: None,
        vae_temporal_scale: None,
        activation_dtype_width: u32::try_from(compute_dtype(spec).size_in_bytes()).ok(),
    }
}

/// The numeric tier a load of `spec` runs: the precision and the tier it names (only generation
/// accepts one, and it packs the text tower and the backbone at that same tier — sc-26266).
/// Nothing is promoted above the selected precision.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

// ================================================================================================
// Contracts.
// ================================================================================================

/// Refuse a quantized tier the generation loader cannot build (sc-26266): it packs the q8 / q4
/// tiers only, and only under the bf16 compute policy (upstream's FP32 path is the dense
/// reference, not a tier) — the MLX twin's rule. The loader and the contract factories share this
/// one refusal.
pub(crate) fn refuse_unloadable_quant(model_id: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    match spec.quantize {
        None | Some(gen_core::Quant::Q4 | gen_core::Quant::Q8) => {}
        Some(other) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{model_id}: LoadSpec::quantize {other:?} is not a tier of the Candle Iris-3B \
                 route; it packs the q8 and q4 tiers"
            )))
        }
    }
    if spec.quantize.is_some() && spec.precision != gen_core::Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: a quantized tier runs under the bf16 compute policy; LoadSpec::precision \
             {:?} is the dense reference path only",
            spec.precision
        )));
    }
    Ok(())
}

/// No Iris loader has a deferred (block-streaming) materialization shape, so a spec asking for one
/// has no contract — and, because every loader prices its contract first, no load either.
fn refuse_deferred_materialization(model_id: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    if spec.load_shape == LoadShape::DeferredMaterialization {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: LoadSpec::load_shape = DeferredMaterialization is not supported (the Iris \
             backbone has no block-streaming loader; every block is materialized at load)"
        )));
    }
    Ok(())
}

fn base_contract(model_id: &str, task: IrisTask, spec: &LoadSpec) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        model_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            // Rung 4 is not implemented, so nothing materializes a window; the field has no default
            // and must still state what one would do. The loader is an mmap read of dense tensors.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(spec);
    let fingerprint = calibration_fingerprint_for(task, spec.precision, spec.quantize);
    contract.calibration = Some(MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    contract
}

fn not_applicable(reason: &str) -> MemoryStrategySupport {
    MemoryStrategySupport::StructurallyNotApplicable {
        reason: reason.to_owned(),
    }
}

/// The generation contract with no asset facts — the registry conformance surface.
pub fn weights_free_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    crate::model::refuse_unsupported_spec(spec)?;
    refuse_deferred_materialization(MODEL_ID, spec)?;
    let mut contract = base_contract(MODEL_ID, IrisTask::Generation, spec);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: true,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            // No conditioning-token variable: the text window is the checkpoint's fixed
            // `text_len`, whatever the prompt.
            MemoryFormulaVariable::AttentionChunkSize,
        ],
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode => not_applicable(
                "iris_3b denoises in pixel space: there is no latent, no autoencoder and so no \
                 decoder scratch to tile (the decode phase is the [-1, 1] -> RGB8 conversion)",
            ),
            // Applicable — the trunk is a block stack — but no loader re-materializes it.
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    Ok(contract)
}

/// The executable generation contract for a real load — [`weights_free_memory_strategy_contract`]
/// plus the on-disk inventory: the backbone at the compute dtype, the language tower at the dtype
/// the production device runs it in, and the adapter stack as a typed auxiliary component.
pub fn generation_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = weights_free_memory_strategy_contract(spec)?;
    let resources = GenerationResources::from_spec(spec, MODEL_ID)?;
    let config = IrisConfig::from_dir(&resources.backbone_dir)?;
    let compute_width = compute_dtype(spec).size_in_bytes() as u64;
    let quant = BackboneQuant::for_load(spec.quantize)?;
    let transformer = backbone_bytes(
        &resources.backbone_dir.join(BACKBONE_WEIGHTS_FILE),
        compute_width,
        quant,
    )?;
    let conditioning = text_encoder_bytes(
        &resources.text_encoder_dir,
        production_tower_dtype(&config.text_encoder)?.size_in_bytes() as u64,
        spec.quantize,
    )?;
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: conditioning.saturating_add(transformer),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: 0,
        overlay_bytes: 0,
    };
    price_adapter_stack(
        &mut contract,
        adapter_overlay(spec, &resources.backbone_dir)?,
    );
    Ok(contract)
}

/// Declare a non-zero adapter stack as typed auxiliary components beside the base components
/// (never inside `base_bytes`), with `OverlayBytes` in the formula. Iris adapters are forward-time
/// residuals — never folded into the base weight — so both classes are resident for the render. A
/// class the stack does not contain declares nothing (a zero-byte component is not a component).
fn price_adapter_stack(
    contract: &mut MemoryProviderContract,
    overlay: crate::adapters::AdapterFootprint,
) {
    if overlay.total() == 0 {
        return;
    }
    contract.asset_facts.overlay_bytes = overlay.total();
    let MemoryFormulaKind::PhaseEnvelope { phases, variables } = &contract.formula else {
        return;
    };
    let mut variables = variables.clone();
    variables.push(MemoryFormulaVariable::OverlayBytes);
    let resident_components = [
        (
            ADAPTER_RESIDUALS_COMPONENT_ID,
            overlay.forward_residual_bytes,
        ),
        (
            ADAPTER_DELTAS_COMPONENT_ID,
            overlay.materialized_delta_bytes,
        ),
    ]
    .into_iter()
    .filter(|(_, bytes)| *bytes > 0)
    .map(|(id, resident_bytes)| gen_core::MemoryResidentComponent {
        id: id.to_owned(),
        kind: gen_core::MemoryComponentKind::AdapterStack,
        resident_bytes,
        bounded_by: None,
        residency: gen_core::MemoryComponentResidency::WholeRender,
    })
    .collect();
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: phases.clone(),
        variables,
        resident_components,
    };
}

/// The shape both downstream tasks share: one backbone forward on shipped conditioning, resident
/// only.
fn weights_free_downstream_contract(
    model_id: &str,
    task: IrisTask,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    refuse_deferred_materialization(model_id, spec)?;
    let mut contract = base_contract(model_id, task, spec);
    // No lifecycle phases and no hooks: one forward, nothing to stage, tile or chunk on request.
    contract.lifecycle = MemoryLifecycleCapabilities::default();
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
        ],
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => not_applicable(&format!(
                "{model_id} loads no text encoder (it runs on the export's shipped empty-prompt \
                 states), so the backbone is the only component and there is nothing to shed"
            )),
            MemoryStrategy::BoundedDecode => not_applicable(&format!(
                "{model_id} predicts in pixel space: there is no latent, no autoencoder and so no \
                 decoder scratch to tile"
            )),
            // Applicable — the forward is the same backbone attention and block stack as
            // generation — but the task's request type carries no `GenerationMemory`, so nothing
            // can select a budget or a window.
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::Missing
            }
        };
    }
    Ok(contract)
}

fn price_downstream_export(
    contract: &mut MemoryProviderContract,
    export: &TaskExport,
    spec: &LoadSpec,
) -> gen_core::Result<()> {
    let transformer = backbone_bytes(
        &export.weights_path(),
        compute_dtype(spec).size_in_bytes() as u64,
        None,
    )?;
    let conditioning = empty_prompt_bytes(&export.dir.join(EMPTY_PROMPT_FILE))?;
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: conditioning.saturating_add(transformer),
        // The task's whole conditioning residency: the shipped empty-prompt states on the device.
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: 0,
        overlay_bytes: 0,
    };
    Ok(())
}

/// The depth contract with no asset facts — the registry conformance surface.
pub fn weights_free_depth_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    crate::depth::refuse_unsupported_spec(spec)?;
    weights_free_downstream_contract(DEPTH_MODEL_ID, IrisTask::Depth, spec)
}

/// The executable depth contract for a real load: the widened backbone and its 1×1 reducer at the
/// compute dtype, plus the empty-prompt states.
pub fn depth_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = weights_free_depth_contract(spec)?;
    let export = TaskExport::from_spec(spec, IrisTask::Depth, DEPTH_MODEL_ID)?;
    price_downstream_export(&mut contract, &export, spec)?;
    Ok(contract)
}

/// The restoration contract with no asset facts — the registry conformance surface.
pub fn weights_free_restoration_contract(
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    crate::restoration::refuse_unsupported_spec(spec)?;
    weights_free_downstream_contract(RESTORATION_MODEL_ID, IrisTask::Restoration, spec)
}

/// The executable restoration contract for a real load: the fine-tuned backbone at the compute
/// dtype, plus the empty-prompt states.
pub fn restoration_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = weights_free_restoration_contract(spec)?;
    let resources = RestorationResources::from_spec(spec, RESTORATION_MODEL_ID)?;
    price_downstream_export(&mut contract, &resources.export, spec)?;
    Ok(contract)
}

// ================================================================================================
// Admission geometry of the two downstream routes.
// ================================================================================================

/// The geometry a depth request is admitted at: the grid the model actually runs on (the source
/// resized to the request's resolution cap and snapped to the patch), one image, no reference —
/// the source is the transform's subject, not a conditioning reference.
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
/// side), which is what the host canvases scale with. One image, no reference.
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
/// the device activation, whatever the output size.
pub fn restoration_forward_pixels(plan: &RestorationPlan) -> u64 {
    if plan.single_pass() {
        u64::from(plan.padded.width) * u64::from(plan.padded.height)
    } else {
        u64::from(plan.tile) * u64::from(plan.tile)
    }
}

// ================================================================================================
// Safety checks and request scopes.
// ================================================================================================

fn unsupported(message: String) -> gen_core::Error {
    gen_core::Error::Unsupported(message)
}

/// The generation safety check: the shared handshake, then this route's own gate.
pub fn generation_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if !matches!(context.mode, MemoryMode::TextToImage) {
            return Err(unsupported(format!(
                "{MODEL_ID}: the generation route is text-to-image; got {:?}",
                context.mode
            )));
        }
        if context.geometry.reference_count != 0 {
            return Err(unsupported(format!(
                "{MODEL_ID}: the generation route takes no reference images"
            )));
        }
        if context.geometry.frames > 1 {
            return Err(unsupported(format!(
                "{MODEL_ID}: a still-image route cannot admit {} frames",
                context.geometry.frames
            )));
        }
        if context.use_pid {
            return Err(unsupported(format!(
                "{MODEL_ID}: there is no PiD decoder on this route"
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

fn downstream_safety_check(
    model_id: &str,
    mode: &str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.mode.as_key() != mode {
            return Err(unsupported(format!(
                "{model_id}: this route's memory mode is `{mode}`; got `{}`",
                context.mode.as_key()
            )));
        }
        if context.geometry.reference_count != 0 || context.geometry.batch != 1 {
            return Err(unsupported(format!(
                "{model_id}: one source image per request and no reference images"
            )));
        }
        if context.geometry.frames > 1 {
            return Err(unsupported(format!(
                "{model_id}: a still-image route cannot admit {} frames",
                context.geometry.frames
            )));
        }
        if context.use_pid {
            return Err(unsupported(format!(
                "{model_id}: there is no PiD decoder on this route"
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

/// The depth safety check ([`DEPTH_MEMORY_MODE`], resident only).
pub fn depth_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    downstream_safety_check(DEPTH_MODEL_ID, DEPTH_MEMORY_MODE, spec, contract, context)
}

/// The restoration safety check ([`RESTORE_MEMORY_MODE`], resident only).
pub fn restoration_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    downstream_safety_check(
        RESTORATION_MODEL_ID,
        RESTORE_MEMORY_MODE,
        spec,
        contract,
        context,
    )
}

/// The provider-owned valid fixture of each optimized rung generation implements.
pub(crate) fn generation_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    Ok(vec![MemoryBehaviorFixture::new(
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            loaded_tier(spec),
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                // The loaded adapter stack is part of the route identity, so a fixture for an
                // adapted load is keyed apart from a plain one.
                overlay: (!spec.adapters.is_empty()).then(|| "lora".to_owned()),
            },
        )?,
    )])
}

/// Images a request renders: one per prompt of a prompt batch, `count` times.
pub fn request_images(request: &GenerationRequest) -> u64 {
    request.prompt_batch.len().max(1) as u64 * u64::from(request.count)
}

/// The generation request scope: the shared Candle lifecycle, plus the one axis the shared
/// geometry cannot see. A prompt batch denoises its rows **together** (one batched solve per
/// `count` repetition), so the admitted `MemoryGeometry::batch` bounds the request's total images
/// ([`request_images`]), not `count` alone — an admitted single-image request cannot be widened
/// into an N-row batch at N times the activation.
struct GenerationRequestScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    admitted_images: u32,
}

impl MemoryRequestScope for GenerationRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        let images = request_images(request);
        if images > u64::from(self.admitted_images) {
            return Err(unsupported(format!(
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

    candle_gen::gen_core::delegate_memory_request_scope!(core);

    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.core.finish(outcome)
    }
}

/// Open the generation request scope for an admitted selection. The scope writes the canonical
/// `GenerationMemory` onto the request, which `Iris3b::generate` honours.
pub(crate) fn generation_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } =
        generation_safety_check(spec, contract, context)
    {
        return Err(unsupported(reason));
    }
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        MODEL_ID,
        candle_gen::default_device()?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        // Rung 4 is not selectable, so no window is ever checked against this count; a resolved
        // load states its trunk depth, a weights-free surface has none to state.
        contract.architecture_facts.transformer_blocks.unwrap_or(0) as usize,
        |_use_pid, _edge, _overlap| {
            Err(unsupported(format!(
                "{MODEL_ID}: there is no decoder on this route, so no decode geometry is admitted"
            )))
        },
    )?;
    if contract.engages_selection(&context.selection, MemoryStrategy::BoundedAttention) {
        config.attention_chunk_size = context.selection.parameters.attention_chunk_size;
    }
    Ok(Some(Box::new(GenerationRequestScope {
        core: candle_gen::request_scope::CandleRequestScopeCore::new(config),
        admitted_images: context.geometry.batch,
    })))
}

/// The downstream tasks implement no optimized rung, so they own no optimized fixture.
fn downstream_valid_fixtures(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    Ok(Vec::new())
}

/// Admission for a downstream request. A resident selection that passes the route's safety check
/// needs no scope — neither `DepthRequest` nor `TransformRequest` carries a memory block to
/// configure — and anything else is refused.
fn downstream_begin_request(
    decision: MemorySafetyDecision,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    match decision {
        MemorySafetyDecision::Accept => Ok(None),
        MemorySafetyDecision::Reject { reason } => Err(unsupported(reason)),
    }
}

// ================================================================================================
// Registrations.
// ================================================================================================

/// The registry-load witnesses the two encoder-free routes publish: the dense bf16 tier (neither
/// loader quantizes) under both offload policies, on the eager materialization shape only (no
/// route has a deferred loader, so that shape has no contract). The downstream loaders have one
/// component and treat the policy as advisory, so their two witnesses are the same contract.
pub fn memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.selector.tier == MemoryContractSurfaceTier::Bf16
                && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

/// The registry-load witnesses of generation (sc-26266): the dense tier and the q8 / q4 tiers the
/// loader packs, eagerly materialized (there is no deferred loader), under both offload policies
/// (`Sequential` is the staged load).
pub fn generation_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| surface.selector.load_shape == LoadShape::EagerMaterialization)
        .collect()
}

/// The generation route's shared-ladder registration.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: generation_contract,
    safety_check: generation_safety_check,
};

/// The generation route's weights-free contract fixture.
pub const MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_memory_strategy_contract,
        surface_specs: generation_memory_contract_surface_specs,
    };

/// The generation route's behavior registration: provider-owned valid fixtures and the executable
/// admission seam.
pub const MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: generation_valid_fixtures,
        begin_request: generation_begin_request,
    };

/// The depth route's registration (a composed route: depth is not a gen-core `Generator`).
pub const DEPTH_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: DEPTH_MODEL_ID,
    contract: depth_contract,
    safety_check: depth_safety_check,
};

/// The depth route's weights-free contract fixture.
pub const DEPTH_MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: DEPTH_MODEL_ID,
        contract: weights_free_depth_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The depth route's behavior registration: no optimized fixture, and admission through
/// [`depth_safety_check`].
pub const DEPTH_MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: DEPTH_MODEL_ID,
        valid_fixtures: downstream_valid_fixtures,
        begin_request: |spec, contract, context| {
            downstream_begin_request(depth_safety_check(spec, contract, context))
        },
    };

/// The restoration route's registration (a composed route: restoration is a `Transform`).
pub const RESTORE_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: RESTORATION_MODEL_ID,
        contract: restoration_contract,
        safety_check: restoration_safety_check,
    };

/// The restoration route's weights-free contract fixture.
pub const RESTORE_MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: RESTORATION_MODEL_ID,
        contract: weights_free_restoration_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The restoration route's behavior registration: no optimized fixture, and admission through
/// [`restoration_safety_check`].
pub const RESTORE_MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: RESTORATION_MODEL_ID,
        valid_fixtures: downstream_valid_fixtures,
        begin_request: |spec, contract, context| {
            downstream_begin_request(restoration_safety_check(spec, contract, context))
        },
    };

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
    use candle_gen::gen_core::{
        AdapterKind, AdapterSpec, OffloadPolicy, Precision, Quant, WeightsSource,
    };

    use crate::nn::{safetensors_shapes, Checkpoint};

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mlx-gen/mlx-gen-iris/tests/fixtures")
    }

    fn generation_spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(fixtures().join("tiny-snapshot/iris")));
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
        );
        spec
    }

    fn depth_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(fixtures().join("tiny-depth")))
    }

    fn restoration_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            fixtures().join("tiny-snapshot/upscaler"),
        ))
    }

    /// A spec naming nothing on disk — the registry's weights-free shape.
    fn unresolved() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ))
    }

    fn support(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> &MemoryStrategySupport {
        &contract.capability(strategy).unwrap().support
    }

    /// Each route declares exactly what its engine does, rung by rung.
    ///
    /// *Mutations that red this:* declaring `BoundedDecode` `Implemented` for any task; declaring
    /// `StagedResidency` anything but not-applicable for a downstream task; publishing an attention
    /// budget on a downstream task.
    #[test]
    fn each_route_declares_the_rungs_its_engine_runs() {
        let generation = weights_free_memory_strategy_contract(&unresolved()).unwrap();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                support(&generation, strategy),
                &MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        assert!(matches!(
            support(&generation, MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert_eq!(
            support(&generation, MemoryStrategy::BoundedTransformerResidency),
            &MemoryStrategySupport::Missing
        );
        assert_eq!(
            generation
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            [ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(
            u64::from(ATTENTION_CHUNK_SIZE),
            candle_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET
        );
        assert!(generation.lifecycle.attention_chunking && !generation.lifecycle.decode_tiling);
        // The text window is a fixed `text_len`, so the formula has no conditioning-token input.
        for variable in [
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::AttentionChunkSize,
        ] {
            assert!(generation.formula.uses(variable), "{variable:?}");
        }
        assert!(!generation
            .formula
            .uses(MemoryFormulaVariable::ConditioningTokenCount));
        gen_core_testkit::memory_strategy_conformance(&generation);

        for (contract, id) in [
            (
                weights_free_depth_contract(&unresolved()).unwrap(),
                DEPTH_MODEL_ID,
            ),
            (
                weights_free_restoration_contract(&unresolved()).unwrap(),
                RESTORATION_MODEL_ID,
            ),
        ] {
            assert_eq!(contract.provider_id, id);
            assert_eq!(
                support(&contract, MemoryStrategy::Resident),
                &MemoryStrategySupport::Implemented
            );
            for strategy in [
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ] {
                assert!(
                    matches!(
                        support(&contract, strategy),
                        MemoryStrategySupport::StructurallyNotApplicable { .. }
                    ),
                    "{id}: {strategy:?}"
                );
            }
            for strategy in [
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    support(&contract, strategy),
                    &MemoryStrategySupport::Missing,
                    "{id}: {strategy:?}"
                );
            }
            assert!(contract
                .strategies
                .iter()
                .all(|capability| capability.parameters == MemoryParameterRanges::default()));
            assert_eq!(contract.lifecycle, MemoryLifecycleCapabilities::default());
            assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
            gen_core_testkit::memory_strategy_conformance(&contract);
        }
    }

    /// The three routes never share calibration evidence, and each identity is a valid fingerprint.
    #[test]
    fn each_route_publishes_its_own_calibration_fingerprint() {
        let fingerprints = [
            MEMORY_CALIBRATION_FINGERPRINT,
            DEPTH_MEMORY_CALIBRATION_FINGERPRINT,
            RESTORE_MEMORY_CALIBRATION_FINGERPRINT,
        ];
        for fingerprint in fingerprints {
            gen_core::validate_calibration_fingerprint(fingerprint).unwrap();
        }
        assert_eq!(
            fingerprints.iter().collect::<BTreeSet<_>>().len(),
            3,
            "one identity per route"
        );
        for (contract, fingerprint) in [
            (
                weights_free_memory_strategy_contract(&unresolved()).unwrap(),
                MEMORY_CALIBRATION_FINGERPRINT,
            ),
            (
                weights_free_depth_contract(&unresolved()).unwrap(),
                DEPTH_MEMORY_CALIBRATION_FINGERPRINT,
            ),
            (
                weights_free_restoration_contract(&unresolved()).unwrap(),
                RESTORE_MEMORY_CALIBRATION_FINGERPRINT,
            ),
        ] {
            assert_eq!(contract.calibration.unwrap().fingerprint, fingerprint);
        }
        assert_eq!(DEPTH_MEMORY_MODE, IrisTask::Depth.name());
        assert_eq!(RESTORE_MEMORY_MODE, IrisTask::Restoration.name());
    }

    /// The calibration identity is keyed on (task, tier), mirroring the MLX twin: dense bf16 keeps
    /// its retained string, and the fp32 dense reference and the q8 / q4 tiers each get their own.
    /// A record whose evidence identity is one quantized tier's never passes the safety check of
    /// the other tier's load (q8 never satisfies q4 and vice versa), while its own tier passes.
    ///
    /// *Mutation that reds this:* returning one shared string for q8 and q4 from
    /// `calibration_fingerprint_for`.
    #[test]
    fn a_record_of_one_tier_never_satisfies_another_tiers_request() {
        let expected = [
            (
                IrisTask::Generation,
                [
                    "iris-3b-candle-derived-2026-10-12-v1",
                    "iris-3b-fp32-candle-derived-2026-10-12-v1",
                    "iris-3b-q8-candle-derived-2026-10-12-v1",
                    "iris-3b-q4-candle-derived-2026-10-12-v1",
                ],
            ),
            (
                IrisTask::Depth,
                [
                    "iris-3b-depth-candle-derived-2026-10-12-v1",
                    "iris-3b-depth-fp32-candle-derived-2026-10-12-v1",
                    "iris-3b-depth-q8-candle-derived-2026-10-12-v1",
                    "iris-3b-depth-q4-candle-derived-2026-10-12-v1",
                ],
            ),
            (
                IrisTask::Restoration,
                [
                    "iris-3b-restore-candle-derived-2026-10-12-v1",
                    "iris-3b-restore-fp32-candle-derived-2026-10-12-v1",
                    "iris-3b-restore-q8-candle-derived-2026-10-12-v1",
                    "iris-3b-restore-q4-candle-derived-2026-10-12-v1",
                ],
            ),
        ];
        let tiers = [
            (Precision::Bf16, None),
            (Precision::Fp32, None),
            (Precision::Bf16, Some(Quant::Q8)),
            (Precision::Bf16, Some(Quant::Q4)),
        ];
        let mut seen = BTreeSet::new();
        for (task, strings) in expected {
            assert_eq!(
                calibration_fingerprint_for(task, Precision::Bf16, None),
                calibration_fingerprint(task),
                "{task:?}: the dense bf16 cell keeps its retained string"
            );
            for ((precision, quant), want) in tiers.into_iter().zip(strings) {
                let fingerprint = calibration_fingerprint_for(task, precision, quant);
                assert_eq!(fingerprint, want, "{task:?} {precision:?} {quant:?}");
                gen_core::validate_calibration_fingerprint(&fingerprint).unwrap();
                assert!(
                    seen.insert(fingerprint),
                    "{task:?}: tiers share an identity"
                );
            }
        }

        // Every published contract carries its own tier's identity (the dense tiers on all three
        // routes; q8 / q4 on generation, the only route whose loader packs them).
        let published: [(IrisTask, ContractFactory); 3] = [
            (IrisTask::Generation, weights_free_memory_strategy_contract),
            (IrisTask::Depth, weights_free_depth_contract),
            (IrisTask::Restoration, weights_free_restoration_contract),
        ];
        for (task, factory) in published {
            for precision in [Precision::Bf16, Precision::Fp32] {
                let mut spec = unresolved();
                spec.precision = precision;
                assert_eq!(
                    factory(&spec).unwrap().calibration.unwrap().fingerprint,
                    calibration_fingerprint_for(task, precision, None)
                );
            }
        }
        let spec = |quant: Quant| unresolved().with_quant(quant);
        let contract = |quant: Quant| weights_free_memory_strategy_contract(&spec(quant)).unwrap();
        for quant in [Quant::Q8, Quant::Q4] {
            assert_eq!(
                contract(quant).calibration.unwrap().fingerprint,
                calibration_fingerprint_for(IrisTask::Generation, Precision::Bf16, Some(quant))
            );
        }

        // A record carrying the request's tier but another tier's evidence identity is refused by
        // the registered safety check; the request's own record passes.
        for (record_tier, request_tier) in [(Quant::Q8, Quant::Q4), (Quant::Q4, Quant::Q8)] {
            let request = contract(request_tier);
            let own = fixture_context(
                &spec(request_tier),
                &request,
                MemoryStrategy::StagedResidency,
            );
            assert_eq!(
                generation_safety_check(&spec(request_tier), &request, &own),
                MemorySafetyDecision::Accept,
                "{request_tier:?}"
            );
            let mut foreign = own.clone();
            foreign.calibration_fingerprint =
                contract(record_tier).calibration.unwrap().fingerprint;
            assert!(
                matches!(
                    generation_safety_check(&spec(request_tier), &request, &foreign),
                    MemorySafetyDecision::Reject { .. }
                ),
                "a {record_tier:?} record satisfied a {request_tier:?} request"
            );
        }
    }

    /// The axes come from each task's own `config.yaml`, and a weights-free surface publishes none.
    ///
    /// *Mutation that reds this:* returning `MemoryArchitectureFacts::default()` from
    /// `architecture_facts` unconditionally.
    #[test]
    fn architecture_facts_are_read_from_the_config_and_absent_without_one() {
        for (contract, id) in [
            (generation_contract(&generation_spec()).unwrap(), MODEL_ID),
            (depth_contract(&depth_spec()).unwrap(), DEPTH_MODEL_ID),
            (
                restoration_contract(&restoration_spec()).unwrap(),
                RESTORATION_MODEL_ID,
            ),
        ] {
            assert_eq!(
                contract.architecture_facts,
                gen_core::MemoryArchitectureFacts {
                    // The MINIATURE geometry (the release is 20 heads x 128, 24 blocks, patch 16)
                    // — proof the axes are read, not restated from the preset.
                    attention_heads: Some(4),
                    head_dim: Some(16),
                    transformer_blocks: Some(3),
                    patch_size: Some(4),
                    latent_channels: None,
                    vae_spatial_scale: None,
                    vae_temporal_scale: None,
                    // The default bf16 compute.
                    activation_dtype_width: Some(2),
                },
                "{id}"
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
            gen_core_testkit::memory_strategy_conformance(&contract);
        }
        let mut fp32 = generation_spec();
        fp32.precision = Precision::Fp32;
        assert_eq!(
            generation_contract(&fp32)
                .unwrap()
                .architecture_facts
                .activation_dtype_width,
            Some(4)
        );
        for contract in [
            weights_free_memory_strategy_contract(&unresolved()).unwrap(),
            weights_free_depth_contract(&unresolved()).unwrap(),
            weights_free_restoration_contract(&unresolved()).unwrap(),
        ] {
            assert!(contract.architecture_facts.is_empty());
        }
    }

    fn width(dtype: DType) -> u64 {
        dtype.size_in_bytes() as u64
    }

    /// Bytes the model built from `checkpoint` holds: every consumed tensor at the dtype the
    /// loader recorded for it.
    fn held(
        checkpoint: &Checkpoint,
        shapes: &std::collections::BTreeMap<String, Vec<usize>>,
        prefix: &str,
    ) -> u64 {
        let stored = checkpoint.stored_dtypes();
        assert!(!stored.is_empty());
        stored
            .iter()
            .map(|(key, dtype)| {
                let elements: usize = shapes[&format!("{prefix}{key}")].iter().product();
                elements as u64 * width(*dtype)
            })
            .sum()
    }

    /// The pricing rule against the loader itself: build the real `IrisDiT` from each committed
    /// checkpoint and sum every tensor at the dtype the loader recorded holding it in.
    ///
    /// *Mutations that red this:* pricing every tensor at the compute width (the f32 norm gains,
    /// modulation tables and text position table stop matching under bf16), pricing a projection
    /// bias at f32, or pricing a rank-1 `.weight` as a projection.
    #[test]
    fn the_backbone_is_priced_at_the_widths_the_loader_holds_each_tensor_in() {
        let device = Device::Cpu;
        for compute in [DType::BF16, DType::F32] {
            for dir in ["tiny-snapshot/iris", "tiny-snapshot/upscaler"] {
                let dir = fixtures().join(dir);
                let file = dir.join(BACKBONE_WEIGHTS_FILE);
                let config = IrisConfig::from_dir(&dir).unwrap();
                let checkpoint = Checkpoint::open(&file, &device).unwrap();
                crate::IrisDiT::from_checkpoint(&checkpoint, &config.model, compute, &device)
                    .unwrap();
                let shapes = safetensors_shapes(&file).unwrap();
                assert_eq!(
                    checkpoint.stored_dtypes().len(),
                    shapes.len(),
                    "every tensor of the checkpoint is held by the model"
                );
                assert_eq!(
                    backbone_bytes(&file, width(compute), None).unwrap(),
                    held(&checkpoint, &shapes, ""),
                    "{} at {compute:?}",
                    dir.display()
                );
            }

            // The depth export: the widened backbone under `pixel.` plus the 1x1 reducer, which the
            // loader casts to the compute dtype outside the `Loader`.
            let dir = fixtures().join("tiny-depth");
            let file = dir.join(BACKBONE_WEIGHTS_FILE);
            let config = IrisConfig::from_dir(&dir).unwrap();
            let checkpoint = Checkpoint::open_prefixed(&file, &device, "pixel.").unwrap();
            crate::IrisDiT::from_checkpoint_widened(
                &checkpoint,
                &config.model,
                compute,
                &device,
                config.model.in_channels + 1,
            )
            .unwrap();
            let shapes = safetensors_shapes(&file).unwrap();
            let reducer: usize = ["depth_reducer.weight", "depth_reducer.bias"]
                .iter()
                .map(|key| shapes[*key].iter().product::<usize>())
                .sum();
            assert_eq!(
                checkpoint.stored_dtypes().len() + 2,
                shapes.len(),
                "the backbone view plus the two reducer tensors is the whole export"
            );
            assert_eq!(
                backbone_bytes(&file, width(compute), None).unwrap(),
                held(&checkpoint, &shapes, "pixel.") + reducer as u64 * width(compute),
                "depth at {compute:?}"
            );
        }
        // Not vacuous: the two widths genuinely disagree on some tensors.
        let file = fixtures()
            .join("tiny-snapshot/iris")
            .join(BACKBONE_WEIGHTS_FILE);
        let (bf16, f32) = (
            backbone_bytes(&file, 2, None).unwrap(),
            backbone_bytes(&file, 4, None).unwrap(),
        );
        assert!(
            bf16 < f32 && bf16 * 2 > f32,
            "some tensors stay f32 under bf16"
        );
    }

    /// The tower is priced over the tensors it keeps: the language model, and the LM head only
    /// when it is untied. The vision tower never reaches the device.
    ///
    /// *Mutation that reds this:* pricing every tensor of the shard (`|_| true`).
    #[test]
    fn the_text_encoder_is_priced_over_the_resident_tensors_only() {
        assert!(text_encoder_keeps(
            "model.language_model.layers.0.mlp.up_proj.weight",
            true
        ));
        assert!(!text_encoder_keeps(
            "model.visual.blocks.0.attn.qkv.weight",
            false
        ));
        assert!(text_encoder_keeps("lm_head.weight", false));
        assert!(!text_encoder_keeps("lm_head.weight", true));

        // A synthetic snapshot carrying a vision tower and an untied head beside the tower.
        let tmp = tempfile::tempdir().unwrap();
        let source = fixtures().join("tiny-snapshot/text_encoder");
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::copy(source.join(name), tmp.path().join(name)).unwrap();
        }
        let zeros = |shape: (usize, usize)| {
            candle_gen::candle_core::Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
        };
        let tensors = std::collections::HashMap::from([
            (
                "model.language_model.layers.0.mlp.up_proj.weight".to_owned(),
                zeros((8, 4)),
            ),
            (
                "model.visual.blocks.0.attn.qkv.weight".to_owned(),
                zeros((64, 64)),
            ),
            ("lm_head.weight".to_owned(), zeros((16, 4))),
        ]);
        candle_gen::candle_core::safetensors::save(&tensors, tmp.path().join("model.safetensors"))
            .unwrap();
        let tied = candle_llm::ModelConfig::from_dir(tmp.path())
            .unwrap()
            .tie_word_embeddings;
        let head = if tied { 0 } else { 16 * 4 };
        assert_eq!(
            text_encoder_bytes(tmp.path(), 2, None).unwrap(),
            (8 * 4 + head) * 2
        );
        assert_eq!(
            text_encoder_bytes(tmp.path(), 4, None).unwrap(),
            (8 * 4 + head) * 4
        );

        // The committed miniature tower: priced over exactly the kept tensors.
        let mut kept = 0_u64;
        let tied = candle_llm::ModelConfig::from_dir(&source)
            .unwrap()
            .tie_word_embeddings;
        for file in crate::text_encoder::shard_files(&source).unwrap() {
            for header in safetensors_path_tensor_headers(&file).unwrap() {
                if text_encoder_keeps(&header.name, tied) {
                    kept += header.materialized_bytes(2).unwrap();
                }
            }
        }
        assert!(kept > 0);
        assert_eq!(text_encoder_bytes(&source, 2, None).unwrap(), kept);

        // The loaded tower's dtype is the priced width on the production device.
        let config = IrisConfig::from_dir(&fixtures().join("tiny-snapshot/iris")).unwrap();
        let device = candle_gen::default_device().unwrap();
        assert_eq!(
            production_tower_dtype(&config.text_encoder).unwrap(),
            crate::text_encoder::tower_dtype(&config.text_encoder, &device).unwrap()
        );
    }

    /// The loaded inventory is the sum of what each loader holds, decomposed honestly.
    #[test]
    fn each_resolved_contract_prices_its_own_components() {
        let spec = generation_spec();
        let contract = generation_contract(&spec).unwrap();
        let config = IrisConfig::from_dir(&fixtures().join("tiny-snapshot/iris")).unwrap();
        let tower = width(production_tower_dtype(&config.text_encoder).unwrap());
        let transformer = backbone_bytes(
            &fixtures()
                .join("tiny-snapshot/iris")
                .join(BACKBONE_WEIGHTS_FILE),
            2,
            None,
        )
        .unwrap();
        let conditioning =
            text_encoder_bytes(&fixtures().join("tiny-snapshot/text_encoder"), tower, None)
                .unwrap();
        assert_eq!(
            contract.asset_facts,
            MemoryAssetFacts {
                base_bytes: transformer + conditioning,
                conditioning_bytes: conditioning,
                transformer_bytes: transformer,
                decoder_bytes: 0,
                overlay_bytes: 0,
            }
        );
        assert!(transformer > 0 && conditioning > 0);

        // The FP32 path widens the compute-width tensors of the backbone and nothing else.
        let mut fp32 = spec.clone();
        fp32.precision = Precision::Fp32;
        let fp32 = generation_contract(&fp32).unwrap();
        assert_eq!(fp32.asset_facts.conditioning_bytes, conditioning);
        assert!(fp32.asset_facts.transformer_bytes > transformer);

        for (contract, dir) in [
            (depth_contract(&depth_spec()).unwrap(), "tiny-depth"),
            (
                restoration_contract(&restoration_spec()).unwrap(),
                "tiny-snapshot/upscaler",
            ),
        ] {
            let dir = fixtures().join(dir);
            let transformer = backbone_bytes(&dir.join(BACKBONE_WEIGHTS_FILE), 2, None).unwrap();
            let config = IrisConfig::from_dir(&dir).unwrap();
            let model = &config.model;
            // The shipped `[1, T, L, D]` empty-prompt states, f32 on the device.
            let conditioning =
                (model.text_len * model.text_lap_num_layers * model.text_dim * 4) as u64;
            assert_eq!(
                contract.asset_facts,
                MemoryAssetFacts {
                    base_bytes: transformer + conditioning,
                    conditioning_bytes: conditioning,
                    transformer_bytes: transformer,
                    decoder_bytes: 0,
                    overlay_bytes: 0,
                },
                "{}",
                dir.display()
            );
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// Bytes the loader's installed residuals hold, by class — counted from the residual tensors
    /// `merge_adapters` builds, with a structured LoKr asked for its own device bytes after one
    /// forward at the kernel dtype (which is when it caches its copies).
    fn installed(adapters: &[AdapterSpec], compute: DType) -> crate::adapters::AdapterFootprint {
        use crate::nn::Residual;
        use candle_gen::candle_core::Tensor;

        let kernel = production_kernel_dtype(compute);
        let device = candle_gen::default_device().unwrap();
        let shapes = safetensors_shapes(
            &fixtures()
                .join("tiny-snapshot/iris")
                .join(BACKBONE_WEIGHTS_FILE),
        )
        .unwrap();
        let merged =
            crate::adapters::merge_adapters(&shapes, adapters, IrisTask::Generation, MODEL_ID)
                .unwrap();
        let mut footprint = crate::adapters::AdapterFootprint::default();
        for residual in merged.residuals.values().flatten() {
            match residual {
                Residual::Lora { a, b } => {
                    assert_eq!((a.dtype(), b.dtype()), (DType::F32, DType::F32));
                    footprint.forward_residual_bytes +=
                        ((a.elem_count() + b.elem_count()) * 4) as u64;
                }
                Residual::Kron(factors) => {
                    let factors = factors.to_device(&device).unwrap();
                    let (_, inputs) = factors.shape();
                    factors
                        .residual(&Tensor::zeros((1, inputs), kernel, &device).unwrap())
                        .unwrap();
                    footprint.forward_residual_bytes += factors.device_bytes() as u64;
                }
                Residual::LowRank { a, b } => {
                    footprint.forward_residual_bytes +=
                        (a.elem_count() + b.elem_count()) as u64 * width(compute);
                }
                Residual::Delta(delta) => {
                    footprint.materialized_delta_bytes +=
                        delta.elem_count() as u64 * width(compute);
                }
            }
        }
        footprint
    }

    /// A stamped third-party LoHa over the miniature's `[64, 64]` `blocks.2.q_proj`, rank `rank`
    /// on both Hadamard pairs.
    fn loha_file(dir: &Path, rank: usize) -> AdapterSpec {
        use candle_gen::candle_core::Tensor;
        let zeros = |shape: (usize, usize)| Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
        let tensors = [
            ("blocks.2.q_proj.hada_w1_a", zeros((64, rank))),
            ("blocks.2.q_proj.hada_w1_b", zeros((rank, 64))),
            ("blocks.2.q_proj.hada_w2_a", zeros((64, rank))),
            ("blocks.2.q_proj.hada_w2_b", zeros((rank, 64))),
        ];
        let meta: std::collections::HashMap<String, String> = [
            ("family", "iris"),
            ("irisTask", "generation"),
            ("baseModel", MODEL_ID),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
        let path = dir.join(format!("loha-r{rank}.safetensors"));
        let list: Vec<(&str, &Tensor)> = tensors.iter().map(|(key, t)| (*key, t)).collect();
        safetensors::serialize_to_file(list, Some(meta), &path).unwrap();
        AdapterSpec::new(path, 1.0, AdapterKind::Lora)
    }

    fn fixture_adapter(file: &str, kind: AdapterKind) -> AdapterSpec {
        AdapterSpec::new(fixtures().join(file), 1.0, kind)
    }

    /// The header-only footprint is, byte for byte and class for class, what the loader installs:
    /// a LoRA's f32 factors, a structured LoKr's factors plus its cached copies, a LoHa's low-rank
    /// factorisation below the break-even rank and its dense delta at or above it.
    ///
    /// *Mutations that red this:* pricing a LoRA at the compute width; pricing every LoHa as a
    /// low-rank residual (drop the break-even filter); pricing a structured LoKr at its stored
    /// factor bytes.
    #[test]
    fn the_header_only_adapter_footprint_is_what_the_loader_installs() {
        let tmp = tempfile::tempdir().unwrap();
        let lora = fixture_adapter("iris_lora.safetensors", AdapterKind::Lora);
        let lokr = fixture_adapter("iris_lokr.safetensors", AdapterKind::Lokr);
        let low_rank = loha_file(tmp.path(), 2);
        let dense = loha_file(tmp.path(), 8);
        let backbone = fixtures().join("tiny-snapshot/iris");
        for precision in [Precision::Bf16, Precision::Fp32] {
            let mut plain = generation_spec();
            plain.precision = precision;
            let compute = compute_dtype(&plain);
            let mut classes = (0, 0);
            for stack in [
                vec![lora.clone()],
                vec![lokr.clone()],
                vec![low_rank.clone()],
                vec![dense.clone()],
                vec![lora.clone(), lokr.clone(), low_rank.clone(), dense.clone()],
            ] {
                let spec = plain.clone().with_adapters(stack.clone());
                let priced = adapter_overlay(&spec, &backbone).unwrap();
                assert_eq!(
                    priced,
                    installed(&stack, compute),
                    "{precision:?}: {stack:?}"
                );
                assert!(priced.total() > 0);
                classes = (
                    classes.0 + u64::from(priced.forward_residual_bytes > 0),
                    classes.1 + u64::from(priced.materialized_delta_bytes > 0),
                );
            }
            // Four stacks apply factors (LoRA, LoKr, low-rank LoHa, the mixed stack); two hold a
            // delta (the dense LoHa, the mixed stack).
            assert_eq!(classes, (4, 2), "{precision:?}");
        }
        // The break-even rank: 4·(64 + 64) < 64·64 factorises; 64·(64 + 64) does not.
        let spec = generation_spec().with_adapters(vec![low_rank]);
        assert_eq!(
            adapter_overlay(&spec, &backbone).unwrap(),
            crate::adapters::AdapterFootprint {
                forward_residual_bytes: 4 * (64 + 64) * 2,
                materialized_delta_bytes: 0,
            }
        );
        let spec = generation_spec().with_adapters(vec![dense]);
        assert_eq!(
            adapter_overlay(&spec, &backbone).unwrap(),
            crate::adapters::AdapterFootprint {
                forward_residual_bytes: 0,
                materialized_delta_bytes: 64 * 64 * 2,
            }
        );
    }

    /// Pricing reads the adapter's header and nothing else: a file cut off right after its header
    /// prices exactly like the whole file, though the loader cannot read it.
    ///
    /// *Mutation that reds this:* pricing through `merge_adapters` (it reads the tensors).
    #[test]
    fn adapter_pricing_reads_only_the_safetensors_header() {
        let tmp = tempfile::tempdir().unwrap();
        let whole = fixtures().join("iris_lokr.safetensors");
        let bytes = std::fs::read(&whole).unwrap();
        let header = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(header < bytes.len());
        let cut = tmp.path().join("header-only.safetensors");
        std::fs::write(&cut, &bytes[..header]).unwrap();

        let backbone = fixtures().join("tiny-snapshot/iris");
        let priced = |path: std::path::PathBuf| {
            let spec = generation_spec().with_adapters(vec![AdapterSpec::new(
                path,
                1.0,
                AdapterKind::Lokr,
            )]);
            adapter_overlay(&spec, &backbone).unwrap()
        };
        assert_eq!(priced(cut.clone()), priced(whole));
        let shapes = safetensors_shapes(&backbone.join(BACKBONE_WEIGHTS_FILE)).unwrap();
        assert!(crate::adapters::merge_adapters(
            &shapes,
            &[AdapterSpec::new(cut, 1.0, AdapterKind::Lokr)],
            IrisTask::Generation,
            MODEL_ID,
        )
        .is_err());
    }

    /// A non-empty adapter stack is declared as typed auxiliary components beside the base — one
    /// per class the stack actually contains — and an adapter the loader refuses from its header
    /// has no contract.
    ///
    /// *Mutations that red this:* dropping the `price_adapter_stack` call from
    /// `generation_contract`; declaring both classes under one component id.
    #[test]
    fn a_non_empty_adapter_stack_prices_typed_overlay_components() {
        let tmp = tempfile::tempdir().unwrap();
        let plain = generation_spec();
        let base = generation_contract(&plain).unwrap();
        assert!(!base.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert!(base.resident_components().is_empty());

        let lora = fixture_adapter("iris_lora.safetensors", AdapterKind::Lora);
        let dense = loha_file(tmp.path(), 8);
        let backbone = fixtures().join("tiny-snapshot/iris");
        let components = |stack: Vec<AdapterSpec>| {
            let spec = plain.clone().with_adapters(stack);
            let overlay = adapter_overlay(&spec, &backbone).unwrap();
            let contract = generation_contract(&spec).unwrap();
            assert_eq!(contract.asset_facts.overlay_bytes, overlay.total());
            assert_eq!(
                contract.asset_facts.base_bytes, base.asset_facts.base_bytes,
                "the overlay is declared beside the base, never inside it"
            );
            assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
            assert!(
                contract.conformance_errors().is_empty(),
                "{:?}",
                contract.conformance_errors()
            );
            assert!(contract.total_resident_bytes() > base.total_resident_bytes());
            let declared: Vec<(String, u64)> = contract
                .resident_components()
                .iter()
                .map(|component| (component.id.clone(), component.steady_state_bytes()))
                .collect();
            (overlay, declared)
        };

        // A LoRA is applied as factors only: one component, the residuals.
        let (overlay, declared) = components(vec![lora.clone()]);
        assert_eq!(
            declared,
            [(
                "iris_3b.adapters.forward_residuals".to_owned(),
                overlay.forward_residual_bytes
            )]
        );
        // A dense LoHa is held as a delta only: one component, the deltas.
        let (overlay, declared) = components(vec![dense.clone()]);
        assert_eq!(
            declared,
            [(
                "iris_3b.adapters.materialized_deltas".to_owned(),
                overlay.materialized_delta_bytes
            )]
        );
        // Both together: both components, each at its own bytes.
        let (overlay, declared) = components(vec![lora, dense]);
        assert_eq!(
            declared,
            [
                (
                    ADAPTER_RESIDUALS_COMPONENT_ID.to_owned(),
                    overlay.forward_residual_bytes
                ),
                (
                    ADAPTER_DELTAS_COMPONENT_ID.to_owned(),
                    overlay.materialized_delta_bytes
                ),
            ]
        );

        // Refusals the loader makes are made from the header: another task's adapter, a kind the
        // file contradicts, per-pass scales, and a file that is not on disk.
        let refused = |adapter: AdapterSpec| {
            generation_contract(&plain.clone().with_adapters(vec![adapter]))
                .expect_err("a refused adapter has no contract")
                .to_string()
        };
        let wrong_task = refused(fixture_adapter(
            "iris_lora_depth_task.safetensors",
            AdapterKind::Lora,
        ));
        assert!(wrong_task.contains("task"), "{wrong_task}");
        let mislabeled = refused(fixture_adapter("iris_lora.safetensors", AdapterKind::Lokr));
        assert!(mislabeled.contains("declared LoKr"), "{mislabeled}");
        let mislabeled = refused(fixture_adapter("iris_lokr.safetensors", AdapterKind::Lora));
        assert!(mislabeled.contains("networkType=lokr"), "{mislabeled}");
        let mut scaled = fixture_adapter("iris_lora.safetensors", AdapterKind::Lora);
        scaled.moe_expert = Some(gen_core::MoeExpert::High);
        assert!(refused(scaled).contains("MoE expert"));
        refused(fixture_adapter("absent.safetensors", AdapterKind::Lora));
    }

    fn fixture_context(
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryRunContext {
        generation_valid_fixtures(spec, contract, strategy)
            .unwrap()
            .pop()
            .expect("an implemented optimized rung owns a fixture")
            .context
    }

    /// The generation gate accepts its own route and refuses every crossing.
    ///
    /// *Mutation that reds this:* returning `Ok(())` from the route gate unconditionally.
    #[test]
    fn the_generation_gate_refuses_other_modes_references_pid_and_a_crossed_tier() {
        let spec = unresolved();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedAttention,
        ] {
            let context = fixture_context(&spec, &contract, strategy);
            assert_eq!(
                generation_safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Accept,
                "{strategy:?}"
            );

            let mut edit = context.clone();
            edit.mode = MemoryMode::Edit;
            let mut referenced = context.clone();
            referenced.geometry.reference_count = 1;
            referenced.has_reference = true;
            let mut pid = context.clone();
            pid.use_pid = true;
            let mut crossed = context.clone();
            crossed.selection.tier.quant = Some(Quant::Q4);
            let mut fp32 = context.clone();
            fp32.selection.tier.precision = Precision::Fp32;
            let mut clip = context.clone();
            clip.geometry.frames = 2;
            for (name, context) in [
                ("frames", clip),
                ("edit", edit),
                ("reference", referenced),
                ("pid", pid),
                ("q4", crossed),
                ("fp32", fp32),
            ] {
                assert!(
                    matches!(
                        generation_safety_check(&spec, &contract, &context),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{strategy:?}: {name}"
                );
            }
        }
        // Unpublished rungs and budgets are refused by the shared selection check.
        let mut decode = fixture_context(&spec, &contract, MemoryStrategy::StagedResidency);
        decode.selection.strategy = MemoryStrategy::BoundedDecode;
        let mut window = decode.clone();
        window.selection.strategy = MemoryStrategy::BoundedTransformerResidency;
        let mut budget = fixture_context(&spec, &contract, MemoryStrategy::BoundedAttention);
        budget.selection.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        for context in [decode, window, budget] {
            assert!(matches!(
                generation_safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        // A rung that is not implemented owns no fixture.
        assert!(
            generation_valid_fixtures(&spec, &contract, MemoryStrategy::BoundedDecode)
                .unwrap()
                .is_empty()
        );
    }

    /// The scope writes the canonical request memory for each rung, admits exactly the published
    /// attention budget it was opened with, and refuses a prompt batch wider than the admitted
    /// image count.
    ///
    /// *Mutation that reds this:* dropping the `request_images` check from
    /// `GenerationRequestScope::configure_request` (a 1-image admission then accepts a 3-row batch).
    #[test]
    fn the_generation_scope_configures_the_request_and_bounds_a_prompt_batch() {
        let spec = unresolved();
        let contract = weights_free_memory_strategy_contract(&spec).unwrap();

        let staged = fixture_context(&spec, &contract, MemoryStrategy::StagedResidency);
        let mut request = GenerationRequest {
            width: staged.geometry.width,
            height: staged.geometry.height,
            ..Default::default()
        };
        let mut scope = generation_begin_request(&spec, &contract, &staged)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.stage_residency && !memory.chunk_attention);
        assert!(crate::model::stage_residency_for(&spec, &request));
        assert_eq!(crate::model::attention_budget_for(&request), None);
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE).is_err());
        assert!(scope.configure_decode(512, 64, staged.geometry).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        let bounded = fixture_context(&spec, &contract, MemoryStrategy::BoundedAttention);
        let mut request = GenerationRequest {
            width: bounded.geometry.width,
            height: bounded.geometry.height,
            ..Default::default()
        };
        let mut scope = generation_begin_request(&spec, &contract, &bounded)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.unwrap();
        assert!(memory.chunk_attention && !memory.stage_residency);
        assert_eq!(
            crate::model::attention_budget_for(&request),
            Some(ATTENTION_CHUNK_SIZE as usize)
        );
        assert!(!crate::model::stage_residency_for(&spec, &request));
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE / 2).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        // Admitted for one image: a three-prompt batch is three rows of the same solve.
        let batch = |prompts: usize, count: u32| GenerationRequest {
            width: staged.geometry.width,
            height: staged.geometry.height,
            count,
            prompt_batch: (0..prompts).map(|i| format!("prompt {i}")).collect(),
            ..Default::default()
        };
        assert_eq!(request_images(&batch(3, 2)), 6);
        assert_eq!(request_images(&batch(0, 2)), 2);
        let mut scope = generation_begin_request(&spec, &contract, &staged)
            .unwrap()
            .unwrap();
        assert!(scope.configure_request(&mut batch(3, 1)).is_err());
        scope.configure_request(&mut batch(1, 1)).unwrap();
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        let mut wide = staged.clone();
        wide.geometry.batch = 6;
        let mut scope = generation_begin_request(&spec, &contract, &wide)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut batch(3, 2)).unwrap();
        assert!(scope.configure_request(&mut batch(4, 2)).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        // A `Sequential` load stages every request, whatever the request says.
        let sequential = unresolved().with_offload_policy(OffloadPolicy::Sequential);
        assert!(crate::model::stage_residency_for(
            &sequential,
            &GenerationRequest::default()
        ));
        assert!(!crate::model::stage_residency_for(
            &spec,
            &GenerationRequest::default()
        ));
    }

    fn downstream_context(contract: &MemoryProviderContract, mode: &str) -> MemoryRunContext {
        let calibration = contract.calibration.clone().unwrap();
        MemoryRunContext {
            reference_resources: None,
            selection: gen_core::MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: loaded_tier(&unresolved()),
            },
            optimization_authority: gen_core::MemoryOptimizationAuthority::Resident,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            load_shape: calibration.load_shape,
            mode: MemoryMode::Other(mode.to_owned()),
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: 8 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1024 * 1024 * 1024,
            cache_state: gen_core::MemoryCacheState::Cold,
            evidence_revision: "weights-free-provider-behavior".to_owned(),
        }
    }

    type SafetyCheck =
        fn(&LoadSpec, &MemoryProviderContract, &MemoryRunContext) -> MemorySafetyDecision;

    /// A downstream route admits its own resident request — with no scope, because its request
    /// type has no memory block to configure — and refuses every optimized selection, the other
    /// task's mode, and each handshake mutation.
    ///
    /// *Mutation that reds this:* dropping the mode comparison from `downstream_safety_check` (the
    /// depth route then admits a restoration context).
    #[test]
    fn a_downstream_route_admits_only_its_own_resident_request() {
        let spec = unresolved();
        let routes: [(
            MemoryProviderContract,
            &str,
            &str,
            SafetyCheck,
            gen_core::MemoryBehaviorRegistration,
        ); 2] = [
            (
                weights_free_depth_contract(&spec).unwrap(),
                DEPTH_MEMORY_MODE,
                RESTORE_MEMORY_MODE,
                depth_safety_check,
                DEPTH_MEMORY_BEHAVIOR_REGISTRATION,
            ),
            (
                weights_free_restoration_contract(&spec).unwrap(),
                RESTORE_MEMORY_MODE,
                DEPTH_MEMORY_MODE,
                restoration_safety_check,
                RESTORE_MEMORY_BEHAVIOR_REGISTRATION,
            ),
        ];
        for (contract, mode, other_mode, check, behavior) in routes {
            let context = downstream_context(&contract, mode);
            assert_eq!(
                check(&spec, &contract, &context),
                MemorySafetyDecision::Accept
            );
            assert!((behavior.begin_request)(&spec, &contract, &context)
                .unwrap()
                .is_none());

            let mut crossed_mode = context.clone();
            crossed_mode.mode = MemoryMode::Other(other_mode.to_owned());
            let mut text_to_image = context.clone();
            text_to_image.mode = MemoryMode::TextToImage;
            let mut batch = context.clone();
            batch.geometry.batch = 2;
            let mut referenced = context.clone();
            referenced.geometry.reference_count = 1;
            referenced.has_reference = true;
            let mut fingerprint = context.clone();
            fingerprint.calibration_fingerprint.push_str("-mutated");
            let mut tier = context.clone();
            tier.selection.tier.precision = Precision::Fp32;
            let mut over_budget = context.clone();
            over_budget.predicted_peak_bytes = u64::MAX;
            let mut clip = context.clone();
            clip.geometry.frames = 2;
            let mut refused = vec![
                clip,
                crossed_mode,
                text_to_image,
                batch,
                referenced,
                fingerprint,
                tier,
                over_budget,
            ];
            for strategy in MemoryStrategy::ALL {
                if strategy.is_optimized() {
                    let mut optimized = context.clone();
                    optimized.selection.strategy = strategy;
                    optimized.optimization_authority =
                        gen_core::MemoryOptimizationAuthority::Calibrated;
                    refused.push(optimized);
                    assert!((behavior.valid_fixtures)(&spec, &contract, strategy)
                        .unwrap()
                        .is_empty());
                }
            }
            for context in refused {
                assert!(
                    matches!(
                        check(&spec, &contract, &context),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{mode}: {context:?}"
                );
                assert!((behavior.begin_request)(&spec, &contract, &context).is_err());
            }
        }
    }

    type ContractFactory = fn(&LoadSpec) -> gen_core::Result<MemoryProviderContract>;

    /// The encoder-free routes witness the dense tier and generation the dense, q8 and q4 tiers
    /// (sc-26266), each under both offload policies on the eager shape; a spec no Iris loader can
    /// serve has no contract on either path — a deferred shape anywhere, a quantized tier on the
    /// encoder-free routes, and NVFP4 or a quantized tier under FP32 compute on generation.
    ///
    /// *Mutations that red this:* dropping `refuse_deferred_materialization` from a contract
    /// factory; publishing the shared twelve-cell Candle witness set unfiltered; filtering
    /// generation to bf16; dropping either arm of `refuse_unloadable_quant`.
    #[test]
    fn the_surface_witnesses_are_eager_and_unsupported_specs_have_no_contract() {
        let ids = |surfaces: &[MemoryContractSurfaceSpec]| {
            let mut ids: Vec<_> = surfaces
                .iter()
                .map(|surface| surface.selector.id())
                .collect();
            ids.sort();
            ids
        };
        let dense = memory_contract_surface_specs();
        assert_eq!(
            ids(&dense),
            ["bf16:resident:eager", "bf16:sequential:eager"]
        );
        let generation = generation_memory_contract_surface_specs();
        assert_eq!(
            ids(&generation),
            [
                "bf16:resident:eager",
                "bf16:sequential:eager",
                "q4:resident:eager",
                "q4:sequential:eager",
                "q8:resident:eager",
                "q8:sequential:eager",
            ]
        );
        let factories: [(ContractFactory, &[MemoryContractSurfaceSpec]); 3] = [
            (weights_free_memory_strategy_contract, &generation),
            (weights_free_depth_contract, &dense),
            (weights_free_restoration_contract, &dense),
        ];
        for (factory, surfaces) in factories {
            for surface in surfaces {
                let contract = factory(&surface.spec).unwrap();
                assert_eq!(contract.load_shape, LoadShape::EagerMaterialization);
                assert!(contract.conformance_errors().is_empty());
            }
            let deferred = unresolved().with_load_shape(LoadShape::DeferredMaterialization);
            assert!(factory(&deferred).is_err(), "no deferred loader");
            assert!(
                factory(&unresolved().with_quant(Quant::Nvfp4)).is_err(),
                "no NVFP4 tier"
            );
            let mut fp32 = unresolved().with_quant(Quant::Q8);
            fp32.precision = Precision::Fp32;
            assert!(factory(&fp32).is_err(), "no quantized FP32 tier");
        }
        for factory in [
            weights_free_depth_contract,
            weights_free_restoration_contract,
        ] {
            for quant in [Quant::Q4, Quant::Q8] {
                assert!(
                    factory(&unresolved().with_quant(quant)).is_err(),
                    "the encoder-free routes ship one dense tier"
                );
            }
        }
        for quant in [Quant::Q4, Quant::Q8] {
            let contract =
                weights_free_memory_strategy_contract(&unresolved().with_quant(quant)).unwrap();
            assert_eq!(
                loaded_tier(&unresolved().with_quant(quant)).quant,
                Some(quant)
            );
            assert!(contract.conformance_errors().is_empty());
        }
        let deferred = |spec: LoadSpec| spec.with_load_shape(LoadShape::DeferredMaterialization);
        assert!(generation_contract(&deferred(generation_spec())).is_err());
        assert!(depth_contract(&deferred(depth_spec())).is_err());
        assert!(restoration_contract(&deferred(restoration_spec())).is_err());
    }

    /// The restoration admission geometry is the size the model runs at, and the largest forward
    /// is one tile however large the output.
    ///
    /// *Mutation that reds this:* returning the padded area from `restoration_forward_pixels`
    /// unconditionally (a tiled plan's forward is then priced at the whole image).
    #[test]
    fn restoration_tiling_bounds_the_forward_at_one_tile() {
        use candle_gen::gen_core::iris::restoration::{
            plan, Dims, RestorationOptions, TileGeometry,
        };
        use candle_gen::gen_core::InputSizing;

        let geometry = TileGeometry::RELEASE;
        let options = RestorationOptions {
            scale: 4.0,
            input_sizing: InputSizing::Original,
            color_fix: true,
        };
        // 1024x768 at 4x -> 4096x3072: tiled.
        let tiled = plan(
            Dims::new(1024, 768),
            &options,
            geometry,
            RESTORATION_MODEL_ID,
        )
        .unwrap();
        assert!(!tiled.single_pass() && tiled.forward_count() > 1);
        assert_eq!(restoration_forward_pixels(&tiled), 1024 * 1024);
        assert_eq!(
            restoration_memory_geometry(&tiled),
            MemoryGeometry {
                width: 4096,
                height: 3072,
                batch: 1,
                frames: 1,
                reference_count: 0,
            }
        );
        // 128x128 at 4x -> 512x512, enlarged to one 1024 tile: a single forward.
        let single = plan(
            Dims::new(128, 128),
            &options,
            geometry,
            RESTORATION_MODEL_ID,
        )
        .unwrap();
        assert!(single.single_pass());
        assert_eq!(
            restoration_forward_pixels(&single),
            u64::from(single.padded.width) * u64::from(single.padded.height)
        );
        assert_eq!(
            (
                restoration_memory_geometry(&single).width,
                restoration_memory_geometry(&single).height
            ),
            (single.processing.width, single.processing.height)
        );

        let depth = DepthInputPlan {
            source_width: 1920,
            source_height: 1080,
            model_width: 1024,
            model_height: 576,
        };
        assert_eq!(
            depth_memory_geometry(&depth),
            MemoryGeometry {
                width: 1024,
                height: 576,
                batch: 1,
                frames: 1,
                reference_count: 0,
            }
        );
    }

    /// The crate's own registry carries all three routes on the shared seams, and the registry-wide
    /// conformance walks the catalog runs pass over it.
    ///
    /// *Mutation that reds this:* dropping any `register_*` call from
    /// `register_memory_contract_surfaces`.
    #[test]
    fn the_registry_publishes_three_conforming_memory_routes() {
        let registry = crate::provider_registry().unwrap();
        let expected: BTreeSet<&str> = [MODEL_ID, DEPTH_MODEL_ID, RESTORATION_MODEL_ID]
            .into_iter()
            .collect();
        assert_eq!(
            registry
                .memory_strategy_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            registry
                .memory_contract_fixture_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            registry
                .memory_behavior_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );

        gen_core_testkit::memory_contract_surface_registry_conformance(&registry);
        gen_core_testkit::memory_contract_surface_registry_facts_conformance(&registry, None);
        gen_core_testkit::memory_strategy_registry_conformance(
            &registry,
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        );

        let surfaces = registry.memory_contract_surfaces().unwrap();
        // Generation: three tiers x two policies (sc-26266); depth and restoration: bf16 x two.
        assert_eq!(surfaces.len(), 3 * 2 + 2 + 2, "witnesses per route");
        for surface in &surfaces {
            // Generation pairs with its registered generator; the two downstream routes are
            // generator-less and so composed.
            assert_eq!(
                surface.composed,
                surface.contract.provider_id != MODEL_ID,
                "{}",
                surface.contract.provider_id
            );
        }
    }

    /// The registered production contract is the one the loaded seam publishes, for each route.
    ///
    /// *Mutation that reds this:* building the loaded contract from a different spec than the one
    /// handed to `load` (e.g. `LoadSpec::new(spec.weights.clone())`).
    #[test]
    fn each_loaded_route_publishes_its_registered_contract() {
        let mut spec = generation_spec();
        spec.precision = Precision::Fp32;
        let generator = crate::model::load(&spec).unwrap();
        let contract = generator.memory_strategy_contract().unwrap();
        assert_eq!(contract, &(MEMORY_REGISTRATION.contract)(&spec).unwrap());
        assert_eq!(contract.architecture_facts.activation_dtype_width, Some(4));
        let context = fixture_context(&spec, contract, MemoryStrategy::StagedResidency);
        assert_eq!(
            generator.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );
        assert!(generator
            .begin_memory_strategy_request(&context)
            .unwrap()
            .is_some());

        let mut spec = restoration_spec();
        spec.precision = Precision::Fp32;
        let restorer = crate::IrisRestorer::load_on(&spec, &Device::Cpu).unwrap();
        let contract = restorer.memory_strategy_contract();
        assert_eq!(
            contract,
            &(RESTORE_MEMORY_REGISTRATION.contract)(&spec).unwrap()
        );
        let mut context = downstream_context(contract, RESTORE_MEMORY_MODE);
        context.selection.tier = loaded_tier(&spec);
        assert_eq!(
            restorer.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );

        let mut spec = depth_spec();
        spec.precision = Precision::Fp32;
        let depth = crate::depth::load_depth(&spec).unwrap();
        let contract = depth.memory_strategy_contract();
        assert_eq!(
            contract,
            &(DEPTH_MEMORY_REGISTRATION.contract)(&spec).unwrap()
        );
        // A model built from an already-resolved export publishes the contract of the load it is.
        let export = TaskExport::from_dir(
            &fixtures().join("tiny-depth"),
            IrisTask::Depth,
            DEPTH_MODEL_ID,
        )
        .unwrap();
        let built = crate::depth::load_depth_export(export, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(built.memory_strategy_contract(), contract);
        let mut context = downstream_context(contract, DEPTH_MEMORY_MODE);
        context.selection.tier = loaded_tier(&spec);
        assert_eq!(
            depth.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );
        context.mode = MemoryMode::Other(RESTORE_MEMORY_MODE.to_owned());
        assert!(matches!(
            depth.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// Every loader prices its contract before it reads a weight, so a spec whose contract cannot
    /// be priced fails `load` with the pricing reason — on every offload policy, including the
    /// `Sequential` one that would otherwise only meet the problem at the first render.
    ///
    /// *Mutations that red this:* pricing the weights-free contract at load instead of the
    /// resolved one (the `Sequential` load with a missing shard then succeeds); dropping
    /// `refuse_deferred_materialization` from a contract factory (a deferred spec then loads).
    #[test]
    fn a_load_whose_contract_cannot_be_priced_fails_with_the_pricing_reason() {
        let tmp = tempfile::tempdir().unwrap();
        let source = fixtures().join("tiny-snapshot/text_encoder");
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::copy(source.join(name), tmp.path().join(name)).unwrap();
        }
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"model.language_model.norm.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            // A text-encoder shard the index names is missing.
            let mut spec = generation_spec().with_offload_policy(policy);
            spec.components.insert(
                TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(tmp.path().to_path_buf()),
            );
            let priced = (MEMORY_REGISTRATION.contract)(&spec)
                .expect_err("an incomplete text encoder has no contract")
                .to_string();
            assert!(priced.contains("model-00001-of-00001"), "{priced}");
            let loaded = crate::model::load(&spec)
                .err()
                .expect("the load must fail")
                .to_string();
            assert_eq!(
                loaded, priced,
                "{policy:?}: load reports the pricing reason"
            );

            // An adapter the loader would refuse.
            let spec = generation_spec()
                .with_offload_policy(policy)
                .with_adapters(vec![AdapterSpec::new(
                    fixtures().join("iris_lora_depth_task.safetensors"),
                    1.0,
                    AdapterKind::Lora,
                )]);
            let loaded = crate::model::load(&spec)
                .err()
                .expect("the load must fail")
                .to_string();
            assert!(loaded.contains("task"), "{policy:?}: {loaded}");
        }

        // No Iris loader has a deferred materialization shape: a typed refusal at load.
        let deferred = |spec: LoadSpec| spec.with_load_shape(LoadShape::DeferredMaterialization);
        let refusals = [
            crate::model::load(&deferred(generation_spec()))
                .err()
                .map(|e| e.to_string()),
            crate::IrisRestorer::load_on(&deferred(restoration_spec()), &Device::Cpu)
                .err()
                .map(|e| gen_core::Error::from(e).to_string()),
            crate::depth::load_depth(&deferred(depth_spec()))
                .err()
                .map(|e| gen_core::Error::from(e).to_string()),
        ];
        for refusal in refusals {
            let refusal = refusal.expect("a deferred spec must not load");
            assert!(refusal.contains("DeferredMaterialization"), "{refusal}");
        }
        assert!(matches!(
            crate::model::load(&deferred(generation_spec())),
            Err(gen_core::Error::Unsupported(_))
        ));
    }

    /// The miniature text tower with an MLP width the GGML block divides (its released 48 does
    /// not, so the miniature has no quantized tier): `config.json` with `intermediate_size: 64`
    /// and a `model.safetensors` of the same keys with the MLP shapes widened, zero-filled.
    /// Pricing reads headers only. Returns the snapshot directory.
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
        let tensors: std::collections::HashMap<String, candle_gen::candle_core::Tensor> =
            safetensors_path_tensor_headers(source.join("model.safetensors"))
                .unwrap()
                .into_iter()
                .map(|header| {
                    let mut shape = header.shape.clone();
                    if header.name.contains(".mlp.gate_proj.")
                        || header.name.contains(".mlp.up_proj.")
                    {
                        shape[0] = 64;
                    } else if header.name.contains(".mlp.down_proj.") {
                        shape[1] = 64;
                    }
                    let zeros =
                        candle_gen::candle_core::Tensor::zeros(shape, DType::F32, &Device::Cpu)
                            .unwrap();
                    (header.name, zeros)
                })
                .collect();
        candle_gen::candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))
            .unwrap();
        dir
    }

    /// Bytes per `[out, in]` GGML weight: Q8_0 is 34 B / 32 weights, Q4_0 18 B / 32 and Q4_K
    /// 144 B / 256 (the same 9/16 B per weight), plus candle's CUDA row padding on a cuda build.
    fn ggml_bytes(out: usize, input: usize, bits: u32) -> u64 {
        let per_32 = if bits == 8 { 34 } else { 18 };
        let pad = if cfg!(feature = "cuda") {
            512 / 32 * per_32
        } else {
            0
        };
        (out * input / 32 * per_32 + pad) as u64
    }

    /// sc-26266: a quantized generation load prices what its loaders hold at the tier — every
    /// backbone projection at least 64 wide packed (the unaligned ones over an input zero-padded to
    /// the next multiple of 64), the text tower's attention/MLP projections packed, everything else
    /// as on the dense tier — and an adapted load prices the same packed backbone (the tier is one
    /// layout whatever the adapters; the stack is an overlay). Expectations are recomputed here
    /// from the headers, not through the provider's layout rule.
    ///
    /// *Mutations that red this:* pricing a quantized load at bf16; quantizing the token
    /// embedding; dropping the padding (with or without adapters); a block at the wrong width.
    #[test]
    fn a_quantized_generation_load_prices_the_packed_projections() {
        let scratch = tempfile::tempdir().unwrap();
        let tower = write_aligned_tiny_text_encoder(scratch.path());
        let mut spec = generation_spec();
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(tower.clone()),
        );
        let backbone_file = fixtures()
            .join("tiny-snapshot/iris")
            .join(BACKBONE_WEIGHTS_FILE);
        let backbone_headers = safetensors_path_tensor_headers(&backbone_file).unwrap();
        let dense_backbone = backbone_bytes(&backbone_file, 2, None).unwrap();
        let backbone_at = |bits: u32, pad: bool| {
            let mut total = dense_backbone;
            for header in &backbone_headers {
                if let [out, input] = header.shape.as_slice() {
                    if header.name.ends_with(".weight") && *input >= 64 && (input % 64 == 0 || pad)
                    {
                        total -= (out * input * 2) as u64;
                        total += ggml_bytes(*out, input.div_ceil(64) * 64, bits);
                    }
                }
            }
            total
        };
        let config = IrisConfig::from_dir(&fixtures().join("tiny-snapshot/iris")).unwrap();
        let tower_width = width(production_tower_dtype(&config.text_encoder).unwrap());
        let dense_tower = text_encoder_bytes(&tower, tower_width, None).unwrap();
        let tower_at = |bits: u32| {
            let mut total = dense_tower;
            for header in safetensors_path_tensor_headers(tower.join("model.safetensors")).unwrap()
            {
                if let [out, input] = header.shape.as_slice() {
                    if header.name.starts_with("model.language_model.layers.") {
                        total -= (out * input) as u64 * tower_width;
                        total += ggml_bytes(*out, *input, bits);
                    }
                }
            }
            total
        };
        // The miniature carries unaligned projections (the 170-wide SwiGLU down projections), so
        // padding is observable.
        assert!(backbone_at(4, true) < backbone_at(4, false));

        let bf16 = generation_contract(&spec).unwrap().asset_facts;
        assert_eq!(bf16.transformer_bytes, dense_backbone);
        assert_eq!(bf16.conditioning_bytes, dense_tower);
        let mut previous = bf16;
        for (quant, bits) in [(Quant::Q8, 8), (Quant::Q4, 4)] {
            let contract = generation_contract(&spec.clone().with_quant(quant)).unwrap();
            gen_core_testkit::memory_strategy_conformance(&contract);
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
            let facts = contract.asset_facts;
            assert_eq!(
                facts.transformer_bytes,
                backbone_at(bits, true),
                "{quant:?}"
            );
            assert_eq!(facts.conditioning_bytes, tower_at(bits), "{quant:?}");
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
        adapted.adapters = vec![fixture_adapter("iris_lora.safetensors", AdapterKind::Lora)];
        assert_eq!(
            generation_contract(&adapted)
                .unwrap()
                .asset_facts
                .transformer_bytes,
            backbone_at(4, true)
        );

        // The released miniature tower has no GGML-aligned MLP width, so it has no quantized tier.
        match generation_contract(&generation_spec().with_quant(Quant::Q4)) {
            Err(gen_core::Error::Unsupported(message)) => {
                assert!(message.contains("block size"), "{message}")
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
        for quant in [None, Some(Quant::Q8), Some(Quant::Q4)] {
            let mut spec = unresolved();
            spec.quantize = quant;
            let contract = weights_free_memory_strategy_contract(&spec).unwrap();
            for selected in [None, Some(Quant::Q8), Some(Quant::Q4)] {
                let mut at = unresolved();
                at.quantize = selected;
                let mut context =
                    fixture_context(&spec, &contract, MemoryStrategy::StagedResidency);
                context.selection.tier = loaded_tier(&at);
                let decision = generation_safety_check(&spec, &contract, &context);
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
}
