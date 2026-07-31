//! The six Mage-Flow variants, their [`ModelDescriptor`]s, and the explicit registration
//! constants and the registered RL [`Generator`] implementation.
//!
//! ## The variant matrix
//!
//! `microsoft` publishes **six** repositories whose `transformer/`, `vae/`, `scheduler/` and
//! `text_encoder/` JSON configs are byte-identical; only the transformer *weights* and the
//! README's default `steps`/`cfg` differ. **No config flag distinguishes a variant, or even
//! generation from editing** — the edit path differs purely by input-sequence assembly, so the
//! same backbone serves both. Turbo and Edit-Turbo are full distilled checkpoints, not LoRAs.
//!
//! | id | repo | task | steps | cfg |
//! | --- | --- | --- | --- | --- |
//! | `mage_flow` | `microsoft/Mage-Flow` (RL) | gen | 20 | 5.0 |
//! | `mage_flow_base` | `microsoft/Mage-Flow-Base` | gen | 30 | 5.0 |
//! | `mage_flow_turbo` | `microsoft/Mage-Flow-Turbo` | gen | 4 | 1.0 (off) |
//! | `mage_flow_edit` | `microsoft/Mage-Flow-Edit` | edit | 30 | 5.0 |
//! | `mage_flow_edit_base` | `microsoft/Mage-Flow-Edit-Base` | edit | 30 | 5.0 |
//! | `mage_flow_edit_turbo` | `microsoft/Mage-Flow-Edit-Turbo` | edit | 4 | 1.0 (off) |
//!
//! Each variant has its own id (rather than one id plus a switch), keeping the variant part
//! of the worker's model cache key.
//!
//! All generation and edit IDs are composed into the shipped platform catalog after their owning
//! stories validated the shared production paths and checkpoint-specific defaults.
//!
//! [`mlx_gen_catalog::provider_registry`]: https://docs.rs/mlx-gen-catalog

use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, AdapterResidencyMode, Error as CoreError, GenerationMemory,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryParameterRanges, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryResidentComponent, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision,
    MemorySelection, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LoadShape, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, Quant, Result, WeightsSource,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::config::{FAMILY, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE};
use crate::pipeline::MageComponentDirs;
use crate::{resolve_gs_key, GenerationSample, MageFlowPipeline};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "mage-flow-generation-peak-v1";
pub const STREAMING_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "mage-flow-generation-peak-v2-sequential-deferred-text-window";
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::TextEncoder;
// Kept false on the measurement branch. The Apple/Metal evidence run must land the per-tier request
// envelope and justify TRANSFORMER_WINDOW_SIZES before the provider publishes rung 4.
const STREAMING_CALIBRATION_AVAILABLE: bool = false;

/// Compatibility contract for callers that have only a tier. Without the load policy, shape, and
/// resolved component format, the text-encoder rung necessarily remains unavailable.
pub fn memory_strategy_contract(provider_id: &str, tier: Option<Quant>) -> MemoryProviderContract {
    memory_strategy_contract_with_adapters(
        provider_id,
        tier,
        &[],
        OffloadPolicy::Resident,
        LoadShape::EagerMaterialization,
        false,
    )
}

/// Build the load-exact Mage contract. Mage installs every adapter as a forward-time residual after
/// quantization, so a fully sizeable stack is independently resident and is part of the predicted
/// peak. An unreadable stack stays undeclared; the consumer can distinguish that evidence gap from
/// an adapter-free load and fail closed.
pub fn memory_strategy_contract_for_spec(
    provider_id: &str,
    spec: &LoadSpec,
) -> MemoryProviderContract {
    memory_strategy_contract_for_spec_with_text_stream(provider_id, spec, false)
}

fn memory_strategy_contract_for_spec_with_text_stream(
    provider_id: &str,
    spec: &LoadSpec,
    transfer_only_text_stream: bool,
) -> MemoryProviderContract {
    memory_strategy_contract_with_adapters(
        provider_id,
        spec.quantize,
        &spec.adapters,
        spec.offload_policy,
        spec.load_shape,
        transfer_only_text_stream,
    )
}

fn memory_strategy_contract_with_adapters(
    provider_id: &str,
    tier: Option<Quant>,
    adapters: &[mlx_gen::AdapterSpec],
    offload_policy: OffloadPolicy,
    load_shape: LoadShape,
    transfer_only_text_stream: bool,
) -> MemoryProviderContract {
    let streamable = STREAMING_CALIBRATION_AVAILABLE
        && transfer_only_text_stream
        && !provider_id.starts_with("mage_flow_edit")
        && matches!(offload_policy, OffloadPolicy::Sequential)
        && matches!(load_shape, LoadShape::DeferredMaterialization);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let mut variables = vec![
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
    ];
    if streamable {
        variables.push(MemoryFormulaVariable::ConditioningTokenCount);
        variables.push(MemoryFormulaVariable::TransformerWindowSize);
    }
    let adapter_bytes = adapter_stack_resident_bytes(adapters, AdapterResidencyMode::Additive);
    contract.formula = if let Some(adapter_bytes) = adapter_bytes.filter(|bytes| *bytes > 0) {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        contract.asset_facts.overlay_bytes = adapter_bytes;
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: vec![MemoryResidentComponent {
                id: "adapter_stack".to_owned(),
                kind: MemoryComponentKind::AdapterStack,
                resident_bytes: adapter_bytes,
                bounded_by: None,
            }],
        }
    } else {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    };
    contract.strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::Resident
                || (strategy == MemoryStrategy::BoundedTransformerResidency && streamable)
            {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            },
            parameters: if strategy == MemoryStrategy::BoundedTransformerResidency && streamable {
                MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                }
            } else {
                MemoryParameterRanges::default()
            },
        })
        .collect();
    contract.load_shape = load_shape;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        transformer_window_materialization: streamable,
        ..Default::default()
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(if streamable {
        STREAMING_MEMORY_CALIBRATION_FINGERPRINT
    } else {
        MEMORY_CALIBRATION_FINGERPRINT
    }));
    contract.asset_facts.base_bytes =
        (crate::memory::generation_resident_gb(tier) * 1_000_000_000.0).round() as u64;
    contract
}

struct MageMemoryScope {
    selection: MemorySelection,
    geometry: MemoryGeometry,
    finished: bool,
}

impl MageMemoryScope {
    fn synchronize_and_release(&mut self) -> CoreResult<()> {
        // `mlx_eval` is synchronous. Evaluating a sentinel on MLX's ordered default stream is a
        // terminal barrier for work queued by this request, including an error/cancellation exit
        // after a progress callback. Only after that barrier may allocator-retained buffers be
        // evicted for the next warm request.
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl MemoryRequestScope for MageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        if !matches!(
            self.selection.strategy,
            MemoryStrategy::Resident | MemoryStrategy::BoundedTransformerResidency
        ) {
            return Err(CoreError::Unsupported(
                "mage_flow: only resident and text-encoder block-window strategies are implemented"
                    .into(),
            ));
        }
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "mage_flow: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        request.memory = Some(if self.selection.strategy == MemoryStrategy::Resident {
            GenerationMemory::default()
        } else {
            GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: self.selection.parameters.transformer_window_size,
                transformer_window_component: Some(self.selection.parameters.window_component()),
                ..Default::default()
            }
        });
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
            "mage_flow: bounded decode is reserved for SC-15509".into(),
        ))
    }

    fn configure_attention(&mut self, _chunk_size: u32) -> CoreResult<()> {
        Err(CoreError::Unsupported(
            "mage_flow: bounded attention is reserved for SC-15509".into(),
        ))
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        let Some(window) = self.selection.parameters.transformer_window_size else {
            return Err(CoreError::Unsupported(
                "mage_flow: transformer-window hook used without an admitted window".into(),
            ));
        };
        if block_count == 0 || block_count > window || first_block >= 36 {
            return Err(CoreError::Unsupported(format!(
                "mage_flow: invalid encoder window first={first_block} count={block_count} for size {window}"
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> CoreResult<()> {
        self.synchronize_and_release()
    }
}

impl Drop for MageMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.synchronize_and_release();
        }
    }
}

/// Which published checkpoint a registered id serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MageVariant {
    /// `microsoft/Mage-Flow` — the Diffusion-NFT RL checkpoint, 20 steps.
    Rl,
    /// `microsoft/Mage-Flow-Base` — the training target, 30 steps.
    Base,
    /// `microsoft/Mage-Flow-Turbo` — Decoupled-DMD distilled, 4 steps, CFG off.
    Turbo,
    /// `microsoft/Mage-Flow-Edit` — instruction editing, 30 steps.
    Edit,
    /// `microsoft/Mage-Flow-Edit-Base` — instruction editing, 30 steps.
    EditBase,
    /// `microsoft/Mage-Flow-Edit-Turbo` — distilled instruction editing, 4 steps, CFG off.
    EditTurbo,
}

impl MageVariant {
    /// Registry id — always prefixed with [`FAMILY`].
    pub const fn id(self) -> &'static str {
        match self {
            Self::Rl => "mage_flow",
            Self::Base => "mage_flow_base",
            Self::Turbo => "mage_flow_turbo",
            Self::Edit => "mage_flow_edit",
            Self::EditBase => "mage_flow_edit_base",
            Self::EditTurbo => "mage_flow_edit_turbo",
        }
    }

    /// The upstream Hugging Face repository this variant's weights come from.
    pub const fn upstream_repo(self) -> &'static str {
        match self {
            Self::Rl => "microsoft/Mage-Flow",
            Self::Base => "microsoft/Mage-Flow-Base",
            Self::Turbo => "microsoft/Mage-Flow-Turbo",
            Self::Edit => "microsoft/Mage-Flow-Edit",
            Self::EditBase => "microsoft/Mage-Flow-Edit-Base",
            Self::EditTurbo => "microsoft/Mage-Flow-Edit-Turbo",
        }
    }

    /// `true` for the instruction-editing checkpoints, which consume reference images.
    pub const fn is_edit(self) -> bool {
        matches!(self, Self::Edit | Self::EditBase | Self::EditTurbo)
    }

    /// `true` for the Decoupled-DMD distilled checkpoints (4 steps, CFG off).
    pub const fn is_distilled(self) -> bool {
        matches!(self, Self::Turbo | Self::EditTurbo)
    }

    /// Published default step count, used when a request omits `steps`.
    pub const fn default_steps(self) -> u32 {
        match self {
            Self::Rl => 20,
            Self::Base | Self::Edit | Self::EditBase => 30,
            Self::Turbo | Self::EditTurbo => 4,
        }
    }

    /// Published default guidance scale. The distilled variants default to **1.0**, at which the
    /// reference builds no unconditional branch at all (`pipeline.py:326`, `:535`) — so CFG is
    /// genuinely off, not merely weightless.
    pub const fn default_cfg(self) -> f32 {
        if self.is_distilled() {
            1.0
        } else {
            5.0
        }
    }

    /// Every variant, in registration order.
    pub const ALL: [MageVariant; 6] = [
        Self::Rl,
        Self::Base,
        Self::Turbo,
        Self::Edit,
        Self::EditBase,
        Self::EditTurbo,
    ];
}

/// Every registered Mage-Flow id, in registration order.
pub const MODEL_IDS: [&str; 6] = [
    "mage_flow",
    "mage_flow_base",
    "mage_flow_turbo",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
];

/// Maximum homogeneous output count exposed through the platform request surface.
pub const MAX_COUNT: u32 = 8;

/// Immutable upstream revision used to establish the Turbo checkpoint fingerprint.
pub const TURBO_SNAPSHOT_REVISION: &str = "8523c9d1ae3cbe2148241e4769c918d0ab158ef8";
/// Immutable upstream revision used to establish the Base checkpoint fingerprint.
pub const BASE_SNAPSHOT_REVISION: &str = "59a9cfd58cf6ecef28245852c6bdace3f12428a2";
/// Immutable upstream revision used to establish the Edit-Base checkpoint fingerprint.
pub const EDIT_BASE_SNAPSHOT_REVISION: &str = "8654a7bc0283ab2946385230b5b2eb944e0b76ea";
/// Immutable upstream revision used to establish the Edit-Turbo checkpoint fingerprint.
pub const EDIT_TURBO_SNAPSHOT_REVISION: &str = "14427bd7627d3a25436497a5939e1096f6a0d523";
/// Immutable upstream revision used to establish the primary Edit checkpoint fingerprint.
pub const EDIT_SNAPSHOT_REVISION: &str = "b01d524f86498b7dabcc4b3572c6d264d786a16e";
// Every identity tensor is a **bias**, deliberately (sc-14980). Biases are never quantized, so a
// bias fingerprint is byte-identical in the dense flat snapshot and in every pre-quantized
// `<tier>/transformer/` artifact — one pinned hash per variant verifies all three tiers, with no
// per-tier constant and no weakening of the check on the packed path.
//
// Turbo previously pinned `img_in.weight`, which the Q4/Q8 packs rewrite into u32 codes; that hash
// could not survive a tier artifact. `transformer_blocks.0.attn.add_k_proj.bias` replaces it and is
// strictly stronger: measured over all six published checkpoints its first 4096 bytes yield **six
// distinct** digests, so this one tensor discriminates every variant (`img_in.bias`, by contrast,
// collides Base with RL and Edit-Base with Edit).
const TURBO_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const BASE_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const EDIT_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const TURBO_IDENTITY_BYTES: usize = 4096;
const BASE_IDENTITY_BYTES: usize = 4096;
const EDIT_IDENTITY_BYTES: usize = 4096;
const TURBO_IDENTITY_SHA256: &str =
    "52d3e3d2bcbb655f4575b71757081da3406dd13e5c58ef73173e070ff1c4767f";
const BASE_IDENTITY_SHA256: &str =
    "c6597b08e4efe45f7bbb5d2470c68e7975d71ca26dce13a1fb34db18ca6a9e3e";
const EDIT_BASE_IDENTITY_SHA256: &str =
    "bb53a04c20e5df443bb093c3f24027f9391f6d65e3edd60ed96546b050db717b";
const EDIT_TURBO_IDENTITY_SHA256: &str =
    "d387be05845ea0e0fc6b2bec5c05bccb3808c25a0123d9e2b3459e2e7f9705df";
const EDIT_IDENTITY_SHA256: &str =
    "bd24b2009764136298499d60750ded8ebdfa7950981d116e9937588471b2ecab";

/// Build a variant's weights-free descriptor.
///
/// Capability fields that later stories own are left at their conservative `Default` (`false` /
/// empty) rather than pre-announced: quant tiers are sc-14046, LoRA/LoKr routing is sc-14057, and
/// the curated sampler/scheduler menus are sc-14041's once the flow-match loop exists. A
/// descriptor is a promise to the worker, so the scaffold promises only what it can point at in
/// the published configs.
pub fn descriptor_for(variant: MageVariant) -> ModelDescriptor {
    ModelDescriptor {
        // The text encoder (8.875 GB) and VAE (0.345 GB) are BIT-IDENTICAL across all six Mage
        // variants — only the 8.232 GB DiT differs — so the SceneWorks mirrors host them once in a
        // shared components repo and stage them as caller-provisioned co-requisite dirs
        // (sc-14979): 58.65 GB for a full six-variant install instead of 105.04 GB. The DiT still
        // arrives as the base `WeightsSource::Dir`. A spec that stages neither falls back to the
        // flat published layout — see `resolve_component_dirs`.
        required_components: REQUIRED_COMPONENTS,
        id: variant.id(),
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Real CFG (`guidance_embed: false`) on the undistilled checkpoints; the distilled
            // ones run at cfg 1.0, where the reference never builds the negative branch.
            supports_negative_prompt: !variant.is_distilled(),
            supports_guidance: !variant.is_distilled(),
            conditioning: if variant.is_edit() {
                vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                ]
            } else {
                Vec::new()
            },
            // LoRA and LoKr both install through the one strict seam
            // ([`crate::adapters::apply_mage_adapters`] → `apply_adapters_strict`), applied in
            // [`assemble`] for EVERY variant — the adapter host is `MageTransformer`, which the
            // edit and generate variants share verbatim. Stated as engine capability, not product
            // exposure: which variants a user may attach an adapter to is decided by the catalog
            // manifest's `loraCompatibility` and the router, not here (sc-15328).
            supports_lora: true,
            supports_lokr: true,
            // Q4/Q8 tiers are sc-14046; `&[]` means dense-only, which is what the scaffold is.
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            // A platform request has one geometry/prompt and `count` independent seeds. The
            // pipeline additionally exposes heterogeneous geometry/prompt packs directly.
            max_count: MAX_COUNT,
            mac_only: true,
            ..Default::default()
        },
    }
}

/// Construct a Mage-Flow generator from a [`LoadSpec`].
///
/// `spec.adapters` carries LoRA/LoKr adapters to install on the DiT (sc-15328). They are applied
/// during assembly, AFTER the per-component tier quantization, through the strict shared seam
/// [`crate::adapters::apply_mage_adapters`] — stacked and mixed LoRA/LoKr, erroring rather than
/// silently dropping an unmatched target. An empty `adapters` is the unchanged no-adapter load.
pub fn load(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(
            "mage_flow variants support bf16/Q4/Q8 checkpoints".into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "mage_flow expects a diffusers snapshot directory".into(),
            ))
        }
    };
    match variant {
        MageVariant::Base => verify_checkpoint_identity(
            root,
            variant,
            BASE_SNAPSHOT_REVISION,
            BASE_IDENTITY_TENSOR,
            BASE_IDENTITY_BYTES,
            &[3072],
            BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Turbo => verify_checkpoint_identity(
            root,
            variant,
            TURBO_SNAPSHOT_REVISION,
            TURBO_IDENTITY_TENSOR,
            TURBO_IDENTITY_BYTES,
            &[3072],
            TURBO_IDENTITY_SHA256,
        )?,
        MageVariant::EditBase => verify_checkpoint_identity(
            root,
            variant,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Edit => verify_checkpoint_identity(
            root,
            variant,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_IDENTITY_SHA256,
        )?,
        MageVariant::EditTurbo => verify_checkpoint_identity(
            root,
            variant,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_TURBO_IDENTITY_SHA256,
        )?,
        _ => {}
    }
    let dirs = resolve_component_dirs(root, spec)?;
    assemble(variant, spec, dirs)
}

/// Construct a Mage-Flow generator from a caller-owned **fine-tuned transformer** (sc-15036,
/// epic 14034 F6) — the artifact a full base fine-tune (sc-14056) writes.
///
/// Two things distinguish this from [`load`], and both are forced by what a fine-tune *is*:
///
/// 1. **`spec.weights` is the fine-tuned `transformer/` component directory itself** (a
///    `config.json` + `diffusion_pytorch_model.safetensors` pair, exactly what the trainer's
///    `save_full_checkpoint` emits), NOT a diffusers snapshot root. A training run produces the
///    DiT alone; it never re-emits the text encoder or VAE, so there is no snapshot root to point
///    at and no flat-layout sibling to fall back to. Both shared components must therefore be
///    caller-staged in [`LoadSpec::components`] — normally the installed base model's own
///    `text_encoder/` + `vae/`, which a fine-tune leaves untouched and is numerically paired with
///    by construction. A missing one is a typed error here rather than a mid-load "No such file".
/// 2. **The pinned-checkpoint identity verification is skipped.** That guard exists to catch one
///    *published* variant's snapshot staged under another published variant's id, and it works by
///    hashing a prefix of `transformer_blocks.0.attn.add_k_proj.bias`. A full fine-tune trains
///    every DiT weight including that bias, so the guard would reject the user's own trained
///    checkpoint **by construction** — it cannot distinguish "fine-tuned from Base" from "the
///    wrong published checkpoint", because the caller is the only one who knows. `variant` states
///    which published checkpoint the run started from, and with it the architecture, the
///    sampling regime (steps / CFG / distillation) and the edit-vs-generate input assembly the
///    fine-tune inherits.
///
/// Deliberately kept off the registry `load(id, spec)` path (like Krea's
/// `load_from_native_dit_file`): a fine-tune is a caller-owned artifact at an arbitrary path, not
/// a published id, so it is reached through this explicit API rather than by resolving an id.
pub fn load_finetuned(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(
            "mage_flow fine-tuned checkpoints load as bf16/Q4/Q8".into(),
        ));
    }
    // Unlike [`load`], a fine-tuned checkpoint keeps refusing adapters (sc-15328). A Mage adapter is
    // trained against, and its residual is calibrated for, the *published* base weights; a full
    // fine-tune moves every DiT weight (sc-15277 measured ~96% of `img_in.weight` changed), so
    // stacking one on top composes two independent deltas the pair was never fit for. Refused here,
    // loudly and terminally, rather than silently honoured — and the router must not queue the
    // combination in the first place.
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "mage_flow fine-tuned checkpoints cannot take LoRA/LoKr adapters: the adapter is fit \
             against the published base weights, which a full fine-tune has moved. Render the \
             adapter on the base model, or use the fine-tune without adapters."
                .into(),
        ));
    }
    let transformer = match &spec.weights {
        WeightsSource::Dir(dir) => dir.clone(),
        WeightsSource::File(file) => {
            return Err(Error::Msg(format!(
                "mage_flow: a fine-tuned checkpoint is a transformer DIRECTORY (config.json + \
                 diffusion_pytorch_model.safetensors), got the file {}",
                file.display()
            )))
        }
    };
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            // No flat-layout fallback: a fine-tune dir has no component siblings, so silently
            // probing `<transformer>/text_encoder` would only turn a staging bug into a confusing
            // deep load failure.
            None => Err(Error::Msg(format!(
                "mage_flow: loading a fine-tuned transformer requires the '{id}' component to be \
                 staged from the installed base model — a training run produces the transformer \
                 alone"
            ))),
        }
    };
    let dirs = MageComponentDirs {
        transformer,
        text_encoder: staged(COMPONENT_TEXT_ENCODER)?,
        vae: staged(COMPONENT_VAE)?,
    };
    assemble(variant, spec, dirs)
}

/// Build the pipeline + generator from already-resolved component dirs — the half [`load`] and
/// [`load_finetuned`] share once each has decided *where* the components live.
fn assemble(
    variant: MageVariant,
    spec: &LoadSpec,
    dirs: MageComponentDirs,
) -> Result<Box<dyn Generator>> {
    let part = if variant.is_edit() {
        crate::vae::VaePart::Both
    } else {
        crate::vae::VaePart::Decode
    };
    let stream_text_encoder = !variant.is_edit()
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && crate::pipeline::text_encoder_window_is_transfer_only(
            &dirs.text_encoder,
            spec.quantize.map(Quant::bits),
        )?;
    let mut pipeline = MageFlowPipeline::load_components_with_text_stream(
        &dirs,
        spec.quantize.map(Quant::bits),
        part,
        stream_text_encoder,
    )?;
    // Install LoRA/LoKr adapters AFTER the per-component tier quantization (sc-15328), matching the
    // Chroma/FLUX composition: the adapter is a forward-time residual over the quantized base, so a
    // Q4/Q8 tier and a bf16 tier take the same path. No-op when `spec.adapters` is empty; any
    // unmatched target errors loudly rather than being silently dropped (`apply_adapters_strict`).
    crate::adapters::apply_mage_adapters(&mut pipeline.transformer, &spec.adapters)?;
    Ok(Box::new(MageFlow {
        variant,
        descriptor: descriptor_for(variant),
        tier: spec.quantize,
        memory_strategy_contract: memory_strategy_contract_for_spec_with_text_stream(
            variant.id(),
            spec,
            stream_text_encoder,
        ),
        pipeline,
    }))
}

/// The caller-provisioned component ids Mage-Flow advertises (sc-14979).
pub const COMPONENT_TEXT_ENCODER: &str = "text_encoder";
/// The caller-provisioned VAE component id (sc-14979).
pub const COMPONENT_VAE: &str = "vae";
/// Both shared components, in descriptor order.
pub const REQUIRED_COMPONENTS: &[&str] = &[COMPONENT_TEXT_ENCODER, COMPONENT_VAE];

/// Resolve where each component's weights live for this load.
///
/// **Split layout (the SceneWorks mirrors, sc-14980/sc-14979).** `spec.weights` is the variant's
/// per-tier dir (`<variant snapshot>/<tier>/`), holding the DiT alone; the text encoder and VAE —
/// bit-identical across all six variants — are staged by the caller in [`LoadSpec::components`] as
/// exact component dirs resolved from the shared components mirror. Six installs cost 58.65 GB
/// instead of 105.04 GB.
///
/// **Flat layout (upstream snapshots, existing installs, arbitrary user paths).** No components are
/// staged and every component sits directly under `spec.weights`. This fallback is why the split is
/// not a breaking change: a repo/revision without tier subdirs, and every `#[ignore]`d real-weights
/// test that points at a raw `microsoft/Mage-Flow*` snapshot, keeps loading unchanged.
///
/// The two are distinguished per component, not globally, so a partially-staged spec is still
/// coherent. Unknown component ids are rejected rather than ignored.
pub(crate) fn resolve_component_dirs(root: &Path, spec: &LoadSpec) -> Result<MageComponentDirs> {
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str, fallback: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            None => Ok(root.join(fallback)),
        }
    };
    Ok(MageComponentDirs {
        transformer: root.join("transformer"),
        text_encoder: staged(COMPONENT_TEXT_ENCODER, "text_encoder")?,
        vae: staged(COMPONENT_VAE, "vae")?,
    })
}

/// Verify bytes from a weight-bearing tensor, not a path or model-card label.
///
/// All Mage-Flow variants share byte-identical configs and tensor schemas, so those cannot detect
/// one checkpoint accidentally routed under another variant id. The caller supplies a tensor,
/// byte count, and hash pinned to the immutable upstream revision. Base deliberately uses an
/// attention bias because its `img_in.weight` prefix is byte-identical to RL's.
fn verify_checkpoint_identity(
    root: &Path,
    variant: MageVariant,
    revision: &str,
    tensor_name: &str,
    identity_bytes: usize,
    expected_shape: &[u64],
    expected_sha256: &str,
) -> Result<()> {
    let id = variant.id();
    let path = root
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let mut file = std::fs::File::open(&path).map_err(|error| {
        Error::Msg(format!(
            "{id}: cannot open transformer checkpoint {}: {error}",
            path.display()
        ))
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 1_048_576 {
        return Err(Error::Msg(format!(
            "{id}: invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let metadata: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        Error::Msg(format!(
            "{id}: invalid safetensors header in {}: {error}",
            path.display()
        ))
    })?;
    let tensor = metadata
        .get(tensor_name)
        .ok_or_else(|| Error::Msg(format!("{id}: missing {tensor_name}")))?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16")
        || tensor.get("shape").and_then(serde_json::Value::as_array)
            != Some(
                &expected_shape
                    .iter()
                    .map(|&dimension| serde_json::json!(dimension))
                    .collect::<Vec<_>>(),
            )
    {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has the wrong dtype or shape"
        )));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Msg(format!("{id}: {tensor_name} has no data offsets")))?;
    let start = offsets.first().and_then(serde_json::Value::as_u64);
    let end = offsets.get(1).and_then(serde_json::Value::as_u64);
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has invalid data offsets"
        )));
    };
    if end.saturating_sub(start) < identity_bytes as u64 {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} is too short for identity verification"
        )));
    }
    file.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut bytes = vec![0u8; identity_bytes];
    file.read_exact(&mut bytes)?;
    let got = format!("{:x}", Sha256::digest(bytes));
    if got != expected_sha256 {
        return Err(Error::Msg(format!(
            "{id}: checkpoint fingerprint mismatch for {tensor_name} \
             (expected revision {revision}, got sha256 {got}); \
             another Mage-Flow checkpoint cannot serve the {id} id"
        )));
    }
    Ok(())
}

pub struct MageFlow {
    variant: MageVariant,
    descriptor: ModelDescriptor,
    tier: Option<Quant>,
    memory_strategy_contract: MemoryProviderContract,
    pipeline: MageFlowPipeline,
}

fn request_context_error(
    provider_id: &str,
    variant: MageVariant,
    tier: Option<Quant>,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<String> {
    if context.selection.tier.precision != Precision::Bf16 || context.selection.tier.quant != tier {
        return Some(format!(
            "{provider_id}: request tier {:?} does not match loaded BF16/{tier:?}",
            context.selection.tier
        ));
    }
    let expected_mode = if variant.is_edit() {
        MemoryMode::Edit
    } else {
        MemoryMode::TextToImage
    };
    if context.mode != expected_mode {
        return Some(format!(
            "{provider_id}: request mode {:?} does not match {expected_mode:?}",
            context.mode
        ));
    }
    let expected_calibration = contract.calibration.as_ref();
    if context.calibration_abi != mlx_gen::gen_core::MEMORY_CALIBRATION_ABI
        || expected_calibration
            .is_none_or(|identity| context.calibration_fingerprint != identity.fingerprint)
    {
        return Some(format!(
            "{provider_id}: request calibration identity {}/{:?} does not match provider {}/{:?}",
            context.calibration_abi,
            context.calibration_fingerprint,
            mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            expected_calibration.map(|identity| identity.fingerprint.as_str())
        ));
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return Some(error.to_string());
    }
    if context.budget.total_bytes == 0 {
        return Some(format!("{provider_id}: request budget is unavailable"));
    }
    let required_total_peak_bytes = ((crate::memory::generation_peak_gb(
        tier,
        context.geometry.width,
        context.geometry.height,
        context.geometry.batch,
    ) * 1_000_000_000.0)
        .round() as u64)
        .saturating_add(contract.auxiliary_resident_bytes());
    let maximum_resident_credit = contract.total_resident_bytes();
    let credited_resident_bytes =
        required_total_peak_bytes.saturating_sub(context.predicted_peak_bytes);
    if context.predicted_peak_bytes > required_total_peak_bytes
        || credited_resident_bytes > maximum_resident_credit
        || credited_resident_bytes > context.budget.committed_bytes
    {
        return Some(format!(
            "{provider_id}: caller peak {} is inconsistent with provider total {}, resident \
             envelope {}, and committed bytes {}",
            context.predicted_peak_bytes,
            required_total_peak_bytes,
            maximum_resident_credit,
            context.budget.committed_bytes
        ));
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return Some(format!(
            "{provider_id}: predicted incremental peak {} exceeds effective budget {}",
            context.predicted_peak_bytes,
            context.budget.effective_bytes()
        ));
    }
    None
}

impl Generator for MageFlow {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.memory_strategy_contract)
    }

    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        if let Some(reason) = request_context_error(
            self.descriptor.id,
            self.variant,
            self.tier,
            &self.memory_strategy_contract,
            context,
        ) {
            return MemorySafetyDecision::Reject { reason };
        }
        let safe_gb = match crate::memory::production_safe_budget_gb() {
            Ok(safe_gb) => safe_gb,
            Err(error) => {
                return MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        };
        match crate::memory::ensure_generation_fits(
            self.tier,
            context.geometry.width,
            context.geometry.height,
            context.geometry.batch,
            safe_gb,
        ) {
            Ok(()) => MemorySafetyDecision::Accept,
            Err(error) => MemorySafetyDecision::Reject {
                reason: error.to_string(),
            },
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> CoreResult<Option<Box<dyn MemoryRequestScope + '_>>> {
        if let MemorySafetyDecision::Reject { reason } = self.memory_strategy_safety_check(context)
        {
            return Err(CoreError::Unsupported(reason));
        }
        Ok(Some(Box::new(MageMemoryScope {
            selection: context.selection,
            geometry: context.geometry,
            finished: false,
        })))
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_generation_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        crate::memory::ensure_generation_fits(
            self.tier,
            req.width,
            req.height,
            req.count,
            crate::memory::production_safe_budget_gb()?,
        )?;
        if req.cancel.is_cancelled() {
            return Err(mlx_gen::gen_core::Error::Canceled);
        }
        let steps = req.steps.unwrap_or(self.variant.default_steps());
        let cfg = req.guidance.unwrap_or(self.variant.default_cfg());
        let seed = req.seed.unwrap_or(0) as i64;
        let key = resolve_gs_key(None)?;
        let negative_prompt = req.negative_prompt.as_deref().unwrap_or(" ");
        if self.variant.is_edit() {
            let references = edit_references(req)?;
            let mut images = Vec::with_capacity(req.count as usize);
            for index in 0..req.count {
                let trace = self.pipeline.edit_trace(
                    &req.prompt,
                    negative_prompt,
                    &references,
                    req.height,
                    req.width,
                    steps as usize,
                    cfg,
                    seed.wrapping_add(index as i64),
                    &key,
                    false,
                    on_progress,
                )?;
                if req.cancel.is_cancelled() {
                    return Err(CoreError::Canceled);
                }
                mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
                images.push(Image {
                    width: req.width,
                    height: req.height,
                    pixels: trace
                        .image_u8
                        .try_as_slice::<u8>()
                        .map_err(|error| {
                            Error::Msg(format!(
                                "mage_flow edit: RGB8 output is not host-readable: {error}"
                            ))
                        })?
                        .to_vec(),
                });
            }
            return Ok(GenerationOutput::Images(images));
        }
        let samples = (0..req.count)
            .map(|index| GenerationSample {
                prompt: &req.prompt,
                negative_prompt,
                height: req.height,
                width: req.width,
                seed: seed.wrapping_add(index as i64),
            })
            .collect::<Vec<_>>();
        let encoder_window =
            resolve_encoder_window(req, self.pipeline.text_encoder.is_streamable())?;
        let traces = self
            .pipeline
            .generate_batch_trace_with_encoder_window(
                &samples,
                steps as usize,
                cfg,
                &key,
                false,
                encoder_window,
                &req.cancel,
                on_progress,
            )?
            .samples;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }
        let mut images = Vec::with_capacity(traces.len());
        for trace in traces {
            mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
            let pixels = trace
                .image_u8
                .try_as_slice::<u8>()
                .map_err(|e| {
                    Error::Msg(format!("mage_flow: RGB8 output is not host-readable: {e}"))
                })?
                .to_vec();
            images.push(Image {
                width: req.width,
                height: req.height,
                pixels,
            });
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn resolve_encoder_window(req: &GenerationRequest, streamable: bool) -> Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory.transformer_window_component.unwrap_or_default();
    if component != TransformerComponent::TextEncoder {
        return Err(Error::Msg(format!(
            "mage_flow implements rung 4 only for TextEncoder, got {component:?}"
        )));
    }
    if !streamable {
        return Err(Error::Msg(
            "mage_flow text-encoder window requires a Sequential + DeferredMaterialization load"
                .into(),
        ));
    }
    let window = memory.transformer_window_size.ok_or_else(|| {
        Error::Msg("mage_flow text-encoder window selected without a window size".into())
    })?;
    usize::try_from(window)
        .ok()
        .filter(|window| *window > 0)
        .map(Some)
        .ok_or_else(|| Error::Msg("mage_flow text-encoder window must be greater than zero".into()))
}

fn edit_references(req: &GenerationRequest) -> Result<Vec<image::RgbImage>> {
    let mut images = Vec::new();
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::Reference { image, .. } => images.push(image),
            Conditioning::MultiReference { images: refs } => images.extend(refs),
            _ => {}
        }
    }
    if images.is_empty() {
        return Err(Error::Msg(
            "mage_flow edit: Reference or MultiReference conditioning is required".into(),
        ));
    }
    images
        .into_iter()
        .map(|image| {
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()).ok_or_else(
                || Error::Msg("mage_flow edit: reference image is not valid RGB8".into()),
            )
        })
        .collect()
}

fn validate_generation_request(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<()> {
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !req.width.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
        || !req.height.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
    {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "mage_flow dimensions must be divisible by {REQUIRED_SIZE_MULTIPLE}"
        )));
    }
    Ok(())
}

/// Every side of a Mage-Flow request must be a multiple of this (the VAE's 16× downsample;
/// `patch_size == 1` adds no further stride). Re-exported at the model layer because SceneWorks
/// pins each advertised resolution bucket to an engine stride constant.
pub const REQUIRED_SIZE_MULTIPLE: u32 = SIZE_MULTIPLE;

/// Per-component on-disk footprint used by the worker's staged-residency fit gate.
///
/// Mage-Flow quantizes all three weight-bearing components, so the accounting must follow the
/// selected snapshot tree rather than a transformer-only approximation. Missing/unreadable
/// subdirectories contribute zero bytes here; checkpoint identity/load validation separately rejects
/// missing required components before generation.
///
/// **Resolves through [`resolve_component_dirs`], not the spec's weights root** (sc-15154). On the
/// SPLIT layout the root is the variant's per-tier dir and holds the DiT *alone* — the text encoder
/// and VAE are staged in [`LoadSpec::components`] from the shared mirror. Summing subdirs of the
/// root therefore reported the DiT's bytes as the whole model: 2.33 GB for a q4 tier whose real
/// install is 7.00 GB, and 8.23 GB for bf16's 17.46 GB. The worker's fit gate adds a flat activation
/// headroom to this number, so the shortfall surfaced as an over-budget message quoting a figure
/// that tracked neither the tier's weights nor its measured peak. The flat-layout fallback is
/// unchanged: with nothing staged, `resolve_component_dirs` returns `root/<component>` exactly as
/// before.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::gen_core::Error::Msg(
            "mage_flow: per-component footprint requires a snapshot directory, not a single \
             .safetensors file"
                .to_owned(),
        ));
    };
    let dirs = resolve_component_dirs(root, spec)?;
    Ok(mlx_gen::PerComponentBytes {
        text_encoder: mlx_gen::safetensors_path_bytes(dirs.text_encoder),
        dit: mlx_gen::safetensors_path_bytes(dirs.transformer),
        vae: mlx_gen::safetensors_path_bytes(dirs.vae),
    })
}

macro_rules! mage_registrations {
    ( $( $variant:ident => ( $descriptor_fn:ident, $load_fn:ident, $registration:ident ) ),+ $(,)? ) => {
        $(
            /// This variant's weights-free descriptor (see [`descriptor_for`]).
            pub fn $descriptor_fn() -> ModelDescriptor {
                descriptor_for(MageVariant::$variant)
            }

            fn $load_fn(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
                load(MageVariant::$variant, spec)
            }

            mlx_gen::register_generators! {
                pub const $registration = $descriptor_fn => $load_fn;
                footprint = component_footprint
            }
        )+

        /// The explicit registration constants, in variant order — the surface a catalog crate
        /// composes.
        pub const REGISTRATIONS: &[mlx_gen::registry::ModelRegistration] = &[ $($registration),+ ];
    };
}

mage_registrations! {
    Rl => (descriptor, load_rl, REGISTRATION),
    Base => (descriptor_base, load_base, REGISTRATION_BASE),
    Turbo => (descriptor_turbo, load_turbo, REGISTRATION_TURBO),
    Edit => (descriptor_edit, load_edit, REGISTRATION_EDIT),
    EditBase => (descriptor_edit_base, load_edit_base, REGISTRATION_EDIT_BASE),
    EditTurbo => (descriptor_edit_turbo, load_edit_turbo, REGISTRATION_EDIT_TURBO),
}

macro_rules! mage_memory_registration {
    ($name:ident, $id:literal) => {
        pub const $name: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $id,
                contract: |spec| Ok(memory_strategy_contract_for_spec($id, spec)),
            };
    };
}

mage_memory_registration!(MEMORY_REGISTRATION, "mage_flow");
mage_memory_registration!(MEMORY_REGISTRATION_BASE, "mage_flow_base");
mage_memory_registration!(MEMORY_REGISTRATION_TURBO, "mage_flow_turbo");
mage_memory_registration!(MEMORY_REGISTRATION_EDIT, "mage_flow_edit");
mage_memory_registration!(MEMORY_REGISTRATION_EDIT_BASE, "mage_flow_edit_base");
mage_memory_registration!(MEMORY_REGISTRATION_EDIT_TURBO, "mage_flow_edit_turbo");

#[cfg(test)]
mod tests {
    use super::*;

    fn q4_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q4)
    }

    #[test]
    fn memory_strategy_contract_is_truthful_resident_only_mlx_adoption() {
        use mlx_gen::gen_core::{MemoryStrategySupport, MEMORY_CALIBRATION_ABI};

        let contract = memory_strategy_contract("mage_flow", Some(Quant::Q4));
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.provider_id, "mage_flow");
        assert_eq!(
            contract
                .calibration
                .as_ref()
                .map(|identity| (identity.abi, identity.fingerprint.as_str())),
            Some((MEMORY_CALIBRATION_ABI, MEMORY_CALIBRATION_FINGERPRINT))
        );
        assert!(matches!(
            contract
                .capability(MemoryStrategy::Resident)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        ));
        for strategy in MemoryStrategy::ALL
            .into_iter()
            .filter(|strategy| *strategy != MemoryStrategy::Resident)
        {
            assert!(matches!(
                contract
                    .capability(strategy)
                    .map(|capability| &capability.support),
                Some(MemoryStrategySupport::Missing)
            ));
        }
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            }
        ));
    }

    #[test]
    fn adapter_contract_adds_load_exact_residency_and_preserves_missing_evidence() {
        let root =
            std::env::temp_dir().join(format!("mage-memory-adapters-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let adapter = root.join("mage.safetensors");
        std::fs::write(&adapter, vec![0_u8; 4096]).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![
            mlx_gen::AdapterSpec::new(adapter, 1.0, mlx_gen::AdapterKind::Lora),
        ]);
        let contract = memory_strategy_contract_for_spec("mage_flow", &spec);

        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.auxiliary_resident_bytes(), 4096);
        assert_eq!(contract.asset_facts.overlay_bytes, 4096);
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert_eq!(
            contract
                .predicted_peak_from_base(100)
                .predicted_peak_bytes(),
            4196
        );

        let missing = LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![
            mlx_gen::AdapterSpec::new(
                root.join("missing.safetensors"),
                1.0,
                mlx_gen::AdapterKind::Lora,
            ),
        ]);
        let missing_contract = memory_strategy_contract_for_spec("mage_flow", &missing);
        assert_eq!(missing_contract.auxiliary_resident_bytes(), 0);
        assert!(!missing_contract
            .formula
            .uses(MemoryFormulaVariable::OverlayBytes));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_calibrated_transfer_only_sequential_deferred_generation_loads_publish_the_window() {
        use mlx_gen::gen_core::MemoryStrategySupport;

        for (policy, shape, transfer_only, implemented) in [
            (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
                true,
                true,
            ),
            (
                OffloadPolicy::Resident,
                LoadShape::DeferredMaterialization,
                true,
                false,
            ),
            (
                OffloadPolicy::Sequential,
                LoadShape::EagerMaterialization,
                true,
                false,
            ),
            (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
                false,
                false,
            ),
        ] {
            let mut spec = q4_spec().with_offload_policy(policy);
            spec.load_shape = shape;
            let contract = memory_strategy_contract_for_spec_with_text_stream(
                "mage_flow",
                &spec,
                transfer_only,
            );
            let rung = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert_eq!(
                matches!(rung.support, MemoryStrategySupport::Implemented),
                implemented && STREAMING_CALIBRATION_AVAILABLE,
                "policy={policy:?} shape={shape:?} transfer_only={transfer_only}"
            );
            if implemented && STREAMING_CALIBRATION_AVAILABLE {
                assert_eq!(
                    rung.parameters.transformer_window_components,
                    vec![TransformerComponent::TextEncoder]
                );
                assert_eq!(rung.parameters.transformer_window_sizes, vec![1]);
                assert!(contract.lifecycle.transformer_window_materialization);
                assert!(contract.conformance_errors().is_empty());
            }
        }

        let mut edit = q4_spec().with_offload_policy(OffloadPolicy::Sequential);
        edit.load_shape = LoadShape::DeferredMaterialization;
        let contract =
            memory_strategy_contract_for_spec_with_text_stream("mage_flow_edit", &edit, true);
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Missing)
        ));
    }

    #[test]
    fn resident_safety_recomputes_peak_and_binds_calibration_identity() {
        use mlx_gen::gen_core::{
            MemoryBudget, MemoryCacheState, MemoryNumericTier, MemoryStrategyParameters,
            MEMORY_CALIBRATION_ABI,
        };

        let contract = memory_strategy_contract("mage_flow", Some(Quant::Q4));
        let required = (crate::memory::generation_peak_gb(Some(Quant::Q4), 512, 512, 1)
            * 1_000_000_000.0)
            .round() as u64;
        let valid = MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
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
                total_bytes: required + 1_000_000_000,
                committed_bytes: contract.asset_facts.base_bytes,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: required - contract.asset_facts.base_bytes,
            cache_state: MemoryCacheState::Warm,
            evidence_revision: "test".to_owned(),
        };
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &valid
        )
        .is_none());

        let mut wrong_identity = valid.clone();
        wrong_identity.calibration_fingerprint = "stale".to_owned();
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &wrong_identity
        )
        .unwrap()
        .contains("calibration identity"));

        let mut zero_zero = valid.clone();
        zero_zero.budget.total_bytes = 0;
        zero_zero.budget.committed_bytes = 0;
        zero_zero.predicted_peak_bytes = 0;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &zero_zero
        )
        .unwrap()
        .contains("budget is unavailable"));

        let mut underreported = valid;
        underreported.predicted_peak_bytes = 0;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &underreported
        )
        .unwrap()
        .contains("inconsistent"));

        let mut uncharged_resident_credit = underreported;
        uncharged_resident_credit.predicted_peak_bytes = required - contract.asset_facts.base_bytes;
        uncharged_resident_credit.budget.committed_bytes = 0;
        uncharged_resident_credit.budget.total_bytes =
            uncharged_resident_credit.predicted_peak_bytes;
        assert!(request_context_error(
            "mage_flow",
            MageVariant::Rl,
            Some(Quant::Q4),
            &contract,
            &uncharged_resident_credit
        )
        .unwrap()
        .contains("committed bytes"));
    }

    #[test]
    fn resident_scope_reapplies_request_state_after_cancel_cleanup() {
        let selection = MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: mlx_gen::gen_core::MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: crate::quant::COMPONENT_PRECISION_FLOORS,
            },
        };
        let geometry = MemoryGeometry {
            width: 1024,
            height: 768,
            batch: 3,
            frames: 1,
        };
        let mut canceled = MageMemoryScope {
            selection,
            geometry,
            finished: false,
        };
        let mut first = GenerationRequest {
            prompt: "first".to_owned(),
            width: 1024,
            height: 768,
            count: 1,
            ..Default::default()
        };
        canceled.configure_request(&mut first).unwrap();
        assert_eq!(first.memory, Some(GenerationMemory::default()));
        canceled.finish(MemoryRunOutcome::Canceled).unwrap();
        assert!(canceled.finished);

        let mut warm = MageMemoryScope {
            selection,
            geometry,
            finished: false,
        };
        let mut follow_up = GenerationRequest {
            prompt: "follow-up".to_owned(),
            width: 1024,
            height: 768,
            count: 1,
            ..Default::default()
        };
        warm.configure_request(&mut follow_up).unwrap();
        warm.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(warm.finished, "a warm follow-up owns fresh terminal state");
    }

    /// sc-15154 — the footprint must follow the SPLIT layout's staged components, not the tier dir.
    ///
    /// The discriminating case: the same fake tier tree scanned with and without the components
    /// staged. A footprint that sums subdirs of `spec.weights` scores the split spec at the DiT's
    /// bytes alone and cannot tell the two specs apart, which is exactly what made the worker's
    /// over-budget message quote a figure unrelated to the tier's real install.
    #[test]
    fn the_footprint_counts_staged_components_not_just_the_tier_dir() {
        let root = std::env::temp_dir().join(format!(
            "mage-footprint-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let write = |dir: std::path::PathBuf, bytes: usize| {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
            dir
        };
        // SPLIT: the variant tier dir holds the DiT; the shared mirror holds the TE + VAE.
        let tier = root.join("q4");
        write(tier.join("transformer"), 300);
        let te = write(root.join("shared/q4/text_encoder"), 700);
        let vae = write(root.join("shared/q4/vae"), 50);

        let split = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier.clone()))
            .with_component(COMPONENT_TEXT_ENCODER, mlx_gen::WeightsSource::Dir(te))
            .with_component(COMPONENT_VAE, mlx_gen::WeightsSource::Dir(vae));
        let got = component_footprint(&split).unwrap();
        assert_eq!(
            (got.dit, got.text_encoder, got.vae),
            (300, 700, 50),
            "the staged text encoder and VAE are part of what this tier loads"
        );

        // ...and the same spec with nothing staged sees only the DiT — the pre-fix behavior, kept
        // here so the assertion above is visibly about the staging and not about the tree.
        let unstaged = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier));
        let got = component_footprint(&unstaged).unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 0, 0));

        // FLAT (upstream snapshots / legacy installs): every component under the root, nothing
        // staged. Unchanged by this fix.
        let flat = root.join("flat");
        write(flat.join("transformer"), 300);
        write(flat.join("text_encoder"), 700);
        write(flat.join("vae"), 50);
        let got = component_footprint(&mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(flat)))
            .unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 700, 50));

        std::fs::remove_dir_all(root).ok();
    }

    /// A minimal but structurally valid safetensors file carrying exactly the Base identity tensor
    /// (`transformer_blocks.0.attn.add_k_proj.bias`, BF16, `[3072]`) filled with `fill` — enough for
    /// [`verify_checkpoint_identity`] to parse the header, seek, and hash. Nothing loads these
    /// weights; the tests below only exercise which *guard* fires.
    fn load_error(result: Result<Box<dyn Generator>>, context: &str) -> String {
        match result {
            Ok(_) => panic!("{context}"),
            Err(error) => error.to_string(),
        }
    }

    fn write_identity_only_checkpoint(path: &Path, fill: u8) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let payload = 3072 * 2; // BF16 [3072]
        let header = format!(
            "{{\"{BASE_IDENTITY_TENSOR}\":{{\"dtype\":\"BF16\",\"shape\":[3072],\"data_offsets\":[0,{payload}]}}}}"
        );
        let mut bytes = Vec::with_capacity(8 + header.len() + payload);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(std::iter::repeat_n(fill, payload));
        std::fs::write(path, bytes).unwrap();
    }

    /// sc-15036 — `load_finetuned` must get PAST the pinned-checkpoint identity guard that `load`
    /// enforces, because a full base fine-tune (sc-14056) rewrites every DiT weight *including*
    /// `transformer_blocks.0.attn.add_k_proj.bias`, so `load` rejects the user's own trained
    /// checkpoint by construction.
    ///
    /// Discriminating in both directions on ONE fabricated checkpoint:
    ///   * `load(Base, …)` must fail with the **fingerprint-mismatch** message — delete the guard
    ///     and this half fails;
    ///   * `load_finetuned(Base, …)` must fail with something else entirely (it reaches component
    ///     staging) — route `load_finetuned` back through the guard and this half fails.
    #[test]
    fn load_finetuned_bypasses_the_pinned_checkpoint_identity_guard() {
        let root = std::env::temp_dir().join(format!(
            "mage-finetuned-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let transformer = root.join("transformer");
        write_identity_only_checkpoint(
            &transformer.join("diffusion_pytorch_model.safetensors"),
            0x5a,
        );

        // `load` sees a snapshot ROOT and hashes `<root>/transformer/…`: the fill is not the pinned
        // Base fingerprint, so the guard fires.
        let published = load_error(
            load(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "a checkpoint whose identity tensor moved must not load as published Base",
        );
        assert!(
            published.contains("checkpoint fingerprint mismatch"),
            "expected the identity guard to fire, got: {published}"
        );

        // `load_finetuned` is handed the SAME root — deliberately, so the mutation "delegate to
        // `load`" is caught: under it this call would report the fingerprint mismatch above. The
        // real entrypoint treats the path as the transformer dir itself and never opens
        // `<path>/transformer`, so it gets past identity and fails later, at the actual load.
        // Pid-keyed so the "this path does not exist" premise cannot be broken by a leftover from,
        // or a concurrent, second `cargo test` process sharing `$TMPDIR`.
        let staged = std::env::temp_dir().join(format!(
            "mage-finetuned-nonexistent-component-{}",
            std::process::id()
        ));
        let finetuned = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(staged.clone()))
                    .with_component(COMPONENT_VAE, WeightsSource::Dir(staged)),
            ),
            "the fabricated checkpoint has no real components to load",
        );
        assert!(
            !finetuned.contains("checkpoint fingerprint"),
            "load_finetuned must not enforce the published-checkpoint fingerprint, got: {finetuned}"
        );
        // ...and the transformer dir it DID read is the one it was handed.
        assert!(
            transformer.is_dir(),
            "fixture sanity: the nested published-layout transformer dir exists"
        );

        std::fs::remove_dir_all(root).ok();
    }

    /// sc-15328 — `load` must ACCEPT `spec.adapters` (they install in [`assemble`] via
    /// [`crate::adapters::apply_mage_adapters`]), while [`load_finetuned`] must keep refusing them
    /// with a message that says why.
    ///
    /// Discriminating in both directions on one fixture, so neither half can pass vacuously:
    ///
    ///   * `load` carrying an adapter must get PAST the entry guard and fail on the *next* thing it
    ///     checks — the published-checkpoint fingerprint. Restore `|| !spec.adapters.is_empty()` to
    ///     `load`'s guard and this half fails, because the error becomes the `Unsupported` one.
    ///   * `load_finetuned` carrying the same adapter must fail on the adapter refusal
    ///     SPECIFICALLY — not on the missing components it would otherwise hit. Drop that guard and
    ///     this half fails.
    ///
    /// The adapter path is deliberately nonexistent: neither call may get far enough to read it,
    /// which is what makes the first half about the *guard* rather than about adapter loading.
    #[test]
    fn load_takes_adapters_while_a_fine_tuned_checkpoint_still_refuses_them() {
        let root = std::env::temp_dir().join(format!(
            "mage-adapters-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        write_identity_only_checkpoint(
            &root
                .join("transformer")
                .join("diffusion_pytorch_model.safetensors"),
            0x5a,
        );
        let adapters = vec![mlx_gen::runtime::AdapterSpec::new(
            std::env::temp_dir().join(format!(
                "mage-adapter-never-read-{}.safetensors",
                std::process::id()
            )),
            0.8,
            mlx_gen::runtime::AdapterKind::Lora,
        )];

        let published = load_error(
            load(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(adapters.clone()),
            ),
            "the fabricated checkpoint is not the pinned Base, so this must still fail",
        );
        assert!(
            published.contains("checkpoint fingerprint mismatch"),
            "an adapter must no longer be refused at `load`'s entry guard — it should reach the \
             identity check like any other load, got: {published}"
        );

        let staged = std::env::temp_dir().join(format!(
            "mage-adapters-nonexistent-component-{}",
            std::process::id()
        ));
        let finetuned = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(staged.clone()))
                    .with_component(COMPONENT_VAE, WeightsSource::Dir(staged))
                    .with_adapters(adapters),
            ),
            "a fine-tuned checkpoint must not accept adapters",
        );
        assert!(
            finetuned.contains("cannot take LoRA/LoKr adapters"),
            "a fine-tune + adapter must be refused explicitly, and BEFORE the component staging it \
             would otherwise trip over, got: {finetuned}"
        );

        std::fs::remove_dir_all(root).ok();
    }

    /// sc-15328 — the descriptor is the engine's capability statement, and every Mage variant hosts
    /// adapters through the same `MageTransformer`. A variant that advertised neither would leave
    /// the app's `supports_adapters()` reading `false` for a model that demonstrably takes them.
    #[test]
    fn every_variant_advertises_lora_and_lokr() {
        for registration in REGISTRATIONS {
            let descriptor = (registration.descriptor)();
            assert!(
                descriptor.capabilities.supports_lora && descriptor.capabilities.supports_lokr,
                "{} must advertise supports_lora + supports_lokr: `assemble` installs both through \
                 `apply_mage_adapters` for every variant",
                descriptor.id
            );
        }
    }

    /// sc-15036 — the shared components are REQUIRED for a fine-tune and there is deliberately no
    /// flat-layout fallback: a training run emits the transformer alone, so probing
    /// `<transformer>/text_encoder` would turn a staging bug into a confusing deep load failure.
    /// Each missing id must be named. Also pins that a FILE weights source is refused (a fine-tune
    /// is a directory).
    #[test]
    fn load_finetuned_requires_both_shared_components_to_be_staged() {
        let root = std::env::temp_dir().join(format!(
            "mage-finetuned-components-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dir = |name: &str| WeightsSource::Dir(root.join(name));

        let bare = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "no components staged",
        );
        assert!(
            bare.contains(COMPONENT_TEXT_ENCODER),
            "the missing component must be named, got: {bare}"
        );

        let vae_only = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, dir("te")),
            ),
            "the VAE is still missing",
        );
        assert!(
            vae_only.contains(COMPONENT_VAE),
            "the missing VAE must be named, got: {vae_only}"
        );

        let as_file = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::File(
                    root.join("diffusion_pytorch_model.safetensors"),
                )),
            ),
            "a fine-tune is a transformer directory, not a single file",
        );
        assert!(
            as_file.contains("transformer DIRECTORY"),
            "expected the directory-shape refusal, got: {as_file}"
        );

        std::fs::remove_dir_all(root).ok();
    }

    /// sc-15036 real-weights end-to-end (epic 14034 F6): TRAIN a full base fine-tune, then RENDER
    /// with it through [`load_finetuned`], pairing the trained transformer with the base snapshot's
    /// own text encoder + VAE — the exact assembly the SceneWorks `mage_finetuned` worker lane
    /// performs.
    ///
    /// This is the claim the story exists to make true, so it is proved on real weights rather than
    /// asserted: before it, the checkpoint could not be loaded at all (the pinned-fingerprint guard
    /// rejects a retrained `add_k_proj.bias` by construction).
    ///
    /// The training step is deliberately GENTLE (4 steps at lr 1e-7, resolution 64) — this test is
    /// about the load + pairing seam, not convergence, and a gentle run is what makes the render a
    /// meaningful assertion. Measured on this checkpoint: at 10 steps / lr 1e-5 the run genuinely
    /// collapses the model onto its two-solid-swatch dataset and renders a FLAT FIELD, which would
    /// pass any "did we get pixels" check while telling you nothing about whether the trained
    /// transformer was correctly paired with the base's text encoder and VAE. At this budget the
    /// fine-tuned checkpoint still renders the base's own image, so the structure assertions below
    /// — dynamic range plus non-repeating rows, the same pair `base_real_weights.rs` uses — fail if
    /// the assembly is wrong in any way that degrades the decode.
    ///
    ///     MAGE_BASE_SNAPSHOT=<flat Mage-Flow-Base snapshot> \
    ///     MAGE_FINETUNE_RENDER_OUT=/tmp/finetuned.png \
    ///     cargo test -p mlx-gen-mage --lib finetune_then_render -- --ignored --nocapture
    #[test]
    #[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT) and an authorized Metal device"]
    fn finetune_then_render_through_load_finetuned() {
        use crate::transformer::{TRANSFORMER_CONFIG_FILE, TRANSFORMER_WEIGHTS_FILE};
        use mlx_gen::train::{TrainingConfig, TrainingItem, TrainingRequest};

        let Ok(root) = std::env::var("MAGE_BASE_SNAPSHOT") else {
            return;
        };
        let root = std::path::PathBuf::from(&root);
        let tmp = std::env::temp_dir().join(format!("mage_finetune_render_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // --- train (tiny) ---
        let mut items = Vec::new();
        for (i, colour) in [[220u8, 60, 40], [40, 90, 210]].into_iter().enumerate() {
            let path = tmp.join(format!("swatch_{i}.png"));
            let mut im = image::RgbImage::new(96, 96);
            for px in im.pixels_mut() {
                *px = image::Rgb(colour);
            }
            im.save(&path).unwrap();
            items.push(TrainingItem::captioned(
                path,
                format!("a solid colour swatch {i}"),
            ));
        }
        let out_dir = tmp.join("finetune");
        let mut trainer =
            crate::training::load_trainer(&LoadSpec::new(WeightsSource::Dir(root.clone())))
                .unwrap();
        let output = trainer
            .train(
                &TrainingRequest {
                    items,
                    config: TrainingConfig {
                        full_finetune: true,
                        steps: 4,
                        resolution: 64,
                        learning_rate: 1e-7,
                        train_dtype: "f32".into(),
                        save_every: 0,
                        sample_every: 0,
                        seed: 7,
                        ..Default::default()
                    },
                    output_dir: out_dir.clone(),
                    file_name: "finetune.safetensors".into(),
                    trigger_words: vec![],
                    cancel: mlx_gen::CancelFlag::new(),
                },
                &mut |_| {},
            )
            .expect("the full fine-tune runs");
        drop(trainer);
        println!(
            "[sc-15036] trained {} steps, final loss {:.5}; checkpoint at {}",
            output.steps,
            output.final_loss,
            out_dir.display()
        );
        // The artifact really is a transformer component dir, not an adapter file.
        assert!(out_dir.join(TRANSFORMER_CONFIG_FILE).is_file());
        assert!(out_dir.join(TRANSFORMER_WEIGHTS_FILE).is_file());

        // --- render through the fine-tuned entrypoint ---
        // `spec.weights` is the trained transformer dir; the shared components come from the
        // INSTALLED base, exactly as the worker lane stages them.
        let spec = LoadSpec::new(WeightsSource::Dir(out_dir.clone()))
            .with_component(
                COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir(root.join("text_encoder")),
            )
            .with_component(COMPONENT_VAE, WeightsSource::Dir(root.join("vae")));
        let model = match load_finetuned(MageVariant::Base, &spec) {
            Ok(model) => model,
            Err(error) => panic!("the fine-tuned checkpoint must load: {error}"),
        };

        let request = GenerationRequest {
            prompt: "a red apple on a wooden table, soft daylight".to_owned(),
            width: 512,
            height: 512,
            count: 1,
            seed: Some(11),
            steps: Some(20),
            guidance: Some(5.0),
            ..Default::default()
        };
        let out = model
            .generate(&request, &mut |_| {})
            .expect("the fine-tuned checkpoint renders");
        let GenerationOutput::Images(images) = out else {
            panic!("expected images");
        };
        let image = images.into_iter().next().expect("one image");
        assert_eq!((image.width, image.height), (512, 512));
        // Real STRUCTURE, not merely non-blank: full dynamic range and non-repeating rows. A flat
        // field (what a heavier fine-tune on this dataset legitimately produces, and also what a
        // mis-paired text encoder or a broken VAE decode produces) fails both.
        let (min, max) = image
            .pixels
            .iter()
            .fold((u8::MAX, u8::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            max.saturating_sub(min) >= 32,
            "the fine-tuned render has collapsed dynamic range: {min}..={max}"
        );
        let repeated_rows = image
            .pixels
            .chunks_exact(512 * 3)
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|rows| rows[0] == rows[1])
            .count();
        println!(
            "[sc-15036] fine-tuned render dynamic range {min}..={max}; repeated adjacent rows \
             {repeated_rows}/511"
        );
        assert!(
            repeated_rows < 51,
            "the fine-tuned render has {repeated_rows} repeated adjacent rows — the trained \
             transformer is not correctly paired with the base's text encoder / VAE"
        );

        if let Ok(png) = std::env::var("MAGE_FINETUNE_RENDER_OUT") {
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
                .expect("rgb buffer")
                .save(&png)
                .expect("png writes");
            println!("[sc-15036] wrote {png}");
        }

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn variant_table_matches_the_published_defaults() {
        // Pinned against the six model cards (epic sc-14034 ground-truth reference). Deliberately
        // asserts the *non-uniform* values — RL 20 vs Base 30 vs Turbo 4 — so a table that
        // collapsed to a single default could not pass.
        let table: Vec<(&str, bool, bool, u32, f32)> = MageVariant::ALL
            .iter()
            .map(|v| {
                (
                    v.id(),
                    v.is_edit(),
                    v.is_distilled(),
                    v.default_steps(),
                    v.default_cfg(),
                )
            })
            .collect();
        assert_eq!(
            table,
            vec![
                ("mage_flow", false, false, 20, 5.0),
                ("mage_flow_base", false, false, 30, 5.0),
                ("mage_flow_turbo", false, true, 4, 1.0),
                ("mage_flow_edit", true, false, 30, 5.0),
                ("mage_flow_edit_base", true, false, 30, 5.0),
                ("mage_flow_edit_turbo", true, true, 4, 1.0),
            ]
        );
    }

    #[test]
    fn registrations_cover_every_variant_in_order() {
        let ids: Vec<&str> = REGISTRATIONS
            .iter()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, MODEL_IDS.to_vec());
        assert_eq!(
            ids,
            MageVariant::ALL.iter().map(|v| v.id()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn edit_variants_advertise_reference_conditioning_and_gen_variants_do_not() {
        let edit = descriptor_for(MageVariant::Edit);
        assert_eq!(
            edit.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        let gen = descriptor_for(MageVariant::Rl);
        assert!(gen.capabilities.conditioning.is_empty());
        // Distillation, not task, drives the CFG surface.
        assert!(gen.capabilities.supports_guidance);
        assert!(
            !descriptor_for(MageVariant::Turbo)
                .capabilities
                .supports_guidance
        );
        assert!(
            descriptor_for(MageVariant::Edit)
                .capabilities
                .supports_guidance
        );
    }

    #[test]
    fn every_edit_variant_enters_the_full_multimodal_loader() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        for variant in [
            MageVariant::Edit,
            MageVariant::EditBase,
            MageVariant::EditTurbo,
        ] {
            let err = load(variant, &spec)
                .err()
                .expect("missing edit snapshot must fail");
            assert!(
                !matches!(err, Error::Unsupported(_)),
                "{} must enter the multimodal component loader: {err}",
                variant.id(),
            );
        }
    }

    #[test]
    fn edit_variant_defaults_and_cfg_surfaces_are_exact() {
        let base = descriptor_edit_base();
        assert_eq!(base.id, "mage_flow_edit_base");
        assert_eq!(MageVariant::EditBase.default_steps(), 30);
        assert_eq!(MageVariant::EditBase.default_cfg(), 5.0);
        assert!(base.capabilities.supports_guidance);
        assert!(base.capabilities.supports_negative_prompt);

        let turbo = descriptor_edit_turbo();
        assert_eq!(turbo.id, "mage_flow_edit_turbo");
        assert_eq!(MageVariant::EditTurbo.default_steps(), 4);
        assert_eq!(MageVariant::EditTurbo.default_cfg(), 1.0);
        assert!(!crate::pipeline::uses_cfg(
            MageVariant::EditTurbo.default_cfg()
        ));
        assert!(!turbo.capabilities.supports_guidance);
        assert!(!turbo.capabilities.supports_negative_prompt);
    }

    #[test]
    #[ignore = "needs complete MAGE_EDIT_SNAPSHOT, MAGE_EDIT_BASE_SNAPSHOT, and MAGE_EDIT_TURBO_SNAPSHOT"]
    fn complete_edit_snapshots_are_config_identical_and_checkpoint_distinct() {
        let root = |name: &str| {
            std::path::PathBuf::from(
                std::env::var(name).unwrap_or_else(|_| panic!("set {name} to a complete snapshot")),
            )
        };
        let edit = root("MAGE_EDIT_SNAPSHOT");
        let base = root("MAGE_EDIT_BASE_SNAPSHOT");
        let turbo = root("MAGE_EDIT_TURBO_SNAPSHOT");
        for relative in [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/chat_template.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/video_preprocessor_config.json",
            "text_encoder/vocab.json",
            "vae/config.json",
        ] {
            let expected = std::fs::read(edit.join(relative)).unwrap();
            assert_eq!(
                std::fs::read(base.join(relative)).unwrap(),
                expected,
                "Edit-Base {relative} must be byte-identical to Edit RL"
            );
            assert_eq!(
                std::fs::read(turbo.join(relative)).unwrap(),
                expected,
                "Edit-Turbo {relative} must be byte-identical to Edit RL"
            );
        }

        let check = |root: &Path, variant, revision, hash| {
            verify_checkpoint_identity(
                root,
                variant,
                revision,
                EDIT_IDENTITY_TENSOR,
                EDIT_IDENTITY_BYTES,
                &[3072],
                hash,
            )
        };
        check(
            &edit,
            MageVariant::Edit,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &base,
            MageVariant::EditBase,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_BASE_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &turbo,
            MageVariant::EditTurbo,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_TURBO_IDENTITY_SHA256,
        )
        .unwrap();

        for wrong in [&edit, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditBase,
                    EDIT_BASE_SNAPSHOT_REVISION,
                    EDIT_BASE_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Base must reject RL and Turbo transformer weights"
            );
        }
        for wrong in [&base, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::Edit,
                    EDIT_SNAPSHOT_REVISION,
                    EDIT_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit must reject Base and Turbo transformer weights"
            );
        }
        for wrong in [&edit, &base] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditTurbo,
                    EDIT_TURBO_SNAPSHOT_REVISION,
                    EDIT_TURBO_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Turbo must reject RL and Base transformer weights"
            );
        }
    }

    #[test]
    fn edit_reference_shapes_are_required_and_preserve_order() {
        let image = |byte| Image {
            width: 1,
            height: 1,
            pixels: vec![byte, byte, byte],
        };
        let request = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: image(1),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![image(2), image(3)],
                },
            ],
            ..Default::default()
        };
        let refs = edit_references(&request).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].as_raw(), &[1, 1, 1]);
        assert_eq!(refs[1].as_raw(), &[2, 2, 2]);
        assert_eq!(refs[2].as_raw(), &[3, 3, 3]);
        assert!(edit_references(&GenerationRequest::default()).is_err());
        let malformed = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(edit_references(&malformed).is_err());
    }

    #[test]
    fn rl_load_enters_the_real_snapshot_loader() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow".into()));
        let err = load(MageVariant::Rl, &spec)
            .err()
            .expect("missing snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "RL must not regress to the scaffold refusal: {err}"
        );
    }

    #[test]
    fn base_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_base().id, "mage_flow_base");
        assert_eq!(
            MageVariant::Base.upstream_repo(),
            "microsoft/Mage-Flow-Base"
        );
        assert_eq!((REGISTRATION_BASE.descriptor)().id, "mage_flow_base");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-base".into()));
        let err = load_base(&spec)
            .err()
            .expect("missing Base snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Base must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn base_platform_defaults_are_thirty_steps_with_real_cfg() {
        assert_eq!(MageVariant::Base.default_steps(), 30);
        assert_eq!(MageVariant::Base.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Base);
        assert!(descriptor.capabilities.supports_guidance);
        assert!(descriptor.capabilities.supports_negative_prompt);
        let request = GenerationRequest {
            prompt: "test".into(),
            negative_prompt: Some("artifact".into()),
            width: 1024,
            height: 1024,
            guidance: Some(5.0),
            ..Default::default()
        };
        validate_generation_request(&descriptor, &request).unwrap();
    }

    #[test]
    fn turbo_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_turbo().id, "mage_flow_turbo");
        assert_eq!(
            MageVariant::Turbo.upstream_repo(),
            "microsoft/Mage-Flow-Turbo"
        );
        assert_eq!((REGISTRATION_TURBO.descriptor)().id, "mage_flow_turbo");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-turbo".into()));
        let err = load_turbo(&spec)
            .err()
            .expect("missing Turbo snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Turbo must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn turbo_platform_defaults_are_four_steps_with_cfg_and_negative_prompt_off() {
        assert_eq!(MageVariant::Turbo.default_steps(), 4);
        assert_eq!(MageVariant::Turbo.default_cfg(), 1.0);
        let descriptor = descriptor_for(MageVariant::Turbo);
        assert!(!descriptor.capabilities.supports_guidance);
        assert!(!descriptor.capabilities.supports_negative_prompt);
        let plain = GenerationRequest {
            prompt: "test".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &plain).unwrap();
        let mut negative = plain.clone();
        negative.negative_prompt = Some("must not be encoded".into());
        assert!(validate_generation_request(&descriptor, &negative).is_err());
        let mut guided = plain;
        guided.guidance = Some(2.0);
        assert!(validate_generation_request(&descriptor, &guided).is_err());
    }

    #[test]
    fn rl_platform_defaults_and_exact_native_sizes_validate() {
        assert_eq!(MageVariant::Rl.default_steps(), 20);
        assert_eq!(MageVariant::Rl.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Rl);
        assert_eq!(descriptor.capabilities.max_count, MAX_COUNT);
        for &(width, height) in &[
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (512, 2048),
            (2048, 512),
            (1232, 688),
        ] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            validate_generation_request(&descriptor, &req).unwrap();
        }
        for &(width, height) in &[(496, 512), (512, 2064), (513, 512)] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            assert!(
                validate_generation_request(&descriptor, &req).is_err(),
                "{width}x{height} must be rejected"
            );
        }
        let mut batch = GenerationRequest {
            prompt: "test".into(),
            width: 2048,
            height: 2048,
            count: MAX_COUNT,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &batch).unwrap();
        batch.count += 1;
        assert!(validate_generation_request(&descriptor, &batch).is_err());
    }
}
