//! Explicit model + transform discovery: provider crates publish registration constants, family
//! crates add them to a [`ProviderRegistryBuilder`], and platform catalogs select the families they
//! ship. This is the Rust equivalent of an ordinary DI composition root with resolve-by-id.

use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
use crate::audio_transform::{AudioTransform, AudioTransformDescriptor, AudioTransformKind};
use crate::caption::{Captioner, CaptionerDescriptor};
use crate::checkpoint_codec::{CheckpointCodecRegistration, CheckpointCodecRegistry};
use crate::generator::{
    reject_unsupported_adapters, ConditioningKind, Generator, Modality, ModelDescriptor,
    StepSupport,
};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::memory_strategy::{
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport,
};
use crate::runtime::{LoadSpec, Quant, WeightsSource};
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor};
use crate::transcribe::{Transcriber, TranscriberDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor};
use crate::weightsmeta::safetensors_path_bytes;
use crate::{Error, Result};

use std::path::Path;

/// The provider-owned per-component resident-weight estimate (bytes), used by pre-load fit gates for
/// staged residency (sc-10894/sc-11924). Each component defaults to its summed on-disk
/// `.safetensors` size; a provider whose load materializes a larger representation must replace that
/// component with the conservative in-memory size:
///
/// - `text_encoder` — the phase-A prompt encoder(s) that [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
///   drops *before* the heavy render bundle loads (one or more, e.g. SDXL's two CLIPs, SD3's three);
/// - `dit` — the heavy transformer / U-Net (the "DiT"), the dominant render-phase component;
/// - `vae` — the autoencoder, co-resident with the DiT through the render.
///
/// Why a provider owns this rather than the consumer inferring it: the Sequential/staged peak is
/// `max(text_encoder, dit + vae)` (the encoder is freed before the renderer materializes — see
/// [`LoadPhase::Renderer`](crate::runtime::LoadPhase::Renderer)), not the resident sum. A consumer that
/// guesses the text-encoder size from `text_encoder*` subdir NAMING reads **zero** for any family whose
/// encoder is not under such a subdir — or has no separable encoder at all (a flat unified checkpoint) —
/// collapsing the staged peak back to the resident peak so no saving is ever selected. Each provider,
/// by contrast, computes the split from the exact subdir paths its own loader resolves.
///
/// The default constructors are tensor-free on-disk sums ([`crate::safetensors_dir_bytes`]) — **zero**
/// MLX allocation and no whole-file reads — so this remains safe in a pre-load gate. A component a
/// model does not have (or cannot separate) is `0`.
///
/// **On-disk byte SUMS, not load-exact.** Each field totals *every* `.safetensors` under the named
/// path(s), which can exceed what a single load materializes: one component dir may ship multiple
/// interchangeable variant files (anima's `diffusion_models/` holds the base/aesthetic/turbo DiTs, but
/// a run loads exactly one — so `dit` over-counts by the unused variants), or side-by-side dtype shards
/// (an SD3 `text_encoder_3/` carrying both f32 and fp16 double-counts). Today the worker consumes only
/// `text_encoder` plus the true whole-model total; `dit` / `vae` are **informational** for a future
/// consumer, which must treat them as an upper-bound on-disk footprint, not the resident size of one load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerComponentBytes {
    pub text_encoder: u64,
    pub dit: u64,
    pub vae: u64,
}

impl PerComponentBytes {
    /// Best-effort footprint for a diffusers-style snapshot: sum the `.safetensors` bytes under each
    /// named component subdir of the spec's weights DIRECTORY. Each list is the exact subdir(s) the
    /// caller's own loader resolves — `["text_encoder", "text_encoder_2"]` for the two SDXL CLIPs,
    /// `["unet"]` / `["transformer"]` for the DiT, `["vae"]` — so the paths are always correct per
    /// engine. A subdir that is absent contributes `0` ([`crate::safetensors_dir_bytes`]).
    ///
    /// Each name may be a component *subdir* OR a flat component *file* ([`safetensors_path_bytes`]),
    /// so this also covers the bernini / anima flat-file layouts. Errors only when `spec.weights` is a
    /// single [`WeightsSource::File`]: a one-file checkpoint has no component tree to split (the consumer
    /// then falls back to whole-file / resident accounting).
    pub fn from_spec_subdirs(
        spec: &LoadSpec,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.as_path(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "per-component footprint requires a snapshot directory, not a single .safetensors \
                     file"
                    .to_owned(),
            )),
        };
        Ok(Self::from_root_subdirs(root, text_encoder, dit, vae))
    }

    /// Sum each component's `.safetensors` bytes under an already-resolved `root` — for a provider whose
    /// component tree is NOT directly under `spec.weights` (e.g. anima's `split_files/` nesting resolves
    /// the root itself, then names `text_encoders` / `diffusion_models` / `vae` under it). Each name is a
    /// subdir or a flat file ([`safetensors_path_bytes`]); a missing one contributes `0`. Infallible —
    /// the root is the caller's to validate.
    pub fn from_root_subdirs(
        root: &Path,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Self {
        let sum = |names: &[&str]| -> u64 {
            names
                .iter()
                .map(|n| safetensors_path_bytes(root.join(n)))
                .sum()
        };
        Self {
            text_encoder: sum(text_encoder),
            dit: sum(dit),
            vae: sum(vae),
        }
    }
}

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model, and the optional [`footprint`](Self::footprint) size seam.
/// ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
#[derive(Clone, Copy)]
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    /// Optional per-component on-disk footprint (sc-10894) — `Some` for a provider that has declared its
    /// [`PerComponentBytes`] split (via `register_generators! { … ; footprint = … }`), `None` otherwise.
    /// `None` is the default so **every** provider that does not set it registers unchanged; a consumer
    /// reaching [`footprint`](Self::footprint) then gets `Ok(None)` and falls back to its own
    /// accounting. Mirrors the [`load`](Self::load) fn-pointer shape (a spec in, a `Result` out).
    pub footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>,
}

/// The validated on-disk shape of a caller-owned model routed through a registered generator.
///
/// The shape is explicit because family identity alone cannot prove that a File, fused checkpoint,
/// component directory, or ComfyUI tree is loadable by the same provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportedModelSource {
    TransformerFile,
    FusedCheckpoint,
    TransformerDirectory,
    ComfyUiTree,
}

/// The request surface selected for an imported source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportedModelOperation {
    Generate,
    Edit,
    Pose,
    MultiPhase,
}

/// A platform backend that may bind a portable checkpoint adapter to a real generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointBackend {
    Mlx,
    Candle,
}

impl CheckpointBackend {
    fn descriptor_label(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }
}

/// One named on-disk dialect accepted by a family checkpoint adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointDialectRegistration {
    pub id: &'static str,
    pub source: ImportedModelSource,
}

/// Header-only tensor evidence that identifies one registered dialect.
///
/// The inspector owns wildcard/cardinality evaluation; the adapter owns the stable signature id
/// and the exact tensor names that must all be present before that signature can claim a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointSignatureRegistration {
    pub id: &'static str,
    pub dialect: &'static str,
    pub required_tensor_names: &'static [&'static str],
}

/// A component role and its accepted cardinality in the portable checkpoint graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointComponentRegistration {
    pub role: &'static str,
    pub min_count: u16,
    pub max_count: u16,
}

/// Families that may satisfy one component role's resident/base dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointBaseCompatibilityRegistration {
    pub component_role: &'static str,
    pub compatible_families: &'static [&'static str],
}

/// Stable canonical key-mapping authority for one checkpoint dialect.
///
/// The mapping id names the provider-owned exhaustive mapper exercised by family migration tests;
/// callers never infer a mapper from a family allow-list.
///
/// # `plan_driven_backends`: which declarations are backed by code
///
/// A `mapping_id` is a *declaration*. Whether some crate actually ships a
/// [`LogicalKeyMapping`](crate::checkpoint_codec::LogicalKeyMapping) with that id is a separate
/// fact, and for most shipped families the answer is **no**: their loaders read the checkpoint on
/// their own native path and never compile a
/// [`LogicalWeightPlan`](crate::checkpoint_codec::LogicalWeightPlan). Declaring that difference
/// here is what lets each catalog's conformance test prove reachability in both directions —
/// a backend listed here **must** ship the impl, and a backend not listed **must not**, so neither
/// an unbacked declaration nor an undeclared plan route can appear silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCanonicalMappingRegistration {
    pub dialect: &'static str,
    pub mapping_id: &'static str,
    /// Backends that ship a real `LogicalKeyMapping` implementation whose `mapping_id()` equals
    /// [`mapping_id`](Self::mapping_id). Empty = **loader-native**: no backend routes this dialect
    /// through the plan compiler today, and the id names the correspondence only. Must be a
    /// duplicate-free subset of the adapter's
    /// [`eligible_backends`](CheckpointAdapterRegistration::eligible_backends).
    pub plan_driven_backends: &'static [CheckpointBackend],
}

/// Stable config-recovery authority for one semantic config field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointConfigRecoveryRegistration {
    pub field: &'static str,
    pub recovery_id: &'static str,
}

/// Portable capability policy for one imported operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointAdapterCapabilityRegistration {
    pub operation: ImportedModelOperation,
    /// The binding projects the real provider descriptor instead of copying a second capability
    /// table into SceneWorks.
    pub inherit_provider_capabilities: bool,
    /// Whether a backend binding may inherit the provider's LoRA/LoKr support for this operation.
    pub supports_adapter_inheritance: bool,
}

/// A real platform-provider binding for one portable family adapter route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointBackendBindingRegistration {
    pub backend: CheckpointBackend,
    pub source: ImportedModelSource,
    pub operation: ImportedModelOperation,
    pub provider_id: &'static str,
    pub required_components: Option<&'static [&'static str]>,
    pub inherit_adapters: bool,
}

/// The explicit temporary projection into the legacy imported-model family namespace.
///
/// [`CheckpointAdapterRegistration::family`] remains the provider-truth portable identity used
/// for binding and cross-catalog conformance. This separate, validated field exists only so the
/// derived [`ImportedModelRegistration`] view can remain byte-compatible while its consumers are
/// migrated away from the legacy family spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportedModelCompatibilityProjectionRegistration {
    pub family: &'static str,
}

/// The complete portable checkpoint-adapter authority for one model family.
///
/// Platform catalogs register this metadata together with only the bindings they actually ship.
/// [`ProviderRegistryBuilder::build`] validates the portable graph, refuses missing or dangling
/// implementations, and derives the temporary [`ImportedModelRegistration`] compatibility view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointAdapterRegistration {
    pub adapter_id: &'static str,
    pub family: &'static str,
    pub compatibility_projection: ImportedModelCompatibilityProjectionRegistration,
    pub signatures: &'static [CheckpointSignatureRegistration],
    pub dialects: &'static [CheckpointDialectRegistration],
    pub component_topology: &'static [CheckpointComponentRegistration],
    pub base_compatibility: &'static [CheckpointBaseCompatibilityRegistration],
    pub canonical_mappings: &'static [CheckpointCanonicalMappingRegistration],
    pub config_recovery: &'static [CheckpointConfigRecoveryRegistration],
    pub eligible_backends: &'static [CheckpointBackend],
    pub backend_bindings: &'static [CheckpointBackendBindingRegistration],
    pub operations: &'static [ImportedModelOperation],
    pub capabilities: &'static [CheckpointAdapterCapabilityRegistration],
}

impl CheckpointAdapterRegistration {
    /// Whether two platform registrations describe the same portable checkpoint contract.
    ///
    /// Backend bindings are deliberately excluded: provider ids and supported operations may be
    /// asymmetric between MLX and Candle, while every other field remains shared family authority.
    pub fn has_same_portable_metadata(&self, other: &Self) -> bool {
        self.adapter_id == other.adapter_id
            && self.family == other.family
            && self.compatibility_projection == other.compatibility_projection
            && self.signatures == other.signatures
            && self.dialects == other.dialects
            && self.component_topology == other.component_topology
            && self.base_compatibility == other.base_compatibility
            && self.canonical_mappings == other.canonical_mappings
            && self.config_recovery == other.config_recovery
            && self.eligible_backends == other.eligible_backends
            && self.operations == other.operations
            && self.capabilities == other.capabilities
    }
}

/// Portable Krea 2 adapter metadata shared by the MLX and Candle platform bindings.
pub const KREA_2_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration =
    CheckpointAdapterRegistration {
        adapter_id: "krea-2-v1",
        family: "krea_2",
        compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
            family: "krea_2",
        },
        signatures: &[
            CheckpointSignatureRegistration {
                id: "krea-2-native-v1",
                dialect: "krea-native",
                required_tensor_names: &["model.diffusion_model.blocks.0.attn.wq.weight"],
            },
            CheckpointSignatureRegistration {
                id: "krea-2-diffusers-v1",
                dialect: "diffusers",
                required_tensor_names: &["transformer_blocks.0.attn.to_q.weight"],
            },
        ],
        dialects: &[
            CheckpointDialectRegistration {
                id: "krea-native",
                source: ImportedModelSource::TransformerFile,
            },
            CheckpointDialectRegistration {
                id: "diffusers",
                source: ImportedModelSource::TransformerFile,
            },
        ],
        component_topology: &[
            CheckpointComponentRegistration {
                role: "transformer",
                min_count: 1,
                max_count: 1,
            },
            CheckpointComponentRegistration {
                role: "base-snapshot",
                min_count: 1,
                max_count: 1,
            },
        ],
        base_compatibility: &[CheckpointBaseCompatibilityRegistration {
            component_role: "base-snapshot",
            compatible_families: &["krea_2"],
        }],
        canonical_mappings: &[
            CheckpointCanonicalMappingRegistration {
                dialect: "krea-native",
                mapping_id: "krea-native-to-diffusers-v1",
                // BOTH engines: `mlx_gen_krea::KreaNativeToDiffusersMapping` and, since sc-20651,
                // `candle_gen_krea::native_mapping::KreaNativeToDiffusersMapping`. One dialect, one
                // canonical mapping id, two implementations — which is the shape this field exists
                // to make checkable rather than assumed.
                plan_driven_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
            },
            // NOT `identity-v1`. A Krea 2 checkpoint may carry *undescribed* fp8, and
            // `IdentityKeyMapping` accepts every on-disk key — including a scale companion under an
            // unrecognised suffix, which then plans as a unit-scale fp8 weight and decodes silently
            // wrong (see the `IdentityKeyMapping` doc comment). `krea-2-diffusers-v1` is identity
            // over the keys the Krea 2 architecture actually contains and REFUSES everything else.
            CheckpointCanonicalMappingRegistration {
                dialect: "diffusers",
                mapping_id: "krea-2-diffusers-v1",
                plan_driven_backends: &[CheckpointBackend::Mlx],
            },
        ],
        config_recovery: &[
            CheckpointConfigRecoveryRegistration {
                field: "architecture",
                recovery_id: "krea-signature-v1",
            },
            CheckpointConfigRecoveryRegistration {
                field: "hidden-size",
                recovery_id: "krea-tensor-shape-v1",
            },
        ],
        eligible_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
        backend_bindings: &[],
        operations: &[
            ImportedModelOperation::Generate,
            ImportedModelOperation::Edit,
            ImportedModelOperation::Pose,
            ImportedModelOperation::MultiPhase,
        ],
        capabilities: &[
            CheckpointAdapterCapabilityRegistration {
                operation: ImportedModelOperation::Generate,
                inherit_provider_capabilities: true,
                supports_adapter_inheritance: true,
            },
            CheckpointAdapterCapabilityRegistration {
                operation: ImportedModelOperation::Edit,
                inherit_provider_capabilities: true,
                supports_adapter_inheritance: true,
            },
            CheckpointAdapterCapabilityRegistration {
                operation: ImportedModelOperation::Pose,
                inherit_provider_capabilities: true,
                supports_adapter_inheritance: true,
            },
            CheckpointAdapterCapabilityRegistration {
                operation: ImportedModelOperation::MultiPhase,
                inherit_provider_capabilities: true,
                supports_adapter_inheritance: true,
            },
        ],
    };

/// Portable SDXL fused-checkpoint adapter metadata shared by both platform bindings.
pub const SDXL_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration = CheckpointAdapterRegistration {
    adapter_id: "sdxl-fused-v1",
    family: "sdxl",
    compatibility_projection: ImportedModelCompatibilityProjectionRegistration { family: "sdxl" },
    signatures: &[CheckpointSignatureRegistration {
        id: "sdxl-ldm-v1",
        dialect: "ldm",
        required_tensor_names: &["model.diffusion_model.input_blocks.0.0.weight"],
    }],
    dialects: &[CheckpointDialectRegistration {
        id: "ldm",
        source: ImportedModelSource::FusedCheckpoint,
    }],
    component_topology: &[
        CheckpointComponentRegistration {
            role: "fused-checkpoint",
            min_count: 1,
            max_count: 1,
        },
        CheckpointComponentRegistration {
            role: "text-encoders",
            min_count: 1,
            max_count: 2,
        },
    ],
    base_compatibility: &[CheckpointBaseCompatibilityRegistration {
        component_role: "text-encoders",
        compatible_families: &["sdxl"],
    }],
    canonical_mappings: &[CheckpointCanonicalMappingRegistration {
        dialect: "ldm",
        mapping_id: "sdxl-ldm-to-diffusers-v1",
        // Loader-native: the SDXL LDM loaders do their own key translation and never compile a
        // `LogicalWeightPlan`, so no crate ships a `LogicalKeyMapping` with this id.
        plan_driven_backends: &[],
    }],
    config_recovery: &[
        CheckpointConfigRecoveryRegistration {
            field: "architecture",
            recovery_id: "sdxl-signature-v1",
        },
        CheckpointConfigRecoveryRegistration {
            field: "prediction-type",
            recovery_id: "sdxl-metadata-or-default-v1",
        },
    ],
    eligible_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
    backend_bindings: &[],
    operations: &[
        ImportedModelOperation::Generate,
        ImportedModelOperation::Edit,
    ],
    capabilities: &[
        CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: true,
        },
        CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Edit,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: true,
        },
    ],
};

/// Portable Mage-Flow fine-tune directory adapter metadata.
pub const MAGE_FLOW_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration =
    CheckpointAdapterRegistration {
        adapter_id: "mage-flow-diffusers-v1",
        family: "mage_flow",
        compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
            family: "mage-flow",
        },
        signatures: &[CheckpointSignatureRegistration {
            id: "mage-flow-diffusers-v1",
            dialect: "diffusers",
            required_tensor_names: &["transformer_blocks.0.attn.to_q.weight"],
        }],
        dialects: &[CheckpointDialectRegistration {
            id: "diffusers",
            source: ImportedModelSource::TransformerDirectory,
        }],
        component_topology: &[
            CheckpointComponentRegistration {
                role: "transformer",
                min_count: 1,
                max_count: 1,
            },
            CheckpointComponentRegistration {
                role: "base-snapshot",
                min_count: 1,
                max_count: 1,
            },
        ],
        base_compatibility: &[CheckpointBaseCompatibilityRegistration {
            component_role: "base-snapshot",
            compatible_families: &["mage_flow"],
        }],
        canonical_mappings: &[CheckpointCanonicalMappingRegistration {
            dialect: "diffusers",
            mapping_id: "identity-v1",
            // Loader-native: the Mage-Flow diffusers-directory loader reads the snapshot directly.
            plan_driven_backends: &[],
        }],
        config_recovery: &[CheckpointConfigRecoveryRegistration {
            field: "architecture",
            recovery_id: "mage-config-json-v1",
        }],
        eligible_backends: &[CheckpointBackend::Mlx],
        backend_bindings: &[],
        operations: &[ImportedModelOperation::Generate],
        capabilities: &[CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: false,
        }],
    };

/// Portable Z-Image ComfyUI adapter metadata.
pub const Z_IMAGE_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration =
    CheckpointAdapterRegistration {
        adapter_id: "z-image-comfyui-v1",
        family: "z-image",
        compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
            family: "z-image",
        },
        signatures: &[CheckpointSignatureRegistration {
            id: "z-image-comfyui-v1",
            dialect: "comfyui",
            required_tensor_names: &["model.diffusion_model.x_embedder.weight"],
        }],
        dialects: &[CheckpointDialectRegistration {
            id: "comfyui",
            source: ImportedModelSource::ComfyUiTree,
        }],
        component_topology: &[
            CheckpointComponentRegistration {
                role: "transformer",
                min_count: 1,
                max_count: 1,
            },
            CheckpointComponentRegistration {
                role: "base-snapshot",
                min_count: 1,
                max_count: 1,
            },
        ],
        base_compatibility: &[CheckpointBaseCompatibilityRegistration {
            component_role: "base-snapshot",
            compatible_families: &["z-image"],
        }],
        canonical_mappings: &[CheckpointCanonicalMappingRegistration {
            dialect: "comfyui",
            mapping_id: "z-image-comfyui-to-diffusers-v1",
            // Loader-native (see `SDXL_CHECKPOINT_ADAPTER`).
            plan_driven_backends: &[],
        }],
        config_recovery: &[CheckpointConfigRecoveryRegistration {
            field: "architecture",
            recovery_id: "z-image-signature-v1",
        }],
        eligible_backends: &[CheckpointBackend::Candle],
        backend_bindings: &[],
        operations: &[ImportedModelOperation::Generate],
        capabilities: &[CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: true,
        }],
    };

/// Portable Qwen-Image ComfyUI adapter metadata.
pub const QWEN_IMAGE_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration =
    CheckpointAdapterRegistration {
        adapter_id: "qwen-image-comfyui-v1",
        family: "qwen-image",
        compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
            family: "qwen-image",
        },
        signatures: &[CheckpointSignatureRegistration {
            id: "qwen-image-comfyui-v1",
            dialect: "comfyui",
            required_tensor_names: &["model.diffusion_model.img_in.weight"],
        }],
        dialects: &[CheckpointDialectRegistration {
            id: "comfyui",
            source: ImportedModelSource::ComfyUiTree,
        }],
        component_topology: &[
            CheckpointComponentRegistration {
                role: "transformer",
                min_count: 1,
                max_count: 1,
            },
            CheckpointComponentRegistration {
                role: "base-snapshot",
                min_count: 1,
                max_count: 1,
            },
        ],
        base_compatibility: &[CheckpointBaseCompatibilityRegistration {
            component_role: "base-snapshot",
            compatible_families: &["qwen-image"],
        }],
        canonical_mappings: &[CheckpointCanonicalMappingRegistration {
            dialect: "comfyui",
            mapping_id: "qwen-image-comfyui-to-diffusers-v1",
            // Loader-native (see `SDXL_CHECKPOINT_ADAPTER`).
            plan_driven_backends: &[],
        }],
        config_recovery: &[CheckpointConfigRecoveryRegistration {
            field: "architecture",
            recovery_id: "qwen-image-signature-v1",
        }],
        eligible_backends: &[CheckpointBackend::Candle],
        backend_bindings: &[],
        operations: &[ImportedModelOperation::Generate],
        capabilities: &[CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: true,
        }],
    };

/// Portable FLUX.2 ComfyUI adapter metadata.
pub const FLUX2_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration = CheckpointAdapterRegistration {
    adapter_id: "flux2-comfyui-v1",
    family: "flux2",
    compatibility_projection: ImportedModelCompatibilityProjectionRegistration { family: "flux2" },
    signatures: &[CheckpointSignatureRegistration {
        id: "flux2-comfyui-v1",
        dialect: "comfyui",
        required_tensor_names: &["model.diffusion_model.double_blocks.0.img_attn.qkv.weight"],
    }],
    dialects: &[CheckpointDialectRegistration {
        id: "comfyui",
        source: ImportedModelSource::ComfyUiTree,
    }],
    component_topology: &[
        CheckpointComponentRegistration {
            role: "transformer",
            min_count: 1,
            max_count: 1,
        },
        CheckpointComponentRegistration {
            role: "base-snapshot",
            min_count: 1,
            max_count: 1,
        },
    ],
    base_compatibility: &[CheckpointBaseCompatibilityRegistration {
        component_role: "base-snapshot",
        compatible_families: &["flux2"],
    }],
    canonical_mappings: &[CheckpointCanonicalMappingRegistration {
        dialect: "comfyui",
        mapping_id: "flux2-comfyui-to-diffusers-v1",
        // Loader-native (see `SDXL_CHECKPOINT_ADAPTER`).
        plan_driven_backends: &[],
    }],
    config_recovery: &[CheckpointConfigRecoveryRegistration {
        field: "architecture",
        recovery_id: "flux2-signature-v1",
    }],
    eligible_backends: &[CheckpointBackend::Candle],
    backend_bindings: &[],
    operations: &[ImportedModelOperation::Generate],
    capabilities: &[CheckpointAdapterCapabilityRegistration {
        operation: ImportedModelOperation::Generate,
        inherit_provider_capabilities: true,
        supports_adapter_inheritance: true,
    }],
};

/// Portable Wan 2.2 ComfyUI adapter metadata (epic 20398, sc-20644).
///
/// # The dual-expert backbone this declares, and what it exposes
///
/// Wan 2.2's ComfyUI distribution is the one shipped family whose checkpoint has **two** backbones:
/// a high-noise and a low-noise expert, both `blocks.N.{self_attn,cross_attn,ffn}` DiTs, selected
/// per denoise step. That is why [`CheckpointAdapterRegistration::component_topology`] declares
/// `transformer-high` and `transformer-low` as two distinct single-count roles rather than one
/// `transformer` with `max_count: 2` — the two are not interchangeable, and a plan that recorded
/// them as two instances of one role could not say which is which.
///
/// # Spelling: hyphens here, underscores in the plan
///
/// SceneWorks' `checkpoint_inspector` emits matching plan-layer roles for these two experts,
/// spelled with UNDERSCORES (`transformer_high` / `transformer_low`) like every other layer role it
/// emits. The hyphenated spelling here is this file's own topology convention — the same split that
/// already exists between the `base-snapshot` topology role and the `base_snapshot` component id.
/// The projection between the two vocabularies is `-` → `_`, and it is pinned from both sides in
/// the `mapping_id` posture: the conformance test below asserts the projection of these roles, and
/// SceneWorks asserts the projected literals are what its inspector actually emits. Nothing
/// structural enforces the tie, so either side drifting alone would turn a compiled Wan plan into
/// two roles no lane recognizes.
///
/// # One binding, and why not two
///
/// [`ImportedModelOperation`] has no video vocabulary: T2V and I2V are the same `Generate`
/// operation distinguished by an `i2v` flag on the loader, not by the operation enum. So at most one
/// binding is expressible for this (backend, source). Registering I2V as `Edit` would be a lie about
/// what the enum means.
pub const WAN_CHECKPOINT_ADAPTER: CheckpointAdapterRegistration = CheckpointAdapterRegistration {
    adapter_id: "wan-comfyui-v1",
    // The PORTABLE family is the generator's own (`wan`) — the registry build refuses an adapter
    // whose family does not match the generator it binds. The PROJECTION is `wan-video`, which is
    // what `checkpoint_inspector::normalize_family` records in a compiled plan and what SceneWorks
    // keys its adapter lookup on. Wan is the second family after Mage-Flow whose two spellings
    // differ, and for the same reason.
    family: "wan",
    compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
        family: "wan-video",
    },
    signatures: &[CheckpointSignatureRegistration {
        id: "wan-comfyui-v1",
        dialect: "comfyui",
        // The pair that separates Wan from its neighbours: a `self_attn` DiT block WITH an `ffn`
        // (Anima has adaln modulation; LTX has attn1/attn2 and no ffn). Mirrors SceneWorks'
        // `base_weights::detect_transformer_family` so the inspector and the adapter agree.
        required_tensor_names: &["blocks.0.self_attn.q.weight", "blocks.0.ffn.0.weight"],
    }],
    dialects: &[CheckpointDialectRegistration {
        id: "comfyui",
        source: ImportedModelSource::ComfyUiTree,
    }],
    component_topology: &[
        CheckpointComponentRegistration {
            role: "transformer-high",
            min_count: 1,
            max_count: 1,
        },
        CheckpointComponentRegistration {
            role: "transformer-low",
            min_count: 1,
            max_count: 1,
        },
        CheckpointComponentRegistration {
            role: "base-snapshot",
            min_count: 1,
            max_count: 1,
        },
    ],
    base_compatibility: &[CheckpointBaseCompatibilityRegistration {
        component_role: "base-snapshot",
        compatible_families: &["wan"],
    }],
    canonical_mappings: &[CheckpointCanonicalMappingRegistration {
        dialect: "comfyui",
        mapping_id: "wan-comfyui-to-diffusers-v1",
        // Plan-driven on Candle: `candle_gen_wan::gguf::WanNativeToDiffusersMapping` is the
        // refusing implementation of this id, and the GGUF DiT route (sc-20649) compiles its plan
        // through it against the registered `gguf-container-v1` codec.
        plan_driven_backends: &[CheckpointBackend::Candle],
    }],
    config_recovery: &[CheckpointConfigRecoveryRegistration {
        field: "architecture",
        recovery_id: "wan-signature-v1",
    }],
    eligible_backends: &[CheckpointBackend::Candle],
    backend_bindings: &[],
    operations: &[ImportedModelOperation::Generate],
    capabilities: &[CheckpointAdapterCapabilityRegistration {
        operation: ImportedModelOperation::Generate,
        inherit_provider_capabilities: true,
        // `load_from_comfyui_experts` takes no adapters: the ComfyUI expert pair loads in place with
        // no LoRA seam, so inheriting the provider's adapter flags would advertise a capability the
        // imported route does not have.
        supports_adapter_inheritance: false,
    }],
};

/// Provider-owned route from one imported source shape and operation to the ordinary generator
/// registration that actually validates and loads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportedModelRegistration {
    /// Historical family spelling emitted by the adapter's explicit compatibility projection.
    /// Portable code must use [`CheckpointAdapterRegistration::family`] instead.
    pub family: &'static str,
    pub source: ImportedModelSource,
    pub operation: ImportedModelOperation,
    pub provider_id: &'static str,
    /// Source-shape-specific staged components. When present, this replaces the ordinary
    /// provider descriptor's component list because an imported loader can consume a different
    /// artifact shape from the published snapshot loader (for example an SDXL fused checkpoint
    /// still needs caller-staged tokenizer assets).
    pub required_components: Option<&'static [&'static str]>,
    /// Whether this source shape inherits the provider's LoRA/LoKr flags. Set false only for a
    /// structural loader refusal that is narrower than the ordinary provider (for example a full
    /// Mage fine-tune whose moved base weights cannot safely take an adapter fitted to the published
    /// checkpoint).
    pub inherit_adapters: bool,
}

/// Provider-owned encoder-contract alias for a real route assembled outside an ordinary
/// [`ModelRegistration`].
///
/// Bespoke edit/control routes often reuse a registered base generator's prompt encoder while
/// loading the denoiser through a platform-specific path.  The alias keeps that relationship in
/// the inference composition root: callers resolve the authored route id and never need to know or
/// hardcode which registered base provider owns its [`crate::EncoderContract`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderContractRouteRegistration {
    /// The externally routed provider id (for example a composed control route).
    pub route_id: &'static str,
    /// The ordinary registered generator whose descriptor is the sole contract oracle.
    pub provider_id: &'static str,
}

/// Optional, pre-load memory-strategy contract registration for one provider route id.
///
/// Kept separate from [`ModelRegistration`] so every existing provider remains source-compatible:
/// ordinary generators opt in with [`ProviderRegistryBuilder::register_memory_strategy`] as they
/// migrate. A platform composition root may instead use
/// [`ProviderRegistryBuilder::register_composed_memory_strategy`] for a real route assembled outside
/// a single gen-core [`Generator`] (for example a base generator plus a native control overlay).
#[derive(Clone, Copy)]
pub struct MemoryRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    /// The provider's real admission check, callable before weights are loaded. The load spec lets
    /// tier-sensitive providers reproduce the loaded generator's exact check without opening any
    /// weight files. This must be the same function the loaded [`Generator`] delegates to; registry
    /// conformance uses it to prove that a route-specific request is rejected at admission rather
    /// than later during generation.
    pub safety_check:
        fn(&LoadSpec, &MemoryProviderContract, &MemoryRunContext) -> MemorySafetyDecision,
}

/// Provider-owned, weights-free contract fixture paired with a [`MemoryRegistration`].
///
/// Catalog conformance uses this factory when required model assets are unavailable. Production
/// registry resolution never consults it and continues to call [`MemoryRegistration::contract`].
/// The fixture must preserve the route declaration and inject zero asset facts without filesystem
/// traversal.
#[derive(Clone, Copy)]
pub struct MemoryContractFixtureRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    /// Complete, weights-free registry-load surface for this provider.
    ///
    /// A caller-selected `LoadSpec` is not a contract inventory: it can hide a rung whose
    /// availability changes with numeric tier, residency policy, or materialization shape.  The
    /// provider therefore owns an explicit finite witness set.  Catalog dumps and conformance walk
    /// every witness; they never substitute one convenient default spec.
    pub surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
}

/// Optional selector-aware resolver for a [`MemoryContractFixtureRegistration`].
///
/// Most providers can construct a weights-free contract from the witness [`LoadSpec`] alone. A
/// prepacked provider cannot: `LoadSpec::quantize` means "pack this dense source at load time", while
/// [`MemoryContractSurfaceSelector::tier`] names the already-resolved artifact tier. Those are
/// deliberately different production shapes. Registering this additive resolver lets such a
/// provider consume the explicit selector tier without overloading `LoadSpec::quantize` or teaching
/// a downstream capability consumer a provider-specific interpretation.
#[derive(Clone, Copy)]
pub struct MemoryContractSurfaceResolverRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&MemoryContractSurfaceSpec) -> Result<MemoryProviderContract>,
}

/// Explicit, weights-free proof that a [`MemoryRegistration`] has no optimized contract surface.
///
/// Resident-only providers still participate in the reconciliation gate: they must publish this
/// typed witness rather than disappearing from the inventory by omission. Every declared selector
/// is constructed and checked to ensure all non-resident strategies remain [`MemoryStrategySupport::Missing`].
#[derive(Clone, Copy)]
pub struct ResidentOnlyMemoryContractRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    pub surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
}

/// Numeric tier named by a weights-free memory-contract surface witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryContractSurfaceTier {
    Bf16,
    Q4,
    Q8,
    Nvfp4,
}

impl MemoryContractSurfaceTier {
    fn load_spec(self) -> LoadSpec {
        let spec = LoadSpec::new(crate::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        match self {
            Self::Bf16 => spec,
            Self::Q4 => spec.with_quant(crate::Quant::Q4),
            Self::Q8 => spec.with_quant(crate::Quant::Q8),
            Self::Nvfp4 => spec.with_quant(crate::Quant::Nvfp4),
        }
    }
}

/// Exact selector axes for one shipped registry-load contract witness.
///
/// Registry routes consume provisioned snapshot directories. Single-file imports are a separate,
/// ad-hoc loader surface and are intentionally not allowed to stand in for the catalog contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryContractSurfaceSelector {
    /// Numeric tier of the already-resolved catalog artifact. This is an output fact, not a request
    /// to quantize a dense [`LoadSpec`] at load time.
    pub tier: MemoryContractSurfaceTier,
    pub offload_policy: crate::OffloadPolicy,
    pub load_shape: crate::LoadShape,
}

impl MemoryContractSurfaceSelector {
    fn matches_spec(self, spec: &LoadSpec) -> bool {
        let tier_matches = match self.tier {
            MemoryContractSurfaceTier::Bf16 => {
                spec.quantize.is_none() && spec.precision == crate::Precision::Bf16
            }
            MemoryContractSurfaceTier::Q4 => spec.quantize == Some(crate::Quant::Q4),
            MemoryContractSurfaceTier::Q8 => spec.quantize == Some(crate::Quant::Q8),
            MemoryContractSurfaceTier::Nvfp4 => spec.quantize == Some(crate::Quant::Nvfp4),
        };
        tier_matches
            && self.offload_policy == spec.offload_policy
            && self.load_shape == spec.load_shape
    }

    pub fn id(self) -> &'static str {
        match (self.tier, self.offload_policy, self.load_shape) {
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "bf16:resident:eager",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "bf16:resident:deferred",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "bf16:sequential:eager",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "bf16:sequential:deferred",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "q4:resident:eager",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "q4:resident:deferred",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "q4:sequential:eager",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "q4:sequential:deferred",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "q8:resident:eager",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "q8:resident:deferred",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "q8:sequential:eager",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "q8:sequential:deferred",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "nvfp4:resident:eager",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "nvfp4:resident:deferred",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "nvfp4:sequential:eager",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "nvfp4:sequential:deferred",
        }
    }
}

/// A provider-owned, weights-free input to one contract surface.
pub struct MemoryContractSurfaceSpec {
    pub selector: MemoryContractSurfaceSelector,
    pub spec: LoadSpec,
}

impl MemoryContractSurfaceSpec {
    /// Explicit resolved artifact tier supplied to selector-aware provider fixtures.
    pub const fn resolved_artifact_tier(&self) -> MemoryContractSurfaceTier {
        self.selector.tier
    }
}

/// One fully constructed contract surface returned by [`ProviderRegistry::memory_contract_surfaces`].
pub struct MemoryContractSurface {
    pub selector: MemoryContractSurfaceSelector,
    pub spec: LoadSpec,
    pub contract: MemoryProviderContract,
    pub composed: bool,
}

impl MemoryContractSurface {
    /// Explicit resolved artifact tier emitted to downstream capability facts.
    pub const fn resolved_artifact_tier(&self) -> MemoryContractSurfaceTier {
        self.selector.tier
    }
}

fn registry_memory_contract_surface_specs(
    tiers: &[MemoryContractSurfaceTier],
) -> Vec<MemoryContractSurfaceSpec> {
    let mut surfaces = Vec::with_capacity(tiers.len() * 4);
    for &tier in tiers {
        for offload_policy in [
            crate::OffloadPolicy::Resident,
            crate::OffloadPolicy::Sequential,
        ] {
            for load_shape in [
                crate::LoadShape::EagerMaterialization,
                crate::LoadShape::DeferredMaterialization,
            ] {
                let selector = MemoryContractSurfaceSelector {
                    tier,
                    offload_policy,
                    load_shape,
                };
                let spec = tier
                    .load_spec()
                    .with_offload_policy(offload_policy)
                    .with_load_shape(load_shape);
                surfaces.push(MemoryContractSurfaceSpec { selector, spec });
            }
        }
    }
    surfaces
}

/// Complete numeric/load-policy surface shipped by the MLX registry.
pub fn mlx_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
    ])
}

/// Common numeric/load-policy surface shipped by Candle registry providers.
pub fn candle_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
    ])
}

/// Candle surface for providers that additionally expose the explicit NVFP4 load tier.
pub fn candle_nvfp4_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
        MemoryContractSurfaceTier::Nvfp4,
    ])
}

/// Provider-owned, weights-free executable fixture for one implemented memory strategy.
///
/// The request is intentionally provider-owned: edit/control routes can supply their real mode and
/// conditioning shape instead of being forced through a synthetic text-to-image fixture.
pub struct MemoryBehaviorFixture {
    pub context: crate::MemoryRunContext,
    pub request: crate::GenerationRequest,
    /// Exact provider-owned load identity used by executable conformance. `None` reuses the
    /// catalog's common weights-free spec; providers with route/path-sensitive admission attach a
    /// synthetic but canonical receipt so their positive control exercises production validation.
    pub load_spec: Option<LoadSpec>,
}

impl MemoryBehaviorFixture {
    /// Build a fixture whose request mirrors `context.geometry`.
    ///
    /// All five geometry axes are carried across: width, height, batch (as
    /// [`count`](crate::GenerationRequest::count)), reference count (as that many
    /// [`Conditioning::Reference`](crate::Conditioning::Reference) entries) and
    /// [`frames`](crate::GenerationRequest::frames). The fixture is therefore self-consistent by
    /// construction — a provider re-grading the request against the geometry it just admitted sees
    /// the same numbers on both sides.
    ///
    /// **The `frames` mapping is `u32` → `Option<u32>` (sc-19591).**
    /// [`MemoryGeometry::frames`](crate::MemoryGeometry) is a concrete frame count, while
    /// [`GenerationRequest::frames`](crate::GenerationRequest::frames) is optional, where `None`
    /// means *unstated* and delegates to the provider's own default clip length — every consumer
    /// resolves it as `request.frames.unwrap_or(<that provider's default>)`, which is
    /// `default_frames` in `MlxRequestScopeCore::configure_request` and
    /// `CandleRequestScopeCore::configure_request`, and `1` in, for example,
    /// `candle_gen_flux2::memory_strategy`. A shared builder cannot know that per-provider default,
    /// so it *states* the geometry's count instead of delegating: `Some(frames)` re-reads as the
    /// admitted geometry for every provider, whereas `None` would only do so for one whose default
    /// happens to equal the geometry. A geometry of zero frames states no clip length at all and no
    /// provider can render zero frames, so it maps back to the unstated `None` rather than to an
    /// explicit zero-frame request.
    ///
    /// Providers whose frame lattice excludes 1 — `mlx-gen-minimax-h3` (`17n+5`, minimum 124
    /// frames) is the first — need only declare the frame count on `context.geometry`; no post-hoc
    /// override of `fixture.request.frames` is required. Guarded in `memory_strategy.rs`'s tests,
    /// alongside `behavior_fixture_preserves_exact_reference_cardinality`, by
    /// `behavior_fixture_propagates_a_multi_frame_geometry`,
    /// `behavior_fixture_propagates_a_single_frame_geometry` and
    /// `behavior_fixture_leaves_a_zero_frame_geometry_unstated`.
    pub fn new(context: crate::MemoryRunContext) -> Self {
        let mut request = crate::GenerationRequest {
            width: context.geometry.width,
            height: context.geometry.height,
            count: context.geometry.batch,
            frames: (context.geometry.frames > 0).then_some(context.geometry.frames),
            use_pid: context.use_pid,
            ..Default::default()
        };
        for _ in 0..context.geometry.reference_count {
            request.conditioning.push(crate::Conditioning::Reference {
                image: crate::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                },
                strength: Some(1.0),
            });
        }
        Self {
            context,
            request,
            load_spec: None,
        }
    }

    /// Bind this positive-control fixture to its provider-owned exact load identity.
    pub fn with_load_spec(mut self, load_spec: LoadSpec) -> Self {
        self.load_spec = Some(load_spec);
        self
    }
}

pub type MemoryBehaviorBeginRequest = fn(
    &LoadSpec,
    &MemoryProviderContract,
    &crate::MemoryRunContext,
) -> Result<Option<Box<dyn crate::MemoryRequestScope>>>;

/// Additive executable conformance seam paired by `provider_id` with a [`MemoryRegistration`].
/// Existing resident-only adopters need no behavior registration; every provider advertising an
/// optimized implemented rung must register one.
#[derive(Clone, Copy)]
pub struct MemoryBehaviorRegistration {
    pub provider_id: &'static str,
    pub valid_fixtures: fn(
        &LoadSpec,
        &MemoryProviderContract,
        crate::MemoryStrategy,
    ) -> Result<Vec<MemoryBehaviorFixture>>,
    pub begin_request: MemoryBehaviorBeginRequest,
}

/// Provider-owned, weights-free activation measurements for one registered generator route.
///
/// Kept separate from [`ModelRegistration`] and [`crate::Capabilities`] so providers opt in without a
/// breaking migration of every unrelated descriptor. The anchor is a static, family-route-wide,
/// warm 1024² transient; omission is the explicit unmeasured state.
#[derive(Clone, Copy)]
pub struct ActivationMemoryRegistration {
    pub provider_id: &'static str,
    pub anchor: crate::ActivationMemoryAnchor,
}

/// A transform provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

/// An audio-transform provider's registration (parallel to [`TransformRegistration`]; sc-12839). The
/// audio sibling of the (image) transform: a non-prompt audio→audio / audio→stems transform (voice
/// conversion, stem separation, super-resolution) resolved and loaded exactly like every other kind.
#[derive(Clone, Copy)]
pub struct AudioTransformRegistration {
    pub descriptor: fn() -> AudioTransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioTransform>>,
}

/// A trainer provider's registration (parallel to [`ModelRegistration`]) — `descriptor` for
/// introspection, `load` to construct the trainer with its (frozen) base model from a [`LoadSpec`].
#[derive(Clone, Copy)]
pub struct TrainerRegistration {
    pub descriptor: fn() -> TrainerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Trainer>>,
}

/// A captioner provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct CaptionerRegistration {
    pub descriptor: fn() -> CaptionerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Captioner>>,
}

/// A transcriber provider's registration (parallel to [`CaptionerRegistration`]; sc-12850). The
/// audio-to-text sibling of the captioner: an ASR provider resolved and loaded exactly like every
/// other kind.
#[derive(Clone, Copy)]
pub struct TranscriberRegistration {
    pub descriptor: fn() -> TranscriberDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transcriber>>,
}

/// An image-embedder provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct ImageEmbedderRegistration {
    pub descriptor: fn() -> ImageEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn ImageEmbedder>>,
}

/// A text-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]). Used by the
/// worker's `dataset_analysis` job for caption/image alignment in CLIP's joint space.
#[derive(Clone, Copy)]
pub struct TextEmbedderRegistration {
    pub descriptor: fn() -> TextEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextEmbedder>>,
}

/// A voice-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12838).
/// The audio-identity sibling of the (unregistered) face embedder: a cloned-voice speaker vector
/// that conditions TTS via [`Conditioning::VoiceEmbedding`](crate::generator::Conditioning::VoiceEmbedding).
#[derive(Clone, Copy)]
pub struct VoiceEmbedderRegistration {
    pub descriptor: fn() -> VoiceEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn VoiceEmbedder>>,
}

/// An audio-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12851).
/// The semantic audio-text (CLAP-style) sibling of the image embedder: a whole-clip vector in a
/// joint audio-text space, for retrieval / search / auto-tagging.
#[derive(Clone, Copy)]
pub struct AudioEmbedderRegistration {
    pub descriptor: fn() -> AudioEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioEmbedder>>,
}

/// Builder for an ordinary, explicit generative-media provider registry.
///
/// Platform bundles add exactly the registrations they ship.
#[derive(Default)]
pub struct ProviderRegistryBuilder {
    generators: Vec<ModelRegistration>,
    checkpoint_adapters: Vec<CheckpointAdapterRegistration>,
    checkpoint_codecs: Vec<CheckpointCodecRegistration>,
    encoder_contract_routes: Vec<EncoderContractRouteRegistration>,
    memory_strategy: Vec<MemoryRegistration>,
    memory_contract_fixture: Vec<MemoryContractFixtureRegistration>,
    memory_contract_surface_resolver: Vec<MemoryContractSurfaceResolverRegistration>,
    resident_only_memory_contract: Vec<ResidentOnlyMemoryContractRegistration>,
    memory_behavior: Vec<MemoryBehaviorRegistration>,
    activation_memory: Vec<ActivationMemoryRegistration>,
    composed_memory_strategy_ids: Vec<&'static str>,
    transforms: Vec<TransformRegistration>,
    audio_transforms: Vec<AudioTransformRegistration>,
    trainers: Vec<TrainerRegistration>,
    captioners: Vec<CaptionerRegistration>,
    transcribers: Vec<TranscriberRegistration>,
    image_embedders: Vec<ImageEmbedderRegistration>,
    text_embedders: Vec<TextEmbedderRegistration>,
    voice_embedders: Vec<VoiceEmbedderRegistration>,
    audio_embedders: Vec<AudioEmbedderRegistration>,
    rejected_quants: Vec<(Quant, &'static str)>,
}

macro_rules! builder_registration_method {
    ($name:ident, $field:ident, $registration:ty) => {
        pub fn $name(mut self, registration: $registration) -> Self {
            self.$field.push(registration);
            self
        }
    };
}

fn validate_checkpoint_adapters(
    adapters: &[CheckpointAdapterRegistration],
    generators: &[ModelRegistration],
) -> Result<Vec<ImportedModelRegistration>> {
    fn require_surface(
        adapter: &CheckpointAdapterRegistration,
        name: &str,
        present: bool,
    ) -> Result<()> {
        if present {
            Ok(())
        } else {
            Err(Error::Msg(format!(
                "checkpoint-adapter '{}' {name} must not be empty",
                adapter.adapter_id
            )))
        }
    }

    let mut adapter_ids = std::collections::BTreeSet::new();
    let mut families = std::collections::BTreeMap::new();
    let mut compatibility_families = std::collections::BTreeMap::new();
    let mut projected_routes = std::collections::BTreeSet::new();
    let mut imported_models = Vec::new();

    for adapter in adapters {
        if !is_registry_ident(adapter.adapter_id) || !is_registry_ident(adapter.family) {
            return Err(Error::Msg(format!(
                "checkpoint-adapter identity {:?}/{:?} must use non-empty lowercase registry identifiers",
                adapter.adapter_id, adapter.family
            )));
        }
        if !is_registry_ident(adapter.compatibility_projection.family) {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' compatibility family {:?} must use a non-empty lowercase registry identifier",
                adapter.adapter_id, adapter.compatibility_projection.family
            )));
        }
        if !adapter_ids.insert(adapter.adapter_id) {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter id '{}'",
                adapter.adapter_id
            )));
        }
        if families
            .insert(adapter.family, adapter.adapter_id)
            .is_some()
        {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter family '{}'",
                adapter.family
            )));
        }
        if compatibility_families
            .insert(adapter.compatibility_projection.family, adapter.adapter_id)
            .is_some()
        {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter compatibility family '{}'",
                adapter.compatibility_projection.family
            )));
        }
    }
    for (compatibility_family, adapter_id) in &compatibility_families {
        if let Some(portable_owner) = families.get(compatibility_family) {
            if portable_owner != adapter_id {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter compatibility family '{}' for '{}' collides with portable family owned by '{}'",
                    compatibility_family, adapter_id, portable_owner
                )));
            }
        }
    }

    for adapter in adapters {
        require_surface(adapter, "signatures", !adapter.signatures.is_empty())?;
        require_surface(adapter, "dialects", !adapter.dialects.is_empty())?;
        require_surface(
            adapter,
            "component topology",
            !adapter.component_topology.is_empty(),
        )?;
        require_surface(
            adapter,
            "base compatibility",
            !adapter.base_compatibility.is_empty(),
        )?;
        require_surface(
            adapter,
            "canonical mappings",
            !adapter.canonical_mappings.is_empty(),
        )?;
        require_surface(
            adapter,
            "config recovery",
            !adapter.config_recovery.is_empty(),
        )?;
        require_surface(
            adapter,
            "eligible backends",
            !adapter.eligible_backends.is_empty(),
        )?;
        if adapter.backend_bindings.is_empty() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' has no backend bindings",
                adapter.adapter_id
            )));
        }
        require_surface(adapter, "operations", !adapter.operations.is_empty())?;
        require_surface(adapter, "capabilities", !adapter.capabilities.is_empty())?;

        let mut dialect_ids = std::collections::BTreeSet::new();
        for dialect in adapter.dialects {
            if !is_registry_ident(dialect.id) || !dialect_ids.insert(dialect.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate dialect '{}'",
                    adapter.adapter_id, dialect.id
                )));
            }
        }

        let mut signature_ids = std::collections::BTreeSet::new();
        let mut signature_dialects = std::collections::BTreeSet::new();
        for signature in adapter.signatures {
            if !is_registry_ident(signature.id) || !signature_ids.insert(signature.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate signature '{}'",
                    adapter.adapter_id, signature.id
                )));
            }
            if !dialect_ids.contains(signature.dialect) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' signature '{}' targets unknown dialect '{}'",
                    adapter.adapter_id, signature.id, signature.dialect
                )));
            }
            if signature.required_tensor_names.is_empty() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' signature '{}' has no required tensor names",
                    adapter.adapter_id, signature.id
                )));
            }
            let mut tensor_names = std::collections::BTreeSet::new();
            for name in signature.required_tensor_names {
                if name.is_empty()
                    || name.chars().any(char::is_whitespace)
                    || !tensor_names.insert(*name)
                {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' signature '{}' has an empty, whitespace-containing, or duplicate tensor name",
                        adapter.adapter_id, signature.id
                    )));
                }
            }
            signature_dialects.insert(signature.dialect);
        }
        for dialect in adapter.dialects {
            if !signature_dialects.contains(dialect.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' dialect '{}' has no signature",
                    adapter.adapter_id, dialect.id
                )));
            }
        }

        let mut component_roles = std::collections::BTreeSet::new();
        for component in adapter.component_topology {
            if !is_registry_ident(component.role)
                || !component_roles.insert(component.role)
                || component.min_count == 0
                || component.max_count < component.min_count
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate component role '{}'",
                    adapter.adapter_id, component.role
                )));
            }
        }
        let mut base_roles = std::collections::BTreeSet::new();
        for base in adapter.base_compatibility {
            if !component_roles.contains(base.component_role)
                || !base_roles.insert(base.component_role)
                || base.compatible_families.is_empty()
                || base
                    .compatible_families
                    .iter()
                    .any(|family| !is_registry_ident(family))
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed, duplicate, or dangling base compatibility for role '{}'",
                    adapter.adapter_id, base.component_role
                )));
            }
            let unique: std::collections::BTreeSet<_> =
                base.compatible_families.iter().copied().collect();
            if unique.len() != base.compatible_families.len() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' repeats a compatible base family for role '{}'",
                    adapter.adapter_id, base.component_role
                )));
            }
        }

        let mut mapped_dialects = std::collections::BTreeSet::new();
        for mapping in adapter.canonical_mappings {
            if !dialect_ids.contains(mapping.dialect)
                || !is_registry_ident(mapping.mapping_id)
                || !mapped_dialects.insert(mapping.dialect)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed, duplicate, or dangling canonical mapping for dialect '{}'",
                    adapter.adapter_id, mapping.dialect
                )));
            }
        }
        if mapped_dialects.len() != dialect_ids.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' must map every declared dialect",
                adapter.adapter_id
            )));
        }

        let mut recovery_fields = std::collections::BTreeSet::new();
        for recovery in adapter.config_recovery {
            if !is_registry_ident(recovery.field)
                || !is_registry_ident(recovery.recovery_id)
                || !recovery_fields.insert(recovery.field)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate config recovery for field '{}'",
                    adapter.adapter_id, recovery.field
                )));
            }
        }

        let eligible_backends: std::collections::BTreeSet<_> =
            adapter.eligible_backends.iter().copied().collect();
        if eligible_backends.len() != adapter.eligible_backends.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' repeats an eligible backend",
                adapter.adapter_id
            )));
        }
        // A dialect cannot be plan-driven on a backend the adapter is not eligible for, and cannot
        // name the same backend twice: the catalog conformance tests read this list as the exact
        // set of backends that must ship an implementation of `mapping_id`.
        for mapping in adapter.canonical_mappings {
            let plan_driven: std::collections::BTreeSet<_> =
                mapping.plan_driven_backends.iter().copied().collect();
            if plan_driven.len() != mapping.plan_driven_backends.len()
                || !plan_driven.is_subset(&eligible_backends)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' canonical mapping for dialect '{}' declares a repeated or ineligible plan-driven backend",
                    adapter.adapter_id, mapping.dialect
                )));
            }
        }

        let operations: std::collections::BTreeSet<_> =
            adapter.operations.iter().copied().collect();
        if operations.len() != adapter.operations.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' repeats an operation",
                adapter.adapter_id
            )));
        }
        let mut capability_operations = std::collections::BTreeSet::new();
        for capability in adapter.capabilities {
            if !operations.contains(&capability.operation)
                || !capability_operations.insert(capability.operation)
                || !capability.inherit_provider_capabilities
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has a duplicate, undeclared, or non-projectable capability for {:?}",
                    adapter.adapter_id, capability.operation
                )));
            }
        }
        if capability_operations != operations {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' must own capability policy for every operation",
                adapter.adapter_id
            )));
        }

        let mut binding_keys = std::collections::BTreeSet::new();
        let mut binding_backends = std::collections::BTreeSet::new();
        for binding in adapter.backend_bindings {
            if !adapter
                .dialects
                .iter()
                .any(|dialect| dialect.source == binding.source)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} has no dialect for its source shape",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            if !eligible_backends.contains(&binding.backend)
                || !operations.contains(&binding.operation)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} is not declared portable metadata",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            let capability = adapter
                .capabilities
                .iter()
                .find(|capability| capability.operation == binding.operation)
                .expect("capability-operation equality validated above");
            if binding.inherit_adapters && !capability.supports_adapter_inheritance {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} inherits adapters contrary to capability policy",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            let binding_key = (binding.backend, binding.source, binding.operation);
            if !binding_keys.insert(binding_key) {
                return Err(Error::Msg(format!(
                    "duplicate checkpoint-adapter binding '{}' ({:?}/{:?}/{:?})",
                    adapter.adapter_id, binding.backend, binding.source, binding.operation
                )));
            }
            binding_backends.insert(binding.backend);
            let projected_key = (
                adapter.compatibility_projection.family,
                binding.source,
                binding.operation,
            );
            if !projected_routes.insert(projected_key) {
                return Err(Error::Msg(format!(
                    "duplicate imported-model route for family '{}' ({:?}/{:?})",
                    adapter.compatibility_projection.family, binding.source, binding.operation
                )));
            }
            if !is_registry_ident(binding.provider_id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding provider id is malformed",
                    adapter.adapter_id
                )));
            }
            if let Some(required) = binding.required_components {
                if required.is_empty() {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' binding {:?}/{:?} declares an empty required-components override; use None instead",
                        adapter.adapter_id, binding.source, binding.operation
                    )));
                }
                let mut component_ids = std::collections::BTreeSet::new();
                for component in required {
                    if !is_registry_ident(component) || !component_ids.insert(*component) {
                        return Err(Error::Msg(format!(
                            "checkpoint-adapter '{}' binding {:?}/{:?} has a malformed or duplicate required component",
                            adapter.adapter_id, binding.source, binding.operation
                        )));
                    }
                }
            }
            let Some(descriptor) = generators
                .iter()
                .map(|generator| (generator.descriptor)())
                .find(|descriptor| descriptor.id == binding.provider_id)
            else {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} targets unregistered generator '{}'",
                    adapter.adapter_id, binding.source, binding.operation, binding.provider_id
                )));
            };
            if descriptor.backend != binding.backend.descriptor_label() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding backend '{}' does not match generator backend '{}'",
                    adapter.adapter_id,
                    binding.backend.descriptor_label(),
                    descriptor.backend
                )));
            }
            if descriptor.family != adapter.family {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' family '{}' does not match generator '{}' family '{}'",
                    adapter.adapter_id, adapter.family, binding.provider_id, descriptor.family
                )));
            }
            imported_models.push(ImportedModelRegistration {
                family: adapter.compatibility_projection.family,
                source: binding.source,
                operation: binding.operation,
                provider_id: binding.provider_id,
                required_components: binding.required_components,
                inherit_adapters: binding.inherit_adapters,
            });
        }
        if let [sole_backend] = adapter.eligible_backends {
            for operation in adapter.operations {
                if !adapter.backend_bindings.iter().any(|binding| {
                    binding.backend == *sole_backend && binding.operation == *operation
                }) {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' operation {:?} has no binding on sole eligible backend '{}'",
                        adapter.adapter_id,
                        operation,
                        sole_backend.descriptor_label()
                    )));
                }
            }
        }
        let shipped_family_backends: std::collections::BTreeSet<_> = generators
            .iter()
            .map(|generator| (generator.descriptor)())
            .filter(|descriptor| descriptor.family == adapter.family)
            .filter_map(|descriptor| match descriptor.backend {
                "mlx" => Some(CheckpointBackend::Mlx),
                "candle" => Some(CheckpointBackend::Candle),
                _ => None,
            })
            .filter(|backend| eligible_backends.contains(backend))
            .collect();
        for backend in shipped_family_backends {
            if !binding_backends.contains(&backend) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has no binding for shipped eligible backend '{}'",
                    adapter.adapter_id,
                    backend.descriptor_label()
                )));
            }
        }
    }
    Ok(imported_models)
}

impl ProviderRegistryBuilder {
    /// Start an empty explicit registry.
    pub fn new() -> Self {
        Self::default()
    }

    builder_registration_method!(register_generator, generators, ModelRegistration);
    builder_registration_method!(
        register_checkpoint_adapter,
        checkpoint_adapters,
        CheckpointAdapterRegistration
    );
    /// Register one portable checkpoint codec row. Codecs are engine-level: a platform catalog
    /// registers its backend's baseline table exactly once, and a family adapter never carries a
    /// codec table of its own. `build` refuses duplicate ids and two codecs claiming one stored
    /// encoding, so a composed catalog that registers the same row twice fails closed instead of
    /// silently shipping duplicate rows.
    pub fn register_checkpoint_codec(mut self, registration: CheckpointCodecRegistration) -> Self {
        self.checkpoint_codecs.push(registration);
        self
    }
    builder_registration_method!(
        register_encoder_contract_route,
        encoder_contract_routes,
        EncoderContractRouteRegistration
    );
    builder_registration_method!(
        register_activation_memory,
        activation_memory,
        ActivationMemoryRegistration
    );
    builder_registration_method!(
        register_memory_strategy,
        memory_strategy,
        MemoryRegistration
    );
    builder_registration_method!(
        register_memory_contract_fixture,
        memory_contract_fixture,
        MemoryContractFixtureRegistration
    );
    builder_registration_method!(
        register_memory_contract_surface_resolver,
        memory_contract_surface_resolver,
        MemoryContractSurfaceResolverRegistration
    );
    builder_registration_method!(
        register_resident_only_memory_contract,
        resident_only_memory_contract,
        ResidentOnlyMemoryContractRegistration
    );
    builder_registration_method!(
        register_memory_behavior,
        memory_behavior,
        MemoryBehaviorRegistration
    );

    /// Register memory policy for a real platform-composed route that is not represented by a
    /// standalone gen-core [`Generator`] registration.
    ///
    /// This explicit seam preserves the ordinary registration invariant: calling
    /// [`register_memory_strategy`](Self::register_memory_strategy) for an unmatched id is still an
    /// error. Composition roots must opt in route by route, so a typo cannot silently become a
    /// provider contract with no executable owner.
    pub fn register_composed_memory_strategy(mut self, registration: MemoryRegistration) -> Self {
        self.composed_memory_strategy_ids
            .push(registration.provider_id);
        self.memory_strategy.push(registration);
        self
    }
    builder_registration_method!(register_transform, transforms, TransformRegistration);
    builder_registration_method!(
        register_audio_transform,
        audio_transforms,
        AudioTransformRegistration
    );
    builder_registration_method!(register_trainer, trainers, TrainerRegistration);
    builder_registration_method!(register_captioner, captioners, CaptionerRegistration);
    builder_registration_method!(register_transcriber, transcribers, TranscriberRegistration);
    builder_registration_method!(
        register_image_embedder,
        image_embedders,
        ImageEmbedderRegistration
    );
    builder_registration_method!(
        register_text_embedder,
        text_embedders,
        TextEmbedderRegistration
    );
    builder_registration_method!(
        register_voice_embedder,
        voice_embedders,
        VoiceEmbedderRegistration
    );
    builder_registration_method!(
        register_audio_embedder,
        audio_embedders,
        AudioEmbedderRegistration
    );

    /// Declare that this platform's backend has **no implementation** of quant tier `quant`, so every
    /// `load*` through the built registry rejects a [`LoadSpec`] requesting it with `reason`.
    ///
    /// A defense-in-depth guard for the rule that **a quant tier is a creative choice** (epic 11037
    /// SC#5): a tier a backend cannot actually serve must fail loudly at the composition boundary, and
    /// must never be quietly coerced into whatever the backend *can* do. That coercion is a live hazard
    /// wherever the tier's element width collides with a tier the backend does implement — e.g.
    /// [`Quant::Nvfp4`] reports 4 bits, so a backend that keys its
    /// quantizer off [`Quant::bits`](crate::runtime::Quant::bits) alone would silently int4-affine
    /// quantize an NVFP4 request and hand back different numerics under the tier the caller picked.
    ///
    /// This is a *platform capability* statement, not a tensor concern: the mechanism stays
    /// backend-neutral and each catalog names the tiers its own backend leaves unimplemented (the MLX
    /// catalog rejects `Nvfp4`; the CUDA candle catalog, which implements it, does not).
    pub fn reject_quant(mut self, quant: Quant, reason: &'static str) -> Self {
        self.rejected_quants.push((quant, reason));
        self
    }

    /// Validate per-kind id uniqueness and produce the immutable registry.
    pub fn build(self) -> Result<ProviderRegistry> {
        macro_rules! ensure_unique {
            ($field:ident, $kind:literal) => {{
                let mut ids = std::collections::BTreeSet::new();
                for registration in &self.$field {
                    let id = (registration.descriptor)().id;
                    if !ids.insert(id) {
                        return Err(Error::Msg(format!(
                            concat!("duplicate ", $kind, " id '{id}' in explicit registry"),
                            id = id
                        )));
                    }
                }
            }};
        }
        ensure_unique!(generators, "generator");
        {
            let mut route_ids = std::collections::BTreeSet::new();
            for registration in &self.encoder_contract_routes {
                if !is_registry_ident(registration.route_id) {
                    return Err(Error::Msg(
                        "encoder-contract route id must be a non-empty lowercase registry identifier"
                            .to_owned(),
                    ));
                }
                if !route_ids.insert(registration.route_id) {
                    return Err(Error::Msg(format!(
                        "duplicate encoder-contract route id '{}'",
                        registration.route_id
                    )));
                }
            }
            for registration in &self.encoder_contract_routes {
                if self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.route_id)
                {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' shadows a registered generator",
                        registration.route_id
                    )));
                }
                let Some(descriptor) = self
                    .generators
                    .iter()
                    .map(|generator| (generator.descriptor)())
                    .find(|descriptor| descriptor.id == registration.provider_id)
                else {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' targets unregistered generator '{}'",
                        registration.route_id, registration.provider_id
                    )));
                };
                if descriptor.encoder_contract.is_none() {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' targets generator '{}' with no encoder contract",
                        registration.route_id, registration.provider_id
                    )));
                }
            }
        }
        let imported_models =
            validate_checkpoint_adapters(&self.checkpoint_adapters, &self.generators)?;
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.activation_memory {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate activation-memory provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "activation-memory registration '{}' has no matching generator",
                        registration.provider_id
                    )));
                }
                if registration.anchor.bytes_1024 == 0 {
                    return Err(Error::Msg(format!(
                        "activation-memory registration '{}' is zero — omit unmeasured routes",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.memory_contract_fixture {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-contract fixture provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-contract fixture '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
            let mut resolver_ids = std::collections::BTreeSet::new();
            for registration in &self.memory_contract_surface_resolver {
                if !resolver_ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-contract surface resolver provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !ids.contains(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "memory-contract surface resolver '{}' has no matching contract-surface fixture",
                        registration.provider_id
                    )));
                }
            }
            let mut resident_only_ids = std::collections::BTreeSet::new();
            for registration in &self.resident_only_memory_contract {
                if !resident_only_ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate resident-only memory-contract witness provider id '{}'",
                        registration.provider_id
                    )));
                }
                if ids.contains(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has both a contract-surface fixture and a resident-only witness",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "resident-only memory-contract witness '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.memory_behavior {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-behavior provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-behavior registration '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            let mut composed_ids = std::collections::BTreeSet::new();
            for id in &self.composed_memory_strategy_ids {
                if !composed_ids.insert(*id) {
                    return Err(Error::Msg(format!(
                        "duplicate composed memory-strategy route id '{id}'"
                    )));
                }
            }
            for registration in &self.memory_strategy {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-strategy provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                    && !composed_ids.contains(registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-strategy contract '{}' has no matching generator registration",
                        registration.provider_id
                    )));
                }
            }
        }
        ensure_unique!(transforms, "transform");
        ensure_unique!(audio_transforms, "audio transform");
        ensure_unique!(trainers, "trainer");
        ensure_unique!(captioners, "captioner");
        ensure_unique!(transcribers, "transcriber");
        ensure_unique!(image_embedders, "image embedder");
        ensure_unique!(text_embedders, "text embedder");
        ensure_unique!(voice_embedders, "voice embedder");
        ensure_unique!(audio_embedders, "audio embedder");
        let checkpoint_codecs =
            CheckpointCodecRegistry::new(self.checkpoint_codecs.iter().copied())?;

        Ok(ProviderRegistry {
            generators: self.generators.into_boxed_slice(),
            checkpoint_adapters: self.checkpoint_adapters.into_boxed_slice(),
            checkpoint_codecs,
            imported_models: imported_models.into_boxed_slice(),
            encoder_contract_routes: self.encoder_contract_routes.into_boxed_slice(),
            memory_strategy: self.memory_strategy.into_boxed_slice(),
            memory_contract_fixture: self.memory_contract_fixture.into_boxed_slice(),
            memory_contract_surface_resolver: self
                .memory_contract_surface_resolver
                .into_boxed_slice(),
            resident_only_memory_contract: self.resident_only_memory_contract.into_boxed_slice(),
            memory_behavior: self.memory_behavior.into_boxed_slice(),
            activation_memory: self.activation_memory.into_boxed_slice(),
            composed_memory_strategy_ids: self.composed_memory_strategy_ids.into_boxed_slice(),
            transforms: self.transforms.into_boxed_slice(),
            audio_transforms: self.audio_transforms.into_boxed_slice(),
            trainers: self.trainers.into_boxed_slice(),
            captioners: self.captioners.into_boxed_slice(),
            transcribers: self.transcribers.into_boxed_slice(),
            image_embedders: self.image_embedders.into_boxed_slice(),
            text_embedders: self.text_embedders.into_boxed_slice(),
            voice_embedders: self.voice_embedders.into_boxed_slice(),
            audio_embedders: self.audio_embedders.into_boxed_slice(),
            rejected_quants: self.rejected_quants.into_boxed_slice(),
        })
    }
}

/// An immutable, explicit catalog of generative-media providers.
pub struct ProviderRegistry {
    generators: Box<[ModelRegistration]>,
    checkpoint_adapters: Box<[CheckpointAdapterRegistration]>,
    checkpoint_codecs: CheckpointCodecRegistry,
    imported_models: Box<[ImportedModelRegistration]>,
    encoder_contract_routes: Box<[EncoderContractRouteRegistration]>,
    memory_strategy: Box<[MemoryRegistration]>,
    memory_contract_fixture: Box<[MemoryContractFixtureRegistration]>,
    memory_contract_surface_resolver: Box<[MemoryContractSurfaceResolverRegistration]>,
    resident_only_memory_contract: Box<[ResidentOnlyMemoryContractRegistration]>,
    memory_behavior: Box<[MemoryBehaviorRegistration]>,
    activation_memory: Box<[ActivationMemoryRegistration]>,
    composed_memory_strategy_ids: Box<[&'static str]>,
    transforms: Box<[TransformRegistration]>,
    audio_transforms: Box<[AudioTransformRegistration]>,
    trainers: Box<[TrainerRegistration]>,
    captioners: Box<[CaptionerRegistration]>,
    transcribers: Box<[TranscriberRegistration]>,
    image_embedders: Box<[ImageEmbedderRegistration]>,
    text_embedders: Box<[TextEmbedderRegistration]>,
    voice_embedders: Box<[VoiceEmbedderRegistration]>,
    audio_embedders: Box<[AudioEmbedderRegistration]>,
    rejected_quants: Box<[(Quant, &'static str)]>,
}

macro_rules! explicit_registry_kind {
    (
        $iter:ident, $load:ident, $field:ident, $registration:ty,
        $kind:literal, $trait:ty
    ) => {
        pub fn $iter(&self) -> impl ExactSizeIterator<Item = &$registration> {
            self.$field.iter()
        }

        pub fn $load(&self, id: &str, spec: &LoadSpec) -> Result<Box<$trait>> {
            let registration = self
                .$iter()
                .find(|registration| (registration.descriptor)().id == id)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        concat!("no ", $kind, " registered for id '{id}'"),
                        id = id
                    ))
                })?;
            self.ensure_quant_supported(id, spec)?;
            spec.read_prepared_files_unchanged(|| (registration.load)(spec))
        }
    };
}

impl ProviderRegistry {
    /// Every validated portable family checkpoint adapter in this platform catalog.
    pub fn checkpoint_adapters(
        &self,
    ) -> impl ExactSizeIterator<Item = &CheckpointAdapterRegistration> {
        self.checkpoint_adapters.iter()
    }

    /// The validated codec table this platform catalog registered (engine-level, registered once).
    pub fn checkpoint_codecs(&self) -> &CheckpointCodecRegistry {
        &self.checkpoint_codecs
    }

    /// Check that two platform catalogs agree on the stable family/adapter-id mapping and portable
    /// checkpoint metadata, and collectively bind every backend and operation they actually ship.
    ///
    /// A family that declares only MLX eligibility is allowed to be absent from a Candle catalog
    /// (and vice versa). A family eligible for both must be present in both platform catalogs and
    /// retain at least one real binding for each backend. Per-platform provider ids and operation
    /// subsets remain intentionally asymmetric.
    pub fn checkpoint_adapter_catalog_conformance_errors(&self, other: &Self) -> Vec<String> {
        fn catalog_backends(
            registry: &ProviderRegistry,
        ) -> std::collections::BTreeSet<CheckpointBackend> {
            registry
                .generators
                .iter()
                .filter_map(|registration| match (registration.descriptor)().backend {
                    "mlx" => Some(CheckpointBackend::Mlx),
                    "candle" => Some(CheckpointBackend::Candle),
                    _ => None,
                })
                .collect()
        }

        fn ordered_pair<'a>(left: &'a str, right: &'a str) -> (&'a str, &'a str) {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        }

        let self_backends = catalog_backends(self);
        let other_backends = catalog_backends(other);
        let catalog_backends: std::collections::BTreeSet<_> =
            self_backends.union(&other_backends).copied().collect();
        let self_adapters_by_id: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.adapter_id, adapter))
            .collect();
        let other_adapters_by_id: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.adapter_id, adapter))
            .collect();
        let adapter_ids: std::collections::BTreeSet<_> = self_adapters_by_id
            .keys()
            .chain(other_adapters_by_id.keys())
            .copied()
            .collect();
        let self_adapters_by_family: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.family, adapter))
            .collect();
        let other_adapters_by_family: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.family, adapter))
            .collect();
        let families: std::collections::BTreeSet<_> = self_adapters_by_family
            .keys()
            .chain(other_adapters_by_family.keys())
            .copied()
            .collect();
        let self_adapters_by_compatibility_family: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.compatibility_projection.family, adapter))
            .collect();
        let other_adapters_by_compatibility_family: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.compatibility_projection.family, adapter))
            .collect();
        let compatibility_families: std::collections::BTreeSet<_> =
            self_adapters_by_compatibility_family
                .keys()
                .chain(other_adapters_by_compatibility_family.keys())
                .copied()
                .collect();
        let mut errors = Vec::new();

        for compatibility_family in compatibility_families {
            let left = self_adapters_by_compatibility_family
                .get(compatibility_family)
                .copied();
            let right = other_adapters_by_compatibility_family
                .get(compatibility_family)
                .copied();
            if let (Some(left), Some(right)) = (left, right) {
                if left.family != right.family && left.adapter_id != right.adapter_id {
                    let (first, second) =
                        if (left.family, left.adapter_id) <= (right.family, right.adapter_id) {
                            (left, right)
                        } else {
                            (right, left)
                        };
                    errors.push(format!(
                        "checkpoint-adapter compatibility family '{compatibility_family}' maps to different portable authorities '{}' (id '{}') and '{}' (id '{}') across catalogs",
                        first.family, first.adapter_id, second.family, second.adapter_id
                    ));
                }
            }
        }

        for adapter_id in adapter_ids {
            let left = self_adapters_by_id.get(adapter_id).copied();
            let right = other_adapters_by_id.get(adapter_id).copied();
            if let (Some(left), Some(right)) = (left, right) {
                if left.family != right.family {
                    let (first_family, second_family) = ordered_pair(left.family, right.family);
                    errors.push(format!(
                        "checkpoint-adapter id '{adapter_id}' maps to different families '{}' and '{}' across catalogs",
                        first_family, second_family
                    ));
                }
            }
        }

        for family in families {
            let left = self_adapters_by_family.get(family).copied();
            let right = other_adapters_by_family.get(family).copied();
            let authority = left.or(right).expect("family came from one catalog");

            match (left, right) {
                (Some(left), Some(right)) if left.adapter_id != right.adapter_id => {
                    let (first_id, second_id) = ordered_pair(left.adapter_id, right.adapter_id);
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' maps to different adapter ids '{}' and '{}' across catalogs",
                        first_id, second_id
                    ));
                    continue;
                }
                (Some(left), Some(right)) if !left.has_same_portable_metadata(right) => {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') portable metadata differs across catalogs",
                        left.adapter_id
                    ));
                    continue;
                }
                (Some(left), None)
                    if other_adapters_by_id
                        .get(left.adapter_id)
                        .is_some_and(|right| right.family != family) =>
                {
                    continue;
                }
                (None, Some(right))
                    if self_adapters_by_id
                        .get(right.adapter_id)
                        .is_some_and(|left| left.family != family) =>
                {
                    continue;
                }
                (Some(_), None)
                    if authority
                        .eligible_backends
                        .iter()
                        .any(|backend| other_backends.contains(backend)) =>
                {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') is missing from a catalog that ships an eligible backend",
                        authority.adapter_id
                    ));
                }
                (None, Some(_))
                    if authority
                        .eligible_backends
                        .iter()
                        .any(|backend| self_backends.contains(backend)) =>
                {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') is missing from a catalog that ships an eligible backend",
                        authority.adapter_id
                    ));
                }
                _ => {}
            }

            for backend in authority
                .eligible_backends
                .iter()
                .copied()
                .filter(|backend| catalog_backends.contains(backend))
            {
                let bound = left
                    .into_iter()
                    .chain(right)
                    .flat_map(|adapter| adapter.backend_bindings)
                    .any(|binding| binding.backend == backend);
                if !bound {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') has no binding for shipped eligible backend '{}'",
                        authority.adapter_id,
                        backend.descriptor_label()
                    ));
                }
            }

            for operation in authority.operations {
                let bound = left
                    .into_iter()
                    .chain(right)
                    .flat_map(|adapter| adapter.backend_bindings)
                    .any(|binding| binding.operation == *operation);
                if !bound {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') operation {:?} has no binding across eligible catalogs",
                        authority.adapter_id, operation
                    ));
                }
            }
        }

        errors
    }

    /// Resolve the encoder contract for an ordinary generator or an explicitly registered bespoke
    /// route. Unknown routes, and ordinary generators that do not support substitution, return
    /// `None`; aliases are validated at registry construction and cannot target another alias or a
    /// provider without a contract.
    pub fn provider_encoder_contract(&self, id: &str) -> Option<crate::EncoderContract> {
        if let Some(descriptor) = self
            .generators
            .iter()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == id)
        {
            return descriptor.encoder_contract;
        }
        let provider_id = self
            .encoder_contract_routes
            .iter()
            .find(|registration| registration.route_id == id)?
            .provider_id;
        self.generators
            .iter()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == provider_id)?
            .encoder_contract
    }

    /// Every provider-owned bespoke route participating in encoder-contract lookup.
    pub fn encoder_contract_routes(
        &self,
    ) -> impl ExactSizeIterator<Item = &EncoderContractRouteRegistration> {
        self.encoder_contract_routes.iter()
    }

    /// Compatibility projection of every adapter-owned imported route in this platform catalog.
    ///
    /// New code consumes [`Self::checkpoint_adapters`]. Existing SceneWorks consumers can continue
    /// using this view until their family migration moves them to persisted import plans.
    pub fn imported_models(&self) -> impl ExactSizeIterator<Item = &ImportedModelRegistration> {
        self.imported_models.iter()
    }

    /// Resolve an exact imported source shape and operation to the descriptor of the generator that
    /// will actually load it. Missing is an explicit unsupported answer.
    pub fn imported_model_descriptor(
        &self,
        family: &str,
        source: ImportedModelSource,
        operation: ImportedModelOperation,
    ) -> Option<ModelDescriptor> {
        let route = self.imported_models.iter().find(|route| {
            route.family == family && route.source == source && route.operation == operation
        })?;
        let mut descriptor = self
            .generators
            .iter()
            .find(|registration| (registration.descriptor)().id == route.provider_id)
            .map(|registration| (registration.descriptor)())?;
        if !route.inherit_adapters {
            descriptor.capabilities.supports_lora = false;
            descriptor.capabilities.supports_lokr = false;
        }
        if let Some(required_components) = route.required_components {
            descriptor.required_components = required_components;
        }
        Some(descriptor)
    }

    /// Admit — or refuse — an **adapter-bearing** load against an imported-model route (sc-21483,
    /// epic 11037 E6).
    ///
    /// [`Self::imported_model_descriptor`] already withdraws `supports_lora`/`supports_lokr` from a
    /// route whose binding declares `inherit_adapters = false`, but a withdrawn capability that
    /// nothing consults is indistinguishable from an ignored adapter: the model would load, the
    /// adapter would be dropped, and the render would silently be the un-adapted base. This is the
    /// gate that turns that into a typed [`Error::Unsupported`] refusal at admission.
    ///
    /// A route that does inherit adapters, an adapter-free spec, and an unrouted family are all
    /// `Ok(())` — this decides adapter admission only, never whether the route exists.
    pub fn ensure_imported_model_adapters_allowed(
        &self,
        family: &str,
        source: ImportedModelSource,
        operation: ImportedModelOperation,
        spec: &LoadSpec,
    ) -> Result<()> {
        if spec.adapters.is_empty() {
            return Ok(());
        }
        let Some(descriptor) = self.imported_model_descriptor(family, source, operation) else {
            return Ok(());
        };
        reject_unsupported_adapters(descriptor.id, &descriptor.capabilities, spec.adapters.len())
    }

    /// Provider-owned warm 1024² activation transient for `id`.
    /// `Ok(None)` is the compatibility-safe unmeasured state, including for a known platform-composed
    /// memory route with no standalone generator registration; genuinely unknown ids remain errors.
    pub fn activation_memory_bytes_1024(&self, id: &str) -> Result<Option<u64>> {
        let has_generator = self
            .generators()
            .any(|registration| (registration.descriptor)().id == id);
        let is_composed_route = self.composed_memory_strategy_ids.contains(&id);
        if !has_generator && !is_composed_route {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        Ok(self
            .activation_memory
            .iter()
            .find(|registration| registration.provider_id == id)
            .map(|registration| registration.anchor.bytes_1024))
    }

    /// Every adopted memory-strategy registration in this explicit catalog.
    pub fn memory_strategy_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryRegistration> {
        self.memory_strategy.iter()
    }

    pub fn memory_contract_fixture_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryContractFixtureRegistration> {
        self.memory_contract_fixture.iter()
    }

    pub fn memory_contract_surface_resolver_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryContractSurfaceResolverRegistration> {
        self.memory_contract_surface_resolver.iter()
    }

    pub fn resident_only_memory_contract_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ResidentOnlyMemoryContractRegistration> {
        self.resident_only_memory_contract.iter()
    }

    /// Construct every provider-owned registry-load contract witness without opening weights.
    ///
    /// Coverage is fail-closed in both directions: each memory registration must have exactly one
    /// paired surface fixture or explicit resident-only witness (builder validation rejects overlap
    /// and orphans). Every declaration must publish a non-empty unique selector set and construct a
    /// contract for the paired provider id. Resident-only declarations are checked across the same
    /// axes but omitted from the returned optimized-surface inventory. This is the seam consumed by
    /// generated capability dumps; it deliberately has no caller-supplied `LoadSpec`.
    pub fn memory_contract_surfaces(&self) -> Result<Vec<MemoryContractSurface>> {
        let mut out = Vec::new();
        for registration in &self.memory_strategy {
            let fixture = self
                .memory_contract_fixture
                .iter()
                .find(|fixture| fixture.provider_id == registration.provider_id);
            let resident_only = self
                .resident_only_memory_contract
                .iter()
                .find(|witness| witness.provider_id == registration.provider_id);
            let surface_resolver = self
                .memory_contract_surface_resolver
                .iter()
                .find(|resolver| resolver.provider_id == registration.provider_id);
            let (surface_specs, contract_factory, is_resident_only) = match (fixture, resident_only) {
                (Some(fixture), None) => (fixture.surface_specs, fixture.contract, false),
                (None, Some(witness)) => (witness.surface_specs, witness.contract, true),
                (None, None) => {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has neither a weights-free contract-surface fixture nor a resident-only witness",
                        registration.provider_id
                    )))
                }
                (Some(_), Some(_)) => {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has both a contract-surface fixture and a resident-only witness",
                        registration.provider_id
                    )))
                }
            };
            let surface_specs = surface_specs();
            if surface_specs.is_empty() {
                return Err(Error::Msg(format!(
                    "memory-contract witness '{}' publishes no surface selectors",
                    registration.provider_id
                )));
            }
            let mut selectors = std::collections::BTreeSet::new();
            for surface in surface_specs {
                if !surface.selector.matches_spec(&surface.spec) {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' selector '{}' does not match its LoadSpec",
                        registration.provider_id,
                        surface.selector.id()
                    )));
                }
                if !selectors.insert(surface.selector.id()) {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' repeats surface selector '{}'",
                        registration.provider_id,
                        surface.selector.id()
                    )));
                }
                let contract = match surface_resolver {
                    Some(resolver) => (resolver.contract)(&surface),
                    None => contract_factory(&surface.spec),
                }
                .map_err(|error| {
                    Error::Msg(format!(
                        "memory-contract witness '{}' failed surface '{}': {error}",
                        registration.provider_id,
                        surface.selector.id()
                    ))
                })?;
                if contract.provider_id != registration.provider_id {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' surface '{}' returned contract for '{}'",
                        registration.provider_id,
                        surface.selector.id(),
                        contract.provider_id
                    )));
                }
                if is_resident_only {
                    let optimized = contract.strategies.iter().find(|capability| {
                        capability.strategy != MemoryStrategy::Resident
                            && capability.support != MemoryStrategySupport::Missing
                    });
                    if let Some(capability) = optimized {
                        return Err(Error::Msg(format!(
                            "resident-only memory-contract witness '{}' surface '{}' exposes {:?} as {:?}",
                            registration.provider_id,
                            surface.selector.id(),
                            capability.strategy,
                            capability.support
                        )));
                    }
                    let errors = contract.conformance_errors();
                    if !errors.is_empty() {
                        return Err(Error::Msg(format!(
                            "resident-only memory-contract witness '{}' surface '{}' is malformed: {}",
                            registration.provider_id,
                            surface.selector.id(),
                            errors.join("; ")
                        )));
                    }
                    continue;
                }
                out.push(MemoryContractSurface {
                    selector: surface.selector,
                    spec: surface.spec,
                    contract,
                    composed: self
                        .composed_memory_strategy_ids
                        .contains(&registration.provider_id),
                });
            }
        }
        Ok(out)
    }

    pub fn memory_behavior_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryBehaviorRegistration> {
        self.memory_behavior.iter()
    }

    /// Reject a [`LoadSpec`] whose requested quant tier this platform's backend does not implement,
    /// as declared by [`ProviderRegistryBuilder::reject_quant`].
    ///
    /// The single boundary every registry-routed load of every provider kind passes through, so one
    /// check covers the whole catalog — the composition root states the platform's tier support once
    /// instead of each provider re-deriving it. Runs *after* id resolution so an unknown id still
    /// reports as an unknown id.
    fn ensure_quant_supported(&self, id: &str, spec: &LoadSpec) -> Result<()> {
        let Some(quant) = spec.quantize else {
            return Ok(());
        };
        match self.rejected_quants.iter().find(|(q, _)| *q == quant) {
            Some((_, reason)) => Err(Error::Unsupported(format!(
                "quant tier {quant:?} is not implemented by this runtime's backend \
                 (requested for '{id}'): {reason}. Refusing to load rather than silently \
                 serving a different tier's numerics."
            ))),
            None => Ok(()),
        }
    }

    explicit_registry_kind!(
        generators,
        load,
        generators,
        ModelRegistration,
        "generator",
        dyn Generator
    );
    explicit_registry_kind!(
        transforms,
        load_transform,
        transforms,
        TransformRegistration,
        "transform",
        dyn Transform
    );
    explicit_registry_kind!(
        audio_transforms,
        load_audio_transform,
        audio_transforms,
        AudioTransformRegistration,
        "audio transform",
        dyn AudioTransform
    );
    explicit_registry_kind!(
        trainers,
        load_trainer,
        trainers,
        TrainerRegistration,
        "trainer",
        dyn Trainer
    );
    explicit_registry_kind!(
        captioners,
        load_captioner,
        captioners,
        CaptionerRegistration,
        "captioner",
        dyn Captioner
    );
    explicit_registry_kind!(
        transcribers,
        load_transcriber,
        transcribers,
        TranscriberRegistration,
        "transcriber",
        dyn Transcriber
    );
    explicit_registry_kind!(
        image_embedders,
        load_image_embedder,
        image_embedders,
        ImageEmbedderRegistration,
        "image embedder",
        dyn ImageEmbedder
    );
    explicit_registry_kind!(
        text_embedders,
        load_text_embedder,
        text_embedders,
        TextEmbedderRegistration,
        "text embedder",
        dyn TextEmbedder
    );
    explicit_registry_kind!(
        voice_embedders,
        load_voice_embedder,
        voice_embedders,
        VoiceEmbedderRegistration,
        "voice embedder",
        dyn VoiceEmbedder
    );
    explicit_registry_kind!(
        audio_embedders,
        load_audio_embedder,
        audio_embedders,
        AudioEmbedderRegistration,
        "audio embedder",
        dyn AudioEmbedder
    );

    /// Return the provider-owned on-disk component footprint for generator `id`, when declared.
    ///
    /// The lookup is scoped to this explicit runtime catalog. `Ok(None)` means the provider does not
    /// declare a split; unknown ids and provider accounting failures remain errors so consumers can
    /// deliberately choose whether to fail open.
    pub fn footprint(&self, id: &str, spec: &LoadSpec) -> Result<Option<PerComponentBytes>> {
        let registration = self
            .generators()
            .find(|registration| (registration.descriptor)().id == id)
            .ok_or_else(|| Error::Msg(format!("no generator registered for id '{id}'")))?;
        match registration.footprint {
            Some(footprint) => spec
                .read_prepared_files_unchanged(|| footprint(spec))
                .map(Some),
            None => Ok(None),
        }
    }

    /// Return the provider-owned memory-strategy contract for `id`, when adopted.
    ///
    /// `Ok(None)` is the compatibility-safe resident-only/unverified state. Unknown provider ids
    /// remain errors; a malformed adopted contract also fails instead of being silently trusted.
    pub fn memory_strategy_contract(
        &self,
        id: &str,
        spec: &LoadSpec,
    ) -> Result<Option<MemoryProviderContract>> {
        let has_generator = self
            .generators()
            .any(|registration| (registration.descriptor)().id == id);
        let is_composed_route = self.composed_memory_strategy_ids.contains(&id);
        if !has_generator && !is_composed_route {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        let Some(registration) = self
            .memory_strategy
            .iter()
            .find(|registration| registration.provider_id == id)
        else {
            return Ok(None);
        };
        let contract = spec.read_prepared_files_unchanged(|| (registration.contract)(spec))?;
        let errors = contract.conformance_errors();
        if !errors.is_empty() {
            return Err(Error::Msg(format!(
                "memory-strategy contract '{id}' is malformed: {}",
                errors.join("; ")
            )));
        }
        Ok(Some(contract))
    }

    /// Run the weights-free descriptor conformance sweep over this explicit catalog.
    pub fn descriptor_conformance_errors(&self) -> Vec<String> {
        descriptor_conformance_errors_for(
            &self.generators,
            &self.transforms,
            &self.audio_transforms,
            &self.trainers,
            &self.captioners,
            &self.transcribers,
            &self.image_embedders,
            &self.text_embedders,
            &self.voice_embedders,
            &self.audio_embedders,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Descriptor-level conformance sweep (sc-9098, F-009)
// ---------------------------------------------------------------------------------------------

/// An identifier-shaped registry string: non-empty lowercase `a-z0-9` with `_`/`-`/`.`/`/`
/// separators — the shape every shipped id/family/backend uses (`z_image_turbo`, `image-embed`,
/// `mlx`, and HF-repo-style captioner ids like `fancyfeast/llama-joycaption-beta-one-hf-llava`).
/// Rejects whitespace/uppercase/unicode, which would break worker payload routing and log grepping.
fn is_registry_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
        })
}

/// Push an error for every malformed identity field (shared by all descriptor kinds).
fn check_identity(errs: &mut Vec<String>, ctx: &str, fields: &[(&str, &str)]) {
    for (name, value) in fields {
        if !is_registry_ident(value) {
            errs.push(format!(
                "{ctx}: {name} {value:?} is not a valid registry identifier \
                 (non-empty lowercase [a-z0-9_.-/])"
            ));
        }
    }
}

/// Push an error for empty/whitespace/duplicate entries in a descriptor's curated name list
/// (samplers / schedulers / guidance methods).
fn check_name_list(errs: &mut Vec<String>, ctx: &str, list_name: &str, names: &[&str]) {
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() || n.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: {list_name}[{i}] {n:?} is empty or contains whitespace"
            ));
        }
        if names[..i].contains(n) {
            errs.push(format!("{ctx}: duplicate {list_name} entry {n:?}"));
        }
    }
}

/// The weights-free invariants a generator [`ModelDescriptor`] must satisfy — everything checkable
/// from `(registration.descriptor)()` alone, with no model load (sc-9098, F-009):
///
/// - `id` / `family` / `backend` are non-empty registry identifiers,
/// - `max_count ≥ 1`, and `1 ≤ min_size ≤ max_size` for the visual modalities (a `Default` 0 bound
///   rejects every request with a confusing "out of range 0..=0" — the F-084 footgun, enforced here
///   for *every* linked visual descriptor rather than only when a request happens to reach
///   `validate_request`); the size range is **skipped for `Modality::Audio`**, whose generators emit
///   a track with no width/height and leave the bounds at the unused 0 (matching the size-skipping
///   `validate_request_audio` floor, sc-12834/sc-13314),
/// - any explicit-size grid multiple advertised by [`SizeFloor`](crate::SizeFloor) is non-zero and
///   no larger than the visual descriptor's `max_size`,
/// - `samplers` / `schedulers` / `supported_guidance_methods` entries are non-empty, whitespace-free
///   and duplicate-free (name *shape* only — resolvability is per-engine: several families advertise
///   native sampler names alongside the gen-core curated set),
/// - `conditioning` is duplicate-free, and the video-frame kinds
///   ([`Keyframe`](ConditioningKind::Keyframe) / [`VideoClip`](ConditioningKind::VideoClip) /
///   [`ControlClip`](ConditioningKind::ControlClip) / [`VideoSync`](ConditioningKind::VideoSync) /
///   [`ReferenceVideo`](ConditioningKind::ReferenceVideo)) are
///   not advertised by `Image`-modality models — an `Image` model cannot consume video frames (the LTX
///   clip kinds ride `Video`/`Both`; the `VideoSync` Foley condition rides a `Modality::Audio`
///   video→audio model, sc-13436; the `ReferenceVideo` motion reference rides a video model whose
///   *references* need not share the output modality, sc-17149).
///
/// Returns one message per violation (empty = conformant). Public so a provider's own tests can
/// target a single descriptor; [`ProviderRegistry::descriptor_conformance_errors`] sweeps a catalog.
pub fn model_descriptor_errors(d: &ModelDescriptor) -> Vec<String> {
    let mut errs = Vec::new();
    let ctx = format!("generator '{}'", d.id);
    check_identity(
        &mut errs,
        &ctx,
        &[("id", d.id), ("family", d.family), ("backend", d.backend)],
    );
    if let Some(contract) = d.encoder_contract {
        if let Err(error) = contract.validate_definition() {
            errs.push(format!("{ctx}: {error}"));
        }
    }
    let caps = &d.capabilities;
    if caps.max_count == 0 {
        errs.push(format!(
            "{ctx}: max_count is 0 — every request would be rejected"
        ));
    }
    // Size bounds only mean something for the visual modalities. A `Modality::Audio` generator emits
    // a `GenerationOutput::Audio` track with no width/height, so its `min_size`/`max_size` are unused
    // and left at the natural `Default` 0 — exactly the fields `validate_request_audio` skips the
    // range check for (sc-12834). Enforcing the visual `1 <= min_size <= max_size` floor on an audio
    // descriptor would force a nominal placeholder bound purely to satisfy the sweep (the sc-13314
    // wart), so the check is skipped for `Audio` and stays strict for `Image`/`Video`/`Both`, which
    // genuinely carry a spatial size range.
    if d.modality != Modality::Audio {
        if caps.min_size == 0 || caps.max_size == 0 {
            errs.push(format!(
                "{ctx}: min_size={} max_size={} — size bounds left at the Default 0",
                caps.min_size, caps.max_size
            ));
        } else if caps.min_size > caps.max_size {
            errs.push(format!(
                "{ctx}: min_size {} > max_size {}",
                caps.min_size, caps.max_size
            ));
        }
    }
    if let Some(multiple) = caps.size_floor.explicit_size_multiple() {
        if multiple == 0 {
            errs.push(format!(
                "{ctx}: explicit-size multiple is 0 — grid validation would be undefined"
            ));
        } else if d.modality != Modality::Audio && multiple > caps.max_size {
            errs.push(format!(
                "{ctx}: explicit-size multiple {multiple} > max_size {} — no explicit size can pass",
                caps.max_size
            ));
        }
    }
    // The advertised step surface must be satisfiable (sc-19559). Each of these declares a
    // constraint that refuses EVERY step count, which the shared floor would then enforce on every
    // request — the descriptor mistake is silent otherwise, because `Unconstrained` (the common
    // case) never reaches here.
    match &caps.supported_steps {
        StepSupport::Unconstrained => {}
        StepSupport::Exact(counts) => {
            if counts.is_empty() {
                errs.push(format!(
                    "{ctx}: supported_steps is an EMPTY exact menu — no step count would be \
                     admitted; use StepSupport::Unconstrained for 'no constraint'"
                ));
            }
            if counts.contains(&0) {
                errs.push(format!(
                    "{ctx}: supported_steps advertises 0 steps, which the shared floor always \
                     refuses (an explicit 0 renders undenoised noise)"
                ));
            }
            for (i, c) in counts.iter().enumerate() {
                if counts[..i].contains(c) {
                    errs.push(format!("{ctx}: duplicate supported step count {c}"));
                }
            }
        }
        StepSupport::Range { min, max } => {
            if *min == 0 {
                errs.push(format!(
                    "{ctx}: supported_steps range starts at 0, which the shared floor always \
                     refuses (an explicit 0 renders undenoised noise)"
                ));
            }
            if min > max {
                errs.push(format!(
                    "{ctx}: supported_steps range {min}..={max} is empty — no step count would be \
                     admitted"
                ));
            }
        }
    }
    check_name_list(&mut errs, &ctx, "sampler", &caps.samplers);
    check_name_list(&mut errs, &ctx, "scheduler", &caps.schedulers);
    check_name_list(
        &mut errs,
        &ctx,
        "guidance_method",
        &caps.supported_guidance_methods,
    );
    for (i, k) in caps.conditioning.iter().enumerate() {
        if caps.conditioning[..i].contains(k) {
            errs.push(format!("{ctx}: duplicate conditioning kind {k:?}"));
        }
        let is_video_kind = matches!(
            k,
            ConditioningKind::Keyframe
                | ConditioningKind::VideoClip
                | ConditioningKind::ControlClip
                | ConditioningKind::VideoSync
                | ConditioningKind::ReferenceVideo
        );
        if is_video_kind && d.modality == Modality::Image {
            errs.push(format!(
                "{ctx}: advertises video conditioning {k:?} but modality is Image"
            ));
        }
    }
    // Multi-turn path-A consistency (sc-14150): `supports_conversation_history` is the opt-in flag,
    // but the request carrier is a `Conditioning::ConversationHistory` gated by the conditioning
    // allowlist — so a descriptor that sets the flag without also advertising the kind would set the
    // shared floor's keyed check to pass while the allowlist still rejects every conversation (the
    // "flag on, kind missing" footgun). Cross-check them here so a provider cannot half-wire path A.
    if caps.supports_conversation_history
        && !caps
            .conditioning
            .contains(&ConditioningKind::ConversationHistory)
    {
        errs.push(format!(
            "{ctx}: supports_conversation_history is set but ConditioningKind::ConversationHistory \
             is not in `conditioning` — path-A requests would be rejected by the allowlist"
        ));
    }
    if let Some(space) = d.denoiser_output_latent_space {
        let validation = space.validation();
        if validation.zero_channels {
            errs.push(format!("{ctx}: latent-space channel count is 0"));
        }
        if validation.zero_spatial_compression {
            errs.push(format!(
                "{ctx}: latent-space spatial compression is {}x{} — both factors must be non-zero",
                space.spatial_compression.height, space.spatial_compression.width
            ));
        }
        if let crate::latent::LatentPatchLayout::Packed {
            patch_height,
            patch_width,
        } = space.patch_layout
        {
            if validation.zero_packed_patch {
                errs.push(format!(
                    "{ctx}: packed latent patch is {patch_height}x{patch_width} — both factors must be non-zero"
                ));
            }
        }
        match space.normalization {
            crate::latent::LatentNormalization::Affine { .. } => {
                let (scale, shift) = space
                    .normalization
                    .affine_values()
                    .expect("matched affine normalization");
                if validation.invalid_affine {
                    errs.push(format!(
                        "{ctx}: affine latent normalization has invalid scale={scale:?} shift={shift:?}"
                    ));
                }
            }
            crate::latent::LatentNormalization::PerChannel(stats) => {
                if validation.per_channel_count_mismatch {
                    errs.push(format!(
                        "{ctx}: latent space declares {} channels but normalization {:?} hashes {}",
                        space.channels, stats.identity, stats.channels
                    ));
                }
                if validation.invalid_per_channel_metadata {
                    errs.push(format!(
                        "{ctx}: per-channel latent normalization must have a non-empty, whitespace-free identity and non-zero content hash"
                    ));
                }
            }
            crate::latent::LatentNormalization::LearnedPerChannel { .. } => {
                if validation.invalid_learned_identity {
                    errs.push(format!(
                        "{ctx}: learned latent normalization identity must be non-empty and whitespace-free"
                    ));
                }
            }
            crate::latent::LatentNormalization::Identity => {}
        }
    }
    // Required components (sc-13658): the weights-free advertisement of the named model components a
    // consumer must provision (see `ModelDescriptor::required_components`). Each declared id must be a
    // non-empty, whitespace-free registry token, and the set must be duplicate-free — a blank or
    // repeated id would be an unstageable / ambiguous `LoadSpec::components` key. `&[]` (the shipped
    // value for every image/video provider and every single-file audio model) is trivially
    // conformant.
    check_name_list(&mut errs, &ctx, "required_component", d.required_components);
    errs
}

/// Push duplicate-id errors for one registry kind.
fn check_unique_ids(errs: &mut Vec<String>, kind: &str, ids: &[&str]) {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            errs.push(format!(
                "{kind} id '{id}' is registered more than once (first-wins shadows the rest)"
            ));
        }
    }
}

/// Weights-free descriptor-level conformance sweep over one explicit provider catalog (sc-9098,
/// F-009): generators through [`model_descriptor_errors`], plus identity
/// and capability-bound checks and per-kind id uniqueness for trainers, captioners, transforms and
/// image/text/voice embedders. No `load` is ever called, so it runs by default (no weights, no Metal) —
/// each provider crate invokes it from a default test, giving every cataloged id at least
/// descriptor-level coverage; behavioral conformance (progress/cancel/seed) stays weights-gated in
/// the `gen-core-testkit` suite.
///
/// Returns one message per violation (empty = conformant).
// One `&[Registration]` slice per provider kind, so the arg count tracks the number of kinds (10 as
// of the sc-12851 audio-embedder kind) rather than any avoidable coupling — the alternative is a
// throwaway "all registrations" struct that adds no clarity.
#[allow(clippy::too_many_arguments)]
fn descriptor_conformance_errors_for(
    generator_registrations: &[ModelRegistration],
    transform_registrations: &[TransformRegistration],
    audio_transform_registrations: &[AudioTransformRegistration],
    trainer_registrations: &[TrainerRegistration],
    captioner_registrations: &[CaptionerRegistration],
    transcriber_registrations: &[TranscriberRegistration],
    image_embedder_registrations: &[ImageEmbedderRegistration],
    text_embedder_registrations: &[TextEmbedderRegistration],
    voice_embedder_registrations: &[VoiceEmbedderRegistration],
    audio_embedder_registrations: &[AudioEmbedderRegistration],
) -> Vec<String> {
    let mut errs = Vec::new();

    let gen_descs: Vec<ModelDescriptor> = generator_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &gen_descs {
        errs.extend(model_descriptor_errors(d));
    }
    let gen_ids: Vec<&str> = gen_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "generator", &gen_ids);

    let trainer_descs: Vec<TrainerDescriptor> = trainer_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &trainer_descs {
        let ctx = format!("trainer '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let trainer_ids: Vec<&str> = trainer_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "trainer", &trainer_ids);

    let cap_descs: Vec<CaptionerDescriptor> = captioner_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &cap_descs {
        let ctx = format!("captioner '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.min_image_size == 0 || c.max_image_size < c.min_image_size {
            errs.push(format!(
                "{ctx}: image-size bounds incoherent (min {} max {})",
                c.min_image_size, c.max_image_size
            ));
        }
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no caption could be produced"
            ));
        }
    }
    let cap_ids: Vec<&str> = cap_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "captioner", &cap_ids);

    // Transcribers (sc-12850): identity, capability-bound coherence (a non-zero token ceiling and a
    // positive max clip duration — the audio twin of the captioner's max_new_tokens/image-size
    // checks), and id uniqueness.
    let asr_descs: Vec<TranscriberDescriptor> = transcriber_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &asr_descs {
        let ctx = format!("transcriber '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no transcript could be produced"
            ));
        }
        if !c.max_audio_seconds.is_finite() || c.max_audio_seconds <= 0.0 {
            errs.push(format!(
                "{ctx}: max_audio_seconds is {} — no audio could be accepted",
                c.max_audio_seconds
            ));
        }
    }
    let asr_ids: Vec<&str> = asr_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transcriber", &asr_ids);

    let tf_descs: Vec<TransformDescriptor> = transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &tf_descs {
        let ctx = format!("transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let tf_ids: Vec<&str> = tf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transform", &tf_ids);

    // Audio transforms (sc-12839): identity, a kind/stem-count coherence check (a separator must
    // advertise ≥ 2 stems; the single-output kinds advertise 0), and id uniqueness.
    let atf_descs: Vec<AudioTransformDescriptor> = audio_transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &atf_descs {
        let ctx = format!("audio transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let caps = &d.capabilities;
        match caps.kind {
            AudioTransformKind::StemSeparation if caps.stem_count < 2 => errs.push(format!(
                "{ctx}: StemSeparation advertises stem_count {} (a separator must produce ≥ 2 stems)",
                caps.stem_count
            )),
            AudioTransformKind::VoiceConversion | AudioTransformKind::SuperResolution
                if caps.stem_count != 0 =>
            {
                errs.push(format!(
                    "{ctx}: {:?} advertises stem_count {} — only StemSeparation produces stems",
                    caps.kind, caps.stem_count
                ))
            }
            _ => {}
        }
    }
    let atf_ids: Vec<&str> = atf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio transform", &atf_ids);

    let ie_descs: Vec<ImageEmbedderDescriptor> = image_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    let te_descs: Vec<TextEmbedderDescriptor> = text_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    // Audio embedders (sc-12851) carry a joint audio-text `space` exactly like image/text
    // embedders, so they run through the same identity + non-zero-dim + non-empty-space check.
    let ae_descs: Vec<AudioEmbedderDescriptor> = audio_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for (ctx_kind, id, family, backend, dim, space) in ie_descs
        .iter()
        .map(|d| {
            (
                "image embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        })
        .chain(te_descs.iter().map(|d| {
            (
                "text embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
        .chain(ae_descs.iter().map(|d| {
            (
                "audio embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
    {
        let ctx = format!("{ctx_kind} '{id}'");
        check_identity(
            &mut errs,
            &ctx,
            &[("id", id), ("family", family), ("backend", backend)],
        );
        if dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
        if space.is_empty() {
            errs.push(format!("{ctx}: embedding space is empty"));
        }
    }
    let ie_ids: Vec<&str> = ie_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "image embedder", &ie_ids);
    let te_ids: Vec<&str> = te_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "text embedder", &te_ids);
    let ae_ids: Vec<&str> = ae_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio embedder", &ae_ids);

    // Voice embedders (sc-12838) carry no embedding `space` (they are the audio-identity sibling of
    // the face embedder, not a cross-encoder cosine space) — check identity, a non-zero dim, and
    // id uniqueness only.
    let ve_descs: Vec<VoiceEmbedderDescriptor> = voice_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &ve_descs {
        let ctx = format!("voice embedder '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        if d.embedding_dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
    }
    let ve_ids: Vec<&str> = ve_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "voice embedder", &ve_ids);

    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
    use crate::audio_transform::{
        AudioTarget, AudioTransform, AudioTransformCapabilities, AudioTransformDescriptor,
        AudioTransformKind, AudioTransformRequest,
    };
    use crate::caption::{
        CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
    };
    use crate::generator::{
        ActivationMemoryAnchor, Capabilities, GenerationOutput, GenerationRequest, Modality,
        ModelDescriptor, SizeFloor,
    };
    use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
    use crate::media::{AudioTrack, Image};
    use crate::runtime::{Progress, WeightsSource};
    use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
    use crate::train::{
        Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    };
    use crate::transcribe::{
        TranscribeCapabilities, TranscribeRequest, Transcriber, TranscriberDescriptor,
        TranscriptOutput,
    };
    use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
    use std::path::PathBuf;

    struct DummyGen {
        desc: ModelDescriptor,
    }

    impl Generator for DummyGen {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    /// Small-but-coherent capabilities for the dummy registrations: the descriptor sweep runs over
    /// the explicit fixture catalog, so the dummies must carry real bounds (a
    /// `Capabilities::default()` has the F-084 all-zero bounds the sweep exists to reject).
    fn dummy_caps() -> Capabilities {
        Capabilities {
            min_size: 64,
            max_size: 512,
            max_count: 1,
            ..Default::default()
        }
    }

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_GENERATOR_REGISTRATION = dummy_descriptor => dummy_load
    }

    fn dummy_candle_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_candle_test_model",
            backend: "candle",
            ..dummy_descriptor()
        }
    }

    crate::register_generators! {
        const DUMMY_CANDLE_GENERATOR_REGISTRATION = dummy_candle_descriptor => dummy_load
    }

    fn dummy_other_candle_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_other_candle_model",
            family: "other",
            backend: "candle",
            ..dummy_descriptor()
        }
    }

    crate::register_generators! {
        const DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION = dummy_other_candle_descriptor => dummy_load
    }

    fn dummy_mage_mlx_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "mage_flow_base",
            family: "mage_flow",
            backend: "mlx",
            ..dummy_descriptor()
        }
    }

    crate::register_generators! {
        const DUMMY_MAGE_MLX_GENERATOR_REGISTRATION = dummy_mage_mlx_descriptor => dummy_load
    }

    fn dummy_legacy_mage_candle_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_legacy_mage_candle_model",
            family: "mage-flow",
            backend: "candle",
            ..dummy_descriptor()
        }
    }

    crate::register_generators! {
        const DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION =
            dummy_legacy_mage_candle_descriptor => dummy_load
    }

    struct DummyDelegatedGen {
        descriptor: ModelDescriptor,
    }

    impl DummyDelegatedGen {
        fn generate_impl(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    crate::impl_generator!(DummyDelegatedGen {
        validate: |_s, _req| Ok::<(), Error>(()),
        generate: generate_impl,
    });

    fn dummy_delegated_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_delegated_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_delegated_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyDelegatedGen {
            descriptor: dummy_delegated_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_DELEGATED_GENERATOR_REGISTRATION =
            dummy_delegated_descriptor => dummy_delegated_load
    }

    // A dummy generator that DECLARES a per-component footprint (sc-10894), exercising the
    // `; footprint = …` macro arm and the [`footprint`] entry point. Its text encoder is under a
    // non-standard `mllm/` subdir (the real boogu layout) — a naming a `text_encoder*` guesser would
    // read as ZERO — so the provider-owned split is what finds it.
    fn dummy_footprint_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_footprint_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_footprint_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_footprint_descriptor(),
        }))
    }

    fn dummy_footprint(spec: &LoadSpec) -> Result<PerComponentBytes> {
        PerComponentBytes::from_spec_subdirs(spec, &["mllm"], &["transformer"], &["vae"])
    }

    crate::register_generators! {
        const DUMMY_FOOTPRINT_GENERATOR_REGISTRATION =
            dummy_footprint_descriptor => dummy_footprint_load;
        footprint = dummy_footprint
    }

    #[cfg(unix)]
    static PREPARED_CALLBACK_REBIND: std::sync::Mutex<Option<(PathBuf, PathBuf, PathBuf)>> =
        std::sync::Mutex::new(None);

    #[cfg(unix)]
    fn rebind_prepared_callback_file() -> Result<()> {
        let (selected, staged_b, staged_a) = PREPARED_CALLBACK_REBIND
            .lock()
            .expect("callback rebinding lock")
            .take()
            .expect("callback rebinding fixture");
        std::fs::rename(staged_b, &selected)?;
        std::fs::rename(staged_a, &selected)?;
        Ok(())
    }

    #[cfg(unix)]
    fn prepared_callback_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "prepared_callback_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    #[cfg(unix)]
    fn prepared_callback_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        rebind_prepared_callback_file()?;
        Ok(Box::new(DummyGen {
            desc: prepared_callback_descriptor(),
        }))
    }

    #[cfg(unix)]
    fn prepared_callback_footprint(_spec: &LoadSpec) -> Result<PerComponentBytes> {
        rebind_prepared_callback_file()?;
        Ok(PerComponentBytes::default())
    }

    #[cfg(unix)]
    fn prepared_callback_memory_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
        rebind_prepared_callback_file()?;
        Ok(MemoryProviderContract::compatibility_default(
            "prepared_callback_model",
            crate::memory_strategy::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        ))
    }

    struct DummyTrainer {
        desc: TrainerDescriptor,
    }

    impl Trainer for DummyTrainer {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.desc
        }

        fn validate(&self, _req: &TrainingRequest) -> Result<()> {
            Ok(())
        }

        fn train(
            &mut self,
            _req: &TrainingRequest,
            _on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> Result<TrainingOutput> {
            Ok(TrainingOutput {
                adapter_path: PathBuf::from("/tmp/dummy.safetensors"),
                steps: 0,
                final_loss: 0.0,
            })
        }
    }

    fn dummy_trainer_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_test_trainer",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
            // Adapter-only: no full base fine-tune path (sc-14056). The shared
            // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
            supports_full_finetune: false,
        }
    }

    fn dummy_trainer_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_trainer_descriptor(),
        }))
    }

    crate::register_trainer! {
        const DUMMY_TRAINER_REGISTRATION = dummy_trainer_descriptor => dummy_trainer_load
    }

    // Multi-provider fixtures verify that independently named constants compose into one catalog.
    fn dummy_multi_gen_a_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_multi_gen_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_b_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "dummy_multi_gen_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_a_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_a_descriptor(),
        }))
    }

    fn dummy_multi_gen_b_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_b_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_MULTI_GENERATOR_A_REGISTRATION =
            dummy_multi_gen_a_descriptor => dummy_multi_gen_a_load
    }
    crate::register_generators! {
        const DUMMY_MULTI_GENERATOR_B_REGISTRATION =
            dummy_multi_gen_b_descriptor => dummy_multi_gen_b_load
    }

    fn dummy_multi_trainer_a_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
            // Adapter-only: no full base fine-tune path (sc-14056). The shared
            // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
            supports_full_finetune: false,
        }
    }

    fn dummy_multi_trainer_b_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
            // Adapter-only: no full base fine-tune path (sc-14056). The shared
            // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
            supports_full_finetune: false,
        }
    }

    fn dummy_multi_trainer_a_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_a_descriptor(),
        }))
    }

    fn dummy_multi_trainer_b_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_b_descriptor(),
        }))
    }

    crate::register_trainer! {
        const DUMMY_MULTI_TRAINER_A_REGISTRATION =
            dummy_multi_trainer_a_descriptor => dummy_multi_trainer_a_load
    }
    crate::register_trainer! {
        const DUMMY_MULTI_TRAINER_B_REGISTRATION =
            dummy_multi_trainer_b_descriptor => dummy_multi_trainer_b_load
    }

    struct DummyCaptioner {
        desc: CaptionerDescriptor,
    }

    impl Captioner for DummyCaptioner {
        fn descriptor(&self) -> &CaptionerDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &CaptionRequest) -> Result<()> {
            Ok(())
        }
        fn caption(
            &self,
            _req: &CaptionRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<CaptionOutput> {
            Ok(CaptionOutput {
                text: "caption".to_owned(),
                generated_tokens: Some(1),
                finish_reason: None,
            })
        }
    }

    fn dummy_captioner_descriptor() -> CaptionerDescriptor {
        CaptionerDescriptor {
            id: "dummy_test_captioner",
            family: "test",
            backend: "mlx",
            capabilities: CaptionCapabilities {
                min_image_size: 1,
                max_image_size: 4096,
                max_prompt_chars: 4000,
                max_name_chars: 120,
                max_extra_options: 16,
                max_extra_option_chars: 500,
                max_trigger_words: 32,
                max_trigger_word_chars: 120,
                max_new_tokens: 1024,
                ..Default::default()
            },
        }
    }

    fn dummy_captioner_load(_spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
        Ok(Box::new(DummyCaptioner {
            desc: dummy_captioner_descriptor(),
        }))
    }

    crate::register_captioner! {
        const DUMMY_CAPTIONER_REGISTRATION = dummy_captioner_descriptor => dummy_captioner_load
    }

    struct DummyTranscriber {
        desc: TranscriberDescriptor,
    }

    impl Transcriber for DummyTranscriber {
        fn descriptor(&self) -> &TranscriberDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &TranscribeRequest) -> Result<()> {
            Ok(())
        }
        fn transcribe(
            &self,
            _req: &TranscribeRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<TranscriptOutput> {
            Ok(TranscriptOutput {
                text: "transcript".to_owned(),
                generated_tokens: Some(1),
                ..Default::default()
            })
        }
    }

    fn dummy_transcriber_descriptor() -> TranscriberDescriptor {
        TranscriberDescriptor {
            id: "dummy_test_transcriber",
            family: "test",
            backend: "candle",
            capabilities: TranscribeCapabilities {
                languages: vec!["en"],
                supports_segment_timestamps: true,
                max_audio_seconds: 30.0,
                max_new_tokens: 448,
                ..Default::default()
            },
        }
    }

    fn dummy_transcriber_load(_spec: &LoadSpec) -> Result<Box<dyn Transcriber>> {
        Ok(Box::new(DummyTranscriber {
            desc: dummy_transcriber_descriptor(),
        }))
    }

    crate::register_transcriber! {
        const DUMMY_TRANSCRIBER_REGISTRATION =
            dummy_transcriber_descriptor => dummy_transcriber_load
    }

    fn dummy_registry() -> ProviderRegistry {
        ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_DELEGATED_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_FOOTPRINT_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_A_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_B_REGISTRATION)
            .register_trainer(DUMMY_TRAINER_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_A_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_B_REGISTRATION)
            .register_captioner(DUMMY_CAPTIONER_REGISTRATION)
            .register_transcriber(DUMMY_TRANSCRIBER_REGISTRATION)
            .register_text_embedder(DUMMY_TEXT_EMBEDDER_REGISTRATION)
            .register_image_embedder(DUMMY_IMAGE_EMBEDDER_REGISTRATION)
            .register_voice_embedder(DUMMY_VOICE_EMBEDDER_REGISTRATION)
            .register_audio_embedder(DUMMY_AUDIO_EMBEDDER_REGISTRATION)
            .build()
            .unwrap()
    }

    #[test]
    fn registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry
            .load("dummy_test_model", &spec)
            .expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_test_model");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[cfg(unix)]
    #[test]
    fn prepared_file_identity_guards_load_footprint_and_memory_callbacks() {
        use std::os::unix::fs::symlink;

        fn spec_with_rebinding_callback(tag: &str) -> (tempfile::TempDir, LoadSpec) {
            let dir = tempfile::tempdir().expect("temp dir");
            let first = dir.path().join(format!("{tag}-blob-a"));
            let second = dir.path().join(format!("{tag}-blob-b"));
            let selected = dir.path().join(format!("{tag}.safetensors"));
            let staged_b = dir.path().join(format!("{tag}-staged-b.safetensors"));
            let staged_a = dir.path().join(format!("{tag}-staged-a.safetensors"));
            std::fs::write(&first, b"same-size-a").expect("write A");
            std::fs::write(&second, b"same-size-b").expect("write B");
            symlink(&first, &selected).expect("select A");
            symlink(&second, &staged_b).expect("stage B link");
            symlink(&first, &staged_a).expect("stage replacement A link");
            let mut spec = LoadSpec::new(WeightsSource::File(selected.clone()));
            spec.prepare_file_sources().expect("prepare A identity");
            *PREPARED_CALLBACK_REBIND
                .lock()
                .expect("callback rebinding lock") = Some((selected, staged_b, staged_a));
            (dir, spec)
        }

        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: prepared_callback_descriptor,
                load: prepared_callback_load,
                footprint: Some(prepared_callback_footprint),
            })
            .register_memory_strategy(MemoryRegistration {
                provider_id: "prepared_callback_model",
                contract: prepared_callback_memory_contract,
                safety_check:
                    crate::memory_strategy::default_registered_memory_strategy_safety_check,
            })
            .build()
            .expect("callback-boundary registry");

        let (_load_dir, load_spec) = spec_with_rebinding_callback("load");
        let load_error = registry
            .load("prepared_callback_model", &load_spec)
            .err()
            .expect("load callback A -> B -> recreated A must fail")
            .to_string();
        assert!(load_error.contains("entry changed"), "got: {load_error}");

        let (_footprint_dir, footprint_spec) = spec_with_rebinding_callback("footprint");
        let footprint_error = registry
            .footprint("prepared_callback_model", &footprint_spec)
            .expect_err("footprint callback A -> B -> recreated A must fail")
            .to_string();
        assert!(
            footprint_error.contains("entry changed"),
            "got: {footprint_error}"
        );

        let (_memory_dir, memory_spec) = spec_with_rebinding_callback("memory");
        let memory_error = registry
            .memory_strategy_contract("prepared_callback_model", &memory_spec)
            .expect_err("memory callback A -> B -> recreated A must fail")
            .to_string();
        assert!(
            memory_error.contains("entry changed"),
            "got: {memory_error}"
        );
    }

    #[test]
    fn explicit_registry_resolves_minimal_catalog() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .build()
            .unwrap();
        assert_eq!(registry.generators().len(), 1);
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/tmp")));
        assert_eq!(
            registry
                .load("dummy_test_model", &spec)
                .unwrap()
                .descriptor()
                .id,
            "dummy_test_model"
        );
        assert!(registry.trainers().next().is_none());
    }

    #[test]
    fn encoder_contract_route_registration_fails_closed() {
        let missing_target = ProviderRegistryBuilder::new()
            .register_encoder_contract_route(EncoderContractRouteRegistration {
                route_id: "dummy_route",
                provider_id: "missing_provider",
            })
            .build()
            .err()
            .expect("an encoder route cannot target an unregistered provider")
            .to_string();
        assert!(missing_target.contains("targets unregistered generator"));

        let target_without_contract = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_encoder_contract_route(EncoderContractRouteRegistration {
                route_id: "dummy_route",
                provider_id: "dummy_test_model",
            })
            .build()
            .err()
            .expect("an encoder route cannot target a provider without a contract")
            .to_string();
        assert!(target_without_contract.contains("with no encoder contract"));

        let duplicate = ProviderRegistryBuilder::new()
            .register_encoder_contract_route(EncoderContractRouteRegistration {
                route_id: "dummy_route",
                provider_id: "missing_a",
            })
            .register_encoder_contract_route(EncoderContractRouteRegistration {
                route_id: "dummy_route",
                provider_id: "missing_b",
            })
            .build()
            .err()
            .expect("duplicate encoder routes must fail before resolution")
            .to_string();
        assert!(duplicate.contains("duplicate encoder-contract route id"));

        let ordinary = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .build()
            .unwrap();
        assert_eq!(ordinary.provider_encoder_contract("dummy_test_model"), None);
        assert_eq!(ordinary.provider_encoder_contract("unknown_route"), None);
    }

    fn composed_memory_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
        Ok(MemoryProviderContract::compatibility_default(
            "dummy_composed_route",
            crate::memory_strategy::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        ))
    }

    const DUMMY_COMPOSED_MEMORY_REGISTRATION: MemoryRegistration = MemoryRegistration {
        provider_id: "dummy_composed_route",
        contract: composed_memory_contract,
        safety_check: crate::memory_strategy::default_registered_memory_strategy_safety_check,
    };

    fn production_contract_requires_assets(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
        Err(Error::Msg("production factory requires assets".to_owned()))
    }

    fn weights_free_fixture_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
        Ok(MemoryProviderContract::compatibility_default(
            "dummy_weights_free_route",
            crate::memory_strategy::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        ))
    }

    const DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION: MemoryRegistration = MemoryRegistration {
        provider_id: "dummy_weights_free_route",
        contract: production_contract_requires_assets,
        safety_check: crate::memory_strategy::default_registered_memory_strategy_safety_check,
    };

    const DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE: MemoryContractFixtureRegistration =
        MemoryContractFixtureRegistration {
            provider_id: "dummy_weights_free_route",
            contract: weights_free_fixture_contract,
            surface_specs: mlx_memory_contract_surface_specs,
        };

    fn selector_aware_fixture_contract(
        surface: &MemoryContractSurfaceSpec,
    ) -> Result<MemoryProviderContract> {
        let mut contract = weights_free_fixture_contract(&surface.spec)?;
        contract.load_shape = surface.spec.load_shape;
        if surface.selector.tier == MemoryContractSurfaceTier::Q4
            && surface.selector.load_shape == crate::LoadShape::DeferredMaterialization
        {
            let capability = contract
                .strategies
                .iter_mut()
                .find(|capability| {
                    capability.strategy == MemoryStrategy::BoundedTransformerResidency
                })
                .unwrap();
            capability.support = MemoryStrategySupport::Implemented;
            capability.parameters.transformer_window_sizes = vec![1];
            capability.parameters.transformer_window_components =
                vec![crate::memory_strategy::TransformerComponent::Dit];
            contract.lifecycle.transformer_window_materialization = true;
        }
        Ok(contract)
    }

    const DUMMY_SURFACE_RESOLVER: MemoryContractSurfaceResolverRegistration =
        MemoryContractSurfaceResolverRegistration {
            provider_id: "dummy_weights_free_route",
            contract: selector_aware_fixture_contract,
        };

    const DUMMY_RESIDENT_ONLY_WITNESS: ResidentOnlyMemoryContractRegistration =
        ResidentOnlyMemoryContractRegistration {
            provider_id: "dummy_weights_free_route",
            contract: weights_free_fixture_contract,
            surface_specs: mlx_memory_contract_surface_specs,
        };

    fn false_resident_only_contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
        let mut contract = weights_free_fixture_contract(spec)?;
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .unwrap()
            .support = MemoryStrategySupport::Implemented;
        Ok(contract)
    }

    fn empty_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
        Vec::new()
    }

    fn duplicate_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
        let first = mlx_memory_contract_surface_specs().remove(0);
        let duplicate = mlx_memory_contract_surface_specs().remove(0);
        vec![first, duplicate]
    }

    fn mismatched_surface_spec() -> Vec<MemoryContractSurfaceSpec> {
        let mut surface = mlx_memory_contract_surface_specs().remove(0);
        surface.selector.tier = MemoryContractSurfaceTier::Q4;
        vec![surface]
    }

    fn wrong_provider_fixture_contract(_spec: &LoadSpec) -> Result<MemoryProviderContract> {
        Ok(MemoryProviderContract::compatibility_default(
            "wrong_provider",
            crate::memory_strategy::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        ))
    }

    #[test]
    fn production_registry_resolution_never_uses_the_weights_free_factory() {
        let registry = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .build()
            .unwrap();
        let error = registry
            .memory_strategy_contract(
                "dummy_weights_free_route",
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("production factory requires assets"),
            "{error}"
        );
    }

    #[test]
    fn weights_free_contract_fixtures_are_unique_and_paired() {
        let orphan = ProviderRegistryBuilder::new()
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .build()
            .err()
            .expect("an orphan fixture must fail")
            .to_string();
        assert!(
            orphan.contains("has no matching memory strategy"),
            "{orphan}"
        );

        let duplicate = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .build()
            .err()
            .expect("duplicate fixtures must fail")
            .to_string();
        assert!(
            duplicate.contains("duplicate memory-contract fixture provider id"),
            "{duplicate}"
        );

        let orphan_resolver = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
            .build()
            .err()
            .expect("a selector-aware resolver without a finite fixture must fail")
            .to_string();
        assert!(orphan_resolver.contains("has no matching contract-surface fixture"));

        let duplicate_resolver = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
            .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
            .build()
            .err()
            .expect("duplicate selector-aware resolvers must fail")
            .to_string();
        assert!(duplicate_resolver.contains("duplicate memory-contract surface resolver"));
    }

    #[test]
    fn selector_aware_resolver_receives_the_explicit_resolved_artifact_tier() {
        let registry = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .register_memory_contract_surface_resolver(DUMMY_SURFACE_RESOLVER)
            .build()
            .unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        let implemented: Vec<_> = surfaces
            .iter()
            .filter(|surface| {
                surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support
                    == MemoryStrategySupport::Implemented
            })
            .map(|surface| surface.selector)
            .collect();
        assert_eq!(implemented.len(), 2);
        assert!(implemented.iter().all(|selector| {
            selector.tier == MemoryContractSurfaceTier::Q4
                && selector.load_shape == crate::LoadShape::DeferredMaterialization
        }));
        assert_eq!(
            registry
                .memory_contract_surface_resolver_registrations()
                .len(),
            1
        );
    }

    #[test]
    fn contract_surface_inventory_fails_closed_on_missing_empty_duplicate_and_wrong_provider() {
        let missing = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .build()
            .unwrap()
            .memory_contract_surfaces()
            .err()
            .expect("a missing fixture must fail")
            .to_string();
        assert!(
            missing.contains(
                "neither a weights-free contract-surface fixture nor a resident-only witness"
            ),
            "{missing}"
        );

        for (surface_specs, expected) in [
            (
                empty_surface_specs as fn() -> Vec<MemoryContractSurfaceSpec>,
                "publishes no surface selectors",
            ),
            (duplicate_surface_specs, "repeats surface selector"),
            (mismatched_surface_spec, "does not match its LoadSpec"),
        ] {
            let error = ProviderRegistryBuilder::new()
                .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
                .register_memory_contract_fixture(MemoryContractFixtureRegistration {
                    provider_id: "dummy_weights_free_route",
                    contract: weights_free_fixture_contract,
                    surface_specs,
                })
                .build()
                .unwrap()
                .memory_contract_surfaces()
                .err()
                .expect("an invalid surface inventory must fail")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }

        let wrong_provider = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(MemoryContractFixtureRegistration {
                provider_id: "dummy_weights_free_route",
                contract: wrong_provider_fixture_contract,
                surface_specs: mlx_memory_contract_surface_specs,
            })
            .build()
            .unwrap()
            .memory_contract_surfaces()
            .err()
            .expect("a wrong-provider fixture must fail")
            .to_string();
        assert!(
            wrong_provider.contains("returned contract for 'wrong_provider'"),
            "{wrong_provider}"
        );
    }

    #[test]
    fn resident_only_witnesses_are_explicit_excluded_and_mutation_checked() {
        let orphan = ProviderRegistryBuilder::new()
            .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
            .build()
            .err()
            .expect("a resident-only witness must pair with a memory strategy")
            .to_string();
        assert!(
            orphan.contains("has no matching memory strategy"),
            "{orphan}"
        );

        let registry = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
            .build()
            .unwrap();
        assert_eq!(
            registry.resident_only_memory_contract_registrations().len(),
            1
        );
        assert!(registry.memory_contract_surfaces().unwrap().is_empty());

        let overlap = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_memory_contract_fixture(DUMMY_WEIGHTS_FREE_CONTRACT_FIXTURE)
            .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
            .build()
            .err()
            .expect("a route cannot be both enumerated and resident-only")
            .to_string();
        assert!(overlap.contains("has both a contract-surface fixture and a resident-only witness"));

        let duplicate = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
            .register_resident_only_memory_contract(DUMMY_RESIDENT_ONLY_WITNESS)
            .build()
            .err()
            .expect("resident-only witnesses must be unique")
            .to_string();
        assert!(
            duplicate.contains("duplicate resident-only memory-contract witness provider id"),
            "{duplicate}"
        );

        let mutated = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_WEIGHTS_FREE_MEMORY_REGISTRATION)
            .register_resident_only_memory_contract(ResidentOnlyMemoryContractRegistration {
                contract: false_resident_only_contract,
                ..DUMMY_RESIDENT_ONLY_WITNESS
            })
            .build()
            .unwrap()
            .memory_contract_surfaces()
            .err()
            .expect("an optimized rung cannot hide behind a resident-only witness")
            .to_string();
        assert!(
            mutated.contains("exposes BoundedDecode as Implemented"),
            "{mutated}"
        );
    }

    #[test]
    fn unmatched_memory_strategy_requires_explicit_composed_route_registration() {
        let error = ProviderRegistryBuilder::new()
            .register_memory_strategy(DUMMY_COMPOSED_MEMORY_REGISTRATION)
            .build()
            .err()
            .expect("an ordinary memory registration must still match a generator");
        assert_eq!(
            error.to_string(),
            "memory-strategy contract 'dummy_composed_route' has no matching generator registration"
        );

        let registry = ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(DUMMY_COMPOSED_MEMORY_REGISTRATION)
            .build()
            .expect("an explicitly composed route is a valid memory-contract owner");
        let contract = registry
            .memory_strategy_contract(
                "dummy_composed_route",
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .expect("the composed route resolves")
            .expect("the composed route has a contract");
        assert_eq!(contract.provider_id, "dummy_composed_route");
        assert_eq!(
            registry
                .activation_memory_bytes_1024("dummy_composed_route")
                .expect("a known composed route has a truthful unmeasured activation state"),
            None
        );
        assert!(registry
            .activation_memory_bytes_1024("unknown_composed_route")
            .is_err());
    }

    /// A tier the platform declared unimplemented is rejected at the load boundary — loudly, naming
    /// the tier, the id, and the platform's reason (epic 11037 SC#5: a quant tier is a creative
    /// choice, never silently substituted). `dummy_load` would otherwise *succeed*, so this pins that
    /// the guard fires ahead of the provider rather than leaving the coercion to the backend.
    #[test]
    fn rejected_quant_tier_fails_loudly_at_load() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();

        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        let error = registry
            .load("dummy_test_model", &spec)
            .err()
            .expect("a rejected tier must not reach the provider");
        match error {
            Error::Unsupported(message) => assert_eq!(
                message,
                "quant tier Nvfp4 is not implemented by this runtime's backend \
                 (requested for 'dummy_test_model'): no FP4 quantizer on this backend. Refusing to \
                 load rather than silently serving a different tier's numerics."
            ),
            other => panic!("a rejected quant tier is a capability gap, got {other:?}"),
        }
    }

    /// The guard is scoped to the declared tiers: an unrejected tier (and a dense, `None` load) still
    /// reaches the provider untouched, and a catalog that declares nothing rejects nothing.
    #[test]
    fn unrejected_quant_tiers_still_load() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();

        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            spec.quantize = quant;
            assert!(
                registry.load("dummy_test_model", &spec).is_ok(),
                "{quant:?} must still load"
            );
        }

        // A catalog whose backend implements every tier (the CUDA candle catalog) declares no
        // rejection and is unaffected.
        let permissive = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .build()
            .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        assert!(permissive.load("dummy_test_model", &spec).is_ok());
    }

    /// An unknown id reports as an unknown id even when the spec also carries a rejected tier — the
    /// guard runs after id resolution so the caller sees the primary fault.
    #[test]
    fn unknown_id_wins_over_rejected_quant() {
        let registry = ProviderRegistryBuilder::new()
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        let error = registry
            .load("nope", &spec)
            .err()
            .expect("unknown id must fail")
            .to_string();
        assert!(
            error.contains("no generator registered for id 'nope'"),
            "{error}"
        );
    }

    #[test]
    fn explicit_registry_rejects_duplicate_ids_deterministically() {
        let registration = ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        };
        let error = ProviderRegistryBuilder::new()
            .register_generator(registration)
            .register_generator(registration)
            .build()
            .err()
            .expect("duplicate registry must fail");
        assert_eq!(
            error.to_string(),
            "duplicate generator id 'dummy_test_model' in explicit registry"
        );
    }

    #[test]
    fn checkpoint_codecs_register_once_per_catalog_and_duplicate_rows_fail_closed() {
        use crate::checkpoint_codec::{WeightEncoding, DENSE_BF16_CODEC};

        let registry = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_codec(DENSE_BF16_CODEC)
            .build()
            .expect("one baseline codec row builds");
        assert_eq!(
            registry
                .checkpoint_codecs()
                .codecs()
                .copied()
                .collect::<Vec<_>>(),
            [DENSE_BF16_CODEC]
        );
        assert_eq!(
            registry
                .checkpoint_codecs()
                .for_encoding(WeightEncoding::DenseBf16),
            Some(&DENSE_BF16_CODEC)
        );
        assert!(registry
            .checkpoint_codecs()
            .for_encoding(WeightEncoding::Fp8E4M3)
            .is_none());

        // A composed catalog that registers the same portable row twice (the withdrawn sc-20638
        // draft's duplicate codec-suite rows) is refused at build, not deduplicated silently.
        let duplicated = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_codec(DENSE_BF16_CODEC)
            .register_checkpoint_codec(DENSE_BF16_CODEC)
            .build();
        assert!(
            duplicated
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("duplicate checkpoint codec id")),
            "duplicate codec rows must fail the build: {:?}",
            duplicated.as_ref().err().map(ToString::to_string)
        );

        let empty = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .build()
            .expect("a catalog without codecs still builds");
        assert!(empty.checkpoint_codecs().is_empty());
    }

    #[test]
    fn imported_routes_are_exact_and_project_the_selected_provider() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .expect("exact imported route");

        let selected = registry
            .imported_model_descriptor(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
            )
            .expect("registered shape and operation resolve");
        assert_eq!(selected.id, "dummy_test_model");
        assert_eq!(selected.required_components, &["imported_component"]);
        assert!(registry
            .imported_model_descriptor(
                "test",
                ImportedModelSource::FusedCheckpoint,
                ImportedModelOperation::Generate,
            )
            .is_none());
        assert!(registry
            .imported_model_descriptor(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Edit,
            )
            .is_none());
    }

    #[test]
    fn imported_route_can_withdraw_structurally_invalid_adapter_inheritance() {
        fn adapter_descriptor() -> ModelDescriptor {
            let mut descriptor = dummy_descriptor();
            descriptor.id = "dummy_adapter_model";
            descriptor.capabilities.supports_lora = true;
            descriptor.capabilities.supports_lokr = true;
            descriptor
        }
        fn adapter_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
            Ok(Box::new(DummyGen {
                desc: adapter_descriptor(),
            }))
        }

        const RESTRICTED_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                backend: CheckpointBackend::Mlx,
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::Generate,
                provider_id: "dummy_adapter_model",
                required_components: None,
                inherit_adapters: false,
            }];
        const RESTRICTED_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] =
            &[CheckpointAdapterCapabilityRegistration {
                operation: ImportedModelOperation::Generate,
                inherit_provider_capabilities: true,
                supports_adapter_inheritance: false,
            }];
        let mut adapter = fixture_adapter(RESTRICTED_BINDING);
        adapter.capabilities = RESTRICTED_CAPABILITIES;

        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: adapter_descriptor,
                load: adapter_load,
                footprint: None,
            })
            .register_checkpoint_adapter(adapter)
            .build()
            .expect("structurally restricted imported route");

        let ordinary = (registry.generators().next().unwrap().descriptor)();
        assert!(ordinary.capabilities.supports_lora);
        assert!(ordinary.capabilities.supports_lokr);
        let imported = registry
            .imported_model_descriptor(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
            )
            .unwrap();
        assert!(!imported.capabilities.supports_lora);
        assert!(!imported.capabilities.supports_lokr);

        // sc-21483 (epic 11037 E6): the withdrawn capability must be *observable*. An adapter-bearing
        // request against this route is a typed capability refusal, never a silently dropped adapter.
        let mut spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("weights")));
        spec.adapters = vec![crate::runtime::AdapterSpec::new(
            std::path::PathBuf::from("adapter.safetensors"),
            1.0,
            crate::runtime::AdapterKind::Lora,
        )];
        let error = registry
            .ensure_imported_model_adapters_allowed(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
                &spec,
            )
            .expect_err("an adapter-bearing request on a non-inheriting route is refused");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "the refusal must be a typed capability error, got {error:?}"
        );
        assert!(
            error.to_string().contains("does not inherit adapters"),
            "{error}"
        );

        // The same route with no adapter selected loads normally…
        let bare = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from("weights")));
        registry
            .ensure_imported_model_adapters_allowed(
                "test",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
                &bare,
            )
            .expect("an adapter-free request is unaffected");

        // …and so does an adapter-bearing request against a route that DOES inherit adapters
        // (`imported_route_resolves_descriptor` covers the inheriting fixture), proving the gate
        // keys off the binding rather than blanket-refusing adapters.
        registry
            .ensure_imported_model_adapters_allowed(
                "unrouted_family",
                ImportedModelSource::TransformerFile,
                ImportedModelOperation::Generate,
                &spec,
            )
            .expect("an unrouted family is not this gate's decision");
    }

    /// sc-21483: the shared refusal is capability-shaped, not route-shaped — a descriptor that
    /// advertises either adapter form admits, one that advertises neither refuses.
    #[test]
    fn adapter_refusal_keys_off_the_advertised_capability() {
        let mut capabilities = dummy_descriptor().capabilities;
        capabilities.supports_lora = false;
        capabilities.supports_lokr = false;
        assert!(reject_unsupported_adapters("m", &capabilities, 0).is_ok());
        assert!(reject_unsupported_adapters("m", &capabilities, 1).is_err());

        capabilities.supports_lokr = true;
        assert!(reject_unsupported_adapters("m", &capabilities, 2).is_ok());
        capabilities.supports_lokr = false;
        capabilities.supports_lora = true;
        assert!(reject_unsupported_adapters("m", &capabilities, 2).is_ok());
    }

    const FIXTURE_DIALECTS: &[CheckpointDialectRegistration] = &[CheckpointDialectRegistration {
        id: "fixture-diffusers",
        source: ImportedModelSource::TransformerFile,
    }];
    const FIXTURE_SIGNATURES: &[CheckpointSignatureRegistration] =
        &[CheckpointSignatureRegistration {
            id: "fixture-transformer-v1",
            dialect: "fixture-diffusers",
            required_tensor_names: &["transformer.weight"],
        }];
    const FIXTURE_COMPONENTS: &[CheckpointComponentRegistration] =
        &[CheckpointComponentRegistration {
            role: "transformer",
            min_count: 1,
            max_count: 1,
        }];
    const FIXTURE_BASES: &[CheckpointBaseCompatibilityRegistration] =
        &[CheckpointBaseCompatibilityRegistration {
            component_role: "transformer",
            compatible_families: &["test"],
        }];
    const FIXTURE_MAPPINGS: &[CheckpointCanonicalMappingRegistration] =
        &[CheckpointCanonicalMappingRegistration {
            dialect: "fixture-diffusers",
            mapping_id: "fixture-identity-v1",
            plan_driven_backends: &[],
        }];
    const FIXTURE_RECOVERY: &[CheckpointConfigRecoveryRegistration] =
        &[CheckpointConfigRecoveryRegistration {
            field: "hidden-size",
            recovery_id: "fixture-tensor-shape-v1",
        }];
    const FIXTURE_OPERATIONS: &[ImportedModelOperation] = &[ImportedModelOperation::Generate];
    const FIXTURE_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] =
        &[CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Generate,
            inherit_provider_capabilities: true,
            supports_adapter_inheritance: true,
        }];
    const FIXTURE_MLX_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Mlx,
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Generate,
            provider_id: "dummy_test_model",
            required_components: Some(&["imported_component"]),
            inherit_adapters: true,
        }];
    const FIXTURE_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Candle,
            provider_id: "dummy_candle_test_model",
            ..FIXTURE_MLX_BINDINGS[0]
        }];
    const FIXTURE_TWO_OPERATIONS: &[ImportedModelOperation] = &[
        ImportedModelOperation::Generate,
        ImportedModelOperation::Edit,
    ];
    const FIXTURE_TWO_CAPABILITIES: &[CheckpointAdapterCapabilityRegistration] = &[
        FIXTURE_CAPABILITIES[0],
        CheckpointAdapterCapabilityRegistration {
            operation: ImportedModelOperation::Edit,
            ..FIXTURE_CAPABILITIES[0]
        },
    ];
    const FIXTURE_CANDLE_GENERATE_AND_EDIT_BINDINGS: &[CheckpointBackendBindingRegistration] = &[
        FIXTURE_CANDLE_BINDINGS[0],
        CheckpointBackendBindingRegistration {
            operation: ImportedModelOperation::Edit,
            ..FIXTURE_CANDLE_BINDINGS[0]
        },
    ];
    const FIXTURE_OTHER_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            provider_id: "dummy_other_candle_model",
            ..FIXTURE_CANDLE_BINDINGS[0]
        }];
    const FIXTURE_MAGE_MLX_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Mlx,
            source: ImportedModelSource::TransformerDirectory,
            operation: ImportedModelOperation::Generate,
            provider_id: "mage_flow_base",
            required_components: Some(&["base_snapshot"]),
            inherit_adapters: false,
        }];
    const FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS: &[CheckpointBackendBindingRegistration] =
        &[CheckpointBackendBindingRegistration {
            backend: CheckpointBackend::Candle,
            provider_id: "dummy_legacy_mage_candle_model",
            ..FIXTURE_MLX_BINDINGS[0]
        }];

    fn fixture_adapter(
        bindings: &'static [CheckpointBackendBindingRegistration],
    ) -> CheckpointAdapterRegistration {
        CheckpointAdapterRegistration {
            adapter_id: "fixture-family-v1",
            family: "test",
            compatibility_projection: ImportedModelCompatibilityProjectionRegistration {
                family: "test",
            },
            signatures: FIXTURE_SIGNATURES,
            dialects: FIXTURE_DIALECTS,
            component_topology: FIXTURE_COMPONENTS,
            base_compatibility: FIXTURE_BASES,
            canonical_mappings: FIXTURE_MAPPINGS,
            config_recovery: FIXTURE_RECOVERY,
            eligible_backends: &[CheckpointBackend::Mlx],
            backend_bindings: bindings,
            operations: FIXTURE_OPERATIONS,
            capabilities: FIXTURE_CAPABILITIES,
        }
    }

    fn cross_platform_fixture_adapter(
        bindings: &'static [CheckpointBackendBindingRegistration],
    ) -> CheckpointAdapterRegistration {
        CheckpointAdapterRegistration {
            eligible_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
            backend_bindings: bindings,
            ..fixture_adapter(bindings)
        }
    }

    fn two_operation_cross_platform_fixture_adapter(
        bindings: &'static [CheckpointBackendBindingRegistration],
    ) -> CheckpointAdapterRegistration {
        CheckpointAdapterRegistration {
            operations: FIXTURE_TWO_OPERATIONS,
            capabilities: FIXTURE_TWO_CAPABILITIES,
            ..cross_platform_fixture_adapter(bindings)
        }
    }

    #[test]
    fn checkpoint_adapter_is_the_authority_and_projects_legacy_import_routes() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .expect("one adapter registration is a complete fixture-family addition");

        let adapter = registry.checkpoint_adapters().next().unwrap();
        assert_eq!(adapter.adapter_id, "fixture-family-v1");
        assert_eq!(adapter.signatures, FIXTURE_SIGNATURES);
        assert_eq!(adapter.dialects, FIXTURE_DIALECTS);
        assert_eq!(adapter.component_topology, FIXTURE_COMPONENTS);
        assert_eq!(adapter.base_compatibility, FIXTURE_BASES);
        assert_eq!(adapter.canonical_mappings, FIXTURE_MAPPINGS);
        assert_eq!(adapter.config_recovery, FIXTURE_RECOVERY);
        assert_eq!(adapter.operations, FIXTURE_OPERATIONS);
        assert_eq!(adapter.capabilities, FIXTURE_CAPABILITIES);

        let projected: Vec<_> = registry.imported_models().copied().collect();
        assert_eq!(
            projected,
            [ImportedModelRegistration {
                family: "test",
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::Generate,
                provider_id: "dummy_test_model",
                required_components: Some(&["imported_component"]),
                inherit_adapters: true,
            }]
        );
        assert_eq!(
            registry
                .imported_model_descriptor(
                    "test",
                    ImportedModelSource::TransformerFile,
                    ImportedModelOperation::Generate,
                )
                .unwrap()
                .id,
            "dummy_test_model"
        );
    }

    #[test]
    fn mage_adapter_binds_provider_truth_and_preserves_the_exact_legacy_family() {
        let adapter = CheckpointAdapterRegistration {
            backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
            ..MAGE_FLOW_CHECKPOINT_ADAPTER
        };
        assert_eq!(adapter.family, "mage_flow");

        let registry = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(adapter)
            .build()
            .expect("provider-truth Mage family must bind the real MLX route");

        assert_eq!(
            registry.imported_models().copied().collect::<Vec<_>>(),
            [ImportedModelRegistration {
                family: "mage-flow",
                source: ImportedModelSource::TransformerDirectory,
                operation: ImportedModelOperation::Generate,
                provider_id: "mage_flow_base",
                required_components: Some(&["base_snapshot"]),
                inherit_adapters: false,
            }]
        );
    }

    /// sc-20644 Wan row — the dual-expert backbone is DECLARED, distinctly, and the declaration is
    /// what a later inspector/plan-layer change has to satisfy.
    ///
    /// The two experts are two single-count roles, not one role with `max_count: 2`. That
    /// distinction is the whole point: a high-noise and a low-noise expert are selected per denoise
    /// step and are not interchangeable, so a plan that recorded them as two instances of one role
    /// could not say which is which — which is exactly the state SceneWorks' inspector is in today
    /// (both map to the single path role `transformer`).
    ///
    /// Also pins the two things about this family that are easy to get wrong: it is Candle-only, and
    /// it does NOT inherit adapter flags, because `load_from_comfyui_experts` has no LoRA seam.
    ///
    /// Failing mutations: collapse the two roles into one `transformer` with `max_count: 2`; set
    /// `supports_adapter_inheritance: true`; add `CheckpointBackend::Mlx`.
    #[test]
    fn the_wan_adapter_declares_two_distinct_single_count_experts() {
        let topology = WAN_CHECKPOINT_ADAPTER.component_topology;
        let backbones: Vec<&CheckpointComponentRegistration> = topology
            .iter()
            .filter(|component| component.role.starts_with("transformer"))
            .collect();
        assert_eq!(
            backbones.len(),
            2,
            "Wan declares TWO backbones; got {:?}",
            topology.iter().map(|c| c.role).collect::<Vec<_>>()
        );
        for component in &backbones {
            assert_eq!(
                (component.min_count, component.max_count),
                (1, 1),
                "{}: each expert is exactly one artifact, not one role holding two",
                component.role
            );
        }
        let mut roles: Vec<&str> = backbones.iter().map(|c| c.role).collect();
        roles.sort_unstable();
        assert_eq!(roles, ["transformer-high", "transformer-low"]);

        assert_eq!(
            WAN_CHECKPOINT_ADAPTER.eligible_backends,
            [CheckpointBackend::Candle],
            "the ComfyUI Wan expert pair loads on Candle only"
        );

        // The inference half of the cross-repo spelling tie (sc-20644 review minor 8). These
        // topology roles are hyphenated; SceneWorks' plan-layer roles are underscored, and the two
        // are joined by one projection that nothing structural enforces. Pinned in the `mapping_id`
        // posture: this asserts the projection, SceneWorks asserts the projected literals are what
        // its inspector emits. Either side drifting alone turns a Wan plan into two unrecognized
        // roles.
        let project = |topology_role: &str| topology_role.replace('-', "_");
        assert_eq!(project("transformer-high"), "transformer_high");
        assert_eq!(project("transformer-low"), "transformer_low");
        assert_ne!(
            "transformer-high", "transformer_high",
            "fixture check: the two spellings genuinely differ, so the projection is not vacuous"
        );
        // The precedent this follows rather than invents — the component id both repos already
        // share is spelled the same two ways.
        assert_eq!(
            project("base-snapshot"),
            crate::runtime::BASE_SNAPSHOT_COMPONENT
        );
        assert_eq!(
            WAN_CHECKPOINT_ADAPTER
                .dialects
                .iter()
                .map(|dialect| dialect.source)
                .collect::<Vec<_>>(),
            [ImportedModelSource::ComfyUiTree]
        );
        let generate = WAN_CHECKPOINT_ADAPTER
            .capabilities
            .iter()
            .find(|capability| capability.operation == ImportedModelOperation::Generate)
            .expect("Wan binds Generate");
        assert!(
            !generate.supports_adapter_inheritance,
            "`load_from_comfyui_experts` takes no adapters, so the imported route must not \
             advertise the provider's LoRA flags"
        );
    }

    #[test]
    fn portable_adapter_families_and_legacy_projections_are_explicit_for_every_family() {
        let identities = [
            (&KREA_2_CHECKPOINT_ADAPTER, "krea_2", "krea_2"),
            (&SDXL_CHECKPOINT_ADAPTER, "sdxl", "sdxl"),
            (&MAGE_FLOW_CHECKPOINT_ADAPTER, "mage_flow", "mage-flow"),
            (&Z_IMAGE_CHECKPOINT_ADAPTER, "z-image", "z-image"),
            (&QWEN_IMAGE_CHECKPOINT_ADAPTER, "qwen-image", "qwen-image"),
            (&FLUX2_CHECKPOINT_ADAPTER, "flux2", "flux2"),
            (&WAN_CHECKPOINT_ADAPTER, "wan", "wan-video"),
        ];

        for (adapter, portable_family, legacy_family) in identities {
            assert_eq!(adapter.family, portable_family, "{}", adapter.adapter_id);
            assert_eq!(
                adapter.compatibility_projection.family, legacy_family,
                "{}",
                adapter.adapter_id
            );
        }
        assert_eq!(
            MAGE_FLOW_CHECKPOINT_ADAPTER.base_compatibility,
            [CheckpointBaseCompatibilityRegistration {
                component_role: "base-snapshot",
                compatible_families: &["mage_flow"],
            }]
        );
    }

    #[test]
    fn compatibility_projection_rejects_alias_collisions_and_duplicate_legacy_families() {
        let mage = CheckpointAdapterRegistration {
            backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
            ..MAGE_FLOW_CHECKPOINT_ADAPTER
        };
        let mut legacy_family_owner = fixture_adapter(FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS);
        legacy_family_owner.adapter_id = "legacy-mage-family-v1";
        legacy_family_owner.family = "mage-flow";
        legacy_family_owner.compatibility_projection.family = "legacy-mage";
        legacy_family_owner.eligible_backends = &[CheckpointBackend::Candle];

        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(mage)
            .register_checkpoint_adapter(legacy_family_owner)
            .build()
            .err()
            .expect("an explicit legacy alias cannot shadow another portable authority")
            .to_string();
        assert_eq!(
            error,
            "checkpoint-adapter compatibility family 'mage-flow' for 'mage-flow-diffusers-v1' collides with portable family owned by 'legacy-mage-family-v1'"
        );

        let first = fixture_adapter(FIXTURE_MLX_BINDINGS);
        let mut second = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
        second.adapter_id = "other-family-v1";
        second.family = "other";
        second.eligible_backends = &[CheckpointBackend::Candle];
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(first)
            .register_checkpoint_adapter(second)
            .build()
            .err()
            .expect("legacy compatibility families must be unique")
            .to_string();
        assert_eq!(
            error,
            "duplicate checkpoint-adapter compatibility family 'test'"
        );
    }

    #[test]
    fn checkpoint_adapter_registry_rejects_duplicate_malformed_and_implementation_free_entries() {
        let duplicate = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .err()
            .expect("duplicate adapters fail")
            .to_string();
        assert!(
            duplicate.contains("duplicate checkpoint-adapter id"),
            "{duplicate}"
        );

        let implementation_free = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(&[]))
            .build()
            .err()
            .expect("implementation-free adapters fail")
            .to_string();
        assert!(
            implementation_free.contains("has no backend bindings"),
            "{implementation_free}"
        );

        let mut malformed = fixture_adapter(FIXTURE_MLX_BINDINGS);
        malformed.signatures = &[];
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(malformed)
            .build()
            .err()
            .expect("signature-free adapter fails")
            .to_string();
        assert!(error.contains("signatures must not be empty"), "{error}");

        let malformed_surfaces: [fn(&mut CheckpointAdapterRegistration); 8] = [
            |adapter| adapter.dialects = &[],
            |adapter| adapter.component_topology = &[],
            |adapter| adapter.base_compatibility = &[],
            |adapter| adapter.canonical_mappings = &[],
            |adapter| adapter.config_recovery = &[],
            |adapter| adapter.eligible_backends = &[],
            |adapter| adapter.operations = &[],
            |adapter| adapter.capabilities = &[],
        ];
        for mutate in malformed_surfaces {
            let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
            mutate(&mut adapter);
            assert!(
                ProviderRegistryBuilder::new()
                    .register_generator(DUMMY_GENERATOR_REGISTRATION)
                    .register_checkpoint_adapter(adapter)
                    .build()
                    .is_err(),
                "removing a required portable adapter surface must fail"
            );
        }

        let mut same_family = fixture_adapter(FIXTURE_MLX_BINDINGS);
        same_family.adapter_id = "fixture-family-v2";
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .register_checkpoint_adapter(same_family)
            .build()
            .err()
            .expect("two adapter ids must not claim one family")
            .to_string();
        assert!(
            error.contains("duplicate checkpoint-adapter family 'test'"),
            "{error}"
        );

        let mut orphan_operation = fixture_adapter(FIXTURE_MLX_BINDINGS);
        orphan_operation.operations = FIXTURE_TWO_OPERATIONS;
        orphan_operation.capabilities = FIXTURE_TWO_CAPABILITIES;
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(orphan_operation)
            .build()
            .err()
            .expect("a sole-backend operation without a binding must fail")
            .to_string();
        assert_eq!(
            error,
            "checkpoint-adapter 'fixture-family-v1' operation Edit has no binding on sole eligible backend 'mlx'"
        );
    }

    #[test]
    fn checkpoint_adapter_registry_rejects_mutated_metadata_and_provider_routes() {
        const EMPTY_COMPONENTS_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                required_components: Some(&[]),
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const DUPLICATE_COMPONENTS_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                required_components: Some(&["same", "same"]),
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const MALFORMED_PROVIDER_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                provider_id: "Bad Provider",
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const UNKNOWN_SOURCE_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                source: ImportedModelSource::FusedCheckpoint,
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const WHITESPACE_TENSOR_SIGNATURE: &[CheckpointSignatureRegistration] =
            &[CheckpointSignatureRegistration {
                required_tensor_names: &["transformer bad.weight"],
                ..FIXTURE_SIGNATURES[0]
            }];
        const INVALID_CARDINALITY: &[CheckpointComponentRegistration] =
            &[CheckpointComponentRegistration {
                min_count: 0,
                ..FIXTURE_COMPONENTS[0]
            }];
        const DUPLICATE_OPERATIONS: &[ImportedModelOperation] = &[
            ImportedModelOperation::Generate,
            ImportedModelOperation::Generate,
        ];
        const DUPLICATE_BACKENDS: &[CheckpointBackend] =
            &[CheckpointBackend::Mlx, CheckpointBackend::Mlx];
        const NON_PROJECTABLE_CAPABILITY: &[CheckpointAdapterCapabilityRegistration] =
            &[CheckpointAdapterCapabilityRegistration {
                inherit_provider_capabilities: false,
                ..FIXTURE_CAPABILITIES[0]
            }];
        const NON_INHERITING_CAPABILITY: &[CheckpointAdapterCapabilityRegistration] =
            &[CheckpointAdapterCapabilityRegistration {
                supports_adapter_inheritance: false,
                ..FIXTURE_CAPABILITIES[0]
            }];

        type AdapterMutation = (&'static str, fn(&mut CheckpointAdapterRegistration));
        let mutations: [AdapterMutation; 11] = [
            ("identity", |adapter| adapter.family = "Bad Family"),
            ("compatibility projection", |adapter| {
                adapter.compatibility_projection.family = "Bad Legacy Family"
            }),
            ("tensor name", |adapter| {
                adapter.signatures = WHITESPACE_TENSOR_SIGNATURE
            }),
            ("cardinality", |adapter| {
                adapter.component_topology = INVALID_CARDINALITY
            }),
            ("eligible backend", |adapter| {
                adapter.eligible_backends = DUPLICATE_BACKENDS
            }),
            ("operation", |adapter| {
                adapter.operations = DUPLICATE_OPERATIONS
            }),
            ("capability projection", |adapter| {
                adapter.capabilities = NON_PROJECTABLE_CAPABILITY
            }),
            ("source dialect", |adapter| {
                adapter.backend_bindings = UNKNOWN_SOURCE_BINDING
            }),
            ("provider id", |adapter| {
                adapter.backend_bindings = MALFORMED_PROVIDER_BINDING
            }),
            ("empty required components", |adapter| {
                adapter.backend_bindings = EMPTY_COMPONENTS_BINDING
            }),
            ("duplicate required components", |adapter| {
                adapter.backend_bindings = DUPLICATE_COMPONENTS_BINDING
            }),
        ];
        for (surface, mutate) in mutations {
            let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
            mutate(&mut adapter);
            assert!(
                ProviderRegistryBuilder::new()
                    .register_generator(DUMMY_GENERATOR_REGISTRATION)
                    .register_checkpoint_adapter(adapter)
                    .build()
                    .is_err(),
                "mutating {surface} must fail registry construction"
            );
        }

        let mut wrong_family = fixture_adapter(FIXTURE_MLX_BINDINGS);
        wrong_family.family = "other";
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(wrong_family)
            .build()
            .err()
            .expect("a provider from another family must fail")
            .to_string();
        assert!(error.contains("does not match generator"), "{error}");

        let mut contradictory_inheritance = fixture_adapter(FIXTURE_MLX_BINDINGS);
        contradictory_inheritance.capabilities = NON_INHERITING_CAPABILITY;
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(contradictory_inheritance)
            .build()
            .err()
            .expect("binding inheritance must obey portable capability policy")
            .to_string();
        assert!(
            error.contains("inherits adapters contrary to capability policy"),
            "{error}"
        );
    }

    #[test]
    fn checkpoint_adapter_registry_rejects_dangling_or_contradictory_metadata() {
        const UNKNOWN_PROVIDER: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                provider_id: "not_registered",
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const WRONG_BACKEND: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                backend: CheckpointBackend::Candle,
                ..FIXTURE_MLX_BINDINGS[0]
            }];
        const DUPLICATE_BINDINGS: &[CheckpointBackendBindingRegistration] =
            &[FIXTURE_MLX_BINDINGS[0], FIXTURE_MLX_BINDINGS[0]];

        for (bindings, expected) in [
            (UNKNOWN_PROVIDER, "targets unregistered generator"),
            (DUPLICATE_BINDINGS, "duplicate checkpoint-adapter binding"),
        ] {
            let error = ProviderRegistryBuilder::new()
                .register_generator(DUMMY_GENERATOR_REGISTRATION)
                .register_checkpoint_adapter(fixture_adapter(bindings))
                .build()
                .err()
                .expect("contradictory binding fails")
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?}, got: {error}"
            );
        }
        let mut wrong_backend = fixture_adapter(WRONG_BACKEND);
        wrong_backend.eligible_backends = &[CheckpointBackend::Mlx, CheckpointBackend::Candle];
        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(wrong_backend)
            .build()
            .err()
            .expect("backend/provider mismatch fails")
            .to_string();
        assert!(
            error.contains("does not match generator backend"),
            "{error}"
        );

        const UNKNOWN_DIALECT_SIGNATURE: &[CheckpointSignatureRegistration] =
            &[CheckpointSignatureRegistration {
                dialect: "missing",
                ..FIXTURE_SIGNATURES[0]
            }];
        const UNKNOWN_DIALECT_MAPPING: &[CheckpointCanonicalMappingRegistration] =
            &[CheckpointCanonicalMappingRegistration {
                dialect: "missing",
                ..FIXTURE_MAPPINGS[0]
            }];
        const UNKNOWN_COMPONENT_BASE: &[CheckpointBaseCompatibilityRegistration] =
            &[CheckpointBaseCompatibilityRegistration {
                component_role: "missing",
                ..FIXTURE_BASES[0]
            }];
        const UNDECLARED_OPERATION_BINDING: &[CheckpointBackendBindingRegistration] =
            &[CheckpointBackendBindingRegistration {
                operation: ImportedModelOperation::Edit,
                ..FIXTURE_MLX_BINDINGS[0]
            }];

        let mutations: [fn(&mut CheckpointAdapterRegistration); 4] = [
            |adapter| adapter.signatures = UNKNOWN_DIALECT_SIGNATURE,
            |adapter| adapter.canonical_mappings = UNKNOWN_DIALECT_MAPPING,
            |adapter| adapter.base_compatibility = UNKNOWN_COMPONENT_BASE,
            |adapter| adapter.backend_bindings = UNDECLARED_OPERATION_BINDING,
        ];
        for mutate in mutations {
            let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
            mutate(&mut adapter);
            assert!(
                ProviderRegistryBuilder::new()
                    .register_generator(DUMMY_GENERATOR_REGISTRATION)
                    .register_checkpoint_adapter(adapter)
                    .build()
                    .is_err(),
                "dangling portable metadata must fail"
            );
        }

        let error = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .err()
            .expect("a shipped eligible backend without a family binding must fail")
            .to_string();
        assert!(
            error.contains("no binding for shipped eligible backend 'candle'"),
            "{error}"
        );
    }

    /// A canonical mapping may only claim a plan-driven backend the adapter is eligible for, and
    /// may not claim one twice. Each catalog's conformance test reads
    /// `plan_driven_backends` as the exact set of backends that must ship a `LogicalKeyMapping`
    /// with that id, so an ineligible or repeated entry would make the reachability proof
    /// unsatisfiable (or vacuous) rather than merely untidy.
    #[test]
    fn canonical_mapping_plan_driven_backends_must_be_eligible_and_unique() {
        const INELIGIBLE: &[CheckpointCanonicalMappingRegistration] =
            &[CheckpointCanonicalMappingRegistration {
                // The fixture adapter is MLX-eligible only.
                plan_driven_backends: &[CheckpointBackend::Candle],
                ..FIXTURE_MAPPINGS[0]
            }];
        const REPEATED: &[CheckpointCanonicalMappingRegistration] =
            &[CheckpointCanonicalMappingRegistration {
                plan_driven_backends: &[CheckpointBackend::Mlx, CheckpointBackend::Mlx],
                ..FIXTURE_MAPPINGS[0]
            }];
        for mapping in [INELIGIBLE, REPEATED] {
            let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
            adapter.canonical_mappings = mapping;
            let error = ProviderRegistryBuilder::new()
                .register_generator(DUMMY_GENERATOR_REGISTRATION)
                .register_checkpoint_adapter(adapter)
                .build()
                .err()
                .expect("a repeated or ineligible plan-driven backend must fail the build")
                .to_string();
            assert!(
                error.contains("repeated or ineligible plan-driven backend"),
                "{error}"
            );
        }

        // And the honest declaration still builds.
        const ELIGIBLE: &[CheckpointCanonicalMappingRegistration] =
            &[CheckpointCanonicalMappingRegistration {
                plan_driven_backends: &[CheckpointBackend::Mlx],
                ..FIXTURE_MAPPINGS[0]
            }];
        let mut adapter = fixture_adapter(FIXTURE_MLX_BINDINGS);
        adapter.canonical_mappings = ELIGIBLE;
        ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(adapter)
            .build()
            .expect("an eligible plan-driven backend builds");
    }

    /// The shipped posture, pinned per (adapter, dialect): which mapping id is the authority and
    /// which backends actually implement it. Two facts this guards, both of which were live
    /// defects before sc-20651:
    ///
    /// * the Krea 2 `diffusers` dialect must NOT be `identity-v1` — that mapping accepts every
    ///   on-disk key and decodes undescribed fp8 at unit scale, silently wrong (its own doc comment
    ///   forbids the use); and
    /// * a `mapping_id` with no implementation anywhere must say so (`plan_driven_backends: &[]`)
    ///   rather than reading like a backed route. The catalog tests then prove the non-empty
    ///   entries resolve and the empty ones have nothing masquerading behind them.
    #[test]
    fn shipped_canonical_mapping_posture_is_pinned_per_dialect() {
        use crate::checkpoint_codec::IdentityKeyMapping;

        /// One pinned mapping row: `(dialect, mapping_id, plan_driven_backends)`.
        type MappingRow = (&'static str, &'static str, &'static [CheckpointBackend]);

        let expected: &[(&str, &[MappingRow])] = &[
            (
                KREA_2_CHECKPOINT_ADAPTER.adapter_id,
                &[
                    (
                        // Implemented on BOTH engines since sc-20651 — one dialect, one canonical
                        // mapping id, two implementations.
                        "krea-native",
                        "krea-native-to-diffusers-v1",
                        &[CheckpointBackend::Mlx, CheckpointBackend::Candle],
                    ),
                    (
                        "diffusers",
                        "krea-2-diffusers-v1",
                        &[CheckpointBackend::Mlx],
                    ),
                ],
            ),
            (
                SDXL_CHECKPOINT_ADAPTER.adapter_id,
                &[("ldm", "sdxl-ldm-to-diffusers-v1", &[])],
            ),
            (
                MAGE_FLOW_CHECKPOINT_ADAPTER.adapter_id,
                &[("diffusers", "identity-v1", &[])],
            ),
            (
                Z_IMAGE_CHECKPOINT_ADAPTER.adapter_id,
                &[("comfyui", "z-image-comfyui-to-diffusers-v1", &[])],
            ),
            (
                QWEN_IMAGE_CHECKPOINT_ADAPTER.adapter_id,
                &[("comfyui", "qwen-image-comfyui-to-diffusers-v1", &[])],
            ),
            (
                FLUX2_CHECKPOINT_ADAPTER.adapter_id,
                &[("comfyui", "flux2-comfyui-to-diffusers-v1", &[])],
            ),
            (
                WAN_CHECKPOINT_ADAPTER.adapter_id,
                &[(
                    "comfyui",
                    "wan-comfyui-to-diffusers-v1",
                    &[CheckpointBackend::Candle],
                )],
            ),
        ];
        let shipped: &[&CheckpointAdapterRegistration] = &[
            &KREA_2_CHECKPOINT_ADAPTER,
            &SDXL_CHECKPOINT_ADAPTER,
            &MAGE_FLOW_CHECKPOINT_ADAPTER,
            &Z_IMAGE_CHECKPOINT_ADAPTER,
            &QWEN_IMAGE_CHECKPOINT_ADAPTER,
            &FLUX2_CHECKPOINT_ADAPTER,
            &WAN_CHECKPOINT_ADAPTER,
        ];
        assert_eq!(
            shipped.len(),
            expected.len(),
            "every shipped adapter's mapping posture must be pinned here"
        );
        for (adapter, (adapter_id, rows)) in shipped.iter().zip(expected) {
            assert_eq!(adapter.adapter_id, *adapter_id);
            let actual: Vec<_> = adapter
                .canonical_mappings
                .iter()
                .map(|mapping| {
                    (
                        mapping.dialect,
                        mapping.mapping_id,
                        mapping.plan_driven_backends,
                    )
                })
                .collect();
            let expected_rows: Vec<_> = rows.to_vec();
            assert_eq!(actual, expected_rows, "adapter {adapter_id}");
            assert!(
                !adapter
                    .canonical_mappings
                    .iter()
                    .any(
                        |mapping| mapping.mapping_id == IdentityKeyMapping::MAPPING_ID
                            && !mapping.plan_driven_backends.is_empty()
                    ),
                "adapter {adapter_id} routes a plan through `identity-v1`, which decodes \
                 undescribed fp8 at unit scale — silently wrong rather than refused"
            );
        }
    }

    #[test]
    fn checkpoint_adapter_catalog_conformance_preserves_truthful_backend_asymmetry() {
        let mlx = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .unwrap();
        let candle = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS))
            .build()
            .unwrap();
        assert_eq!(
            mlx.checkpoint_adapter_catalog_conformance_errors(&candle),
            Vec::<String>::new()
        );

        let mlx_only = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .unwrap();
        let candle_without_adapter = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .build()
            .unwrap();
        assert_eq!(
            mlx_only.checkpoint_adapter_catalog_conformance_errors(&candle_without_adapter),
            Vec::<String>::new(),
            "an explicitly MLX-only family must not fabricate a Candle route"
        );

        let mlx_generate_only = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
                FIXTURE_MLX_BINDINGS,
            ))
            .build()
            .unwrap();
        let candle_generate_only = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
                FIXTURE_CANDLE_BINDINGS,
            ))
            .build()
            .unwrap();
        assert_eq!(
            mlx_generate_only
                .checkpoint_adapter_catalog_conformance_errors(&candle_generate_only),
            ["checkpoint-adapter family 'test' (id 'fixture-family-v1') operation Edit has no binding across eligible catalogs"]
        );

        let candle_implements_edit = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(two_operation_cross_platform_fixture_adapter(
                FIXTURE_CANDLE_GENERATE_AND_EDIT_BINDINGS,
            ))
            .build()
            .unwrap();
        assert_eq!(
            mlx_generate_only
                .checkpoint_adapter_catalog_conformance_errors(&candle_implements_edit),
            Vec::<String>::new(),
            "one eligible backend may omit an operation when another real binding implements it"
        );
    }

    #[test]
    fn checkpoint_adapter_catalog_conformance_rejects_family_id_drift_deterministically() {
        let mlx = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .unwrap();

        let mut different_id = fixture_adapter(FIXTURE_CANDLE_BINDINGS);
        different_id.adapter_id = "fixture-family-v2";
        different_id.eligible_backends = &[CheckpointBackend::Candle];
        let candle_different_id = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(different_id)
            .build()
            .unwrap();
        let expected_id_drift = ["checkpoint-adapter family 'test' maps to different adapter ids 'fixture-family-v1' and 'fixture-family-v2' across catalogs"];
        assert_eq!(
            mlx.checkpoint_adapter_catalog_conformance_errors(&candle_different_id),
            expected_id_drift
        );
        assert_eq!(
            candle_different_id.checkpoint_adapter_catalog_conformance_errors(&mlx),
            expected_id_drift,
            "family/id drift diagnostics must not depend on catalog argument order"
        );

        let mut different_family = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
        different_family.family = "other";
        different_family.eligible_backends = &[CheckpointBackend::Candle];
        let candle_different_family = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(different_family)
            .build()
            .unwrap();
        let expected_family_drift = ["checkpoint-adapter id 'fixture-family-v1' maps to different families 'other' and 'test' across catalogs"];
        assert_eq!(
            mlx.checkpoint_adapter_catalog_conformance_errors(&candle_different_family),
            expected_family_drift
        );
        assert_eq!(
            candle_different_family.checkpoint_adapter_catalog_conformance_errors(&mlx),
            expected_family_drift,
            "adapter-id/family drift diagnostics must not depend on catalog argument order"
        );

        let mage_mlx = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_MAGE_MLX_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(CheckpointAdapterRegistration {
                backend_bindings: FIXTURE_MAGE_MLX_BINDINGS,
                ..MAGE_FLOW_CHECKPOINT_ADAPTER
            })
            .build()
            .unwrap();
        let mut normalized_legacy_family = fixture_adapter(FIXTURE_LEGACY_MAGE_CANDLE_BINDINGS);
        normalized_legacy_family.adapter_id = MAGE_FLOW_CHECKPOINT_ADAPTER.adapter_id;
        normalized_legacy_family.family = "mage-flow";
        normalized_legacy_family.compatibility_projection.family = "mage-flow";
        normalized_legacy_family.eligible_backends = &[CheckpointBackend::Candle];
        let mage_candle_with_normalized_family = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_LEGACY_MAGE_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(normalized_legacy_family)
            .build()
            .unwrap();
        let expected_normalization_drift = ["checkpoint-adapter id 'mage-flow-diffusers-v1' maps to different families 'mage-flow' and 'mage_flow' across catalogs"];
        assert_eq!(
            mage_mlx
                .checkpoint_adapter_catalog_conformance_errors(&mage_candle_with_normalized_family),
            expected_normalization_drift,
            "hyphen/underscore normalization must not hide portable-family drift"
        );
        assert_eq!(
            mage_candle_with_normalized_family
                .checkpoint_adapter_catalog_conformance_errors(&mage_mlx),
            expected_normalization_drift,
            "normalization-drift diagnostics must not depend on catalog argument order"
        );

        let mut mlx_shared_projection = fixture_adapter(FIXTURE_MLX_BINDINGS);
        mlx_shared_projection.compatibility_projection.family = "shared-legacy-family";
        let mlx_shared_projection = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(mlx_shared_projection)
            .build()
            .unwrap();
        let mut candle_shared_projection = fixture_adapter(FIXTURE_OTHER_CANDLE_BINDINGS);
        candle_shared_projection.adapter_id = "other-family-v1";
        candle_shared_projection.family = "other";
        candle_shared_projection.compatibility_projection.family = "shared-legacy-family";
        candle_shared_projection.eligible_backends = &[CheckpointBackend::Candle];
        let candle_shared_projection = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_OTHER_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(candle_shared_projection)
            .build()
            .unwrap();
        let expected_alias_ownership_drift = ["checkpoint-adapter compatibility family 'shared-legacy-family' maps to different portable authorities 'other' (id 'other-family-v1') and 'test' (id 'fixture-family-v1') across catalogs"];
        assert_eq!(
            mlx_shared_projection
                .checkpoint_adapter_catalog_conformance_errors(&candle_shared_projection),
            expected_alias_ownership_drift
        );
        assert_eq!(
            candle_shared_projection
                .checkpoint_adapter_catalog_conformance_errors(&mlx_shared_projection),
            expected_alias_ownership_drift,
            "compatibility-family ownership diagnostics must not depend on catalog argument order"
        );
    }

    #[test]
    fn checkpoint_adapter_catalog_conformance_detects_metadata_and_binding_mutations() {
        const MUTATED_RECOVERY: &[CheckpointConfigRecoveryRegistration] =
            &[CheckpointConfigRecoveryRegistration {
                field: "hidden-size",
                recovery_id: "mutated-tensor-shape-v1",
            }];

        let mlx = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(cross_platform_fixture_adapter(FIXTURE_MLX_BINDINGS))
            .build()
            .unwrap();
        let mut mutated = cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS);
        mutated.config_recovery = MUTATED_RECOVERY;
        let candle_with_metadata_drift = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(mutated)
            .build()
            .unwrap();
        let expected_metadata_drift = vec![
            "checkpoint-adapter family 'test' (id 'fixture-family-v1') portable metadata differs across catalogs"
                .to_owned(),
        ];
        assert_eq!(
            mlx.checkpoint_adapter_catalog_conformance_errors(&candle_with_metadata_drift),
            expected_metadata_drift
        );
        assert_eq!(
            candle_with_metadata_drift.checkpoint_adapter_catalog_conformance_errors(&mlx),
            expected_metadata_drift
        );

        let mut mutated_projection = cross_platform_fixture_adapter(FIXTURE_CANDLE_BINDINGS);
        mutated_projection.compatibility_projection.family = "test-legacy";
        let candle_with_projection_drift = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .register_checkpoint_adapter(mutated_projection)
            .build()
            .unwrap();
        assert_eq!(
            mlx.checkpoint_adapter_catalog_conformance_errors(&candle_with_projection_drift),
            expected_metadata_drift,
            "legacy projection drift is portable-metadata drift"
        );
        assert_eq!(
            candle_with_projection_drift.checkpoint_adapter_catalog_conformance_errors(&mlx),
            expected_metadata_drift,
            "projection-drift diagnostics must not depend on catalog argument order"
        );

        let candle_without_adapter = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_CANDLE_GENERATOR_REGISTRATION)
            .build()
            .unwrap();
        let errors = mlx.checkpoint_adapter_catalog_conformance_errors(&candle_without_adapter);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("missing from a catalog")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("no binding for shipped eligible backend 'candle'")),
            "{errors:?}"
        );
    }

    #[test]
    fn unknown_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry.load("no_such_model", &spec).is_err());
    }

    #[test]
    fn dummy_appears_in_iteration() {
        assert!(dummy_registry()
            .generators()
            .any(|r| (r.descriptor)().id == "dummy_test_model"));
    }

    #[test]
    fn macro_delegated_generator_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry
            .load("dummy_delegated_test_model", &spec)
            .expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_delegated_test_model");
        g.validate(&GenerationRequest::default()).unwrap();
    }

    #[test]
    fn macro_registered_trainer_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_trainer("dummy_test_trainer", &spec)
            .expect("dummy trainer is registered");
        assert_eq!(t.descriptor().id, "dummy_test_trainer");
        assert!(registry
            .trainers()
            .any(|r| (r.descriptor)().id == "dummy_test_trainer"));
    }

    #[test]
    fn multiple_generator_constants_compose() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_gen_a", "dummy_multi_gen_b"] {
            let g = registry
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
        }
    }

    #[test]
    fn multiple_trainer_constants_compose() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_trainer_a", "dummy_multi_trainer_b"] {
            let t = registry
                .load_trainer(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(t.descriptor().id, id);
        }
    }

    #[test]
    fn captioner_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let c = registry
            .load_captioner("dummy_test_captioner", &spec)
            .expect("dummy captioner is registered");
        assert_eq!(c.descriptor().id, "dummy_test_captioner");
    }

    #[test]
    fn unknown_captioner_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry.load_captioner("no_such_captioner", &spec).is_err());
    }

    #[test]
    fn dummy_captioner_appears_in_iteration() {
        assert!(dummy_registry()
            .captioners()
            .any(|r| (r.descriptor)().id == "dummy_test_captioner"));
    }

    #[test]
    fn transcriber_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_transcriber("dummy_test_transcriber", &spec)
            .expect("dummy transcriber is registered");
        assert_eq!(t.descriptor().id, "dummy_test_transcriber");
    }

    #[test]
    fn unknown_transcriber_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_transcriber("no_such_transcriber", &spec)
            .is_err());
    }

    #[test]
    fn dummy_transcriber_appears_in_iteration() {
        assert!(dummy_registry()
            .transcribers()
            .any(|r| (r.descriptor)().id == "dummy_test_transcriber"));
    }

    struct DummyTextEmbedder {
        desc: TextEmbedderDescriptor,
    }

    impl TextEmbedder for DummyTextEmbedder {
        fn descriptor(&self) -> &TextEmbedderDescriptor {
            &self.desc
        }

        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32, 1.0])
        }
    }

    fn dummy_text_embedder_descriptor() -> TextEmbedderDescriptor {
        TextEmbedderDescriptor {
            id: "dummy_test_text_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_text_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn TextEmbedder>> {
        Ok(Box::new(DummyTextEmbedder {
            desc: dummy_text_embedder_descriptor(),
        }))
    }

    crate::register_text_embedder! {
        const DUMMY_TEXT_EMBEDDER_REGISTRATION =
            dummy_text_embedder_descriptor => dummy_text_embedder_load
    }

    #[test]
    fn text_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_text_embedder("dummy_test_text_embedder", &spec)
            .expect("dummy text embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_text_embedder");
        assert_eq!(e.embed_text("clip").unwrap(), vec![4.0, 1.0]);
        assert_eq!(
            e.embed_text_batch(&["a", "abcd"]).unwrap(),
            vec![vec![1.0, 1.0], vec![4.0, 1.0]]
        );
    }

    #[test]
    fn unknown_text_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_text_embedder("no_such_text_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_text_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .text_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_text_embedder"));
    }

    struct DummyImageEmbedder {
        desc: ImageEmbedderDescriptor,
    }

    impl ImageEmbedder for DummyImageEmbedder {
        fn descriptor(&self) -> &ImageEmbedderDescriptor {
            &self.desc
        }

        fn embed(&self, image: &Image) -> Result<Vec<f32>> {
            Ok(vec![image.width as f32, image.height as f32])
        }
    }

    fn dummy_image_embedder_descriptor() -> ImageEmbedderDescriptor {
        ImageEmbedderDescriptor {
            id: "dummy_test_image_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_image_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn ImageEmbedder>> {
        Ok(Box::new(DummyImageEmbedder {
            desc: dummy_image_embedder_descriptor(),
        }))
    }

    crate::register_image_embedder! {
        const DUMMY_IMAGE_EMBEDDER_REGISTRATION =
            dummy_image_embedder_descriptor => dummy_image_embedder_load
    }

    #[test]
    fn image_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_image_embedder("dummy_test_image_embedder", &spec)
            .expect("dummy image embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_image_embedder");
        let img = Image {
            width: 7,
            height: 3,
            pixels: vec![0; 7 * 3 * 3],
        };
        assert_eq!(e.embed(&img).unwrap(), vec![7.0, 3.0]);
    }

    #[test]
    fn unknown_image_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_image_embedder("no_such_image_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_image_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .image_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_image_embedder"));
    }

    // ---- audio transforms (sc-12839) --------------------------------------------------------

    /// A weights-free stub audio transform: `apply` returns `out_tracks` copies of the source clip
    /// (one for the single-output kinds, one per stem for separation), retargeted to the request's
    /// sample rate — enough to exercise register→resolve→apply for all three shapes without a tensor
    /// backend.
    struct DummyAudioTransform {
        desc: AudioTransformDescriptor,
        out_tracks: usize,
    }

    impl AudioTransform for DummyAudioTransform {
        fn descriptor(&self) -> &AudioTransformDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &AudioTransformRequest) -> Result<()> {
            Ok(())
        }
        fn apply(
            &self,
            req: &AudioTransformRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<Vec<AudioTrack>> {
            let rate = match req.target {
                AudioTarget::Preserve => req.audio.sample_rate,
                AudioTarget::SampleRate(r) => r,
            };
            Ok(vec![
                AudioTrack {
                    sample_rate: rate,
                    ..req.audio.clone()
                };
                self.out_tracks
            ])
        }
    }

    fn dummy_voice_conversion_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_voice_conversion",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::VoiceConversion,
                ..Default::default()
            },
        }
    }

    fn dummy_stem_separation_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_stem_separation",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::StemSeparation,
                stem_count: 4,
                ..Default::default()
            },
        }
    }

    fn dummy_super_resolution_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_super_resolution",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::SuperResolution,
                is_diffusion: true,
                supports_resample: true,
                ..Default::default()
            },
        }
    }

    fn dummy_voice_conversion_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_voice_conversion_descriptor(),
            out_tracks: 1,
        }))
    }

    fn dummy_stem_separation_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_stem_separation_descriptor(),
            out_tracks: 4,
        }))
    }

    fn dummy_super_resolution_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_super_resolution_descriptor(),
            out_tracks: 1,
        }))
    }

    crate::register_audio_transform! {
        const DUMMY_VOICE_CONVERSION_REGISTRATION =
            dummy_voice_conversion_descriptor => dummy_voice_conversion_load
    }
    crate::register_audio_transform! {
        const DUMMY_STEM_SEPARATION_REGISTRATION =
            dummy_stem_separation_descriptor => dummy_stem_separation_load
    }
    crate::register_audio_transform! {
        const DUMMY_SUPER_RESOLUTION_REGISTRATION =
            dummy_super_resolution_descriptor => dummy_super_resolution_load
    }

    fn audio_transform_registry() -> ProviderRegistry {
        ProviderRegistryBuilder::new()
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .register_audio_transform(DUMMY_STEM_SEPARATION_REGISTRATION)
            .register_audio_transform(DUMMY_SUPER_RESOLUTION_REGISTRATION)
            .build()
            .unwrap()
    }

    #[test]
    fn audio_transform_registry_resolves_by_id() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_audio_transform("dummy_voice_conversion", &spec)
            .expect("dummy voice conversion is registered");
        assert_eq!(t.descriptor().id, "dummy_voice_conversion");
    }

    #[test]
    fn unknown_audio_transform_id_errors() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_audio_transform("no_such_audio_transform", &spec)
            .is_err());
    }

    #[test]
    fn audio_transforms_appear_in_iteration() {
        let registry = audio_transform_registry();
        assert_eq!(registry.audio_transforms().len(), 3);
        assert!(registry
            .audio_transforms()
            .any(|r| (r.descriptor)().id == "dummy_stem_separation"));
    }

    #[test]
    fn duplicate_audio_transform_id_is_rejected() {
        let err = ProviderRegistryBuilder::new()
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .build()
            .err()
            .expect("duplicate audio transform id must fail");
        assert_eq!(
            err.to_string(),
            "duplicate audio transform id 'dummy_voice_conversion' in explicit registry"
        );
    }

    #[test]
    fn audio_transform_descriptors_pass_conformance() {
        assert!(audio_transform_registry()
            .descriptor_conformance_errors()
            .is_empty());
    }

    /// The sc-12839 acceptance path end to end, weights-free: register the three non-prompt
    /// audio→audio shapes, then resolve + `apply` each and assert the output shape.
    #[test]
    fn all_three_audio_transform_shapes_resolve_and_apply() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let source = dummy_audio_track(1024, 16_000);

        // audio→audio: voice conversion, rate-preserving, single output.
        let vc = registry
            .load_audio_transform("dummy_voice_conversion", &spec)
            .unwrap();
        let converted = vc
            .apply(
                &AudioTransformRequest {
                    audio: source.clone(),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].sample_rate, 16_000);

        // audio→Vec<audio>: stem separation, multi output.
        let stems = registry
            .load_audio_transform("dummy_stem_separation", &spec)
            .unwrap();
        let separated = stems
            .apply(
                &AudioTransformRequest {
                    audio: source.clone(),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(separated.len(), 4);
        assert_eq!(
            stems.descriptor().capabilities.stem_count as usize,
            separated.len()
        );

        // audio→audio: super-resolution / bandwidth extension to a higher rate, single output.
        let sr = registry
            .load_audio_transform("dummy_super_resolution", &spec)
            .unwrap();
        let restored = sr
            .apply(
                &AudioTransformRequest {
                    audio: source,
                    target: AudioTarget::SampleRate(48_000),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].sample_rate, 48_000);
    }

    fn dummy_audio_track(samples: usize, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    // Malformed audio-transform descriptors exercising the kind/stem_count coherence guard's
    // *rejection* branches (an inverted condition would otherwise pass the whole suite, since the
    // positive test only asserts an empty sweep). The `load` fn is never invoked by the sweep.
    fn bad_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_voice_conversion_descriptor(),
            out_tracks: 1,
        }))
    }

    fn separator_stem_count(id: &'static str, stem_count: u16) -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id,
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::StemSeparation,
                stem_count,
                ..Default::default()
            },
        }
    }
    fn single_output_stem_count(
        id: &'static str,
        kind: AudioTransformKind,
    ) -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id,
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind,
                stem_count: 3,
                ..Default::default()
            },
        }
    }

    fn bad_stems_zero_descriptor() -> AudioTransformDescriptor {
        separator_stem_count("bad_stems_zero", 0)
    }
    fn bad_stems_one_descriptor() -> AudioTransformDescriptor {
        separator_stem_count("bad_stems_one", 1)
    }
    fn bad_vc_stems_descriptor() -> AudioTransformDescriptor {
        single_output_stem_count("bad_vc_stems", AudioTransformKind::VoiceConversion)
    }
    fn bad_sr_stems_descriptor() -> AudioTransformDescriptor {
        single_output_stem_count("bad_sr_stems", AudioTransformKind::SuperResolution)
    }

    crate::register_audio_transform! {
        const BAD_STEMS_ZERO_REGISTRATION = bad_stems_zero_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_STEMS_ONE_REGISTRATION = bad_stems_one_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_VC_STEMS_REGISTRATION = bad_vc_stems_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_SR_STEMS_REGISTRATION = bad_sr_stems_descriptor => bad_load
    }

    #[test]
    fn audio_transform_kind_stem_count_incoherence_is_rejected() {
        let errs = ProviderRegistryBuilder::new()
            .register_audio_transform(BAD_STEMS_ZERO_REGISTRATION)
            .register_audio_transform(BAD_STEMS_ONE_REGISTRATION)
            .register_audio_transform(BAD_VC_STEMS_REGISTRATION)
            .register_audio_transform(BAD_SR_STEMS_REGISTRATION)
            .build()
            .unwrap()
            .descriptor_conformance_errors();
        let has = |needle: &str| errs.iter().any(|e| e.contains(needle));

        // A separator advertising < 2 stems (0 and 1) is rejected with the specific message.
        assert!(
            has("audio transform 'bad_stems_zero': StemSeparation advertises stem_count 0 (a separator must produce ≥ 2 stems)"),
            "{errs:?}"
        );
        assert!(
            has("audio transform 'bad_stems_one': StemSeparation advertises stem_count 1 (a separator must produce ≥ 2 stems)"),
            "{errs:?}"
        );
        // A single-output kind advertising any stems is rejected.
        assert!(
            has("audio transform 'bad_vc_stems': VoiceConversion advertises stem_count 3 — only StemSeparation produces stems"),
            "{errs:?}"
        );
        assert!(
            has("audio transform 'bad_sr_stems': SuperResolution advertises stem_count 3 — only StemSeparation produces stems"),
            "{errs:?}"
        );
    }

    struct DummyVoiceEmbedder {
        desc: VoiceEmbedderDescriptor,
    }

    impl VoiceEmbedder for DummyVoiceEmbedder {
        fn descriptor(&self) -> &VoiceEmbedderDescriptor {
            &self.desc
        }

        fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding> {
            Ok(vec![audio.samples.len() as f32; self.desc.embedding_dim])
        }
    }

    fn dummy_voice_embedder_descriptor() -> VoiceEmbedderDescriptor {
        VoiceEmbedderDescriptor {
            id: "dummy_test_voice_embedder",
            family: "voice",
            backend: "candle",
            embedding_dim: 4,
            mac_only: false,
        }
    }

    fn dummy_voice_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn VoiceEmbedder>> {
        Ok(Box::new(DummyVoiceEmbedder {
            desc: dummy_voice_embedder_descriptor(),
        }))
    }

    crate::register_voice_embedder! {
        const DUMMY_VOICE_EMBEDDER_REGISTRATION =
            dummy_voice_embedder_descriptor => dummy_voice_embedder_load
    }

    fn dummy_audio(samples: usize) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn voice_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_voice_embedder("dummy_test_voice_embedder", &spec)
            .expect("dummy voice embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_voice_embedder");
        assert_eq!(e.embed(&dummy_audio(3)).unwrap(), vec![3.0; 4]);
    }

    #[test]
    fn unknown_voice_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_voice_embedder("no_such_voice_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_voice_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .voice_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_voice_embedder"));
    }

    // ---- audio embedders (sc-12851) ---------------------------------------------------------

    /// A weights-free joint audio-text embedder: both `embed` and `embed_text` return a **one-hot**
    /// unit vector, keyed off the audio clip's length (a proxy for its semantic "category") and off
    /// the text's length respectively. That is deliberately enough structure to drive the DoD
    /// cross-modal ranking test without a tensor backend: a text query lands on the one-hot index of
    /// exactly one clip, so cosine ranks that clip first — and the test FAILS if the stub ignored the
    /// audio (every clip identical) or if the audio/text vectors were not the same length (not joint).
    struct DummyAudioEmbedder {
        desc: AudioEmbedderDescriptor,
    }

    impl DummyAudioEmbedder {
        fn one_hot(&self, index: usize) -> Vec<f32> {
            let mut v = vec![0.0; self.desc.embedding_dim];
            v[index % self.desc.embedding_dim] = 1.0;
            v
        }
    }

    impl AudioEmbedder for DummyAudioEmbedder {
        fn descriptor(&self) -> &AudioEmbedderDescriptor {
            &self.desc
        }
        fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>> {
            Ok(self.one_hot(audio.samples.len()))
        }
        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.one_hot(text.len()))
        }
    }

    fn dummy_audio_embedder_descriptor() -> AudioEmbedderDescriptor {
        AudioEmbedderDescriptor {
            id: "dummy_test_audio_embedder",
            family: "audio-embed",
            backend: "candle",
            embedding_dim: 4,
            space: "test-space",
            mac_only: false,
        }
    }

    fn dummy_audio_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn AudioEmbedder>> {
        Ok(Box::new(DummyAudioEmbedder {
            desc: dummy_audio_embedder_descriptor(),
        }))
    }

    crate::register_audio_embedder! {
        const DUMMY_AUDIO_EMBEDDER_REGISTRATION =
            dummy_audio_embedder_descriptor => dummy_audio_embedder_load
    }

    #[test]
    fn audio_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_audio_embedder("dummy_test_audio_embedder", &spec)
            .expect("dummy audio embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_audio_embedder");
        // audio and text embeddings share the joint dim.
        assert_eq!(e.embed(&dummy_audio(1)).unwrap().len(), 4);
        assert_eq!(e.embed_text("q").unwrap().len(), 4);
    }

    #[test]
    fn unknown_audio_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_audio_embedder("no_such_audio_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_audio_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .audio_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_audio_embedder"));
    }

    /// The sc-12851 acceptance path end to end, weights-free: resolve the joint embedder, embed a
    /// SET of audio clips spanning "categories", embed a TEXT query, and assert the semantically
    /// matching clip ranks HIGHEST by cosine over the others — the cross-modal retrieval the DoD
    /// pins. Designed to fail if the embedder ignored the audio (all clips equidistant) or the audio
    /// and text vectors were not in one joint space.
    #[test]
    fn audio_text_query_ranks_the_matching_clip_highest() {
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }

        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_audio_embedder("dummy_test_audio_embedder", &spec)
            .expect("registered");

        // Three clips whose lengths map to distinct one-hot categories (1, 2, 3).
        let clips = [dummy_audio(1), dummy_audio(2), dummy_audio(3)];
        let clip_vecs: Vec<Vec<f32>> = clips.iter().map(|c| e.embed(c).unwrap()).collect();

        // A 3-char query lands on category 3 → the third clip is the match.
        let query = e.embed_text("abc").unwrap();
        let scores: Vec<f32> = clip_vecs.iter().map(|v| cosine(&query, v)).collect();

        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            best, 2,
            "the matching clip must rank first, scores={scores:?}"
        );
        assert!(scores[2] > scores[0] && scores[2] > scores[1], "{scores:?}");
    }

    /// The sc-12838 acceptance path end to end, weights-free: resolve a registered voice embedder,
    /// drive `embed()` over a reference clip, then feed the resulting embedding into a stub audio
    /// [`Generator`]'s conditioning (`Conditioning::VoiceEmbedding`) and validate — a cloned voice
    /// driving TTS, the audio mirror of a face embedding conditioning InstantID/PuLID.
    #[test]
    fn voice_embedding_resolves_embeds_and_conditions_a_generator() {
        use crate::generator::{Conditioning, ConditioningKind};

        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));

        // resolve → embed
        let embedder = registry
            .load_voice_embedder("dummy_test_voice_embedder", &spec)
            .expect("registered");
        let embedding = embedder.embed(&dummy_audio(5)).unwrap();
        assert_eq!(embedding.len(), 4);

        // A stub audio generator that advertises VoiceEmbedding conditioning.
        let tts = DummyGen {
            desc: ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: None,
                control_kinds: None,
                required_components: &[],
                id: "dummy_tts",
                family: "test",
                backend: "mlx",
                modality: Modality::Audio,
                capabilities: Capabilities {
                    max_count: 1,
                    conditioning: vec![ConditioningKind::VoiceEmbedding],
                    audio_sample_rates: vec![24_000],
                    ..Default::default()
                },
            },
        };

        // feed the embedding into the generator's conditioning and validate (size-skipping audio floor)
        let req = GenerationRequest {
            conditioning: vec![Conditioning::VoiceEmbedding {
                embedding,
                strength: None,
            }],
            steps: Some(1),
            ..Default::default()
        };
        tts.descriptor()
            .capabilities
            .validate_request_audio("dummy_tts", &req)
            .expect("a cloned-voice embedding is accepted conditioning for the TTS generator");

        // A generator that does NOT advertise it rejects the same conditioning.
        let no_voice = Capabilities {
            max_count: 1,
            audio_sample_rates: vec![24_000],
            ..Default::default()
        };
        assert!(no_voice.validate_request_audio("dummy_tts", &req).is_err());
    }

    /// The sweep (sc-9098, F-009) is clean over the explicit dummy catalog.
    #[test]
    fn descriptor_sweep_is_clean_over_dummy_catalog() {
        let errs = dummy_registry().descriptor_conformance_errors();
        assert!(
            errs.is_empty(),
            "descriptor conformance FAILED:\n  - {}",
            errs.join("\n  - ")
        );
    }

    #[test]
    fn activation_memory_anchor_selection_is_provider_route_wide() {
        const REGISTRATION: ActivationMemoryRegistration = ActivationMemoryRegistration {
            provider_id: "dummy_test_model",
            anchor: ActivationMemoryAnchor { bytes_1024: 8 },
        };
        let registry = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_activation_memory(REGISTRATION)
            .build()
            .unwrap();
        assert_eq!(
            registry
                .activation_memory_bytes_1024("dummy_test_model")
                .unwrap(),
            Some(8)
        );
    }

    #[test]
    fn activation_memory_registration_fails_closed_and_preserves_unmeasured_state() {
        const REGISTRATION: ActivationMemoryRegistration = ActivationMemoryRegistration {
            provider_id: "dummy_test_model",
            anchor: ActivationMemoryAnchor { bytes_1024: 8 },
        };
        let duplicate = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_activation_memory(REGISTRATION)
            .register_activation_memory(REGISTRATION)
            .build()
            .err()
            .expect("duplicate activation registration must fail")
            .to_string();
        assert!(duplicate.contains("duplicate activation-memory provider id"));

        let orphan = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_activation_memory(ActivationMemoryRegistration {
                provider_id: "no_such_model",
                anchor: ActivationMemoryAnchor { bytes_1024: 8 },
            })
            .build()
            .err()
            .expect("orphan activation registration must fail")
            .to_string();
        assert!(orphan.contains("has no matching generator"));

        let zero = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_activation_memory(ActivationMemoryRegistration {
                provider_id: "dummy_test_model",
                anchor: ActivationMemoryAnchor { bytes_1024: 0 },
            })
            .build()
            .err()
            .expect("zero activation registration must fail")
            .to_string();
        assert!(zero.contains("is zero"));

        let unmeasured = ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .build()
            .unwrap();
        assert_eq!(
            unmeasured
                .activation_memory_bytes_1024("dummy_test_model")
                .unwrap(),
            None
        );
        assert!(unmeasured
            .activation_memory_bytes_1024("no_such_model")
            .is_err());
    }

    /// sc-19559 — an unsatisfiable step declaration is a descriptor mistake, not a runtime
    /// surprise. Each shape refuses EVERY step count, and the shared floor would enforce it on
    /// every request, so the sweep has to name it before a catalog ships.
    ///
    /// Each case is built and asserted individually: a single descriptor carrying all of them at
    /// once would pass even if only one check fired.
    #[test]
    fn model_descriptor_errors_flags_an_unsatisfiable_step_declaration() {
        let with = |steps: StepSupport| ModelDescriptor {
            capabilities: Capabilities {
                supported_steps: steps,
                ..dummy_descriptor().capabilities
            },
            ..dummy_descriptor()
        };
        let errs = |steps: StepSupport| model_descriptor_errors(&with(steps));

        // The baseline: the coherent descriptor this varies from is clean, so every message below
        // is attributable to the step declaration alone.
        assert!(errs(StepSupport::Unconstrained).is_empty());
        assert!(errs(StepSupport::Exact(vec![8])).is_empty());
        assert!(errs(StepSupport::Range { min: 1, max: 200 }).is_empty());

        let empty_menu = errs(StepSupport::Exact(Vec::new()));
        assert!(
            empty_menu.iter().any(|e| e.contains("EMPTY exact menu")),
            "{empty_menu:?}"
        );
        let zero_menu = errs(StepSupport::Exact(vec![0, 8]));
        assert!(
            zero_menu.iter().any(|e| e.contains("advertises 0 steps")),
            "{zero_menu:?}"
        );
        let dupe_menu = errs(StepSupport::Exact(vec![8, 8]));
        assert!(
            dupe_menu
                .iter()
                .any(|e| e.contains("duplicate supported step count 8")),
            "{dupe_menu:?}"
        );
        let zero_range = errs(StepSupport::Range { min: 0, max: 200 });
        assert!(
            zero_range.iter().any(|e| e.contains("range starts at 0")),
            "{zero_range:?}"
        );
        let inverted = errs(StepSupport::Range { min: 30, max: 8 });
        assert!(
            inverted.iter().any(|e| e.contains("30..=8 is empty")),
            "{inverted:?}"
        );
    }

    /// Each per-descriptor invariant fires: identity shape, zero/inverted bounds, duplicate or
    /// malformed curated names, duplicate conditioning, video conditioning on an Image model.
    #[test]
    fn model_descriptor_errors_flags_each_violation() {
        // A fully-coherent descriptor produces no errors.
        assert!(model_descriptor_errors(&dummy_descriptor()).is_empty());

        let broken = ModelDescriptor {
            encoder_contract: Some(crate::EncoderContract {
                architecture: "qwen3",
                hidden_size: 8,
                intermediate_size: 12,
                num_hidden_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: 3,
                head_dim: 4,
                vocab_size: 16,
                output_width: 8,
                loaded_hidden_layers: 2,
                requires_final_norm: true,
                requires_lm_head: false,
                hidden_activation: "silu",
                attention_dropout: crate::EncoderConfigFloat::new(0.0),
                rms_norm_eps: crate::EncoderConfigFloat::new(1e-6),
                qk_norm_eps: Some(crate::EncoderConfigFloat::new(1e-6)),
                rope_theta: crate::EncoderConfigFloat::new(1_000_000.0),
                max_position_embeddings: 4_096,
                attention_bias: crate::EncoderConfigBool::Required(false),
                tie_word_embeddings: crate::EncoderConfigBool::Required(true),
                tokenizer: crate::EncoderTokenizerContract {
                    family: "test-qwen3",
                    binding: crate::EncoderTokenizerBinding::RetainBase,
                    artifact_candidates: &["tokenizer/tokenizer.json"],
                    required_tokens: &[],
                },
                prompt_executions: &[crate::EncoderPromptExecutionContract {
                    purpose: "test",
                    template: crate::EncoderPromptTemplate::QwenInstruct,
                    add_special_tokens: true,
                    length: crate::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 8 },
                    padding: crate::EncoderPromptPadding::None,
                    prefix_trim: 0,
                }],
                bos_token_id: None,
                eos_token_id: None,
                image_token_id: None,
                vision_start_token_id: None,
                vision_end_token_id: None,
                mrope_section: &[],
                mrope_interleaved: None,
                selected_hidden_layers: &[2],
                packing: None,
                dense_storage_dtype_probe: None,
            }),
            denoiser_output_latent_space: None,
            control_kinds: None,
            // Blank + duplicate required-component ids (sc-13658) — unstageable / ambiguous keys.
            required_components: &["", "voice_embedding", "voice_embedding"],
            id: "Bad Id", // uppercase + whitespace
            family: "",   // empty
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                min_size: 512,
                max_size: 256, // inverted
                max_count: 0,  // zero
                size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 0 },
                samplers: vec!["euler", "euler", "bad name"], // duplicate + whitespace
                conditioning: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::Reference, // duplicate
                    ConditioningKind::VideoClip, // video kind on an Image model
                ],
                ..Default::default()
            },
        };
        let errs = model_descriptor_errors(&broken);
        let has = |needle: &str| errs.iter().any(|e| e.contains(needle));
        assert!(has("id \"Bad Id\""), "{errs:?}");
        assert!(has("family \"\""), "{errs:?}");
        assert!(has("invalid text encoder contract"), "{errs:?}");
        assert!(has("max_count is 0"), "{errs:?}");
        assert!(has("min_size 512 > max_size 256"), "{errs:?}");
        assert!(has("explicit-size multiple is 0"), "{errs:?}");
        assert!(has("duplicate sampler entry \"euler\""), "{errs:?}");
        assert!(has("sampler[2] \"bad name\""), "{errs:?}");
        assert!(has("duplicate conditioning kind Reference"), "{errs:?}");
        assert!(has("video conditioning VideoClip"), "{errs:?}");
        // The required-component id list is validated like the curated name lists: a blank id and a
        // duplicate id are both flagged (sc-13658).
        assert!(has("required_component[0] \"\""), "{errs:?}");
        assert!(
            has("duplicate required_component entry \"voice_embedding\""),
            "{errs:?}"
        );

        // All-zero bounds report the Default-0 message (F-084), not the inverted-bounds one.
        let zeroed = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "zeroed",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
        };
        assert!(model_descriptor_errors(&zeroed)
            .iter()
            .any(|e| e.contains("left at the Default 0")));
    }

    /// The multi-turn path-A cross-check (sc-14150): `supports_conversation_history` without the
    /// matching `ConditioningKind::ConversationHistory` in `conditioning` is flagged (the "flag on,
    /// kind missing" footgun where every path-A request would be rejected by the allowlist), and a
    /// descriptor that wires both is conformant.
    #[test]
    fn model_descriptor_errors_flags_conversation_history_flag_without_kind() {
        let half_wired = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "convo",
            family: "test",
            backend: "candle",
            modality: Modality::Audio,
            capabilities: Capabilities {
                max_count: 1,
                supports_conversation_history: true,
                // ConditioningKind::ConversationHistory deliberately NOT advertised.
                ..Default::default()
            },
        };
        assert!(model_descriptor_errors(&half_wired)
            .iter()
            .any(|e| e.contains("supports_conversation_history is set but")));

        // Wiring both the flag and the kind is conformant.
        let wired = ModelDescriptor {
            capabilities: Capabilities {
                max_count: 1,
                supports_conversation_history: true,
                conditioning: vec![ConditioningKind::ConversationHistory],
                ..Default::default()
            },
            ..half_wired
        };
        assert!(model_descriptor_errors(&wired).is_empty());
    }

    /// The size-bounds floor is exempt for `Modality::Audio` (sc-13314): a pure-audio generator has
    /// no width/height, so `min_size`/`max_size` left at the unused `Default` 0 must NOT be flagged —
    /// mirroring the size-skipping `validate_request_audio` floor. The exemption is scoped to the
    /// size axis: every other invariant (identity, `max_count`, curated-name shape) still fires for an
    /// audio descriptor.
    #[test]
    fn audio_descriptor_with_zero_size_bounds_passes_sweep() {
        let audio = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "zeroed_audio",
            family: "test",
            backend: "candle",
            modality: Modality::Audio,
            capabilities: Capabilities {
                // No spatial dims — bounds stay at the natural unused 0.
                min_size: 0,
                max_size: 0,
                max_count: 1,
                audio_sample_rates: vec![24_000],
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&audio).is_empty(),
            "an audio descriptor with unused (0) size bounds must pass the sweep: {:?}",
            model_descriptor_errors(&audio)
        );

        // The exemption is only the size axis: a broken `max_count` on the same audio descriptor is
        // still reported.
        let audio_bad_count = ModelDescriptor {
            required_components: &[],
            capabilities: Capabilities {
                max_count: 0,
                ..audio.capabilities.clone()
            },
            ..audio.clone()
        };
        assert!(
            model_descriptor_errors(&audio_bad_count)
                .iter()
                .any(|e| e.contains("max_count is 0")),
            "the audio exemption must not relax the non-size invariants"
        );
    }

    /// The strictness the sweep exists for is preserved for the visual modalities: an `Image`/`Video`
    /// descriptor with invalid size bounds is STILL flagged (the audio exemption above does not weaken
    /// the check for modalities that genuinely carry a spatial size range).
    #[test]
    fn visual_descriptor_with_invalid_size_bounds_still_fails_sweep() {
        // Video, zero bounds → the Default-0 footgun still fires.
        let video_zero = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "video_zero",
            family: "test",
            backend: "mlx",
            modality: Modality::Video,
            capabilities: Capabilities {
                min_size: 0,
                max_size: 0,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&video_zero)
                .iter()
                .any(|e| e.contains("left at the Default 0")),
            "a Video descriptor with zero size bounds must still be rejected"
        );

        // Image, inverted bounds → the inverted-bounds message still fires.
        let image_inverted = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "image_inverted",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                min_size: 512,
                max_size: 256,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&image_inverted)
                .iter()
                .any(|e| e.contains("min_size 512 > max_size 256")),
            "an Image descriptor with inverted size bounds must still be rejected"
        );

        // `Both` (emits image or video) is a visual modality too — zero bounds still fail.
        let both_zero = ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
            id: "both_zero",
            family: "test",
            backend: "mlx",
            modality: Modality::Both,
            capabilities: Capabilities {
                min_size: 0,
                max_size: 0,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&both_zero)
                .iter()
                .any(|e| e.contains("left at the Default 0")),
            "a Both-modality descriptor with zero size bounds must still be rejected"
        );
    }

    /// Build a synthetic diffusers-style snapshot with a `bytes`-sized `model.safetensors` under each
    /// named subdir.
    ///
    /// Returns the `TempDir` guard, not a bare `PathBuf` (sc-17755): the tree leaves on `Drop`,
    /// including out of a panicking test, which the trailing `remove_dir_all(..).ok()` lines this
    /// replaced could not do. Callers bind it for the whole test and read `tmp.path()`.
    fn synthetic_snapshot(tag: &str, subdirs: &[(&str, usize)]) -> tempfile::TempDir {
        let tmp = tempfile::Builder::new()
            .prefix(&format!("gencore_footprint_{tag}_"))
            .tempdir()
            .expect("fixture temp dir");
        let root = tmp.path();
        for (sub, bytes) in subdirs {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; *bytes]).unwrap();
        }
        tmp
    }

    /// Guards the sc-17755 fix for [`synthetic_snapshot`]: the assembled tree leaves with the guard.
    /// Revert it to a bare `create_dir_all` on a `temp_dir()` join and this goes RED.
    #[test]
    fn synthetic_snapshot_is_removed_on_drop() {
        let (root, shard) = {
            let tmp = synthetic_snapshot("drop-guard", &[("transformer", 16)]);
            let root = tmp.path().to_path_buf();
            let shard = root.join("transformer/model.safetensors");
            assert!(shard.is_file(), "snapshot shard not written");
            (root, shard)
        };
        assert!(!shard.exists(), "shard survived: {}", shard.display());
        assert!(!root.exists(), "snapshot root survived: {}", root.display());
    }

    /// sc-10894: a provider that declared a footprint returns the per-component on-disk split, resolved
    /// from the exact subdirs its loader uses — including a text encoder under a NON-`text_encoder`
    /// subdir (`mllm/`, the boogu layout) that a name-guessing consumer would read as zero.
    #[test]
    fn footprint_returns_provider_component_split() {
        let root = synthetic_snapshot(
            "split",
            &[("mllm", 1500), ("transformer", 9000), ("vae", 400)],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));

        let fp = dummy_registry()
            .footprint("dummy_footprint_model", &spec)
            .expect("registered + declares a footprint")
            .expect("Some — the provider computed the split");
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 1500,
                dit: 9000,
                vae: 400,
            }
        );
        // The whole point: the text encoder is NON-zero even though it is not under `text_encoder*`.
        assert!(fp.text_encoder > 0, "mllm/ text encoder must be measured");
    }

    /// A registered generator that declares NO footprint yields `Ok(None)` (the consumer falls back);
    /// an unknown id is an `Err`.
    #[test]
    fn footprint_is_none_without_declaration_and_errs_on_unknown_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // `dummy_test_model` is registered but declares no footprint.
        let registry = dummy_registry();
        assert_eq!(registry.footprint("dummy_test_model", &spec).unwrap(), None);
        // Unknown id → Err (a fail-open consumer treats it like None).
        assert!(registry.footprint("no_such_model", &spec).is_err());
    }

    /// sc-10894: `from_spec_subdirs` sums each component's subdir(s) (SD3's three text encoders here),
    /// treats a missing subdir as `0`, and errors on a single-`File` source (no tree to split).
    #[test]
    fn per_component_bytes_from_spec_subdirs_and_file_guard() {
        let root = synthetic_snapshot(
            "sd3",
            &[
                ("text_encoder", 100),
                ("text_encoder_2", 200),
                ("text_encoder_3", 4000),
                ("transformer", 8000),
                ("vae", 300),
            ],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
        let fp = PerComponentBytes::from_spec_subdirs(
            &spec,
            &["text_encoder", "text_encoder_2", "text_encoder_3"],
            &["transformer"],
            &["vae"],
        )
        .unwrap();
        assert_eq!(fp.text_encoder, 4300); // 100 + 200 + 4000
        assert_eq!(fp.dit, 8000);
        assert_eq!(fp.vae, 300);

        // A named-but-absent subdir contributes 0 (does not error).
        let fp_missing =
            PerComponentBytes::from_spec_subdirs(&spec, &["nope"], &["transformer"], &["vae"])
                .unwrap();
        assert_eq!(fp_missing.text_encoder, 0);

        // A single-file source has no component tree → Err (consumer falls back to whole-file).
        let file_spec = LoadSpec::new(WeightsSource::File(
            root.path().join("transformer/model.safetensors"),
        ));
        assert!(
            PerComponentBytes::from_spec_subdirs(&file_spec, &["te"], &["dit"], &["vae"]).is_err()
        );
    }

    /// sc-10894: `from_root_subdirs` sums a component named by a flat FILE (the bernini/anima layout —
    /// `t5_encoder.safetensors` at the root, not a `text_encoder/` subdir) as well as a subdir, against
    /// an already-resolved root.
    #[test]
    fn per_component_bytes_from_root_subdirs_handles_flat_files() {
        let tmp = tempfile::Builder::new()
            .prefix("gencore_footprint_flat_")
            .tempdir()
            .expect("fixture temp dir");
        let root = tmp.path();
        // bernini-style flat component files at the root.
        std::fs::write(root.join("t5_encoder.safetensors"), vec![0u8; 2000]).unwrap();
        std::fs::write(root.join("low_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("high_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("vae.safetensors"), vec![0u8; 500]).unwrap();

        let fp = PerComponentBytes::from_root_subdirs(
            root,
            &["t5_encoder.safetensors"],
            &[
                "low_noise_model.safetensors",
                "high_noise_model.safetensors",
            ],
            &["vae.safetensors"],
        );
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 2000,
                dit: 12000,
                vae: 500,
            }
        );
    }
}
