//! Explicit model + transform discovery: provider crates publish registration constants, family
//! crates add them to a [`ProviderRegistryBuilder`], and platform catalogs select the families they
//! ship. This is the Rust equivalent of an ordinary DI composition root with resolve-by-id.

use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
use crate::audio_transform::{AudioTransform, AudioTransformDescriptor, AudioTransformKind};
use crate::caption::{Captioner, CaptionerDescriptor};
use crate::checkpoint_codec::{CheckpointCodecRegistration, CheckpointCodecRegistry};
use crate::generator::{ConditioningKind, Generator, Modality, ModelDescriptor, StepSupport};
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
    signatures: &[
        CheckpointSignatureRegistration {
            id: "flux2-comfyui-v1",
            dialect: "comfyui",
            required_tensor_names: &["model.diffusion_model.double_blocks.0.img_attn.qkv.weight"],
        },
        // The bare-keyed FLUX.2 Klein transformer single file (sc-21485).
        //
        // A signature is an ALL-PRESENT name list, so it can only assert what a file *has* — it
        // cannot express "and no `guidance_in.*`". The discrimination therefore has to come from a
        // name only klein spells: the community klein export writes its per-head norms as
        // `...norm.query_norm.weight`, while the BFL-official FLUX.2-**dev** export spells the same
        // tensor `...norm.query_norm.scale` (see `candle-gen-flux2::single_file`, whose mapping
        // refuses the `.scale` spelling). The first two names alone match a bare-keyed dev file
        // too, which would import a 32B dev checkpoint as `flux2_klein_9b` and only fail later, at
        // plan time.
        //
        // So: the qkv name pins the fused-attention topology, the AdaLN name pins the final-layer
        // topology the half-swap transform is declared for, and the per-head-norm name is what
        // makes this klein rather than dev.
        CheckpointSignatureRegistration {
            id: "flux2-klein-bfl-v1",
            dialect: "bfl",
            required_tensor_names: &[
                "double_blocks.0.img_attn.qkv.weight",
                "double_blocks.0.img_attn.norm.query_norm.weight",
                "final_layer.adaLN_modulation.1.weight",
            ],
        },
    ],
    dialects: &[
        CheckpointDialectRegistration {
            id: "comfyui",
            source: ImportedModelSource::ComfyUiTree,
        },
        // A standalone transformer-only `.safetensors` in the original (ComfyUI/BFL) key
        // convention — the shape `wikeeyang/Flux2-Klein-9B-True-V2` ships (sc-21485). Distinct
        // from `comfyui`: that source is a file inside a ComfyUI folder tree and binds to the
        // 32B dev provider; this one is the 9B klein community single file.
        CheckpointDialectRegistration {
            id: "bfl",
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
        compatible_families: &["flux2"],
    }],
    canonical_mappings: &[
        CheckpointCanonicalMappingRegistration {
            dialect: "comfyui",
            mapping_id: "flux2-comfyui-to-diffusers-v1",
            // Loader-native (see `SDXL_CHECKPOINT_ADAPTER`).
            plan_driven_backends: &[],
        },
        // Plan-driven on Candle (sc-21485): `candle_gen_flux2::single_file` ships the
        // `LogicalKeyMapping` with this id — renames plus the declarative fused-QKV row slices
        // and the AdaLN half swap — and the klein single-file loader plans through it.
        CheckpointCanonicalMappingRegistration {
            dialect: "bfl",
            mapping_id: "flux2-bfl-to-diffusers-v1",
            plan_driven_backends: &[CheckpointBackend::Candle],
        },
    ],
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
    Fp8Rowwise,
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
            Self::Fp8Rowwise => spec.with_quant(crate::Quant::Fp8Rowwise),
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
            MemoryContractSurfaceTier::Fp8Rowwise => {
                spec.quantize == Some(crate::Quant::Fp8Rowwise)
            }
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
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "fp8r:resident:eager",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "fp8r:resident:deferred",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "fp8r:sequential:eager",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "fp8r:sequential:deferred",
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

/// Candle CUDA surface for providers that load the publisher FP8R and NVFP4 tiers as shipped
/// (FLUX3, sc-26241) beside the dense and native q4/q8 tiers. CUDA-only formats are normal: a
/// backend that cannot run them simply does not advertise this surface.
pub fn candle_publisher_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
        MemoryContractSurfaceTier::Nvfp4,
        MemoryContractSurfaceTier::Fp8Rowwise,
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

/// Provider-owned PRE-FIT pricing of a request's reference work (sc-26241): the additional bytes
/// the resolved [`crate::MemoryReferenceResourceFacts`] need on the loaded package `spec`. Opt-in
/// and separate from [`MemoryRegistration`], like [`ActivationMemoryRegistration`]; omission is
/// the explicit unpriced state, and a request without references prices zero.
#[derive(Clone, Copy)]
pub struct ReferenceWorkspaceRegistration {
    pub provider_id: &'static str,
    pub reference_workspace_bytes:
        fn(&LoadSpec, &crate::MemoryReferenceResourceFacts) -> Result<u64>,
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
    reference_workspace: Vec<ReferenceWorkspaceRegistration>,
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
        register_reference_workspace,
        reference_workspace,
        ReferenceWorkspaceRegistration
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
            for registration in &self.reference_workspace {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate reference-workspace provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "reference-workspace registration '{}' has no matching generator",
                        registration.provider_id
                    )));
                }
            }
        }
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
            reference_workspace: self.reference_workspace.into_boxed_slice(),
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
    reference_workspace: Box<[ReferenceWorkspaceRegistration]>,
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
            self.ensure_quant_supported(id, spec, $kind)?;
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
    ///
    /// A route whose binding declares `inherit_adapters = false` has `supports_lora`/`supports_lokr`
    /// **withdrawn** from the descriptor returned here. That withdrawal is *advertisement only* —
    /// this function admits nothing and refuses nothing. Turning it into a typed refusal is the job
    /// of the **provider-build seam gate**: every provider's `build(spec, descriptor)` calls
    /// [`crate::reject_unsupported_adapters`] with the descriptor it was handed (see
    /// `candle_gen_krea`'s `build`), which is the single gate for the sc-21483 / epic-11037-E6
    /// silent-un-adapted-render class. There is deliberately no second, registry-side wrapper: a
    /// guard that no load route calls is indistinguishable from no guard at all, so a consumer that
    /// resolves a descriptor here is responsible for passing *that* descriptor — not the provider's
    /// own — into the build seam.
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

    /// Provider-owned PRE-FIT bytes of a request's reference work for generator `id` on `spec`.
    /// `facts == None` (no references) is `Ok(Some(0))` for a registered provider, so decisions
    /// without references are unchanged; an unregistered provider is `Ok(None)` (unpriced); an
    /// unknown id is an error.
    pub fn reference_workspace_bytes(
        &self,
        id: &str,
        spec: &LoadSpec,
        facts: Option<&crate::MemoryReferenceResourceFacts>,
    ) -> Result<Option<u64>> {
        if !self
            .generators()
            .any(|registration| (registration.descriptor)().id == id)
        {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        let Some(registration) = self
            .reference_workspace
            .iter()
            .find(|registration| registration.provider_id == id)
        else {
            return Ok(None);
        };
        match facts {
            None => Ok(Some(0)),
            Some(facts) => (registration.reference_workspace_bytes)(spec, facts).map(Some),
        }
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
    /// as declared by [`ProviderRegistryBuilder::reject_quant`], or whose YuE2 AR mode a different
    /// provider would silently ignore.
    ///
    /// The single boundary every registry-routed load of every provider kind passes through, so one
    /// check covers the whole catalog — the composition root states the platform's tier support once
    /// instead of each provider re-deriving it. Runs *after* id resolution so an unknown id still
    /// reports as an unknown id.
    fn ensure_quant_supported(&self, id: &str, spec: &LoadSpec, kind: &str) -> Result<()> {
        if spec.yue2_ar_mode != crate::Yue2ArMode::Native && id != "yue2" {
            return Err(Error::Unsupported(format!(
                "the YuE2 AR mode {:?} cannot be used by provider '{id}'",
                spec.yue2_ar_mode
            )));
        }
        if spec.yue2_compute_policy != crate::Yue2ComputePolicy::Legacy && id != "yue2" {
            return Err(Error::Unsupported(format!(
                "the YuE2 compute policy {:?} cannot be used by provider '{id}'",
                spec.yue2_compute_policy
            )));
        }
        let Some(quant) = spec.quantize else {
            return Ok(());
        };
        match self.rejected_quants.iter().find(|(q, _)| *q == quant) {
            Some((_, reason)) => Err(Error::Unsupported(format!(
                "quant tier {quant:?} is not implemented by this runtime's backend \
                 (requested for '{id}'): {reason}. Refusing to load rather than silently \
                 serving a different tier's numerics."
            ))),
            None => {
                // A NEW packed float format never inherits affine Q8 support. Reuse
                // the descriptor's exact typed list as the one opt-in source; no
                // parallel builder allow-list or model-id exception exists.
                if quant == Quant::Fp8Rowwise
                    && (kind != "generator"
                        || !self.generators().any(|registration| {
                            let descriptor = (registration.descriptor)();
                            descriptor.id == id
                                && descriptor.capabilities.supported_quants.contains(&quant)
                        }))
                {
                    return Err(Error::Unsupported(format!("provider '{id}' does not advertise the exact fp8r packed format (it is not Q8); install/convert a compatible artifact and select a supported native tier")));
                }
                Ok(())
            }
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
        if spec.quantize == Some(Quant::Fp8Rowwise) {
            self.ensure_quant_supported(id, spec, "generator")?;
        }
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
        if spec.quantize == Some(Quant::Fp8Rowwise) {
            self.ensure_quant_supported(id, spec, "generator")?;
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
mod tests;
