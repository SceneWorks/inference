//! The published checkpoint-adapter registrations (Krea 2, SDXL, Mage, Z-Image, Qwen-Image, FLUX.2, Wan).

use super::*;

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
