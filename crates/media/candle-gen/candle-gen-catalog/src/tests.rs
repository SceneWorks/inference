use candle_gen::gen_core::ConditioningKind;

#[test]
fn checkpoint_codecs_register_once_and_every_row_has_an_engine_implementation() {
    use candle_gen::logical_weights::{BASELINE_CODECS, CODEC_IMPLEMENTATION_IDS};

    use candle_gen::gen_core::GGUF_CONTAINER_CODEC;

    let registry = super::provider_registry().unwrap();
    let registered: Vec<_> = registry.checkpoint_codecs().codecs().copied().collect();
    // The engine's safetensors table, then the one GGUF container row the catalog adds because
    // its implementation lives in `candle-gen-wan` rather than in the engine crate. Still
    // exactly one composition point: no family `register_providers` may add a row.
    let expected: Vec<_> = BASELINE_CODECS
        .iter()
        .copied()
        .chain(std::iter::once(GGUF_CONTAINER_CODEC))
        .collect();
    assert_eq!(
        registered, expected,
        "the composed Candle catalog must carry the engine codec table exactly once plus the \
             GGUF container row; no family crate may add or repeat a row"
    );
    let mut registered_ids: Vec<&str> = registered.iter().map(|codec| codec.codec_id).collect();
    registered_ids.sort_unstable();
    let mut implemented: Vec<&str> = CODEC_IMPLEMENTATION_IDS
        .iter()
        .copied()
        .chain(std::iter::once(
            candle_gen_wan::GGUF_CODEC_IMPLEMENTATION_ID,
        ))
        .collect();
    implemented.sort_unstable();
    assert_eq!(
        registered_ids, implemented,
        "every registered codec row needs a decode implementation and vice versa"
    );
    // The GGUF row is reachable by its own container encoding, and — critically — is NOT
    // reachable from any safetensors dtype: `WeightEncoding::from_dtype` can never produce
    // `GgufContainer`, so a safetensors U8/fp8 tensor cannot be routed into the GGUF decoder.
    assert!(registry
        .checkpoint_codecs()
        .for_encoding(candle_gen::gen_core::WeightEncoding::GgufContainer)
        .is_some_and(|codec| codec.codec_id == GGUF_CONTAINER_CODEC.codec_id));
    // The plan-side registry `candle-gen-wan` compiles against carries the same row.
    assert_eq!(
        candle_gen_wan::gguf_codec_registry()
            .for_encoding(candle_gen::gen_core::WeightEncoding::GgufContainer)
            .map(|codec| codec.codec_id),
        Some(GGUF_CONTAINER_CODEC.codec_id)
    );
    assert!(registry
        .checkpoint_codecs()
        .for_encoding(candle_gen::gen_core::WeightEncoding::DenseBf16)
        .is_some());
    assert!(registry
        .checkpoint_codecs()
        .for_encoding(candle_gen::gen_core::WeightEncoding::Fp8E4M3)
        .is_some());
}

/// Every `mapping_id` the composed Candle catalog registers resolves to a real
/// `LogicalKeyMapping` on this backend, **or** is explicitly declared not plan-driven here.
///
/// Both directions matter, and both were broken before sc-20651: five declared mapping ids had
/// no implementation anywhere (they read like backed routes and were not), and nothing proved
/// that a mapping the Candle lane really does plan through is declared as such. The mirror of
/// `mlx-gen-catalog`'s `canonical_mappings_are_backed_by_declared_implementations`.
#[test]
fn canonical_mappings_are_backed_by_declared_implementations() {
    use candle_gen::gen_core::{CheckpointBackend, LogicalKeyMapping};

    let registry = super::provider_registry().unwrap();
    // The complete set of `LogicalKeyMapping` implementations reachable from the Candle
    // platform. Adding one without declaring it (or declaring one without adding it) fails
    // below.
    //
    // The Krea mapping carries a file-detected namespace prefix and an optional architecture
    // config; neither affects `mapping_id`, so the id-surface check below uses the
    // no-config, bare-prefix form.
    let krea = candle_gen_krea::native_mapping::KreaNativeToDiffusersMapping::without_config("");
    // The FLUX.2 mapping carries the variant architecture config (it declares logical shapes
    // and the fused-transform geometry from it); the id surface is config-independent.
    let flux2_cfg = candle_gen_flux2::config::Flux2Variant::Klein9b.config();
    let flux2 = candle_gen_flux2::Flux2BflToDiffusersMapping::new(&flux2_cfg);
    let implementations: &[&dyn LogicalKeyMapping] =
        &[&candle_gen_wan::WanNativeToDiffusersMapping, &krea, &flux2];

    let mut declared_here = 0usize;
    for adapter in registry.checkpoint_adapters() {
        for mapping in adapter.canonical_mappings {
            let implemented = implementations
                .iter()
                .any(|implementation| implementation.mapping_id() == mapping.mapping_id);
            let declared = mapping
                .plan_driven_backends
                .contains(&CheckpointBackend::Candle);
            assert_eq!(
                implemented,
                declared,
                "adapter {} dialect {:?}: mapping id {:?} is {} on Candle but {} — a declared \
                     mapping must resolve to a real implementation, and an implementation must be \
                     declared (loader-native dialects declare no backend at all)",
                adapter.adapter_id,
                mapping.dialect,
                mapping.mapping_id,
                if implemented {
                    "implemented"
                } else {
                    "unimplemented"
                },
                if declared {
                    "declared plan-driven"
                } else {
                    "declared loader-native"
                },
            );
            declared_here += usize::from(declared);
        }
    }
    assert_eq!(
        declared_here,
        implementations.len(),
        "every Candle mapping implementation must be claimed by exactly one registered dialect"
    );

    // And the Wan mapping really is the refusing remap the GGUF route plans through: a native
    // key resolves to its diffusers name, and a foreign one refuses rather than passing
    // through unchanged.
    let wan = candle_gen_wan::WanNativeToDiffusersMapping;
    assert_eq!(
        wan.logical_key("blocks.0.self_attn.q.weight").as_deref(),
        Some("blocks.0.attn1.to_q.weight")
    );
    assert_eq!(wan.logical_key("vace_blocks.0.before_proj.weight"), None);

    // ...and the Krea mapping really is the native-mmdit remap the Kitchen NVFP4 import plans
    // through: a native key resolves to its diffusers name, a foreign one refuses.
    assert_eq!(
        krea.logical_key("blocks.0.attn.wq.weight").as_deref(),
        Some("transformer_blocks.0.attn.to_q.weight")
    );
    assert_eq!(krea.logical_key("blocks.0.attn.bogus"), None);
}

#[test]
fn checkpoint_adapter_catalog_uses_shared_portable_authority_and_real_candle_bindings() {
    use candle_gen::gen_core::{
        CheckpointBackend, ImportedModelOperation, ImportedModelRegistration, ImportedModelSource,
        BASE_SNAPSHOT_COMPONENT, FLUX2_CHECKPOINT_ADAPTER, KREA_2_CHECKPOINT_ADAPTER,
        QWEN_IMAGE_CHECKPOINT_ADAPTER, SDXL_CHECKPOINT_ADAPTER, WAN_CHECKPOINT_ADAPTER,
        Z_IMAGE_CHECKPOINT_ADAPTER,
    };

    let registry = super::provider_registry().unwrap();
    let expected = [
        &FLUX2_CHECKPOINT_ADAPTER,
        &KREA_2_CHECKPOINT_ADAPTER,
        &QWEN_IMAGE_CHECKPOINT_ADAPTER,
        &SDXL_CHECKPOINT_ADAPTER,
        // The Wan 2.2 ComfyUI expert pair (sc-20644). Registering it widened this frozen
        // corpus by one row; the corpus is a SHAPE assertion, so it is rewritten here in the
        // same change rather than exempted.
        &WAN_CHECKPOINT_ADAPTER,
        &Z_IMAGE_CHECKPOINT_ADAPTER,
    ];
    let adapters: Vec<_> = registry.checkpoint_adapters().collect();
    assert_eq!(adapters.len(), expected.len());
    for portable in expected {
        let bound = adapters
            .iter()
            .copied()
            .find(|adapter| adapter.adapter_id == portable.adapter_id)
            .unwrap_or_else(|| panic!("missing Candle adapter {}", portable.adapter_id));
        assert!(
            bound.has_same_portable_metadata(portable),
            "{} drifted from portable metadata",
            portable.adapter_id
        );
        assert!(bound
            .backend_bindings
            .iter()
            .all(|binding| binding.backend == CheckpointBackend::Candle));
    }

    let binding_operations = |adapter_id| {
        adapters
            .iter()
            .copied()
            .find(|adapter| adapter.adapter_id == adapter_id)
            .unwrap()
            .backend_bindings
            .iter()
            .map(|binding| binding.operation)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        binding_operations(KREA_2_CHECKPOINT_ADAPTER.adapter_id),
        [
            ImportedModelOperation::Generate,
            ImportedModelOperation::Edit,
            ImportedModelOperation::MultiPhase,
        ],
        "Candle truthfully omits the MLX-only Krea pose route"
    );
    assert_eq!(
        binding_operations(SDXL_CHECKPOINT_ADAPTER.adapter_id),
        [ImportedModelOperation::Generate],
        "Candle truthfully omits the fused SDXL edit route (sc-20651 review). The previous \
             expectation here asserted that Candle 'implements' it; it did not. The binding \
             existed, but it named the txt2img `sdxl` provider, whose descriptor declares no \
             `Reference` conditioning — so the capability floor refused every request the route \
             admitted. The Candle SDXL edit stack (`edit_provider::SdxlEdit`) is a name-driven \
             provider that needs a diffusers snapshot dir and a staged fp16-fix VAE; it has no \
             `LdmComponents`-fed constructor and so cannot serve a fused single-file import. \
             `SDXL_CHECKPOINT_ADAPTER.eligible_backends` is `[Mlx, Candle]`, so the per-operation \
             completeness check does not oblige Candle to bind Edit — exactly as Candle Krea \
             omits the MLX-only pose route above. MLX keeps its Edit binding and honors it."
    );
    for adapter in &adapters {
        if adapter.eligible_backends == [CheckpointBackend::Candle] {
            for operation in adapter.operations {
                assert!(
                    adapter
                        .backend_bindings
                        .iter()
                        .any(|binding| binding.operation == *operation),
                    "single-backend Candle adapter {} leaves {operation:?} implementation-free",
                    adapter.adapter_id
                );
            }
        }
    }
    let expected_legacy_projection = [
        ImportedModelRegistration {
            family: "flux2",
            source: ImportedModelSource::ComfyUiTree,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_flux2::config::FLUX2_DEV_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        // sc-21485: the klein universal BFL transformer single file routes to
        // `flux2_klein_9b` — a separate artifact shape from the dev ComfyUI tree above.
        ImportedModelRegistration {
            family: "flux2",
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_flux2::config::FLUX2_KLEIN_9B_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        ImportedModelRegistration {
            family: "krea_2",
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_krea::KREA_2_TURBO_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        ImportedModelRegistration {
            family: "krea_2",
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::Edit,
            provider_id: candle_gen_krea::KREA_2_TURBO_EDIT_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        ImportedModelRegistration {
            family: "krea_2",
            source: ImportedModelSource::TransformerFile,
            operation: ImportedModelOperation::MultiPhase,
            provider_id: candle_gen_krea::KREA_2_RAW_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        ImportedModelRegistration {
            family: "qwen-image",
            source: ImportedModelSource::ComfyUiTree,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_qwen_image::config::MODEL_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
        ImportedModelRegistration {
            family: "sdxl",
            source: ImportedModelSource::FusedCheckpoint,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_sdxl::MODEL_ID,
            required_components: Some(&["tokenizer_clip_l", "tokenizer_clip_bigg"]),
            inherit_adapters: true,
        },
        // No `sdxl` + `FusedCheckpoint` + `Edit` row: Candle does not bind that operation
        // (sc-20651 review). See the `binding_operations` expectation above for why.
        ImportedModelRegistration {
            // Wan's imported route takes NO caller-staged components: the UMT5 encoder, VAE
            // and tokenizer come from a resident snapshot tier the caller resolves, not from
            // `LoadSpec::components`. And `inherit_adapters` is false because
            // `load_from_comfyui_experts` has no adapter seam (sc-20644).
            family: "wan-video",
            source: ImportedModelSource::ComfyUiTree,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_wan::config::MODEL_ID_T2V_14B,
            required_components: None,
            inherit_adapters: false,
        },
        ImportedModelRegistration {
            family: "z-image",
            source: ImportedModelSource::ComfyUiTree,
            operation: ImportedModelOperation::Generate,
            provider_id: candle_gen_z_image::MODEL_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        },
    ];
    assert_eq!(
        registry.imported_models().copied().collect::<Vec<_>>(),
        expected_legacy_projection,
        "the legacy catalog surface must be the exact pre-registry compatibility projection"
    );
}

#[test]
fn modelled_video_provider_ids_have_typed_vae_assignments() {
    let registry = super::provider_registry().unwrap();
    let registered: Vec<&str> = registry
        .generators()
        .map(|registration| (registration.descriptor)().id)
        .collect();
    let expected = [
        (
            candle_gen_ltx::config::MODEL_ID,
            candle_gen_ltx::VAE_TILING,
            Some(2_725_804_800),
        ),
        (
            candle_gen_wan::config::MODEL_ID,
            candle_gen_wan::WAN_Z48_VAE_TILING,
            Some(382_730_240),
        ),
        (
            candle_gen_wan::config::MODEL_ID_T2V_14B,
            candle_gen_wan::WAN_Z16_VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_wan::config::MODEL_ID_I2V_14B,
            candle_gen_wan::WAN_Z16_VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_wan::config::MODEL_ID_VACE,
            candle_gen_wan::WAN_Z16_VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_wan::model_vace_fun::MODEL_ID_VACE_FUN,
            candle_gen_wan::model_vace_fun::ProviderVae::VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_bernini::MODEL_ID,
            candle_gen_bernini::VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_bernini::bernini::MODEL_ID,
            candle_gen_bernini::VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_scail2::MODEL_ID,
            candle_gen_scail2::VAE_TILING,
            Some(265_830_400),
        ),
        (
            candle_gen_svd::config::MODEL_ID,
            candle_gen_svd::VAE_TILING,
            None,
        ),
    ];

    for (provider_id, tiling, peak_bytes) in expected {
        assert!(
            registered.contains(&provider_id),
            "unregistered {provider_id}"
        );
        assert_eq!(
            super::vae_tiling(provider_id),
            Some(tiling),
            "{provider_id}"
        );
        assert_eq!(
            super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9)
                .map(|profile| profile.working_set_bytes()),
            peak_bytes,
            "{provider_id}"
        );
    }
    assert_eq!(super::vae_tiling("ltx_2_3"), None);
    assert_eq!(super::vae_tiling(candle_gen_mochi::MODEL_ID), None);
    assert_eq!(super::vae_tiling("krea_realtime_14b"), None);
    assert_eq!(super::vae_tiling("not_a_provider"), None);
    for provider_id in [
        "ltx_2_3",
        candle_gen_svd::config::MODEL_ID,
        candle_gen_mochi::MODEL_ID,
        "krea_realtime_14b",
        "not_a_provider",
    ] {
        assert_eq!(
            super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9),
            None
        );
    }
    assert_eq!(
        super::conservative_video_decode_memory_profile(candle_gen_ltx::config::MODEL_ID, 64, 0, 9,),
        None
    );
    assert_eq!(
        super::conservative_video_decode_memory_profile(
            candle_gen_wan::config::MODEL_ID_T2V_14B,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ),
        None
    );

    let ltx = super::conservative_video_decode_memory_profile(
        candle_gen_ltx::config::MODEL_ID,
        64,
        64,
        9,
    )
    .unwrap();
    assert_eq!(ltx.resident_decoder_bytes_included(), 0);
    assert_eq!(
        ltx.checked_composed_peak(2_500_000_000, 2_500_000_000),
        Some(5_225_804_800)
    );
    assert_eq!(
        ltx.checked_composed_peak(2_900_000_000, 2_900_000_000),
        Some(5_625_804_800)
    );

    let wan = super::conservative_video_decode_memory_profile(
        candle_gen_wan::config::MODEL_ID_T2V_14B,
        64,
        64,
        9,
    )
    .unwrap();
    assert_eq!(wan.resident_decoder_bytes_included(), 0);
    assert_eq!(
        wan.checked_composed_peak(1_000_000_000, 600_000_000),
        Some(1_265_830_400)
    );
}

#[test]
fn every_registered_generator_advertises_its_exact_latent_space() {
    use candle_gen::gen_core::{
        LatentSpace, FLUX1_LATENT_SPACE, FLUX2_PACKED_LATENT_SPACE, LTX_VIDEO_LATENT_SPACE,
        MAGE_LATENT_SPACE, MOCHI_VIDEO_LATENT_SPACE, QWEN_IMAGE_2_1_Z64_LATENT_SPACE,
        QWEN_KREA_Z16_LATENT_SPACE, SANA_LATENT_SPACE, SD3_LATENT_SPACE, SDXL_LATENT_SPACE,
        SEEDVR2_VIDEO_LATENT_SPACE, SVD_LATENT_SPACE, WAN_Z16_VIDEO_LATENT_SPACE,
        WAN_Z48_LATENT_SPACE,
    };

    fn expected(
        descriptor: &candle_gen::gen_core::ModelDescriptor,
    ) -> Option<&'static LatentSpace> {
        match descriptor.family {
            "anima" | "qwen-image" | "krea_2" => Some(&QWEN_KREA_Z16_LATENT_SPACE),
            // Qwen-Image 2.1's 64-channel RGBA autoencoder is its own lineage (sc-24109).
            "qwen-image-2-1" => Some(&QWEN_IMAGE_2_1_Z64_LATENT_SPACE),
            "wan" if descriptor.id == "wan2_2_ti2v_5b" => Some(&WAN_Z48_LATENT_SPACE),
            "bernini" | "scail2" | "wan" => Some(&WAN_Z16_VIDEO_LATENT_SPACE),
            "flux" | "boogu" | "chroma" | "z-image" => Some(&FLUX1_LATENT_SPACE),
            "stable-diffusion-3" => Some(&SD3_LATENT_SPACE),
            "sdxl" | "kolors" => Some(&SDXL_LATENT_SPACE),
            "flux2" | "ideogram" | "lens" => Some(&FLUX2_PACKED_LATENT_SPACE),
            // FLUX 3's packed 16× latent (its own autoencoder with BatchNorm-stat tokens)
            // has no shared decoder consumer, so nothing is advertised (sc-26235).
            "flux3" => None,
            "ltx" => Some(&LTX_VIDEO_LATENT_SPACE),
            "mage_flow" => Some(&MAGE_LATENT_SPACE),
            "mochi" => Some(&MOCHI_VIDEO_LATENT_SPACE),
            "moss_image" => Some(&FLUX2_PACKED_LATENT_SPACE),
            "sana" => Some(&SANA_LATENT_SPACE),
            "seedvr2" => Some(&SEEDVR2_VIDEO_LATENT_SPACE),
            // Supra2-IMG decodes with stabilityai/sd-vae-ft-mse: SD's 4-channel f8 latent at
            // scale 0.18215, the same space SVD's VAE consumes (sc-24214, as on MLX).
            "svd" | "supra2" => Some(&SVD_LATENT_SPACE),
            // SenseNova's flow head emits RGB patches directly; there is no latent decoder seam.
            "sensenova-u1" => None,
            // Iris-3B generates RGB pixels directly (patch DiT + PiT pixel head, no VAE).
            "iris" => None,
            // MiniMax-H3's denoiser emits a 24-channel joint audio+video latent on the
            // 17-frame clip lattice (token-dropped, seam-blended dual decode — see the crate's
            // `chunking` module). No `LatentTemporalLaw` variant expresses that mapping and no
            // external decoder can consume it, so the descriptor deliberately advertises
            // nothing and fails closed against every decoder swap.
            "minimax_h3" => None,
            family => panic!(
                "{} has unclassified latent lineage for registered family {family}",
                descriptor.id
            ),
        }
    }

    let registry = super::provider_registry().unwrap();
    for registration in registry.generators() {
        let descriptor = (registration.descriptor)();
        assert_eq!(
            descriptor.denoiser_output_latent_space,
            expected(&descriptor),
            "{} must advertise the latent space its decoder consumes",
            descriptor.id
        );
    }
}

/// AC (sc-22661 / epic SC-22657 E1+E2): every registered contract surface in the composed media
/// catalog publishes an honest byte decomposition and no fabricated architecture axis.
///
/// This is the registry-wide half of the story's acceptance test. The surfaces are built
/// **weights-free** — the registry names the sentinel snapshot
/// `/__sceneworks_memory_contract_surface__`, which is not on disk — so every provider here
/// lands on the walk's Candle arm: `check_memory_contract_asset_facts` is the byte half that
/// must hold, and `MemoryArchitectureFacts::default()` is the required E2 state. That second
/// rule catches the converse defect — a provider that published an architecture axis on a
/// contract built with nothing to read would have hardcoded it from its own provider id.
///
/// It cannot, on its own, catch a provider that derives *nothing* — `default()` is exactly what
/// that provider publishes here too. `every_materializable_provider_derives_geometry_from_a_snapshot_root`
/// is the arm that does, and the two run over the same composed registry.
///
/// `runtime-cuda` runs this walk over the CUDA bundle's registry; this one keeps the
/// composition root itself covered on a lane that needs no accelerator.
#[test]
fn every_registered_contract_surface_publishes_honest_facts() {
    let registry = super::memory_contract_surface_registry().unwrap();
    gen_core_testkit::memory_contract_surface_registry_facts_conformance(&registry, None);
    // Non-vacuous: the walk must have had surfaces to reject.
    assert!(
        !registry.memory_contract_surfaces().unwrap().is_empty(),
        "the composed catalog must publish contract surfaces for the facts walk to check"
    );
}

/// The providers whose admission accepts a synthetic snapshot root, and which must therefore
/// derive at least one architecture axis from it (sc-22661).
///
/// This is the non-vacuous half of the E2 story. A provider that returned
/// `MemoryArchitectureFacts::default()` unconditionally satisfies every weights-free assertion
/// in `every_registered_contract_surface_publishes_honest_facts` — that walk *requires*
/// `default()` on the Candle arm — so only rebuilding the contract against a materialized root
/// can separate "derives nothing" from "has nothing to derive yet". Reverting any one of these
/// crates' `architecture_facts` to `::default()` turns the assertion below red.
///
/// The catalog's remaining providers are absent for one reason in four shapes, each a
/// *pre-existing admission rule* rather than anything about facts: the route demands an exact
/// resolved catalog route (chroma, sd3, sdxl, qwen-image-edit), an exact immutable turnkey
/// repo/revision (boogu, sana, ideogram, kolors' base tier), a specific tier subdirectory or
/// component file on disk (scail2's `dit.safetensors`, ltx-2.5's video VAE, anima's
/// `diffusion_models/`, sensenova's shards, svd's unquantized surface, ltx-2.3's plain split q4
/// tier), or a whole snapshot layout to walk (krea, bernini, qwen-image). Standing those up is
/// building each provider's own load fixture — which is exactly what each crate's own
/// `architecture_facts_*` test does, and which `every_memory_route_crate_reaches_the_catalog`
/// proves every memory-route crate has.
const MATERIALIZED_ROOT_PROVIDERS: &[&str] = &[
    "candle_kolors_control",
    "candle_kolors_ipadapter",
    "flux1_dev",
    "flux1_schnell",
    "flux2_dev",
    "flux2_klein_9b",
    "krea_2_turbo_control",
    "lens",
    "lens_turbo",
    "mage_flow",
    "mage_flow_base",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
    "mage_flow_turbo",
    "minimax_h3",
    "z_image",
    "z_image_control",
    "z_image_turbo",
    "z_image_turbo_control",
];

/// AC (sc-22661 / epic SC-22657 E2): the registry-level walk is non-vacuous. Every provider
/// that can be handed a materialized snapshot root must derive geometry from it — the assertion
/// an unconditional `MemoryArchitectureFacts::default()` fails, and the one the weights-free
/// walk structurally cannot make.
#[test]
fn every_materializable_provider_derives_geometry_from_a_snapshot_root() {
    use std::collections::BTreeSet;

    let tmp = tempfile::tempdir().unwrap();
    let root = synthetic_snapshot_root(tmp.path());
    let expected = MATERIALIZED_ROOT_PROVIDERS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected.len(),
        MATERIALIZED_ROOT_PROVIDERS.len(),
        "the materialized-root list must not repeat a provider"
    );
    let registry = super::memory_contract_surface_registry().unwrap();
    let registered = registry
        .memory_strategy_registrations()
        .map(|registration| registration.provider_id)
        .collect::<BTreeSet<_>>();
    assert!(
        expected.is_subset(&registered),
        "the materialized-root list names providers this catalog does not register: {:?}",
        expected.difference(&registered).collect::<Vec<_>>()
    );

    let lookup = |provider_id: &str| expected.contains(provider_id).then(|| root.clone());
    let coverage = gen_core_testkit::memory_contract_surface_registry_facts_conformance(
        &registry,
        Some(&lookup),
    );
    assert_eq!(
        coverage.materialized_providers_checked,
        MATERIALIZED_ROOT_PROVIDERS.len(),
        "every listed provider must have been rebuilt against the synthetic root"
    );
}

/// A synthetic snapshot root carrying the component layouts and config keys the catalog's
/// providers read.
fn synthetic_snapshot_root(base: &std::path::Path) -> std::path::PathBuf {
    let root = base.join("sc22661-synthetic-snapshot");
    for (component, config) in SYNTHETIC_COMPONENT_CONFIGS {
        let dir = if component.is_empty() {
            root.clone()
        } else {
            root.join(component)
        };
        std::fs::create_dir_all(&dir).expect("synthetic component dir");
        std::fs::write(dir.join("config.json"), config).expect("synthetic component config");
    }
    root
}

const SYNTHETIC_COMPONENT_CONFIGS: &[(&str, &str)] = &[
    (
        "",
        r#"{"patch_size": 2, "llm_config": {"hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "head_dim": 128}}"#,
    ),
    (
        "transformer",
        r#"{"in_channels": 64, "out_channels": 64, "hidden_size": 3072, "num_heads": 24,
                "num_attention_heads": 24, "attention_head_dim": 128, "depth": 12,
                "num_layers": 19, "num_single_layers": 38, "patch_size": 2, "caption_channels": 2304,
                "num_key_value_heads": 8, "cross_attention_dim": 2048, "context_in_dim": 2560,
                "axes_dim": [16, 56, 56], "checkpoint": false}"#,
    ),
    (
        "unet",
        r#"{"in_channels": 4, "block_out_channels": [320, 640, 1280],
                "attention_head_dim": [5, 10, 20], "layers_per_block": 2, "num_attention_heads": 20}"#,
    ),
    (
        "vae",
        r#"{"latent_channels": 16, "z_channels": 16, "block_out_channels": [128, 256, 512, 512],
                "scale_factor_spatial": 8, "temperal_downsample": [false, true, true],
                "patch_size": [1, 2, 2], "dim_mult": [1, 2, 4, 4]}"#,
    ),
];

#[test]
fn every_registered_memory_strategy_rejects_cross_route_decode_geometry() {
    use std::collections::{BTreeMap, BTreeSet};

    let registry = super::provider_registry().unwrap();
    let contract_registry = super::memory_contract_surface_registry().unwrap();
    gen_core_testkit::memory_contract_surface_registry_conformance(&contract_registry);
    let surfaces = contract_registry.memory_contract_surfaces().unwrap();

    // Shape, not population. The registration/surface/composed totals used to be pinned as
    // hand-maintained numbers (`24`/`23`, `21 * 12 + 2 * 16 [+ 6]`, `4`), which any legitimate
    // catalog growth re-trips while saying nothing about *which* provider moved.
    //
    // Scope, stated precisely: every claim below reads the same registry, so it constrains that
    // registry's *internal consistency* — a witness set that is orphaned, doubled, ragged, or
    // mismatched against the composed-route seam. It cannot, on its own, notice a registration
    // that vanished together with its fixture, because that shrinks both sides at once. The
    // registration set is anchored to something outside the registry — the provider crate
    // sources — by `every_memory_route_crate_reaches_the_catalog`, which is what replaced the
    // `24`/`23` pin. Read the two together.

    // Coverage is exact in both directions: each memory-strategy registration either publishes
    // optimized surfaces or is a resident-only witness, and nothing else reaches the inventory.
    let strategy_ids = contract_registry
        .memory_strategy_registrations()
        .map(|registration| registration.provider_id)
        .collect::<BTreeSet<_>>();
    let resident_only_ids = contract_registry
        .resident_only_memory_contract_registrations()
        .map(|registration| registration.provider_id)
        .collect::<BTreeSet<_>>();
    assert!(
        resident_only_ids.is_subset(&strategy_ids),
        "resident-only witnesses without a memory-strategy registration: {:?}",
        resident_only_ids
            .difference(&strategy_ids)
            .collect::<Vec<_>>()
    );
    let surfaced_ids = surfaces
        .iter()
        .map(|surface| surface.contract.provider_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        surfaced_ids,
        strategy_ids
            .difference(&resident_only_ids)
            .copied()
            .collect::<BTreeSet<_>>(),
        "optimized surfaces must cover exactly the non-resident-only memory registrations"
    );

    // What the old `21 * 12 + 2 * 16 + 6` arithmetic was really asserting, per provider: a
    // witness set is a full rectangle over the residency/materialization axes for every tier it
    // publishes (12 = 3 tiers x 2 policies x 2 shapes, 16 = 4 tiers x 4, and TI2V-5B's 6 =
    // 3 tiers x 2 policies x its single eager shape). Stated as a rectangle it is derived, so it
    // holds at any catalog size — and on the tier and shape axes it is stronger, because a
    // provider that dropped one (policy, shape) cell only moved the old global sum by
    // coincidence.
    //
    // A rectangle alone cannot see a *whole* axis collapse, though: drop every Sequential
    // surface and the remaining Resident-only set is still a perfect rectangle. The residency
    // axis is the one with a universe that does not vary by provider — every Candle memory route
    // is witnessed under both offload policies — so it is pinned to the `OffloadPolicy` enum
    // rather than to the provider's own set. Tier and shape stay per-provider derived: those
    // genuinely differ (TI2V-5B has no deferred block loader, tiers vary with what a route
    // ships).
    //
    // `MemoryContractSurfaceSelector::id` is the stable `tier:policy:shape` encoding of the three
    // axes, so splitting it recovers them without needing `Ord` on the runtime enums.
    let axes = |id: &'static str| {
        let mut parts = id.split(':');
        let tier = parts.next().expect("selector ids start with a tier");
        let policy = parts.next().expect("selector ids name a residency policy");
        let shape = parts
            .next()
            .expect("selector ids name a materialization shape");
        assert_eq!(parts.next(), None, "unexpected selector id shape: {id:?}");
        (tier, policy, shape)
    };
    let offload_policy_universe = {
        use candle_gen::gen_core::{
            LoadShape, MemoryContractSurfaceSelector, MemoryContractSurfaceTier, OffloadPolicy,
        };
        // Exhaustiveness guard: a new `OffloadPolicy` variant has to be classified here on
        // purpose instead of quietly shrinking the universe every provider is measured against.
        fn _every_policy_is_listed(policy: OffloadPolicy) {
            match policy {
                OffloadPolicy::Resident | OffloadPolicy::Sequential => {}
            }
        }
        let tokens = [OffloadPolicy::Resident, OffloadPolicy::Sequential]
            .into_iter()
            .map(|offload_policy| {
                axes(
                    MemoryContractSurfaceSelector {
                        tier: MemoryContractSurfaceTier::Bf16,
                        offload_policy,
                        load_shape: LoadShape::EagerMaterialization,
                    }
                    .id(),
                )
                .1
            })
            .collect::<Vec<_>>();
        let universe = tokens.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(
            universe.len(),
            tokens.len(),
            "each OffloadPolicy must encode a distinct selector token"
        );
        universe
    };

    let mut published = BTreeMap::<&str, BTreeSet<_>>::new();
    for surface in &surfaces {
        published
            .entry(surface.contract.provider_id.as_str())
            .or_default()
            .insert(axes(surface.selector.id()));
    }
    for (provider_id, selectors) in &published {
        let tiers = selectors
            .iter()
            .map(|(tier, ..)| *tier)
            .collect::<BTreeSet<_>>();
        let policies = selectors
            .iter()
            .map(|(_, policy, _)| *policy)
            .collect::<BTreeSet<_>>();
        let shapes = selectors
            .iter()
            .map(|(.., shape)| *shape)
            .collect::<BTreeSet<_>>();
        let mut rectangle = BTreeSet::new();
        for tier in &tiers {
            for policy in &policies {
                for shape in &shapes {
                    rectangle.insert((*tier, *policy, *shape));
                }
            }
        }
        assert_eq!(
            *selectors, rectangle,
            "{provider_id} publishes a ragged witness set: every tier it declares must be \
                 witnessed on every residency/materialization combination it supports"
        );
        assert_eq!(
                policies, offload_policy_universe,
                "{provider_id} witnesses only the {policies:?} residency policies; every Candle \
                 memory route must be witnessed across the whole OffloadPolicy universe \
                 {offload_policy_universe:?}. A rectangle survives a whole axis collapsing, so this \
                 is checked against the enum rather than against the provider's own set"
            );
    }

    // Composed routes are exactly the memory registrations with no standalone generator:
    // `register_memory_strategy` rejects an unmatched id, so `register_composed_memory_strategy`
    // is the only seam that admits one. Deriving the set that way asserts the `composed` flag
    // agrees with the registration seam rather than pinning how many composed routes exist.
    let generator_ids = contract_registry
        .generators()
        .map(|registration| (registration.descriptor)().id)
        .collect::<BTreeSet<_>>();
    let observed_composed = surfaces
        .iter()
        .filter(|surface| surface.composed)
        .map(|surface| surface.contract.provider_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        observed_composed,
        surfaced_ids
            .iter()
            .copied()
            .filter(|id| !generator_ids.contains(id))
            .collect::<BTreeSet<_>>(),
        "the composed flag must mark exactly the generator-less memory routes"
    );
    assert!(
        !observed_composed.is_empty(),
        "the Candle catalog publishes composed memory routes; none were observed"
    );

    let spec = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
        "/nonexistent".into(),
    ));
    gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
}

#[test]
fn krea_raw_and_edit_publish_exact_request_scoped_candle_surfaces() {
    use candle_gen::gen_core::{MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport};

    let registry = super::memory_contract_surface_registry().unwrap();
    let surfaces = registry.memory_contract_surfaces().unwrap();
    for provider_id in ["krea_2_raw", "krea_2_edit", "krea_2_turbo_edit"] {
        let provider_surfaces: Vec<_> = surfaces
            .iter()
            .filter(|surface| surface.contract.provider_id == provider_id)
            .collect();
        assert_eq!(provider_surfaces.len(), 12, "{provider_id}");
        let tiers = provider_surfaces
            .iter()
            .map(|surface| surface.resolved_artifact_tier())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            tiers,
            [
                MemoryContractSurfaceTier::Bf16,
                MemoryContractSurfaceTier::Q4,
                MemoryContractSurfaceTier::Q8,
            ]
            .into_iter()
            .collect(),
            "{provider_id}"
        );
        // sc-22735: these are weights-free declarations, so their identities live in the
        // registry-behavior namespace and carry their own (route, declared tier) key. The
        // production strings — `krea-2-<route>-<tier>-cuda-staged-residency-v1` — are what a
        // measured anchor binds to, and no catalog surface may republish one.
        let mut declared = std::collections::BTreeSet::new();
        for surface in provider_surfaces {
            assert!(!surface.composed, "{provider_id}");
            assert_eq!(
                surface.contract.asset_facts,
                candle_gen::gen_core::MemoryAssetFacts::default(),
                "{provider_id}"
            );
            let fingerprint = &surface.contract.calibration.as_ref().unwrap().fingerprint;
            let tier = match surface.resolved_artifact_tier() {
                MemoryContractSurfaceTier::Bf16 => "bf16",
                MemoryContractSurfaceTier::Q4 => "q4",
                MemoryContractSurfaceTier::Q8 => "q8",
                other => panic!("{provider_id}: unexpected surface tier {other:?}"),
            };
            assert_eq!(
                fingerprint,
                &format!(
                    "krea-2-candle-registry-behavior-v1-{}-{tier}",
                    provider_id.replace('_', "-")
                ),
                "{provider_id}:{}",
                surface.selector.id()
            );
            for cell in ["q4", "q8", "bf16"] {
                assert_ne!(
                    fingerprint,
                    &format!(
                        "{}-{cell}-cuda-staged-residency-v1",
                        provider_id.replace('_', "-")
                    ),
                    "{provider_id}: a weights-free surface must never republish a production \
                         calibration identity"
                );
            }
            declared.insert(fingerprint.clone());
            for strategy in MemoryStrategy::ALL {
                let expected = matches!(
                    strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                );
                assert_eq!(
                    surface.contract.capability(strategy).unwrap().support
                        == MemoryStrategySupport::Implemented,
                    expected,
                    "{provider_id}:{}:{strategy:?}",
                    surface.selector.id()
                );
            }
        }
        assert_eq!(
            declared.len(),
            3,
            "{provider_id}: one declaration per artifact tier: {declared:?}"
        );
    }
}

#[test]
fn z_image_and_lens_publish_exact_typed_candle_rung_four_surfaces() {
    use candle_gen::gen_core::{
        LoadShape, MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport, OffloadPolicy,
    };

    let registry = super::memory_contract_surface_registry().unwrap();
    let surfaces = registry.memory_contract_surfaces().unwrap();
    for (provider_id, expected_count, composed, fingerprint) in [
        (
            "z_image",
            6,
            false,
            "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2",
        ),
        (
            "z_image_turbo",
            6,
            false,
            "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2",
        ),
        (
            "z_image_control",
            6,
            true,
            "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2",
        ),
        (
            "z_image_turbo_control",
            6,
            true,
            "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2",
        ),
        // sc-22732: the Lens weights-free surfaces publish a per-cell static behavior identity
        // instead of leaking the measured q4 Lens-Turbo production string onto all 24 of them,
        // so these two entries are the route-exact PREFIX of that namespace.
        ("lens", 2, false, "lens-candle-registry-behavior-v1-lens-"),
        (
            "lens_turbo",
            2,
            false,
            "lens-candle-registry-behavior-v1-lens-turbo-",
        ),
    ] {
        let provider_surfaces: Vec<_> = surfaces
            .iter()
            .filter(|surface| surface.contract.provider_id == provider_id)
            .collect();
        assert_eq!(provider_surfaces.len(), 12, "{provider_id}");
        let mut implemented = 0;
        for surface in provider_surfaces {
            let expected = if provider_id.starts_with("lens") {
                matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Q4 | MemoryContractSurfaceTier::Q8
                ) && surface.selector.offload_policy == OffloadPolicy::Sequential
                    && surface.selector.load_shape == LoadShape::DeferredMaterialization
            } else {
                matches!(
                    surface.resolved_artifact_tier(),
                    MemoryContractSurfaceTier::Bf16
                        | MemoryContractSurfaceTier::Q4
                        | MemoryContractSurfaceTier::Q8
                ) && surface.selector.load_shape == LoadShape::DeferredMaterialization
            };
            let rung = surface
                .contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .expect("complete Candle ladder");
            assert_eq!(
                rung.support,
                if expected {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                "{}:{}",
                provider_id,
                surface.selector.id()
            );
            implemented += usize::from(expected);
            assert_eq!(surface.composed, composed, "{provider_id}");
            let published = &surface.contract.calibration.as_ref().unwrap().fingerprint;
            if provider_id.starts_with("lens") {
                assert!(
                    published.starts_with(fingerprint),
                    "{provider_id}:{} published {published}",
                    surface.selector.id()
                );
                // `…-v1-lens-` is itself a prefix of `…-v1-lens-turbo-`, so the check above
                // cannot tell the base route's identity from the turbo route's. Pin the
                // discrimination explicitly rather than leaving it to prefix arithmetic.
                assert_eq!(
                    published.starts_with("lens-candle-registry-behavior-v1-lens-turbo-"),
                    provider_id == "lens_turbo",
                    "{provider_id}:{} published {published}",
                    surface.selector.id()
                );
                // The weights-free namespace must never be the measured production string.
                assert_ne!(
                    published.as_str(),
                    "lens-candle-cuda-shared-ladder-device-format-blocks-v1",
                    "{provider_id}:{} leaks the measured production identity",
                    surface.selector.id()
                );
            } else {
                assert_eq!(published, fingerprint, "{provider_id}");
            }
            assert_eq!(
                surface.contract.asset_facts,
                candle_gen::gen_core::MemoryAssetFacts::default(),
                "{provider_id}: weights-free surfaces cannot claim real inventory"
            );
            if composed {
                assert!(
                    surface.spec.control.is_some(),
                    "{provider_id}:{} missing mandatory control source",
                    surface.selector.id()
                );
            }
        }
        assert_eq!(implemented, expected_count, "{provider_id}");
    }
}

#[test]
fn cfg_capability_matrix_matches_the_registered_candle_render_paths() {
    let registry = super::provider_registry().unwrap();
    let descriptor = |id: &str| {
        registry
            .generators()
            .find_map(|registration| {
                let descriptor = (registration.descriptor)();
                (descriptor.id == id).then_some(descriptor)
            })
            .unwrap_or_else(|| panic!("{id} missing from Candle catalog"))
    };

    let klein = descriptor("flux2_klein_9b").capabilities;
    assert!(klein.supports_guidance);
    assert!(klein.supports_negative_prompt);
    assert!(!klein.supports_true_cfg);

    for id in ["sd3_5_large", "sd3_5_medium"] {
        let capabilities = descriptor(id).capabilities;
        assert!(capabilities.supports_guidance, "{id}");
        assert!(capabilities.supports_negative_prompt, "{id}");
        assert!(!capabilities.supports_true_cfg, "{id}");
    }
    let turbo = descriptor("sd3_5_large_turbo").capabilities;
    assert!(!turbo.supports_guidance);
    assert!(!turbo.supports_negative_prompt);
    assert!(!turbo.supports_true_cfg);

    for id in ["sensenova_u1_8b", "sensenova_u1_8b_fast"] {
        let capabilities = descriptor(id).capabilities;
        assert!(capabilities.supports_guidance, "{id}");
        assert!(!capabilities.supports_negative_prompt, "{id}");
        assert!(capabilities.supports_true_cfg, "{id} image CFG");
        assert_eq!(
            capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            "{id} registered conditioned-image surface"
        );
    }
}

#[test]
fn complete_catalog_has_stable_conforming_surface() {
    let registry = super::provider_registry().unwrap();
    let generators: Vec<String> = registry
        .generators()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();
    let trainers: Vec<String> = registry
        .trainers()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();
    let captioners: Vec<String> = registry
        .captioners()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();
    let image_embedders: Vec<String> = registry
        .image_embedders()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();
    let text_embedders: Vec<String> = registry
        .text_embedders()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();

    // sc-25683: the Iris-3B restorer is the catalog's first (image) transform.
    let transforms: Vec<String> = registry
        .transforms()
        .map(|r| (r.descriptor)().id.to_string())
        .collect();
    assert_eq!(transforms, ["iris_3b_restore"]);
    assert!(registry
        .transforms()
        .all(|r| (r.descriptor)().backend == "candle"));
    assert_eq!(
        registry.descriptor_conformance_errors(),
        Vec::<String>::new()
    );
    assert!(registry
        .generators()
        .all(|r| (r.descriptor)().backend == "candle"));
    assert!(registry
        .trainers()
        .all(|r| (r.descriptor)().backend == "candle"));
    assert_eq!(
        generators,
        [
            "anima_base",
            "anima_aesthetic",
            "anima_turbo",
            "bernini_renderer",
            "bernini",
            "boogu_image",
            "boogu_image_turbo",
            "boogu_image_edit",
            "chroma1_hd",
            "chroma1_base",
            "chroma1_flash",
            "flux1_schnell",
            "flux1_dev",
            "flux2_klein_9b",
            "flux2_dev",
            "flux3_klein",
            "flux3_klein_base",
            "ideogram_4",
            "ideogram_4_turbo",
            "iris_3b",
            "kolors",
            "krea_2_turbo",
            "krea_2_raw",
            "krea_2_edit",
            "krea_2_turbo_edit",
            "lens_turbo",
            "lens",
            "ltx_2_3_distilled",
            "ltx_2_5_distilled",
            "mage_flow",
            "mage_flow_base",
            "mage_flow_turbo",
            "mage_flow_edit",
            "mage_flow_edit_base",
            "mage_flow_edit_turbo",
            "minimax_h3",
            "mochi_1",
            "moss_image_base",
            "moss_image_sft",
            "qwen_image",
            "qwen_image_2_1",
            "qwen_image_2_1_turbo",
            "sana_1600m",
            "sana_sprint_1600m",
            "scail2_14b",
            "sd3_5_large",
            "sd3_5_large_turbo",
            "sd3_5_medium",
            "sdxl",
            "seedvr2",
            "seedvr2_3b",
            "seedvr2_7b",
            "sensenova_u1_8b",
            "sensenova_u1_8b_fast",
            "supra2_img",
            "svd_xt",
            "wan2_2_ti2v_5b",
            "wan2_2_t2v_14b",
            "wan2_2_i2v_14b",
            "wan_vace",
            "wan2_2_vace_fun_14b",
            "z_image_turbo",
            "z_image",
        ]
    );
    assert_eq!(
        trainers,
        [
            "anima_base",
            "flux3_klein_base",
            "iris_3b",
            "iris_3b_depth",
            "iris_3b_restore",
            "kolors",
            "krea_2_raw",
            "krea_2_control",
            "lens",
            "ltx_2_3",
            "ltx_2_5_distilled",
            "mage_flow_base",
            "moss_image_base",
            "moss_image_sft",
            "qwen_image_2_1",
            "sd3_5_large",
            "sd3_5_medium",
            "sdxl",
            "wan2_2_t2v_14b",
            "wan2_2_i2v_14b",
            "wan2_2_ti2v_5b",
            "z_image_turbo",
        ]
    );
    assert_eq!(
        captioners,
        ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
    );
    assert_eq!(image_embedders, ["clip_vit_l14"]);
    assert_eq!(text_embedders, ["clip_vit_l14_text"]);

    // sc-16667: the pinned surface and the model-weight licence mapping move together — this is
    // where a surface change and a mapping change meet. Fifteen of the seventeen trainer ids are
    // also generator ids (`krea_2_control` and `ltx_2_3` are trainer-only), which is why 57
    // generators + 2 trainer-only ids + 1 captioner + 2 embedders are 62 distinct ids (sc-24109
    // adds `qwen_image_2_1`, a generator with its own component row; its sc-24160 trainer
    // reuses that row; sc-25680 adds `iris_3b`, a generator with its own complete row; sc-25683
    // adds the `iris_3b_restore` transform with its own complete row: 63 / 53; the sc-25686
    // `iris_3b` trainer reuses the generator's row and adds no id; the sc-25687 `iris_3b_depth`
    // trainer is a new id with its own complete row: 64 / 54; the sc-25688 `iris_3b_restore`
    // trainer reuses the transform's id and row and adds no id; sc-25901 adds
    // `qwen_image_2_1_turbo`, a generator with its own complete row: 65 / 55; sc-26235 adds
    // `flux3_klein`, a generator with its own complete row: 66 / 56; sc-26239 adds
    // `flux3_klein_base`, a generator with its own complete row: 67 / 57; the sc-26240
    // `flux3_klein_base` trainer reuses that generator's id and row and adds no id; sc-24214
    // adds `supra2_img`, a generator with its own complete row: 70 / 60 with the MOSS pair).
    //
    // Registration is never conditioned on the mapping: 53 < 63 because ten ids load nothing
    // the shared checkpoint table covers, and they ship exactly as before. That gap is a hole in
    // our metadata for CI to report, and `licenses::tests` pins which ten and why — as
    // `#[cfg(test)]` data, so no gate can read it and suppress them.
    let distinct: std::collections::BTreeSet<&String> = generators
        .iter()
        .chain(&trainers)
        .chain(&captioners)
        .chain(&image_embedders)
        .chain(&text_embedders)
        .chain(&transforms)
        .collect();
    assert_eq!(distinct.len(), 70);
    assert_eq!(super::provider_components().len(), 60);
}

/// The manifest emitter runs on **this** catalog's three slices, and its output is
/// deterministic.
///
/// Deliberately cheap. Byte-stability of the shared emitter was settled in sc-16663 (#406,
/// #411) and the audio lane gates its bytes against a committed file; there is no committed
/// Candle media manifest to compare against, because merging the three catalogs into one
/// release artifact is sc-16664's job. So this pins only what is this crate's to pin: the
/// function is reachable, emits the schema-3 shape with all three layers present, and returns
/// the same bytes call to call.
#[test]
fn component_licenses_manifest_json_is_well_formed_and_stable() {
    let generated = super::component_licenses_manifest_json();
    assert!(!generated.is_empty());

    let parsed: serde_json::Value =
        serde_json::from_str(&generated).expect("manifest is valid JSON");
    assert_eq!(parsed["schema_version"], 3);
    assert_eq!(parsed["kind"], "model-weight-licenses");
    // A consumer reads one document and finds all three layers in it.
    for section in ["families", "components", "providers"] {
        assert!(
            parsed[section]
                .as_array()
                .is_some_and(|rows| !rows.is_empty()),
            "manifest section {section:?} is missing or empty"
        );
    }
    assert_eq!(
        parsed["providers"].as_array().expect("providers").len(),
        super::provider_components().len()
    );

    assert_eq!(
        super::component_licenses_manifest_json(),
        generated,
        "the emitter must be deterministic — a merged release artifact compares bytes"
    );
}

#[test]
fn mage_is_shipped_with_truthful_quant_surface() {
    let registry = super::provider_registry().expect("catalog");
    for id in [
        "mage_flow",
        "mage_flow_base",
        "mage_flow_turbo",
        "mage_flow_edit",
        "mage_flow_edit_base",
        "mage_flow_edit_turbo",
    ] {
        let descriptor = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == id)
            .unwrap_or_else(|| panic!("{id} missing from Candle catalog"));
        assert_eq!(descriptor.backend, "candle");
        assert!(!descriptor.capabilities.mac_only);
        assert_eq!(
            descriptor.capabilities.supported_quants,
            &[
                super::media::gen_core::Quant::Q4,
                super::media::gen_core::Quant::Q8
            ]
        );
    }
}

/// Pin the NVFP4 tier's catalog surface (epic 11037, sc-11042 Option A): the FP4 tier is exposed
/// **only** under the `cuda` feature (consumer Blackwell `sm_120`); the CPU candle bundle surfaces
/// no advanced tier. The MLX/macOS runtime uses a separate `mlx-gen-catalog` with no such surface —
/// the third leg of the pinned platform difference.
#[test]
fn nvfp4_tier_surface_is_cuda_only() {
    #[cfg(feature = "cuda")]
    assert_eq!(
        super::nvfp4_quant_tiers(),
        &[super::media::gen_core::Quant::Nvfp4],
        "NVFP4 must be surfaced on the cuda catalog"
    );
    #[cfg(not(feature = "cuda"))]
    assert!(
        super::nvfp4_quant_tiers().is_empty(),
        "NVFP4 must NOT be surfaced on a non-cuda (CPU) candle catalog"
    );

    // `SURFACES_NVFP4_TIER` is what dependent bundles read instead of a `cfg!` they cannot write;
    // it must never drift from the tier list itself.
    assert_eq!(
        super::SURFACES_NVFP4_TIER,
        !super::nvfp4_quant_tiers().is_empty(),
        "SURFACES_NVFP4_TIER must agree with nvfp4_quant_tiers()"
    );
}

#[test]
fn krea_cuda_memory_contract_is_not_exposed_by_the_cpu_catalog() {
    let registry = super::provider_registry().expect("catalog");
    let registered = registry
        .memory_strategy_registrations()
        .any(|registration| registration.provider_id == "krea_2_turbo");
    assert_eq!(registered, cfg!(feature = "cuda"));
}

#[test]
fn krea_catalog_rejects_missing_assets_when_cuda_contract_is_registered() {
    let registry = super::provider_registry().expect("catalog");
    let directory = tempfile::tempdir().unwrap();
    let spec = super::media::gen_core::LoadSpec::new(super::media::gen_core::WeightsSource::Dir(
        directory.path().join("missing"),
    ));
    let contract = registry.memory_strategy_contract("krea_2_turbo", &spec);
    #[cfg(feature = "cuda")]
    assert!(
        contract.is_err(),
        "CUDA production admission requires real assets"
    );
    #[cfg(not(feature = "cuda"))]
    assert!(contract.expect("known Krea generator").is_none());
}

/// sc-27309: Wan's sampler menus, pinned against the MLX twin (`mlx-gen-wan`). The dense 5B and both
/// VACE routes advertise exactly the MLX menus (the native flow-SNR `uni_pc` / `euler` / `dpmpp_2m`,
/// the 5B's gen-core fold-ins, the legacy aliases), so a recipe validates identically on both
/// platforms. The 14B MoE routes are the one deliberate difference: MLX also folds the gen-core
/// solvers (`euler_ancestral`, `heun`, `dpmpp_sde`, `ddim`) into its expert-routed denoise, while the
/// Candle 14B denoise drives only its native schedulers — so those four names are refused here.
#[test]
fn wan_sampler_menus_match_mlx_except_the_pinned_14b_fold_in_gap() {
    const NATIVE: [&str; 5] = ["uni_pc", "euler", "dpmpp_2m", "unipc", "dpmpp2m"];
    const MLX_FULL: [&str; 9] = [
        "uni_pc",
        "euler",
        "dpmpp_2m",
        "euler_ancestral",
        "heun",
        "dpmpp_sde",
        "ddim",
        "unipc",
        "dpmpp2m",
    ];
    let registry = super::provider_registry().unwrap();
    let menu = |id: &str| -> Vec<&'static str> {
        let descriptor = registry
            .generators()
            .map(|r| (r.descriptor)())
            .find(|d| d.id == id)
            .unwrap_or_else(|| panic!("{id} is not registered"));
        descriptor.capabilities.samplers
    };
    assert_eq!(menu("wan2_2_ti2v_5b"), MLX_FULL);
    for id in ["wan_vace", "wan2_2_vace_fun_14b"] {
        assert_eq!(menu(id), NATIVE, "{id}");
    }
    for id in ["wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
        assert_eq!(
            menu(id),
            NATIVE,
            "{id}: the Candle 14B denoise has no gen-core fold-in (MLX advertises one)"
        );
    }
}
