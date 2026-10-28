/// Complements the real header/packing math tests: a stock literal at the render call site
/// must not survive merely because a helper can express the selected geometry.
#[test]
fn render_wires_selected_patch_and_actual_reference_channels_before_conditioning() {
    let src = include_str!("../model.rs")
        .split("\n#[cfg(test)]\n")
        .next()
        .unwrap();
    assert!(
        !src.contains("PATCH_SIZE"),
        "render conditioning cannot use the stock patch constant"
    );
    let compact: String = src.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(
        compact
            .matches("letpatch=self.task_dit_config(task)?.patch_size;")
            .count(),
        2
    );
    assert_eq!(
        compact.matches("keyframe_condition_rows(&c,patch,").count(),
        2
    );
    assert!(compact.contains("build_condition_rows(&vae,&pixels,&anchors,patch,"));
    assert!(compact.contains("reference_audio_channels.push(t.shape()[0]);"));
    assert!(compact.contains("ref2va_layout_with_audio_channels(geometry,&text_tags,&geometries,&reference_audio_channels,patch"));
}

#[test]
fn selected_task_patch_header_is_resolved_without_tensor_payloads() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("tier/transformer");
    let reference = base.with_file_name("transformer_ref");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::create_dir_all(&reference).unwrap();
    let model = generator_at(base.to_str().unwrap());
    let stock = r#"{"num_attention_heads":56,"attention_head_dim":128,"hidden_size":5376,"num_layers":50,"num_refiner_layers":2,"ffn_dim":14336,"in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],"text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":5376,"time_embed_dim":2688,"rope_freq_dim":16,"rope_theta":10000.0,"norm_eps":0.00001,"qk_norm_eps":0.00001,"final_norm_eps":0.00001}"#;
    std::fs::write(base.join("config.json"), stock).unwrap();
    std::fs::write(
        reference.join("config.json"),
        stock.replace("[1,2,2]", "[1,1,1]"),
    )
    .unwrap();
    // These selected tier directories have no shards: any accidental DiT load fails.
    assert_eq!(
        model
            .task_dit_config(MiniMaxH3Task::T2va)
            .unwrap()
            .patch_size,
        [1, 2, 2]
    );
    assert_eq!(
        model
            .task_dit_config(MiniMaxH3Task::Fl2va)
            .unwrap()
            .patch_size,
        [1, 2, 2]
    );
    assert_eq!(
        model
            .task_dit_config(MiniMaxH3Task::Ref2va)
            .unwrap()
            .patch_size,
        [1, 1, 1]
    );
    std::fs::write(
        base.join("config.json"),
        stock.replace("[1,2,2]", "[1,1,1]"),
    )
    .unwrap();
    assert_eq!(
        model
            .task_dit_config(MiniMaxH3Task::Fl2va)
            .unwrap()
            .patch_size,
        [1, 1, 1]
    );
    std::fs::remove_file(base.join("config.json")).unwrap();
    let error = model
        .task_dit_config(MiniMaxH3Task::T2va)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("no task config")
            && error.contains("transformer")
            && error.contains("reimport")
            && error.contains("select a partition"),
        "{error}"
    );
    assert!(
        error.contains(base.join("config.json").to_str().unwrap()),
        "{error}"
    );
    assert_eq!(
        model
            .task_dit_config(MiniMaxH3Task::Ref2va)
            .unwrap()
            .patch_size,
        [1, 1, 1]
    );
}

#[test]
fn malformed_veda_controls_fail_at_native_request_boundary() {
    for enabled in [false, true] {
        for bad in [
            mlx_gen::gen_core::VedaBudget::KeyTiles(0),
            mlx_gen::gen_core::VedaBudget::SparsityPercent(f64::NAN),
            mlx_gen::gen_core::VedaBudget::SparsityPercent(100.0),
        ] {
            let mut request = request(1344, 768);
            request.veda = Some(mlx_gen::gen_core::VedaRequest {
                enabled,
                generated: bad,
                ..Default::default()
            });
            let error = validate_request(&descriptor().capabilities, &request)
                .unwrap_err()
                .to_string();
            assert!(error.contains("generated: Veda"), "{error}");
        }
        for steps in [4, 8, 50] {
            let mut request = request(1344, 768);
            request.steps = Some(steps);
            request.veda = Some(mlx_gen::gen_core::VedaRequest {
                enabled,
                full_attention_steps: vec![steps as usize],
                ..Default::default()
            });
            let error = validate_request(&descriptor().capabilities, &request)
                .unwrap_err()
                .to_string();
            assert!(error.contains("zero-based index"), "{error}");
            request.veda.as_mut().unwrap().full_attention_steps = vec![steps as usize - 1];
            request.veda.as_mut().unwrap().full_attention_layers = vec![50];
            assert!(validate_request(&descriptor().capabilities, &request)
                .unwrap_err()
                .to_string()
                .contains("full_attention_layers"));
            request.veda.as_mut().unwrap().full_attention_layers = vec![49];
            validate_request(&descriptor().capabilities, &request).unwrap();
        }
    }
}

use super::*;
use mlx_gen::gen_core::CancelFlag;
use mlx_gen::runtime::{AdapterKind, MoeExpert};

#[test]
fn fl2va_phase_boundaries_preserve_the_actionable_metal_timeout() {
    const TIMEOUT: &str = "[METAL] Command buffer execution failed: Caused GPU Timeout Error \
             (00000002:kIOGPUCommandBufferCallbackErrorTimeout)";

    for phase in [FL2VA_GROUNDED_QWEN3_VL_PHASE, FL2VA_KEYFRAME_VAE_PHASE] {
        let error = in_fl2va_phase::<()>(phase, Err(Error::Msg(TIMEOUT.into())))
            .expect_err("injected Metal timeout must remain a failure")
            .to_string();
        assert!(error.starts_with(phase), "missing FL2VA phase: {error}");
        assert!(
            error.contains("kIOGPUCommandBufferCallbackErrorTimeout"),
            "the original driver classification was hidden: {error}"
        );
    }
}

#[test]
fn fl2va_phase_boundaries_do_not_stringify_typed_request_outcomes() {
    assert_eq!(
        in_fl2va_phase(FL2VA_KEYFRAME_VAE_PHASE, Ok(3.25_f32))
            .expect("a successful phase remains numerically transparent"),
        3.25
    );
    assert!(matches!(
        in_fl2va_phase::<()>(FL2VA_GROUNDED_QWEN3_VL_PHASE, Err(Error::Canceled)),
        Err(Error::Canceled)
    ));
    assert!(matches!(
        in_fl2va_phase::<()>(
            FL2VA_KEYFRAME_VAE_PHASE,
            Err(Error::Unsupported("fixture".into()))
        ),
        Err(Error::Unsupported(message)) if message == "fixture"
    ));
    assert!(matches!(
        in_fl2va_phase::<()>(
            FL2VA_KEYFRAME_VAE_PHASE,
            Err(Error::MissingTensor("fixture.weight".into()))
        ),
        Err(Error::MissingTensor(key)) if key == "fixture.weight"
    ));
}

fn request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a slow pan across a rainy street at night".into(),
        width,
        height,
        cancel: CancelFlag::default(),
        ..Default::default()
    }
}

/// A generator over a snapshot that does not exist, at an explicit load shape.
///
/// The contract is resolved from a spec carrying **that same shape** rather than being pinned
/// here, because `load` resolves the two from one `LoadSpec` and a fixture that let them
/// disagree would be testing a state the loader cannot produce (sc-18650).
pub(crate) fn generator_with(dit_dir: &str, load_shape: mlx_gen::gen_core::LoadShape) -> MiniMaxH3 {
    let spec =
        LoadSpec::new(WeightsSource::Dir(PathBuf::from("/snap"))).with_load_shape(load_shape);
    MiniMaxH3 {
        fixture: None,
        veda_component: None,
        veda_report: Default::default(),
        descriptor: descriptor(),
        root: PathBuf::from("/snap"),
        dit_dir: PathBuf::from(dit_dir),
        text_encoder_dir: PathBuf::from("/snap/text_encoder"),
        dtype: Dtype::Bfloat16,
        adapters: Vec::new(),
        load_shape,
        memory_strategy: crate::memory_strategy::contract_for(&spec).expect("contract"),
        memory_tier: crate::memory_strategy::numeric_tier(&spec),
    }
}

/// The shape rung 4 is `Implemented` at. NOT `LoadSpec::new`'s default — that is
/// `EagerMaterialization`; a caller must ask for the deferred shape, and the resident-load
/// refusal below constructs the eager twin explicitly.
fn generator_at(dit_dir: &str) -> MiniMaxH3 {
    generator_with(
        dit_dir,
        mlx_gen::gen_core::LoadShape::DeferredMaterialization,
    )
}

/// Write the structurally complete **weights-free** snapshot [`load`] probes for.
///
/// Every path here is stat'd or parsed as JSON by the loader; not one tensor is read. The
/// shards are **sparse** files of the measured component sizes with valid dense headers, so
/// tier detection parses real tensor facts while stat accounting preserves the family's asset
/// facts without materializing the sparse payload. That is what lets the assertions below distinguish the
/// production contract from the zero-footprint weights-free one.
fn weightless_snapshot(root: &Path) {
    for (dir, probe) in [
        (BASE_DIT_PARTITION, "config.json"),
        (REFERENCE_DIT_PARTITION, "config.json"),
        ("text_encoder", "config.json"),
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("tokenizer", "tokenizer.json"),
        ("FL2VA/audio_vae", "config.json"),
        ("FL2VA/audio_vae", "config.yaml"),
        ("FL2VA/audio_vae", "metadata.json"),
    ] {
        let dir = root.join(dir);
        std::fs::create_dir_all(&dir).expect("component dir");
        std::fs::write(dir.join(probe), b"{}").expect("probe document");
    }
    for (component, bytes) in [
        (BASE_DIT_PARTITION, crate::memory_strategy::DIT_BF16_BYTES),
        ("text_encoder", crate::memory_strategy::TEXT_ENCODER_BYTES),
        ("vae", crate::memory_strategy::VIDEO_VAE_BYTES),
        ("audio_vae", crate::memory_strategy::AUDIO_VAE_BYTES),
    ] {
        let mut shard =
            std::fs::File::create(root.join(component).join("model.safetensors")).expect("shard");
        use std::io::Write;
        // Header and both payload intervals cover the exact sparse file, including odd
        // measured lengths. A real BF16 tensor proves dense storage; U8 padding is not a
        // pretend floating parameter and must not alter stat-derived component charges.
        let mut padding = 0_u64;
        let header = loop {
            let header = format!(
                r#"{{"w":{{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}},"padding":{{"dtype":"U8","shape":[{padding:20}],"data_offsets":[2,{:20}]}}}}"#,
                padding + 2
            );
            let remaining = (bytes)
                .checked_sub(8 + header.len() as u64 + 2)
                .expect("fixture fits header");
            if padding == remaining {
                break header;
            }
            padding = remaining;
        };
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        shard.write_all(header.as_bytes()).unwrap();
        shard.set_len(bytes).expect("sparse shard");
    }
}

/// **The loaded generator publishes the provider's memory-strategy contract** (sc-18650).
///
/// This is the seam SceneWorks admission reads: `Generator::memory_strategy_contract` on the
/// boxed generator `load` hands back. `mlx_gen::impl_generator!` emits only `descriptor` /
/// `validate` / `generate`, so before this story the vtable slot resolved to the trait default
/// `None` and every MiniMax-H3 render skipped request-geometry admission entirely — while
/// `memory_strategy::contract_for` built a full contract that nothing ever asked for.
///
/// Asserted through `dyn Generator` rather than on the concrete struct, because the default is
/// what a `Box<dyn Generator>` would dispatch to.
///
/// Both load shapes are driven, because rung 4's support is the one entry that varies with the
/// shape — a single-shape test would pass against a contract pinned to either constant.
#[test]
fn a_loaded_generator_publishes_the_providers_memory_strategy_contract() {
    use mlx_gen::gen_core::{LoadShape, MemoryStrategy, MemoryStrategySupport};
    let root = tempfile::tempdir().expect("fixture root");
    weightless_snapshot(root.path());

    for (load_shape, rung4) in [
        (
            LoadShape::DeferredMaterialization,
            MemoryStrategySupport::Implemented,
        ),
        (
            LoadShape::EagerMaterialization,
            MemoryStrategySupport::Missing,
        ),
    ] {
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()))
            .with_load_shape(load_shape);
        let generator = load(&spec).expect("weights-free load");
        let published = generator
            .memory_strategy_contract()
            .expect("a loaded generator must publish its memory-strategy contract");

        // It is the provider's own contract for THIS spec — not some other shape's, and not the
        // resident-only compatibility default the trait falls back to.
        assert_eq!(
            published,
            &crate::memory_strategy::contract_for(&spec).expect("contract")
        );
        assert_eq!(published.load_shape, load_shape);
        // ...and the PRODUCTION one. The weights-free fixture contract differs from it only in
        // asset facts, so every assertion below would hold against it too; this is the one that
        // proves `load` resolved the snapshot it had just probed.
        assert_ne!(
            published,
            &crate::memory_strategy::weights_free_contract(&spec).expect("fixture contract"),
            "the published contract must carry the resolved snapshot's asset facts"
        );
        assert_eq!(
            published.asset_facts.transformer_bytes,
            crate::memory_strategy::DIT_BF16_BYTES
        );

        // The rungs it advertises are `memory_strategy::strategies()`'s, for this load shape.
        let support = |strategy: MemoryStrategy| {
            published
                .capability(strategy)
                .unwrap_or_else(|| panic!("{strategy:?} must appear in the strategy table"))
                .support
                .clone()
        };
        for implemented in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                support(implemented),
                MemoryStrategySupport::Implemented,
                "{implemented:?}"
            );
        }
        // Rung 3 is measured inapplicable on MLX for this family, at every shape.
        assert!(
            matches!(
                support(MemoryStrategy::BoundedAttention),
                MemoryStrategySupport::StructurallyNotApplicable { .. }
            ),
            "rung 3 must stay structurally inapplicable, got {:?}",
            support(MemoryStrategy::BoundedAttention)
        );
        // Rung 4 exists only where its shared prerequisite — the deferred shape — does.
        assert_eq!(
            support(MemoryStrategy::BoundedTransformerResidency),
            rung4,
            "{load_shape:?}"
        );
    }
}

/// **The published safety check and request scope refuse what this provider cannot run**
/// (sc-18650).
///
/// Publishing a contract without these two is worse than publishing nothing: the trait's own
/// `memory_strategy_safety_check` default would accept any selection the capability table
/// allows **without** this family's `route_gate`, and its `begin_memory_strategy_request`
/// default hard-errors for a contract declaring implemented optimized rungs. So each arm here
/// is a live guard, not a restatement of `memory_strategy`'s own tests:
///
/// * the accepted arm dies if `begin_memory_strategy_request` is dropped (the default errors);
/// * the geometry arms die if `memory_strategy_safety_check` is dropped (the default runs no
///   route gate and admits both);
/// * the rung-4 arm is the same context judged by two generators, so it cannot pass by
///   accident: it fails if the shape stops reaching the declaration.
#[test]
fn the_published_admission_refuses_selections_this_provider_cannot_serve() {
    use mlx_gen::gen_core::{LoadShape, MemorySafetyDecision, MemoryStrategy};
    let root = tempfile::tempdir().expect("fixture root");
    weightless_snapshot(root.path());
    let spec_at = |load_shape| {
        LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf())).with_load_shape(load_shape)
    };
    let deferred_spec = spec_at(LoadShape::DeferredMaterialization);
    let deferred = load(&deferred_spec).expect("weights-free load");
    let contract = deferred.memory_strategy_contract().expect("contract");

    // The provider's own fixture builder supplies the contexts, so every parameter is one this
    // contract published — a hand-rolled context would risk testing the parameter validator
    // instead of the rung.
    let fixture_for = |strategy| {
        crate::memory_strategy::registered_fixture(&deferred_spec, contract, strategy)
            .expect("fixtures")
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("{strategy:?} must publish at least one route fixture"))
    };

    // --- accepted: the legal rung-2 selection admits, and opens a real scope ----------------
    let admitted = fixture_for(MemoryStrategy::BoundedDecode).context;
    assert_eq!(
        deferred.memory_strategy_safety_check(&admitted),
        MemorySafetyDecision::Accept
    );
    assert!(
        deferred
            .begin_memory_strategy_request(&admitted)
            .expect("an admitted selection must open a scope, not error")
            .is_some(),
        "an implemented optimized rung must hand back a live request scope"
    );

    // --- refused: a rung this LOADED route does not have ------------------------------------
    // The identical context, judged by the eager twin whose contract declares rung 4 `Missing`.
    let streamed = fixture_for(MemoryStrategy::BoundedTransformerResidency).context;
    assert_eq!(
        deferred.memory_strategy_safety_check(&streamed),
        MemorySafetyDecision::Accept,
        "the deferred load implements rung 4"
    );
    let eager = load(&spec_at(LoadShape::EagerMaterialization)).expect("weights-free load");
    assert!(
        matches!(
            eager.memory_strategy_safety_check(&streamed),
            MemorySafetyDecision::Reject { .. }
        ),
        "a resident load must refuse block streaming, not silently accept it"
    );
    assert!(
        eager.begin_memory_strategy_request(&streamed).is_err(),
        "a refused selection must not open a scope"
    );

    // --- refused: a geometry the render itself would reject ---------------------------------
    // Off the `17n + 5` lattice. Only `route_gate` knows this, and only the override runs it.
    let mut off_lattice = admitted.clone();
    off_lattice.geometry.frames += 1;
    let MemorySafetyDecision::Reject { reason } =
        deferred.memory_strategy_safety_check(&off_lattice)
    else {
        panic!("an off-lattice frame count must be refused");
    };
    assert!(reason.contains("17n+5"), "{reason}");
    assert!(
        deferred
            .begin_memory_strategy_request(&off_lattice)
            .is_err(),
        "a refused geometry must not open a scope"
    );

    // --- refused: a decode tile outside the PUBLISHED PARAMETER DOMAIN -----------------------
    //
    // Deliberately not titled "`validate_decode_geometry` runs": it does not, and cannot, from
    // here. `contract.validate_selection` checks the owned parameter domains *before*
    // `route_gate` is ever called, and rung 2's published domain is the same singleton
    // `validate_decode_geometry` admits — so no value exists that clears one and trips the
    // other, and every out-of-domain tile is refused by the domain check. That predicate's own
    // seam is the request scope's `configure_decode`, which `memory_strategy`'s rung-2 chain
    // test drives end to end. What this arm pins is the domain check itself, by its message.
    //
    // Only the edge is moved: the fixture selection already carries both published parameters,
    // so this is a one-axis mutation of a complete, admitted selection rather than a selection
    // with a hole in it — and the assertion names which axis was refused.
    let mut retiled = admitted;
    retiled.selection.parameters.decode_tile_edge =
        Some(crate::memory_strategy::DECODE_TILE_EDGE + SPATIAL_STRIDE);
    let MemorySafetyDecision::Reject { reason } = deferred.memory_strategy_safety_check(&retiled)
    else {
        panic!("an out-of-domain decode tile must be refused");
    };
    assert!(
        reason.contains("outside the declared production candidates"),
        "{reason}"
    );

    // --- refused: rung 4 on an ADAPTER-CARRYING deferred load (sc-18650) ----------------------
    //
    // The blocker this arm exists for. `requested_transformer_window` has always refused a
    // windowed render with adapters, but until this story the contract did not know: `streamable`
    // read the load shape alone, so this exact spec published rung 4 `Implemented`, accepted the
    // selection, opened a scope, and only then hit a hard `Unsupported` inside `generate`.
    //
    // `MemoryRunContext` carries no adapter axis, so this cannot be a route-gate refusal — the
    // declaration is the only seam that can see it, which is why the assertion is on `support`
    // as well as on the two admission seams.
    let lora = root.path().join("turbo.safetensors");
    std::fs::write(&lora, b"\x00").expect("adapter file");
    let adapted_spec =
        spec_at(LoadShape::DeferredMaterialization).with_adapters(vec![AdapterSpec {
            path: lora,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }]);
    let adapted = load(&adapted_spec).expect("weights-free load with an adapter");
    let adapted_contract = adapted.memory_strategy_contract().expect("contract");
    assert_eq!(
        adapted_contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4 must appear in the strategy table")
            .support,
        mlx_gen::gen_core::MemoryStrategySupport::Missing,
        "an adapter load must declare rung 4 Missing — the window would drop the factors"
    );
    assert!(
        matches!(
            adapted.memory_strategy_safety_check(&streamed),
            MemorySafetyDecision::Reject { .. }
        ),
        "an adapter load must refuse block streaming at the safety check"
    );
    assert!(
        adapted.begin_memory_strategy_request(&streamed).is_err(),
        "an adapter load must not open a streaming scope"
    );
    // ...and the adapter's resident bytes are charged rather than declared away (sc-18650).
    assert_eq!(
        adapted_contract.asset_facts.overlay_bytes, 1,
        "the forward-time residual factors are resident for the whole render"
    );
    assert_eq!(
        contract.asset_facts.overlay_bytes, 0,
        "a render with no adapter has no overlay"
    );
}

/// **The rung-4 admission mirrors the contract in both directions** (sc-18662).
///
/// `requested_transformer_window` is the single seam every render's residency decision goes
/// through — `generate_impl` resolves it once and threads it to the conditioning, the DiT load
/// and the `JointDit` construction on all three routes. Each guard is mutated individually:
/// a streamed request must come back `Some(1)` (not `None` — a silent downgrade to the
/// resident path would be this test's false green), and each refusal must name its reason.
#[test]
fn transformer_streaming_is_admitted_and_refused_exactly_as_declared() {
    use mlx_gen::gen_core::{GenerationMemory, LoadShape, TransformerComponent};
    let streamed = |memory: GenerationMemory| GenerationRequest {
        prompt: "a slow pan across a rainy street at night".into(),
        memory: Some(memory),
        ..Default::default()
    };
    let stream_request = GenerationMemory {
        stage_residency: true,
        stream_transformer_blocks: true,
        transformer_window_size: Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TransformerComponent::Both),
        ..Default::default()
    };
    let generator = generator_at("/snap/transformer");

    // The happy path admits the published window — `Some`, never a silent `None`.
    assert_eq!(
        generator
            .requested_transformer_window(&streamed(stream_request))
            .expect("the declared selection must be admitted"),
        Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
    );
    // An untouched request is byte-for-byte unaffected: no memory block, no window.
    assert_eq!(
        generator
            .requested_transformer_window(&request(576, 320))
            .expect("a request without a memory block is the resident path"),
        None
    );
    // `stream_transformer_blocks: false` is the resident path even with parameters present.
    assert_eq!(
        generator
            .requested_transformer_window(&streamed(GenerationMemory {
                stream_transformer_blocks: false,
                ..stream_request
            }))
            .expect("an unstreamed selection is the resident path"),
        None
    );

    // A resident load genuinely does not have the rung (its contract declares `Missing`).
    let resident = generator_with("/snap/transformer", LoadShape::EagerMaterialization);
    let refusal = resident
        .requested_transformer_window(&streamed(stream_request))
        .expect_err("a resident load must refuse streaming, not downgrade it")
        .to_string();
    assert!(refusal.contains("DeferredMaterialization"), "{refusal}");

    // Adapters would be silently dropped from every windowed block — typed refusal.
    let adapted = MiniMaxH3 {
        adapters: vec![AdapterSpec {
            path: PathBuf::from("/nonexistent.safetensors"),
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        ..generator_at("/snap/transformer")
    };
    let refusal = adapted
        .requested_transformer_window(&streamed(stream_request))
        .expect_err("a streamed render with adapters must be refused")
        .to_string();
    assert!(refusal.contains("adapter"), "{refusal}");

    // The component defaults to `Dit` (SC-15794), which is deliberately outside this
    // provider's published domain — so both the explicit and the defaulted forms refuse.
    for component in [None, Some(TransformerComponent::Dit)] {
        let refusal = generator
            .requested_transformer_window(&streamed(GenerationMemory {
                transformer_window_component: component,
                ..stream_request
            }))
            .expect_err("a Dit-only window must be refused (AC3)")
            .to_string();
        assert!(refusal.contains("Both"), "{refusal}");
    }

    // A window outside the measured singleton domain is an advertised lever that does nothing.
    let refusal = generator
        .requested_transformer_window(&streamed(GenerationMemory {
            transformer_window_size: Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE + 1),
            ..stream_request
        }))
        .expect_err("an out-of-domain window must be refused")
        .to_string();
    assert!(
        refusal.contains("outside the published domain"),
        "{refusal}"
    );
    // `None` means the provider's own default and is admitted.
    assert_eq!(
        generator
            .requested_transformer_window(&streamed(GenerationMemory {
                transformer_window_size: None,
                ..stream_request
            }))
            .expect("a defaulted window size uses the provider's constant"),
        Some(crate::memory_strategy::TRANSFORMER_WINDOW_SIZE)
    );
}

/// **`decode_video` consults the request's rung-2 selection, and does so before it maps the
/// 10.42 GB VAE** (sc-18660 AC4).
///
/// This is the call-site half of reachability. `memory_strategy` proves the selection travels
/// as far as `pipeline::decode_tiling_for`; nothing there can prove `decode_video` *calls* it —
/// a doc comment saying so is not evidence. This drives the real method on a root with no
/// weights at all and pins **which** error comes back:
///
/// * an unadmitted geometry ⇒ the tiling refusal, proving the resolution ran;
/// * an admitted geometry ⇒ the missing-`config.json` error, proving the refusal is not a
///   blanket failure and that the method proceeds to the load.
///
/// Deleting the `decode_tiling_for` call turns the first arm into the second and fails here.
#[test]
fn decode_video_resolves_the_requests_tiling_before_it_loads_the_vae() {
    use mlx_gen::gen_core::GenerationMemory;
    let generator = generator_at("/snap/transformer");
    let latents = mlx_rs::Array::from_slice(&[0.0f32; 24], &[1, 24, 1, 1, 1]);
    let request = |edge: u32, overlap: u32| GenerationRequest {
        prompt: "a slow pan across a rainy street at night".into(),
        memory: Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(edge),
            decode_overlap: Some(overlap),
            ..Default::default()
        }),
        ..Default::default()
    };

    let refused = generator
        .decode_video(&request(128, 32), &latents)
        .expect_err("an unadmitted tile must be refused")
        .to_string();
    assert!(
        refused.contains("bounded decode admits only the reference geometry"),
        "expected the rung-2 refusal, got: {refused}"
    );
    assert!(
        !refused.contains("config.json"),
        "the refusal must fire BEFORE the VAE load, got: {refused}"
    );

    // A starved overlap takes the same path — the corruption it causes is cross-frame and no
    // memory number could see it, so admission is the only place it can be stopped.
    let starved = generator
        .decode_video(
            &request(crate::memory_strategy::DECODE_TILE_EDGE, 0),
            &latents,
        )
        .expect_err("a zero overlap must be refused")
        .to_string();
    assert!(starved.contains("bounded decode admits only the reference geometry"));

    // The control arm: an ADMITTED geometry gets past the resolution and fails at the load,
    // which is what proves the refusal above is specific rather than a blanket rejection.
    let loaded = generator
        .decode_video(
            &request(
                crate::memory_strategy::DECODE_TILE_EDGE,
                crate::memory_strategy::DECODE_OVERLAP,
            ),
            &latents,
        )
        .expect_err("there are no weights at /snap")
        .to_string();
    assert!(
        loaded.contains("config.json"),
        "an admitted geometry must proceed to the load, got: {loaded}"
    );
}

/// `ref2va` reads its partition from **beside the resolved DiT**, not from under the root.
///
/// The sc-17149 / sc-17150 seam. Before tiering, `ref2va` loaded `root.join("transformer_ref")`
/// and that was right because `dit_dir` was always `root/transformer`. On a tiered install the
/// DiT is staged outside the snapshot root entirely — `root/transformer_ref` does not exist, and
/// resolving there would either fail at load or, worse, find a stale dense checkpoint next to a
/// `q4` base and silently mix tiers across two branches of the same render.
#[test]
fn the_reference_partition_follows_the_tier_not_the_root() {
    // Flat upstream layout: the sibling IS the root-relative path, so nothing regresses.
    let flat = generator_at("/snap/transformer");
    assert_eq!(
        flat.task_dit_dir(MiniMaxH3Task::Ref2va),
        PathBuf::from("/snap/transformer_ref")
    );
    assert_eq!(
        flat.task_dit_dir(MiniMaxH3Task::T2va),
        PathBuf::from("/snap/transformer")
    );

    // Tiered/split layout: both partitions come from the SAME tier directory, and neither is
    // under `/snap`. This is the case `root.join(partition)` got wrong.
    let tiered = generator_at("/tiers/q4/transformer");
    assert_eq!(
        tiered.task_dit_dir(MiniMaxH3Task::Ref2va),
        PathBuf::from("/tiers/q4/transformer_ref")
    );
    assert_eq!(
        tiered.task_dit_dir(MiniMaxH3Task::T2va),
        PathBuf::from("/tiers/q4/transformer")
    );
    assert_eq!(
        tiered.task_dit_dir(MiniMaxH3Task::Fl2va),
        tiered.task_dit_dir(MiniMaxH3Task::T2va),
        "fl2va shares the base partition"
    );

    // The two partitions are never the same directory — the whole point of the split.
    for dir in ["/snap/transformer", "/tiers/q8/transformer"] {
        let g = generator_at(dir);
        assert_ne!(
            g.task_dit_dir(MiniMaxH3Task::Ref2va),
            g.task_dit_dir(MiniMaxH3Task::T2va)
        );
    }
}

/// The descriptor's shape: a Mac-only video model with no guidance surface and no conditioning.
#[test]
fn the_descriptor_declares_a_guidance_free_video_model() {
    let d = descriptor();
    assert_eq!(d.id, MODEL_ID);
    assert_eq!(d.backend, "mlx");
    assert!(matches!(d.modality, Modality::Video));
    assert!(!d.capabilities.supports_guidance);
    assert!(!d.capabilities.supports_negative_prompt);
    assert!(!d.capabilities.supports_true_cfg);
    assert_eq!(d.capabilities.max_count, 1);
    assert_eq!(d.capabilities.min_size, SPATIAL_STRIDE);
    assert!(d.capabilities.mac_only);
    assert!(
        d.capabilities.audio_sample_rates.is_empty(),
        "the audio surface describes an audio REQUEST, which a video model has none of"
    );
}

/// **The geometry gate fires from `validate`, before any weight is read.** This is the
/// acceptance criterion's "rejected, not silently refit" at the contract boundary.
#[test]
fn validate_rejects_off_lattice_geometry() {
    let caps = descriptor().capabilities;

    // An explicit off-lattice frame count.
    for frames in [123u32, 125, 129, 200] {
        let req = GenerationRequest {
            frames: Some(frames),
            ..request(576, 320)
        };
        let e = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(e.contains("17n + 5"), "{frames}: {e}");
    }
    // A canvas that is 16-aligned but not 32-aligned.
    let e = validate_request(&caps, &request(592, 320))
        .unwrap_err()
        .to_string();
    assert!(e.contains("multiple of 32"), "{e}");

    // ...and the legal request passes.
    validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(124),
            ..request(576, 320)
        },
    )
    .unwrap();
}

/// Every resolution SceneWorks advertises for `minimax_h3` and `minimax_h3_ref`.
///
/// Copied from `config/manifests/builtin.models.jsonc` — both partitions declare this exact
/// list, and it drives the resolution `<select>` in `apps/web/src/resolutionOverride.js`. A
/// bucket the menu offers and the engine refuses is a submit error the user cannot avoid, so
/// this list is the acceptance criterion rather than an illustration.
const ADVERTISED_BUCKETS: [(u32, u32); 9] = [
    (1536, 672),
    (672, 1536),
    (1344, 768),
    (768, 1344),
    (1024, 768),
    (768, 1024),
    (768, 768),
    (576, 320),
    (320, 576),
];

/// **The per-edge ceiling is the widest canvas the model's own resolver emits** (sc-17152).
///
/// The number is *derived here*, not restated: a maintained literal inside the test that exists
/// to catch a wrong literal proves nothing. The sweep walks the model's whole legal aspect
/// range and asks [`crate::keyframe::resolve_canvas_size`] — the arithmetic that turns a
/// reference's ratio into a canvas — for each canvas, then asserts `max_size` equals the widest
/// edge that came back. A ceiling below it would refuse a canvas the model resolves to on its
/// own; above it would advertise an edge nothing can produce.
#[test]
fn the_capability_ceiling_is_the_widest_canvas_the_resolver_emits() {
    const STEPS: u32 = 40_000;
    let mut widest = 0u32;
    for i in 0..=STEPS {
        let ratio = crate::keyframe::MIN_ASPECT_RATIO
            + (crate::keyframe::MAX_ASPECT_RATIO - crate::keyframe::MIN_ASPECT_RATIO)
                * f64::from(i)
                / f64::from(STEPS);
        let (h, w) = crate::keyframe::resolve_canvas_size(
            ratio,
            1.0,
            SPATIAL_STRIDE as i32,
            crate::pipeline::CANVAS_SHORT_EDGE as i32,
            i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
        )
        .unwrap();
        widest = widest.max(w as u32).max(h as u32);
    }
    assert_eq!(
        descriptor().capabilities.max_size,
        widest,
        "max_size must be the widest edge `resolve_canvas_size` can emit across 1:4..=4:1"
    );
    assert_eq!(descriptor().capabilities.max_size, MAX_CANVAS_EDGE);

    // The advertised menu must fit under it — this is the defect the story was filed for.
    let advertised = ADVERTISED_BUCKETS
        .iter()
        .flat_map(|&(w, h)| [w, h])
        .max()
        .unwrap();
    assert!(
        advertised <= descriptor().capabilities.max_size,
        "SceneWorks advertises a {advertised} px edge; the engine ceiling is {}",
        descriptor().capabilities.max_size
    );
}

/// **Every advertised bucket validates** — the user-visible half of sc-17152.
#[test]
fn every_advertised_resolution_bucket_validates() {
    let caps = descriptor().capabilities;
    for (w, h) in ADVERTISED_BUCKETS {
        // The manifest list must itself be inside the checkpoint's envelope, or the engine is
        // right to refuse and the manifest is the thing that is wrong.
        assert!(
            u64::from(w) * u64::from(h) <= u64::from(crate::pipeline::CANVAS_MAX_PIXELS),
            "{w}x{h} is over the area budget"
        );
        validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(w, h)
            },
        )
        .unwrap_or_else(|e| panic!("{w}x{h} is advertised but refused: {e}"));
    }
}

/// **Raising the per-edge ceiling did not widen the area envelope** (sc-17152).
///
/// Both shapes here are *inside* `max_size` on both edges and 32-aligned, so the per-edge gate
/// and the stride gate are both satisfied and only the area gate can refuse them — asserted
/// explicitly, because a refusal that came from the edge gate would make this test pass while
/// proving the opposite of what it claims. The error is named rather than checked with
/// `is_err`.
#[test]
fn an_over_area_canvas_is_still_refused_by_the_area_gate() {
    let caps = descriptor().capabilities;
    for (w, h) in [(1536u32, 1536u32), (1344u32, 1344u32)] {
        assert!(
            w <= caps.max_size && h <= caps.max_size,
            "{w}x{h} must be INSIDE the per-edge ceiling or this proves nothing"
        );
        assert!(w.is_multiple_of(SPATIAL_STRIDE) && h.is_multiple_of(SPATIAL_STRIDE));
        assert!(u64::from(w) * u64::from(h) > u64::from(crate::pipeline::CANVAS_MAX_PIXELS));

        let e = validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(w, h)
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("canvas budget"), "{w}x{h}: {e}");
        assert!(
            !e.contains("outside supported range"),
            "{w}x{h} must be refused by the AREA gate, not the per-edge gate: {e}"
        );
    }

    // The one shape the raised ceiling newly admits: the 4:1 canvas the resolver itself emits.
    // The short edge is derived — the ceiling divides the area budget exactly, which is what
    // makes `MAX_CANVAS_EDGE x short` sit at the budget rather than over it.
    let short = crate::pipeline::CANVAS_MAX_PIXELS / MAX_CANVAS_EDGE;
    assert_eq!(short * MAX_CANVAS_EDGE, crate::pipeline::CANVAS_MAX_PIXELS);
    assert!(short.is_multiple_of(SPATIAL_STRIDE));
    for (w, h) in [(MAX_CANVAS_EDGE, short), (short, MAX_CANVAS_EDGE)] {
        validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(w, h)
            },
        )
        .unwrap_or_else(|e| panic!("{w}x{h} is the resolver's own 4:1 canvas: {e}"));
    }

    // …and one stride past it on the long edge is outside the per-edge ceiling again.
    let e = validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(124),
            ..request(MAX_CANVAS_EDGE + SPATIAL_STRIDE, short)
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("outside supported range"), "{e}");
}

/// sc-18729 — **the video sigma shift is a per-request override, and the default is a byte-exact
/// no-op.**
///
/// Three claims, because two of them are individually satisfiable by a wrong implementation:
///
/// 1. `scheduler_shift: None` builds exactly what `JointSchedule::new` builds. An implementation
///    that quietly defaulted to, say, 6.0 would still "honor the knob" by every other check
///    here — this is the arm that pins the base checkpoint's schedule unchanged.
/// 2. `Some(6.0)` moves the **video** grid and leaves the **audio** grid alone. Asserting only
///    that something changed would pass an implementation that fed one shift to both modalities,
///    which is the sc-17146 defect: it moves audio 1.81e-1 at cosine 0.9846.
/// 3. A non-positive shift is refused at `validate`, not 20 minutes later inside
///    `SigmaSchedule::with_shift` — `validate` is the only gate that runs before the 53 GB text
///    encoder maps.
#[test]
fn the_video_sigma_shift_is_overridable_and_defaults_to_the_published_grid() {
    let caps = descriptor().capabilities;

    // (1) The default path is the published pair, exactly.
    let default = JointSchedule::with_shifts(9, VIDEO_SIGMA_SHIFT, AUDIO_SIGMA_SHIFT).unwrap();
    assert_eq!(
        default,
        JointSchedule::new(9).unwrap(),
        "an unset scheduler_shift must reproduce the base checkpoint's schedule exactly"
    );

    // (2) The turbo 768p shift moves the video grid and only the video grid.
    let turbo = JointSchedule::with_shifts(5, 6.0, AUDIO_SIGMA_SHIFT).unwrap();
    assert_ne!(
        turbo.video(),
        default.video(),
        "shift 6.0 must produce a different video sigma grid than 12.0"
    );
    assert_eq!(
        turbo.audio().shift(),
        AUDIO_SIGMA_SHIFT,
        "the audio shift must stay at its published 3.0 — the override is video-only"
    );

    // (3) Refused at the contract boundary, with the knob named.
    for bad in [0.0f32, -1.0] {
        let req = GenerationRequest {
            frames: Some(124),
            scheduler_shift: Some(bad),
            ..request(576, 320)
        };
        let e = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(
            e.contains("scheduler_shift") && e.contains("positive"),
            "shift {bad}: {e}"
        );
    }
    // ...and the documented turbo value passes.
    validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(124),
            scheduler_shift: Some(6.0),
            ..request(576, 320)
        },
    )
    .expect("the documented 768p turbo shift must validate");
}

/// sc-23402 — `validate` is wired to the reference-image short-edge gate, so an out-of-range
/// value is refused **here**, before the 53 GB text encoder maps.
///
/// Deleting the call in `validate_request` leaves the contract-level test in `gen-core` green
/// and reds only this one; that is the point of asserting it through THIS entry point.
#[test]
fn validate_refuses_a_reference_image_short_edge_outside_1024_to_2048() {
    let caps = descriptor().capabilities;
    let with = |edge: u32| GenerationRequest {
        frames: Some(124),
        reference_image_short_edge: Some(edge),
        ..request(576, 320)
    };

    for bad in [1023u32, 2049] {
        let e = validate_request(&caps, &with(bad)).unwrap_err().to_string();
        assert!(
            e.contains("reference_image_short_edge")
                && e.contains(&bad.to_string())
                && e.contains("1024..=2048"),
            "{bad}: {e}"
        );
    }
    validate_request(&caps, &with(1024)).expect("1024 is admitted");
    validate_request(&caps, &with(2048)).expect("2048 is admitted");
    validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(124),
            ..request(576, 320)
        },
    )
    .expect("an absent knob is vacuous");
}

/// `frames` is exact and `duration` is aligned — the split, pinned in both directions.
#[test]
fn frames_are_exact_and_duration_is_aligned() {
    // 5.3 s has no lattice point; it aligns UP to 141 frames.
    let by_duration = GenerationRequest {
        duration: Some(5.3),
        ..request(576, 320)
    };
    assert_eq!(
        request_geometry(&by_duration).unwrap().joint.num_frames,
        141
    );

    // The same clip requested as 127 frames is refused rather than aligned.
    let by_frames = GenerationRequest {
        frames: Some(127),
        ..request(576, 320)
    };
    assert!(request_geometry(&by_frames).is_err());

    // `frames` wins when both are given.
    let both = GenerationRequest {
        frames: Some(124),
        duration: Some(15.0),
        ..request(576, 320)
    };
    assert_eq!(request_geometry(&both).unwrap().joint.num_frames, 124);

    // No frames, no duration: the shortest legal render.
    assert_eq!(
        request_geometry(&request(576, 320))
            .unwrap()
            .joint
            .num_frames,
        crate::pipeline::SMALLEST_LEGAL_FRAMES
    );
}

/// The model generates at 24 fps and says so rather than quietly retiming.
#[test]
fn a_foreign_frame_rate_is_rejected() {
    let req = GenerationRequest {
        fps: Some(30),
        ..request(576, 320)
    };
    let e = request_geometry(&req).unwrap_err().to_string();
    assert!(e.contains("24 fps"), "{e}");
}

fn image(w: u32, h: u32) -> mlx_gen::media::Image {
    mlx_gen::media::Image {
        width: w,
        height: h,
        pixels: vec![0u8; (w * h * 3) as usize],
    }
}

fn image_ref() -> Conditioning {
    Conditioning::Reference {
        image: image(64, 64),
        strength: None,
    }
}

fn track() -> mlx_gen::media::AudioTrack {
    mlx_gen::media::AudioTrack {
        samples: vec![0.0; 64],
        // The engine ships no resampler, so `Ref2VaReferences::new` admits only this rate.
        sample_rate: AUDIO_SAMPLE_RATE,
        channels: 1,
        stems: Vec::new(),
    }
}

fn clip_ref() -> Conditioning {
    Conditioning::ReferenceVideo {
        frames: vec![image(64, 64)],
        fps: 24.0,
        audio: None,
    }
}

fn audio_ref() -> Conditioning {
    Conditioning::ReferenceAudio {
        audio: track(),
        strength: None,
    }
}

fn with(conditioning: Vec<Conditioning>) -> GenerationRequest {
    GenerationRequest {
        frames: Some(124),
        conditioning,
        ..request(576, 320)
    }
}

/// **The task selects the checkpoint, and only `ref2va` moves.**
///
/// The two partitions are byte-different but structurally identical — same config, same 638
/// tensor names — so this mapping is the *only* thing standing between a `ref2va` request and a
/// plausible render off the wrong 66 GB of weights.
#[test]
fn the_task_picks_the_partition_and_only_ref2va_moves() {
    assert_eq!(
        MiniMaxH3Task::resolve(false, false).unwrap(),
        MiniMaxH3Task::T2va
    );
    assert_eq!(
        MiniMaxH3Task::resolve(true, false).unwrap(),
        MiniMaxH3Task::Fl2va
    );
    assert_eq!(
        MiniMaxH3Task::resolve(false, true).unwrap(),
        MiniMaxH3Task::Ref2va
    );

    assert_eq!(MiniMaxH3Task::T2va.partition(), "transformer");
    assert_eq!(MiniMaxH3Task::Fl2va.partition(), "transformer");
    assert_eq!(MiniMaxH3Task::Ref2va.partition(), "transformer_ref");
    // Stated as an inequality too: if someone "simplifies" the match into one arm, the equality
    // assertions above could still be satisfied by a constant.
    assert_ne!(
        MiniMaxH3Task::Ref2va.partition(),
        MiniMaxH3Task::T2va.partition(),
        "ref2va must NOT read the base checkpoint"
    );

    // The vision tower is read by both grounded tasks and by neither text-only one.
    assert!(!MiniMaxH3Task::T2va.needs_vision_tower());
    assert!(MiniMaxH3Task::Fl2va.needs_vision_tower());
    assert!(MiniMaxH3Task::Ref2va.needs_vision_tower());

    // Both at once is refused rather than resolved to one of them.
    let e = MiniMaxH3Task::resolve(true, true).unwrap_err().to_string();
    assert!(e.contains("both keyframes and references"), "{e}");
}

/// **Every `ref2va` cap is enforced at the engine boundary**, by `validate`, before a weight is
/// read — the story's second acceptance criterion.
#[test]
fn the_reference_caps_are_enforced_by_validate() {
    let caps = descriptor().capabilities;
    let err = |c: Vec<Conditioning>| {
        validate_request(&caps, &with(c))
            .expect_err("expected a rejection")
            .to_string()
    };

    // 10 images.
    let e = err(vec![image_ref(); 10]);
    assert!(e.contains("at most 9 image"), "{e}");
    // 4 clips.
    let e = err(vec![clip_ref(); 4]);
    assert!(e.contains("at most 3 video"), "{e}");
    // 4 audio (paired, so the audio-alone rule is not what fires).
    let mut four_audio = vec![image_ref()];
    four_audio.extend(vec![audio_ref(); 4]);
    let e = err(four_audio);
    assert!(e.contains("at most 3 audio"), "{e}");
    // 13 total, with every per-modality cap satisfied.
    let mut thirteen = vec![image_ref(); 9];
    thirteen.extend(vec![clip_ref(); 3]);
    thirteen.push(audio_ref());
    let e = err(thirteen);
    assert!(e.contains("at most 12 references in total"), "{e}");

    // …and the legal saturated shapes are ACCEPTED. Without these arms every assertion above
    // would still pass with a gate that rejected everything.
    let mut twelve = vec![image_ref(); 9];
    twelve.extend(vec![clip_ref(); 2]);
    twelve.push(audio_ref());
    validate_request(&caps, &with(twelve)).expect("9 + 2 + 1 = 12 is legal");
    validate_request(&caps, &with(vec![image_ref(); 9])).expect("9 images is legal");
    validate_request(&caps, &with(vec![clip_ref(); 3])).expect("3 clips is legal");
}

/// An in-context clip is **refused**, not silently ignored (sc-17149). This is the arm that
/// makes dropping `ConditioningKind::VideoClip` from the descriptor a real gate rather than a
/// cosmetic edit: `request_references` no longer reads the variant, so without default-deny a
/// `VideoClip` would sail through and render a plausible clip that dropped the caller's
/// conditioning entirely.
#[test]
fn an_in_context_video_clip_is_refused_as_unsupported() {
    let caps = descriptor().capabilities;
    let e = validate_request(
        &caps,
        &with(vec![Conditioning::VideoClip {
            frames: vec![image(64, 64)],
            frame_idx: 7,
            strength: 0.5,
        }]),
    )
    .unwrap_err();
    assert!(
        matches!(e, Error::Unsupported(_)),
        "an unadvertised kind must be the typed Unsupported, got {e:?}"
    );
}

/// A reference clip carries **its own frame rate**, and the rate survives the mapping.
///
/// This is the hole the old `VideoClip` carrier could not close: it had no rate, so the mapping
/// read `req.fps` — which `request_geometry` rejects unless it is exactly 24.0. A 30 fps
/// reference was therefore conditioned on as 24 fps with nothing raised. Asserting `30.0`
/// *while the request's own fps stays unset* is what pins that the two are separate quantities.
#[test]
fn a_reference_clip_carries_its_own_frame_rate() {
    let req = with(vec![Conditioning::ReferenceVideo {
        frames: vec![image(64, 64)],
        fps: 30.0,
        audio: None,
    }]);
    assert!(req.fps.is_none(), "the request's own output rate is unset");
    let refs = request_references(&req).unwrap().unwrap();
    match &refs.as_slice()[0] {
        Ref2VaReference::Video(v) => assert!(
            (v.fps - 30.0).abs() < f64::EPSILON,
            "the clip's own rate must survive, got {}",
            v.fps
        ),
        other => panic!("expected a video reference, got {other:?}"),
    }
    // And it is legal: a reference does not bind the output rate, so a 30 fps reference on a
    // 24 fps render is a valid request rather than a geometry conflict.
    validate_request(&descriptor().capabilities, &req).expect("a 30 fps reference is legal");
}

/// A reference clip's **own soundtrack** reaches the engine on the clip, not as a standalone
/// audio reference — the distinction the whole carrier exists for.
///
/// The two assertions are one claim each and both are needed: the soundtrack is *present* on the
/// video reference, and it did *not* also become a reference of its own (which is what sending
/// it as a separate `ReferenceAudio` would have done, consuming an audio cap slot and taking its
/// own rotary origin instead of sharing the clip's).
#[test]
fn a_reference_clips_own_soundtrack_rides_the_clip() {
    let req = with(vec![Conditioning::ReferenceVideo {
        frames: vec![image(64, 64)],
        fps: 24.0,
        audio: Some(track()),
    }]);
    let refs = request_references(&req).unwrap().unwrap();
    assert_eq!(refs.as_slice().len(), 1, "one reference, not two");
    match &refs.as_slice()[0] {
        Ref2VaReference::Video(v) => {
            assert!(v.audio.is_some(), "the soundtrack must ride the clip")
        }
        other => panic!("expected a video reference, got {other:?}"),
    }
    assert_eq!(
        refs.audio_count(),
        0,
        "a clip's own soundtrack must not consume a standalone audio-reference slot"
    );
    validate_request(&descriptor().capabilities, &req)
        .expect("a lone video reference with its own soundtrack is legal");

    // The same waveform sent as a *standalone* reference is a different request: two
    // references, and one of them does consume an audio slot.
    let standalone = with(vec![clip_ref(), audio_ref()]);
    let refs = request_references(&standalone).unwrap().unwrap();
    assert_eq!(refs.as_slice().len(), 2);
    assert_eq!(refs.audio_count(), 1);
}

/// A request carrying keyframes **and** references is refused at the boundary — the two are
/// different tasks on different checkpoints.
#[test]
fn keyframes_and_references_together_are_refused() {
    let caps = descriptor().capabilities;
    let e = validate_request(
        &caps,
        &with(vec![
            Conditioning::Keyframe {
                image: image(576, 320),
                frame_idx: 0,
                strength: 1.0,
            },
            image_ref(),
        ]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("both keyframes and references"), "{e}");
}

/// sc-19571 — **refuse, do not ignore.** MiniMax-H3 anchors at the checkpoint's trained-in
/// `KEYFRAME_NOISE_AUG_T`, so `Keyframe.strength` has nothing to weight here; a request asking
/// for a partial pin must be told so rather than rendered as a full one. Gated in BOTH places a
/// request can arrive: `validate_request` (before the 53 GB text encoder maps) and
/// `keyframe_anchors` (the render's own path), so neither can be the only one carrying it.
///
/// Mutation guard: delete the `reject_keyframe_strength` call in `validate_request` and the
/// first `unwrap_err` panics; delete the one in `keyframe_anchors` and the second does.
#[test]
fn a_keyframe_conditioning_strength_is_refused_not_ignored() {
    let caps = descriptor().capabilities;
    let kf = |strength: f32| Conditioning::Keyframe {
        image: image(576, 320),
        frame_idx: 0,
        strength,
    };
    // A full pin — the only thing the checkpoint expresses — is admitted.
    assert!(validate_request(&caps, &with(vec![kf(1.0)])).is_ok());

    let e = validate_request(&caps, &with(vec![kf(0.6)]))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("conditioning strength is not supported") && e.contains("0.999"),
        "{e}"
    );

    // …and the render path refuses it independently of validate.
    let img = image(576, 320);
    let refs = [KeyframeRef {
        image: &img,
        frame_idx: 0,
        strength: 0.6,
    }];
    let e = keyframe_anchors(&refs, 124).unwrap_err().to_string();
    assert!(e.contains("conditioning strength is not supported"), "{e}");
    // `keyframe_images` re-runs the same validation, so it cannot disagree.
    assert!(keyframe_images(&refs, 124).is_err());
}

/// The reference list keeps **request order** across modalities — order is semantic for
/// `ref2va`, and re-sorting it would silently rewrite the request.
#[test]
fn the_reference_list_preserves_request_order_across_modalities() {
    let req = with(vec![clip_ref(), image_ref(), audio_ref(), image_ref()]);
    let refs = request_references(&req).unwrap().unwrap();
    let kinds: Vec<_> = refs.as_slice().iter().map(|r| r.kind()).collect();
    assert_eq!(
        kinds,
        vec![
            crate::reference::ReferenceKind::Video,
            crate::reference::ReferenceKind::Image,
            crate::reference::ReferenceKind::Audio,
            crate::reference::ReferenceKind::Image,
        ],
        "grouping by modality would silently reorder a ref2va request"
    );
    // A request with no reference at all is `None`, which is what makes this the t2va/fl2va
    // discriminator too.
    assert!(request_references(&request(576, 320)).unwrap().is_none());
}

/// An empty prompt and an out-of-range step count are contract errors.
#[test]
fn malformed_requests_are_rejected() {
    let caps = descriptor().capabilities;
    let blank = GenerationRequest {
        prompt: "   ".into(),
        ..request(576, 320)
    };
    assert!(validate_request(&caps, &blank).is_err());
    let too_many = GenerationRequest {
        steps: Some(MAX_STEPS + 1),
        ..request(576, 320)
    };
    assert!(validate_request(&caps, &too_many).is_err());
    let none = GenerationRequest {
        steps: Some(0),
        ..request(576, 320)
    };
    assert!(validate_request(&caps, &none).is_err());
}

/// A missing partition is a load error naming the path, not a mid-render surprise.
#[test]
fn a_snapshot_missing_a_partition_fails_at_load() {
    let dir = tempfile::tempdir().unwrap();
    // `Box<dyn Generator>` is not `Debug`, so the error is destructured rather than unwrapped.
    let message = |spec: &LoadSpec| match load(spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("this spec must not load"),
    };
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let e = message(&spec);
    assert!(e.contains("transformer"), "{e}");
    // A single file is the wrong shape entirely.
    let file = LoadSpec::new(WeightsSource::File(dir.path().join("x.safetensors")));
    let e = message(&file);
    assert!(e.contains("directory"), "{e}");
}

/// sc-18724 — the adapter surface is **declared and reachable**, and the two shared-spec knobs
/// this model cannot honor are refused rather than silently ignored.
///
/// `supports_lora` alone proves nothing: before this slice `load` rejected every adapter spec
/// outright, so a `true` there would have advertised a path no request could take.
#[test]
fn the_adapter_surface_is_declared_and_the_unusable_knobs_are_refused() {
    let caps = descriptor().capabilities;
    assert!(caps.supports_lora);
    assert!(caps.supports_lokr);

    let dir = tempfile::tempdir().unwrap();
    let message = |spec: &LoadSpec| match load(spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("this spec must not load"),
    };
    let missing = dir.path().join("nope.safetensors");
    let with = |adapter: AdapterSpec| {
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        spec.adapters = vec![adapter];
        spec
    };

    // A mistyped path fails at load, naming the file — not 20 minutes into a render.
    let e = message(&with(AdapterSpec::new(
        missing.clone(),
        1.0,
        AdapterKind::Lora,
    )));
    assert!(e.contains("does not exist"), "{e}");
    assert!(e.contains("nope.safetensors"), "{e}");

    // A real file gets past the adapter gate and fails on the (absent) snapshot instead, which
    // is what proves the gate no longer refuses adapters wholesale.
    let real = dir.path().join("real.safetensors");
    std::fs::write(&real, b"not-really-safetensors").unwrap();
    let e = message(&with(AdapterSpec::new(
        real.clone(),
        1.0,
        AdapterKind::Lora,
    )));
    assert!(
        e.contains("transformer") && !e.contains("adapter"),
        "an existing adapter must fall through to the snapshot probe; got {e}"
    );

    // LTX's per-pass strengths and Wan's MoE expert selector have no meaning here.
    let mut per_pass = AdapterSpec::new(real.clone(), 1.0, AdapterKind::Lora);
    per_pass.pass_scales = Some(vec![1.0, 0.5]);
    let e = message(&with(per_pass));
    assert!(e.contains("per-pass scales"), "{e}");
    let mut expert = AdapterSpec::new(real, 1.0, AdapterKind::Lora);
    expert.moe_expert = Some(MoeExpert::High);
    let e = message(&with(expert));
    assert!(e.contains("MoE expert"), "{e}");
}

/// sc-18724 — the render-path seam: `load_task_dit` really folds the configured adapters into
/// the DiT it maps, on the partition the task selects. The install itself is gated in
/// `tests/turbo_lora.rs`; what this adds is that the **generator** reaches it at all.
///
/// `MINIMAX_H3_TURBO_DIT` is a `transformer/` directory (any tier) and `MINIMAX_H3_TURBO_LORA`
/// the downloaded `lightx2v/Minimax-h3-Turbo` dir.
#[test]
#[ignore = "needs a real MiniMax-H3 transformer/ (MINIMAX_H3_TURBO_DIT) + the turbo LoRA \
                (MINIMAX_H3_TURBO_LORA) + Metal"]
fn the_render_path_folds_the_configured_adapters() {
    use mlx_gen::adapters::AdaptableHost;

    let dit_dir = std::env::var("MINIMAX_H3_TURBO_DIT").unwrap_or_default();
    assert!(
        !dit_dir.is_empty(),
        "MINIMAX_H3_TURBO_DIT must point at a MiniMax-H3 `transformer/` directory"
    );
    let lora_dir = std::env::var("MINIMAX_H3_TURBO_LORA").unwrap_or_default();
    assert!(
        !lora_dir.is_empty(),
        "MINIMAX_H3_TURBO_LORA must point at a lightx2v/Minimax-h3-Turbo snapshot dir"
    );
    let lora = PathBuf::from(lora_dir).join("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors");
    assert!(lora.is_file(), "missing {}", lora.display());

    let mut gen = generator_at(&dit_dir);
    gen.adapters = vec![AdapterSpec::new(lora, 1.0, AdapterKind::Lora)];
    let mut dit = gen
        .load_task_dit(MiniMaxH3Task::T2va)
        .unwrap_or_else(|e| panic!("load_task_dit: {e}"));

    // Every one of the 312 published targets carries exactly one residual, reached through the
    // generator rather than by calling the installer directly.
    let cfg = dit.config().clone();
    let mut adapted = 0usize;
    for path in crate::adapters::adapter_target_paths(&cfg) {
        let segs: Vec<&str> = path.split('.').collect();
        let lin = dit
            .adaptable_mut(&segs)
            .unwrap_or_else(|| panic!("unreachable target {path}"));
        assert_eq!(lin.adapters().len(), 1, "{path}");
        adapted += 1;
    }
    println!("[render seam] {adapted} module(s) folded via MiniMaxH3::load_task_dit");
    assert_eq!(adapted, 312);
    crate::model_inference_device::record_model_inference_device();
}

/// The requested step count is model EVALUATIONS; the schedule adds the terminal σ = 0.
#[test]
fn the_requested_step_count_is_evaluations() {
    let s = JointSchedule::new(DEFAULT_STEPS as usize + 1).unwrap();
    assert_eq!(s.num_evals(), DEFAULT_STEPS as usize);
    assert_eq!(DEFAULT_STEPS, 50);
}
