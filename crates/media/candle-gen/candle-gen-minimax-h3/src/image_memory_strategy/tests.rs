use super::*;
use candle_gen::gen_core::{MemoryBudget, MemoryStrategyCapability, TransformerComponent};

fn spec(shape: LoadShape) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir("/minimax-h3-image".into())).with_load_shape(shape)
}

fn support(c: &MemoryProviderContract, strategy: MemoryStrategy) -> MemoryStrategySupport {
    c.capability(strategy).unwrap().support.clone()
}

/// AC1: a deferred load declares all four optimized rungs Implemented — staging, the shipped
/// tiled decode, bounded attention over its two-point chunk domain, and the one-block window over
/// both transformers — and the contract conforms. An eager load cannot stream, so rung 4 is
/// Missing there and every other rung is unchanged.
#[test]
fn a_deferred_load_declares_all_four_optimized_rungs() {
    let c = weights_free_contract(&spec(LoadShape::DeferredMaterialization)).unwrap();
    for strategy in MemoryStrategy::ALL {
        assert_eq!(
            support(&c, strategy),
            MemoryStrategySupport::Implemented,
            "{strategy:?}"
        );
    }
    assert!(
        c.conformance_errors().is_empty(),
        "{:?}",
        c.conformance_errors()
    );
    let params = |s| c.capability(s).unwrap().parameters.clone();
    assert_eq!(
        params(MemoryStrategy::BoundedDecode),
        crate::image_vae::bounded_decode_parameters()
    );
    assert_eq!(
        params(MemoryStrategy::BoundedAttention).attention_chunk_sizes,
        IMAGE_ATTENTION_CHUNK_SIZES.to_vec()
    );
    let window = params(MemoryStrategy::BoundedTransformerResidency);
    assert_eq!(
        window.transformer_window_sizes,
        vec![IMAGE_TRANSFORMER_WINDOW_SIZE]
    );
    // Staging is declared; the weights-free surface knows no block bytes to stream.
    let phase_facts = c.phase_facts.as_ref().unwrap();
    assert_eq!(
        phase_facts.staged_weights,
        gen_core::StagedWeightSchedule::ThreeStage
    );
    assert!(phase_facts.transformer_stream.is_none());
    assert_eq!(
        window.transformer_window_components,
        vec![TransformerComponent::Both]
    );
    assert!(c.lifecycle.attention_chunking && c.lifecycle.transformer_window_materialization);
    assert!(matches!(
        c.backend,
        MemoryBackendRealization::CandleCuda {
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
            ..
        }
    ));

    let eager = weights_free_contract(&spec(LoadShape::EagerMaterialization)).unwrap();
    assert_eq!(
        support(&eager, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Missing
    );
    assert_eq!(
        support(&eager, MemoryStrategy::BoundedAttention),
        MemoryStrategySupport::Implemented
    );
    assert!(!eager.lifecycle.transformer_window_materialization);
    assert!(eager.conformance_errors().is_empty());
    // The packed tiers use the same contract; only the asset bytes differ.
    for tier in [gen_core::Quant::Q4, gen_core::Quant::Q8] {
        let mut q = spec(LoadShape::DeferredMaterialization);
        q.quantize = Some(tier);
        assert!(weights_free_contract(&q).is_ok(), "{tier:?}");
    }
    assert!(!memory_contract_surface_specs().is_empty());
    assert!(validate_decode(Some(256), Some(64)).is_ok());
    assert!(validate_decode(Some(512), Some(64)).is_err());
}

/// The weights-free surface publishes no axis it could only have inferred from the id.
#[test]
fn the_weights_free_surface_publishes_no_architecture_axis() {
    let surface = LoadSpec::new(WeightsSource::Dir(
        "/__sceneworks_memory_contract_surface__".into(),
    ));
    assert!(weights_free_contract(&surface)
        .unwrap()
        .architecture_facts
        .is_empty());
}

fn fixture_context(
    contract: &MemoryProviderContract,
    load: &LoadSpec,
    strategy: MemoryStrategy,
) -> MemoryRunContext {
    (IMAGE_MEMORY_BEHAVIOR.valid_fixtures)(load, contract, strategy)
        .unwrap()
        .remove(0)
        .context
}

/// The route gate admits text-to-image and SW-7092's 1..=16-reference edit, at any positive size,
/// and refuses 17 references, references on a text-to-image request, and a chunk smaller than
/// one materializing query row of the admitted sequence.
#[test]
fn the_route_gate_admits_edit_and_any_size_and_refuses_an_unsplittable_chunk() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let tier = MemoryNumericTier {
        precision: load.precision,
        quant: None,
        component_precision_floors: &[],
    };
    let base = fixture_context(&c, &load, MemoryStrategy::BoundedAttention);
    assert!(route_gate(&c, &base, tier).is_ok());
    let mut odd = base.clone();
    odd.geometry.width = 1000;
    odd.geometry.height = 1001;
    assert!(route_gate(&c, &odd, tier).is_ok());
    let mut edit = base.clone();
    edit.mode = MemoryMode::Edit;
    edit.has_reference = true;
    edit.geometry.reference_count = 16;
    assert!(route_gate(&c, &edit, tier).is_ok());
    edit.geometry.reference_count = 17;
    assert!(route_gate(&c, &edit, tier).is_err());
    let mut t2i_refs = base.clone();
    t2i_refs.geometry.reference_count = 1;
    assert!(route_gate(&c, &t2i_refs, tier).is_err());
    // 16 MP: 20,480 keys x 56 heads is 1.15 Mi elements per query row.
    let mut tight = base.clone();
    tight.geometry.width = 4096;
    tight.geometry.height = 4096;
    tight.selection.parameters.attention_chunk_size = Some(1 << 20);
    assert!(route_gate(&c, &tight, tier).is_err());
    tight.selection.parameters.attention_chunk_size = Some(IMAGE_ATTENTION_CHUNK_SIZES[0]);
    assert!(route_gate(&c, &tight, tier).is_ok());
    // The smallest declared chunk holds one row of the largest admitted edit.
    let mut largest = tight.clone();
    largest.mode = MemoryMode::Edit;
    largest.has_reference = true;
    largest.geometry.width = 5120;
    largest.geometry.height = 4096;
    largest.geometry.reference_count = 16;
    assert!(route_gate(&c, &largest, tier).is_ok());
}

/// SW-7092: admission prices each edit reference as a 2048² VAE copy because the geometry carries
/// only their count, but the VAE copy caps the short edge alone — a 4096x2048 reference keeps
/// 8192 rows. The generator re-prices the admitted selection at the references' actual sizes: a
/// budget that holds the square bound exactly admits a 2048² reference and refuses the wide one.
#[test]
fn the_exact_reference_geometry_is_repriced_against_the_admitted_budget() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let facts = admission::shipped_scale_facts(0);
    let mut context = fixture_context(&c, &load, MemoryStrategy::BoundedTransformerResidency);
    context.mode = MemoryMode::Edit;
    context.has_reference = true;
    context.geometry.width = 2048;
    context.geometry.height = 2048;
    context.geometry.reference_count = 1;
    context.predicted_peak_bytes = 0;
    let bound = admission::estimate(
        &c,
        &facts,
        &admission::ImageRequestShape::with_reference_count(2048, 2048, 1),
        &context.selection,
    )
    .unwrap()
    .peak_device_bytes();
    context.budget = MemoryBudget {
        total_bytes: bound,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    assert!(estimate_gate(Some(&facts), &c, &context).is_ok());
    let request = |w: u32, h: u32| gen_core::GenerationRequest {
        prompt: "edit".into(),
        width: 2048,
        height: 2048,
        conditioning: vec![gen_core::Conditioning::MultiReference {
            images: vec![gen_core::Image {
                width: w,
                height: h,
                pixels: vec![0; (w * h * 3) as usize],
            }],
        }],
        ..Default::default()
    };
    assert!(admit_exact_request(Some(&facts), &c, &context, &request(2048, 2048)).is_ok());
    let wide = request(4096, 2048);
    let rows = admission::ImageRequestRows::of(
        &admission::ImageRequestShape::from_request(&wide).unwrap(),
    );
    assert_eq!(rows.reference_rows, vec![8192]);
    let e = admit_exact_request(Some(&facts), &c, &context, &wide)
        .unwrap_err()
        .to_string();
    assert!(e.contains("GPU budget"), "{e}");
    // The same wide reference against a chunk that holds one row of the square bound but not of
    // its real sequence is refused by the chunk check.
    let square_keys = admission::ImageRequestRows::of(
        &admission::ImageRequestShape::with_reference_count(2048, 2048, 1),
    )
    .sequence_rows();
    let heads = crate::dit::MiniMaxH3DitConfig::image().num_attention_heads as u64;
    let mut chunked = context.clone();
    chunked.budget.total_bytes = u64::MAX / 2;
    chunked.selection.parameters.attention_chunk_size = Some((square_keys * heads) as u32);
    assert!(admit_exact_request(Some(&facts), &c, &chunked, &request(2048, 2048)).is_ok());
    let e = admit_exact_request(Some(&facts), &c, &chunked, &wide)
        .unwrap_err()
        .to_string();
    assert!(e.contains("materializing attention row"), "{e}");
}

/// AC2's admission: given a simulated 40 GB of free device memory, the 16 MP request is refused at
/// the staged rung — its estimate holds the whole DiT — and admitted at the streamed one, so the
/// ladder lands on rung 4 instead of rejecting the request at the resident peak.
#[test]
fn admission_refuses_the_resident_peak_and_admits_the_streamed_route_at_16_mp() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let facts = admission::shipped_scale_facts(0);
    let budget = MemoryBudget {
        total_bytes: 40 * admission::BYTES_PER_GB,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let at = |strategy| {
        let mut context = fixture_context(&c, &load, strategy);
        context.geometry.width = 4096;
        context.geometry.height = 4096;
        context.budget = budget;
        context.predicted_peak_bytes = 0;
        context
    };
    let staged = at(MemoryStrategy::StagedResidency);
    let e = estimate_gate(Some(&facts), &c, &staged)
        .unwrap_err()
        .to_string();
    assert!(e.contains("GPU budget") && e.contains("GiB"), "{e}");
    assert!(matches!(
        safety_check_with(Some(&facts), &load, &c, &staged),
        MemorySafetyDecision::Reject { .. }
    ));
    let streamed = at(MemoryStrategy::BoundedTransformerResidency);
    assert!(estimate_gate(Some(&facts), &c, &streamed).is_ok());
    assert_eq!(
        safety_check_with(Some(&facts), &load, &c, &streamed),
        MemorySafetyDecision::Accept
    );
    // The weights-free surface has no source to price, and admits on the shared checks alone.
    assert_eq!(
        safety_check_with(None, &load, &c, &staged),
        MemorySafetyDecision::Accept
    );
}

/// The request scope installs the selected context as this thread's attention budget and the
/// window policy on the request, and clears both on drop — including after a failure.
#[test]
fn the_request_scope_applies_the_selected_chunk_and_window_and_cleans_up() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let context = fixture_context(&c, &load, MemoryStrategy::BoundedTransformerResidency);
    let chunk = context.selection.parameters.attention_chunk_size;
    assert_eq!(chunk, Some(IMAGE_ATTENTION_CHUNK_SIZES[0]));
    let mut scope = registered_begin_request(None, &load, &c, &context)
        .unwrap()
        .unwrap();
    assert_eq!(
        crate::nn::attention_budget().max_score_elements(),
        u64::from(chunk.unwrap())
    );
    // A second request on the same thread is refused while one is active.
    assert!(registered_begin_request(None, &load, &c, &context).is_err());
    let mut request = gen_core::GenerationRequest {
        width: context.geometry.width,
        height: context.geometry.height,
        ..Default::default()
    };
    scope.configure_request(&mut request).unwrap();
    let memory = request.memory.unwrap();
    assert!(memory.stream_transformer_blocks && memory.chunk_attention);
    assert_eq!(memory.transformer_window_size, Some(1));
    assert_eq!(
        memory.transformer_window_component,
        Some(TransformerComponent::Both)
    );
    drop(scope);
    assert!(crate::memory_strategy::active_context().is_none());
    assert_eq!(
        crate::nn::attention_budget().max_score_elements(),
        gen_core::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS
    );
}

/// Every implemented optimized rung has an executable behavior fixture.
#[test]
fn every_implemented_rung_has_a_behavior_fixture() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    for MemoryStrategyCapability { strategy, .. } in &c.strategies {
        if !strategy.is_optimized() {
            continue;
        }
        let fixtures = (IMAGE_MEMORY_BEHAVIOR.valid_fixtures)(&load, &c, *strategy).unwrap();
        assert!(!fixtures.is_empty(), "{strategy:?}");
        let scope = registered_begin_request(None, &load, &c, &fixtures[0].context).unwrap();
        drop(scope);
    }
}

/// Write a header-plus-zeros safetensors file of `(name, dtype, shape)` tensors.
fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize])]) {
    let width = |dtype: &str| if dtype == "F32" { 4 } else { 2 };
    let mut header = serde_json::Map::new();
    let mut offset = 0_usize;
    for (name, dtype, shape) in tensors {
        let len = shape.iter().product::<usize>() * width(dtype);
        header.insert(
            (*name).into(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len]}),
        );
        offset += len;
    }
    let json = serde_json::Value::Object(header).to_string().into_bytes();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.resize(bytes.len() + offset, 0);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// The VAE decode half is loaded with `DType::F32`, so a bf16-stored decode tensor is priced at
/// 4 bytes per element (the encoder half is not read); the DiT keeps its stored widths.
#[test]
fn asset_facts_price_the_vae_at_the_f32_it_is_loaded_at() {
    let root = tempfile::tempdir().unwrap();
    let te = root.path().join(crate::image_model::IMAGE_TEXT_ENCODER);
    std::fs::create_dir_all(&te).unwrap();
    std::fs::write(te.join("config.json"), "{}").unwrap();
    write_safetensors(
        &te.join("model.safetensors"),
        &[("unread.weight", "BF16", &[4])],
    );
    let dit = root.path().join(crate::image_model::IMAGE_DIT_PARTITION);
    write_safetensors(
        &dit.join("model.safetensors"),
        &[
            ("transformer_blocks.0.attn.to_q.weight", "BF16", &[8, 8]),
            ("pdd_proj_out.0.weight", "F32", &[4, 4]),
        ],
    );
    write_safetensors(
        &root
            .path()
            .join(IMAGE_VAE_COMPONENT)
            .join("model.safetensors"),
        &[
            ("decoder.proj_in.weight", "BF16", &[4, 8]),
            ("post_quant_conv.bias", "F32", &[4]),
            ("encoder.proj_in.weight", "BF16", &[1000]),
        ],
    );
    let facts = asset_facts(&dit, root.path(), &te).unwrap();
    assert_eq!(facts.decoder_bytes, (32 + 4) * VAE_F32_WIDTH);
    assert_eq!(facts.transformer_bytes, 64 * 2 + 16 * 4);
    assert_eq!(facts.conditioning_bytes, 0);
}

/// A non-32-aligned request the generator accepts is admitted at its requested size, and its
/// resources follow the canvas the generator renders on. The request scope still binds the
/// requested size: the canvas itself is not the admitted request.
#[test]
fn a_non_aligned_request_is_admitted_and_sized_at_its_canvas() {
    use gen_core::h3_image::{generation_canvas, MIN_GENERATION_AREA_DEFAULT};

    let canvas = admission_canvas(1000, 1001).unwrap();
    assert_eq!(
        canvas,
        generation_canvas(1000, 1001, MIN_GENERATION_AREA_DEFAULT)
    );
    assert_ne!(canvas, (1000, 1001));
    assert!(admission_canvas(1000, 0).is_err());

    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let mut context = fixture_context(&c, &load, MemoryStrategy::StagedResidency);
    context.geometry.width = 1000;
    context.geometry.height = 1001;
    assert_eq!(
        safety_check_with(None, &load, &c, &context),
        MemorySafetyDecision::Accept
    );
    let request = |width, height| gen_core::GenerationRequest {
        prompt: "a lighthouse at dusk".into(),
        width,
        height,
        ..Default::default()
    };
    let mut scope = registered_begin_request(None, &load, &c, &context)
        .unwrap()
        .unwrap();
    scope.configure_request(&mut request(1000, 1001)).unwrap();
    drop(scope);
    let mut scope = registered_begin_request(None, &load, &c, &context)
        .unwrap()
        .unwrap();
    assert!(scope
        .configure_request(&mut request(canvas.0, canvas.1))
        .is_err());
}

/// SW-7095 AC2's admission: under a budget between the 16 MP streamed and staged estimates,
/// `admission::admit` lands on rung 4, and that exact selection is what the generator's safety
/// check accepts at the same budget, while the staged rung is refused there. This is the
/// selection the real-weights 16 MP gate renders through.
#[test]
fn the_admitted_streamed_selection_is_accepted_at_the_budget_that_chose_it() {
    let load = spec(LoadShape::DeferredMaterialization);
    let c = weights_free_contract(&load).unwrap();
    let facts = admission::shipped_scale_facts(0);
    let shape = admission::ImageRequestShape::text_to_image(4096, 4096);
    let streamed_context = fixture_context(&c, &load, MemoryStrategy::BoundedTransformerResidency);
    let tier = streamed_context.selection.tier;
    let rungs = admission::ladder(&c, tier);
    let at = |strategy| {
        let selection = rungs.iter().find(|s| s.strategy == strategy).unwrap();
        admission::estimate(&c, &facts, &shape, selection).unwrap()
    };
    let staged = at(MemoryStrategy::StagedResidency);
    let streamed = at(MemoryStrategy::BoundedTransformerResidency);
    let budget = MemoryBudget {
        total_bytes: (staged.peak_device_bytes() + streamed.peak_device_bytes()) / 2,
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    };
    let admitted = admission::admit(&c, &facts, &shape, tier, budget).unwrap();
    assert_eq!(
        admitted.estimate.selection.strategy,
        MemoryStrategy::BoundedTransformerResidency
    );
    let in_budget = |mut context: MemoryRunContext| {
        context.geometry.width = 4096;
        context.geometry.height = 4096;
        context.budget = budget;
        context.predicted_peak_bytes = admitted.estimate.peak_device_bytes();
        context
    };
    let mut context = in_budget(streamed_context);
    context.selection = admitted.estimate.selection;
    assert_eq!(
        safety_check_with(Some(&facts), &load, &c, &context),
        MemorySafetyDecision::Accept
    );
    let staged_context = in_budget(fixture_context(&c, &load, MemoryStrategy::StagedResidency));
    assert!(matches!(
        safety_check_with(Some(&facts), &load, &c, &staged_context),
        MemorySafetyDecision::Reject { .. }
    ));
}

/// SW-7096 AC2: configured adapters do not take rung 4 away from the image generator. The
/// streamed window copies each host block with its installed residuals, so a deferred load with
/// LoRAs still declares bounded transformer residency Implemented, conforms, and accepts the
/// rung-4 fixture with the adapter bytes priced.
#[test]
fn rung_4_stays_implemented_with_adapters_configured() {
    let load = spec(LoadShape::DeferredMaterialization).with_adapters(vec![
        gen_core::AdapterSpec::new(
            "/nonexistent/style.safetensors".into(),
            0.75,
            gen_core::AdapterKind::Lora,
        ),
        gen_core::AdapterSpec::new(
            "/nonexistent/detail.safetensors".into(),
            0.5,
            gen_core::AdapterKind::Lora,
        ),
    ]);
    let c = weights_free_contract(&load).unwrap();
    assert_eq!(
        support(&c, MemoryStrategy::BoundedTransformerResidency),
        MemoryStrategySupport::Implemented
    );
    assert!(c.lifecycle.transformer_window_materialization);
    assert!(
        c.conformance_errors().is_empty(),
        "{:?}",
        c.conformance_errors()
    );
    let facts = admission::shipped_scale_facts(admission::BYTES_PER_GB);
    let context = fixture_context(&c, &load, MemoryStrategy::BoundedTransformerResidency);
    assert_eq!(
        safety_check_with(Some(&facts), &load, &c, &context),
        MemorySafetyDecision::Accept
    );
}
