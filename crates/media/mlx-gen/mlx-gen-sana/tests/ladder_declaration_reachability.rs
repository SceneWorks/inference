//! **SC-18607 — the declared MLX SANA ladder, and the weights-free proof that a route reaches it.**
//!
//! Epic 18304's declaration audit is about one defect class: a memory ladder that is *declared* and
//! that no route actually *reaches*. `mlx-gen-sana` already registered real contracts, but every
//! proof that the declaration matched what a load can execute lived in the `#[ignore]`d
//! `memory_ladder_real_weights` harness — which **skips and reports pass** in CI. A skipped proof is
//! not a proof, so the declaration was effectively ungated on every lane that actually runs.
//!
//! This file closes that gap without weights, for **both** catalog entries. It is deliberately not
//! a second copy of the real-weight harness: that file owns peaks, parity and evidence records, and
//! this one owns *declaration exactness* and *route reachability*, which need no snapshot at all.
//!
//! ## Why every assertion here is weights-free but still production-path
//!
//! Two properties of this provider make it possible:
//!
//! 1. a `Sequential` load captures loader closures and touches **no** component weights, so
//!    `provider_registry().load(id, spec)` over a nonexistent directory constructs the real,
//!    registry-resolved generator;
//! 2. `Sana::generate_impl` runs `validate_request` and then
//!    `memory_strategy::validate_request_memory` **before** the residency lifecycle opens anything.
//!
//! So a rung-4 request refused by name on a generator whose snapshot does not exist proves the
//! refusal is reached by the production `generate` seam, and a published rungs-1-3 request that
//! fails with a *different* error proves the refusal is specific rather than a blanket rejection.
//!
//! ## SANA's shape does not transfer from a sibling family
//!
//! `attn1` is ReLU **linear** attention with no score tensor at all, `attn2` hand-rolls its softmax
//! over a fixed 300-slot caption axis, and there is no RoPE anywhere. That is why SANA publishes its
//! own score budgets instead of the shared 64 Mi operating point, and why rung 4 is *implemented,
//! measured and WITHHELD* rather than absent: see `memory_strategy::TRANSFORMER_WINDOW_WITHHELD`.
//! Nothing here imports another family's axes.

use mlx_gen::gen_core::{
    GenerationMemory, GenerationRequest, MemoryStrategy, MemoryStrategySupport,
    TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, WeightsSource};
use mlx_gen_sana::memory_strategy as ms;

/// The pair. Sharing one provider module is explicitly not what makes an entry covered, so every
/// assertion in this file runs for both ids.
const ENTRIES: [&str; 2] = [mlx_gen_sana::MODEL_ID, mlx_gen_sana::SPRINT_MODEL_ID];

/// The published rungs 1-3 selection — the whole ceiling a selector can compose while rung 4 is
/// withheld.
fn published_selection() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        decode_tile_edge: Some(mlx_gen_sana::pipeline::DECODE_TILE_EDGE as u32),
        decode_overlap: Some(mlx_gen_sana::pipeline::DECODE_OVERLAP as u32),
        chunk_attention: true,
        attention_chunk_size: Some(ms::ATTENTION_CHUNK_SIZE),
        ..Default::default()
    }
}

/// A rung-4 selection, which the production path must refuse by name.
fn rung_four_selection(window: u32, component: TransformerComponent) -> GenerationMemory {
    GenerationMemory {
        stream_transformer_blocks: true,
        transformer_window_size: Some(window),
        transformer_window_component: Some(component),
        ..published_selection()
    }
}

/// A `Sequential` + `DeferredMaterialization` load over a directory that does not exist: the exact
/// shape that constructs the generator without opening a single weight file.
fn weightless_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(
        "/nonexistent/sc-18607-sana-declaration".into(),
    ))
    .with_offload_policy(OffloadPolicy::Sequential)
    .with_load_shape(LoadShape::DeferredMaterialization)
}

fn probe_request(memory: GenerationMemory) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        // 256 is the smallest advertised output and a multiple of the DC-AE 32x stride.
        width: 256,
        height: 256,
        count: 1,
        steps: Some(1),
        seed: Some(1234),
        memory: Some(memory),
        ..Default::default()
    }
}

/// Whether an error is one of the memory contract's own refusals (as opposed to the absent snapshot).
fn is_memory_refusal(error: &str) -> bool {
    error.contains("WITHHELD")
        || error.contains("outside the published domain")
        || error.contains("bounded transformer residency")
        || error.contains("attention_chunk_size")
        || error.contains("tile")
}

/// **Every declared surface, for both entries, says exactly what the engine implements.**
///
/// This walks the provider's own finite witness set — the same seam a generated capability dump
/// consumes — rather than one convenient `LoadSpec`, so a rung whose availability changes with
/// residency policy or materialization shape cannot hide behind a default.
#[test]
fn every_declared_surface_is_exact_for_both_entries() {
    let registry = mlx_gen_sana::provider_registry().expect("SANA provider registry");
    let surfaces = registry
        .memory_contract_surfaces()
        .expect("every SANA surface must construct weights-free");

    for id in ENTRIES {
        let mine: Vec<_> = surfaces
            .iter()
            .filter(|surface| surface.contract.provider_id == id)
            .collect();
        // 3 artifact tiers x {Resident, Sequential} x {Eager, Deferred}.
        assert_eq!(mine.len(), 12, "{id} must publish the whole MLX surface");

        let mut staged_surfaces = 0;
        for surface in &mine {
            let contract = &surface.contract;
            let selector = surface.selector;
            assert!(
                contract.conformance_errors().is_empty(),
                "{id} {}: {:?}",
                selector.id(),
                contract.conformance_errors()
            );
            assert!(!surface.composed, "{id} is a single-generator route");

            let support = |strategy| {
                &contract
                    .capability(strategy)
                    .unwrap_or_else(|| panic!("{id} {} omits {strategy:?}", selector.id()))
                    .support
            };
            let staged = selector.offload_policy == OffloadPolicy::Sequential;
            staged_surfaces += usize::from(staged);

            assert_eq!(
                support(MemoryStrategy::Resident),
                &MemoryStrategySupport::Implemented,
                "{id} {}",
                selector.id()
            );
            // Rung 1 is a LOAD-time mechanism here: the shared `Residency` seam is driven by
            // `spec.offload_policy`, so a Resident load genuinely cannot stage.
            assert_eq!(
                support(MemoryStrategy::StagedResidency),
                if staged {
                    &MemoryStrategySupport::Implemented
                } else {
                    &MemoryStrategySupport::Missing
                },
                "{id} {}",
                selector.id()
            );
            // Rungs 2 and 3 bound scratch, not residency, so they ride every selector.
            assert_eq!(
                support(MemoryStrategy::BoundedDecode),
                &MemoryStrategySupport::Implemented,
                "{id} {}",
                selector.id()
            );
            assert_eq!(
                support(MemoryStrategy::BoundedAttention),
                &MemoryStrategySupport::Implemented,
                "{id} {}",
                selector.id()
            );
            // Rung 4 is implemented, output-preserving, measured, and WITHHELD.
            assert_eq!(
                support(MemoryStrategy::BoundedTransformerResidency),
                &MemoryStrategySupport::Missing,
                "{id} {}",
                selector.id()
            );

            // The capability, the lifecycle hook and the published parameter domain are ONE binding.
            assert!(
                !contract.lifecycle.transformer_window_materialization,
                "{id} {}",
                selector.id()
            );
            let window = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert!(window.parameters.transformer_window_sizes.is_empty());
            assert!(window.parameters.transformer_window_components.is_empty());
            assert_eq!(contract.lifecycle.synchronized_phase_release, staged);
            assert!(contract.lifecycle.decode_tiling);
            assert!(contract.lifecycle.attention_chunking);

            // SANA's own budgets, not the sibling families' 64 Mi operating point.
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                ms::ATTENTION_CHUNK_SIZES.to_vec(),
                "{id} {}",
                selector.id()
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges,
                ms::DECODE_TILE_EDGES.to_vec(),
                "{id} {}",
                selector.id()
            );
            // A weights-free witness must inject no asset facts.
            assert_eq!(contract.asset_facts, Default::default(), "{id}");
        }
        assert_eq!(
            staged_surfaces, 6,
            "{id} stages on exactly the Sequential half"
        );
    }

    // The pair publishes the same ladder per selector — but that is asserted, never assumed from
    // the shared module.
    for base in surfaces
        .iter()
        .filter(|surface| surface.contract.provider_id == mlx_gen_sana::MODEL_ID)
    {
        let sprint = surfaces
            .iter()
            .find(|surface| {
                surface.contract.provider_id == mlx_gen_sana::SPRINT_MODEL_ID
                    && surface.selector == base.selector
            })
            .unwrap_or_else(|| panic!("sprint is missing selector {}", base.selector.id()));
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                base.contract.capability(strategy),
                sprint.contract.capability(strategy),
                "{} diverges at {strategy:?}",
                base.selector.id()
            );
        }
        assert_eq!(base.contract.calibration, sprint.contract.calibration);
    }
}

/// **The declared ladder is reached by the production `generate` path — for both entries, without
/// a single weight file.**
///
/// This is the story's core claim. A declaration that only a shared-contract admission call can see
/// is exactly the defect epic 18304 exists to fix, so the refusal and the admission are both driven
/// through `Generator::generate` on a registry-resolved generator.
#[test]
fn the_production_route_reaches_the_declared_ladder_for_both_entries() {
    for id in ENTRIES {
        let spec = weightless_spec();
        let model = mlx_gen_sana::provider_registry()
            .expect("SANA provider registry")
            .load(id, &spec)
            .unwrap_or_else(|error| {
                panic!("{id}: a Sequential load must construct the generator weights-free: {error}")
            });

        // The loaded generator exposes the exact contract, not a default.
        let contract = model
            .memory_strategy_contract()
            .unwrap_or_else(|| panic!("{id}: the loaded generator must expose its contract"));
        assert_eq!(contract.provider_id, id);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        // The published composition CLEARS the memory gate and then fails on the absent snapshot.
        // Without this half, a blanket rejection would satisfy every refusal assertion below.
        let admitted = model
            .generate(&probe_request(published_selection()), &mut |_| {})
            .expect_err("the snapshot does not exist, so the render itself must still fail")
            .to_string();
        assert!(
            !is_memory_refusal(&admitted),
            "{id}: the published rungs 1-3 composition was refused by the memory gate: {admitted}"
        );

        // …and every rung-4 selection is refused BY NAME on the same path. Published-shaped cadences
        // and unpublished ones alike: the rung is withheld, so the whole domain is out.
        let mut admitted_rung_four = Vec::new();
        for window in ms::TRANSFORMER_WINDOW_SIZES
            .iter()
            .copied()
            .chain([0_u32, 3, 6, 7, 11, 20, 21, 70])
        {
            match model.generate(
                &probe_request(rung_four_selection(
                    window,
                    ms::TRANSFORMER_WINDOW_COMPONENT,
                )),
                &mut |_| {},
            ) {
                Ok(_) => admitted_rung_four.push(format!("cadence {window} rendered")),
                Err(error) => {
                    let error = error.to_string();
                    if !error.contains("WITHHELD") {
                        admitted_rung_four
                            .push(format!("cadence {window} failed elsewhere: {error}"));
                    }
                }
            }
        }
        for component in [
            TransformerComponent::Dit,
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            let error = model
                .generate(
                    &probe_request(rung_four_selection(ms::TRANSFORMER_WINDOW_SIZE, component)),
                    &mut |_| {},
                )
                .map(|_| "rendered".to_owned())
                .unwrap_or_else(|error| error.to_string());
            if !error.contains("WITHHELD") {
                admitted_rung_four.push(format!("component {component:?}: {error}"));
            }
        }
        assert!(
            admitted_rung_four.is_empty(),
            "{id}: the production path did not refuse the withheld rung by name: \
             {admitted_rung_four:?}"
        );

        // Every rejected rung-2/rung-3 parameter is refused on the same path too, so the published
        // domains are enforced where a caller can actually set them.
        let mut silently_admitted = Vec::new();
        for rejected in ms::ATTENTION_CHUNK_SIZES_REJECTED
            .iter()
            .copied()
            .chain([0, 7, 999])
        {
            assert!(!ms::ATTENTION_CHUNK_SIZES.contains(&rejected));
            let memory = GenerationMemory {
                attention_chunk_size: Some(rejected),
                ..published_selection()
            };
            let error = model
                .generate(&probe_request(memory), &mut |_| {})
                .map(|_| "rendered".to_owned())
                .unwrap_or_else(|error| error.to_string());
            if !error.contains("outside the published domain") {
                silently_admitted.push(format!("attention budget {rejected}: {error}"));
            }
        }
        for rejected in ms::DECODE_TILE_EDGES_REJECTED {
            let memory = GenerationMemory {
                decode_tile_edge: Some(*rejected),
                ..published_selection()
            };
            let error = model
                .generate(&probe_request(memory), &mut |_| {})
                .map(|_| "rendered".to_owned())
                .unwrap_or_else(|error| error.to_string());
            if !is_memory_refusal(&error) {
                silently_admitted.push(format!("decode edge {rejected}: {error}"));
            }
        }
        assert!(
            silently_admitted.is_empty(),
            "{id}: the production path admitted unpublished parameters: {silently_admitted:?}"
        );

        // Every published value must remain reachable, or the gate above would be over-broad.
        for edge in ms::DECODE_TILE_EDGES {
            let memory = GenerationMemory {
                decode_tile_edge: Some(*edge),
                ..published_selection()
            };
            let error = model
                .generate(&probe_request(memory), &mut |_| {})
                .expect_err("the snapshot does not exist")
                .to_string();
            assert!(
                !is_memory_refusal(&error),
                "{id}: published decode edge {edge} was refused: {error}"
            );
        }
        for budget in ms::ATTENTION_CHUNK_SIZES {
            let memory = GenerationMemory {
                attention_chunk_size: Some(*budget),
                ..published_selection()
            };
            let error = model
                .generate(&probe_request(memory), &mut |_| {})
                .expect_err("the snapshot does not exist")
                .to_string();
            assert!(
                !is_memory_refusal(&error),
                "{id}: published budget {budget} was refused: {error}"
            );
        }
    }
}

/// **The registered behavior fixture round-trips into a selection the production path admits.**
///
/// The declaration, the registered request scope and the production gate are three separate pieces
/// of code that can disagree. This walks the whole loop for both entries: the provider's own fixture
/// for each implemented rung, through the registered scope's `configure_request`, into
/// `Generator::generate` on a real registry-resolved generator. A scope that wrote a selection the
/// production gate refuses would be a ladder nothing can execute — declared, unreachable.
#[test]
fn the_registered_scope_produces_a_selection_the_production_path_admits() {
    let registry = mlx_gen_sana::provider_registry().expect("SANA provider registry");
    for id in ENTRIES {
        let spec = weightless_spec();
        let contract = ms::memory_strategy_contract(id, &spec).expect("contract");
        let behavior = registry
            .memory_behavior_registrations()
            .find(|registration| registration.provider_id == id)
            .unwrap_or_else(|| panic!("{id} must register an executable memory behavior"));
        let model = registry.load(id, &spec).expect("weights-free load");

        let mut exercised = 0_usize;
        for strategy in MemoryStrategy::ALL {
            let fixtures = (behavior.valid_fixtures)(&spec, &contract, strategy).unwrap();
            let implemented = contract
                .capability(strategy)
                .map(|capability| &capability.support)
                == Some(&MemoryStrategySupport::Implemented)
                && strategy.is_optimized();
            if !implemented {
                assert!(
                    fixtures.is_empty(),
                    "{id}: {strategy:?} is not an implemented optimized rung but still has a fixture"
                );
                continue;
            }
            assert_eq!(fixtures.len(), 1, "{id} {strategy:?}");
            let fixture = &fixtures[0];
            assert!(!fixture.context.use_pid, "{id}: SANA has no PiD decoder");
            assert_eq!(fixture.context.geometry.reference_count, 0);

            assert_eq!(
                model.memory_strategy_safety_check(&fixture.context),
                mlx_gen::gen_core::MemorySafetyDecision::Accept,
                "{id} {strategy:?}"
            );
            let mut scope = model
                .begin_memory_strategy_request(&fixture.context)
                .unwrap()
                .unwrap_or_else(|| panic!("{id} {strategy:?} must open a request scope"));
            let mut request = fixture.request.clone();
            scope.configure_request(&mut request).unwrap();
            let memory = request
                .memory
                .unwrap_or_else(|| panic!("{id} {strategy:?} wrote no request memory"));
            assert!(
                !memory.stream_transformer_blocks,
                "{id} {strategy:?}: the withheld rung must never be selected"
            );

            // Round-trip the scope's own selection through the production gate.
            request.prompt = "a red fox in a snowy forest, photograph".into();
            request.steps = Some(1);
            let error = model
                .generate(&request, &mut |_| {})
                .expect_err("the snapshot does not exist")
                .to_string();
            assert!(
                !is_memory_refusal(&error),
                "{id} {strategy:?}: the registered scope wrote a selection the production path \
                 refuses: {error}"
            );
            exercised += 1;
        }
        // Rungs 1, 2 and 3 — the whole published optimized ceiling.
        assert_eq!(exercised, 3, "{id} must exercise every implemented rung");
    }
}

/// **A load the provider cannot execute is refused by the declaration AND by the loader.**
///
/// Both directions matter. A contract that described a load the loader rejects would declare an
/// unreachable ladder; a loader that accepted a component the contract refuses would run a shape
/// nothing priced.
#[test]
fn unloadable_shapes_are_refused_by_both_the_declaration_and_the_loader() {
    let registry = mlx_gen_sana::provider_registry().expect("SANA provider registry");
    let good = weightless_spec();

    let mut cases: Vec<(&str, LoadSpec)> = Vec::new();
    cases.push((
        "single-file source",
        LoadSpec::new(WeightsSource::File("/nonexistent/sana.safetensors".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization),
    ));
    let mut fp32 = good.clone();
    fp32.precision = mlx_gen::Precision::Fp32;
    cases.push(("precision override", fp32));
    let mut adapted = good.clone();
    adapted.adapters.push(mlx_gen::AdapterSpec::new(
        "/nonexistent/lora.safetensors".into(),
        1.0,
        mlx_gen::AdapterKind::Lora,
    ));
    cases.push(("adapter", adapted));
    let mut control = good.clone();
    control.control = Some(WeightsSource::File(
        "/nonexistent/control.safetensors".into(),
    ));
    cases.push(("control", control));
    let mut ip_adapter = good.clone();
    ip_adapter.ip_adapter = Some(WeightsSource::Dir("/nonexistent/ip".into()));
    cases.push(("ip adapter", ip_adapter));
    let mut text_encoder = good.clone();
    text_encoder.text_encoder = Some(WeightsSource::Dir("/nonexistent/external-text".into()));
    cases.push(("external text encoder", text_encoder));
    let mut component = good.clone();
    component.components.insert(
        "unexpected".to_owned(),
        WeightsSource::File("/nonexistent/unexpected.safetensors".into()),
    );
    cases.push(("named component", component));

    let mut admitted = Vec::new();
    for (label, spec) in &cases {
        for id in ENTRIES {
            if ms::memory_strategy_contract(id, spec).is_ok() {
                admitted.push(format!("{id} declared {label}"));
            }
            if registry.load(id, spec).is_ok() {
                admitted.push(format!("{id} loaded {label}"));
            }
        }
    }
    assert!(
        admitted.is_empty(),
        "these unloadable shapes were still admitted: {admitted:?}"
    );

    // The control spec itself must still be declarable and loadable, or the gate above proves
    // nothing.
    for id in ENTRIES {
        ms::memory_strategy_contract(id, &good).expect("the control spec must be declarable");
        registry
            .load(id, &good)
            .expect("the control spec must load");
    }
}
