//! **sc-18317 — the declared FLUX.2 execution domains, and the weights-free proof that a route
//! reaches them.**
//!
//! Epic 18304's P2 planner selects a graph-evaluation cadence and an FFN chunk per request. The defect
//! class this file guards is the same one the ladder families guard for their rungs: a domain that is
//! *declared* on a descriptor and that no route actually *consumes*, or a value a provider quietly
//! drops instead of refusing.
//!
//! Three properties are asserted for **every** registered FLUX.2 descriptor, with no weights:
//!
//! 1. the declared [`ExecutionSurface`] is internally coherent (`declaration_errors`);
//! 2. a request selecting a declared cadence/chunk is **admitted** by the shared request floor every
//!    provider's `validate` delegates to — so the planner's selection reaches `generate`;
//! 3. a request selecting the one domain FLUX.2 does **not** implement (CFG batching — its guidance is
//!    embedded/distilled, not a two-branch classifier-free batch) is **refused by name**, rather than
//!    accepted and ignored.
//!
//! What happens after admission — the value arriving at the block loops and the result staying
//! equivalent — is `transformer_chunk_equiv.rs`'s job; the two together are the chain from request
//! field to consumer. Declaration/admission needs no snapshot at all, which is why it lives here and
//! not in an `#[ignore]`d real-weight harness that reports pass while skipping.

use mlx_gen::gen_core::{
    CfgBatching, ExecutionValueDomain, FfnChunk, GenerationMemory, GenerationRequest,
    GraphEvalCadence,
};

fn probe_request(memory: Option<GenerationMemory>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy forest, photograph".into(),
        width: 1024,
        height: 1024,
        count: 1,
        steps: Some(1),
        seed: Some(1234),
        memory,
        ..Default::default()
    }
}

/// Every registered FLUX.2 descriptor, as the catalog composes them.
fn descriptors() -> Vec<mlx_gen::gen_core::ModelDescriptor> {
    let registry = mlx_gen_flux2::provider_registry().expect("FLUX.2 provider registry");
    registry
        .generators()
        .map(|registration| (registration.descriptor)())
        .collect()
}

#[test]
fn every_flux2_descriptor_declares_a_coherent_execution_surface() {
    let descriptors = descriptors();
    assert!(
        !descriptors.is_empty(),
        "the FLUX.2 registry must expose at least one generator"
    );
    for descriptor in descriptors {
        let surface = &descriptor.capabilities.execution;
        assert!(
            surface.declaration_errors().is_empty(),
            "{}: {:?}",
            descriptor.id,
            surface.declaration_errors()
        );
        // The two levers `chunk::MemoryConfig` implements, and only those.
        assert_eq!(
            surface.graph_eval_cadence_blocks,
            ExecutionValueDomain::ANY_POSITIVE,
            "{}: cadence domain",
            descriptor.id
        );
        assert_eq!(
            surface.ffn_chunk_rows,
            ExecutionValueDomain::ANY_POSITIVE,
            "{}: FFN chunk domain",
            descriptor.id
        );
        assert!(
            !surface.cfg_batching.is_supported(),
            "{}: FLUX.2 guidance is embedded, so it must not advertise a CFG batching axis",
            descriptor.id
        );
    }
}

#[test]
fn a_declared_cadence_and_chunk_are_admitted_on_every_route() {
    for descriptor in descriptors() {
        for memory in [
            GenerationMemory {
                graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
                ..Default::default()
            },
            GenerationMemory {
                graph_eval_cadence: Some(GraphEvalCadence::new(8).unwrap()),
                ..Default::default()
            },
            GenerationMemory {
                ffn_chunk: Some(FfnChunk::new(4096).unwrap()),
                ..Default::default()
            },
            GenerationMemory {
                graph_eval_cadence: Some(GraphEvalCadence::new(4).unwrap()),
                ffn_chunk: Some(FfnChunk::new(2048).unwrap()),
                ..Default::default()
            },
        ] {
            descriptor
                .capabilities
                .validate_request(descriptor.id, &probe_request(Some(memory)))
                .unwrap_or_else(|error| {
                    panic!(
                        "{}: a declared selection must be admitted: {error}",
                        descriptor.id
                    )
                });
        }
        // And the historical shape — no execution selection at all — is unaffected.
        descriptor
            .capabilities
            .validate_request(descriptor.id, &probe_request(None))
            .unwrap_or_else(|error| panic!("{}: {error}", descriptor.id));
        descriptor
            .capabilities
            .validate_request(
                descriptor.id,
                &probe_request(Some(GenerationMemory::default())),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", descriptor.id));
    }
}

#[test]
fn an_undeclared_cfg_batching_selection_is_refused_by_name() {
    for descriptor in descriptors() {
        for mode in CfgBatching::ALL {
            let request = probe_request(Some(GenerationMemory {
                cfg_batching: Some(mode),
                ..Default::default()
            }));
            let error = descriptor
                .capabilities
                .validate_request(descriptor.id, &request)
                .expect_err("FLUX.2 must refuse a CFG batching selection");
            assert!(
                matches!(error, mlx_gen::gen_core::Error::Unsupported(_)),
                "{}: must be a capability gap, not a range error: {error:?}",
                descriptor.id
            );
            let message = error.to_string();
            assert!(
                message.contains("cfg_batching"),
                "{}: {message}",
                descriptor.id
            );
            assert!(
                message.contains(mode.label()),
                "{}: the refusal must name the rejected mode: {message}",
                descriptor.id
            );
            assert!(
                message.contains("unset"),
                "{}: the refusal must name the remedy: {message}",
                descriptor.id
            );
        }
    }
}
