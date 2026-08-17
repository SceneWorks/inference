//! **sc-18322 — the declared Wan approximate capabilities, and the weights-free proof that selection
//! is refused rather than silently ignored.**
//!
//! Epic 18304's P7 adds one *result-changing* mechanism — a denoise feature cache — behind a contract
//! that binds selection to a quality characterization. Since no characterization artifact family
//! exists yet, the shipped state is **declared + implemented + refused**, and the defect class this
//! file guards is different from the execution-domain suites' by exactly that:
//!
//! 1. the declared [`ApproximationSurface`] is internally coherent, and declares the mechanism on the
//!    dense 5B **and nowhere else** — the MoE 14B and VACE providers have no wired route, so declaring
//!    there would be a claim no code backs;
//! 2. an unset request is admitted unchanged and resolves to
//!    [`ApproximationPlan::Exact`](mlx_gen::gen_core::ApproximationPlan::Exact) on every descriptor —
//!    the request-side half of byte-identical-when-off;
//! 3. every approximate selection is **refused by name**, on the declaring provider and the
//!    non-declaring ones alike, with the two refusals distinguished: an absent mechanism and an
//!    uncharacterized one have different fixes.
//!
//! What happens after the (unreachable) admission — the mechanism reusing a residual, staying
//! byte-identical when off, and keeping the cancel/eval cadence — is
//! `src/feature_cache.rs`'s in-crate tensor suite over the checked-in 2-block fixture. Declaration and
//! refusal need no weights at all, which is why they live here and not in an `#[ignore]`d harness that
//! reports pass while skipping.

use mlx_gen::gen_core::{
    ApproximationPlan, ApproximationRequest, CacheReuseInterval, CacheWarmupSteps,
    CharacterizationRef, Error, FeatureCacheDomain, FeatureCachePolicy, GenerationRequest,
    ModelDescriptor,
};

/// The dense 5B — the only Wan provider whose denoise route implements the cache.
const DECLARING_ID: &str = mlx_gen_wan::MODEL_ID;

fn descriptors() -> Vec<ModelDescriptor> {
    let registry = mlx_gen_wan::provider_registry().expect("Wan provider registry");
    registry
        .generators()
        .map(|registration| (registration.descriptor)())
        .collect()
}

fn probe_request(approximation: Option<ApproximationRequest>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a paper boat drifting down a rain gutter".into(),
        width: 640,
        height: 480,
        count: 1,
        steps: Some(4),
        seed: Some(1234),
        approximation,
        ..Default::default()
    }
}

fn policy(interval: u32, warmup: u32) -> FeatureCachePolicy {
    FeatureCachePolicy::new(CacheReuseInterval::new(interval).expect("declared interval"))
        .with_warmup(CacheWarmupSteps::new(warmup))
}

#[test]
fn exactly_the_dense_5b_declares_a_coherent_denoise_feature_cache() {
    let descriptors = descriptors();
    assert!(
        !descriptors.is_empty(),
        "the Wan registry must expose at least one generator"
    );
    let mut declaring = 0usize;
    for descriptor in &descriptors {
        let surface = &descriptor.capabilities.approximation;
        assert!(
            surface.declaration_errors().is_empty(),
            "{}: {:?}",
            descriptor.id,
            surface.declaration_errors()
        );
        // No provider is SELECTABLE — declaration plus a characterization binding — and none can be
        // until the terminal measurement campaign defines an artifact family. This is the assertion
        // that would redden if someone ever made the uncharacterized path reachable.
        assert!(
            !surface.is_selectable(),
            "{}: no approximate mechanism may be selectable before a characterization artifact \
             family exists",
            descriptor.id
        );
        if descriptor.id == DECLARING_ID {
            declaring += 1;
            assert_eq!(
                surface.denoise_feature_cache,
                FeatureCacheDomain::Implemented {
                    intervals: vec![2, 3, 4],
                    max_warmup_steps: 8,
                },
                "the dense 5B's declared domain must be the mechanism's implemented operating points"
            );
        } else {
            // The MoE 14B providers swap experts mid-trajectory and VACE runs two sequential B=1
            // forwards per step; neither route threads a cache, so neither may declare one.
            assert!(
                surface.is_inert(),
                "{}: this provider has no wired feature-cache route and must declare none",
                descriptor.id
            );
        }
    }
    assert_eq!(
        declaring, 1,
        "exactly one Wan descriptor declares the denoise feature cache"
    );
}

#[test]
fn an_unset_request_is_admitted_and_resolves_to_the_exact_plan() {
    for descriptor in descriptors() {
        for approximation in [None, Some(ApproximationRequest::default())] {
            let request = probe_request(approximation);
            descriptor
                .capabilities
                .validate_request(descriptor.id, &request)
                .unwrap_or_else(|error| {
                    panic!(
                        "{}: an unset approximation must be admitted: {error}",
                        descriptor.id
                    )
                });
            let plan = descriptor
                .capabilities
                .approximation_plan(descriptor.id, &request)
                .unwrap_or_else(|error| panic!("{}: {error}", descriptor.id));
            assert_eq!(
                plan,
                ApproximationPlan::Exact,
                "{}: an unset approximation must resolve to the exact path",
                descriptor.id
            );
        }
    }
}

#[test]
fn a_declared_selection_is_refused_for_want_of_a_characterization() {
    let descriptors = descriptors();
    let declaring: Vec<&ModelDescriptor> = descriptors
        .iter()
        .filter(|descriptor| descriptor.id == DECLARING_ID)
        .collect();
    assert_eq!(declaring.len(), 1, "the dense 5B must be registered");
    for descriptor in declaring {
        // Every corner of the declared domain, so the refusal cannot be an accident of one value
        // sitting outside it.
        for (interval, warmup) in [(2, 0), (3, 4), (4, 8)] {
            let request = probe_request(Some(ApproximationRequest::feature_cache(policy(
                interval, warmup,
            ))));
            let error = descriptor
                .capabilities
                .validate_request(descriptor.id, &request)
                .expect_err("a declared-but-uncharacterized selection must be refused");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "a capability gap, not a range error: {error:?}"
            );
            let message = error.to_string();
            assert!(
                message.contains("quality-characterization artifact reference"),
                "the refusal must name the missing precondition: {message}"
            );
            assert!(
                message.contains("unset"),
                "the refusal must name the remedy: {message}"
            );
            // And NOT the absent-mechanism refusal: this provider does implement the cache, and
            // reporting otherwise would send a caller to implement something that already exists.
            assert!(
                !message.contains("declares no denoise feature cache mechanism"),
                "a declared mechanism must not be refused as absent: {message}"
            );
        }

        // Supplying a reference does not help, because no artifact family is bound — the structural
        // gate. The refusal must be the binding one, naming the reference it could not validate.
        let request = probe_request(Some(ApproximationRequest {
            denoise_feature_cache: Some(policy(2, 0)),
            characterization: Some(
                CharacterizationRef::new("wan-5b-trunk-cache", "sha256:0").expect("reference"),
            ),
        }));
        let message = descriptor
            .capabilities
            .validate_request(descriptor.id, &request)
            .expect_err("a reference cannot be validated with no bound family")
            .to_string();
        assert!(message.contains("binds no"), "{message}");
        assert!(
            message.contains("implemented but not yet selectable"),
            "the refusal must distinguish unimplemented from uncharacterized: {message}"
        );

        // An interval outside the declared domain is refused for the DOMAIN, before the
        // characterization gate — the two are ordered so the actionable reason wins.
        let message = descriptor
            .capabilities
            .validate_request(
                descriptor.id,
                &probe_request(Some(ApproximationRequest::feature_cache(policy(5, 0)))),
            )
            .expect_err("an undeclared interval must be refused")
            .to_string();
        assert!(
            message.contains("intervals [2, 3, 4]"),
            "the domain must be named: {message}"
        );
    }
}

#[test]
fn a_non_declaring_wan_provider_refuses_a_selection_as_an_absent_mechanism() {
    let mut checked = 0usize;
    for descriptor in descriptors() {
        if descriptor.id == DECLARING_ID {
            continue;
        }
        checked += 1;
        let request = probe_request(Some(ApproximationRequest::feature_cache(policy(2, 0))));
        let error = descriptor
            .capabilities
            .validate_request(descriptor.id, &request)
            .expect_err("a provider with no feature cache must refuse a selection");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "{}: a capability gap, not a range error: {error:?}",
            descriptor.id
        );
        let message = error.to_string();
        assert!(
            message.contains("declares no denoise feature cache mechanism"),
            "{}: the refusal must name the absent mechanism: {message}",
            descriptor.id
        );
        assert!(
            message.contains("unset"),
            "{}: the refusal must name the remedy: {message}",
            descriptor.id
        );
    }
    assert!(
        checked > 0,
        "the registry must contain at least one non-declaring Wan provider for this control"
    );
}

#[test]
fn every_unwired_denoise_route_refuses_a_non_exact_plan_by_name() {
    // The provider-side backstop for the routes the mechanism is NOT wired into. Unreachable today —
    // the shared floor refuses every selection first — so it is asserted directly here, which is the
    // only way it can be gated at all before a characterization artifact exists.
    let plan = ApproximationPlan::FeatureCache(policy(2, 0));
    for route in [
        "TI2V mask-blend",
        "curated unified solver",
        "MoE expert swap",
    ] {
        let error = mlx_gen_wan::refuse_unwired_approximation(DECLARING_ID, route, &plan)
            .expect_err("an unwired route must refuse a non-exact plan");
        assert!(
            matches!(error, mlx_gen::Error::Unsupported(_)),
            "{route}: a capability gap, not a range error: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains(route),
            "the route must be named: {message}"
        );
        assert!(
            message.contains("does not implement the denoise feature cache"),
            "{message}"
        );
    }
    // And the exact plan passes on every one of them, so the check above is about the plan rather
    // than about the route names.
    for route in [
        "TI2V mask-blend",
        "curated unified solver",
        "MoE expert swap",
    ] {
        mlx_gen_wan::refuse_unwired_approximation(DECLARING_ID, route, &ApproximationPlan::Exact)
            .unwrap_or_else(|error| panic!("{route} must admit the exact plan: {error}"));
    }
}
