//! **sc-18322 — the declared Wan approximate capabilities, and the weights-free proof that selection
//! is refused rather than silently ignored.**
//!
//! Epic 18304's P7 adds two *result-changing* mechanisms — a denoise feature cache and token pruning —
//! behind a contract
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
    ModelDescriptor, TokenDropStride, TokenPruningDomain, TokenPruningPolicy,
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

fn pruning(stride: u32, warmup: u32) -> TokenPruningPolicy {
    TokenPruningPolicy::new(TokenDropStride::new(stride).expect("declared stride"))
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
            assert_eq!(
                surface.token_pruning,
                TokenPruningDomain::Implemented {
                    drop_strides: vec![2, 3, 4],
                    max_warmup_steps: 8,
                },
                "the dense 5B declares token pruning over the strides its blocks implement"
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
            token_pruning: None,
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
fn a_declared_token_pruning_selection_is_refused_on_its_own_terms() {
    let descriptors = descriptors();
    let declaring = descriptors
        .iter()
        .find(|descriptor| descriptor.id == DECLARING_ID)
        .expect("the dense 5B must be registered");

    // Every declared corner reaches the characterization gate rather than the domain gate.
    for (stride, warmup) in [(2, 0), (3, 4), (4, 8)] {
        let request = probe_request(Some(ApproximationRequest::token_pruning(pruning(
            stride, warmup,
        ))));
        let error = declaring
            .capabilities
            .validate_request(declaring.id, &request)
            .expect_err("a declared-but-uncharacterized pruning selection must be refused");
        assert!(
            matches!(error, Error::Unsupported(_)),
            "a capability gap, not a range error: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("quality-characterization artifact reference"),
            "{message}"
        );
        assert!(
            !message.contains("declares no token pruning mechanism"),
            "a declared mechanism must not be refused as absent: {message}"
        );
    }

    // An undeclared stride is refused for the domain, with the domain named.
    let message = declaring
        .capabilities
        .validate_request(
            declaring.id,
            &probe_request(Some(ApproximationRequest::token_pruning(pruning(5, 0)))),
        )
        .expect_err("an undeclared stride must be refused")
        .to_string();
    assert!(
        message.contains("drop strides [2, 3, 4]"),
        "the domain must be named: {message}"
    );

    // Both mechanisms at once is still one refusal, and it names the shared precondition.
    let both = probe_request(Some(ApproximationRequest {
        denoise_feature_cache: Some(policy(2, 0)),
        token_pruning: Some(pruning(2, 0)),
        characterization: None,
    }));
    let message = declaring
        .capabilities
        .validate_request(declaring.id, &both)
        .expect_err("composing both mechanisms is still uncharacterized")
        .to_string();
    assert!(
        message.contains("quality-characterization artifact reference"),
        "{message}"
    );
}

#[test]
fn a_non_declaring_wan_provider_refuses_a_selection_as_an_absent_mechanism() {
    let mut checked = 0usize;
    for descriptor in descriptors() {
        if descriptor.id == DECLARING_ID {
            continue;
        }
        checked += 1;
        for (selection, absent) in [
            (
                ApproximationRequest::feature_cache(policy(2, 0)),
                "declares no denoise feature cache mechanism",
            ),
            (
                ApproximationRequest::token_pruning(pruning(2, 0)),
                "declares no token pruning mechanism",
            ),
        ] {
            let request = probe_request(Some(selection));
            let error = descriptor
                .capabilities
                .validate_request(descriptor.id, &request)
                .expect_err("a provider with no such mechanism must refuse a selection");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "{}: a capability gap, not a range error: {error:?}",
                descriptor.id
            );
            let message = error.to_string();
            assert!(
                message.contains(absent),
                "{}: the refusal must name the absent mechanism: {message}",
                descriptor.id
            );
            assert!(
                message.contains("unset"),
                "{}: the refusal must name the remedy: {message}",
                descriptor.id
            );
        }
    }
    assert!(
        checked > 0,
        "the registry must contain at least one non-declaring Wan provider for this control"
    );
}

#[test]
fn the_unwired_route_helper_names_the_route_and_the_selected_mechanisms() {
    // Renamed from a claim it could not support. This exercises the HELPER, not any route's use of it:
    // it takes an arbitrary `route` string, so a green result here proves the message is well formed,
    // never that some route calls it. Which routes call it is
    // `every_a14b_route_refuses_a_non_exact_plan` (below) plus the two `Wan::generate_impl` arms, whose
    // coverage is a source fact rather than a weights-free assertion.
    //
    // What it does pin, and what MINOR 6 was: the refusal names the plan's ACTUALLY SELECTED mechanisms
    // rather than a fixed string, so a pruning-only plan is not refused with the cache's name.
    for (plan, expected) in [
        (
            ApproximationPlan::feature_cache_only(policy(2, 0)),
            "denoise feature cache",
        ),
        (
            ApproximationPlan::token_pruning_only(pruning(2, 0)),
            "token pruning",
        ),
        (
            ApproximationPlan::Approximate {
                denoise_feature_cache: Some(policy(2, 0)),
                token_pruning: Some(pruning(2, 0)),
            },
            "denoise feature cache + token pruning",
        ),
    ] {
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
                message.contains(expected),
                "the refusal must name the selected mechanisms ({expected}): {message}"
            );
        }
    }

    // And the exact plan passes on every route, so the checks above are about the plan rather than
    // about the route names.
    for route in [
        "TI2V mask-blend",
        "curated unified solver",
        "MoE expert swap",
    ] {
        mlx_gen_wan::refuse_unwired_approximation(DECLARING_ID, route, &ApproximationPlan::Exact)
            .unwrap_or_else(|error| panic!("{route} must admit the exact plan: {error}"));
    }
}

#[test]
fn every_a14b_route_refuses_a_non_exact_plan_before_touching_weights() {
    // MAJOR 5: `Wan14b::generate_impl` previously never resolved a plan, so the MoE routes had NO call
    // site for the refusal the helper's docs claimed. It resolves one now, at the top of `generate_impl`
    // and before any weight is opened — which is what makes this assertion possible without weights.
    //
    // Today it necessarily reports the shared floor's refusal (the contract admits no approximate
    // selection at all), so what is proven here is that an approximate selection cannot reach an A14B
    // denoise: the request is refused, by name, on the way in. The provider-side `MoE expert swap`
    // refusal sits immediately behind it for the day selection becomes possible.
    let a14b: Vec<ModelDescriptor> = descriptors()
        .into_iter()
        .filter(|descriptor| descriptor.id != DECLARING_ID)
        .collect();
    assert!(
        !a14b.is_empty(),
        "the registry must expose the non-dense Wan providers"
    );
    for descriptor in a14b {
        for selection in [
            ApproximationRequest::feature_cache(policy(2, 0)),
            ApproximationRequest::token_pruning(pruning(2, 0)),
        ] {
            let error = descriptor
                .capabilities
                .validate_request(descriptor.id, &probe_request(Some(selection)))
                .expect_err("an expert-swap provider must refuse an approximate selection");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "{}: {error:?}",
                descriptor.id
            );
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Two properties below are asserted against the crate's own SOURCE, which needs justifying.
//
// Both are compile-time/structural facts that no runtime assertion can reach. "There is no production
// constructor" is a statement about what a `not(test)` build contains — and a test, by construction,
// is a `test` build, so it can never observe the thing it needs to check. "This route calls the
// refusal" is a statement about a code path that requires real weights to execute, and the shared
// floor refuses the request before the provider is ever entered, so even a weights-bearing test would
// exercise the floor rather than the provider guard.
//
// The mutation pass is what forced the issue: both properties survived every behavioural mutation,
// which is the honest signal that they were claimed but not gated. A source assertion is narrow and
// slightly unusual, but it pins exactly the regression that matters and costs nothing to run.
// -------------------------------------------------------------------------------------------------

fn crate_source(relative: &str) -> String {
    let path = format!("{}/{relative}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn neither_mechanism_has_a_production_constructor() {
    // MAJOR 4: `TokenPruner` shipped with a `pub` keep-set constructor and a re-export, so a production
    // build could assemble the mechanism by hand and run it with no plan at all — bypassing the contract
    // entirely. Only `TrunkCache` had the second lock.
    for (file, ty) in [
        ("src/token_pruning.rs", "TokenPruner"),
        ("src/feature_cache.rs", "TrunkCache"),
    ] {
        let source = crate_source(file);
        let marker = "fn from_plan_for_test";
        let at = source
            .find(marker)
            .unwrap_or_else(|| panic!("{file}: {ty} must have a from_plan_for_test constructor"));
        let preceding = &source[..at];
        assert!(
            preceding.trim_end().ends_with("pub(crate)")
                || preceding.trim_end().ends_with("pub(crate) "),
            "{file}: {ty}'s plan constructor must be pub(crate)"
        );
        // The `#[cfg(test)]` must be on the constructor itself, i.e. within the few lines above it.
        let window_start = preceding.len().saturating_sub(160);
        assert!(
            preceding[window_start..].contains("#[cfg(test)]"),
            "{file}: {ty}'s plan constructor must be #[cfg(test)] — without it a production build can \
             reach the uncharacterized mechanism"
        );
    }

    // And the keep-set must not be re-exported: it is the piece a caller would need to assemble a
    // pruned forward without going through `TokenPruner`.
    let lib = crate_source("src/lib.rs");
    assert!(
        !lib.contains("pub use token_pruning::TokenKeepSet"),
        "TokenKeepSet must not be re-exported — TokenPruner is the only reachable pruning surface"
    );
    assert!(
        lib.contains("pub use token_pruning::TokenPruner"),
        "TokenPruner is the surface the pub entry points take, so it must be exported"
    );
}

#[test]
fn every_unwired_denoise_route_has_a_refusal_call_site() {
    // MINOR 5: the helper's doc asserted a guard on three routes, but the MoE routes had no call site at
    // all — `Wan14b::generate_impl` never resolved a plan. The helper takes the route name as a string,
    // so no test of the helper can distinguish "this route refuses" from "this string round-trips".
    let model = crate_source("src/model.rs");
    for route in [
        "TI2V mask-blend",
        "curated unified solver",
        "MoE expert swap",
    ] {
        assert!(
            model.contains(&format!("\"{route}\"")),
            "the {route} route must pass its own name to refuse_unwired_approximation"
        );
    }
    assert_eq!(
        model.matches("refuse_unwired_approximation(").count(),
        3,
        "exactly the three unwired routes call the refusal — a fourth call site, or a missing one, \
         means the guard's documented coverage has drifted from the code"
    );
    // The A14B refusal must sit before any weight is opened, which is what makes it a guard rather than
    // a late error. `validate` is the first statement of `generate_impl`; the refusal is the second.
    let a14b = model
        .rfind("refuse_unwired_approximation(")
        .expect("the MoE call site");
    let validate_before = model[..a14b]
        .rfind("self.validate(req)?;")
        .expect("validate precedes the refusal");
    assert!(
        model[validate_before..a14b].lines().count() < 12,
        "the MoE refusal must follow validate immediately, before any loading work"
    );
}
