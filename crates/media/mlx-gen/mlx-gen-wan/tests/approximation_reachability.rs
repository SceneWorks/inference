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

/// A crate source file with **every** comment removed — `//` to end of line *and* `/* … */` spans,
/// nested ones included — with string, char and raw-string literals left intact.
///
/// Both halves are load-bearing, and each closed a hole this gate shipped with.
///
/// The line-comment stripping exists because the first revision scanned the 160 characters before
/// `fn from_plan_for_test` for the text `#[cfg(test)]`, and `TokenPruner`'s own doc comment says
/// "`#[cfg(test)]`, mirroring …" 127 characters back: deleting the real attribute left the doc match
/// inside the window, so the gate passed and the mutation it exists to kill survived — on the exact
/// type it was written for.
///
/// The block-comment stripping closes the same class through the other syntax: commenting a guard out
/// with `/* … */` while preserving its text would leave every occurrence-count assertion satisfied. No
/// `/* … */` appears in the four scanned files today, so this is prevention rather than repair — but
/// "prose about code cannot satisfy an assertion about code" has to hold for both comment forms or it
/// does not hold at all.
fn crate_code(relative: &str) -> String {
    strip_comments(&crate_source(relative))
}

/// Remove Rust comments from `source`, preserving literals.
///
/// Literal-aware on purpose: a `//` inside a string is not a comment, and blindly stripping it would
/// corrupt the very `"MoE expert swap"`-style route names another gate matches on.
fn strip_comments(source: &str) -> String {
    let bytes: Vec<char> = source.chars().collect();
    let mut out = String::with_capacity(source.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        let next = bytes.get(i + 1).copied();

        // Line comment: drop through to (but keep) the newline, so line structure survives.
        if c == '/' && next == Some('/') {
            while i < bytes.len() && bytes[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // Block comment, honouring Rust's nesting.
        if c == '/' && next == Some('*') {
            let mut depth = 1usize;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == '/' && bytes.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == '*' && bytes.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                } else {
                    // Keep newlines so a commented-out block does not join its neighbours.
                    if bytes[i] == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
            continue;
        }
        // Raw string: `r`, any number of `#`, then `"` — no escapes, closed by `"` + the same `#`s.
        // Only when the `r` does not continue an identifier (otherwise it is just a letter).
        if c == 'r' && !identifier_char_before(&bytes, i) {
            let mut hashes = 0usize;
            let mut j = i + 1;
            while bytes.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if bytes.get(j) == Some(&'"') {
                let closing: String = std::iter::once('"')
                    .chain(std::iter::repeat_n('#', hashes))
                    .collect();
                out.extend(&bytes[i..=j]);
                i = j + 1;
                let tail: String = bytes[i..].iter().collect();
                let end = tail.find(&closing).map_or(bytes.len(), |at| {
                    i + tail[..at + closing.len()].chars().count()
                });
                out.extend(&bytes[i..end.min(bytes.len())]);
                i = end;
                continue;
            }
        }
        // Ordinary string or char literal: copy verbatim, respecting escapes.
        if c == '"' || c == '\'' {
            out.push(c);
            i += 1;
            while i < bytes.len() {
                if bytes[i] == '\\' {
                    out.push(bytes[i]);
                    if let Some(escaped) = bytes.get(i + 1) {
                        out.push(*escaped);
                    }
                    i += 2;
                    continue;
                }
                out.push(bytes[i]);
                let closed = bytes[i] == c;
                i += 1;
                if closed {
                    break;
                }
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn identifier_char_before(chars: &[char], at: usize) -> bool {
    at > 0 && (chars[at - 1].is_alphanumeric() || chars[at - 1] == '_')
}

/// `code` with every run of whitespace collapsed to a single space.
///
/// What the adjacency assertions match against, so they pin **token** order rather than layout. A blank
/// line between `#[cfg(test)]` and its signature is legal and rustfmt-preserved
/// (`blank_lines_upper_bound = 1`); matching raw text would have reddened on it, and hard-coding
/// `\n    ` would additionally have reddened whenever an item's nesting depth changed — both spurious
/// reds costing a full CI cycle to discover.
fn squeezed(code: &str) -> String {
    code.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every region introduced by a column-0 `header`, each ending at its column-0 closing brace.
///
/// Returns *all* of them rather than the last: `model.rs` has **two** `impl Wan14b {` blocks (split by a
/// top-level fn between them), so an `rfind` was correct only by the accident that `generate_impl` sits
/// in the later one. Reordering the impls would have silently retargeted the gate at unrelated code.
fn top_level_blocks<'a>(code: &'a str, header: &str) -> Vec<&'a str> {
    code.match_indices(header)
        .map(|(start, _)| {
            let rest = &code[start + header.len()..];
            let end = rest.find("\n}").map_or(rest.len(), |at| at + 2);
            &rest[..end]
        })
        .collect()
}

#[test]
fn neither_mechanism_has_a_production_constructor() {
    // MAJOR 4: `TokenPruner` shipped with a `pub` keep-set constructor and a re-export, so a production
    // build could assemble the mechanism by hand and run it with no plan at all — bypassing the contract
    // entirely. Only `TrunkCache` had the second lock.
    //
    // Asserted as an adjacent TOKEN sequence over comment-stripped, whitespace-normalised code, so
    // neither prose about the attribute nor a change of layout can decide the outcome. Token adjacency
    // is the smallest thing a real regression must break: relaxing the visibility, dropping the `cfg`,
    // or interposing another item all separate these tokens.
    for (file, ty) in [
        ("src/token_pruning.rs", "TokenPruner"),
        ("src/feature_cache.rs", "TrunkCache"),
    ] {
        let code = crate_code(file);
        assert!(
            code.contains("fn from_plan_for_test"),
            "{file}: {ty} must have a from_plan_for_test constructor"
        );
        assert!(
            squeezed(&code).contains("#[cfg(test)] pub(crate) fn from_plan_for_test"),
            "{file}: {ty}'s plan constructor must be exactly `#[cfg(test)] pub(crate)`, with nothing \
             between the two — without that a production build can reach the uncharacterized mechanism"
        );
        // And there is only the one, so a second, unguarded constructor cannot hide beside it.
        assert_eq!(
            code.matches("fn from_plan_for_test").count(),
            1,
            "{file}: {ty} must have exactly one plan constructor"
        );
    }

    // The keep-set must not be re-exported: it is the piece a caller would need to assemble a pruned
    // forward without going through `TokenPruner`.
    let lib = crate_code("src/lib.rs");
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
    let model = crate_code("src/model.rs");
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

    // The A14B refusal must run BEFORE any loading work, which is what makes it a guard rather than a
    // late error. Scoped to whichever `impl Wan14b` block actually holds `generate_impl` — there are two
    // such blocks, so "the last one" was correct only by accident — and asserted as an ORDERING plus a
    // named-operation check, not as a line distance.
    let wan14b: Vec<&str> = top_level_blocks(&model, "impl Wan14b {")
        .into_iter()
        .filter(|block| block.contains("fn generate_impl("))
        .collect();
    assert_eq!(
        wan14b.len(),
        1,
        "exactly one `impl Wan14b` block must define `generate_impl` — {} do, so this gate would be \
         scoped to the wrong code",
        wan14b.len()
    );
    let body = &wan14b[0][wan14b[0]
        .find("fn generate_impl(")
        .expect("filtered on this")..];
    let validate = body
        .find("self.validate(req)?;")
        .expect("Wan14b::generate_impl must validate");
    let refusal = body
        .find("refuse_unwired_approximation(")
        .expect("Wan14b::generate_impl must refuse an unwired approximate plan");
    assert!(
        validate < refusal,
        "the MoE refusal must follow the shared floor, not precede it"
    );
    // Nothing that loads or embeds may run between them. Naming the operations is more precise than a
    // line budget: a comment is free, an expert load is not.
    let between = &body[validate..refusal];
    for forbidden in ["load_", "embed_text", "denoise", "Weights::"] {
        assert!(
            !between.contains(forbidden),
            "`{forbidden}` runs between the shared floor and the MoE refusal — the refusal must come \
             before any loading work"
        );
    }
}

#[test]
fn every_mechanism_bridge_refuses_a_policy_inert_over_the_trajectory() {
    // Cycle-2 MINOR: `refuse_inert_over_trajectory` has a direct test with discriminating controls, but
    // nothing pinned that the bridges still CALL it — the same compile-time-fact class as the two gates
    // above, and the cheapest to cover, since deleting a call cannot fail any behavioural test.
    //
    // Four bridges: `trunk_cache` and `token_pruner`, each in a `cfg(test)` and a `cfg(not(test))`
    // definition. The property is **guard before construction**, asserted per bridge body — not literal
    // adjacency to the opening brace, which would have reddened on a benign `let _ = steps;` or a
    // `debug_assert!` above the guard while the guard still did its job.
    let model = crate_code("src/model.rs");
    assert!(
        model.contains("fn refuse_inert_over_trajectory("),
        "the guard the bridges call must be defined here"
    );

    let mut bridges = 0usize;
    for signature in ["fn trunk_cache(", "fn token_pruner("] {
        let bodies = top_level_blocks(&model, signature);
        assert_eq!(
            bodies.len(),
            2,
            "{signature} must have exactly its cfg(test) and cfg(not(test)) definitions"
        );
        for body in bodies {
            bridges += 1;
            let guard = body
                .find("refuse_inert_over_trajectory(")
                .unwrap_or_else(|| {
                    panic!(
                        "{signature}: every bridge must refuse a policy inert over the trajectory"
                    )
                });
            // Where the bridge constructs a mechanism, the guard must already have run: a guard after
            // construction has let the uncharacterized path be built before deciding it was refusable.
            if let Some(construction) = body.find("from_plan_for_test") {
                assert!(
                    guard < construction,
                    "{signature}: the inert-policy guard must run BEFORE the mechanism is constructed"
                );
            }
        }
    }
    assert_eq!(bridges, 4, "all four mechanism bridges must be covered");
}
