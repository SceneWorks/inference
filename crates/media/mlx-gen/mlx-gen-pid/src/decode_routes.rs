//! The per-route bounded-decode domain a PiD-eligible provider declares, and its enforcement
//! (SC-15775).
//!
//! [`selected_decode_tiling`](crate::engine::selected_decode_tiling) is a **shared,
//! provider-agnostic** seam: every PiD-eligible provider reaches it through
//! [`resolve_pid_decoder_at_sigma`](crate::engine::resolve_pid_decoder_at_sigma). SC-15510 taught it
//! to honour an explicit
//! [`GenerationMemory::decode_tile_edge`](mlx_gen::gen_core::GenerationMemory) so a shared-contract
//! bounded-decode ("rung 2") selection *executes* on the PiD route instead of being refused.
//!
//! That created a cross-provider hazard. Both routes measure "decode tile edge in output pixels", but
//! they are **disjoint domains**:
//!
//! | route | decoder | legal tile edges |
//! |---|---|---|
//! | native | the family's own VAE | its measured ladder, 256-768 output px in practice |
//! | `use_pid` | the super-resolving PiD student | [`budget::tile_edge_candidates`] — [`TILE_ALIGN`](budget::TILE_ALIGN) multiples from [`MIN_TILE_EDGE`](budget::MIN_TILE_EDGE) (1024) up to [`WATCHDOG_SAFE_EDGE`](budget::WATCHDOG_SAFE_EDGE) |
//!
//! The student decodes a `scale×` super-resolved output, so its tiles are necessarily larger than the
//! whole native output. The moment a second provider's memory-strategy adoption emits **its native VAE
//! ladder** into `decode_tile_edge`, a `use_pid` + rung-2 request on that provider flips from
//! "auto-plan, works" to a hard [`budget::validate_tile`] rejection at generate time. Qwen-Image is the
//! obvious next adopter (its probe ladder is 768/640/512/448/384/320/256) and it is PiD-eligible.
//!
//! The seam deliberately does **not** paper over it: silently falling back to the auto-plan when the
//! selected edge is not a legal PiD tile would execute a different strategy than the selector chose,
//! which is exactly what the shared contract forbids (SC-15449) and what made SC-15615 refuse rungs 2+
//! on PiD in the first place. So an out-of-domain edge has to be loud.
//!
//! ## What this module does about it
//!
//! It moves the obligation from a doc comment ("copy Z-Image's shape") into an API that makes the
//! mistake hard to write and impossible to ship quietly, in four layers:
//!
//! 1. **[`DecodeRoutes::new`] is a *checked* constructor, and the only one.** It takes the *native*
//!    ladder only — the PiD ladder is an associated function, not a constructor argument, so a
//!    provider **cannot supply a PiD-route domain at all**. On top of that it returns [`Result`]: a
//!    native ladder that reaches into the PiD student's tile domain, or is otherwise degenerate,
//!    **does not construct**. So a `DecodeRoutes` value in hand is proof its two routes are disjoint,
//!    and that holds in release builds and for an adopter with no tests of its own.
//!
//!    Stated precisely, because the looser claim was made once and was wrong: an illegal
//!    *declaration* — a `DecodeRoutes` whose routes overlap — is unrepresentable. The *inputs* to an
//!    illegal declaration are of course still writable; what is impossible is getting a value back
//!    from them.
//! 2. **[`assert_decode_routes`] is the weights-free test-suite form** a provider runs beside
//!    `gen_core_testkit::memory_strategy_conformance`. Same predicate as the constructor, so it adds
//!    no second definition of "conforming"; what it adds is *when* the failure lands — a provider that
//!    declares its ladder in a `const` gets a named CI failure instead of a load-time error.
//! 3. **The seam itself asserts** (see [`selected_decode_tiling`](crate::engine::selected_decode_tiling)).
//!    A provider that bypasses this module entirely still trips a `debug_assert` the moment any test
//!    drives a native geometry into a `use_pid` request, so the mis-wiring lands in CI rather than in
//!    a user's generate call. Release behaviour is unchanged: [`budget::validate_tile`] still rejects,
//!    typed, and still never re-plans.
//! 4. **The workspace gate refuses a provider that never reaches for any of the above.** Layers 1-3
//!    help a provider that *calls* them, and layer 3 only fires if some test drives the request — so
//!    `scripts/check-workspace.py`'s `check_pid_decode_route_adoption` fails any workspace member that
//!    depends on this crate, registers a memory-strategy contract, and shows any sign of implementing
//!    rung 2, unless its sources actually **call** the checked constructor and **call**
//!    [`DecodeRoutes::validate`]. It matches on comment-stripped source, so a `// TODO: use
//!    DecodeRoutes` or an unused `use` satisfies nothing. That is what makes the obligation
//!    **automatic** for the next adopter instead of opt-in: it is keyed on facts about the crate, not
//!    on a list of crate names anyone has to remember to extend.
//!
//! [`DecodeRoutes::validate`] is the admission gate itself — the route-aware check a provider's
//! `safety_check` and request scope call, replacing a hand-rolled per-provider `match`.

use crate::budget;

/// How a rejection names the route a selection was built for.
///
/// One spelling, because a provider's admission rejections and this module's conformance errors are
/// read together in the same failure. Private: no caller outside this module has ever needed to spell
/// a route on its own.
fn route_label(use_pid: bool) -> &'static str {
    if use_pid {
        "PiD overlay"
    } else {
        "native VAE"
    }
}

/// Everything wrong with a native declaration, as messages — empty means it conforms.
///
/// The property that matters is **disjointness**: no native candidate may also be a legal PiD tile
/// edge. That is the precondition for the whole route-aware design — if the domains overlapped, an
/// edge on the wire would be ambiguous, admission could not tell which route it was built for, and
/// the seam would happily hand a native geometry to a `scale×` super-resolved decode.
///
/// It runs on the *inputs*, before a [`DecodeRoutes`] exists, because it is [`DecodeRoutes::new`]'s
/// own admission check rather than a separate opt-in pass a caller could forget.
fn conformance_errors(provider_id: &str, native_edges: &[u32], native_overlap: u32) -> Vec<String> {
    let mut errors = Vec::new();
    if native_edges.is_empty() {
        errors.push(format!(
            "{provider_id}: declares no native decode tile-edge candidates, so the native route has \
             no domain to be disjoint from the PiD one"
        ));
    }
    for &edge in native_edges {
        if edge == 0 {
            errors.push(format!(
                "{provider_id}: native decode tile edge 0 is degenerate"
            ));
            continue;
        }
        if budget::is_tile_edge_candidate(edge as i32) {
            errors.push(format!(
                "{provider_id}: native decode tile edge {edge} is ALSO a legal PiD tile edge \
                 ({:?}). The two routes' domains must be disjoint, or the shared \
                 `selected_decode_tiling` seam cannot tell a native geometry from a PiD one and a \
                 `use_pid` + bounded-decode request decodes at the wrong scale or is refused at \
                 generate time (SC-15775).",
                budget::tile_edge_candidates()
            ));
        }
        if native_overlap >= edge {
            errors.push(format!(
                "{provider_id}: native decode overlap {native_overlap} is not smaller than \
                 candidate edge {edge}, which makes the tile split degenerate"
            ));
        }
    }
    if native_overlap == 0 {
        errors.push(format!(
            "{provider_id}: native decode overlap must be positive (a zero-overlap split seams)"
        ));
    }
    // Self-check on the PiD side, so a change to `budget`'s constants that made the published PiD
    // ladder un-honourable fails here rather than at some provider's generate call. This cannot be
    // wrong "per provider" — that is the point of the domain living in this crate — but it can be
    // wrong for everyone at once, which is worse.
    let pid_edges = DecodeRoutes::pid_edges();
    if pid_edges.is_empty() {
        errors.push(format!(
            "{provider_id}: the PiD route publishes no candidates, so a bounded-decode selection on \
             it can never validate"
        ));
    }
    for edge in pid_edges {
        if !budget::is_tile_edge_candidate(edge as i32) {
            errors.push(format!(
                "{provider_id}: published PiD tile edge {edge} is not one of \
                 `budget::tile_edge_candidates` {:?}, so `validate_tile` would refuse the provider's \
                 own advertised candidate",
                budget::tile_edge_candidates()
            ));
        }
        if DecodeRoutes::pid_overlap() == 0 || DecodeRoutes::pid_overlap() >= edge {
            errors.push(format!(
                "{provider_id}: the PiD overlap {} must be in 1..{edge}",
                DecodeRoutes::pid_overlap()
            ));
        }
    }
    errors
}

/// The two bounded-decode routes a PiD-eligible provider runs, and the single place their candidate
/// domains are reconciled (SC-15775).
///
/// Construct it from the family's **native** VAE ladder only, and handle the rejection — an
/// overlapping or degenerate declaration does not produce a value:
///
/// ```no_run
/// use mlx_gen_pid::DecodeRoutes;
///
/// // The provider's own measured native ladder, at its own overlap.
/// let routes = DecodeRoutes::new("my_provider", [768, 640, 512], 64)?;
///
/// // The static contract publishes the union, because `validate_selection` is not route-aware.
/// assert_eq!(routes.published_edges(), vec![2048, 768, 640, 512]);
///
/// // Admission, which IS route-aware, enforces the route's own subset.
/// routes.validate(true, Some(512), Some(64)).unwrap_err();
///
/// // A native ladder reaching into the PiD student's domain does not construct at all.
/// assert!(DecodeRoutes::new("my_provider", [2048, 1024], 256).is_err());
/// # Ok::<(), Vec<String>>(())
/// ```
///
/// There is deliberately **no** way to pass in a PiD-route ladder: [`DecodeRoutes::pid_edges`] is an
/// associated function reading [`budget::tile_edge_candidates`]' own preferred point. A constructor
/// that accepted one would let a provider pass its native ladder for both routes, which is the exact
/// defect this type exists to prevent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeRoutes {
    provider_id: String,
    native_edges: Vec<u32>,
    native_overlap: u32,
}

impl DecodeRoutes {
    /// Declare the native route's ladder, or fail.
    ///
    /// Edges are sorted descending and deduped first, because the shared contract's static validator
    /// rejects duplicate candidates and callers should not have to know that. The normalized
    /// declaration is then checked, and a non-conforming one is **refused rather than repaired**:
    /// silently dropping a native candidate that collides with the PiD domain would publish a
    /// narrower ladder than the provider measured, which is the same class of "executed something
    /// other than what was declared" failure the shared contract exists to prevent.
    ///
    /// `provider_id` is stored, so every message this type produces — construction and admission
    /// alike — names the same provider. That is why [`Self::validate`] does not take one.
    ///
    /// The `Err` is the full list of defects rather than the first: an adopter fixing a hand-copied
    /// ladder wants to see all of them.
    pub fn new(
        provider_id: &str,
        native_edges: impl IntoIterator<Item = u32>,
        native_overlap: u32,
    ) -> Result<Self, Vec<String>> {
        let mut native_edges: Vec<u32> = native_edges.into_iter().collect();
        native_edges.sort_unstable_by(|a, b| b.cmp(a));
        native_edges.dedup();
        let errors = conformance_errors(provider_id, &native_edges, native_overlap);
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Self {
            provider_id: provider_id.to_owned(),
            native_edges,
            native_overlap,
        })
    }

    /// The native VAE route's declared tile edges, descending.
    pub fn native_edges(&self) -> &[u32] {
        &self.native_edges
    }

    /// The **PiD overlay route's** tile-edge ladder — not a provider input.
    ///
    /// One candidate, and that is a measurement rather than a placeholder. [`budget::validate_tile`]
    /// will *accept* seven aligned edges between [`MIN_TILE_EDGE`](budget::MIN_TILE_EDGE) and
    /// [`WATCHDOG_SAFE_EDGE`](budget::WATCHDOG_SAFE_EDGE), but only
    /// [`PREFERRED_TILE_EDGE`](budget::PREFERRED_TILE_EDGE) has seam evidence: sc-10087's A/B measured
    /// 2048 px tiles at 256 px overlap as showing no measurable seam on this student, which is why the
    /// planner itself prefers it. Publishing the unmeasured other six beside a native ladder that
    /// earned its entries by a real-weight sweep would be exactly the double standard this epic keeps
    /// rejecting.
    ///
    /// Widening it needs a per-student sweep against the untiled super-resolved decode. The student is
    /// tied to a *latent space* rather than to a model (see [`registry`](crate::registry)), so such a
    /// sweep would widen this for every provider sharing that student at once — which is the reason
    /// the domain lives here and not in each provider.
    pub fn pid_edges() -> Vec<u32> {
        vec![budget::PREFERRED_TILE_EDGE as u32]
    }

    /// The PiD overlay route's feather overlap — the sc-10087 A/B's 256 px, i.e. the planner's own
    /// [`DEFAULT_TILE_OVERLAP`](budget::DEFAULT_TILE_OVERLAP), which is also what
    /// [`selected_decode_tiling`](crate::engine::selected_decode_tiling) falls back to when a
    /// selection names an edge but no overlap.
    pub fn pid_overlap() -> u32 {
        budget::DEFAULT_TILE_OVERLAP as u32
    }

    /// The `(edges, overlap)` domain of the route a request with this `use_pid` actually runs.
    ///
    /// This is the value admission enforces. It is a *subset* of what the static contract publishes —
    /// narrower at admission, never broader, which is the direction the shared contract allows.
    pub fn domain(&self, use_pid: bool) -> (Vec<u32>, u32) {
        if use_pid {
            (Self::pid_edges(), Self::pid_overlap())
        } else {
            (self.native_edges.clone(), self.native_overlap)
        }
    }

    /// The union of both routes' edges, descending — what the provider's static
    /// `MemoryParameterRanges::decode_tile_edges` must publish.
    ///
    /// The union, not one route: the static validator is route-blind (it never sees `use_pid`), so
    /// publishing only one route's ladder would make every legitimate selection on the *other* route
    /// fail the static check before admission could even see it.
    pub fn published_edges(&self) -> Vec<u32> {
        let mut edges: Vec<u32> = self
            .native_edges
            .iter()
            .copied()
            .chain(Self::pid_edges())
            .collect();
        edges.sort_unstable_by(|a, b| b.cmp(a));
        edges.dedup();
        edges
    }

    /// The union of both routes' overlaps, descending. See [`Self::published_edges`].
    pub fn published_overlaps(&self) -> Vec<u32> {
        let mut overlaps = vec![Self::pid_overlap(), self.native_overlap];
        overlaps.sort_unstable_by(|a, b| b.cmp(a));
        overlaps.dedup();
        overlaps
    }

    /// The **route-aware admission gate**: accept or reject a bounded-decode selection's
    /// `(edge, overlap)` for the route the request will run.
    ///
    /// Call this from the provider's `safety_check` and from its request scope's `configure_decode`,
    /// on any selection that engages
    /// [`MemoryStrategy::BoundedDecode`](mlx_gen::gen_core::MemoryStrategy). It rejects — it never
    /// substitutes a different geometry, and it never falls back to the planner's auto-plan, because
    /// executing a different strategy than the selector chose is the failure the shared contract
    /// exists to prevent.
    ///
    /// `None` is a rejection, not "use the default": at this point the selection has already engaged
    /// the rung, so a missing geometry means the selector named a strategy it did not parameterize.
    pub fn validate(
        &self,
        use_pid: bool,
        edge: Option<u32>,
        overlap: Option<u32>,
    ) -> Result<(), String> {
        let (edges, expected_overlap) = self.domain(use_pid);
        let label = route_label(use_pid);
        let provider_id = &self.provider_id;
        // `None` reads as "named none" rather than `None`, because this reason reaches the caller as a
        // `CoreError::Unsupported` and "decode tile edge None" invites the reader to think the field
        // was the problem rather than its absence.
        match edge {
            Some(edge) if edges.contains(&edge) => {}
            Some(edge) => {
                return Err(format!(
                    "{provider_id}: decode tile edge {edge} is not a {label} route candidate \
                     {edges:?}"
                ));
            }
            None => {
                return Err(format!(
                    "{provider_id}: a bounded decode on the {label} route named no tile edge; the \
                     candidates are {edges:?}"
                ));
            }
        }
        match overlap {
            Some(overlap) if overlap == expected_overlap => {}
            Some(overlap) => {
                return Err(format!(
                    "{provider_id}: the {label} route's decode overlap is {expected_overlap}, got \
                     {overlap}"
                ));
            }
            None => {
                return Err(format!(
                    "{provider_id}: a bounded decode on the {label} route named no overlap; it must \
                     be {expected_overlap}"
                ));
            }
        }
        Ok(())
    }
}

/// Panic-on-failure [`DecodeRoutes::new`], mirroring `gen_core_testkit`'s
/// `memory_strategy_conformance` so a provider's suite reads the same either way. Returns the
/// declaration, so the test can go on using it.
///
/// This is the **test-suite** spelling of the *same* check the constructor already enforces — it adds
/// no second definition of "conforming". What it adds is *when* the failure lands: a provider whose
/// ladder is a `const` (every adopter so far) gets a named failure from its own suite in CI instead of
/// a load-time error from `?` in production, which is the story's "caught by a test rather than by a
/// production rejection".
pub fn assert_decode_routes(
    provider_id: &str,
    native_edges: impl IntoIterator<Item = u32>,
    native_overlap: u32,
) -> DecodeRoutes {
    match DecodeRoutes::new(provider_id, native_edges, native_overlap) {
        Ok(routes) => routes,
        Err(errors) => panic!(
            "PiD decode-route conformance FAILED for '{provider_id}':\n- {}",
            errors.join("\n- ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Z-Image's shipping declaration (SC-15510), which is the shape an adopter copies.
    fn routes() -> DecodeRoutes {
        DecodeRoutes::new("z_image_turbo", [768, 640, 512], 64).unwrap()
    }

    #[test]
    fn the_shipping_declaration_conforms_and_the_domains_are_disjoint() {
        let routes = routes();
        let (native, native_overlap) = routes.domain(false);
        let (pid, pid_overlap) = routes.domain(true);
        assert_eq!(native, vec![768, 640, 512]);
        assert_eq!(native_overlap, 64);
        assert_eq!(pid, vec![budget::PREFERRED_TILE_EDGE as u32]);
        assert_eq!(pid_overlap, budget::DEFAULT_TILE_OVERLAP as u32);
        assert!(
            native.iter().all(|edge| !pid.contains(edge)),
            "the routes' domains must be disjoint"
        );
        // Disjointness is not a coincidence of these numbers: no native edge can be a legal PiD tile,
        // because the student's floor is above the whole native output.
        assert!(native
            .iter()
            .all(|&edge| !budget::is_tile_edge_candidate(edge as i32)));
    }

    /// THE regression this module exists for: the next adopter declares its native VAE ladder into
    /// the PiD-legal range, and **construction** says so instead of a user's generate call saying so.
    #[test]
    fn a_native_ladder_that_reaches_into_the_pid_domain_does_not_construct() {
        // A hypothetical family whose native VAE tiles at PiD-legal edges.
        let errors = DecodeRoutes::new("hypothetical", [2048, 1024], 64).unwrap_err();
        assert_eq!(errors.len(), 2, "{errors:?}");
        for error in &errors {
            assert!(error.contains("is ALSO a legal PiD tile edge"), "{error}");
            assert!(error.contains("SC-15775"), "{error}");
        }
        // One legal edge among otherwise-native ones is still caught.
        let errors = DecodeRoutes::new("hypothetical", [768, 1536], 64).unwrap_err();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("1536"), "{errors:?}");
    }

    /// The adversarial review of the first revision built exactly this from an external crate and it
    /// **constructed**, then admitted a PiD-domain geometry on the native route. Pinned verbatim so
    /// the hole cannot reopen: the point of a checked constructor is that no `DecodeRoutes` value with
    /// overlapping routes exists to be asked in the first place.
    #[test]
    fn the_reviewers_construction_attack_no_longer_yields_a_value() {
        let attack = DecodeRoutes::new("attacker", [2048u32, 1024], 256);
        assert!(
            attack.is_err(),
            "an overlapping native ladder must not construct: {:?}",
            attack.as_ref().map(|routes| routes.native_edges().to_vec())
        );
        let errors = attack.unwrap_err();
        assert!(
            errors.iter().all(|error| error.contains("attacker")),
            "every message names the provider: {errors:?}"
        );
        // The review's follow-on assertions are now unreachable by construction: there is no value on
        // which to call `native_edges`, `domain` or `validate`.
        assert!(DecodeRoutes::new("attacker", [2048u32], 256).is_err());
        assert!(DecodeRoutes::new("attacker", [1024u32], 256).is_err());
    }

    #[test]
    fn a_degenerate_native_declaration_does_not_construct() {
        let errors = DecodeRoutes::new("empty", [], 64).unwrap_err();
        assert!(
            errors[0].contains("no native decode tile-edge candidates"),
            "{errors:?}"
        );

        let errors = DecodeRoutes::new("zero_overlap", [512], 0).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("overlap must be positive")),
            "{errors:?}"
        );

        let errors = DecodeRoutes::new("fat_overlap", [512], 512).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("makes the tile split degenerate")),
            "{errors:?}"
        );

        let errors = DecodeRoutes::new("zero_edge", [0], 64).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("is degenerate")),
            "{errors:?}"
        );
    }

    /// The admission gate is symmetric: each route refuses the other's geometry rather than
    /// re-planning it, and refuses a rung engaged with no geometry at all.
    #[test]
    fn admission_refuses_the_other_routes_geometry_on_both_sides() {
        let routes = routes();
        let pid_edge = DecodeRoutes::pid_edges()[0];
        let pid_overlap = DecodeRoutes::pid_overlap();

        routes.validate(false, Some(512), Some(64)).unwrap();
        routes
            .validate(true, Some(pid_edge), Some(pid_overlap))
            .unwrap();

        // A native edge under `use_pid` — the SC-15775 hazard, at admission.
        let error = routes.validate(true, Some(512), Some(64)).unwrap_err();
        assert!(error.contains("PiD overlay"), "{error}");
        assert!(error.contains("512"), "{error}");
        // The provider the declaration was built for names itself, without the caller repeating it.
        assert!(error.contains("z_image_turbo"), "{error}");

        // A PiD edge without the overlay.
        let error = routes
            .validate(false, Some(pid_edge), Some(pid_overlap))
            .unwrap_err();
        assert!(error.contains("native VAE"), "{error}");

        // Right edge, wrong route's overlap.
        let error = routes.validate(true, Some(pid_edge), Some(64)).unwrap_err();
        assert!(error.contains("overlap"), "{error}");

        // The rung is engaged but nothing was named: a rejection, never a fabricated default.
        assert!(routes.validate(true, None, None).is_err());
        assert!(routes.validate(false, None, Some(64)).is_err());
        assert!(routes.validate(false, Some(512), None).is_err());
    }

    #[test]
    fn the_published_union_covers_both_routes_and_is_sorted_and_deduped() {
        let routes = routes();
        let published = routes.published_edges();
        assert_eq!(published, vec![2048, 768, 640, 512]);
        for &edge in routes.native_edges() {
            assert!(published.contains(&edge));
        }
        for edge in DecodeRoutes::pid_edges() {
            assert!(published.contains(&edge));
        }
        assert_eq!(routes.published_overlaps(), vec![256, 64]);
        // A provider whose native overlap happens to equal the PiD one publishes it once.
        assert_eq!(
            DecodeRoutes::new("p", [768], DecodeRoutes::pid_overlap())
                .unwrap()
                .published_overlaps(),
            vec![DecodeRoutes::pid_overlap()]
        );
        // Duplicate/unsorted native input normalizes rather than reaching the contract's validator.
        assert_eq!(
            DecodeRoutes::new("p", [512, 768, 512], 64)
                .unwrap()
                .native_edges(),
            &[768, 512]
        );
    }

    #[test]
    fn assert_decode_routes_names_the_provider_and_the_defect() {
        let panic = std::panic::catch_unwind(|| assert_decode_routes("hypothetical", [1024], 64))
            .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .expect("panic payload is a String");
        assert!(message.contains("hypothetical"), "{message}");
        assert!(message.contains("ALSO a legal PiD tile edge"), "{message}");
        // The happy path hands the declaration back so a suite can keep using it.
        assert_eq!(
            assert_decode_routes("z_image_turbo", [768, 640, 512], 64).native_edges(),
            &[768, 640, 512]
        );
    }
}
