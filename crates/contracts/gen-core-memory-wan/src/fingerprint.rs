//! Calibration fingerprints, the weights-free contract and the A14B fixture geometry.

use super::*;

/// Sentinel [`repository_policy`] revision for a repository the SceneWorks manifest downloads
/// **without** a `revision` key.
///
/// [`repository_revision`] still requires an HF-cache-shaped
/// `models--<owner>--<name>/snapshots/<40 lowercase hex>[/ <tier>]` path and still returns the
/// concrete snapshot id it found, which the artifact identity hashes — so the seal stays bound to
/// exact content. What it does not do is demand one *particular* id the manifest never promised.
///
/// Every Candle dense leg uses it (sc-22736 review): the manifest ships all three `Wan-AI/…-Diffusers`
/// bf16 downloads (`Wan2.2-TI2V-5B`, `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`) with NO `revision` key
/// and the SceneWorks resolver binds whichever snapshot is staged, so a hard pin here is a promise
/// the manifest never made: the cell reports `runnable` and then fails inside
/// [`repository_revision`] mid-campaign the moment the staged snapshot is any other commit. The
/// content pin is the snapshot id the receipt hashes, exactly as on the T2V leg. The MLX legs and
/// every packed `SceneWorks/…-candle` tier keep their hard revisions: those the manifest DOES pin.
pub(super) const UNPINNED_UPSTREAM_REVISION: &str = "unpinned-upstream-snapshot";

/// The concrete 40-hex snapshot id a synthetic fixture tree must use for a route whose policy
/// revision is [`UNPINNED_UPSTREAM_REVISION`] — the sentinel itself is not a valid snapshot id.
#[doc(hidden)]
pub const FIXTURE_UNPINNED_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// The directory name a fixture (or any caller building a synthetic snapshot tree) must use for
/// `revision`: itself when the policy pins one, and [`FIXTURE_UNPINNED_REVISION`] when it does not.
#[doc(hidden)]
pub fn fixture_revision(revision: &str) -> &str {
    if revision == UNPINNED_UPSTREAM_REVISION {
        FIXTURE_UNPINNED_REVISION
    } else {
        revision
    }
}

/// The production calibration identity this authority publishes for a **sealed** receipt.
///
/// Returns `None` — an opaque, unmeasurable contract — for every route whose identity is owned
/// elsewhere or which ships no tier ladder:
///
/// * [`WanI2vRoute::Ti2v5b`]'s production identity belongs to the per-crate `memory_strategy`
///   modules (`sc-19236-…-mlx-…` in `mlx-gen-wan`, `sc-19223-…-candle-…` in `candle-gen-wan`),
///   which are the registered `MemoryRegistration::contract` for `wan_2_2` on both lanes. Minting a
///   second string here would give one manifest cell two production identities.
/// * The VACE routes are sealed from an artifact *receipt* rather than a tier-suffixed snapshot;
///   the manifest ships them no q4/q8/bf16 ladder, so there is no per-tier cell to name.
///
/// The tier this keys on is **proven against the artifact**, not requested: `repository_revision`
/// refuses a root whose `snapshots/<rev>/<tier>` suffix disagrees with the selected tier (and
/// refuses any suffix at all on a Candle dense root), and `physical_resident_bytes` refuses a
/// packed marker that does not encode the selected Q4/Q8 group-64 layout. A caller cannot reach
/// this function with a tier the on-disk snapshot does not carry.
pub fn production_calibration_fingerprint(
    route: WanI2vRoute,
    backend: WanI2vBackend,
    tier: MemoryNumericTier,
) -> Option<String> {
    let (route_token, backend_token, tier_token) = a14b_cell_tokens(route, backend, tier)?;
    Some(format!(
        "sc-22736-{route_token}-{backend_token}-{tier_token}-v1"
    ))
}

/// The numeric tier a load spec names on `backend`, without opening anything.
///
/// This is the same `tier` resolution every sealed receipt runs — MLX reads the tier off the
/// snapshot's own directory name when `quantize` is unset, Candle reads only `quantize` — exposed
/// so a weights-free registry surface can key on the witness selector's tier rather than
/// reimplementing that split (sc-22736).
pub fn spec_numeric_tier(
    spec: &LoadSpec,
    backend: WanI2vBackend,
) -> gen_core::Result<MemoryNumericTier> {
    tier(spec, backend)
}

/// The `(route, backend, tier)` coordinate every A14B identity minted here is keyed on, or `None`
/// for a cell this authority does not name. [`production_calibration_fingerprint`] documents why
/// each token is part of the key and why the other three routes are silent.
pub(super) fn a14b_cell_tokens(
    route: WanI2vRoute,
    backend: WanI2vBackend,
    tier: MemoryNumericTier,
) -> Option<(&'static str, &'static str, &'static str)> {
    let route_token = match route {
        WanI2vRoute::T2v14b => "wan2-2-t2v-a14b",
        WanI2vRoute::I2v14b => "wan2-2-i2v-a14b",
        WanI2vRoute::Ti2v5b | WanI2vRoute::Vace | WanI2vRoute::VaceFun => return None,
    };
    // The backend token is load-bearing: both lanes seal the same route from *different*
    // repositories with different layouts and different resident arithmetic, so one string across
    // both would let a measurement taken on one lane be read as evidence for the other.
    let backend_token = match backend {
        WanI2vBackend::Mlx => "mlx",
        WanI2vBackend::Candle => "candle",
    };
    let tier_token = match tier.quant {
        None => "dense",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        // `tier` refuses every other quant before a receipt can seal.
        Some(_) => return None,
    };
    Some((route_token, backend_token, tier_token))
}

/// The **weights-free** registry-conformance identity of one A14B cell (sc-22736).
///
/// A catalog walk reads no artifact, so it can never prove a tier and must never publish a
/// measurable identity. The `-weights-free-conformance-` infix puts every string minted here in a
/// namespace no production string can enter, and the disjointness is asserted over the derived cell
/// grid rather than over a frozen list.
pub fn weights_free_calibration_fingerprint(
    route: WanI2vRoute,
    backend: WanI2vBackend,
    tier: MemoryNumericTier,
) -> Option<String> {
    let (route_token, backend_token, tier_token) = a14b_cell_tokens(route, backend, tier)?;
    Some(format!(
        "sc-22736-{route_token}-{backend_token}-{tier_token}-weights-free-conformance-v1"
    ))
}

/// The contract a registry catalog walk resolves for an A14B route whose root is **not**
/// materialized (sc-22736, epic sc-22723 E1).
///
/// Identical in shape to the sealed contract — same strategy support, same lifecycle, same formula,
/// same load shape — with [`MemoryAssetFacts::default`] in place of the header-derived facts and the
/// isolated [`weights_free_calibration_fingerprint`] in place of the production identity. It touches
/// no filesystem at all, which is the whole point: a [`gen_core::MemoryContractFixtureRegistration`] is
/// consulted exactly when the weights are unavailable.
///
/// Only the two A14B routes are served. Ti2V-5B's weights-free fixture belongs to the per-crate
/// `memory_strategy` modules that own its production identity, and the VACE routes register no
/// memory strategy at all, so answering for either here would publish a second fixture for a cell
/// that already has one.
pub fn weights_free_contract(
    provider_id: &str,
    backend: WanI2vBackend,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let route = WanI2vRoute::for_provider(provider_id)?;
    if !matches!(route, WanI2vRoute::T2v14b | WanI2vRoute::I2v14b) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: this authority publishes no weights-free contract for it"
        )));
    }
    let tier = tier(spec, backend)?;
    let mut contract = contract(
        provider_id,
        route,
        backend,
        spec,
        MemoryAssetFacts::default(),
    );
    contract.calibration = weights_free_calibration_fingerprint(route, backend, tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    Ok(contract)
}

/// The one public mode an A14B route admits: T2V carries nothing, I2V carries one Reference.
///
/// Named here rather than at three call sites so the weights-free admission surface and the
/// weights-free behavior fixtures cannot disagree about which carrier the route takes.
pub fn a14b_public_mode_key(route: WanI2vRoute) -> Option<&'static str> {
    match route {
        WanI2vRoute::T2v14b => Some("text_to_video"),
        WanI2vRoute::I2v14b => Some("image_to_video"),
        WanI2vRoute::Ti2v5b | WanI2vRoute::Vace | WanI2vRoute::VaceFun => None,
    }
}

/// The single frame rate the A14B manifest menu publishes (`fps: [16]` on both routes).
///
/// [`WanI2vRoute::accepts_rate`] is still the authority on the `(fps, frames)` pair; this only says
/// which of the rates that authority might admit an A14B cell actually ships, and it is asserted
/// against `accepts_rate` rather than trusted.
pub const A14B_PUBLIC_FPS: u32 = 16;

/// Admission for an A14B route whose weights are **not** materialized (sc-22736, epic sc-22723 E1).
///
/// This is [`validate_context`] with exactly the artifact-bound clauses removed, and nothing else:
///
/// * `ensure_unchanged`, the `artifact_identity` receipt-tail prefix and the `adapter_identity`
///   overlay comparison are all statements about a sealed snapshot, and there is no snapshot here.
///   A weights-free context therefore carries no overlay at all rather than an overlay this
///   function cannot check.
/// * the receipt's embedded frame rate is likewise unavailable, so the rate menu is asked at
///   [`A14B_PUBLIC_FPS`] — which is the only rate either A14B route's manifest ships, and
///   `accepts_rate` still owns whether the frame count belongs to it.
///
/// Everything else — mode, carrier, geometry bucket, batch, PiD/phase exclusion, the selected
/// strategy, and the whole of [`gen_core::standard_memory_strategy_safety_check`] (calibration ABI,
/// fingerprint, load shape and tier) — is the same check the sealed path runs, which is what makes
/// a registry conformance walk's mutation probes bite.
pub fn validate_weights_free_context(
    provider_id: &str,
    backend: WanI2vBackend,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let route = WanI2vRoute::for_provider(provider_id)?;
    let Some(mode_key) = a14b_public_mode_key(route) else {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: this authority has no weights-free admission surface for it"
        )));
    };
    let tier = tier(spec, backend)?;
    let geometry = context.geometry;
    let expected_references = u32::from(route == WanI2vRoute::I2v14b);
    if context.mode.as_key() != mode_key
        || geometry.batch != 1
        || geometry.reference_count != expected_references
        || context.has_reference != (expected_references > 0)
        || context.use_pid
        || context.has_phases
        || context.overlay.is_some()
        || !route
            .public_geometries()
            .contains(&(geometry.width, geometry.height))
        || !route.accepts_rate(A14B_PUBLIC_FPS, geometry.frames)
        || !matches!(
            context.selection.strategy,
            MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                | MemoryStrategy::BoundedDecode
        )
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: crossed weights-free Wan A14B memory context"
        )));
    }
    gen_core::admit_memory_safety(gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(tier),
        None,
    ))
}

/// A geometry every A14B cell admits, for a weights-free behavior fixture: the manifest's own
/// default resolution at its shortest published duration.
///
/// Read back through [`WanI2vRoute::public_geometries`] and [`WanI2vRoute::accepts_rate`] by
/// `a14b_weights_free_fixture_geometry_is_admissible`, so a menu change cannot leave this naming a
/// bucket the route stopped admitting.
pub const A14B_FIXTURE_GEOMETRY: (u32, u32, u32) = (1280, 720, 45);
