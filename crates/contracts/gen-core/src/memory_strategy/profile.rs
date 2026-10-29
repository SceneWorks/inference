//! Table-driven provider memory profiles (sc-27307, review finding F-328).
//!
//! Before this module every adopting provider's `memory_strategy.rs` re-typed the same skeletons
//! around the shared pipeline: resolve the loaded numeric tier from the [`LoadSpec`], run
//! [`standard_memory_strategy_safety_check`] with a provider route gate, wrap it for the
//! weights-free registry, assemble the behavior fixtures, resolve the tier again for
//! `begin_request`, and spell the calibration fingerprint from a route label and a tier label.
//! The copies had drifted in signature (`(spec, contract, context)`, `(contract, tier, context)`,
//! `(provider_id, contract, context, loaded_quant)`) and in where a provider-owned refusal ran
//! relative to the shared handshake.
//!
//! A [`ProviderMemoryProfile`] is the data those skeletons were parameterised over. A provider
//! supplies the profile; the `standard_*` functions below are the one copy of each skeleton:
//!
//! | skeleton                              | standard function                         |
//! |---------------------------------------|-------------------------------------------|
//! | loaded-tier safety check              | [`standard_safety_check`]                 |
//! | the same check, `Result`-shaped       | [`standard_validate_context`]             |
//! | weights-free registry safety check    | [`standard_registered_safety_check`]      |
//! | weights-free `begin_request`          | [`standard_begin_request`]                |
//! | weights-free behavior fixtures        | [`standard_registered_valid_fixture`]     |
//! | calibration fingerprint spelling      | [`MemoryFingerprintTable::fingerprint`]   |
//! | architecture facts weights-free guard | [`standard_architecture_facts`]           |
//!
//! What a provider genuinely owns stays an explicit hook on the profile rather than a hand-rolled
//! copy of the whole skeleton: how its artifact names the loaded tier ([`MemoryProfileTier`]), its
//! route refusals and where they run ([`MemoryProfileGates`]), and how its geometry is derived
//! from a snapshot ([`MemoryArchitectureSource`]). Request-scope construction is backend-owned, so
//! [`standard_begin_request`] takes it as a [`MemoryProfileScope`] argument rather than as profile
//! data (a backend may compile it only on the lane that executes it). `check-workspace.py`
//! ratchets the providers that still define a bespoke skeleton instead of a profile, so the count
//! can only fall.
//!
//! The profile is tensor-neutral by construction: every hook takes contract-level types only.

use super::*;
use crate::LoadSpec;
use std::path::Path;

/// A provider-owned route refusal: `Ok(())` admits, an error refuses.
///
/// It sees the contract, the run context and the numeric tier the provider loaded, which is all a
/// loaded generator and the weights-free registry both hold. A rule that needs anything else (an
/// artifact seal, a prepared identity, the raw [`LoadSpec`]) is a genuine implementation difference
/// and stays a provider-owned safety check.
pub type MemoryProfileGate =
    fn(&MemoryProviderContract, &MemoryRunContext, MemoryNumericTier) -> Result<()>;

/// Resolve the numeric tier a load from `spec` actually materializes, for the contract being
/// admitted (its `provider_id` names the route).
///
/// The contract is an input because a provider's weights-free tier can depend on which contract it
/// is: a registry witness pins its fixture tier while a production contract carries the tier its
/// artifact receipt proved. A resolver that cannot reproduce the contract it is handed refuses.
pub type MemoryProfileTierResolver =
    fn(&LoadSpec, &MemoryProviderContract) -> Result<MemoryNumericTier>;

/// Build the provider's request scope for an already-admitted weights-free request.
///
/// [`standard_begin_request`] runs the profile's safety check before calling this, so the hook
/// owns construction only. A hook that re-runs its own (identical) admission is harmless.
pub type MemoryProfileScope = fn(
    &'static str,
    &LoadSpec,
    &MemoryProviderContract,
    &MemoryRunContext,
    MemoryNumericTier,
) -> Result<Option<Box<dyn MemoryRequestScope>>>;

/// How a weights-free registration names the numeric tier the generator loads.
#[derive(Clone, Copy)]
pub enum MemoryProfileTier {
    /// The requested `spec.precision` and `spec.quantize`, with no component floors.
    Requested,
    /// A fixed compute precision (a backend that always loads one precision) and the requested
    /// `spec.quantize`, with no component floors.
    FixedPrecision(Precision),
    /// Provider-owned resolution, for a provider whose artifact (not the request) decides the
    /// tier — for example a pre-packed transformer whose `config.json` names its width.
    Resolved(MemoryProfileTierResolver),
}

impl MemoryProfileTier {
    /// The numeric tier a load from `spec` materializes for `contract`.
    pub fn resolve(
        self,
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
    ) -> Result<MemoryNumericTier> {
        match self {
            Self::Requested => Ok(MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            }),
            Self::FixedPrecision(precision) => Ok(MemoryNumericTier {
                precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            }),
            Self::Resolved(resolve) => resolve(spec, contract),
        }
    }
}

/// Provider route refusals, by where they run relative to the shared pipeline.
///
/// The three positions exist because the providers' refusals genuinely differ in precedence, and
/// precedence is observable: when a context fails both a provider rule and a shared rule, the
/// rejection names whichever ran first.
#[derive(Clone, Copy, Default)]
pub struct MemoryProfileGates {
    /// Runs before the shared handshake (a structural route refusal that must win over every
    /// shared rejection).
    pub before: Option<MemoryProfileGate>,
    /// Runs inside [`standard_memory_strategy_safety_check`] as its `route_gate`: after the
    /// handshake, tier and selection checks, before the budget check. The ordinary position.
    pub within: Option<MemoryProfileGate>,
    /// Runs after the shared pipeline accepted (a refusal that must not mask a shared rejection).
    pub after: Option<MemoryProfileGate>,
}

impl MemoryProfileGates {
    /// No provider gate: the shared pipeline alone.
    pub const NONE: Self = Self {
        before: None,
        within: None,
        after: None,
    };

    /// Only a `before` gate.
    pub const fn before(gate: MemoryProfileGate) -> Self {
        Self {
            before: Some(gate),
            ..Self::NONE
        }
    }

    /// Only a `within` gate: the ordinary provider route gate.
    pub const fn within(gate: MemoryProfileGate) -> Self {
        Self {
            within: Some(gate),
            ..Self::NONE
        }
    }

    /// Only an `after` gate.
    pub const fn after(gate: MemoryProfileGate) -> Self {
        Self {
            after: Some(gate),
            ..Self::NONE
        }
    }
}

/// The mode of one weights-free behavior fixture route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFixtureMode {
    TextToImage,
    ImageToImage,
    Edit,
    /// A provider-specific mode key ([`MemoryMode::Other`]).
    Other(&'static str),
}

impl MemoryFixtureMode {
    fn mode(self) -> MemoryMode {
        match self {
            Self::TextToImage => MemoryMode::TextToImage,
            Self::ImageToImage => MemoryMode::ImageToImage,
            Self::Edit => MemoryMode::Edit,
            Self::Other(key) => MemoryMode::Other(key.to_owned()),
        }
    }
}

/// Whether a fixture route's context declares request phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFixturePhases {
    Never,
    Always,
    /// When the fixture's strategy engages [`MemoryStrategy::StagedResidency`].
    WhenStaged,
    /// When the load spec asks for [`crate::OffloadPolicy::Sequential`].
    WhenSequentialOffload,
}

/// A fixture route's overlay identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFixtureOverlay {
    None,
    Fixed(&'static str),
    /// The key when the load spec carries adapters, otherwise no overlay.
    WhenAdapters(&'static str),
}

/// One weights-free behavior fixture route: the data of a [`MemoryBehaviorRoute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFixtureRoute {
    pub mode: MemoryFixtureMode,
    pub reference_count: u32,
    pub phases: MemoryFixturePhases,
    pub overlay: MemoryFixtureOverlay,
}

impl MemoryFixtureRoute {
    /// Plain single-phase text-to-image without references or overlays.
    pub const TEXT_TO_IMAGE: Self = Self::new(MemoryFixtureMode::TextToImage, 0);

    /// A single-phase route without overlays.
    pub const fn new(mode: MemoryFixtureMode, reference_count: u32) -> Self {
        Self {
            mode,
            reference_count,
            phases: MemoryFixturePhases::Never,
            overlay: MemoryFixtureOverlay::None,
        }
    }

    /// This route with request phases declared as `phases`.
    pub const fn with_phases(mut self, phases: MemoryFixturePhases) -> Self {
        self.phases = phases;
        self
    }

    /// This route with the overlay identity `overlay`.
    pub const fn with_overlay(mut self, overlay: MemoryFixtureOverlay) -> Self {
        self.overlay = overlay;
        self
    }

    fn route(
        self,
        spec: &LoadSpec,
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
        use_pid: bool,
    ) -> MemoryBehaviorRoute {
        MemoryBehaviorRoute {
            mode: self.mode.mode(),
            reference_count: self.reference_count,
            use_pid,
            has_phases: match self.phases {
                MemoryFixturePhases::Never => false,
                MemoryFixturePhases::Always => true,
                MemoryFixturePhases::WhenStaged => {
                    contract.engages(strategy, MemoryStrategy::StagedResidency)
                }
                MemoryFixturePhases::WhenSequentialOffload => {
                    spec.offload_policy == crate::OffloadPolicy::Sequential
                }
            },
            overlay: match self.overlay {
                MemoryFixtureOverlay::None => None,
                MemoryFixtureOverlay::Fixed(key) => Some(key.to_owned()),
                MemoryFixtureOverlay::WhenAdapters(key) => {
                    (!spec.adapters.is_empty()).then(|| key.to_owned())
                }
            },
        }
    }
}

/// When a route also gets a PiD-decode copy of its fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFixturePid {
    Never,
    /// When the strategy engages [`MemoryStrategy::BoundedDecode`].
    WhenBoundedDecode,
    /// When the strategy engages [`MemoryStrategy::BoundedDecode`] and the contract declares
    /// separate PiD decode routes.
    WhenPidDecodeRoutes,
}

/// The weights-free behavior fixtures a provider publishes for each optimized strategy.
#[derive(Clone, Copy, Debug)]
pub struct MemoryProfileFixtures {
    /// One fixture per route (plus its PiD copy, per [`Self::pid`]), in this order.
    pub routes: &'static [MemoryFixtureRoute],
    /// Publish nothing for a strategy the contract does not declare
    /// [`MemoryStrategySupport::Implemented`].
    pub implemented_only: bool,
    /// Publish nothing when the contract has no calibration identity.
    pub calibrated_only: bool,
    pub pid: MemoryFixturePid,
    /// A provider prompt for every fixture request, instead of the empty default.
    pub prompt: Option<&'static str>,
}

impl MemoryProfileFixtures {
    /// One plain text-to-image fixture per optimized strategy.
    pub const TEXT_TO_IMAGE: Self = Self::routes(&[MemoryFixtureRoute::TEXT_TO_IMAGE]);

    /// The given routes, for every optimized strategy, without PiD copies or a prompt.
    pub const fn routes(routes: &'static [MemoryFixtureRoute]) -> Self {
        Self {
            routes,
            implemented_only: false,
            calibrated_only: false,
            pid: MemoryFixturePid::Never,
            prompt: None,
        }
    }

    /// These fixtures, published only for an implemented strategy.
    pub const fn implemented_only(mut self) -> Self {
        self.implemented_only = true;
        self
    }

    /// These fixtures, published only for a calibrated contract.
    pub const fn calibrated_only(mut self) -> Self {
        self.calibrated_only = true;
        self
    }

    /// These fixtures with PiD copies per `pid`.
    pub const fn with_pid(mut self, pid: MemoryFixturePid) -> Self {
        self.pid = pid;
        self
    }

    /// These fixtures with `prompt` on every request.
    pub const fn with_prompt(mut self, prompt: &'static str) -> Self {
        self.prompt = Some(prompt);
        self
    }
}

/// A pinned calibration fingerprint for one `(route, tier)` cell.
///
/// Pins exist because a fingerprint is an identity: a cell whose measurement predates the
/// template's spelling keeps the name it was measured under, byte for byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFingerprintPin {
    pub route: &'static str,
    pub tier: &'static str,
    pub fingerprint: &'static str,
}

/// The spelling of a provider family's production calibration fingerprints.
#[derive(Clone, Copy, Debug)]
pub struct MemoryFingerprintTable {
    /// The fingerprint with `{route}` and `{tier}` placeholders.
    pub template: &'static str,
    /// Provider id to route label. An id absent here has no production fingerprint.
    pub routes: &'static [(&'static str, &'static str)],
    /// Cells that keep a pinned fingerprint instead of the template.
    pub pins: &'static [MemoryFingerprintPin],
}

impl MemoryFingerprintTable {
    /// The route label of `provider_id`, or `None` when the family does not publish it.
    pub fn route(&self, provider_id: &str) -> Option<&'static str> {
        self.routes
            .iter()
            .find_map(|(id, route)| (*id == provider_id).then_some(*route))
    }

    /// The fingerprint of the `(route, tier)` cell: its pin, otherwise the template.
    pub fn render(&self, route: &str, tier: &str) -> String {
        self.pins
            .iter()
            .find(|pin| pin.route == route && pin.tier == tier)
            .map(|pin| pin.fingerprint.to_owned())
            .unwrap_or_else(|| {
                self.template
                    .replace("{route}", route)
                    .replace("{tier}", tier)
            })
    }

    /// The fingerprint of `provider_id` at `tier`, or `None` for an unpublished provider id.
    pub fn fingerprint(&self, provider_id: &str, tier: &str) -> Option<String> {
        Some(self.render(self.route(provider_id)?, tier))
    }
}

/// Where a provider's [`MemoryArchitectureFacts`] come from.
#[derive(Clone, Copy)]
pub enum MemoryArchitectureSource {
    /// Geometry fixed by the provider, known without a snapshot.
    Declared(MemoryArchitectureFacts),
    /// Geometry derived from a resolved snapshot root. A load with no resolved snapshot directory
    /// (a weights-free sentinel path, or a single-file import) declares no axis.
    Snapshot(fn(&str, &Path, &LoadSpec) -> MemoryArchitectureFacts),
}

/// How a refusal renders as a [`MemorySafetyDecision::Reject`] reason.
///
/// Both conventions are pinned by provider tests, so the profile names the one a provider uses
/// rather than the hoist silently picking one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryRejectionRendering {
    /// The shared pipeline's reason verbatim, and a gate's error rendered once. The gen-core and
    /// MLX convention.
    #[default]
    Reason,
    /// The rendering of the typed [`standard_validate_context`] error, so a shared refusal reads
    /// `unsupported: <reason>`. The Candle convention: a provider's loaded and registered decisions
    /// read exactly as its `begin_request` error does.
    ValidationError,
}

/// The data a provider family's memory-strategy skeletons are parameterised over.
#[derive(Clone, Copy)]
pub struct ProviderMemoryProfile {
    pub tier: MemoryProfileTier,
    pub gates: MemoryProfileGates,
    pub rejection: MemoryRejectionRendering,
    pub fixtures: MemoryProfileFixtures,
    pub fingerprint: Option<MemoryFingerprintTable>,
    pub architecture: Option<MemoryArchitectureSource>,
}

impl ProviderMemoryProfile {
    /// A resident-only profile: requested tier, no provider gates, plain text-to-image fixtures,
    /// no fingerprint or architecture source. Providers override fields from here.
    pub const STANDARD: Self = Self {
        tier: MemoryProfileTier::Requested,
        gates: MemoryProfileGates::NONE,
        rejection: MemoryRejectionRendering::Reason,
        fixtures: MemoryProfileFixtures::TEXT_TO_IMAGE,
        fingerprint: None,
        architecture: None,
    };
}

/// Which stage of the profile pipeline refused a context.
enum ProfileRefusal {
    /// The shared pipeline, with its rendered reason (a `within` gate's error included).
    Shared(String),
    /// A `before` or `after` gate, with its typed error.
    Gate(Error),
}

fn evaluate(
    profile: &ProviderMemoryProfile,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_tier: MemoryNumericTier,
) -> std::result::Result<(), ProfileRefusal> {
    if let Some(before) = profile.gates.before {
        before(contract, context, loaded_tier).map_err(ProfileRefusal::Gate)?;
    }
    let within = profile
        .gates
        .within
        .map(|gate| move || gate(contract, context, loaded_tier));
    let route_gate = within.as_ref().map(|gate| gate as &dyn Fn() -> Result<()>);
    if let MemorySafetyDecision::Reject { reason } =
        standard_memory_strategy_safety_check(contract, context, Some(loaded_tier), route_gate)
    {
        return Err(ProfileRefusal::Shared(reason));
    }
    if let Some(after) = profile.gates.after {
        after(contract, context, loaded_tier).map_err(ProfileRefusal::Gate)?;
    }
    Ok(())
}

/// The profile's safety check for a generator that loaded `loaded_tier`.
///
/// `before` gate, then the shared pipeline with the `within` gate as its route gate, then the
/// `after` gate. A gate's refusal reason is its error's rendering; a shared refusal renders per the
/// profile's [`MemoryRejectionRendering`].
pub fn standard_safety_check(
    profile: &ProviderMemoryProfile,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    match evaluate(profile, contract, context, loaded_tier) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(ProfileRefusal::Shared(reason)) => match profile.rejection {
            MemoryRejectionRendering::Reason => MemorySafetyDecision::Reject { reason },
            MemoryRejectionRendering::ValidationError => {
                MemorySafetyDecision::rejected(Error::Unsupported(reason))
            }
        },
        Err(ProfileRefusal::Gate(error)) => MemorySafetyDecision::rejected(error),
    }
}

/// [`standard_safety_check`] as a `Result`, for a provider's request-scope admission.
///
/// A shared-pipeline refusal is [`Error::Unsupported`] carrying its reason; a `before` or `after`
/// gate's error is returned as the gate typed it, not re-rendered.
pub fn standard_validate_context(
    profile: &ProviderMemoryProfile,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_tier: MemoryNumericTier,
) -> Result<()> {
    evaluate(profile, contract, context, loaded_tier).map_err(|refusal| match refusal {
        ProfileRefusal::Shared(reason) => Error::Unsupported(reason),
        ProfileRefusal::Gate(error) => error,
    })
}

/// The weights-free registry safety check ([`crate::MemoryRegistration::safety_check`]): resolve
/// the tier the generator would load, then [`standard_safety_check`]. An unresolvable tier refuses.
pub fn standard_registered_safety_check(
    profile: &ProviderMemoryProfile,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match profile.tier.resolve(spec, contract) {
        Ok(tier) => standard_safety_check(profile, contract, context, tier),
        Err(error) => MemorySafetyDecision::rejected(error),
    }
}

/// The weights-free `begin_request` ([`crate::MemoryBehaviorRegistration::begin_request`]):
/// resolve the tier (an unresolvable tier is returned as typed), admit the context through
/// [`standard_validate_context`] (so a gate's typed error is not re-rendered into a second
/// `unsupported: ` prefix), then build the request scope with the provider's `scope` hook.
pub fn standard_begin_request(
    profile: &ProviderMemoryProfile,
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    scope: MemoryProfileScope,
) -> Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = profile.tier.resolve(spec, contract)?;
    standard_validate_context(profile, contract, context, tier)?;
    scope(provider_id, spec, contract, context, tier)
}

/// The weights-free behavior fixtures ([`crate::MemoryBehaviorRegistration::valid_fixtures`]) the
/// profile declares for `strategy`: none for the resident rung (or a strategy the profile's
/// [`MemoryProfileFixtures`] gates out), otherwise one per route plus its PiD copy.
pub fn standard_registered_valid_fixture(
    profile: &ProviderMemoryProfile,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> Result<Vec<crate::MemoryBehaviorFixture>> {
    let fixtures = &profile.fixtures;
    if !strategy.is_optimized()
        || (fixtures.implemented_only
            && !matches!(
                contract
                    .capability(strategy)
                    .map(|capability| &capability.support),
                Some(MemoryStrategySupport::Implemented)
            ))
        || (fixtures.calibrated_only && contract.calibration.is_none())
    {
        return Ok(Vec::new());
    }
    let tier = profile.tier.resolve(spec, contract)?;
    let pid_copy = contract.engages(strategy, MemoryStrategy::BoundedDecode)
        && match fixtures.pid {
            MemoryFixturePid::Never => false,
            MemoryFixturePid::WhenBoundedDecode => true,
            MemoryFixturePid::WhenPidDecodeRoutes => contract.pid_decode_routes.is_some(),
        };
    let mut out = Vec::new();
    for route in fixtures.routes {
        for use_pid in [false, true] {
            if use_pid && !pid_copy {
                continue;
            }
            let mut fixture = crate::MemoryBehaviorFixture::new(standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route.route(spec, contract, strategy, use_pid),
            )?);
            if let Some(prompt) = fixtures.prompt {
                fixture.request.prompt = prompt.to_owned();
            }
            out.push(fixture);
        }
    }
    Ok(out)
}

/// The resolved snapshot directory of `spec`, if it names one that exists.
///
/// The "a real snapshot was resolved" signal for [`MemoryArchitectureSource::Snapshot`]: a
/// weights-free contract surface names a sentinel path that is deliberately not on disk, and a
/// single-file import is not a component snapshot.
pub fn resolved_snapshot_root(spec: &LoadSpec) -> Option<&Path> {
    match &spec.weights {
        crate::WeightsSource::Dir(root) if root.is_dir() => Some(root.as_path()),
        _ => None,
    }
}

/// The profile's architecture facts for `provider_id` loaded from `spec`. A profile without an
/// architecture source, or a snapshot source without a resolved snapshot, declares no axis.
pub fn standard_architecture_facts(
    profile: &ProviderMemoryProfile,
    provider_id: &str,
    spec: &LoadSpec,
) -> MemoryArchitectureFacts {
    match profile.architecture {
        None => MemoryArchitectureFacts::default(),
        Some(MemoryArchitectureSource::Declared(facts)) => facts,
        Some(MemoryArchitectureSource::Snapshot(derive)) => resolved_snapshot_root(spec)
            .map(|root| derive(provider_id, root, spec))
            .unwrap_or_default(),
    }
}

/// A [`MemoryRegistration`](crate::MemoryRegistration) whose weights-free safety check is
/// [`standard_registered_safety_check`] over `$profile`.
///
/// `$profile` names a `const` [`ProviderMemoryProfile`]; `$contract` is the registration's
/// contract callback.
#[macro_export]
macro_rules! profile_memory_registration {
    ($profile:expr, $provider_id:expr, $contract:expr $(,)?) => {
        $crate::MemoryRegistration {
            provider_id: $provider_id,
            contract: $contract,
            safety_check: |spec, contract, context| {
                $crate::standard_registered_safety_check(&$profile, spec, contract, context)
            },
        }
    };
}

/// A [`MemoryBehaviorRegistration`](crate::MemoryBehaviorRegistration) whose fixtures and
/// `begin_request` are [`standard_registered_valid_fixture`] and [`standard_begin_request`] over
/// `$profile` for the route `$provider_id`, building its request scope with `$scope` (a
/// [`MemoryProfileScope`]).
#[macro_export]
macro_rules! profile_memory_behavior_registration {
    ($profile:expr, $provider_id:expr, $scope:expr $(,)?) => {
        $crate::MemoryBehaviorRegistration {
            provider_id: $provider_id,
            valid_fixtures: |spec, contract, strategy| {
                $crate::standard_registered_valid_fixture(&$profile, spec, contract, strategy)
            },
            begin_request: |spec, contract, context| {
                $crate::standard_begin_request(
                    &$profile,
                    $provider_id,
                    spec,
                    contract,
                    context,
                    $scope,
                )
            },
        }
    };
}

#[cfg(test)]
mod tests;
