//! The provider memory contract, its runtime semantics, and contract validation.

use super::*;

/// Request-level cache keys must include every axis that can change residency or execution.
///
/// A warm generator is owned by one resolved provider, backend realization, and load shape, so
/// those axes are invariant inside this contract. A caller whose outer cache spans loaded
/// generators must key that outer cache by those loader identities too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheSemantics {
    StrategyTierParametersModeGeometryOverlayAndEngagedComposition,
}

/// Warm generators must not inherit a prior request's memory decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryWarmRunSemantics {
    RevalidateBudgetAndReapplyRequestState,
}

/// Cancellation and error cleanup must synchronize backend work before releasing active state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCleanupSemantics {
    SynchronizeAndReleaseActivePhasesAndWindows,
}

/// Runtime semantics required of every adopting provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRuntimeSemantics {
    pub cache: MemoryCacheSemantics,
    pub warm_run: MemoryWarmRunSemantics,
    pub cancellation: MemoryCleanupSemantics,
    pub error: MemoryCleanupSemantics,
}

impl Default for MemoryRuntimeSemantics {
    fn default() -> Self {
        Self {
            cache:
                MemoryCacheSemantics::StrategyTierParametersModeGeometryOverlayAndEngagedComposition,
            warm_run: MemoryWarmRunSemantics::RevalidateBudgetAndReapplyRequestState,
            cancellation: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
            error: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
        }
    }
}

/// A provider-verified exception to the shared cumulative cost-order composition.
///
/// This can remove only a default engagement edge. It cannot remove a prerequisite, and the provider
/// must name the calibration or executable proof that established the cheaper composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryStrategyEngagementExclusion {
    pub selection: MemoryStrategy,
    pub excluded_rung: MemoryStrategy,
    pub evidence: String,
}

/// Representation of an explicit resident selection in [`crate::GenerationRequest::memory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidentRequestMemory {
    /// Leave the request memory block absent and preserve the loaded provider's historical defaults.
    #[default]
    PreserveLoadDefaults,
    /// Write `Some(GenerationMemory::default())` to disable load-time sequential/windowed defaults.
    ExplicitResident,
}

/// Static provider contract returned before weights are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryProviderContract {
    pub provider_id: String,
    pub backend: MemoryBackendRealization,
    pub strategies: Vec<MemoryStrategyCapability>,
    /// Whether the caller declared semantic decode authority for this loaded route even when no
    /// row survives the exact tier/load-shape projection. This distinguishes a genuinely legacy
    /// empty table from a declared-but-inapplicable/refused table, which must remain fail closed.
    pub decode_geometry_policy_authoritative: bool,
    /// Exact native/PiD route split when this provider supports PiD decode. The ordinary strategy
    /// ranges publish the union; this declaration makes the route distinction visible to weights-free
    /// registry conformance and provider admission checks.
    pub pid_decode_routes: Option<MemoryPidDecodeRoutes>,
    /// The load-time materialization shape this concrete provider instance was built with.
    pub load_shape: LoadShape,
    /// Realization-specific constraints appended to the shared graph. Providers cannot remove or
    /// replace [`MemoryStrategy::requires`].
    pub additional_prerequisites: Vec<(MemoryStrategy, MemoryStrategyPrerequisite)>,
    /// Documented, provider-verified exceptions to the shared cumulative cost-order default.
    /// Prerequisite edges remain authoritative and cannot be removed through this list.
    pub default_engagement_exclusions: Vec<MemoryStrategyEngagementExclusion>,
    /// How an explicit shared-contract `Resident` selection is represented on the generation
    /// request. Most providers preserve their historical load-time defaults by leaving the memory
    /// block absent. Providers whose load defaults themselves enable sequential/windowed execution
    /// opt into an explicit all-disabled block so `Resident` can override those defaults.
    pub resident_request_memory: ResidentRequestMemory,
    pub lifecycle: MemoryLifecycleCapabilities,
    pub formula: MemoryFormulaKind,
    /// `None` is the compatibility default for a provider that has not adopted calibration yet.
    /// Such a provider can run its resident path but can never claim a verified optimized fit.
    pub calibration: Option<MemoryCalibrationIdentity>,
    pub asset_facts: MemoryAssetFacts,
    /// Snapshot-read architecture axes. `MemoryArchitectureFacts::default()` (every axis `None`) is
    /// the compatibility state for a provider that has not adopted the axis yet, and the honest
    /// state for a weights-free contract built with no snapshot on disk.
    pub architecture_facts: MemoryArchitectureFacts,
    /// Phase lifetimes and workspace topology from the actual provider loader.
    pub phase_facts: Option<crate::MemoryPhaseFacts>,
    pub runtime: MemoryRuntimeSemantics,
}

impl MemoryProviderContract {
    /// Safe compatibility view for an existing provider: resident only, optimized rungs missing,
    /// and no calibration identity. This preserves existing behavior without fabricating evidence.
    pub fn compatibility_default(
        provider_id: impl Into<String>,
        backend: MemoryBackendRealization,
    ) -> Self {
        let provider_id = provider_id.into();
        Self {
            provider_id,
            backend,
            strategies: MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| MemoryStrategyCapability {
                    strategy,
                    support: if strategy == MemoryStrategy::Resident {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    parameters: MemoryParameterRanges::default(),
                })
                .collect(),
            decode_geometry_policy_authoritative: false,
            pid_decode_routes: None,
            load_shape: LoadShape::EagerMaterialization,
            additional_prerequisites: Vec::new(),
            default_engagement_exclusions: Vec::new(),
            resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
            lifecycle: MemoryLifecycleCapabilities::default(),
            formula: MemoryFormulaKind::AssetBytesPlusHeadroom,
            calibration: None,
            asset_facts: MemoryAssetFacts::default(),
            phase_facts: None,
            architecture_facts: MemoryArchitectureFacts::default(),
            runtime: MemoryRuntimeSemantics::default(),
        }
    }

    /// Add load-exact auxiliary residency to a provider's existing base prediction.
    ///
    /// Both declaration legs are required: typed components make the total decomposable, while
    /// [`MemoryFormulaVariable::OverlayBytes`] makes the compatibility aggregate load-bearing. A
    /// pre-SC-16065 provider has no typed components, so this returns the byte-identical input scalar.
    pub fn predicted_peak_from_base(&self, base_predicted_peak_bytes: u64) -> MemoryPeakBreakdown {
        let components = self.auxiliary_components().cloned().collect::<Vec<_>>();
        let auxiliary_bytes = self.auxiliary_overlay_bytes_for_prediction();
        MemoryPeakBreakdown {
            predicted_peak_bytes: base_predicted_peak_bytes.saturating_add(auxiliary_bytes),
            unattributed_bytes: base_predicted_peak_bytes,
            components,
        }
    }

    /// Decompose a scalar which already includes this provider's auxiliary residency. The scalar is
    /// never rewritten; this is the inspection seam for run contexts and evidence records.
    pub fn decompose_predicted_peak(&self, predicted_peak_bytes: u64) -> MemoryPeakBreakdown {
        let components = self.auxiliary_components().cloned().collect::<Vec<_>>();
        let auxiliary_bytes = self.auxiliary_overlay_bytes_for_prediction();
        MemoryPeakBreakdown {
            predicted_peak_bytes,
            unattributed_bytes: predicted_peak_bytes.saturating_sub(auxiliary_bytes),
            components,
        }
    }

    fn auxiliary_components(&self) -> impl Iterator<Item = &MemoryResidentComponent> {
        self.formula
            .resident_components()
            .iter()
            .filter(|component| component.kind.is_auxiliary())
    }

    /// Compatibility aggregate attributed to typed auxiliary components in peak calculations.
    ///
    /// The typed-component sum and this aggregate are intentionally separate declaration legs;
    /// conformance proves they agree. Legacy contracts with only an aggregate must not gain a new
    /// contribution merely by calling the decomposition helpers.
    fn auxiliary_overlay_bytes_for_prediction(&self) -> u64 {
        if self.auxiliary_components().next().is_some() {
            self.asset_facts.overlay_bytes
        } else {
            0
        }
    }

    /// Typed resident component declarations, empty for every non-adopting provider.
    pub fn resident_components(&self) -> &[MemoryResidentComponent] {
        self.formula.resident_components()
    }

    /// Bytes the declared components drop before their phase reaches steady state (SC-18665).
    ///
    /// **Zero** for every provider that declares only [`MemoryComponentResidency::WholeRender`], so
    /// this reads the same on a contract written before the field existed as it did before.
    pub fn evicted_component_bytes(&self) -> u64 {
        self.resident_components()
            .iter()
            .fold(0_u64, |total, component| {
                total.saturating_add(component.evicted_bytes())
            })
    }

    /// [`MemoryAssetFacts::transformer_bytes`] corrected for declared intra-transformer evictions:
    /// the transformer residency an estimate for the **post-precompute steady state** should
    /// charge, rather than the load-exact total that is only true at the precompute instant.
    ///
    /// Only [`MemoryComponentKind::TransformerSubStack`] declarations move this. An auxiliary
    /// network's residency is not inside `transformer_bytes` and must not be subtracted from it,
    /// and a whole-[`MemoryComponentKind::Transformer`] declaration has nothing to drop. So this
    /// returns `asset_facts.transformer_bytes` **unchanged** for every provider that has not
    /// adopted the sub-stack vocabulary.
    pub fn steady_state_transformer_bytes(&self) -> u64 {
        let evicted = self
            .resident_components()
            .iter()
            .filter(|component| {
                matches!(component.kind, MemoryComponentKind::TransformerSubStack(_))
            })
            .fold(0_u64, |total, component| {
                total.saturating_add(component.evicted_bytes())
            });
        self.asset_facts.transformer_bytes.saturating_sub(evicted)
    }

    /// Load-exact bytes attributable to auxiliary component declarations.
    pub fn auxiliary_resident_bytes(&self) -> u64 {
        self.auxiliary_components().fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        })
    }

    /// Resident bytes safe to credit to a warm provider. A legacy declaration preserves the old
    /// `base_bytes` meaning; an adopter adds the separately declared overlay.
    pub fn total_resident_bytes(&self) -> u64 {
        if self.auxiliary_components().next().is_some() {
            self.asset_facts
                .base_bytes
                .saturating_add(self.asset_facts.overlay_bytes)
        } else {
            self.asset_facts.base_bytes
        }
    }

    pub fn capability(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategyCapability> {
        self.strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
    }

    /// Replace a withheld provider's route-blind rung-2 declaration with sealed, geometry-aware
    /// quality policy. An empty input is a no-op so existing defaults and estimated fallback remain
    /// byte-for-byte compatible until a packaged quality table exists.
    ///
    /// Refused rows remain published as fail-closed knowledge. Only admitted rows contribute to the
    /// selectable edge/overlap union or arm the production decode lifecycle.
    pub fn adopt_decode_geometry_policies(
        &mut self,
        expected_family: &str,
        policies: Vec<MemoryDecodeGeometryPolicy>,
    ) -> Result<bool> {
        if policies.is_empty() {
            return Ok(false);
        }
        self.decode_geometry_policy_authoritative = true;
        if expected_family.trim().is_empty()
            || policies
                .iter()
                .any(|policy| policy.family != expected_family)
        {
            return Err(Error::Unsupported(format!(
                "{}: decode-quality policy does not match expected family {:?}",
                self.provider_id, expected_family
            )));
        }
        let capability = self
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{}: decode-quality policy requires a BoundedDecode capability",
                    self.provider_id
                ))
            })?;
        let mut edges = policies
            .iter()
            .filter(|policy| matches!(policy.disposition, MemoryDecodeQualityDisposition::Admitted))
            .map(|policy| policy.tile_edge)
            .collect::<Vec<_>>();
        let mut overlaps = policies
            .iter()
            .filter(|policy| matches!(policy.disposition, MemoryDecodeQualityDisposition::Admitted))
            .map(|policy| policy.overlap)
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges.dedup();
        overlaps.sort_unstable();
        overlaps.dedup();
        let admitted = !edges.is_empty();
        capability.support = if admitted {
            MemoryStrategySupport::Implemented
        } else {
            MemoryStrategySupport::Missing
        };
        capability.parameters.decode_tile_edges = edges;
        capability.parameters.decode_overlaps = overlaps;
        capability.parameters.decode_geometry_policies = policies;
        self.lifecycle.decode_tiling = admitted;

        let mut errors = Vec::new();
        validate_decode_geometry_policies(self, &mut errors);
        if errors.is_empty() {
            Ok(admitted)
        } else {
            Err(Error::Unsupported(format!(
                "{}: invalid decode-quality policy: {}",
                self.provider_id,
                errors.join("; ")
            )))
        }
    }

    /// Resolve the exact production-latent quality row for bounded decode.
    ///
    /// `Ok(None)` means this provider has not adopted geometry-aware quality policy and retains its
    /// legacy route-blind domain. Once any policy row is declared, every coordinate is closed: an
    /// admitted row returns `Some`, while a recorded failure, an unmeasured coordinate, or a
    /// tile/overlap mismatch returns a typed refusal. The tile pair is part of the exact coordinate:
    /// a selector that has not chosen one must enumerate
    /// [`decode_geometry_policies_for_request`](Self::decode_geometry_policies_for_request) rather
    /// than rely on table order.
    pub fn decode_geometry_policies_for_request(
        &self,
        query: MemoryDecodePolicyQuery<'_>,
    ) -> Result<Vec<&MemoryDecodeGeometryPolicy>> {
        let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) else {
            return Ok(Vec::new());
        };
        let policies = &capability.parameters.decode_geometry_policies;
        if policies.is_empty() {
            return Ok(Vec::new());
        }
        Ok(policies
            .iter()
            .filter(|policy| policy.matches_request_axes(self, query))
            .collect())
    }

    pub fn decode_geometry_policy(
        &self,
        query: MemoryDecodePolicyQuery<'_>,
        selected_parameters: Option<MemoryStrategyParameters>,
    ) -> Result<Option<&MemoryDecodeGeometryPolicy>> {
        let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) else {
            return Ok(None);
        };
        let policies = &capability.parameters.decode_geometry_policies;
        if policies.is_empty() {
            if self.decode_geometry_policy_authoritative {
                return Err(Error::Unsupported(format!(
                    "{}: bounded decode has a declared quality scope but no applicable row for the loaded tier/load shape",
                    self.provider_id
                )));
            }
            return Ok(None);
        }
        let request_rows = self.decode_geometry_policies_for_request(query)?;
        let had_request_rows = !request_rows.is_empty();
        let selected_pair = selected_parameters
            .and_then(|parameters| parameters.decode_tile_edge.zip(parameters.decode_overlap));
        let policy = if let Some((tile_edge, overlap)) = selected_pair {
            request_rows
                .into_iter()
                .find(|policy| policy.matches_coordinate(self, query, tile_edge, overlap))
        } else if request_rows.len() == 1 {
            request_rows.into_iter().next()
        } else if request_rows.is_empty() {
            None
        } else {
            return Err(Error::Unsupported(format!(
                "{}: bounded decode quality lookup for {}x{} matches multiple tile pairs; an exact selected edge/overlap is required",
                self.provider_id, query.geometry.width, query.geometry.height,
            )));
        };
        let Some(policy) = policy else {
            if had_request_rows && selected_pair.is_some() {
                return Err(Error::Unsupported(format!(
                    "{}: bounded decode parameters {:?} do not match any admitted/refused exact tile pair for {}x{}",
                    self.provider_id,
                    selected_pair,
                    query.geometry.width,
                    query.geometry.height,
                )));
            }
            return Err(Error::Unsupported(format!(
                "{}: bounded decode has no admitted production-latent quality record for \
                 backend={} tier={:?} load_shape={:?} mode={} overlay={:?} geometry={}x{}x{} \
                 frames={} references={} use_pid={} tile_pair={:?}",
                self.provider_id,
                self.backend.backend_id(),
                query.tier,
                query.load_shape,
                query.mode_key,
                query.overlay,
                query.geometry.width,
                query.geometry.height,
                query.geometry.batch,
                query.geometry.frames,
                query.geometry.reference_count,
                query.use_pid,
                selected_pair,
            )));
        };
        if let MemoryDecodeQualityDisposition::Refused { reason } = &policy.disposition {
            return Err(Error::Unsupported(format!(
                "{}: bounded decode was refused by production-latent quality admission for \
                 {}x{}: {reason} (evidence {})",
                self.provider_id,
                query.geometry.width,
                query.geometry.height,
                policy.production_evidence_sha256,
            )));
        }
        Ok(Some(policy))
    }

    pub(super) fn support(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategySupport> {
        self.capability(strategy)
            .map(|capability| &capability.support)
    }

    /// Shared prerequisites followed by additive realization-specific constraints.
    pub fn requires(
        &self,
        strategy: MemoryStrategy,
    ) -> impl Iterator<Item = MemoryStrategyPrerequisite> + '_ {
        strategy.requires().iter().copied().chain(
            self.additional_prerequisites
                .iter()
                .filter_map(move |(owner, prerequisite)| {
                    (*owner == strategy).then_some(*prerequisite)
                }),
        )
    }

    /// Whether `rung`'s mechanism is active in a request that selected `strategy`.
    ///
    /// This is the **cost-order selector policy** ([`MemoryStrategy::engages`]) intersected with
    /// what this provider actually implements, and it is the seam that makes the default defeasible:
    /// a provider which does not declare a cheaper rung `Implemented` simply does not engage it, and
    /// [`MemoryProviderContract::validate_selection`] then stops requiring that rung's parameters
    /// rather than refusing the composition. It is deliberately separate from
    /// [`MemoryStrategy::requires`] — engagement is a default, a prerequisite is a constraint.
    pub fn engages(&self, strategy: MemoryStrategy, rung: MemoryStrategy) -> bool {
        let required_by_realization = self.requires(strategy).any(|prerequisite| {
            matches!(
                prerequisite,
                MemoryStrategyPrerequisite::Rung {
                    rung: required,
                    ..
                } if required == rung
            )
        });
        let excludes_default = self
            .default_engagement_exclusions
            .iter()
            .any(|exclusion| exclusion.selection == strategy && exclusion.excluded_rung == rung);
        ((strategy.engages(rung) && !excludes_default) || required_by_realization)
            && matches!(self.support(rung), Some(MemoryStrategySupport::Implemented))
    }

    /// Request-selection-aware engagement. Geometry-quality adoption makes bounded decode a
    /// conditional cost-order default: the exact tile pair is present only on selections for an
    /// admitted request coordinate. Higher independent rungs therefore remain selectable with
    /// `decode_* = None` on off-grid/refused requests, while an exact admitted pair composes bounded
    /// decode as before. Selecting `BoundedDecode` itself never makes its own mechanism optional.
    pub fn engages_selection(&self, selection: &MemorySelection, rung: MemoryStrategy) -> bool {
        if rung == MemoryStrategy::StagedResidency
            && selection.parameters.stage_residency == Some(true)
        {
            return matches!(self.support(rung), Some(MemoryStrategySupport::Implemented));
        }
        let engaged = self.engages(selection.strategy, rung);
        if !engaged || rung != MemoryStrategy::BoundedDecode {
            return engaged;
        }
        let has_geometry_policy = self.decode_geometry_policy_authoritative
            || self
                .capability(MemoryStrategy::BoundedDecode)
                .is_some_and(|capability| {
                    !capability.parameters.decode_geometry_policies.is_empty()
                });
        if !has_geometry_policy || selection.strategy == MemoryStrategy::BoundedDecode {
            return true;
        }
        selection.parameters.decode_tile_edge.is_some()
            && selection.parameters.decode_overlap.is_some()
    }

    /// Canonical composition for one concrete selection, including conditional bounded-decode
    /// engagement.
    pub fn engaged_composition_for_selection(
        &self,
        selection: &MemorySelection,
    ) -> Vec<MemoryStrategy> {
        MemoryStrategy::ALL
            .into_iter()
            .filter(|rung| self.engages_selection(selection, *rung))
            .collect()
    }

    /// Canonical, ordered set of strategy mechanisms active for one selection.
    ///
    /// Evidence records this exact set at measurement time. Walking [`MemoryStrategy::ALL`] keeps
    /// the identity deterministic while routing every engagement decision through the provider
    /// contract, including realization-specific prerequisite edges.
    pub fn engaged_composition(&self, strategy: MemoryStrategy) -> Vec<MemoryStrategy> {
        MemoryStrategy::ALL
            .into_iter()
            .filter(|rung| self.engages(strategy, *rung))
            .collect()
    }

    /// Canonical translation from an admitted shared selection into the tensor-neutral request
    /// controls every provider executes. Engagement is read only through this contract, including
    /// provider prerequisite edges and verified exclusions; provider implementations must not
    /// duplicate this mapping.
    pub fn generation_memory(
        &self,
        selection: &MemorySelection,
    ) -> Option<crate::GenerationMemory> {
        if selection.strategy == MemoryStrategy::Resident {
            return match self.resident_request_memory {
                ResidentRequestMemory::PreserveLoadDefaults => None,
                ResidentRequestMemory::ExplicitResident => Some(crate::GenerationMemory::default()),
            };
        }

        let parameters = selection.parameters;
        let stage_residency = self.engages_selection(selection, MemoryStrategy::StagedResidency);
        let tile_vae_decode = self.engages_selection(selection, MemoryStrategy::BoundedDecode);
        let chunk_attention = self.engages_selection(selection, MemoryStrategy::BoundedAttention);
        let stream_transformer_blocks =
            self.engages_selection(selection, MemoryStrategy::BoundedTransformerResidency);
        Some(crate::GenerationMemory {
            stage_residency,
            tile_vae_decode,
            chunk_attention,
            stream_transformer_blocks,
            decode_tile_edge: tile_vae_decode
                .then_some(parameters.decode_tile_edge)
                .flatten(),
            decode_overlap: tile_vae_decode
                .then_some(parameters.decode_overlap)
                .flatten(),
            attention_chunk_size: chunk_attention
                .then_some(parameters.attention_chunk_size)
                .flatten(),
            transformer_window_size: stream_transformer_blocks
                .then_some(parameters.transformer_window_size)
                .flatten(),
            transformer_window_component: stream_transformer_blocks
                .then(|| parameters.window_component()),
            ..Default::default()
        })
    }

    /// Deterministic provider-declared selection used by weights-free behavioral conformance.
    pub fn representative_selection(
        &self,
        strategy: MemoryStrategy,
        tier: MemoryNumericTier,
        use_pid: bool,
    ) -> crate::Result<MemorySelection> {
        let mut parameters = MemoryStrategyParameters::default();
        if self.engages(strategy, MemoryStrategy::BoundedDecode) {
            if let Some(routes) = &self.pid_decode_routes {
                let route = if use_pid { &routes.pid } else { &routes.native };
                parameters.decode_tile_edge = route.tile_edges.first().copied();
                parameters.decode_overlap = Some(route.tile_overlap);
            } else if let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) {
                parameters.decode_tile_edge =
                    capability.parameters.decode_tile_edges.first().copied();
                parameters.decode_overlap = capability.parameters.decode_overlaps.first().copied();
            }
        }
        if self.engages(strategy, MemoryStrategy::BoundedAttention) {
            parameters.attention_chunk_size = self
                .capability(MemoryStrategy::BoundedAttention)
                .and_then(|capability| {
                    capability.parameters.attention_chunk_sizes.first().copied()
                });
        }
        if self.engages(strategy, MemoryStrategy::BoundedTransformerResidency) {
            if let Some(capability) = self.capability(MemoryStrategy::BoundedTransformerResidency) {
                parameters.transformer_window_size = capability
                    .parameters
                    .transformer_window_sizes
                    .first()
                    .copied();
                parameters.transformer_window_component = capability
                    .parameters
                    .transformer_window_components
                    .first()
                    .copied();
            }
        }
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        self.validate_selection(&selection)?;
        Ok(selection)
    }

    /// Every selection this provider declares it can execute at `strategy` — the full product of
    /// the parameter domains of every rung the strategy engages (the decode route's tile edges and
    /// overlaps, the attention chunk sizes, the transformer window sizes and components) and, where
    /// the provider implements staged residency but the strategy does not already engage it, the
    /// optional phase eviction (`stage_residency: Some(true)`); each one
    /// [`validate_selection`](Self::validate_selection)-checked; never empty (a strategy with no
    /// declared parameters is its single parameterless selection).
    ///
    /// Ordered QUALITY FIRST, cheapest last: within each lever the most resident / largest value
    /// first (unstaged before staged, the largest decode tile and overlap, the largest attention
    /// chunk and transformer window, the narrowest window scope `Dit` → `TextEncoder` → `Both`),
    /// the tile edge varying fastest, then the overlap, chunk, window size, component and staging.
    /// A capture that refuses at one selection can therefore step to the next cheaper one at the
    /// same size.
    pub fn selection_space(
        &self,
        strategy: MemoryStrategy,
        tier: MemoryNumericTier,
        use_pid: bool,
    ) -> crate::Result<Vec<MemorySelection>> {
        // The representative validates the strategy itself (and its prerequisites) once.
        let representative = self.representative_selection(strategy, tier, use_pid)?;
        let descending = |mut v: Vec<u32>| {
            v.sort_unstable_by(|a, b| b.cmp(a));
            v.dedup();
            v
        };
        let or_none = |v: Vec<u32>| -> Vec<Option<u32>> {
            if v.is_empty() {
                vec![None]
            } else {
                v.into_iter().map(Some).collect()
            }
        };
        let (mut tiles, mut overlaps) = (vec![None], vec![None]);
        if self.engages(strategy, MemoryStrategy::BoundedDecode) {
            if let Some(routes) = &self.pid_decode_routes {
                let route = if use_pid { &routes.pid } else { &routes.native };
                tiles = or_none(descending(route.tile_edges.clone()));
                overlaps = vec![Some(route.tile_overlap)];
            } else if let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) {
                tiles = or_none(descending(capability.parameters.decode_tile_edges.clone()));
                overlaps = or_none(descending(capability.parameters.decode_overlaps.clone()));
            }
        }
        let mut chunks = vec![None];
        if self.engages(strategy, MemoryStrategy::BoundedAttention) {
            if let Some(capability) = self.capability(MemoryStrategy::BoundedAttention) {
                chunks = or_none(descending(
                    capability.parameters.attention_chunk_sizes.clone(),
                ));
            }
        }
        let (mut windows, mut components) = (vec![None], vec![None]);
        if self.engages(strategy, MemoryStrategy::BoundedTransformerResidency) {
            if let Some(capability) = self.capability(MemoryStrategy::BoundedTransformerResidency) {
                windows = or_none(descending(
                    capability.parameters.transformer_window_sizes.clone(),
                ));
                let mut declared = capability.parameters.transformer_window_components.clone();
                declared.sort_unstable();
                declared.dedup();
                if !declared.is_empty() {
                    components = declared.into_iter().map(Some).collect();
                }
            }
        }
        let mut stages = vec![representative.parameters.stage_residency];
        if strategy != MemoryStrategy::Resident
            && representative.parameters.stage_residency.is_none()
            && !self.engages(strategy, MemoryStrategy::StagedResidency)
            && matches!(
                self.support(MemoryStrategy::StagedResidency),
                Some(MemoryStrategySupport::Implemented)
            )
        {
            stages.push(Some(true));
        }
        let mut out = Vec::new();
        for &stage_residency in &stages {
            for &component in &components {
                for &window in &windows {
                    for &chunk in &chunks {
                        for &overlap in &overlaps {
                            for &tile in &tiles {
                                let selection = MemorySelection {
                                    strategy,
                                    parameters: MemoryStrategyParameters {
                                        stage_residency,
                                        decode_tile_edge: tile,
                                        decode_overlap: overlap,
                                        attention_chunk_size: chunk,
                                        transformer_window_size: window,
                                        transformer_window_component: component,
                                    },
                                    tier,
                                };
                                if self.validate_selection(&selection).is_ok() {
                                    out.push(selection);
                                }
                            }
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            out.push(representative);
        }
        Ok(out)
    }

    /// What one rung-4 window materialization does for this provider (SC-16090), or `None` if the
    /// realization has not stated it.
    ///
    /// Layered on [`MemoryBackendRealization::window_materialization`] the way
    /// [`MemoryProviderContract::engages`] is layered on [`MemoryStrategy::engages`]: the contract is
    /// where a caller asks, so a later per-provider override has a seam to land in without every caller
    /// changing. It currently forwards unchanged — the fact is already per provider, since the
    /// realization is declared per provider rather than per backend.
    ///
    /// **Not `Option<bool>`**: a caller that only wants "is this conforming" still has to distinguish
    /// *declared non-conforming* from *never declared*, because the two have different fixes.
    pub fn window_materialization(&self) -> Option<&MemoryWindowMaterialization> {
        self.backend.window_materialization()
    }

    /// Static conformance errors. An empty result means the provider declaration is internally
    /// coherent; it does not mean measurements are verified.
    pub fn conformance_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.provider_id.trim().is_empty() {
            errors.push("provider_id must be non-empty".to_owned());
        }

        for strategy in MemoryStrategy::ALL {
            let count = self
                .strategies
                .iter()
                .filter(|capability| capability.strategy == strategy)
                .count();
            if count != 1 {
                errors.push(format!(
                    "strategy {strategy:?} must appear exactly once (found {count})"
                ));
            }
        }
        if self.strategies.len() != MemoryStrategy::ALL.len() {
            errors.push(format!(
                "strategy table must contain exactly five entries (found {})",
                self.strategies.len()
            ));
        }

        let implemented = |strategy| {
            matches!(
                self.capability(strategy).map(|c| &c.support),
                Some(MemoryStrategySupport::Implemented)
            )
        };
        let mut exclusions = std::collections::HashSet::new();
        for exclusion in &self.default_engagement_exclusions {
            if !exclusions.insert((exclusion.selection, exclusion.excluded_rung)) {
                errors.push(format!(
                    "duplicate default engagement exclusion {:?} -> {:?}",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if exclusion.evidence.trim().is_empty() {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} has no verification evidence",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if !exclusion.selection.engages(exclusion.excluded_rung) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} does not remove a shared cost-order edge",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if exclusion.selection == exclusion.excluded_rung
                || exclusion.excluded_rung == MemoryStrategy::Resident
            {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} cannot remove the selected strategy or resident baseline",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if !implemented(exclusion.selection) || !implemented(exclusion.excluded_rung) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} requires both strategies to be implemented",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if self.requires(exclusion.selection).any(|prerequisite| {
                matches!(
                    prerequisite,
                    MemoryStrategyPrerequisite::Rung { rung, .. }
                        if rung == exclusion.excluded_rung
                )
            }) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} cannot remove a prerequisite edge",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
        }
        if implemented(MemoryStrategy::StagedResidency) {
            for phase in [
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ] {
                if !self.lifecycle.phases.contains(&phase) {
                    errors.push(format!(
                        "StagedResidency requires a {phase:?} lifecycle phase"
                    ));
                }
            }
            if !self.lifecycle.synchronized_phase_release {
                errors.push(
                    "StagedResidency requires synchronized release of completed phases".to_owned(),
                );
            }
        }
        if implemented(MemoryStrategy::BoundedDecode) && !self.lifecycle.decode_tiling {
            errors.push("BoundedDecode requires the decode_tiling hook".to_owned());
        }
        if let Some(routes) = &self.pid_decode_routes {
            validate_pid_decode_routes(self, routes, &mut errors);
        }
        validate_decode_geometry_policies(self, &mut errors);
        if implemented(MemoryStrategy::BoundedAttention) && !self.lifecycle.attention_chunking {
            errors.push("BoundedAttention requires the attention_chunking hook".to_owned());
        }
        if implemented(MemoryStrategy::BoundedTransformerResidency) {
            if !self.lifecycle.transformer_window_materialization {
                errors.push(
                    "BoundedTransformerResidency requires the transformer-window hook".to_owned(),
                );
            }
            // SC-16090. What is refused here is an UNDECLARED window realization, never a slow one.
            //
            // A provider that honestly declares `HostFormatConversion` and names its owner story has no
            // error and keeps every rung, including rung 4 — deliberately, because the selector reaches
            // rung 4 only when nothing cheaper fits, so refusing it would trade a slow render for no
            // render. Cost is not what this rule reads.
            //
            // Be clear about the blast radius, because it is wider than rung 4: like every other rule
            // in this function, an error here is CONTRACT-level, and a caller that bails on a non-empty
            // `conformance_errors` (the shared selector does) loses rungs 0-4, not just rung 4. That is
            // the right severity for the case that produces it — a declaration this contract cannot
            // interpret — and the wrong severity for a cost verdict, which is a second reason this rule
            // does not attempt one.
            match self.backend.window_materialization() {
                Some(MemoryWindowMaterialization::HostFormatConversion {
                    converts,
                    owner_story,
                }) => {
                    // `owner_story` is the promise; `converts` is the pointer a reviewer follows to the
                    // code. Both are load-bearing, so both are required to be present. Neither can be
                    // checked for TRUTH here — a wrong pointer or a closed story still passes.
                    if owner_story.trim().is_empty() {
                        errors.push(format!(
                            "BoundedTransformerResidency on a HostFormatConversion realization \
                             ({converts}) must name the owner_story that removes it (SC-16090)"
                        ));
                    }
                    if converts.trim().is_empty() {
                        errors.push(
                            "BoundedTransformerResidency on a HostFormatConversion realization must \
                             say what it converts, so a reviewer can find it (SC-16090)"
                                .to_owned(),
                        );
                    }
                }
                Some(MemoryWindowMaterialization::DeviceFormatTransfer) => {}
                // Unstated. The actionable fix is naming the mechanism, not naming a story — see
                // `MemoryBackendRealization::window_materialization`.
                None => errors.push(
                    "BoundedTransformerResidency requires the realization to state what a window \
                     materialization does; an MLX realization does so by declaring \
                     lazy_or_mmap_materialization (SC-16090)"
                        .to_owned(),
                ),
            }
        }

        for capability in &self.strategies {
            if let MemoryStrategySupport::StructurallyNotApplicable { reason } = &capability.support
            {
                if reason.trim().is_empty() {
                    errors.push(format!(
                        "{:?} is StructurallyNotApplicable without a reason",
                        capability.strategy
                    ));
                }
                let has_implementation_hook = match capability.strategy {
                    MemoryStrategy::Resident => false,
                    MemoryStrategy::StagedResidency => {
                        !self.lifecycle.phases.is_empty()
                            || self.lifecycle.synchronized_phase_release
                    }
                    MemoryStrategy::BoundedDecode => self.lifecycle.decode_tiling,
                    MemoryStrategy::BoundedAttention => self.lifecycle.attention_chunking,
                    MemoryStrategy::BoundedTransformerResidency => {
                        self.lifecycle.transformer_window_materialization
                    }
                };
                if has_implementation_hook {
                    errors.push(format!(
                        "{:?} cannot be StructurallyNotApplicable while its implementation hook is declared",
                        capability.strategy
                    ));
                }
            }
            validate_ranges(capability, &mut errors);
            validate_owned_parameter_domain(capability, &mut errors);
        }

        let mut component_ids = std::collections::HashSet::new();
        for component in self.formula.resident_components() {
            if component.id.trim().is_empty() {
                errors.push("resident component id must be non-empty".to_owned());
            } else if !component_ids.insert(component.id.as_str()) {
                errors.push(format!(
                    "resident component id {:?} must be unique",
                    component.id
                ));
            }
            if component.resident_bytes == 0 {
                errors.push(format!(
                    "resident component {:?} must declare non-zero bytes",
                    component.id
                ));
            }
            if let Some(strategy) = component.bounded_by {
                if !implemented(strategy) {
                    errors.push(format!(
                        "resident component {:?} is bounded by {strategy:?}, but that strategy is not implemented",
                        component.id
                    ));
                }
            }
            // SC-18665. A sub-stack is part of the transformer total, so it cannot exceed it. Only
            // checked against a RESOLVED footprint: a weights-free fixture legitimately declares an
            // architecture-derived sub-stack against zero asset facts.
            if matches!(component.kind, MemoryComponentKind::TransformerSubStack(_))
                && self.asset_facts.transformer_bytes > 0
                && component.resident_bytes > self.asset_facts.transformer_bytes
            {
                errors.push(format!(
                    "resident component {:?} declares {} B inside a transformer charged at {} B; a \
                     sub-stack cannot exceed the stack that contains it",
                    component.id, component.resident_bytes, self.asset_facts.transformer_bytes
                ));
            }
            if let MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in,
                retained_bytes,
                evidence,
            } = &component.residency
            {
                if *retained_bytes >= component.resident_bytes {
                    errors.push(format!(
                        "resident component {:?} retains {retained_bytes} B of its {} B, so its \
                         declared eviction excludes nothing",
                        component.id, component.resident_bytes
                    ));
                }
                if evidence.trim().is_empty() {
                    errors.push(format!(
                        "resident component {:?} declares an eviction with no verification evidence",
                        component.id
                    ));
                }
                if !self.lifecycle.phases.contains(precomputed_in) {
                    errors.push(format!(
                        "resident component {:?} is precomputed in {precomputed_in:?}, which is not \
                         a declared lifecycle phase",
                        component.id
                    ));
                }
            }
        }
        let auxiliary_bytes = self.auxiliary_resident_bytes();
        match self
            .asset_facts
            .conditioning_bytes
            .checked_add(self.asset_facts.transformer_bytes)
            .and_then(|bytes| bytes.checked_add(self.asset_facts.decoder_bytes))
        {
            Some(base_components) if base_components != self.asset_facts.base_bytes => {
                errors.push(format!(
                    "base_bytes {} must equal base component bytes {} and exclude overlay_bytes",
                    self.asset_facts.base_bytes, base_components
                ));
            }
            None => errors.push("base component byte sum overflow".to_owned()),
            _ => {}
        }
        if self.asset_facts.overlay_bytes > 0
            && self.formula.uses(MemoryFormulaVariable::AssetBytes)
            && self.formula.uses(MemoryFormulaVariable::OverlayBytes)
            && auxiliary_bytes == 0
        {
            errors.push(
                "a non-zero overlay declared with AssetBytes and OverlayBytes must use typed auxiliary resident components"
                    .to_owned(),
            );
        }
        if auxiliary_bytes > 0 {
            if auxiliary_bytes != self.asset_facts.overlay_bytes {
                errors.push(format!(
                    "overlay_bytes {} must equal declared auxiliary component bytes {}",
                    self.asset_facts.overlay_bytes, auxiliary_bytes
                ));
            }
            if !self.formula.uses(MemoryFormulaVariable::OverlayBytes) {
                errors.push(
                    "a provider with auxiliary resident components must include OverlayBytes in its formula"
                        .to_owned(),
                );
            }
        }

        if let Some(calibration) = &self.calibration {
            if calibration.abi != MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "calibration ABI {} does not match contract ABI {}",
                    calibration.abi, MEMORY_CALIBRATION_ABI
                ));
            }
            if let Err(reason) = validate_calibration_fingerprint(&calibration.fingerprint) {
                errors.push(format!("calibration fingerprint {reason}"));
            }
            if calibration.load_shape != self.load_shape {
                errors.push(format!(
                    "calibration load shape {:?} does not match contract load shape {:?}",
                    calibration.load_shape, self.load_shape
                ));
            }
        }

        if self.backend.backend_id().is_empty() {
            errors.push("backend realization must have an id".to_owned());
        }
        errors
    }

    /// Validate a worker-owned selection against static provider capability and parameter ranges.
    pub fn validate_selection(&self, selection: &MemorySelection) -> Result<()> {
        if selection.strategy == MemoryStrategy::Resident
            && selection.parameters.stage_residency.is_some()
        {
            return Err(Error::Unsupported(
                "a resident selection cannot request phase eviction".to_owned(),
            ));
        }
        if selection.parameters.stage_residency == Some(false) {
            return Err(Error::Unsupported(
                "explicit staging can enable phase eviction, not disable a prerequisite".to_owned(),
            ));
        }
        if selection.parameters.stage_residency == Some(true)
            && !matches!(
                self.support(MemoryStrategy::StagedResidency),
                Some(MemoryStrategySupport::Implemented)
            )
        {
            return Err(Error::Unsupported(format!(
                "{} does not implement staged residency",
                self.provider_id
            )));
        }
        let capability = self.capability(selection.strategy).ok_or_else(|| {
            Error::Unsupported(format!(
                "{} does not declare {:?}",
                self.provider_id, selection.strategy
            ))
        })?;
        if !matches!(capability.support, MemoryStrategySupport::Implemented) {
            return Err(Error::Unsupported(format!(
                "{} cannot execute {:?}: {:?}",
                self.provider_id, selection.strategy, capability.support
            )));
        }
        // SC-15805: walk the DECLARED prerequisite graph, never `< selection.strategy`. The numeric
        // order is a cost ordering; reading it as a dependency refused compositions that are
        // perfectly correct (rungs 2 and 3 bound scratch and depend on nothing).
        for prerequisite in self.requires(selection.strategy) {
            match prerequisite {
                MemoryStrategyPrerequisite::Rung { rung, scope } => match scope {
                    MemoryPrerequisiteScope::EngagedInSameRequest => {
                        if self.engages_selection(selection, rung) {
                            continue;
                        }
                        // `StructurallyNotApplicable` satisfies the edge vacuously: it asserts the
                        // architecture has no such component to shed.
                        if matches!(
                            self.support(rung),
                            Some(MemoryStrategySupport::StructurallyNotApplicable { .. })
                        ) {
                            continue;
                        }
                        let (why, caveat) = match self.support(rung) {
                            Some(MemoryStrategySupport::Implemented) => (
                                "the selected composition does not engage it".to_owned(),
                                format!(
                                    "This prerequisite is engagement, not availability — rung \
                                     {rung:?} being implemented, or engaged by some other request, \
                                     does not satisfy it."
                                ),
                            ),
                            support => (
                                match support {
                                    Some(support) => {
                                        format!("the provider declares it {support:?}")
                                    }
                                    None => {
                                        "the provider does not declare it at all".to_owned()
                                    }
                                },
                                format!(
                                    "Declaring rung {rung:?} Implemented — or \
                                     StructurallyNotApplicable, if this architecture has no such \
                                     component to stage — is what makes {:?} selectable. Engagement \
                                     is per request.",
                                    selection.strategy
                                ),
                            ),
                        };
                        return Err(Error::Unsupported(format!(
                            "{} cannot execute {:?}: it requires rung {rung:?} to be ENGAGED IN THE \
                             SAME REQUEST, and {why}. {caveat}",
                            self.provider_id, selection.strategy
                        )));
                    }
                },
                MemoryStrategyPrerequisite::LoadShape(required) => {
                    if self.load_shape == required {
                        continue;
                    }
                    return Err(Error::Unsupported(format!(
                        "{} cannot execute {:?}: it requires load shape {required:?}, but this \
                         generator was loaded with {:?}",
                        self.provider_id, selection.strategy, self.load_shape
                    )));
                }
            }
        }
        validate_selected_parameters(selection, self)
            .map_err(|message| Error::Unsupported(format!("{}: {message}", self.provider_id)))
    }
}

fn validate_pid_decode_routes(
    contract: &MemoryProviderContract,
    routes: &MemoryPidDecodeRoutes,
    errors: &mut Vec<String>,
) {
    let Some(bounded_decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        errors.push("PiD decode routes require a BoundedDecode capability".to_owned());
        return;
    };
    if !matches!(bounded_decode.support, MemoryStrategySupport::Implemented) {
        errors.push("PiD decode routes require BoundedDecode to be Implemented".to_owned());
    }

    for (label, domain) in [("native", &routes.native), ("PiD", &routes.pid)] {
        if domain.tile_edges.is_empty() {
            errors.push(format!(
                "{label} decode route must declare at least one tile edge"
            ));
        }
        if domain.tile_edges.contains(&0) {
            errors.push(format!("{label} decode route contains a zero tile edge"));
        }
        let mut unique = domain.tile_edges.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != domain.tile_edges.len() {
            errors.push(format!(
                "{label} decode route contains duplicate tile edges"
            ));
        }
        if domain.tile_overlap == 0 {
            errors.push(format!(
                "{label} decode route must declare a non-zero overlap"
            ));
        }
    }

    if routes.native.tile_overlap == routes.pid.tile_overlap
        && routes
            .native
            .tile_edges
            .iter()
            .any(|edge| routes.pid.tile_edges.contains(edge))
    {
        errors.push("native and PiD decode route geometries must be disjoint".to_owned());
    }

    let mut declared_edges = bounded_decode.parameters.decode_tile_edges.clone();
    declared_edges.sort_unstable();
    declared_edges.dedup();
    let mut route_edges = routes
        .native
        .tile_edges
        .iter()
        .chain(&routes.pid.tile_edges)
        .copied()
        .collect::<Vec<_>>();
    route_edges.sort_unstable();
    route_edges.dedup();
    if declared_edges != route_edges {
        errors.push(format!(
            "BoundedDecode tile edges {declared_edges:?} must equal the native/PiD route union \
             {route_edges:?}"
        ));
    }

    let mut declared_overlaps = bounded_decode.parameters.decode_overlaps.clone();
    declared_overlaps.sort_unstable();
    declared_overlaps.dedup();
    let mut route_overlaps = vec![routes.native.tile_overlap, routes.pid.tile_overlap];
    route_overlaps.sort_unstable();
    route_overlaps.dedup();
    if declared_overlaps != route_overlaps {
        errors.push(format!(
            "BoundedDecode overlaps {declared_overlaps:?} must equal the native/PiD route union \
             {route_overlaps:?}"
        ));
    }
}

fn validate_decode_geometry_policies(contract: &MemoryProviderContract, errors: &mut Vec<String>) {
    let Some(bounded_decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        return;
    };
    let policies = &bounded_decode.parameters.decode_geometry_policies;
    if policies.is_empty() {
        return;
    }

    let implemented = matches!(bounded_decode.support, MemoryStrategySupport::Implemented);
    let mut coordinates = std::collections::HashSet::new();
    let mut admitted_edges = Vec::new();
    let mut admitted_overlaps = Vec::new();
    let mut admitted_count = 0usize;
    let mut family = None;
    let mut resolved_route = None;
    let mut artifact = None;
    for policy in policies {
        if policy.quality_abi != MEMORY_DECODE_QUALITY_ABI {
            errors.push(format!(
                "decode quality ABI {} does not match {}",
                policy.quality_abi, MEMORY_DECODE_QUALITY_ABI
            ));
        }
        if policy.family.trim().is_empty() {
            errors.push("decode quality family must be non-empty".to_owned());
        }
        if let Some(expected) = family {
            if policy.family != expected {
                errors.push(format!(
                    "one loaded provider contract cannot mix decode quality families {:?} and {:?}",
                    expected, policy.family
                ));
            }
        } else {
            family = Some(policy.family.as_str());
        }
        if policy.resolved_route.trim().is_empty() {
            errors.push("decode quality resolved route must be non-empty".to_owned());
        }
        if let Some(expected) = resolved_route {
            if policy.resolved_route != expected {
                errors.push(format!(
                    "one loaded provider contract cannot mix decode quality routes {:?} and {:?}",
                    expected, policy.resolved_route
                ));
            }
        } else {
            resolved_route = Some(policy.resolved_route.as_str());
        }
        if policy.backend != contract.backend.backend_kind() {
            errors.push(format!(
                "decode quality backend {:?} does not match provider backend {:?}",
                policy.backend,
                contract.backend.backend_kind()
            ));
        }
        if policy.load_shape != contract.load_shape {
            errors.push(format!(
                "decode quality load shape {:?} does not match provider load shape {:?}",
                policy.load_shape, contract.load_shape
            ));
        }
        if policy.artifact.repository.trim().is_empty()
            || policy.artifact.variant.trim().is_empty()
            || policy.artifact.fingerprint.trim().is_empty()
            || !is_lower_hex_sha1(&policy.artifact.revision)
        {
            errors.push(
                "decode quality artifact requires repository, exact 40-hex revision, variant, and fingerprint"
                    .to_owned(),
            );
        }
        if let Some(expected) = artifact {
            if &policy.artifact != expected {
                errors.push(
                    "one loaded provider contract cannot mix decode quality artifact identities"
                        .to_owned(),
                );
            }
        } else {
            artifact = Some(&policy.artifact);
        }
        if !is_lower_hex_sha256(&policy.implementation_fingerprint) {
            errors.push(
                "decode quality implementation fingerprint must be a lowercase SHA-256 stamp"
                    .to_owned(),
            );
        }
        if policy.mode.as_key().trim().is_empty() {
            errors.push("decode quality mode must be non-empty".to_owned());
        }
        if policy
            .overlay
            .as_deref()
            .is_some_and(|overlay| overlay.trim().is_empty() || overlay.contains('='))
        {
            errors.push("decode quality overlay must be a non-empty identity axis".to_owned());
        }
        if policy.geometry.width == 0
            || policy.geometry.height == 0
            || policy.geometry.batch == 0
            || policy.geometry.frames == 0
        {
            errors.push("decode quality geometry dimensions must be non-zero".to_owned());
        }
        if policy.use_pid && contract.pid_decode_routes.is_none() {
            errors.push(
                "decode quality cannot admit a PiD route when the provider declares no PiD decode geometry"
                    .to_owned(),
            );
        }
        if policy.tile_edge == 0 || policy.overlap == 0 || policy.overlap >= policy.tile_edge {
            errors.push(format!(
                "decode quality tile edge {} must exceed non-zero overlap {}",
                policy.tile_edge, policy.overlap
            ));
        }
        if policy.metric.trim().is_empty() {
            errors.push("decode quality metric must be non-empty".to_owned());
        }
        if !is_lower_hex_sha256(&policy.production_evidence_sha256) {
            errors.push(
                "decode production evidence SHA-256 must be 64 lowercase hexadecimal characters"
                    .to_owned(),
            );
        } else if policy.production_evidence_sha256 != policy.computed_evidence_sha256() {
            errors.push(format!(
                "decode production evidence SHA-256 does not bind the canonical {}x{} quality record",
                policy.geometry.width, policy.geometry.height
            ));
        }
        if policy.fixtures.len() < 2 {
            errors.push(format!(
                "decode quality coordinate {}x{} requires multiple fixed-seed production latents",
                policy.geometry.width, policy.geometry.height
            ));
        }
        let mut seeds = std::collections::HashSet::new();
        for fixture in &policy.fixtures {
            if !seeds.insert(fixture.seed) {
                errors.push(format!(
                    "decode quality coordinate {}x{} contains duplicate seed {}",
                    policy.geometry.width, policy.geometry.height, fixture.seed
                ));
            }
            if fixture.production_latent_provenance.trim().is_empty() {
                errors.push(format!(
                    "decode quality seed {} has no production-latent provenance",
                    fixture.seed
                ));
            }
            for (label, sha256) in [
                (
                    "production latent",
                    fixture.production_latent_sha256.as_str(),
                ),
                ("dense output", fixture.dense_output_sha256.as_str()),
                ("tiled output", fixture.tiled_output_sha256.as_str()),
            ] {
                if !is_lower_hex_sha256(sha256) {
                    errors.push(format!(
                        "decode quality seed {} {label} SHA-256 must be 64 lowercase hexadecimal characters",
                        fixture.seed
                    ));
                }
            }
        }

        let coordinate = format!(
            "{}|{}|{}|{:?}|{:?}|{}|{:?}|{}x{}x{}x{}r{}|pid={}|tile={}+{}",
            policy.family,
            policy.resolved_route,
            policy.backend.as_key(),
            policy.tier,
            policy.load_shape,
            policy.mode.as_key(),
            policy.overlay,
            policy.geometry.width,
            policy.geometry.height,
            policy.geometry.batch,
            policy.geometry.frames,
            policy.geometry.reference_count,
            policy.use_pid,
            policy.tile_edge,
            policy.overlap,
        );
        if !coordinates.insert(coordinate) {
            errors.push(format!(
                "duplicate decode quality coordinate for {}x{}",
                policy.geometry.width, policy.geometry.height
            ));
        }

        match &policy.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                admitted_count += 1;
                admitted_edges.push(policy.tile_edge);
                admitted_overlaps.push(policy.overlap);
                if !implemented {
                    errors.push(
                        "an admitted decode quality row requires BoundedDecode Implemented"
                            .to_owned(),
                    );
                }
                if policy
                    .fixtures
                    .iter()
                    .any(|fixture| fixture.observed_error > policy.maximum_error)
                {
                    errors.push(format!(
                        "admitted decode quality coordinate {}x{} exceeds its {} maximum",
                        policy.geometry.width, policy.geometry.height, policy.maximum_error
                    ));
                }
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                if reason.trim().is_empty() {
                    errors.push("refused decode quality row requires a reason".to_owned());
                }
                if policy
                    .fixtures
                    .iter()
                    .all(|fixture| fixture.observed_error <= policy.maximum_error)
                {
                    errors.push(format!(
                        "refused decode quality coordinate {}x{} has no failing fixture",
                        policy.geometry.width, policy.geometry.height
                    ));
                }
            }
        }
    }

    if implemented && admitted_count == 0 {
        errors.push(
            "geometry-aware BoundedDecode requires at least one admitted quality row".to_owned(),
        );
    }
    admitted_edges.sort_unstable();
    admitted_edges.dedup();
    admitted_overlaps.sort_unstable();
    admitted_overlaps.dedup();
    let mut declared_edges = bounded_decode.parameters.decode_tile_edges.clone();
    declared_edges.sort_unstable();
    declared_edges.dedup();
    let mut declared_overlaps = bounded_decode.parameters.decode_overlaps.clone();
    declared_overlaps.sort_unstable();
    declared_overlaps.dedup();
    if implemented && admitted_edges != declared_edges {
        errors.push(format!(
            "BoundedDecode tile edges {declared_edges:?} must equal admitted geometry-policy union {admitted_edges:?}"
        ));
    }
    if implemented && admitted_overlaps != declared_overlaps {
        errors.push(format!(
            "BoundedDecode overlaps {declared_overlaps:?} must equal admitted geometry-policy union {admitted_overlaps:?}"
        ));
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_hex_sha1(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
