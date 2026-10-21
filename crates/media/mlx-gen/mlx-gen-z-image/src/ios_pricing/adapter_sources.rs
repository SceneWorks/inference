//! Z-only consumption of the unfused source-case channel. No legacy semantic-v1 witness is used
//! as a native allocation/lifetime certificate. The public guard remains until this ledger and
//! native producer correspondence are independently qualified.

use std::collections::{BTreeMap, BTreeSet};

use mlx_gen::adapters::facts::{linear_header_facts, AdapterFacts, TensorFacts};
use mlx_gen::adapters::loader::source_facts::{
    visit_configured_adapter_source_cases, AdapterDeclaredOrigin, AdapterFileEpoch,
    AdapterSourceCase, AdapterSourceFailure, AdapterSourceKey, AdapterSourceOutcome,
    AdapterSourcePrefix, AdapterSourceSummary, AdapterSourceWork,
};
use mlx_gen::adapters::{AdaptableHost, Adapter, LokrFactors};
use mlx_gen::gen_core::{LoadSpec, WeightsSource};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosWindowPrice};
use mlx_gen::ios_quantum::Backend;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use super::{carrier, price_source_quanta, Facts, Model, Priced, Route};

/// Rich PRIVATE failure preserves committed prefix prices and CURRENT source provenance before
/// an outward legacy Error-only compatibility mapping. Prefix is NEVER replayed as another case.
#[derive(Debug)]
pub(super) struct AdapterPricingFailure {
    pub error: Error,
    pub source_prefix: AdapterSourcePrefix,
    pub attempted_case_id: Option<u64>,
    pub committed: Vec<IosWindowPrice>,
}

pub(super) struct AdapterPricingReport {
    pub pricing: IosPricing,
    pub source_summary: AdapterSourceSummary,
    pub refusal_prefixes: Vec<(u64, Vec<IosWindowPrice>)>,
}

impl AdapterPricingFailure {
    /// The legacy Error-only boundary consumes actual abort provenance through the existing
    /// metadata listener BEFORE returning the original error. No tensor payloads, user text,
    /// replay, charging, materialization or admission decision are introduced here.
    fn into_legacy_error(self) -> Error {
        self.into_legacy_error_with(|kind, fields| {
            mlx_gen::ios_quantum::diagnostic_metadata("z_image", kind, fields);
        })
    }

    fn into_legacy_error_with(
        self,
        mut emit: impl FnMut(&'static str, &[(&'static str, u64)]),
    ) -> Error {
        emit(
            "adapter-source-pricing-abort",
            &[
                ("schema", 1),
                ("case_present", u64::from(self.attempted_case_id.is_some())),
                ("case_id", self.attempted_case_id.unwrap_or(0)),
                ("source_files", self.source_prefix.files.len() as u64),
                ("source_work", self.source_prefix.work.len() as u64),
                ("retained_slots", self.source_prefix.retained.len() as u64),
                ("applied", self.source_prefix.counts.applied as u64),
                (
                    "unmatched",
                    self.source_prefix.counts.unmatched.len() as u64,
                ),
                ("committed_windows", self.committed.len() as u64),
            ],
        );
        for (ordinal, file) in self.source_prefix.files.iter().enumerate() {
            emit(
                "adapter-source-pricing-abort-file",
                &[
                    ("ordinal", ordinal as u64),
                    ("epoch", file.epoch.0 as u64),
                    ("header_keys", file.headers.len() as u64),
                ],
            );
        }
        for (ordinal, work) in self.source_prefix.work.iter().enumerate() {
            // Schema1 tags distinguish ATTEMPT from COMPLETE; zero-valued unrelated fields
            // never erase the ordered source event or manufacture a successful allocation.
            let (kind, epoch, items, attempt, prepared, incomplete) = match work {
                AdapterSourceWork::FileMapAttempt(epoch) => (0, *epoch, 0, 0, 0, 0),
                AdapterSourceWork::FileMap(epoch) => (1, *epoch, 0, 0, 0, 0),
                AdapterSourceWork::Parse(family) => {
                    (2, family.epoch, family.actions.len(), 0, 0, 0)
                }
                AdapterSourceWork::Constructor(work) => (
                    3,
                    work.epoch,
                    work.operands.len(),
                    work.attempt,
                    work.prepared.len(),
                    work.operations.iter().filter(|op| !op.completed).count(),
                ),
                AdapterSourceWork::BudgetProbe { epoch, .. } => (4, *epoch, 0, 0, 0, 0),
                AdapterSourceWork::Install {
                    epoch,
                    retained_index,
                    constructor_attempt,
                    ..
                } => (5, *epoch, *retained_index, *constructor_attempt, 0, 0),
                AdapterSourceWork::MaterializeAccessedAttempt { epoch, keys } => {
                    (6, *epoch, keys.len(), 0, 0, 0)
                }
                AdapterSourceWork::MaterializeAccessedComplete(epoch) => (7, *epoch, 0, 0, 0, 0),
            };
            emit(
                "adapter-source-pricing-abort-work",
                &[
                    ("ordinal", ordinal as u64),
                    ("work_kind", kind),
                    ("epoch", epoch.0 as u64),
                    ("items", items as u64),
                    ("constructor_attempt", attempt as u64),
                    ("prepared", prepared as u64),
                    ("incomplete_operations", incomplete as u64),
                ],
            );
        }
        for (ordinal, window) in self.committed.iter().enumerate() {
            let phase = match window.phase {
                IosPhase::Load => 0,
                IosPhase::Conditioning => 1,
                IosPhase::Encode => 2,
                IosPhase::Denoise => 3,
                IosPhase::Decode => 4,
            };
            emit(
                "adapter-source-pricing-abort-committed-window",
                &[
                    ("ordinal", ordinal as u64),
                    ("phase", phase),
                    ("resident_bytes", window.resident_bytes),
                    ("held_bytes", window.held_bytes),
                    ("scratch_bytes", window.scratch_bytes),
                ],
            );
        }
        self.error // SAME owned typed kind/code/cause; never a formatted Msg wrapper.
    }
}

/// The actual visitor callback's commit state. A whole case is priced locally FIRST; an Err
/// preserves every prior commit and never commits/replays the current prefix as a second case.
#[derive(Default)]
struct AdapterCasePrices {
    committed: BTreeMap<u64, Vec<IosWindowPrice>>,
    refusal_prefixes: Vec<(u64, Vec<IosWindowPrice>)>,
    pricing: IosPricing,
    last_refused_prefix: Option<AdapterSourcePrefix>,
}

impl AdapterCasePrices {
    fn consume(
        &mut self,
        case: &AdapterSourceCase,
        price: impl FnOnce() -> Result<IosPricing>,
    ) -> Result<()> {
        if self.committed.contains_key(&case.case_id) {
            return Err(Error::Msg("adapter source case delivered twice".into()));
        }
        let local = price()?; // ALL fallible work precedes the single case-ID commit.
        let windows = local.windows;
        if matches!(case.outcome, AdapterSourceOutcome::Refused(_)) {
            self.refusal_prefixes.push((case.case_id, windows.clone()));
            self.last_refused_prefix = Some(case.prefix.clone());
        }
        self.committed.insert(case.case_id, windows.clone());
        for window in windows {
            self.pricing.take_window(window);
        }
        self.pricing.absent_phases.extend(local.absent_phases);
        Ok(())
    }

    fn finish(
        self,
        provider: &str,
        visited: std::result::Result<AdapterSourceSummary, Box<AdapterSourceFailure>>,
    ) -> std::result::Result<AdapterPricingReport, AdapterPricingFailure> {
        match visited {
            Ok(summary) if summary.enumeration_complete && summary.complete_cases != 0 => {
                Ok(AdapterPricingReport {
                    pricing: self.pricing,
                    source_summary: summary,
                    refusal_prefixes: self.refusal_prefixes,
                })
            }
            Ok(_) => Err(AdapterPricingFailure {
                error: Error::Unsupported(format!(
                    "{provider}: every native adapter source case refuses"
                )),
                source_prefix: self.last_refused_prefix.unwrap_or_default(),
                attempted_case_id: None,
                committed: self.committed.into_values().flatten().collect(),
            }),
            Err(failure) => {
                let AdapterSourceFailure {
                    error,
                    prefix,
                    attempted_case_id,
                    ..
                } = *failure;
                Err(AdapterPricingFailure {
                    error,
                    source_prefix: prefix,
                    attempted_case_id,
                    committed: self.committed.into_values().flatten().collect(),
                })
            }
        }
    }
}

fn cap(facts: &TensorFacts) -> u64 {
    mlx_gen::ios_quantum::allocated_bytes(facts.bytes())
}

#[derive(Default)]
pub(super) struct Ledger {
    raw: BTreeMap<AdapterSourceKey, u64>,
    /// Distinct alpha READ/CAST attempts and constructor scalars, never deduped by source KEY.
    eager: BTreeMap<(AdapterFileEpoch, usize, &'static str), u64>,
    constructors: BTreeMap<(AdapterFileEpoch, usize), (u64, u64)>,
    slots: BTreeMap<usize, (AdapterFileEpoch, usize)>,
    file_roots: BTreeMap<AdapterFileEpoch, u64>,
    // Raw buffers/parents are conservatively retained here. Descriptor scopes/cache/queued work
    // do not authorize native Data/process release credits. This is source-capacity union only.
}

impl Ledger {
    fn raw_key(
        &mut self,
        prefix: &AdapterSourcePrefix,
        epoch: AdapterFileEpoch,
        key: &str,
    ) -> Result<()> {
        let file = prefix
            .files
            .iter()
            .find(|f| f.epoch == epoch)
            .ok_or_else(|| Error::Msg("adapter source epoch missing".into()))?;
        let facts = file
            .headers
            .get(key)
            .ok_or_else(|| Error::Msg(format!("adapter source key missing: {key}")))?;
        self.raw
            .entry(AdapterSourceKey {
                epoch,
                key: key.into(),
            })
            .or_insert_with(|| cap(facts));
        Ok(())
    }

    pub(super) fn held(&self) -> Result<u64> {
        self.raw
            .values()
            .chain(self.eager.values())
            .try_fold(0u64, |sum, &n| {
                sum.checked_add(n)
                    .ok_or_else(|| Error::Msg("adapter source root union overflow".into()))
            })
    }

    pub(super) fn apply(&mut self, prefix: &AdapterSourcePrefix) -> Result<()> {
        for event in &prefix.work {
            match event {
                AdapterSourceWork::FileMapAttempt(_)
                | AdapterSourceWork::FileMap(_)
                | AdapterSourceWork::BudgetProbe { .. }
                | AdapterSourceWork::MaterializeAccessedComplete(_) => {}
                AdapterSourceWork::Parse(family) => {
                    let file = prefix
                        .files
                        .iter()
                        .find(|f| f.epoch == family.epoch)
                        .ok_or_else(|| Error::Msg("parse source epoch missing".into()))?;
                    for action in family.checked_alpha_attempt_upper()? {
                        self.raw_key(prefix, family.epoch, &action.key)?;
                        let raw = &file.headers[&action.key];
                        if raw.dtype != Dtype::Float32 {
                            let cast = TensorFacts {
                                shape: raw.shape.clone(),
                                dtype: Dtype::Float32,
                                strides: Vec::new(),
                            };
                            // FULL tensor shape, not <=8B semantic-control read. Per-action ID,
                            // not per-key dedup, preserves distinct repeated native work attempts.
                            self.eager
                                .entry((family.epoch, action.id.ordinal, "alpha-F32-attempt"))
                                .or_insert_with(|| cap(&cast));
                        }
                    }
                }
                AdapterSourceWork::Constructor(work) => {
                    let mut eager = 0u64;
                    let mut deferred = 0u64;
                    for value in &work.prepared {
                        match &value.origin {
                            AdapterDeclaredOrigin::Source(_) => {} // source/view same root capacity
                            AdapterDeclaredOrigin::ImmediateScalar => {
                                eager = eager.checked_add(cap(&value.facts)).ok_or_else(|| {
                                    Error::Msg("adapter immediate source bytes overflow".into())
                                })?
                            }
                            AdapterDeclaredOrigin::DeferredCreation => {
                                deferred =
                                    deferred.checked_add(cap(&value.facts)).ok_or_else(|| {
                                        Error::Msg("adapter deferred source bytes overflow".into())
                                    })?
                            }
                        }
                    }
                    self.eager
                        .insert((work.epoch, work.attempt, "constructor-scalars"), eager);
                    if self
                        .constructors
                        .insert((work.epoch, work.attempt), (eager, deferred))
                        .is_some()
                    {
                        return Err(Error::Msg(
                            "adapter constructor identity repeated in SAME prefix".into(),
                        ));
                    }
                }
                AdapterSourceWork::Install {
                    epoch,
                    retained_index,
                    canonical_target,
                    constructor_attempt,
                } => {
                    let endpoint = prefix.retained.get(*retained_index).ok_or_else(|| {
                        Error::Msg("installed global slot has no retained endpoint".into())
                    })?;
                    let constructor = prefix
                        .work
                        .iter()
                        .find_map(|event| match event {
                            AdapterSourceWork::Constructor(work)
                                if work.epoch == *epoch && work.attempt == *constructor_attempt =>
                            {
                                Some(work)
                            }
                            _ => None,
                        })
                        .ok_or_else(|| {
                            Error::Msg("installed slot has no SAME epoch/constructor".into())
                        })?;
                    if endpoint.0 != *canonical_target
                        || constructor.retained.as_ref() != Some(&endpoint.1)
                        || !self
                            .constructors
                            .contains_key(&(*epoch, *constructor_attempt))
                        || *retained_index != self.slots.len()
                        || self
                            .slots
                            .insert(*retained_index, (*epoch, *constructor_attempt))
                            .is_some()
                    {
                        return Err(Error::Msg(
                            "adapter global slot/epoch/endpoint chronology disagrees".into(),
                        ));
                    }
                }
                AdapterSourceWork::MaterializeAccessedAttempt { epoch, keys } => {
                    for key in keys {
                        self.raw_key(prefix, *epoch, key)?;
                    }
                }
            }
        }
        if self.slots.len() != prefix.retained.len() {
            return Err(Error::Msg(
                "retained endpoints lack exact global install slots".into(),
            ));
        }
        for file in &prefix.files {
            let total = self
                .raw
                .iter()
                .filter(|(k, _)| k.epoch == file.epoch)
                .map(|(_, &n)| n)
                .chain(
                    self.eager
                        .iter()
                        .filter(|((e, _, _), _)| *e == file.epoch)
                        .map(|(_, &n)| n),
                )
                .try_fold(0u64, |sum, n| {
                    sum.checked_add(n)
                        .ok_or_else(|| Error::Msg("file source union overflow".into()))
                })?;
            self.file_roots.insert(file.epoch, total);
        }
        Ok(())
    }
}

/// PiD replaces the VAE forward, not the already-created Z adapter source slots. The same
/// global slot is counted once whether resident, captured or locally installed. No label/target
/// subtraction or source-prefix replay is performed at the later decode source operation.
pub(super) fn pid_decode_source_owners(
    case: &AdapterSourceCase,
    ledger: &Ledger,
    positive_z_forward: bool,
) -> Result<u64> {
    if !positive_z_forward {
        return Ok(0);
    }
    let mut deferred = 0u64;
    for (slot, (_, facts)) in case.prefix.retained.iter().enumerate() {
        if facts.disabled() {
            continue;
        }
        let constructor = ledger.slots.get(&slot).ok_or_else(|| {
            Error::Msg("PiD decode missing SAME global adapter source slot".into())
        })?;
        let created = ledger
            .constructors
            .get(constructor)
            .ok_or_else(|| Error::Msg("PiD decode missing SAME adapter constructor".into()))?
            .1;
        deferred = deferred
            .checked_add(created)
            .ok_or_else(|| Error::Msg("PiD adapter endpoint source union overflow".into()))?;
    }
    Ok(deferred)
}

fn metadata_adapter(facts: &AdapterFacts) -> Result<Adapter> {
    // Endpoint proxies ONLY for the SAME existing production scratch functions. Their source
    // parents/creation history live on the ledger, never inferred from these proxy Arrays.
    Ok(match facts {
        AdapterFacts::Lora { a, b, disabled } => Adapter::Lora {
            a: carrier(&a.shape, a.dtype)?,
            b: carrier(&b.shape, b.dtype)?,
            scale: if *disabled { 0.0 } else { 1.0 },
        },
        AdapterFacts::Materialized { delta, disabled } => Adapter::Lokr {
            delta: carrier(&delta.shape, delta.dtype)?,
            scale: if *disabled { 0.0 } else { 1.0 },
        },
        AdapterFacts::Structured { w1, w2, disabled } => Adapter::LokrStructured {
            factors: LokrFactors {
                w1: carrier(&w1.shape, w1.dtype)?,
                w2: carrier(&w2.shape, w2.dtype)?,
                a: w1.shape[0],
                c: w1.shape[1],
                b: w2.shape[0],
                d: w2.shape[1],
                scale: if *disabled { 0.0 } else { 1.0 },
            },
        },
    })
}

fn base_mut(model: &mut Model) -> &mut crate::transformer::ZImageTransformer {
    match &mut model.dit {
        super::Dit::Base(base) => base,
        super::Dit::Control(control) => &mut control.base,
    }
}

fn set_metadata(model: &mut Model, target: &str, adapters: Vec<Adapter>) -> Result<()> {
    let parts = target.split('.').collect::<Vec<_>>();
    base_mut(model)
        .adaptable_mut(&parts)
        .ok_or_else(|| {
            Error::Msg(format!(
                "metadata adapter target is not actual Z namespace: {target}"
            ))
        })?
        .set_adapters(adapters.clone());
    if let ["layers", index, rest @ ..] = parts.as_slice() {
        let index = index
            .parse::<usize>()
            .map_err(|_| Error::Msg("invalid local adapter block".into()))?;
        model
            .dit_blocks
            .get_mut(index)
            .ok_or_else(|| Error::Msg("local metadata block missing".into()))?
            .0
            .adaptable_mut(rest)
            .ok_or_else(|| Error::Msg("local metadata adapter target missing".into()))?
            .set_adapters(adapters);
    }
    Ok(())
}

pub(super) fn install_metadata(
    model: &mut Model,
    case: &AdapterSourceCase,
    previous: &mut BTreeSet<String>,
) -> Result<()> {
    for target in std::mem::take(previous) {
        set_metadata(model, &target, Vec::new())?;
    }
    let mut targets = BTreeMap::<String, Vec<Adapter>>::new();
    for (target, facts) in &case.prefix.retained {
        targets
            .entry(target.clone())
            .or_default()
            .push(metadata_adapter(facts)?);
    }
    for (target, adapters) in targets {
        set_metadata(model, &target, adapters)?;
        previous.insert(target);
    }
    Ok(())
}

pub(super) fn load_windows(
    model: &Model,
    case: &AdapterSourceCase,
    quanta: &[Priced],
    ledger: &Ledger,
    staged: bool,
) -> Result<Vec<IosWindowPrice>> {
    let last_load = quanta
        .iter()
        .rev()
        .find(|q| q.window.phase == IosPhase::Load)
        .ok_or_else(|| Error::Msg("adapter load lacks SAME source base load window".into()))?;
    // VAE_LOAD's guard precedes VAE creation; reusing that resident subtotal would omit VAE at
    // adapter load. Ask the SAME source cursor at the ACTUAL next load label/history instead.
    let te = model
        .facts
        .te
        .resident(|name| model.te_constructor_keys.contains(name));
    let structural = model
        .dit_resident()
        .checked_add(model.control_resident())
        .and_then(|n| n.checked_add(if staged { 0 } else { te }))
        .ok_or_else(|| Error::Msg("adapter heavy source structural overflow".into()))?;
    let mut cursor =
        super::QuantizationCursor::new(super::QuantizationHistory::RequestFirst, false);
    let (base_resident, base_created, _) = cursor
        .window_with_vae(
            &model.facts,
            IosPhase::Load,
            mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM,
            0,
            structural,
            staged,
            last_load.source_vae.as_deref(),
        )?
        .ok_or_else(|| Error::Msg("adapter base source prefix missing".into()))?;
    let mut out = Vec::new();
    let mut prior = 0u64;
    for file in &case.prefix.files {
        let raw_floor = file.headers.values().try_fold(0u64, |sum, t| {
            sum.checked_add(t.bytes())
                .ok_or_else(|| Error::Msg("adapter raw floor overflow".into()))
        })?;
        let current = ledger.file_roots.get(&file.epoch).copied().unwrap_or(0);
        let mut window = last_load.window.clone();
        window.label = mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM.into();
        window.class = Some(format!(
            "z-image/adapter-source/case-{}/file-{}/load-prefix/staged-{staged}",
            case.case_id, file.epoch.0
        ));
        // The existing before_load floor and constructor source union belong to this SAME file
        // history. No independent upper is subtracted or multiplied by captured/local handles.
        window.resident_bytes = base_resident
            .checked_add(prior)
            .ok_or_else(|| Error::Msg("adapter load resident overflow".into()))?;
        window.held_bytes = last_load
            .source_held
            .checked_add(base_created)
            .ok_or_else(|| Error::Msg("adapter base-created source overflow".into()))?;
        window.scratch_bytes = raw_floor.max(current);
        window.owner_domain = format!(
            "{}/adapter-case-{}/file-{}",
            last_load.window.owner_domain, case.case_id, file.epoch.0
        );
        out.push(window);
        prior = prior
            .checked_add(current)
            .ok_or_else(|| Error::Msg("adapter prefix carry overflow".into()))?;
    }
    if let AdapterSourceOutcome::Refused(
        mlx_gen::adapters::loader::source_facts::AdapterSourceRefusal::Header { file, .. },
    ) = &case.outcome
    {
        let mut window = last_load.window.clone();
        window.label = mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM.into();
        window.class = Some(format!(
            "z-image/adapter-source/case-{}/file-{file}/header-refusal-prefix",
            case.case_id
        ));
        window.resident_bytes = base_resident
            .checked_add(ledger.held()?)
            .ok_or_else(|| Error::Msg("header-refusal carry overflow".into()))?;
        window.held_bytes = last_load
            .source_held
            .checked_add(base_created)
            .ok_or_else(|| Error::Msg("header-refusal created overflow".into()))?;
        // No tensor payload work for an unreadable current header. Host metadata IO/heap is a
        // separate unpriced domain, NOT asserted zero native/process demand.
        window.scratch_bytes = 0;
        out.push(window);
    }
    Ok(out)
}

/// A class is evidence for exactly ONE recorded operation, never permission to pick the first
/// same-label block. Missing or ambiguous correspondence aborts this case before its commit.
fn exact_source_operation_index<'a>(
    label: &str,
    class: &str,
    operations: impl IntoIterator<Item = (usize, &'a str, usize)>,
) -> Result<usize> {
    let mut matches = operations
        .into_iter()
        .filter_map(|(index, source_label, point)| {
            (source_label == label
                && class.contains(&format!("/{source_label}/source-op-{point}/")))
            .then_some(index)
        });
    let index = matches
        .next()
        .ok_or_else(|| Error::Msg("SAME adapter source operation identity missing".into()))?;
    if matches.next().is_some() {
        return Err(Error::Msg(
            "SAME adapter source operation identity is ambiguous".into(),
        ));
    }
    Ok(index)
}

/// ONE global endpoint slot, irrespective of capture/local handle clones. Raw roots and eager
/// alpha/scalar work are positive parents on Ledger::held, even when this slot is disabled or a
/// solve has no model callback. Local base/quant epochs remain in the original source quanta.
fn slot_window_owners(
    endpoint: &AdapterFacts,
    deferred: u64,
    first: (IosPhase, &str, usize, usize),
    current: (IosPhase, &str, usize, usize),
    no_forward: bool,
    recurrent: bool,
) -> (u64, u64) {
    if endpoint.disabled() || no_forward {
        return (0, 0);
    }
    let same =
        current.1 == first.1 && (current.2 == first.2 || first.1 == crate::quantum::DIT_EMBED);
    let warm = (recurrent && current.0 == IosPhase::Denoise)
        || current.0 == IosPhase::Decode
        || (current.0 == IosPhase::Denoise && first.3 < current.3 && !same);
    if warm {
        (deferred, 0)
    } else if current.0 == first.0 && same {
        (0, deferred)
    } else {
        (0, 0)
    }
}

pub(super) fn decorate(
    model: &Model,
    case: &AdapterSourceCase,
    ledger: &Ledger,
    quanta: &[Priced],
    pricing: &mut IosPricing,
    route: &Route,
) -> Result<()> {
    let facts = &model.facts;
    let held = ledger.held()?;
    let positions = case
        .prefix
        .retained
        .iter()
        .map(|(target, _)| facts.quantization_position(1, target))
        .collect::<Result<Vec<_>>>()?;
    for window in &mut pricing.windows {
        if window.phase == IosPhase::Load {
            continue;
        }
        if window.phase == IosPhase::Conditioning && route.execution.stage_residency {
            continue;
        }
        let class = window.class.as_deref().unwrap_or("");
        let no_forward = class.contains("no-model-forward");
        let recurrent = class.contains("denoise-recurrent") || class.contains("serial-later-final");
        let operation_index = exact_source_operation_index(
            &window.label,
            class,
            quanta
                .iter()
                .enumerate()
                .map(|(index, q)| (index, q.label, q.source_point)),
        )?;
        let operation = &quanta[operation_index];
        let mut before = 0u64;
        let mut current_creation = 0u64;
        for (slot, (_, retained)) in case.prefix.retained.iter().enumerate() {
            let constructor = ledger
                .slots
                .get(&slot)
                .ok_or_else(|| Error::Msg("retained slot source recipe missing".into()))?;
            let deferred = ledger
                .constructors
                .get(constructor)
                .ok_or_else(|| Error::Msg("retained constructor missing".into()))?
                .1;
            let point = positions[slot];
            let first = quanta
                .iter()
                .position(|q| q.label == point.1 && q.source_point == point.2)
                .ok_or_else(|| {
                    Error::Msg("adapter projection first-use operation missing".into())
                })?;
            let (resident, created) = slot_window_owners(
                retained,
                deferred,
                (point.0, point.1, point.2, first),
                (
                    window.phase,
                    operation.label,
                    operation.source_point,
                    operation_index,
                ),
                no_forward,
                recurrent,
            );
            before = before
                .checked_add(resident)
                .ok_or_else(|| Error::Msg("adapter warm union overflow".into()))?;
            current_creation = current_creation
                .checked_add(created)
                .ok_or_else(|| Error::Msg("adapter first-use creation overflow".into()))?;
        }
        // Each alternative cold/recurrent/zero/late/serial window is priced independently, never
        // by a mutable cursor over an already-enveloped/sorted collection of unrelated histories.
        window.resident_bytes = window
            .resident_bytes
            .checked_add(held)
            .and_then(|n| n.checked_add(before))
            .ok_or_else(|| Error::Msg("adapter retained union overflow".into()))?;
        window.held_bytes = window
            .held_bytes
            .checked_add(current_creation)
            .ok_or_else(|| Error::Msg("adapter current union overflow".into()))?;
        window.owner_domain = format!("{}/adapter-case-{}", window.owner_domain, case.case_id);
    }
    // Load is per file, AFTER actual heavy base/VAE quant construction; source statements remain
    // separate from staging/native Data release. Output/recurrent families are generated upstream,
    // not by replaying requested N or multiplying a local adapter payload by window count.
    for window in load_windows(model, case, quanta, ledger, route.execution.stage_residency)? {
        pricing.take_window(window);
    }
    Ok(())
}

pub(super) fn price_request(
    provider: &str,
    spec: &LoadSpec,
    model: &mut Model,
    route: &Route,
    sampling: &mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
    initial_width: u64,
) -> std::result::Result<AdapterPricingReport, AdapterPricingFailure> {
    let setup = (|| -> Result<_> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(Error::Unsupported(
                "Z adapter source pricing requires snapshot directory".into(),
            ));
        };
        let mut headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(
            root.join("transformer"),
        )
        .map_err(Error::from)?
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect::<BTreeMap<_, _>>();
        crate::loader::remap_transformer_header_keys(&mut headers);
        let cfg = model.facts.dit_cfg.clone();
        let kohya = crate::transformer::adapter_header_targets(&cfg, false)
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>();
        let mut projections = BTreeMap::new();
        for (path, header) in crate::transformer::adapter_header_targets(&cfg, true) {
            projections.insert(
                path,
                linear_header_facts(
                    &headers,
                    &header,
                    mlx_gen::quant::DEFAULT_GROUP_SIZE,
                    spec.quantize.map(|q| q.bits()),
                )?,
            );
        }
        Ok((cfg, kohya, projections))
    })();
    let (cfg, kohya, projections) = setup.map_err(|error| AdapterPricingFailure {
        error,
        source_prefix: AdapterSourcePrefix::default(),
        attempted_case_id: None,
        committed: Vec::new(),
    })?;
    let mut projection = |path: &str| -> Result<_> {
        let Some(canonical) = crate::transformer::canonical_adapter_path(&cfg, path) else {
            return Ok(None);
        };
        Ok(projections
            .get(&canonical)
            .cloned()
            .map(|facts| (canonical, facts)))
    };
    let mut prices = AdapterCasePrices::default();
    let mut previous = BTreeSet::new();
    let mut consume = |case: &AdapterSourceCase| -> Result<()> {
        prices.consume(case, || {
            let mut ledger = Ledger::default();
            ledger.apply(&case.prefix)?;
            install_metadata(model, case, &mut previous)?;
            let mut local = IosPricing::default();
            model.visit_quanta(route, Backend::Metal, &mut |quanta| {
                if matches!(case.outcome, AdapterSourceOutcome::Refused(_)) {
                    for window in load_windows(
                        model,
                        case,
                        quanta,
                        &ledger,
                        route.execution.stage_residency,
                    )? {
                        local.take_window(window);
                    }
                } else {
                    let mut priced =
                        price_source_quanta(&model.facts, quanta, route, sampling, initial_width)?;
                    decorate(model, case, &ledger, quanta, &mut priced, route)?;
                    for window in priced.windows {
                        local.take_window(window);
                    }
                    local.absent_phases.extend(priced.absent_phases);
                }
                Ok(())
            })?;
            Ok(local)
        })
    };
    let visited = visit_configured_adapter_source_cases(
        &spec.adapters,
        &kohya,
        mlx_gen::adapters::DEFAULT_LYCORIS_DTYPE,
        &mut projection,
        &mut consume,
    );
    drop(consume);
    prices.finish(provider, visited)
}

/// Reachable from BOTH actual public request/token pricing entrypoints through price_controls.
/// This source-feedback phase preserves the pre-existing outward adapter refusal AFTER full
/// ledger production/consumption. It is not an API-only or test-only disconnected source builder,
/// and is NOT a permanent unsupported/default finish: native correspondence/complete consumer
/// histories must be qualified before that existing guard is removed in the next exact checkpoint.
pub(super) fn price_before_existing_guard(
    req: &mlx_gen::ios_pricing::IosPricingRequest<'_>,
    route: &Route,
    sampling: &mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
    initial_width: u64,
) -> Result<IosPricing> {
    let mut component_spec = req.spec.clone();
    component_spec.adapters.clear(); // ONLY base component metadata extraction; not the request
    let facts = Facts::from_spec(req.provider, &component_spec)?;
    let mut model = Model::new(facts)?;
    let report = price_request(
        req.provider,
        req.spec,
        &mut model,
        route,
        sampling,
        initial_width,
    )
    .map_err(AdapterPricingFailure::into_legacy_error)?; // Consume provenance, preserve SAME typed cause.
    if !report.source_summary.enumeration_complete || report.source_summary.complete_cases == 0 {
        return Err(Error::Msg(
            "adapter source pricing did not complete constructor histories".into(),
        ));
    }
    // Keep SAME existing public guard until source-case native/consumer qualification. The report
    // contains actual ADAPTER_LOAD/first-use/disabled/capture-local unions for private conformance;
    // it is not treated as native/process/physical evidence or a successful admission envelope.
    let _priced_windows = report.pricing;
    let _refused_prefixes = report.refusal_prefixes;
    Err(Error::Unsupported(format!(
        "{}: iOS pricing does not price LoRA/LoKr adapter residuals",
        req.provider
    )))
}

#[cfg(test)]
mod source_ledger_tests {
    use super::*;
    use mlx_gen::adapters::loader::source_facts::{
        AdapterActionId, AdapterConstructorWork, AdapterDeclaredTensor, AdapterFileSources,
        AdapterParseAction, AdapterParseStop, NativeUnorderedParsePrefix, ParseActionKind,
    };
    use mlx_gen::adapters::loader::UnfusedAdapterFormat;
    use std::io::Write;
    use std::sync::Arc;

    fn t(shape: &[i32], dtype: Dtype) -> TensorFacts {
        TensorFacts {
            shape: shape.to_vec(),
            dtype,
            strides: Vec::new(),
        }
    }

    #[test]
    fn raw_union_is_by_epoch_key_but_alpha_and_constructor_work_is_by_action() {
        let epoch = AdapterFileEpoch(0);
        let alpha = |ordinal| AdapterActionId { epoch, ordinal };
        let file = AdapterFileSources {
            epoch,
            format: UnfusedAdapterFormat::PeftLora,
            headers: Arc::new(BTreeMap::from([
                ("p.alpha".into(), t(&[7], Dtype::Float16)),
                ("transformer.p.alpha".into(), t(&[7], Dtype::Bfloat16)),
                ("orphan.lora_A.weight".into(), t(&[3, 8], Dtype::Bfloat16)),
            ])),
        };
        let actions = vec![
            AdapterParseAction {
                id: alpha(0),
                key: "p.alpha".into(),
                kind: ParseActionKind::Alpha {
                    group: "p".into(),
                    empty: false,
                },
            },
            AdapterParseAction {
                id: alpha(1),
                key: "transformer.p.alpha".into(),
                kind: ParseActionKind::Alpha {
                    group: "p".into(),
                    empty: false,
                },
            },
        ];
        let family = NativeUnorderedParsePrefix {
            epoch,
            actions,
            must_visit: [alpha(0), alpha(1)].into_iter().collect(),
            may_visit: [alpha(0), alpha(1)].into_iter().collect(),
            stop: AdapterParseStop::End,
            generic_lora_alpha_conflicts: true,
        };
        let mut prefix = AdapterSourcePrefix {
            files: vec![file],
            ..AdapterSourcePrefix::default()
        };
        prefix.work.push(AdapterSourceWork::Parse(family));
        prefix
            .work
            .push(AdapterSourceWork::MaterializeAccessedAttempt {
                epoch,
                keys: [
                    "p.alpha".into(),
                    "transformer.p.alpha".into(),
                    "orphan.lora_A.weight".into(),
                ]
                .into_iter()
                .collect(),
            });
        for attempt in [5, 6] {
            prefix.work.push(AdapterSourceWork::Constructor(Box::new(
                AdapterConstructorWork {
                    epoch,
                    group: format!("g{attempt}"),
                    recipe: None,
                    attempt,
                    probe: false,
                    operands: Vec::new(),
                    retained: None,
                    operations: Vec::new(),
                    prepared: vec![AdapterDeclaredTensor {
                        facts: t(&[1], Dtype::Float32),
                        origin: AdapterDeclaredOrigin::ImmediateScalar,
                    }],
                },
            )));
        }
        let mut ledger = Ledger::default();
        ledger.apply(&prefix).unwrap();
        // Raw each alpha14 + orphan48, two FULL alphaF32 casts28 each, two scalar attempts4 each.
        // A later raw-materialize action repeats each alpha SOURCE but must not drop either cast.
        assert_eq!(ledger.raw.len(), 3);
        assert_eq!(ledger.held().unwrap(), 14 + 14 + 48 + 28 + 28 + 4 + 4);
        assert_eq!(ledger.eager.len(), 4);
        let mut repeated = prefix.clone();
        repeated.files.push(AdapterFileSources {
            epoch: AdapterFileEpoch(1),
            headers: Arc::clone(&repeated.files[0].headers),
            format: UnfusedAdapterFormat::PeftLora,
        });
        repeated
            .work
            .push(AdapterSourceWork::MaterializeAccessedAttempt {
                epoch: AdapterFileEpoch(1),
                keys: ["p.alpha".into()].into_iter().collect(),
            });
        let mut ledger = Ledger::default();
        ledger.apply(&repeated).unwrap();
        assert_eq!(
            ledger.raw.len(),
            4,
            "reopened literal-path epoch is not native source/Data dedup"
        );
        assert_eq!(ledger.held().unwrap(), 14 + 14 + 48 + 28 + 28 + 4 + 4 + 14);
    }
    #[test]
    fn operation_identity_never_falls_back_to_first_same_label() {
        let operations = [(0, "DIT_BLOCK", 0), (1, "DIT_BLOCK", 1)];
        assert_eq!(
            exact_source_operation_index(
                "DIT_BLOCK",
                "z-image/text/denoise/DIT_BLOCK/source-op-1/request-first/",
                operations
            )
            .unwrap(),
            1
        );
        assert!(exact_source_operation_index(
            "DIT_BLOCK",
            "z-image/text/denoise/DIT_BLOCK/source-op-9/request-first/",
            operations
        )
        .is_err());
        assert!(exact_source_operation_index(
            "DIT_BLOCK",
            "z-image/text/denoise/DIT_BLOCK/request-first/",
            operations
        )
        .is_err());
        assert!(exact_source_operation_index(
            "DIT_BLOCK",
            "/DIT_BLOCK/source-op-1/request-first/",
            [(0, "DIT_BLOCK", 1), (1, "DIT_BLOCK", 1)]
        )
        .is_err());
    }

    #[test]
    fn retained_endpoint_requires_the_exact_epoch_constructor_and_global_slot() {
        let epoch = AdapterFileEpoch(0);
        let facts = AdapterFacts::Materialized {
            delta: t(&[8, 8], Dtype::Bfloat16),
            disabled: true,
        };
        let mut prefix = AdapterSourcePrefix::default();
        prefix.retained.push(("p".into(), facts.clone()));
        prefix.work.push(AdapterSourceWork::Constructor(Box::new(
            AdapterConstructorWork {
                epoch,
                group: "p".into(),
                recipe: None,
                attempt: 3,
                probe: false,
                operands: Vec::new(),
                prepared: Vec::new(),
                operations: Vec::new(),
                retained: Some(facts),
            },
        )));
        prefix.work.push(AdapterSourceWork::Install {
            epoch,
            retained_index: 0,
            canonical_target: "p".into(),
            constructor_attempt: 3,
        });
        let mut ledger = Ledger::default();
        ledger.apply(&prefix).unwrap();
        assert_eq!(ledger.slots, BTreeMap::from([(0, (epoch, 3))]));
        let mut wrong_epoch = prefix.clone();
        let Some(AdapterSourceWork::Install { epoch, .. }) = wrong_epoch.work.last_mut() else {
            panic!("independent install fixture");
        };
        *epoch = AdapterFileEpoch(1);
        assert!(Ledger::default().apply(&wrong_epoch).is_err());
        let mut repeated = prefix.clone();
        repeated.work.push(prefix.work.last().unwrap().clone());
        assert!(
            Ledger::default().apply(&repeated).is_err(),
            "captured/local clones are not another source slot install"
        );
    }

    #[test]
    fn same_global_slot_consumes_cold_recurrent_zero_late_and_serial_history_owners() {
        let endpoint = AdapterFacts::Materialized {
            delta: t(&[8, 8], Dtype::Bfloat16),
            disabled: false,
        };
        let first = (IosPhase::Denoise, "DIT_BLOCK", 2, 7);
        // Literal whole-slot capacity, not output of another pricing helper used as our oracle.
        let rows = [
            (
                (IosPhase::Conditioning, "TE_LAYER", 0, 1),
                false,
                false,
                (0, 0),
            ),
            ((IosPhase::Denoise, "DIT_BLOCK", 1, 6), false, false, (0, 0)),
            (
                (IosPhase::Denoise, "DIT_BLOCK", 2, 7),
                false,
                false,
                (0, 128),
            ),
            (
                (IosPhase::Denoise, "DIT_BLOCK", 3, 8),
                false,
                false,
                (128, 0),
            ),
            (
                (IosPhase::Denoise, "DIT_BLOCK", 1, 6),
                false,
                true,
                (128, 0),
            ),
            ((IosPhase::Denoise, "DIT_BLOCK", 2, 7), true, false, (0, 0)),
            (
                (IosPhase::Decode, "VAE_DECODE_IN", 0, 9),
                false,
                false,
                (128, 0),
            ),
            (
                (IosPhase::Decode, "VAE_DECODE_OUT", 0, 12),
                false,
                false,
                (128, 0),
            ),
            (
                (IosPhase::Decode, "VAE_DECODE_OUT", 0, 12),
                true,
                false,
                (0, 0),
            ),
        ];
        for (current, zero, recurrent, expected) in rows {
            assert_eq!(
                slot_window_owners(&endpoint, 128, first, current, zero, recurrent),
                expected,
                "SLOT-HISTORY independent endpoint/window comparison {current:?}"
            );
        }
        let disabled = AdapterFacts::Materialized {
            delta: t(&[8, 8], Dtype::Bfloat16),
            disabled: true,
        };
        assert_eq!(
            slot_window_owners(&disabled, 128, first, first, false, false),
            (0, 0)
        );
        // Alternative input dtype quanta for SAME embedding operation both begin cold.
        for point in [0, 1] {
            assert_eq!(
                slot_window_owners(
                    &endpoint,
                    128,
                    (IosPhase::Denoise, crate::quantum::DIT_EMBED, 0, 3),
                    (
                        IosPhase::Denoise,
                        crate::quantum::DIT_EMBED,
                        point,
                        3 + point
                    ),
                    false,
                    false
                ),
                (0, 128)
            );
        }
    }

    fn prior_windows() -> Vec<IosWindowPrice> {
        [(11, 13, 17), (19, 23, 29)]
            .into_iter()
            .enumerate()
            .map(|(index, (resident, held, scratch))| IosWindowPrice {
                phase: IosPhase::Load,
                class: Some(format!("independent-case0-window{index}")),
                label: mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM.into(),
                resident_bytes: resident,
                held_bytes: held,
                scratch_bytes: scratch,
                basis: None,
                owner_domain: format!("independent/case0/window{index}"),
            })
            .collect()
    }

    fn local_prices() -> IosPricing {
        let mut local = IosPricing::default();
        for window in prior_windows() {
            local.take_window(window);
        }
        local
    }

    fn failed(
        result: std::result::Result<AdapterPricingReport, AdapterPricingFailure>,
    ) -> AdapterPricingFailure {
        match result {
            Err(failure) => failure,
            Ok(_) => panic!("aborted/all-refused cases cannot produce a successful envelope"),
        }
    }

    fn metadata_adapter_file(root: &std::path::Path, paired: bool) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        header.insert(
            "p.lora_A.weight".into(),
            serde_json::json!({"dtype":"F32","shape":[1,8],"data_offsets":[0,32]}),
        );
        if paired {
            header.insert(
                "p.lora_B.weight".into(),
                serde_json::json!({"dtype":"F32","shape":[8,1],"data_offsets":[32,64]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let path = root.join("metadata-only.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + if paired { 64 } else { 32 })
            .unwrap();
        path
    }

    fn projection(path: &str) -> Result<Option<(String, mlx_gen::adapters::LinearFacts)>> {
        Ok((path == "p").then(|| {
            (
                "p".into(),
                mlx_gen::adapters::LinearFacts {
                    base_shape: vec![8, 8],
                    is_quantized: false,
                    has_live_adapters: false,
                    adapter_count: 0,
                    bias_shape: None,
                },
            )
        }))
    }

    fn callback_abort_fixture() -> (AdapterPricingFailure, AdapterSourceCase, usize) {
        let dir = tempfile::tempdir().unwrap();
        let spec = mlx_gen::runtime::AdapterSpec {
            path: metadata_adapter_file(dir.path(), true),
            kind: mlx_gen::runtime::AdapterKind::Lora,
            scale: 1.0,
            pass_scales: None,
            moe_expert: None,
        };
        let mut prices = AdapterCasePrices::default();
        let calls = std::cell::Cell::new(0);
        let mut first = None;
        let mut consume = |case: &AdapterSourceCase| {
            if first.is_none() {
                first = Some(case.clone());
            }
            prices.consume(case, || {
                calls.set(calls.get() + 1);
                if case.case_id == 0 {
                    Ok(local_prices())
                } else {
                    Err(Error::Io(std::io::Error::from_raw_os_error(13)))
                }
            })
        };
        let visited = visit_configured_adapter_source_cases(
            &[spec],
            &[],
            Dtype::Bfloat16,
            &mut projection,
            &mut consume,
        );
        drop(consume);
        (
            failed(prices.finish("z_image", visited)),
            first.unwrap(),
            calls.get(),
        )
    }

    #[test]
    fn actual_callback_abort_preserves_prefix_and_previous_whole_windows_once() {
        let (failure, first, calls) = callback_abort_fixture();
        assert_eq!(
            calls, 2,
            "actual source callback consumes refused case0 then aborts current case1 once"
        );
        assert!(
            matches!(first.outcome, AdapterSourceOutcome::Refused(_)),
            "refused DATA can commit prefix prices without abort/replay"
        );
        assert_eq!(failure.attempted_case_id, Some(1));
        assert_eq!(
            failure.committed,
            prior_windows(),
            "both prior WHOLE windows survive in original order, not just max/last"
        );
        assert_eq!(failure.source_prefix.files.len(), 1);
        assert_eq!(failure.source_prefix.retained.len(), 1);
        assert!(matches!(
            failure.source_prefix.work.last(),
            Some(AdapterSourceWork::MaterializeAccessedComplete(
                AdapterFileEpoch(0)
            ))
        ));
        let mut emitted = Vec::new();
        let cause =
            failure.into_legacy_error_with(|kind, fields| emitted.push((kind, fields.to_vec())));
        match cause {
            Error::Io(error) => assert_eq!(error.raw_os_error(), Some(13)),
            other => panic!("Io cause/code replaced: {other:?}"),
        }
        let header = &emitted[0];
        assert_eq!(header.0, "adapter-source-pricing-abort");
        assert!(header.1.contains(&("case_present", 1)) && header.1.contains(&("case_id", 1)));
        let work = emitted
            .iter()
            .filter(|(kind, _)| *kind == "adapter-source-pricing-abort-work")
            .map(|(_, fields)| {
                fields
                    .iter()
                    .find(|(key, _)| *key == "work_kind")
                    .unwrap()
                    .1
            })
            .collect::<Vec<_>>();
        assert_eq!(
            work,
            vec![0, 1, 2, 3, 5, 6, 7],
            "actual ordered file/parse/constructor/install/raw-attempt/complete prefix"
        );
        let windows = emitted
            .iter()
            .filter(|(kind, _)| *kind == "adapter-source-pricing-abort-committed-window")
            .map(|(_, fields)| fields.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            windows,
            vec![
                vec![
                    ("ordinal", 0),
                    ("phase", 0),
                    ("resident_bytes", 11),
                    ("held_bytes", 13),
                    ("scratch_bytes", 17)
                ],
                vec![
                    ("ordinal", 1),
                    ("phase", 0),
                    ("resident_bytes", 19),
                    ("held_bytes", 23),
                    ("scratch_bytes", 29)
                ]
            ],
            "diagnostic consumes EACH previous committed window without charging/replay"
        );
    }

    #[test]
    fn global_abort_and_all_refused_keep_prior_prices_without_valid_envelope() {
        use mlx_gen::adapters::loader::source_facts::AdapterSourceAbortKind;
        let (original, first, _) = callback_abort_fixture();
        let mut prices = AdapterCasePrices::default();
        prices.consume(&first, || Ok(local_prices())).unwrap();
        let retried = std::cell::Cell::new(false);
        assert!(prices
            .consume(&first, || {
                retried.set(true);
                Ok(local_prices())
            })
            .is_err());
        assert!(
            !retried.get(),
            "duplicate case is rejected BEFORE fallible work and cannot doublecharge"
        );
        let mut prefix = original.source_prefix.clone();
        prefix
            .work
            .push(AdapterSourceWork::FileMapAttempt(AdapterFileEpoch(1)));
        let abort = AdapterSourceFailure {
            error: Error::Canceled,
            prefix,
            attempted_case_id: None,
            kind: AdapterSourceAbortKind::Enumerator,
        };
        let failure = failed(prices.finish("z_image", Err(Box::new(abort))));
        assert_eq!(failure.attempted_case_id, None);
        assert_eq!(failure.committed, prior_windows());
        assert!(matches!(
            failure.source_prefix.work.last(),
            Some(AdapterSourceWork::FileMapAttempt(AdapterFileEpoch(1)))
        ));
        let mut events = Vec::new();
        assert!(matches!(
            failure.into_legacy_error_with(|kind, fields| events.push((kind, fields.to_vec()))),
            Error::Canceled
        ));
        assert!(
            events[0].1.contains(&("case_present", 0)),
            "global pending prefix never fabricates a delivered case ID"
        );
        let dir = tempfile::tempdir().unwrap();
        let spec = mlx_gen::runtime::AdapterSpec {
            path: metadata_adapter_file(dir.path(), false),
            kind: mlx_gen::runtime::AdapterKind::Lora,
            scale: 1.0,
            pass_scales: None,
            moe_expert: None,
        };
        let mut prices = AdapterCasePrices::default();
        let mut consume = |case: &AdapterSourceCase| {
            assert!(matches!(case.outcome, AdapterSourceOutcome::Refused(_)));
            prices.consume(case, || Ok(local_prices()))
        };
        let visited = visit_configured_adapter_source_cases(
            &[spec],
            &[],
            Dtype::Bfloat16,
            &mut projection,
            &mut consume,
        );
        drop(consume);
        let failure = failed(prices.finish("z_image", visited));
        assert_eq!(
            failure.committed.len(),
            4,
            "both consumed refused prefixes retain both local windows"
        );
        assert!(matches!(
            failure.source_prefix.work.last(),
            Some(AdapterSourceWork::MaterializeAccessedComplete(_))
        ));
        assert!(
            matches!(
                failure.into_legacy_error_with(|_, _| {}),
                Error::Unsupported(_)
            ),
            "all-refused must never become generation pricing"
        );
    }

    #[test]
    fn actual_legacy_listener_enabled_and_disabled_preserves_typed_causes() {
        fn assert_typed_causes() {
            for (error, expected) in [
                (Error::Canceled, 0),
                (Error::Unsupported("same capability".into()), 1),
                (Error::MissingTensor("same key".into()), 2),
            ] {
                let failure = AdapterPricingFailure {
                    error,
                    source_prefix: AdapterSourcePrefix::default(),
                    attempted_case_id: None,
                    committed: Vec::new(),
                };
                match (failure.into_legacy_error(), expected) {
                    (Error::Canceled, 0) => {}
                    (Error::Unsupported(text), 1) => assert_eq!(text, "same capability"),
                    (Error::MissingTensor(key), 2) => assert_eq!(key, "same key"),
                    other => panic!("typed legacy cause replaced: {other:?}"),
                }
            }
        }
        const CHILD: &str = "ZIMAGE_ADAPTER_ABORT_DIAGNOSTIC_HOST_CHILD";
        const DONE: &str = "ZIMAGE_ADAPTER_ABORT_DIAGNOSTIC_HOST_DONE";
        const FQN: &str="ios_pricing::adapter_sources::source_ledger_tests::actual_legacy_listener_enabled_and_disabled_preserves_typed_causes";
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "HOST diagnostic child assertion failed");
            assert_eq!(
                std::fs::read(marker)
                    .expect("fresh HOST completion marker")
                    .as_slice(),
                b"ok"
            );
            return;
        }
        // Pure HOST metadata/stdio only: no Device/Array/model/native backend call is made here.
        assert!(
            !mlx_gen::ios_quantum::set_quantum_listener(None),
            "fresh isolated child has no prior listener"
        );
        let (failure, _, _) = callback_abort_fixture();
        match failure.into_legacy_error() {
            Error::Io(error) => assert_eq!(error.raw_os_error(), Some(13)),
            other => panic!("disabled listener changed Io cause: {other:?}"),
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        {
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    mlx_gen::ios_quantum::set_quantum_listener(None);
                }
            }
            let _reset = Reset;
            let sink = std::sync::Arc::clone(&captured);
            assert!(!mlx_gen::ios_quantum::set_quantum_listener(Some(Box::new(
                move |mark| {
                    if let mlx_gen::ios_quantum::QuantumMark::Metadata {
                        provider,
                        kind,
                        fields,
                    } = mark
                    {
                        sink.lock().unwrap().push((
                            provider.to_owned(),
                            kind.to_owned(),
                            fields
                                .iter()
                                .map(|(key, value)| ((*key).to_owned(), *value))
                                .collect::<Vec<_>>(),
                        ));
                    }
                }
            ))));
            let (failure, _, calls) = callback_abort_fixture();
            match failure.into_legacy_error() {
                Error::Io(error) => assert_eq!(error.raw_os_error(), Some(13)),
                other => panic!("enabled listener changed Io cause: {other:?}"),
            }
            assert_eq!(
                calls, 2,
                "diagnostic did not replay the aborted callback/prefix"
            );
            assert_typed_causes(); // SAME typed causes with actual listener enabled too.
            let events = captured.lock().unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|(_, kind, _)| kind == "adapter-source-pricing-abort")
                    .count(),
                4
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|(_, kind, _)| kind == "adapter-source-pricing-abort-committed-window")
                    .count(),
                2
            );
            assert!(events.iter().all(|(provider, _, _)| provider == "z_image"));
        }
        // Existing diagnostic path is optional. Removing its listener cannot change typed errors.
        assert!(!mlx_gen::ios_quantum::set_quantum_listener(None));
        assert_typed_causes();
        drop(captured);
        std::fs::write(
            std::env::var(DONE).expect("fresh HOST completion path"),
            b"ok",
        )
        .unwrap();
    }
}

#[cfg(all(test, target_os = "macos"))]
mod native_source_proofs {
    use super::*;
    use mlx_gen::adapters::loader::source_facts::{AdapterParseStop, ObservedAdapterParsePrefix};
    use mlx_gen::adapters::loader::{self, UnfusedAdapterFormat};
    use mlx_gen::adapters::{AdaptableLinear, LinearFacts};
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};
    use mlx_gen::weights::Weights;
    use std::io::Write;

    const CHILD: &str = "ZIMAGE_ADAPTER_SOURCE_FACTORY_CPU_CHILD";
    const DONE: &str = "ZIMAGE_ADAPTER_SOURCE_FACTORY_CPU_DONE";
    const FQN: &str = "ios_pricing::adapter_sources::native_source_proofs::native_unfused_sources_match_all_families_and_constructor_variants";

    struct Completion;
    impl Drop for Completion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "adapter source final CPU sync AFTER owner scopes dropped"
                );
                std::fs::write(std::env::var(DONE).expect("fresh completion path"), b"ok")
                    .expect("completion marker");
            }
        }
    }

    struct Host {
        lin: AdaptableLinear,
    }
    impl AdaptableHost for Host {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            if path == ["p"] {
                Some(&mut self.lin)
            } else {
                None
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["p".into()]
        }
    }

    fn tensor_file(
        root: &std::path::Path,
        name: &str,
        entries: &[(&str, &[usize], &str)],
        metadata: serde_json::Value,
    ) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata);
        let mut offset = 0u64;
        for (key, shape, dtype) in entries {
            let width = match *dtype {
                "I8" => 1,
                "F16" | "BF16" => 2,
                "F32" | "I32" => 4,
                "U64" => 8,
                _ => panic!("literal fixture dtype"),
            };
            let size = shape.iter().fold(width, |n, &d| n * d as u64);
            header.insert((*key).into(), serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+size]}));
            offset += size;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let path = root.join(format!("{name}.safetensors"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
        path
    }

    fn endpoint(facts: &AdapterFacts) -> (&'static str, Vec<(Vec<i32>, Dtype)>, bool) {
        let family = match facts {
            AdapterFacts::Lora { .. } => "lora",
            AdapterFacts::Structured { .. } => "structured",
            AdapterFacts::Materialized { .. } => "materialized",
        };
        (
            family,
            facts
                .tensors()
                .into_iter()
                .map(|t| (t.shape.clone(), t.dtype))
                .collect(),
            facts.disabled(),
        )
    }

    #[test]
    fn native_unfused_sources_match_all_families_and_constructor_variants() {
        let mut completion: Option<Completion> = None; // FIRST None; true final arm LAST below.
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "native source child failed");
            assert_eq!(
                std::fs::read(marker)
                    .expect("fresh normal completion")
                    .as_slice(),
                b"ok"
            );
            return;
        }
        // Before source files, Arrays and native factories. Never use default-stream inference.
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu()
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        eprintln!("ZIMAGE_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
        let mut mask = 0u32;
        {
            let dir = tempfile::tempdir().unwrap();
            use UnfusedAdapterFormat::*;
            type Entry<'a> = (&'a str, &'a [usize], &'a str);
            let cases: Vec<(
                &str,
                UnfusedAdapterFormat,
                bool,
                Vec<Entry<'_>>,
                serde_json::Value,
            )> = vec![
                (
                    "peft",
                    PeftLora,
                    false,
                    vec![
                        ("p.lora_A.weight", &[1, 64], "F16"),
                        ("p.lora_B.weight", &[64, 1], "BF16"),
                        ("p.alpha", &[7], "F16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "kohya",
                    KohyaLora,
                    false,
                    vec![
                        ("lora_unet_p.lora_down.weight", &[1, 64], "BF16"),
                        ("lora_unet_p.lora_up.weight", &[64, 1], "F32"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "peft-lokr",
                    PeftLokr,
                    false,
                    vec![
                        ("p.lokr_w1", &[8, 8], "F16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                    ],
                    serde_json::json!({"networkType":"lokr","rank":"2","alpha":"4"}),
                ),
                (
                    "tp-full",
                    ThirdpartyLokr,
                    false,
                    vec![
                        ("p.lokr_w1", &[8, 8], "F16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-product",
                    ThirdpartyLokr,
                    false,
                    vec![
                        ("p.lokr_w1_a", &[8, 2], "F16"),
                        ("p.lokr_w1_b", &[2, 8], "BF16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-structured",
                    ThirdpartyLokr,
                    true,
                    vec![
                        ("p.lokr_w1_a", &[8, 2], "F16"),
                        ("p.lokr_w1_b", &[2, 8], "BF16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                        ("p.alpha", &[7], "F16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-tucker",
                    ThirdpartyLokr,
                    true,
                    vec![
                        ("p.lokr_w1", &[8, 8], "BF16"),
                        ("p.lokr_t2", &[1, 1, 1, 1], "BF16"),
                        ("p.lokr_w2_a", &[1, 8], "F16"),
                        ("p.lokr_w2_b", &[1, 8], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-shadow",
                    ThirdpartyLokr,
                    true,
                    vec![
                        ("p.lokr_w1", &[8, 8], "BF16"),
                        ("p.lokr_w1_a", &[8, 2], "F16"),
                        ("p.lokr_w1_b", &[2, 8], "F16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                        ("p.lokr_t2", &[1, 1], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "loha",
                    ThirdpartyLoha,
                    false,
                    vec![
                        ("p.hada_w1_a", &[64, 2], "F16"),
                        ("p.hada_w1_b", &[2, 64], "BF16"),
                        ("p.hada_w2_a", &[64, 2], "BF16"),
                        ("p.hada_w2_b", &[2, 64], "F32"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "loha-tucker",
                    ThirdpartyLoha,
                    true,
                    vec![
                        ("p.hada_t1", &[1, 1, 1, 1], "BF16"),
                        ("p.hada_t2", &[1, 1, 1, 1], "BF16"),
                        ("p.hada_w1_a", &[1, 64], "F16"),
                        ("p.hada_w1_b", &[1, 64], "BF16"),
                        ("p.hada_w2_a", &[1, 64], "BF16"),
                        ("p.hada_w2_b", &[1, 64], "F32"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "loha-single-t",
                    ThirdpartyLoha,
                    false,
                    vec![
                        ("p.hada_t1", &[1, 1], "BF16"),
                        ("p.hada_w1_a", &[64, 2], "F16"),
                        ("p.hada_w1_b", &[2, 64], "BF16"),
                        ("p.hada_w2_a", &[64, 2], "BF16"),
                        ("p.hada_w2_b", &[2, 64], "F32"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "loha-batched",
                    ThirdpartyLoha,
                    false,
                    vec![
                        ("p.hada_w1_a", &[1, 64, 2], "F16"),
                        ("p.hada_w1_b", &[1, 2, 64], "BF16"),
                        ("p.hada_w2_a", &[1, 64, 2], "BF16"),
                        ("p.hada_w2_b", &[1, 2, 64], "F32"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-integer-promotion",
                    ThirdpartyLokr,
                    false,
                    vec![
                        ("p.lokr_w1_a", &[8, 2], "U64"),
                        ("p.lokr_w1_b", &[2, 8], "I8"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
                (
                    "tp-full-structured-samewidth",
                    ThirdpartyLokr,
                    true,
                    vec![
                        ("p.lokr_w1", &[8, 8], "F16"),
                        ("p.lokr_w2", &[8, 8], "BF16"),
                    ],
                    serde_json::json!({}),
                ),
            ];
            for (index, (name, format, packed, entries, metadata)) in cases.into_iter().enumerate()
            {
                let path = tensor_file(dir.path(), name, &entries, metadata);
                let spec = AdapterSpec {
                    path: path.clone(),
                    kind: if format == PeftLokr {
                        AdapterKind::Lokr
                    } else {
                        AdapterKind::Lora
                    },
                    scale: 1.0,
                    pass_scales: None,
                    moe_expert: None,
                };
                let mut host = Host {
                    lin: AdaptableLinear::dense(
                        mlx_rs::Array::zeros::<f32>(&[64, 64]).unwrap(),
                        None,
                    ),
                };
                if packed {
                    host.lin.quantize(4, Some(64)).unwrap();
                }
                let base = LinearFacts::of(&host.lin);
                let weights = Weights::from_file(&path).unwrap();
                let native_order = weights.keys().map(str::to_owned).collect::<Vec<_>>();
                assert_eq!(
                    loader::classify_unfused_adapter_format(
                        native_order.iter().map(String::as_str),
                        weights.metadata("networkType"),
                        spec.kind
                    ),
                    format
                );
                let report = match format {
                    PeftLora => loader::apply_lora_peft(&mut host, &weights, spec.scale, None),
                    KohyaLora => loader::apply_lora_kohya(
                        &mut host,
                        &weights,
                        spec.scale,
                        &BTreeMap::from([("p".into(), "p".into())]),
                    ),
                    PeftLokr => loader::apply_lokr(&mut host, &weights, spec.scale),
                    ThirdpartyLokr => {
                        loader::apply_lokr_thirdparty(&mut host, &weights, spec.scale)
                    }
                    ThirdpartyLoha => {
                        loader::apply_loha_thirdparty(&mut host, &weights, spec.scale)
                    }
                    LoraKindMismatch => panic!("literal native fixture family"),
                }
                .unwrap();
                weights.materialize_accessed().unwrap();
                assert_eq!(
                    report.applied, 1,
                    "independent actual native successful push: {name}"
                );
                let actual_keys = weights
                    .accessed_entries()
                    .into_iter()
                    .map(|(key, _)| key)
                    .collect::<BTreeSet<_>>();
                let expected_keys = entries
                    .iter()
                    .map(|(key, _, _)| (*key).to_owned())
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    actual_keys, expected_keys,
                    "actual accessed source set includes shadow/alpha factors: {name}"
                );
                let actual = host
                    .lin
                    .adapters()
                    .iter()
                    .map(AdapterFacts::of)
                    .collect::<Vec<_>>();
                let mut matched = false;
                visit_configured_adapter_source_cases(std::slice::from_ref(&spec),&["p".into()],Dtype::Bfloat16,
                    &mut |target| Ok((target=="p").then(|| ("p".into(),base.clone()))), &mut |case| {
                        if !matches!(case.outcome,AdapterSourceOutcome::Complete) { return Ok(()); }
                        let endpoints = case.prefix.retained.iter().map(|(_,facts)| endpoint(facts)).collect::<Vec<_>>();
                        if endpoints != actual.iter().map(endpoint).collect::<Vec<_>>() { return Ok(()); }
                        let family = case.prefix.work.iter().find_map(|event| match event { AdapterSourceWork::Parse(family) if family.stop==AdapterParseStop::End => Some(family), _=>None }).expect("complete actual parse family");
                        let visits = native_order.iter().map(|key| family.actions.iter().find(|a| &a.key==key).expect("actual native key must be in source action table").id).collect::<Vec<_>>();
                        let requires = native_order.iter().filter(|key| actual_keys.contains(*key)).map(|key| family.actions.iter().find(|a| &a.key==key).unwrap().id).collect::<Vec<_>>();
                        let alpha_attempts = native_order.iter().filter(|key| key.ends_with(".alpha")).map(|key| family.actions.iter().find(|a| &a.key==key).unwrap().id).collect();
                        assert!(family.accepts(&ObservedAdapterParsePrefix { visits,requires,alpha_attempts,alpha_comparisons:Vec::new() }),"actual native observed parse prefix: {name}");
                        assert!(case.prefix.work.iter().any(|event| matches!(event,AdapterSourceWork::MaterializeAccessedAttempt { keys,.. } if keys==&actual_keys)),"actual raw materialization set: {name}");
                        let mut ledger=Ledger::default();
                        ledger.apply(&case.prefix)?;
                        if name=="tp-full-structured-samewidth" {
                            // Independent actual native facts: FULL raww1 F16 -> packed endpoint
                            // BF16. Equal128B widths cannot turn this native conversion into a clone.
                            assert_eq!(weights.require("p.lokr_w1").unwrap().dtype(),Dtype::Float16);
                            let Adapter::LokrStructured {factors}=&host.lin.adapters()[0] else {panic!("actual packed full-factor native family");};
                            assert_eq!((factors.w1.shape(),factors.w1.dtype()),(&[8,8][..],Dtype::Bfloat16));
                            assert_eq!((factors.w2.shape(),factors.w2.dtype()),(&[8,8][..],Dtype::Bfloat16));
                            let work=case.prefix.work.iter().find_map(|event|match event {
                                AdapterSourceWork::Constructor(work) if work.probe&&work.retained.is_some()=>Some(work),_=>None
                            }).expect("actual retained packed constructor");
                            let prepared=work.prepared.iter().filter(|tensor|matches!(tensor.origin,AdapterDeclaredOrigin::DeferredCreation))
                                .map(|tensor|(tensor.facts.shape.clone(),tensor.facts.dtype,tensor.facts.bytes())).collect::<Vec<_>>();
                            // Native build_lokr_factors, full branch: castw2 BF16->F32, multiply
                            // scalarF32, castw2 F32->BF16, then castw1 F16->BF16. No payload read,
                            // no materialize-adapter substitution, and no physical-disjointness claim.
                            assert_eq!(prepared,vec![(vec![8,8],Dtype::Float32,256),(vec![8,8],Dtype::Float32,256),
                                (vec![8,8],Dtype::Bfloat16,128),(vec![8,8],Dtype::Bfloat16,128)],
                                "NATIVE-SAME-WIDTH-CAST full packed factor conversion work/order cannot be hidden by normalized endpoint dtype");
                            assert_eq!(ledger.constructors.values().copied().collect::<Vec<_>>(),vec![(4,768)]);
                            assert_eq!(ledger.held()?,260,"raw128+raw128+immediateF32scalar4 parents remain positive");
                        }
                        matched=true;
                        Ok(())
                    }).unwrap();
                assert!(matched,"NATIVE-ADAPTER-SOURCE: independently installed endpoint/disabled state lacks source case: {name}");
                mask |= 1 << index;
                eprintln!("ZIMAGE_ADAPTER_SOURCE_FACTORY_CASE_V1 index={index} name={name}");
            }
            // Actual native promoted-exact Matmul refusal, not a setup failure. Compare its
            // source attempted prefix independently: two I32 operands, no cast/output prepared.
            let path = tensor_file(
                dir.path(),
                "native-exact-refusal",
                &[
                    ("p.lokr_w1_a", &[8, 2], "I32"),
                    ("p.lokr_w1_b", &[2, 8], "I32"),
                    ("p.lokr_w2", &[8, 8], "BF16"),
                ],
                serde_json::json!({}),
            );
            let weights = Weights::from_file(&path).unwrap();
            let mut host = Host {
                lin: AdaptableLinear::dense(mlx_rs::Array::zeros::<f32>(&[64, 64]).unwrap(), None),
            };
            let base = LinearFacts::of(&host.lin);
            let actual_error = loader::apply_lokr_thirdparty(&mut host, &weights, 1.0).unwrap_err();
            assert!(
                actual_error.to_string().contains("inexact"),
                "independent actual native refusal site: {actual_error}"
            );
            let spec = AdapterSpec {
                path,
                kind: AdapterKind::Lora,
                scale: 1.0,
                pass_scales: None,
                moe_expert: None,
            };
            let mut compared = false;
            visit_configured_adapter_source_cases(&[spec],&["p".into()],Dtype::Bfloat16,
                &mut |_| Ok(Some(("p".into(),base.clone()))),&mut |case| {
                    if !matches!(&case.outcome,AdapterSourceOutcome::Refused(mlx_gen::adapters::loader::source_facts::AdapterSourceRefusal::Constructor { .. })) { return Ok(()); }
                    let work=case.prefix.work.iter().find_map(|event| match event { AdapterSourceWork::Constructor(work)=>Some(work),_=>None }).expect("native failed constructor must preserve attempted work");
                    assert_eq!(work.operations.len(),1,"NATIVE-FAILED-PREFIX actual attempted Matmul, no later cast/work");
                    assert_eq!(work.operations[0].operation,mlx_gen::adapters::loader::source_facts::AdapterConstructorOperation::Matmul);
                    assert!(!work.operations[0].completed);
                    assert_eq!(work.operations[0].operands.iter().map(|t|(t.shape.clone(),t.dtype)).collect::<Vec<_>>(),
                        vec![(vec![8,2],Dtype::Int32),(vec![2,8],Dtype::Int32)]);
                    assert!(work.prepared.is_empty(),"failed descriptor operator must not become successful allocation work");
                    compared=true;
                    Ok(())
                }).unwrap();
            assert!(
                compared,
                "actual native failed operator lacks source refusal comparison"
            );
            eprintln!("ZIMAGE_ADAPTER_SOURCE_FAILED_PREFIX_COMPARED_V1 cases=1");
        } // ALL native/file/host/adapter owners drop before final arm. No release-size claim.
        assert_eq!(
            mask, 16383,
            "all fourteen authored native factory source fixtures must compare"
        );
        eprintln!("ZIMAGE_ADAPTER_SOURCE_FACTORY_BODY_COMPLETED_V1 cases=14 fullmask=16383");
        let _ = completion.replace(Completion);
    }
}
