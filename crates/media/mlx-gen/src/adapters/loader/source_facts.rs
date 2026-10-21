//! Specialized source facts for the UNFUSED autoprefix loader, not a tensor graph executor.
//!
//! No native Arrays, tensor scalar/factor reads, cache queries or runtime caps occur here.
//! Existing semantic-v1 witnesses and native parser policies remain separate. Native-unordered
//! families are conservative contract supersets, not observed HashMap traversal witnesses.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use mlx_rs::Dtype;

use super::super::facts::{native_header_dtype, AdapterFacts, TensorFacts};
use super::super::LinearFacts;
use crate::gen_core::weightsmeta::{self as wm, SafetensorsTensorHeader};
use crate::runtime::AdapterSpec;
use crate::{Error, Result};

/// Relative to THIS visitor/load scope; repeated identical literal paths are separate epochs.
/// This denotes source factories, not disjoint native Data or physical allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterFileEpoch(pub usize);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterSourceKey {
    pub epoch: AdapterFileEpoch,
    pub key: String,
}

/// An action identity is distinct from its source key. A parse read, an alpha cast/read and a later
/// raw materialization may name the SAME source but are not interchangeable work attempts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterActionId {
    pub epoch: AdapterFileEpoch,
    pub ordinal: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseActionKind {
    Factor {
        group: String,
        role: String,
        claim: Option<super::LoraFactorRole>,
    },
    Alpha {
        group: String,
        empty: bool,
    },
    UnmatchedKohya {
        stem: String,
    },
    Ignored,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterParseAction {
    pub id: AdapterActionId,
    pub key: String,
    pub kind: ParseActionKind,
}

/// The actual generic loader stops on the SECOND factor role claim BEFORE requiring that key;
/// alpha validation/read errors instead occur AFTER requiring the alpha key. This distinction is
/// retained even though ordinary raw factor reads merely clone lazy source handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterParseStop {
    End,
    DuplicateRole {
        first: AdapterActionId,
        second: AdapterActionId,
    },
    EmptyAlpha {
        action: AdapterActionId,
    },
    AlphaConflict {
        first: AdapterActionId,
        second: AdapterActionId,
    },
    AlphaReadFailed {
        action: AdapterActionId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedAdapterParsePrefix {
    /// Key-action visits, including skipped keys and the failing second role claim.
    pub visits: Vec<AdapterActionId>,
    /// Successful `Weights::require` calls in actual order, not a provenance key set.
    pub requires: Vec<AdapterActionId>,
    /// Full nonempty alpha F32 conversion/read ATTEMPTS, one per visited alpha action.
    pub alpha_attempts: Vec<AdapterActionId>,
    /// Actual native comparison OUTCOMES, not scalar values or pricing reads.
    pub alpha_comparisons: Vec<(AdapterActionId, AdapterActionId, bool)>,
}

/// A WHOLE conservative native-unordered-contract family, not one sorted traversal and not an
/// assertion that every permutation is realizable by a particular HashMap seed. `actions` is a
/// key-indexed action table only. Order/frontier/multiplicity constraints live in `accepts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeUnorderedParsePrefix {
    pub epoch: AdapterFileEpoch,
    pub actions: Vec<AdapterParseAction>,
    pub must_visit: BTreeSet<AdapterActionId>,
    pub may_visit: BTreeSet<AdapterActionId>,
    pub stop: AdapterParseStop,
    pub generic_lora_alpha_conflicts: bool,
}

impl NativeUnorderedParsePrefix {
    pub fn accepts(&self, observed: &ObservedAdapterParsePrefix) -> bool {
        let table = self
            .actions
            .iter()
            .map(|a| (a.id, a))
            .collect::<BTreeMap<_, _>>();
        let mut visited = BTreeSet::new();
        let mut claimed = BTreeMap::<(String, super::LoraFactorRole), AdapterActionId>::new();
        let mut required = Vec::new();
        let mut alpha_attempts = Vec::new();
        let mut stopped = false;
        let mut last_alpha = BTreeMap::<String, AdapterActionId>::new();
        let mut comparisons = Vec::new();
        for (position, id) in observed.visits.iter().copied().enumerate() {
            if stopped || !visited.insert(id) || !self.may_visit.contains(&id) {
                return false;
            }
            let Some(action) = table.get(&id) else {
                return false;
            };
            if id.epoch != self.epoch {
                return false;
            }
            let terminal = position + 1 == observed.visits.len();
            match &action.kind {
                ParseActionKind::Factor { group, claim, .. } => {
                    if let Some(role) = claim {
                        if let Some(first) = claimed.insert((group.clone(), *role), id) {
                            if self.stop != (AdapterParseStop::DuplicateRole { first, second: id })
                                || !terminal
                            {
                                return false;
                            }
                            stopped = true;
                            continue; // Native claim rejects BEFORE require(second).
                        }
                    }
                    required.push(id);
                }
                ParseActionKind::Alpha { empty, .. } => {
                    required.push(id);
                    if *empty {
                        if self.stop != (AdapterParseStop::EmptyAlpha { action: id }) || !terminal {
                            return false;
                        }
                        stopped = true;
                        continue;
                    }
                    alpha_attempts.push(id);
                    if self.generic_lora_alpha_conflicts
                        && !matches!(self.stop, AdapterParseStop::AlphaReadFailed { action } if action == id)
                    {
                        let ParseActionKind::Alpha { group, .. } = &action.kind else {
                            return false;
                        };
                        if let Some(first) = last_alpha.insert(group.clone(), id) {
                            let agrees = !matches!(self.stop, AdapterParseStop::AlphaConflict { first: a, second: b } if a == first && b == id);
                            comparisons.push((first, id, agrees));
                        }
                    }
                    match self.stop {
                        AdapterParseStop::AlphaConflict { first, second } if second == id => {
                            let Some(first_action) = table.get(&first) else {
                                return false;
                            };
                            let same_group = matches!((&first_action.kind, &action.kind),
                                (ParseActionKind::Alpha { group: a, .. }, ParseActionKind::Alpha { group: b, .. }) if a == b);
                            if !same_group
                                || !visited.contains(&first)
                                || first == second
                                || !terminal
                            {
                                return false;
                            }
                            stopped = true;
                        }
                        AdapterParseStop::AlphaReadFailed { action } if action == id => {
                            if !terminal {
                                return false;
                            }
                            stopped = true;
                        }
                        _ => {}
                    }
                }
                ParseActionKind::UnmatchedKohya { .. } | ParseActionKind::Ignored => {}
            }
        }
        if !self.must_visit.is_subset(&visited) {
            return false;
        }
        match self.stop {
            AdapterParseStop::End => {
                if stopped || visited.len() != self.actions.len() {
                    return false;
                }
            }
            _ if !stopped => return false,
            _ => {}
        }
        // These are VECTORS deliberately: set membership cannot validate attempted work order or
        // multiplicity, and a read after the stop or a repeated alpha attempt is not admissible.
        required == observed.requires
            && alpha_attempts == observed.alpha_attempts
            && comparisons == observed.alpha_comparisons
    }

    pub fn possible_requires(&self) -> BTreeSet<String> {
        self.actions.iter().filter(|a| self.may_visit.contains(&a.id)).filter_map(|a| match &a.kind {
            ParseActionKind::Factor { .. } => {
                if matches!(self.stop, AdapterParseStop::DuplicateRole { second, .. } if second == a.id) {
                    None
                } else { Some(a.key.clone()) }
            }
            ParseActionKind::Alpha { .. } => Some(a.key.clone()),
            _ => None,
        }).collect()
    }

    pub fn possible_alpha_attempts(&self) -> Vec<AdapterActionId> {
        self.actions
            .iter()
            .filter(|a| self.may_visit.contains(&a.id))
            .filter(|a| matches!(a.kind, ParseActionKind::Alpha { empty: false, .. }))
            .map(|a| a.id)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterFactorRecipe {
    Full(String),
    Product { a: String, b: String },
    Tucker { core: String, a: String, b: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterSourceRecipe {
    Lora {
        down: String,
        up: String,
        alpha_fold: bool,
    },
    Lokr {
        left: AdapterFactorRecipe,
        right: AdapterFactorRecipe,
        structured: bool,
    },
    Loha {
        a1: String,
        b1: String,
        a2: String,
        b2: String,
        tucker_pair: Option<(String, String)>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterSourceCondition {
    AlphaAgreement {
        epoch: AdapterFileEpoch,
        first: String,
        second: String,
        agrees: bool,
    },
    PackedCoefficient {
        epoch: AdapterFileEpoch,
        group: String,
        class: PackedCoefficientClass,
    },
    MaterializationBudget {
        epoch: AdapterFileEpoch,
        fits: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedCoefficientClass {
    Zero,
    FiniteNonzero,
    Nonfinite,
}

#[derive(Clone, Debug)]
pub struct AdapterFileSources {
    pub epoch: AdapterFileEpoch,
    pub headers: Arc<BTreeMap<String, TensorFacts>>,
    pub format: super::UnfusedAdapterFormat,
}

/// New shapes/dtypes can be prepared without backing. A view/full same-dtype factor aliases a
/// source root; an actual conversion remains distinct even when F16/BF16 have equal byte widths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterDeclaredOrigin {
    Source(AdapterSourceKey),
    DeferredCreation,
    ImmediateScalar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterDeclaredTensor {
    pub facts: TensorFacts,
    pub origin: AdapterDeclaredOrigin,
}

/// Attempts are recorded BEFORE fallible factory validation. Prepared descriptors alone cannot
/// distinguish a native rejected operator from an operator never reached after an earlier failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterConstructorOperation {
    Transpose,
    Cast(Dtype),
    Matmul,
    Kron,
    Multiply,
    Scalar,
    Reshape(Vec<i32>),
    Tucker { loha_order: bool },
    Sum(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct AdapterConstructorAttempt {
    pub operation: AdapterConstructorOperation,
    pub operands: Vec<TensorFacts>,
    pub completed: bool,
}

#[derive(Clone, Debug)]
pub struct AdapterConstructorWork {
    pub epoch: AdapterFileEpoch,
    pub group: String,
    pub recipe: Option<AdapterSourceRecipe>,
    /// A packed failed structured probe and later fallback are separate action identities.
    pub attempt: usize,
    pub probe: bool,
    pub operands: Vec<AdapterSourceKey>,
    pub prepared: Vec<AdapterDeclaredTensor>,
    pub operations: Vec<AdapterConstructorAttempt>,
    /// Endpoint payloads after actual transpose/fold/cast selection, not an allocation witness.
    pub retained: Option<AdapterFacts>,
}

#[derive(Clone, Debug)]
pub enum AdapterSourceWork {
    FileMapAttempt(AdapterFileEpoch),
    FileMap(AdapterFileEpoch),
    Parse(NativeUnorderedParsePrefix),
    Constructor(Box<AdapterConstructorWork>),
    BudgetProbe {
        epoch: AdapterFileEpoch,
        projected_raw_bytes: u64,
    },
    Install {
        epoch: AdapterFileEpoch,
        retained_index: usize,
        canonical_target: String,
        constructor_attempt: usize,
    },
    MaterializeAccessedAttempt {
        epoch: AdapterFileEpoch,
        keys: BTreeSet<String>,
    },
    MaterializeAccessedComplete(AdapterFileEpoch),
}

#[derive(Clone, Debug, Default)]
pub struct AdapterSourceCounts {
    pub applied: usize,
    /// Z's strict NON-diff entrypoint passes zero. Never infer this from a retained-slot count or
    /// silently transplant this policy into SDXL's separate merge/diff entrypoints.
    pub pre_applied: usize,
    pub unmatched: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AdapterSourcePrefix {
    pub files: Vec<AdapterFileSources>,
    pub work: Vec<AdapterSourceWork>,
    /// GLOBAL append order across input files and native ordered groups, not a target-key map.
    pub retained: Vec<(String, AdapterFacts)>,
    pub counts: AdapterSourceCounts,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterSourceRefusal {
    Header {
        file: usize,
        message: String,
    },
    KindMismatch {
        file: usize,
    },
    Parse {
        file: usize,
        stop: AdapterParseStop,
    },
    Constructor {
        file: usize,
        group: String,
        message: String,
    },
    MaterializationBudget {
        file: usize,
    },
    Materialization {
        file: usize,
    },
    AggregateNoMatch,
    AggregateUnmatched,
}

#[derive(Clone, Debug)]
pub enum AdapterSourceOutcome {
    Complete,
    Refused(AdapterSourceRefusal),
}

#[derive(Clone, Debug)]
pub struct AdapterSourceCase {
    pub case_id: u64,
    pub conditions: Vec<AdapterSourceCondition>,
    pub prefix: AdapterSourcePrefix,
    pub outcome: AdapterSourceOutcome,
}

#[derive(Clone, Debug, Default)]
pub struct AdapterSourceSummary {
    pub delivered_cases: u64,
    pub complete_cases: u64,
    pub refused_cases: u64,
    pub enumeration_complete: bool,
}

#[derive(Debug)]
pub enum AdapterSourceAbortKind {
    Consumer,
    Enumerator,
}

/// A projection callback's answer for one resolved dotted path: the canonical target and its
/// linear facts, or `None` when the path has no projection.
pub type AdapterProjection = Result<Option<(String, LinearFacts)>>;

/// Every fallible visitor stage returns this boxed: the failure carries the whole committed
/// prefix, and a by-value `Err` would make every intermediate `Result` that large.
pub type AdapterSourceResult<T> = std::result::Result<T, Box<AdapterSourceFailure>>;

#[derive(Debug)]
pub struct AdapterSourceFailure {
    pub error: Error,
    pub prefix: AdapterSourcePrefix,
    /// Some means callback was ENTERED, even when it returned Err. This prefix is provenance,
    /// NEVER a replay delta or a second charging channel.
    pub attempted_case_id: Option<u64>,
    pub kind: AdapterSourceAbortKind,
}

struct Delivery<'a> {
    consume: &'a mut dyn FnMut(&AdapterSourceCase) -> Result<()>,
    summary: AdapterSourceSummary,
    next_id: u64,
}

impl Delivery<'_> {
    fn emit(
        &mut self,
        prefix: &AdapterSourcePrefix,
        conditions: &[AdapterSourceCondition],
        outcome: AdapterSourceOutcome,
    ) -> AdapterSourceResult<()> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            Box::new(AdapterSourceFailure {
                error: Error::Msg("adapter source case ordinal overflow".into()),
                prefix: prefix.clone(),
                attempted_case_id: None,
                kind: AdapterSourceAbortKind::Enumerator,
            })
        })?;
        let complete = matches!(outcome, AdapterSourceOutcome::Complete);
        let case = AdapterSourceCase {
            case_id: id,
            conditions: conditions.to_vec(),
            prefix: prefix.clone(),
            outcome,
        };
        (self.consume)(&case).map_err(|error| {
            Box::new(AdapterSourceFailure {
                error,
                prefix: prefix.clone(),
                attempted_case_id: Some(id),
                kind: AdapterSourceAbortKind::Consumer,
            })
        })?;
        self.summary.delivered_cases += 1;
        if complete {
            self.summary.complete_cases += 1;
        } else {
            self.summary.refused_cases += 1;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct RawGroup {
    raw: String,
    factors: BTreeMap<String, String>,
    alpha: Vec<String>,
}

struct FileContext {
    source: AdapterFileSources,
    groups: Vec<RawGroup>,
    actions: Vec<AdapterParseAction>,
    reads: BTreeSet<String>,
    unresolved: Vec<String>,
    metadata: BTreeMap<String, String>,
    lora_meta: Option<wm::LoraAdapterMeta>,
}

fn tensor(header: &SafetensorsTensorHeader) -> Result<TensorFacts> {
    let shape = super::semantic_header_shape(header)?;
    let mut stride = 1usize;
    let mut strides = vec![0; shape.len()];
    for index in (0..shape.len()).rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(shape[index].max(0) as usize)
            .ok_or_else(|| Error::Msg("adapter source stride overflow".into()))?;
    }
    Ok(TensorFacts {
        shape,
        dtype: native_header_dtype(header.dtype)?,
        strides,
    })
}

fn file_context(
    spec: &AdapterSpec,
    ordinal: usize,
    table: &BTreeMap<String, String>,
) -> Result<FileContext> {
    let layout = wm::safetensors_file_tensor_locations(&spec.path).map_err(Error::from)?;
    let metadata = wm::safetensors_file_metadata(&spec.path)
        .map_err(Error::from)?
        .into_iter()
        .collect::<BTreeMap<String, String>>();
    let raw_headers = layout
        .tensors
        .iter()
        .map(|l| l.header.clone())
        .collect::<Vec<_>>();
    let mut headers = BTreeMap::new();
    for header in &raw_headers {
        headers.insert(header.name.clone(), tensor(header)?);
    }
    let format = super::classify_unfused_adapter_format(
        headers.keys().map(String::as_str),
        metadata.get("networkType").map(String::as_str),
        spec.kind,
    );
    let epoch = AdapterFileEpoch(ordinal);
    let prefix = wm::detect_lora_prefix(headers.keys().map(String::as_str)).unwrap_or("");
    let lora = matches!(
        format,
        super::UnfusedAdapterFormat::PeftLora | super::UnfusedAdapterFormat::KohyaLora
    );
    let kohya = format == super::UnfusedAdapterFormat::KohyaLora;
    let peft_lokr = format == super::UnfusedAdapterFormat::PeftLokr;
    let loha = format == super::UnfusedAdapterFormat::ThirdpartyLoha;
    let suffixes = if loha {
        &wm::LOHA_TP_SUFFIXES[..]
    } else if peft_lokr {
        &wm::LOKR_SUFFIXES[..]
    } else {
        &wm::LOKR_TP_SUFFIXES[..]
    };
    let mut groups = BTreeMap::<String, RawGroup>::new();
    let mut actions = Vec::new();
    let mut reads = BTreeSet::new();
    let mut unresolved = BTreeSet::new();
    // Sorted STORAGE only; no sorted execution history is asserted or chosen.
    for (key, facts) in &headers {
        let mut kind = ParseActionKind::Ignored;
        if lora {
            let parsed = if kohya {
                super::kohya_factor_key(key).map(|(stem, role)| {
                    let path = table.get(stem).cloned();
                    (
                        path,
                        stem.to_owned(),
                        match role {
                            super::KohyaRole::Down => Some(super::LoraFactorRole::Down),
                            super::KohyaRole::Up => Some(super::LoraFactorRole::Up),
                            super::KohyaRole::Alpha => None,
                        },
                    )
                })
            } else if let Some((path, role)) = super::peft_lora_factor(key, prefix) {
                Some((Some(path.into()), String::new(), Some(role)))
            } else {
                key.strip_prefix(prefix)
                    .and_then(|k| k.strip_suffix(".alpha"))
                    .or_else(|| key.strip_suffix(".alpha"))
                    .map(|path| (Some(path.into()), String::new(), None))
            };
            if let Some((path, stem, role)) = parsed {
                if let Some(path) = path {
                    let group = groups.entry(path.clone()).or_insert_with(|| RawGroup {
                        raw: path.clone(),
                        ..RawGroup::default()
                    });
                    reads.insert(key.clone());
                    if let Some(role) = role {
                        let name = if role == super::LoraFactorRole::Down {
                            "down"
                        } else {
                            "up"
                        };
                        // Duplicates remain in the ACTION table. A success branch is impossible;
                        // no arbitrary last winner is selected for the generic loader.
                        group
                            .factors
                            .entry(name.into())
                            .or_insert_with(|| key.clone());
                        kind = ParseActionKind::Factor {
                            group: path,
                            role: name.into(),
                            claim: Some(role),
                        };
                    } else {
                        group.alpha.push(key.clone());
                        kind = ParseActionKind::Alpha {
                            group: path,
                            empty: facts.elements() == 0,
                        };
                    }
                } else {
                    unresolved.insert(stem.clone());
                    kind = ParseActionKind::UnmatchedKohya { stem };
                }
            }
        } else if !peft_lokr && key.ends_with(".alpha") {
            let path = key.strip_suffix(".alpha").expect("suffix").to_owned();
            groups
                .entry(path.clone())
                .or_insert_with(|| RawGroup {
                    raw: path.clone(),
                    ..RawGroup::default()
                })
                .alpha
                .push(key.clone());
            reads.insert(key.clone());
            kind = ParseActionKind::Alpha {
                group: path,
                empty: facts.elements() == 0,
            };
        } else if let Some((path, role)) = wm::split_factor_key(key, suffixes) {
            let path = path.to_owned();
            groups
                .entry(path.clone())
                .or_insert_with(|| RawGroup {
                    raw: path.clone(),
                    ..RawGroup::default()
                })
                .factors
                .insert(role.into(), key.clone());
            reads.insert(key.clone());
            kind = ParseActionKind::Factor {
                group: path,
                role: role.into(),
                claim: None,
            };
        }
        actions.push(AdapterParseAction {
            id: AdapterActionId {
                epoch,
                ordinal: actions.len(),
            },
            key: key.clone(),
            kind,
        });
    }
    let lora_meta = if format == super::UnfusedAdapterFormat::PeftLora {
        Some(wm::LoraAdapterMeta::from_metadata(
            metadata
                .get(wm::LORA_ADAPTER_METADATA_KEY)
                .map(String::as_str),
        ))
        .flatten()
    } else {
        None
    };
    Ok(FileContext {
        source: AdapterFileSources {
            epoch,
            headers: Arc::new(headers),
            format,
        },
        groups: groups.into_values().collect(),
        actions,
        reads,
        unresolved: unresolved.into_iter().collect(),
        metadata,
        lora_meta,
    })
}

fn family(ctx: &FileContext, stop: AdapterParseStop) -> NativeUnorderedParsePrefix {
    let mut must_visit = BTreeSet::new();
    match stop {
        AdapterParseStop::End => must_visit.extend(ctx.actions.iter().map(|a| a.id)),
        AdapterParseStop::DuplicateRole { first, second }
        | AdapterParseStop::AlphaConflict { first, second } => {
            must_visit.insert(first);
            must_visit.insert(second);
        }
        AdapterParseStop::EmptyAlpha { action } | AdapterParseStop::AlphaReadFailed { action } => {
            must_visit.insert(action);
        }
    }
    let may_visit = ctx
        .actions
        .iter()
        .filter(|a| match a.kind {
            ParseActionKind::Alpha { empty: true, .. } => {
                matches!(stop, AdapterParseStop::EmptyAlpha { action } if action == a.id)
            }
            _ => true,
        })
        .map(|a| a.id)
        .collect();
    NativeUnorderedParsePrefix {
        epoch: ctx.source.epoch,
        actions: ctx.actions.clone(),
        must_visit,
        may_visit,
        stop,
        generic_lora_alpha_conflicts: ctx.source.format == super::UnfusedAdapterFormat::PeftLora,
    }
}

fn full_tensor(ctx: &FileContext, key: &str) -> Result<TensorFacts> {
    ctx.source
        .headers
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Msg(format!("missing adapter source {key}")))
}

fn required_key(group: &RawGroup, role: &str) -> Result<String> {
    group
        .factors
        .get(role)
        .cloned()
        .ok_or_else(|| Error::Msg(format!("missing adapter factor {role}")))
}

fn inexact(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 | Dtype::Complex64
    )
}

/// Pure copy of pinned native7a1d4f dtype.cpp's 14-type promotion table. This metadata
/// traversal must not call a native factory/query just to determine a descriptor's dtype.
fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
    use Dtype::*;
    let index = |dtype| match dtype {
        Bool => 0,
        Uint8 => 1,
        Uint16 => 2,
        Uint32 => 3,
        Uint64 => 4,
        Int8 => 5,
        Int16 => 6,
        Int32 => 7,
        Int64 => 8,
        Float16 => 9,
        Float32 => 10,
        Float64 => 11,
        Bfloat16 => 12,
        Complex64 => 13,
    };
    const TABLE: [[Dtype; 14]; 14] = [
        [
            Bool, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint8, Uint8, Uint16, Uint32, Uint64, Int16, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint16, Uint16, Uint16, Uint32, Uint64, Int32, Int32, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint32, Uint32, Uint32, Uint32, Uint64, Int64, Int64, Int64, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Uint64, Uint64, Uint64, Uint64, Uint64, Float32, Float32, Float32, Float32, Float16,
            Float32, Float64, Bfloat16, Complex64,
        ],
        [
            Int8, Int16, Int32, Int64, Float32, Int8, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int16, Int16, Int32, Int64, Float32, Int16, Int16, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int32, Int32, Int32, Int64, Float32, Int32, Int32, Int32, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Int64, Int64, Int64, Int64, Float32, Int64, Int64, Int64, Int64, Float16, Float32,
            Float64, Bfloat16, Complex64,
        ],
        [
            Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16,
            Float16, Float32, Float64, Float32, Complex64,
        ],
        [
            Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
            Float32, Float32, Float64, Float32, Complex64,
        ],
        [
            Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
            Float64, Float64, Float64, Float64, Complex64,
        ],
        [
            Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16, Bfloat16,
            Bfloat16, Float32, Float32, Float64, Bfloat16, Complex64,
        ],
        [
            Complex64, Complex64, Complex64, Complex64, Complex64, Complex64, Complex64, Complex64,
            Complex64, Complex64, Complex64, Complex64, Complex64, Complex64,
        ],
    ];
    TABLE[index(a)][index(b)]
}

fn materialized(shape: Vec<i32>, dtype: Dtype) -> TensorFacts {
    TensorFacts {
        shape,
        dtype,
        strides: Vec::new(),
    }
}

fn prepare(
    work: &mut AdapterConstructorWork,
    facts: TensorFacts,
    origin: AdapterDeclaredOrigin,
) -> TensorFacts {
    work.prepared.push(AdapterDeclaredTensor {
        facts: facts.clone(),
        origin,
    });
    facts
}

fn attempt(
    work: &mut AdapterConstructorWork,
    operation: AdapterConstructorOperation,
    operands: &[TensorFacts],
) -> usize {
    let index = work.operations.len();
    work.operations.push(AdapterConstructorAttempt {
        operation,
        operands: operands.to_vec(),
        completed: false,
    });
    index
}

fn cast(work: &mut AdapterConstructorWork, input: TensorFacts, dtype: Dtype) -> TensorFacts {
    let index = attempt(
        work,
        AdapterConstructorOperation::Cast(dtype),
        std::slice::from_ref(&input),
    );
    let output = if input.dtype == dtype {
        input
    } else {
        prepare(
            work,
            materialized(input.shape, dtype),
            AdapterDeclaredOrigin::DeferredCreation,
        )
    };
    work.operations[index].completed = true;
    output
}

fn matmul(
    work: &mut AdapterConstructorWork,
    a: TensorFacts,
    b: TensorFacts,
) -> Result<TensorFacts> {
    let index = attempt(
        work,
        AdapterConstructorOperation::Matmul,
        &[a.clone(), b.clone()],
    );
    // Native validates rank and contraction before promotion; batch broadcasting is LATER.
    if a.shape.is_empty() || b.shape.is_empty() {
        return Err(Error::Msg("adapter matmul cannot use a scalar".into()));
    }
    let contract_a = a.shape[a.shape.len() - 1];
    let contract_b = b.shape[if b.shape.len() == 1 {
        0
    } else {
        b.shape.len() - 2
    }];
    if contract_a != contract_b {
        return Err(Error::Msg(
            "adapter matmul contraction dimensions disagree".into(),
        ));
    }
    let dtype = promote_dtype(a.dtype, b.dtype);
    // Pinned native ops.cpp: rank/contraction validation, then PAIR promotion, then inexact gate.
    // No at_least_float rule: U64+signed can promoteF32, while I32+I32 staysI32 and rejects.
    if !inexact(dtype) {
        return Err(Error::Msg(
            "native matmul promoted dtype is not inexact".into(),
        ));
    }
    let a = if a.dtype == dtype {
        a
    } else {
        cast(work, a, dtype)
    };
    let b = if b.dtype == dtype {
        b
    } else {
        cast(work, b, dtype)
    };
    let shape = super::semantic_matmul_shape(&a.shape, &b.shape)?;
    let output = prepare(
        work,
        materialized(shape, dtype),
        AdapterDeclaredOrigin::DeferredCreation,
    );
    work.operations[index].completed = true;
    Ok(output)
}

fn multiply(
    work: &mut AdapterConstructorWork,
    a: TensorFacts,
    b: TensorFacts,
) -> Result<TensorFacts> {
    let index = attempt(
        work,
        AdapterConstructorOperation::Multiply,
        &[a.clone(), b.clone()],
    );
    let dtype = promote_dtype(a.dtype, b.dtype);
    // Native multiply prepares astype inputs BEFORE broadcast validation, including refusal cases.
    let a = cast(work, a, dtype);
    let b = cast(work, b, dtype);
    let shape = super::semantic_broadcast_shape(&a.shape, &b.shape)?;
    let output = prepare(
        work,
        materialized(shape, dtype),
        AdapterDeclaredOrigin::DeferredCreation,
    );
    work.operations[index].completed = true;
    Ok(output)
}

fn scalar(work: &mut AdapterConstructorWork) -> TensorFacts {
    let index = attempt(work, AdapterConstructorOperation::Scalar, &[]);
    // Existing native constructor makes a strong one-element F32 scalar. No payload is read here.
    let output = prepare(
        work,
        materialized(vec![1], Dtype::Float32),
        AdapterDeclaredOrigin::ImmediateScalar,
    );
    work.operations[index].completed = true;
    output
}

fn reshape(
    work: &mut AdapterConstructorWork,
    input: TensorFacts,
    shape: Vec<i32>,
) -> Result<TensorFacts> {
    let index = attempt(
        work,
        AdapterConstructorOperation::Reshape(shape.clone()),
        std::slice::from_ref(&input),
    );
    super::semantic_reshape_matches(&input.shape, &shape)?;
    let output = if input.shape == shape {
        input
    } else {
        // Possible reshape-copy capacity; never a native Data independence/physical witness.
        prepare(
            work,
            materialized(shape, input.dtype),
            AdapterDeclaredOrigin::DeferredCreation,
        )
    };
    work.operations[index].completed = true;
    Ok(output)
}

fn kron(work: &mut AdapterConstructorWork, a: TensorFacts, b: TensorFacts) -> Result<TensorFacts> {
    let operation = attempt(
        work,
        AdapterConstructorOperation::Kron,
        &[a.clone(), b.clone()],
    );
    if a.elements() == 0 || b.elements() == 0 {
        return Err(Error::Msg("native kron rejects empty operands".into()));
    }
    let rank = a.shape.len().max(b.shape.len());
    let mut a_shape = vec![1; 2 * rank];
    let mut b_shape = vec![1; 2 * rank];
    let mut output = vec![1i32; rank];
    for (ax, &dim) in a.shape.iter().enumerate() {
        let index = rank - a.shape.len() + ax;
        a_shape[2 * index] = dim;
        output[index] = dim;
    }
    for (ax, &dim) in b.shape.iter().enumerate() {
        let index = rank - b.shape.len() + ax;
        b_shape[2 * index + 1] = dim;
        output[index] = output[index]
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg("LoKr output dimension overflow".into()))?;
    }
    let a = reshape(work, a, a_shape)?;
    let b = reshape(work, b, b_shape)?;
    let product = multiply(work, a, b)?;
    let result = reshape(work, product, output)?;
    work.operations[operation].completed = true;
    Ok(result)
}

fn sum_dtype(dtype: Dtype) -> Dtype {
    match dtype {
        Dtype::Bool | Dtype::Int8 | Dtype::Int16 | Dtype::Int32 => Dtype::Int32,
        Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 => Dtype::Uint32,
        other => other,
    }
}

#[derive(Clone)]
struct EinsumTerm {
    labels: Vec<char>,
    facts: TensorFacts,
}

fn term_elements(labels: &BTreeSet<char>, dims: &BTreeMap<char, i32>) -> Result<u64> {
    labels.iter().try_fold(1u64, |n, c| {
        n.checked_mul(dims[c] as u64)
            .ok_or_else(|| Error::Msg("einsum source size overflow".into()))
    })
}

fn labels_set(labels: &[char]) -> BTreeSet<char> {
    labels.iter().copied().collect()
}

fn contraction(
    work: &mut AdapterConstructorWork,
    mut terms: Vec<EinsumTerm>,
    output: &[char],
    dims: &BTreeMap<char, i32>,
) -> Result<EinsumTerm> {
    // Native preprocessing reduces input-only axes, preserving LOCAL sizes (not global broadcast
    // sizes) and applying sum's integer promotion before the can_dot decision.
    for index in 0..terms.len() {
        let other = terms
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != index)
            .flat_map(|(_, t)| t.labels.iter().copied())
            .chain(output.iter().copied())
            .collect::<BTreeSet<_>>();
        let axes = terms[index]
            .labels
            .iter()
            .enumerate()
            .filter_map(|(ax, c)| (!other.contains(c)).then_some(ax))
            .collect::<Vec<_>>();
        if !axes.is_empty() {
            let old = terms[index].facts.clone();
            let operation = attempt(
                work,
                AdapterConstructorOperation::Sum(axes.clone()),
                std::slice::from_ref(&old),
            );
            let labels = terms[index]
                .labels
                .iter()
                .enumerate()
                .filter_map(|(ax, c)| (!axes.contains(&ax)).then_some(*c))
                .collect();
            let shape = old
                .shape
                .iter()
                .enumerate()
                .filter_map(|(ax, d)| (!axes.contains(&ax)).then_some(*d))
                .collect();
            terms[index] = EinsumTerm {
                labels,
                facts: prepare(
                    work,
                    materialized(shape, sum_dtype(old.dtype)),
                    AdapterDeclaredOrigin::DeferredCreation,
                ),
            };
            work.operations[operation].completed = true;
        }
    }
    let output_set = labels_set(output);
    let all = terms
        .iter()
        .flat_map(|t| t.labels.iter().copied())
        .collect::<BTreeSet<_>>();
    let contracted = all
        .difference(&output_set)
        .copied()
        .collect::<BTreeSet<_>>();
    if terms.len() == 2 && !contracted.is_empty() {
        // Native batch_tensordot first broadcasts contracting axes, then transpose/reshape, then
        // matmul (which may reject its promoted exact dtype). Keep that prepared refused prefix.
        let batch = terms[0]
            .labels
            .iter()
            .copied()
            .filter(|c| output_set.contains(c) && terms[1].labels.contains(c))
            .collect::<Vec<_>>();
        let a_concat = terms[0]
            .labels
            .iter()
            .copied()
            .filter(|c| output_set.contains(c) && !batch.contains(c))
            .collect::<Vec<_>>();
        let b_concat = terms[1]
            .labels
            .iter()
            .copied()
            .filter(|c| output_set.contains(c) && !batch.contains(c))
            .collect::<Vec<_>>();
        let product = |labels: &[char], term: &EinsumTerm| -> Result<i32> {
            labels.iter().try_fold(1i32, |n, c| {
                let size = if contracted.contains(c) {
                    dims[c]
                } else {
                    term.facts.shape[term
                        .labels
                        .iter()
                        .position(|x| x == c)
                        .expect("local einsum axis")]
                };
                n.checked_mul(size)
                    .ok_or_else(|| Error::Msg("einsum flattened dimension overflow".into()))
            })
        };
        let contract = terms[0]
            .labels
            .iter()
            .copied()
            .filter(|c| contracted.contains(c))
            .collect::<Vec<_>>();
        let mut matrices = Vec::new();
        for (index, term) in terms.iter().enumerate() {
            let mut expanded = term.facts.clone();
            for (ax, c) in term.labels.iter().enumerate() {
                if contracted.contains(c) {
                    expanded.shape[ax] = dims[c];
                }
            }
            // Broadcast itself is a view, not a successful new allocation. The reached reshape
            // below carries its possible copy capacity; rejected matmul creates no output.
            let mut shape = batch
                .iter()
                .map(|c| {
                    term.facts.shape[term
                        .labels
                        .iter()
                        .position(|x| x == c)
                        .expect("batch einsum axis")]
                })
                .collect::<Vec<_>>();
            if index == 0 {
                shape.extend([product(&a_concat, term)?, product(&contract, term)?]);
            } else {
                shape.extend([product(&contract, term)?, product(&b_concat, term)?]);
            }
            matrices.push(reshape(work, expanded, shape)?);
        }
        let result = matmul(work, matrices.remove(0), matrices.remove(0))?;
        let shape = output.iter().map(|c| dims[c]).collect();
        let result = reshape(work, result, shape)?;
        return Ok(EinsumTerm {
            labels: output.to_vec(),
            facts: result,
        });
    }
    // Native naive einsum does NOT pre-promote all inputs. Align local dimensions, multiply in
    // operand order, and reduce last. E.g. I8*I8*F16 has an I8 first product, not an F16 product.
    let labels = all.iter().copied().collect::<Vec<_>>();
    let mut aligned = Vec::new();
    for term in &terms {
        let shape = labels
            .iter()
            .map(|c| {
                term.labels
                    .iter()
                    .position(|x| x == c)
                    .map_or(1, |ax| term.facts.shape[ax])
            })
            .collect::<Vec<_>>();
        // Transpose is a view; reshape can copy, so keep a positive source-capacity bound. This
        // is not an assertion that the native hash-label tie order is our lexical storage order.
        aligned.push(reshape(work, term.facts.clone(), shape)?);
    }
    let mut result = aligned.remove(0);
    for operand in aligned {
        result = multiply(work, result, operand)?;
    }
    if !contracted.is_empty() {
        let axes = labels
            .iter()
            .enumerate()
            .filter_map(|(ax, c)| contracted.contains(c).then_some(ax))
            .collect::<Vec<_>>();
        let operation = attempt(
            work,
            AdapterConstructorOperation::Sum(axes),
            std::slice::from_ref(&result),
        );
        let shape = output
            .iter()
            .map(|c| {
                result.shape[labels
                    .iter()
                    .position(|x| x == c)
                    .expect("output einsum axis")]
            })
            .collect();
        result = prepare(
            work,
            materialized(shape, sum_dtype(result.dtype)),
            AdapterDeclaredOrigin::DeferredCreation,
        );
        work.operations[operation].completed = true;
    } else {
        result.shape = output
            .iter()
            .map(|c| {
                result.shape[labels
                    .iter()
                    .position(|x| x == c)
                    .expect("output einsum axis")]
            })
            .collect();
    }
    Ok(EinsumTerm {
        labels: output.to_vec(),
        facts: result,
    })
}

fn tucker(
    work: &mut AdapterConstructorWork,
    core: TensorFacts,
    a: TensorFacts,
    b: TensorFacts,
    loha_order: bool,
) -> Result<TensorFacts> {
    let operation = attempt(
        work,
        AdapterConstructorOperation::Tucker { loha_order },
        &[core.clone(), a.clone(), b.clone()],
    );
    let result = (|| -> Result<TensorFacts> {
        if core.shape.len() != 4 || a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(Error::Msg(
                "native Tucker requires4Dcore and2Dfactor matrices".into(),
            ));
        }
        let mut terms = if loha_order {
            vec![
                EinsumTerm {
                    labels: "ijhw".chars().collect(),
                    facts: core,
                },
                EinsumTerm {
                    labels: "jr".chars().collect(),
                    facts: b,
                },
                EinsumTerm {
                    labels: "ip".chars().collect(),
                    facts: a,
                },
            ]
        } else {
            vec![
                EinsumTerm {
                    labels: "ijhw".chars().collect(),
                    facts: core,
                },
                EinsumTerm {
                    labels: "ip".chars().collect(),
                    facts: a,
                },
                EinsumTerm {
                    labels: "jr".chars().collect(),
                    facts: b,
                },
            ]
        };
        let output = "prhw".chars().collect::<Vec<_>>();
        let output_set = labels_set(&output);
        let mut dims = BTreeMap::new();
        for term in &terms {
            for (label, &size) in term.labels.iter().zip(&term.facts.shape) {
                let old = dims.entry(*label).or_insert(size);
                if size != 1 && *old != 1 && size != *old {
                    return Err(Error::Msg(
                        "native einsum dimensions cannot broadcast".into(),
                    ));
                }
                *old = (*old).max(size);
            }
        }
        let all = terms
            .iter()
            .flat_map(|t| t.labels.iter().copied())
            .collect::<BTreeSet<_>>();
        let naive_cost = term_elements(&all, &dims)?
            .checked_mul(3)
            .ok_or_else(|| Error::Msg("einsum source cost overflow".into()))?;
        let mut memory_limit = term_elements(&output_set, &dims)?;
        for term in &terms {
            memory_limit = memory_limit.max(term_elements(&labels_set(&term.labels), &dims)?);
        }
        let mut cost_so_far = 0u64;
        // Only the actual three-operand Tucker equations, not an arbitrary einsum/graph framework.
        while terms.len() > 1 {
            let mut candidates = Vec::new();
            for outer in [false, true] {
                for x in 0..terms.len() {
                    for y in x + 1..terms.len() {
                        let left = labels_set(&terms[x].labels);
                        let right = labels_set(&terms[y].labels);
                        if !outer && left.is_disjoint(&right) {
                            continue;
                        }
                        let combined = left.union(&right).copied().collect::<BTreeSet<_>>();
                        let needed = terms
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != x && *i != y)
                            .flat_map(|(_, t)| t.labels.iter().copied())
                            .chain(output.iter().copied())
                            .collect::<BTreeSet<_>>();
                        let result = combined
                            .intersection(&needed)
                            .copied()
                            .collect::<BTreeSet<_>>();
                        let size = term_elements(&result, &dims)?;
                        if size > memory_limit {
                            continue;
                        }
                        let cost = term_elements(&combined, &dims)?
                            .checked_mul(if combined.len() > result.len() { 2 } else { 1 })
                            .ok_or_else(|| Error::Msg("einsum pair source cost overflow".into()))?;
                        if cost_so_far.checked_add(cost).is_none_or(|n| n > naive_cost) {
                            continue;
                        }
                        let removed = i128::from(term_elements(&left, &dims)?)
                            + i128::from(term_elements(&right, &dims)?)
                            - i128::from(size);
                        candidates.push((removed, cost, x, y, result));
                    }
                }
                if !candidates.is_empty() {
                    break;
                }
            }
            if candidates.is_empty() {
                return Ok(contraction(work, terms, &output, &dims)?.facts);
            }
            // Same native greedy preference: greatest removed-size, then least cost, then original
            // pair traversal on exact ties. Metadata label ties do not alter source shapes/dtypes.
            let mut best = 0;
            for index in 1..candidates.len() {
                if candidates[index].0 > candidates[best].0
                    || (candidates[index].0 == candidates[best].0
                        && candidates[index].1 < candidates[best].1)
                {
                    best = index;
                }
            }
            let (_, cost, x, y, result) = candidates.swap_remove(best);
            let mut labels = result.into_iter().collect::<Vec<_>>();
            labels.sort_by_key(|c| dims[c]);
            if terms.len() == 2 {
                labels = output.clone();
            }
            let right = terms.remove(y);
            let left = terms.remove(x);
            let merged = contraction(work, vec![left, right], &labels, &dims)?;
            terms.push(merged);
            cost_so_far += cost;
        }
        Ok(terms.remove(0).facts)
    })();
    if result.is_ok() {
        work.operations[operation].completed = true;
    }
    result
}

fn factor_recipe(group: &RawGroup, left: bool, allow_tucker: bool) -> Result<AdapterFactorRecipe> {
    let stem = if left { "lokr_w1" } else { "lokr_w2" };
    if let Some(key) = group.factors.get(stem) {
        return Ok(AdapterFactorRecipe::Full(key.clone()));
    }
    let a = required_key(group, &format!("{stem}_a"))?;
    let b = required_key(group, &format!("{stem}_b"))?;
    if allow_tucker && !left {
        if let Some(core) = group.factors.get("lokr_t2") {
            return Ok(AdapterFactorRecipe::Tucker {
                core: core.clone(),
                a,
                b,
            });
        }
    }
    Ok(AdapterFactorRecipe::Product { a, b })
}

fn factor(
    ctx: &FileContext,
    recipe: &AdapterFactorRecipe,
    work: &mut AdapterConstructorWork,
) -> Result<TensorFacts> {
    let mut source = |key: &str| -> Result<TensorFacts> {
        work.operands.push(AdapterSourceKey {
            epoch: ctx.source.epoch,
            key: key.into(),
        });
        full_tensor(ctx, key)
    };
    match recipe {
        AdapterFactorRecipe::Full(key) => {
            let facts = source(key)?;
            Ok(prepare(
                work,
                facts,
                AdapterDeclaredOrigin::Source(AdapterSourceKey {
                    epoch: ctx.source.epoch,
                    key: key.clone(),
                }),
            ))
        }
        AdapterFactorRecipe::Product { a, b } => {
            let a = source(a)?;
            let b = source(b)?;
            matmul(work, a, b)
        }
        AdapterFactorRecipe::Tucker { core, a, b } => {
            let core = source(core)?;
            let a = source(a)?;
            let b = source(b)?;
            tucker(work, core, a, b, false)
        }
    }
}

fn begin_constructor(
    ctx: &FileContext,
    group: &RawGroup,
    attempt: usize,
    probe: bool,
) -> AdapterConstructorWork {
    AdapterConstructorWork {
        epoch: ctx.source.epoch,
        group: group.raw.clone(),
        recipe: None,
        attempt,
        probe,
        operands: Vec::new(),
        prepared: Vec::new(),
        operations: Vec::new(),
        retained: None,
    }
}

/// A failed constructor keeps its attempted work; boxed so the `Result` stays small.
type Construction =
    std::result::Result<AdapterConstructorWork, Box<(Error, AdapterConstructorWork)>>;

fn lora_work(
    ctx: &FileContext,
    group: &RawGroup,
    spec: &AdapterSpec,
    ordinal: usize,
) -> Construction {
    let mut work = begin_constructor(ctx, group, ordinal, false);
    let built = (|| -> Result<AdapterFacts> {
        let down = required_key(group, "down")?;
        let up = required_key(group, "up")?;
        let a = full_tensor(ctx, &down)?;
        let b = full_tensor(ctx, &up)?;
        work.operands
            .extend([down.clone(), up.clone()].map(|key| AdapterSourceKey {
                epoch: ctx.source.epoch,
                key,
            }));
        let operation = attempt(
            &mut work,
            AdapterConstructorOperation::Transpose,
            std::slice::from_ref(&a),
        );
        let mut a_shape = a.shape.clone();
        a_shape.reverse();
        let mut b_shape = b.shape.clone();
        b_shape.reverse();
        let a_view = prepare(
            &mut work,
            materialized(a_shape, a.dtype),
            AdapterDeclaredOrigin::Source(AdapterSourceKey {
                epoch: ctx.source.epoch,
                key: down.clone(),
            }),
        );
        work.operations[operation].completed = true;
        let operation = attempt(
            &mut work,
            AdapterConstructorOperation::Transpose,
            std::slice::from_ref(&b),
        );
        let mut b_view = prepare(
            &mut work,
            materialized(b_shape, b.dtype),
            AdapterDeclaredOrigin::Source(AdapterSourceKey {
                epoch: ctx.source.epoch,
                key: up.clone(),
            }),
        );
        work.operations[operation].completed = true;
        // Native .t() calls precede the exact matrix gate. Do not erase that refused prefix.
        super::validate_lora_shapes(&group.raw, &a.shape, &b.shape)?;
        let fold = !group.alpha.is_empty()
            || ctx
                .lora_meta
                .as_ref()
                .is_some_and(|m| m.effective(&group.raw).0.is_some());
        work.recipe = Some(AdapterSourceRecipe::Lora {
            down,
            up,
            alpha_fold: fold,
        });
        if fold {
            super::validate_lora_alpha_rank(&group.raw, a.shape[0])?;
            let coefficient = scalar(&mut work);
            b_view = multiply(&mut work, b_view, coefficient)?;
        }
        Ok(AdapterFacts::Lora {
            a: TensorFacts::materialized_matrix(a_view.shape[0], a_view.shape[1], a_view.dtype),
            b: TensorFacts::materialized_matrix(b_view.shape[0], b_view.shape[1], b_view.dtype),
            disabled: spec.scale == 0.0,
        })
    })();
    match built {
        Ok(facts) => {
            work.retained = Some(facts);
            Ok(work)
        }
        Err(error) => Err(Box::new((error, work))),
    }
}

fn lokr_rank(ctx: &FileContext, group: &RawGroup) -> Result<Option<f32>> {
    let shape = |role: &str| {
        group
            .factors
            .get(role)
            .map(|k| ctx.source.headers[k].shape.as_slice())
    };
    super::thirdparty_lokr_rank(shape("lokr_w1_a"), shape("lokr_t2"), shape("lokr_w2_a"))
}

fn coefficient_options(
    ctx: &FileContext,
    group: &RawGroup,
    spec: &AdapterSpec,
) -> Result<Vec<PackedCoefficientClass>> {
    use PackedCoefficientClass::*;
    if !spec.scale.is_finite() {
        return Ok(vec![Nonfinite]);
    }
    if ctx.source.format == super::UnfusedAdapterFormat::PeftLokr {
        let (rank, alpha) = wm::parse_rank_alpha(
            ctx.metadata.get("rank").map(String::as_str),
            ctx.metadata.get("alpha").map(String::as_str),
        );
        let coefficient = (alpha / rank) * spec.scale;
        return Ok(vec![if !coefficient.is_finite() {
            Nonfinite
        } else if coefficient == 0.0 {
            Zero
        } else {
            FiniteNonzero
        }]);
    }
    let rank = lokr_rank(ctx, group)?;
    if rank.is_none() || group.alpha.is_empty() {
        return Ok(vec![if spec.scale == 0.0 {
            Zero
        } else {
            FiniteNonzero
        }]);
    }
    // No payload access. These are source-cost conditions, never a semantic-v1 scalar witness.
    // A finite scalar times specScale0 is zero; nonfinite payload remains a refused possibility.
    if spec.scale == 0.0 {
        Ok(vec![Zero, Nonfinite])
    } else {
        Ok(vec![Zero, FiniteNonzero, Nonfinite])
    }
}

/// One LoKr constructor attempt: the target projection, its dtype/coefficient class, the spec,
/// and whether this is the packed structured probe or the materializing fallback.
struct LokrAttempt<'a> {
    base: &'a LinearFacts,
    dtype: Dtype,
    class: PackedCoefficientClass,
    spec: &'a AdapterSpec,
    attempt: usize,
    probe: bool,
}

fn lokr_work(ctx: &FileContext, group: &RawGroup, request: LokrAttempt<'_>) -> Construction {
    let LokrAttempt {
        base,
        dtype,
        class,
        spec,
        attempt,
        probe,
    } = request;
    let mut work = begin_constructor(ctx, group, attempt, probe);
    let built = (|| -> Result<Option<AdapterFacts>> {
        if probe && class == PackedCoefficientClass::Nonfinite {
            return Err(Error::Msg("LoKr: non-finite packed scale".into()));
        }
        if probe && group.factors.contains_key("lokr_t2") {
            // Native packed probe rejects ANY t2 BEFORE choosing even a full w2/factor products.
            return Ok(None);
        }
        if ctx.source.format == super::UnfusedAdapterFormat::ThirdpartyLokr {
            lokr_rank(ctx, group)?;
        }
        if !probe && ctx.source.format == super::UnfusedAdapterFormat::PeftLokr {
            let (rank, alpha) = wm::parse_rank_alpha(
                ctx.metadata.get("rank").map(String::as_str),
                ctx.metadata.get("alpha").map(String::as_str),
            );
            super::super::validate_lokr_metadata_scale(alpha, rank)?;
        }
        let left = factor_recipe(group, true, !probe)?;
        let mut f1 = factor(ctx, &left, &mut work)?;
        let right = factor_recipe(group, false, !probe)?;
        let f2 = factor(ctx, &right, &mut work)?;
        work.recipe = Some(AdapterSourceRecipe::Lokr {
            left,
            right,
            structured: probe,
        });
        if probe {
            if super::super::lokr_structured_layout(&base.base_shape, &f1.shape, &f2.shape, false)
                .is_none()
            {
                return Ok(None); // Prepared probe products are NOT a later fallback's products.
            }
            let coefficient = scalar(&mut work);
            let scaled = multiply(&mut work, f2, coefficient)?;
            let w2 = cast(&mut work, scaled, dtype);
            let w1 = cast(&mut work, f1, dtype);
            return Ok(Some(AdapterFacts::Structured {
                w1: TensorFacts::materialized_matrix(w1.shape[0], w1.shape[1], dtype),
                w2: TensorFacts::materialized_matrix(w2.shape[0], w2.shape[1], dtype),
                disabled: class == PackedCoefficientClass::Zero,
            }));
        }
        if f1.shape.len() < f2.shape.len() {
            let mut shape = f1.shape.clone();
            shape.resize(f2.shape.len(), 1);
            f1 = reshape(&mut work, f1, shape)?;
        }
        let kron = kron(&mut work, f1, f2)?;
        let coefficient = scalar(&mut work);
        let delta = multiply(&mut work, kron, coefficient)?;
        let delta = reshape(&mut work, delta, base.base_shape.clone())?;
        let delta = cast(&mut work, delta, dtype);
        Ok(Some(AdapterFacts::Materialized {
            delta,
            disabled: spec.scale == 0.0,
        }))
    })();
    match built {
        Ok(facts) => {
            work.retained = facts;
            Ok(work)
        }
        Err(error) => Err(Box::new((error, work))),
    }
}

fn loha_work(
    ctx: &FileContext,
    group: &RawGroup,
    base: &LinearFacts,
    dtype: Dtype,
    spec: &AdapterSpec,
    attempt: usize,
) -> Construction {
    let mut work = begin_constructor(ctx, group, attempt, false);
    let built = (|| -> Result<AdapterFacts> {
        let a1 = required_key(group, "hada_w1_a")?;
        let b1 = required_key(group, "hada_w1_b")?;
        let a2 = required_key(group, "hada_w2_a")?;
        let b2 = required_key(group, "hada_w2_b")?;
        super::thirdparty_loha_rank(Some(&full_tensor(ctx, &b1)?.shape))?;
        let pair = match (group.factors.get("hada_t1"), group.factors.get("hada_t2")) {
            (Some(t1), Some(t2)) => Some((t1.clone(), t2.clone())),
            _ => None,
        };
        work.recipe = Some(AdapterSourceRecipe::Loha {
            a1: a1.clone(),
            b1: b1.clone(),
            a2: a2.clone(),
            b2: b2.clone(),
            tucker_pair: pair.clone(),
        });
        work.operands.extend(
            [&a1, &b1, &a2, &b2]
                .into_iter()
                .map(|key| AdapterSourceKey {
                    epoch: ctx.source.epoch,
                    key: key.clone(),
                }),
        );
        let (m1, m2) = if let Some((t1, t2)) = pair {
            work.operands
                .extend([t1.clone(), t2.clone()].map(|key| AdapterSourceKey {
                    epoch: ctx.source.epoch,
                    key,
                }));
            (
                tucker(
                    &mut work,
                    full_tensor(ctx, &t1)?,
                    full_tensor(ctx, &a1)?,
                    full_tensor(ctx, &b1)?,
                    true,
                )?,
                tucker(
                    &mut work,
                    full_tensor(ctx, &t2)?,
                    full_tensor(ctx, &a2)?,
                    full_tensor(ctx, &b2)?,
                    true,
                )?,
            )
        } else {
            (
                matmul(&mut work, full_tensor(ctx, &a1)?, full_tensor(ctx, &b1)?)?,
                matmul(&mut work, full_tensor(ctx, &a2)?, full_tensor(ctx, &b2)?)?,
            )
        };
        let product = multiply(&mut work, m1, m2)?;
        let coefficient = scalar(&mut work);
        let delta = multiply(&mut work, product, coefficient)?;
        let delta = reshape(&mut work, delta, base.base_shape.clone())?;
        let delta = cast(&mut work, delta, dtype);
        Ok(AdapterFacts::Materialized {
            delta,
            disabled: spec.scale == 0.0,
        })
    })();
    match built {
        Ok(facts) => {
            work.retained = Some(facts);
            Ok(work)
        }
        Err(error) => Err(Box::new((error, work))),
    }
}

#[derive(Clone, Default)]
struct State {
    next_file: usize,
    prefix: AdapterSourcePrefix,
    conditions: Vec<AdapterSourceCondition>,
}

#[derive(Clone)]
struct PlannedInstall {
    group: RawGroup,
    canonical: String,
    base: Option<LinearFacts>,
    deferred: Option<AdapterConstructorWork>,
    unmatched: Option<String>,
}

enum Task {
    File(State),
    Lora {
        state: State,
        context: Arc<FileContext>,
        group: usize,
    },
    Probe {
        state: State,
        context: Arc<FileContext>,
        group: usize,
        plans: Vec<PlannedInstall>,
        projected: u64,
    },
    Install {
        state: State,
        context: Arc<FileContext>,
        index: usize,
        plans: Vec<PlannedInstall>,
    },
}

fn refuse(
    delivery: &mut Delivery<'_>,
    state: &State,
    reason: AdapterSourceRefusal,
) -> AdapterSourceResult<()> {
    delivery.emit(
        &state.prefix,
        &state.conditions,
        AdapterSourceOutcome::Refused(reason),
    )
}

fn constructor_failure(
    delivery: &mut Delivery<'_>,
    mut state: State,
    context: &FileContext,
    error: Error,
    work: Option<AdapterConstructorWork>,
    group: &str,
) -> AdapterSourceResult<()> {
    if let Some(work) = work {
        state
            .prefix
            .work
            .push(AdapterSourceWork::Constructor(Box::new(work)));
    }
    refuse(
        delivery,
        &state,
        AdapterSourceRefusal::Constructor {
            file: context.source.epoch.0,
            group: group.into(),
            message: error.to_string(),
        },
    )
}

fn install(
    state: &mut State,
    context: &FileContext,
    canonical: String,
    work: AdapterConstructorWork,
) {
    let facts = work
        .retained
        .clone()
        .expect("only complete source endpoints install");
    let index = state.prefix.retained.len();
    let attempt = work.attempt;
    // Endpoint facts may have been prepared during packed probe. They are still one SAME slot,
    // not a reconstruction repeated at install/capture/local-i boundaries.
    if !state.prefix.work.iter().any(|event| {
        matches!(event,
        AdapterSourceWork::Constructor(old) if old.epoch == work.epoch && old.attempt == attempt)
    }) {
        state
            .prefix
            .work
            .push(AdapterSourceWork::Constructor(Box::new(work)));
    }
    state.prefix.retained.push((canonical.clone(), facts));
    state.prefix.work.push(AdapterSourceWork::Install {
        epoch: context.source.epoch,
        retained_index: index,
        canonical_target: canonical,
        constructor_attempt: attempt,
    });
    state.prefix.counts.applied += 1; // native successful push counter, not retained.len inference
}

fn finish_file(
    delivery: &mut Delivery<'_>,
    state: &mut State,
    context: &FileContext,
) -> AdapterSourceResult<()> {
    state
        .prefix
        .work
        .push(AdapterSourceWork::MaterializeAccessedAttempt {
            epoch: context.source.epoch,
            keys: context.reads.clone(),
        });
    refuse(
        delivery,
        state,
        AdapterSourceRefusal::Materialization {
            file: context.source.epoch.0,
        },
    )?;
    state
        .prefix
        .work
        .push(AdapterSourceWork::MaterializeAccessedComplete(
            context.source.epoch,
        ));
    state
        .prefix
        .counts
        .unmatched
        .extend(context.unresolved.iter().cloned());
    state.next_file += 1;
    Ok(())
}

fn parse_frontiers(
    delivery: &mut Delivery<'_>,
    state: &State,
    context: &FileContext,
) -> AdapterSourceResult<bool> {
    let mut known_fatal = false;
    let mut claims = BTreeMap::<(String, super::LoraFactorRole), Vec<AdapterActionId>>::new();
    let mut alphas = BTreeMap::<String, Vec<AdapterActionId>>::new();
    for action in &context.actions {
        match &action.kind {
            ParseActionKind::Factor {
                group,
                claim: Some(role),
                ..
            } => {
                claims
                    .entry((group.clone(), *role))
                    .or_default()
                    .push(action.id);
            }
            ParseActionKind::Alpha { group, empty } => {
                if *empty {
                    known_fatal = true;
                    let mut refused = state.clone();
                    let stop = AdapterParseStop::EmptyAlpha { action: action.id };
                    refused
                        .prefix
                        .work
                        .push(AdapterSourceWork::Parse(family(context, stop.clone())));
                    refuse(
                        delivery,
                        &refused,
                        AdapterSourceRefusal::Parse {
                            file: context.source.epoch.0,
                            stop,
                        },
                    )?;
                } else {
                    alphas.entry(group.clone()).or_default().push(action.id);
                    // Actual IO/cast/read can fail after earlier work. The attempt survives and
                    // the case stops here, without later raw materialization/constructor claims.
                    let mut refused = state.clone();
                    let stop = AdapterParseStop::AlphaReadFailed { action: action.id };
                    refused
                        .prefix
                        .work
                        .push(AdapterSourceWork::Parse(family(context, stop.clone())));
                    refuse(
                        delivery,
                        &refused,
                        AdapterSourceRefusal::Parse {
                            file: context.source.epoch.0,
                            stop,
                        },
                    )?;
                }
            }
            _ => {}
        }
    }
    for aliases in claims.values() {
        if aliases.len() < 2 {
            continue;
        }
        known_fatal = true;
        // First/second claims are roles in the family constraint, NOT chosen lexical winners.
        for &first in aliases {
            for &second in aliases {
                if first == second {
                    continue;
                }
                let mut refused = state.clone();
                let stop = AdapterParseStop::DuplicateRole { first, second };
                refused
                    .prefix
                    .work
                    .push(AdapterSourceWork::Parse(family(context, stop.clone())));
                refuse(
                    delivery,
                    &refused,
                    AdapterSourceRefusal::Parse {
                        file: context.source.epoch.0,
                        stop,
                    },
                )?;
            }
        }
    }
    if context.source.format == super::UnfusedAdapterFormat::PeftLora {
        for keys in alphas.values() {
            for &first in keys {
                for &second in keys {
                    if first == second {
                        continue;
                    }
                    let mut refused = state.clone();
                    let stop = AdapterParseStop::AlphaConflict { first, second };
                    refused
                        .prefix
                        .work
                        .push(AdapterSourceWork::Parse(family(context, stop.clone())));
                    refuse(
                        delivery,
                        &refused,
                        AdapterSourceRefusal::Parse {
                            file: context.source.epoch.0,
                            stop,
                        },
                    )?;
                }
            }
        }
    }
    Ok(!known_fatal)
}

/// Visit complete source-cost cases and Refused PREFIX CASE DATA for the existing unfused generic
/// strict NON-diff autoprefix entrypoint. Native scalar validation remains native; this cost walker
/// does not read payload values. Cases conservatively cover that entrypoint's unordered contract,
/// not observed/hash-seed-feasible traces or SDXL's separate local visitor/merge chronology.
///
/// `consume` is the SOLE charging channel. Every semantic/source refusal is delivered as data and
/// not then returned again as Err. Consumer Err stops immediately with SAME prefix evidence;
/// callers MUST NOT replay Failure.prefix or report an incomplete envelope as successful.
pub fn visit_configured_adapter_source_cases(
    specs: &[AdapterSpec],
    kohya_paths: &[String],
    lycoris_dtype: Dtype,
    projection: &mut dyn FnMut(&str) -> AdapterProjection,
    consume: &mut dyn FnMut(&AdapterSourceCase) -> Result<()>,
) -> AdapterSourceResult<AdapterSourceSummary> {
    let table = wm::kohya_table(kohya_paths);
    let mut delivery = Delivery {
        consume,
        summary: AdapterSourceSummary::default(),
        next_id: 0,
    };
    let mut stack = vec![Task::File(State::default())];
    // Metadata observations are lazy per reached ordinal. No native file epoch/slot is deduped
    // by literal path; a failed future-file header read cannot hide earlier-file histories.
    let mut contexts: Vec<Option<std::result::Result<Arc<FileContext>, String>>> =
        (0..specs.len()).map(|_| None).collect();
    while let Some(task) = stack.pop() {
        match task {
            Task::File(mut state) => {
                if state.next_file == specs.len() {
                    if !specs.is_empty()
                        && state.prefix.counts.applied == 0
                        && state.prefix.counts.pre_applied == 0
                    {
                        refuse(
                            &mut delivery,
                            &state,
                            AdapterSourceRefusal::AggregateNoMatch,
                        )?;
                    } else if !state.prefix.counts.unmatched.is_empty() {
                        refuse(
                            &mut delivery,
                            &state,
                            AdapterSourceRefusal::AggregateUnmatched,
                        )?;
                    } else {
                        delivery.emit(
                            &state.prefix,
                            &state.conditions,
                            AdapterSourceOutcome::Complete,
                        )?;
                    }
                    continue;
                }
                let index = state.next_file;
                state
                    .prefix
                    .work
                    .push(AdapterSourceWork::FileMapAttempt(AdapterFileEpoch(index)));
                if contexts[index].is_none() {
                    contexts[index] = Some(
                        file_context(&specs[index], index, &table)
                            .map(Arc::new)
                            .map_err(|e| e.to_string()),
                    );
                }
                let context = match contexts[index].as_ref().expect("initialized observation") {
                    Ok(context) => Arc::clone(context),
                    Err(message) => {
                        refuse(
                            &mut delivery,
                            &state,
                            AdapterSourceRefusal::Header {
                                file: index,
                                message: message.clone(),
                            },
                        )?;
                        continue;
                    }
                };
                state.prefix.files.push(context.source.clone());
                state
                    .prefix
                    .work
                    .push(AdapterSourceWork::FileMap(context.source.epoch));
                if context.source.format == super::UnfusedAdapterFormat::LoraKindMismatch {
                    refuse(
                        &mut delivery,
                        &state,
                        AdapterSourceRefusal::KindMismatch { file: index },
                    )?;
                    continue;
                }
                if !parse_frontiers(&mut delivery, &state, &context)? {
                    continue;
                }
                state.prefix.work.push(AdapterSourceWork::Parse(family(
                    &context,
                    AdapterParseStop::End,
                )));
                if matches!(
                    context.source.format,
                    super::UnfusedAdapterFormat::PeftLora | super::UnfusedAdapterFormat::KohyaLora
                ) {
                    stack.push(Task::Lora {
                        state,
                        context,
                        group: 0,
                    });
                } else {
                    stack.push(Task::Probe {
                        state,
                        context,
                        group: 0,
                        plans: Vec::new(),
                        projected: 0,
                    });
                }
            }
            Task::Lora {
                mut state,
                context,
                group,
            } => {
                if group == context.groups.len() {
                    finish_file(&mut delivery, &mut state, &context)?;
                    stack.push(Task::File(state));
                    continue;
                }
                let raw = &context.groups[group];
                if !raw.factors.contains_key("down") || !raw.factors.contains_key("up") {
                    stack.push(Task::Lora {
                        state,
                        context,
                        group: group + 1,
                    });
                    continue; // native partial skip still preserves parse/raw accessed keys
                }
                let projected = projection(&raw.raw);
                let canonical = match projected {
                    Ok(Some((canonical, _))) => canonical,
                    Ok(None) => {
                        state.prefix.counts.unmatched.push(raw.raw.clone());
                        stack.push(Task::Lora {
                            state,
                            context,
                            group: group + 1,
                        });
                        continue;
                    }
                    Err(error) => {
                        constructor_failure(&mut delivery, state, &context, error, None, &raw.raw)?;
                        continue;
                    }
                };
                match lora_work(
                    &context,
                    raw,
                    &specs[state.next_file],
                    state.prefix.work.len(),
                ) {
                    Ok(work) => {
                        install(&mut state, &context, canonical, work);
                        stack.push(Task::Lora {
                            state,
                            context,
                            group: group + 1,
                        });
                    }
                    Err(failed) => {
                        let (error, work) = *failed;
                        constructor_failure(
                            &mut delivery,
                            state,
                            &context,
                            error,
                            Some(work),
                            &raw.raw,
                        )?;
                    }
                }
            }
            Task::Probe {
                mut state,
                context,
                group,
                mut plans,
                projected,
            } => {
                if group == context.groups.len() {
                    if projected != 0 {
                        state.prefix.work.push(AdapterSourceWork::BudgetProbe {
                            epoch: context.source.epoch,
                            projected_raw_bytes: projected,
                        });
                        let mut refused = state.clone();
                        refused
                            .conditions
                            .push(AdapterSourceCondition::MaterializationBudget {
                                epoch: context.source.epoch,
                                fits: false,
                            });
                        refuse(
                            &mut delivery,
                            &refused,
                            AdapterSourceRefusal::MaterializationBudget {
                                file: context.source.epoch.0,
                            },
                        )?;
                        state
                            .conditions
                            .push(AdapterSourceCondition::MaterializationBudget {
                                epoch: context.source.epoch,
                                fits: true,
                            });
                    }
                    stack.push(Task::Install {
                        state,
                        context,
                        index: 0,
                        plans,
                    });
                    continue;
                }
                let raw = &context.groups[group];
                let dotted = if context.source.format == super::UnfusedAdapterFormat::PeftLokr {
                    raw.raw.clone()
                } else {
                    wm::resolve_lokr_path(&raw.raw, &table)
                        .unwrap_or_else(|| super::strip_common_lora_prefix(&raw.raw))
                        .to_owned()
                };
                // Native claim is over RESOLVED DOTTED path, not canonical target aliases.
                let collision = context.groups[..group].iter().any(|previous| {
                    let previous = if context.source.format == super::UnfusedAdapterFormat::PeftLokr
                    {
                        previous.raw.as_str()
                    } else {
                        wm::resolve_lokr_path(&previous.raw, &table)
                            .unwrap_or_else(|| super::strip_common_lora_prefix(&previous.raw))
                    };
                    previous == dotted
                });
                if collision {
                    constructor_failure(
                        &mut delivery,
                        state,
                        &context,
                        Error::Msg("LyCORIS raw keys resolve to same module twice".into()),
                        None,
                        &raw.raw,
                    )?;
                    continue;
                }
                let (canonical, base) = match projection(&dotted) {
                    Ok(Some(facts)) => facts,
                    Ok(None) => {
                        // Native LyCORIS records this plan in pass1, and appends unmatched only
                        // in PASS2 after BudgetProbe. Preserve even that refused chronology.
                        plans.push(PlannedInstall {
                            group: raw.clone(),
                            canonical: String::new(),
                            base: None,
                            deferred: None,
                            unmatched: Some(raw.raw.clone()),
                        });
                        stack.push(Task::Probe {
                            state,
                            context,
                            group: group + 1,
                            plans,
                            projected,
                        });
                        continue;
                    }
                    Err(error) => {
                        constructor_failure(&mut delivery, state, &context, error, None, &raw.raw)?;
                        continue;
                    }
                };
                if base.is_quantized
                    && context.source.format != super::UnfusedAdapterFormat::ThirdpartyLoha
                {
                    let options = match coefficient_options(&context, raw, &specs[state.next_file])
                    {
                        Ok(options) => options,
                        Err(error) => {
                            // ThirdPartyLokr::factors enters self.scale()/rank BEFORE build_lokr_factors.
                            // That failed constructor invocation has no successful tensor descriptor.
                            let work =
                                begin_constructor(&context, raw, state.prefix.work.len(), true);
                            constructor_failure(
                                &mut delivery,
                                state,
                                &context,
                                error,
                                Some(work),
                                &raw.raw,
                            )?;
                            continue;
                        }
                    };
                    for class in options.into_iter().rev() {
                        let mut branch = state.clone();
                        branch
                            .conditions
                            .push(AdapterSourceCondition::PackedCoefficient {
                                epoch: context.source.epoch,
                                group: raw.raw.clone(),
                                class,
                            });
                        match lokr_work(
                            &context,
                            raw,
                            LokrAttempt {
                                base: &base,
                                dtype: lycoris_dtype,
                                class,
                                spec: &specs[branch.next_file],
                                attempt: branch.prefix.work.len(),
                                probe: true,
                            },
                        ) {
                            Ok(work) => {
                                let deferred = work.retained.is_some();
                                branch
                                    .prefix
                                    .work
                                    .push(AdapterSourceWork::Constructor(Box::new(work.clone())));
                                let mut branch_plans = plans.clone();
                                branch_plans.push(PlannedInstall {
                                    group: raw.clone(),
                                    canonical: canonical.clone(),
                                    base: Some(base.clone()),
                                    deferred: if deferred { Some(work) } else { None },
                                    unmatched: None,
                                });
                                let extra = if deferred {
                                    0
                                } else {
                                    TensorFacts {
                                        shape: base.base_shape.clone(),
                                        dtype: lycoris_dtype,
                                        strides: Vec::new(),
                                    }
                                    .bytes()
                                };
                                let Some(projected) = projected.checked_add(extra) else {
                                    constructor_failure(
                                        &mut delivery,
                                        branch,
                                        &context,
                                        Error::Msg("materialization source byte overflow".into()),
                                        None,
                                        &raw.raw,
                                    )?;
                                    continue;
                                };
                                stack.push(Task::Probe {
                                    state: branch,
                                    context: Arc::clone(&context),
                                    group: group + 1,
                                    plans: branch_plans,
                                    projected,
                                });
                            }
                            Err(failed) => {
                                let (error, work) = *failed;
                                constructor_failure(
                                    &mut delivery,
                                    branch,
                                    &context,
                                    error,
                                    Some(work),
                                    &raw.raw,
                                )?;
                            }
                        }
                    }
                } else {
                    let extra = TensorFacts {
                        shape: base.base_shape.clone(),
                        dtype: lycoris_dtype,
                        strides: Vec::new(),
                    }
                    .bytes();
                    let Some(projected) = projected.checked_add(extra) else {
                        constructor_failure(
                            &mut delivery,
                            state,
                            &context,
                            Error::Msg("materialization source byte overflow".into()),
                            None,
                            &raw.raw,
                        )?;
                        continue;
                    };
                    plans.push(PlannedInstall {
                        group: raw.clone(),
                        canonical,
                        base: Some(base),
                        deferred: None,
                        unmatched: None,
                    });
                    stack.push(Task::Probe {
                        state,
                        context,
                        group: group + 1,
                        plans,
                        projected,
                    });
                }
            }
            Task::Install {
                mut state,
                context,
                index,
                plans,
            } => {
                if index == plans.len() {
                    finish_file(&mut delivery, &mut state, &context)?;
                    stack.push(Task::File(state));
                    continue;
                }
                let plan = &plans[index];
                if let Some(raw) = &plan.unmatched {
                    state.prefix.counts.unmatched.push(raw.clone());
                    stack.push(Task::Install {
                        state,
                        context,
                        index: index + 1,
                        plans,
                    });
                    continue;
                }
                let base = plan
                    .base
                    .as_ref()
                    .expect("matched native plan has projection facts");
                // Actual Z's post-quantization target namespace is stable across adapter pushes;
                // no generic SDXL merge/lazy-host policy is inferred from this provider contract.
                if let Some(work) = &plan.deferred {
                    install(&mut state, &context, plan.canonical.clone(), work.clone());
                    stack.push(Task::Install {
                        state,
                        context,
                        index: index + 1,
                        plans,
                    });
                } else {
                    let result =
                        if context.source.format == super::UnfusedAdapterFormat::ThirdpartyLoha {
                            loha_work(
                                &context,
                                &plan.group,
                                base,
                                lycoris_dtype,
                                &specs[state.next_file],
                                state.prefix.work.len(),
                            )
                        } else {
                            lokr_work(
                                &context,
                                &plan.group,
                                LokrAttempt {
                                    base,
                                    dtype: lycoris_dtype,
                                    class: PackedCoefficientClass::FiniteNonzero,
                                    spec: &specs[state.next_file],
                                    attempt: state.prefix.work.len(),
                                    probe: false,
                                },
                            )
                        };
                    match result {
                        Ok(work) => {
                            install(&mut state, &context, plan.canonical.clone(), work);
                            stack.push(Task::Install {
                                state,
                                context,
                                index: index + 1,
                                plans,
                            });
                        }
                        Err(failed) => {
                            let (error, work) = *failed;
                            constructor_failure(
                                &mut delivery,
                                state,
                                &context,
                                error,
                                Some(work),
                                &plan.group.raw,
                            )?;
                        }
                    }
                }
            }
        }
    }
    delivery.summary.enumeration_complete = true;
    Ok(delivery.summary)
}

impl NativeUnorderedParsePrefix {
    /// Per-action positive-work upper for this WHOLE family, not a traversal selected as native.
    /// Every generic alpha key is visited at most once; cast/read attempts occur only after a
    /// nonempty require. All other nonempty alpha actions may precede the exact stop frontier.
    /// Raw source storage may union by key; this vector MUST NOT be reduced to a key union.
    pub fn checked_alpha_attempt_upper(&self) -> Result<Vec<&AdapterParseAction>> {
        let ids = self.actions.iter().map(|a| a.id).collect::<BTreeSet<_>>();
        if ids.len() != self.actions.len()
            || !self.must_visit.is_subset(&self.may_visit)
            || !self.may_visit.is_subset(&ids)
            || self.actions.iter().any(|a| a.id.epoch != self.epoch)
        {
            return Err(Error::Msg(
                "invalid native parse family action identity constraints".into(),
            ));
        }
        let action = |id| {
            self.actions
                .iter()
                .find(|a| a.id == id)
                .ok_or_else(|| Error::Msg("parse frontier action missing".into()))
        };
        match self.stop {
            AdapterParseStop::End if self.must_visit != ids => {
                return Err(Error::Msg("complete parse family omits actions".into()));
            }
            AdapterParseStop::DuplicateRole { first, second } => {
                let (a, b) = (action(first)?, action(second)?);
                if first == second
                    || !matches!((&a.kind, &b.kind),
                    (ParseActionKind::Factor { group: x, claim: Some(rx), .. },
                     ParseActionKind::Factor { group: y, claim: Some(ry), .. }) if x == y && rx == ry)
                {
                    return Err(Error::Msg(
                        "duplicate-role frontier is not actual same-role claim".into(),
                    ));
                }
            }
            AdapterParseStop::EmptyAlpha { action: id } => {
                if !matches!(action(id)?.kind, ParseActionKind::Alpha { empty: true, .. }) {
                    return Err(Error::Msg("empty-alpha frontier is not empty".into()));
                }
            }
            AdapterParseStop::AlphaConflict { first, second } => {
                let (a, b) = (action(first)?, action(second)?);
                if !self.generic_lora_alpha_conflicts
                    || first == second
                    || !matches!((&a.kind, &b.kind),
                    (ParseActionKind::Alpha { group: x, empty: false }, ParseActionKind::Alpha { group: y, empty: false }) if x == y)
                {
                    return Err(Error::Msg(
                        "alpha conflict frontier violates generic parser policy".into(),
                    ));
                }
            }
            AdapterParseStop::AlphaReadFailed { action: id }
                if !matches!(
                    action(id)?.kind,
                    ParseActionKind::Alpha { empty: false, .. }
                ) =>
            {
                return Err(Error::Msg(
                    "alpha read frontier is not readable-shape alpha".into(),
                ));
            }
            _ => {}
        }
        Ok(self
            .actions
            .iter()
            .filter(|a| self.may_visit.contains(&a.id))
            .filter(|a| matches!(a.kind, ParseActionKind::Alpha { empty: false, .. }))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::AdapterKind;
    use std::io::Write;

    fn id(ordinal: usize) -> AdapterActionId {
        AdapterActionId {
            epoch: AdapterFileEpoch(0),
            ordinal,
        }
    }
    fn duplicate_family() -> NativeUnorderedParsePrefix {
        let actions = vec![
            AdapterParseAction {
                id: id(0),
                key: "p.lora_A.weight".into(),
                kind: ParseActionKind::Factor {
                    group: "p".into(),
                    role: "down".into(),
                    claim: Some(super::super::LoraFactorRole::Down),
                },
            },
            AdapterParseAction {
                id: id(1),
                key: "p.lora_down.weight".into(),
                kind: ParseActionKind::Factor {
                    group: "p".into(),
                    role: "down".into(),
                    claim: Some(super::super::LoraFactorRole::Down),
                },
            },
            AdapterParseAction {
                id: id(2),
                key: "p.alpha".into(),
                kind: ParseActionKind::Alpha {
                    group: "p".into(),
                    empty: false,
                },
            },
        ];
        NativeUnorderedParsePrefix {
            epoch: AdapterFileEpoch(0),
            actions,
            must_visit: [id(0), id(1)].into_iter().collect(),
            may_visit: [id(0), id(1), id(2)].into_iter().collect(),
            stop: AdapterParseStop::DuplicateRole {
                first: id(0),
                second: id(1),
            },
            generic_lora_alpha_conflicts: true,
        }
    }

    #[test]
    fn family_validates_actual_stop_order_and_action_multiplicity_not_only_sets() {
        let family = duplicate_family();
        let valid = ObservedAdapterParsePrefix {
            visits: vec![id(2), id(0), id(1)],
            requires: vec![id(2), id(0)],
            alpha_attempts: vec![id(2)],
            alpha_comparisons: Vec::new(),
        };
        assert!(family.accepts(&valid));
        let mut impossible = valid.clone();
        impossible.visits.push(id(2));
        assert!(
            !family.accepts(&impossible),
            "post-stop visit is impossible even if may-set contains key"
        );
        let mut impossible = valid.clone();
        impossible.requires.push(id(1));
        assert!(
            !family.accepts(&impossible),
            "second duplicate claim fails BEFORE require"
        );
        let mut impossible = valid.clone();
        impossible.alpha_attempts.push(id(2));
        assert!(
            !family.accepts(&impossible),
            "repeated work attempt cannot disappear in key dedup"
        );
        let mut impossible = valid;
        impossible.visits = vec![id(1), id(0), id(2)];
        assert!(
            !family.accepts(&impossible),
            "opposite first/second frontier is another family, not this one"
        );
        assert_eq!(
            family
                .checked_alpha_attempt_upper()
                .unwrap()
                .iter()
                .map(|a| a.id)
                .collect::<Vec<_>>(),
            vec![id(2)]
        );
    }

    fn file(
        root: &std::path::Path,
        name: &str,
        entries: &[(&str, &[usize], &str)],
    ) -> std::path::PathBuf {
        file_meta(root, name, entries, serde_json::json!({}))
    }
    fn file_meta(
        root: &std::path::Path,
        name: &str,
        entries: &[(&str, &[usize], &str)],
        metadata: serde_json::Value,
    ) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), metadata);
        let mut offset = 0u64;
        for (key, shape, dtype) in entries {
            let width: u64 = if *dtype == "F32" { 4 } else { 2 };
            let bytes = shape.iter().fold(width, |n, &d| n * d as u64);
            header.insert(
                (*key).into(),
                serde_json::json!({"dtype": dtype, "shape": shape,
                "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
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

    fn spec(path: std::path::PathBuf) -> AdapterSpec {
        AdapterSpec {
            path,
            kind: AdapterKind::Lora,
            scale: 1.0,
            pass_scales: None,
            moe_expert: None,
        }
    }
    fn projection(path: &str) -> Result<Option<(String, LinearFacts)>> {
        Ok(Some((
            path.into(),
            LinearFacts {
                base_shape: vec![8, 8],
                is_quantized: false,
                has_live_adapters: false,
                adapter_count: 0,
                bias_shape: None,
            },
        )))
    }

    #[test]
    fn callback_error_preserves_same_nonempty_prefix_once_without_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(dir.path(), "orphan", &[("p.lora_A.weight", &[1, 8], "F32")]);
        let mut calls = 0;
        let failure = visit_configured_adapter_source_cases(&[spec(path)], &[], Dtype::Bfloat16,
            &mut projection, &mut |case| {
                calls += 1;
                assert!(matches!(case.outcome, AdapterSourceOutcome::Refused(AdapterSourceRefusal::Materialization { file: 0 })));
                assert!(case.prefix.work.iter().any(|w| matches!(w, AdapterSourceWork::MaterializeAccessedAttempt { keys, .. } if keys.contains("p.lora_A.weight"))));
                Err(Error::Msg("deliberate consumer abort AFTER observing refused prefix".into()))
            }).unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(failure.attempted_case_id, Some(0));
        assert!(!failure.prefix.work.is_empty());
        assert!(matches!(failure.kind, AdapterSourceAbortKind::Consumer));
    }

    #[test]
    fn future_file_io_cannot_preempt_an_earlier_known_parse_failure() {
        let dir = tempfile::tempdir().unwrap();
        let first = file(
            dir.path(),
            "duplicate",
            &[
                ("p.lora_A.weight", &[1, 8], "F32"),
                ("p.lora_down.weight", &[1, 8], "F32"),
                ("p.lora_B.weight", &[8, 1], "F32"),
            ],
        );
        let later = dir.path().join("unreadable-later.safetensors");
        let mut cases = Vec::new();
        let summary = visit_configured_adapter_source_cases(
            &[spec(first), spec(later)],
            &[],
            Dtype::Bfloat16,
            &mut projection,
            &mut |case| {
                cases.push(case.clone());
                Ok(())
            },
        )
        .unwrap();
        assert!(summary.enumeration_complete);
        assert_eq!(summary.complete_cases, 0);
        assert!(!cases.is_empty());
        assert!(cases.iter().all(|case| matches!(
            case.outcome,
            AdapterSourceOutcome::Refused(AdapterSourceRefusal::Parse { file: 0, .. })
        )));
        assert!(cases.iter().all(|case| case.prefix.files.len() == 1));
    }

    #[test]
    fn repeated_literal_paths_keep_file_epochs_and_global_same_target_slots() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(
            dir.path(),
            "same",
            &[
                ("p.lora_A.weight", &[1, 8], "F16"),
                ("p.lora_B.weight", &[8, 1], "BF16"),
            ],
        );
        let mut complete = Vec::new();
        let summary = visit_configured_adapter_source_cases(
            &[spec(path.clone()), spec(path)],
            &[],
            Dtype::Bfloat16,
            &mut projection,
            &mut |case| {
                if matches!(case.outcome, AdapterSourceOutcome::Complete) {
                    complete.push(case.clone());
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(summary.enumeration_complete && summary.complete_cases != 0);
        for case in complete {
            assert_eq!(
                case.prefix
                    .files
                    .iter()
                    .map(|f| f.epoch)
                    .collect::<Vec<_>>(),
                vec![AdapterFileEpoch(0), AdapterFileEpoch(1)]
            );
            assert_eq!(
                case.prefix
                    .retained
                    .iter()
                    .map(|(target, _)| target.as_str())
                    .collect::<Vec<_>>(),
                vec!["p", "p"]
            );
            assert_eq!(
                case.prefix
                    .work
                    .iter()
                    .filter_map(|w| match w {
                        AdapterSourceWork::Install { retained_index, .. } => Some(*retained_index),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                vec![0, 1]
            );
            assert_eq!(
                case.prefix.files[0].headers["p.lora_A.weight"].dtype,
                Dtype::Float16
            );
            assert_eq!(
                case.prefix.files[0].headers["p.lora_B.weight"].dtype,
                Dtype::Bfloat16
            );
        }
    }

    #[test]
    fn full_alpha_vector_cast_and_orphan_raw_materialization_survive_in_source_work() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(
            dir.path(),
            "alpha",
            &[
                ("p.lora_A.weight", &[1, 8], "F16"),
                ("p.lora_B.weight", &[8, 1], "BF16"),
                ("p.alpha", &[7], "F16"),
                ("orphan.lora_A.weight", &[3, 8], "BF16"),
            ],
        );
        let mut seen = false;
        visit_configured_adapter_source_cases(&[spec(path)], &[], Dtype::Bfloat16, &mut projection,
            &mut |case| {
                if matches!(case.outcome, AdapterSourceOutcome::Complete) {
                    let alpha = &case.prefix.files[0].headers["p.alpha"];
                    assert_eq!((alpha.shape.clone(), alpha.dtype, alpha.bytes()), (vec![7], Dtype::Float16, 14));
                    let cast = materialized(alpha.shape.clone(), Dtype::Float32);
                    assert_eq!(cast.bytes(), 28, "FULL vector cast is not8B semantic-control read");
                    assert!(case.prefix.work.iter().any(|w| matches!(w,
                        AdapterSourceWork::MaterializeAccessedAttempt { keys, .. } if keys.contains("orphan.lora_A.weight") && keys.contains("p.alpha"))));
                    match &case.prefix.retained[0].1 {
                        AdapterFacts::Lora { a, b, .. } => assert_eq!((a.dtype, b.dtype), (Dtype::Float16, Dtype::Float32)),
                        _ => panic!("wrong actual configured endpoint family"),
                    }
                    seen = true;
                }
                Ok(())
            }).unwrap();
        assert!(seen);
    }

    #[test]
    fn every_existing_unfused_family_produces_and_materializes_its_exact_source_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cases: Vec<(
            &str,
            AdapterKind,
            Vec<(&str, &[usize], &str)>,
            serde_json::Value,
        )> = vec![
            (
                "peft",
                AdapterKind::Lora,
                vec![
                    ("p.lora_A.weight", &[1, 8], "F16"),
                    ("p.lora_B.weight", &[8, 1], "BF16"),
                ],
                serde_json::json!({}),
            ),
            (
                "kohya",
                AdapterKind::Lora,
                vec![
                    ("lora_unet_p.lora_down.weight", &[1, 8], "BF16"),
                    ("lora_unet_p.lora_up.weight", &[8, 1], "F32"),
                ],
                serde_json::json!({}),
            ),
            (
                "peft_lokr",
                AdapterKind::Lokr,
                vec![
                    ("p.lokr_w1", &[2, 2], "F16"),
                    ("p.lokr_w2", &[4, 4], "BF16"),
                ],
                serde_json::json!({"networkType":"lokr","rank":"2","alpha":"4"}),
            ),
            (
                "tp_lokr",
                AdapterKind::Lora,
                vec![
                    ("p.lokr_w1", &[2, 2], "F16"),
                    ("p.lokr_w2", &[4, 4], "BF16"),
                ],
                serde_json::json!({}),
            ),
            (
                "tp_loha",
                AdapterKind::Lora,
                vec![
                    ("p.hada_w1_a", &[8, 1], "F16"),
                    ("p.hada_w1_b", &[1, 8], "BF16"),
                    ("p.hada_w2_a", &[8, 1], "BF16"),
                    ("p.hada_w2_b", &[1, 8], "F32"),
                ],
                serde_json::json!({}),
            ),
        ];
        for (name, kind, entries, metadata) in cases {
            let path = file_meta(dir.path(), name, &entries, metadata);
            let expected = entries
                .iter()
                .map(|(key, _, _)| (*key).to_owned())
                .collect::<BTreeSet<_>>();
            let mut configured = spec(path);
            configured.kind = kind;
            let mut complete = Vec::new();
            let summary = visit_configured_adapter_source_cases(
                &[configured],
                &["p".into()],
                Dtype::Bfloat16,
                &mut projection,
                &mut |case| {
                    if matches!(case.outcome, AdapterSourceOutcome::Complete) {
                        complete.push(case.clone());
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert!(
                summary.enumeration_complete && summary.complete_cases != 0,
                "actual family source plan failed: {name}"
            );
            for case in complete {
                assert_eq!(case.prefix.counts.applied, 1);
                assert_eq!(case.prefix.retained[0].0, "p");
                let materialized = case
                    .prefix
                    .work
                    .iter()
                    .filter_map(|work| match work {
                        AdapterSourceWork::MaterializeAccessedAttempt { keys, .. } => {
                            Some(keys.clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    materialized,
                    vec![expected.clone()],
                    "exact parser sources for {name}"
                );
            }
        }
    }

    #[test]
    fn packed_tucker_probe_and_full_factor_precedence_keep_shadow_raw_sources() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(
            dir.path(),
            "shadow",
            &[
                ("p.lokr_w1", &[2, 2], "BF16"),
                ("p.lokr_w1_a", &[2, 1], "F16"),
                ("p.lokr_w1_b", &[1, 2], "F16"),
                ("p.lokr_w2", &[4, 4], "BF16"),
                ("p.lokr_t2", &[1, 1], "BF16"),
            ],
        );
        let mut packed = |path: &str| {
            let mut facts = projection(path)?.unwrap();
            facts.1.is_quantized = true;
            Ok(Some(facts))
        };
        let mut seen = false;
        visit_configured_adapter_source_cases(&[spec(path)], &["p".into()], Dtype::Bfloat16, &mut packed,
            &mut |case| {
                if matches!(case.outcome, AdapterSourceOutcome::Complete) {
                    let constructors = case.prefix.work.iter().filter_map(|w| match w { AdapterSourceWork::Constructor(w) => Some(w), _ => None }).collect::<Vec<_>>();
                    assert!(constructors[0].probe && constructors[0].retained.is_none());
                    assert!(constructors[0].prepared.is_empty(), "ANYt2 rejects packed probe before factor products, even fullw2");
                    assert!(matches!(constructors.last().unwrap().recipe,
                        Some(AdapterSourceRecipe::Lokr { left: AdapterFactorRecipe::Full(_), right: AdapterFactorRecipe::Full(_), structured: false })));
                    assert!(case.prefix.work.iter().any(|w| matches!(w, AdapterSourceWork::MaterializeAccessedAttempt { keys, .. }
                        if keys.contains("p.lokr_w1_a") && keys.contains("p.lokr_w1_b") && keys.contains("p.lokr_t2"))));
                    seen = true;
                }
                Ok(())
            }).unwrap();
        assert!(seen);
    }

    fn empty_work() -> AdapterConstructorWork {
        AdapterConstructorWork {
            epoch: AdapterFileEpoch(0),
            group: "independent".into(),
            recipe: None,
            attempt: 0,
            probe: false,
            operands: Vec::new(),
            prepared: Vec::new(),
            operations: Vec::new(),
            retained: None,
        }
    }

    #[test]
    fn sequential_naive_products_preserve_integer_intermediate_and_local_shapes() {
        // Hand-authored factory comparison: first operands share i; third only adds j. There is
        // no contraction, so native naive path's first product is I8[2,1], then F16[2,3].
        let mut work = empty_work();
        let terms = vec![
            EinsumTerm {
                labels: vec!['i'],
                facts: materialized(vec![2], Dtype::Int8),
            },
            EinsumTerm {
                labels: vec!['i'],
                facts: materialized(vec![2], Dtype::Int8),
            },
            EinsumTerm {
                labels: vec!['j'],
                facts: materialized(vec![3], Dtype::Float16),
            },
        ];
        let result = contraction(
            &mut work,
            terms,
            &['i', 'j'],
            &BTreeMap::from([('i', 2), ('j', 3)]),
        )
        .unwrap();
        let products = work
            .operations
            .iter()
            .filter(|op| op.operation == AdapterConstructorOperation::Multiply)
            .collect::<Vec<_>>();
        assert_eq!(products.len(), 2);
        assert_eq!(
            products[0]
                .operands
                .iter()
                .map(|t| (t.shape.clone(), t.dtype))
                .collect::<Vec<_>>(),
            vec![(vec![2, 1], Dtype::Int8), (vec![2, 1], Dtype::Int8)]
        );
        assert_eq!(products[1].operands[0].dtype, Dtype::Int8);
        assert_eq!(
            (result.facts.shape, result.facts.dtype),
            (vec![2, 3], Dtype::Float16)
        );
    }

    #[test]
    fn operator_refusals_preserve_only_reached_casts_and_no_successful_output() {
        let mut work = empty_work();
        // Broadcast failure occurs AFTER multiply's F16 -> F32 cast, but no Multiply output.
        assert!(multiply(
            &mut work,
            materialized(vec![2], Dtype::Float16),
            materialized(vec![3], Dtype::Float32)
        )
        .is_err());
        assert_eq!(
            work.operations[0].operation,
            AdapterConstructorOperation::Multiply
        );
        assert!(!work.operations[0].completed);
        assert_eq!(
            work.prepared
                .iter()
                .map(|t| (t.facts.shape.clone(), t.facts.dtype))
                .collect::<Vec<_>>(),
            vec![(vec![2], Dtype::Float32)]
        );
        let mut work = empty_work();
        // Pair exact dtype rejection is BEFORE any cast; the failed Matmul attempt remains.
        assert!(matmul(
            &mut work,
            materialized(vec![2, 2], Dtype::Int32),
            materialized(vec![2, 2], Dtype::Int32)
        )
        .is_err());
        assert_eq!(work.operations.len(), 1);
        assert!(!work.operations[0].completed);
        assert!(work.prepared.is_empty());
        // Batch-broadcast rejection instead follows the pair casts.
        let mut work = empty_work();
        assert!(matmul(
            &mut work,
            materialized(vec![2, 2, 2], Dtype::Float16),
            materialized(vec![3, 2, 2], Dtype::Bfloat16)
        )
        .is_err());
        assert_eq!(
            work.prepared
                .iter()
                .map(|t| t.facts.dtype)
                .collect::<Vec<_>>(),
            vec![Dtype::Float32, Dtype::Float32]
        );
        assert!(!work.operations[0].completed);
        assert_eq!(promote_dtype(Dtype::Uint64, Dtype::Int8), Dtype::Float32);
        assert_eq!(
            promote_dtype(Dtype::Float16, Dtype::Bfloat16),
            Dtype::Float32
        );
    }

    #[test]
    fn failed_raw_materialization_is_consumed_once_before_any_future_file_observation() {
        let dir = tempfile::tempdir().unwrap();
        let first = file(
            dir.path(),
            "raw-prefix",
            &[
                ("p.lora_A.weight", &[1, 8], "F32"),
                ("p.lora_B.weight", &[8, 1], "F32"),
            ],
        );
        let missing = dir.path().join("not-yet-opened.safetensors");
        let mut calls = 0;
        let failure = visit_configured_adapter_source_cases(
            &[spec(first), spec(missing)],
            &[],
            Dtype::Bfloat16,
            &mut projection,
            &mut |case| {
                calls += 1;
                assert!(matches!(
                    case.outcome,
                    AdapterSourceOutcome::Refused(AdapterSourceRefusal::Materialization {
                        file: 0
                    })
                ));
                assert_eq!(
                    case.prefix.retained.len(),
                    1,
                    "native endpoint installed BEFORE raw materialize attempt"
                );
                assert!(matches!(
                    case.prefix.work.last(),
                    Some(AdapterSourceWork::MaterializeAccessedAttempt { .. })
                ));
                assert_eq!(
                    case.prefix.files.len(),
                    1,
                    "future file must not preempt materialization refusal"
                );
                Err(Error::Msg(
                    "deliberate callback abort at raw materialization prefix".into(),
                ))
            },
        )
        .unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(failure.attempted_case_id, Some(0));
        assert_eq!(failure.prefix.retained.len(), 1);
    }

    #[test]
    fn aggregate_zero_relaxation_does_not_change_successful_semantic_v1_json_or_unknown_targets() {
        let dir = tempfile::tempdir().unwrap();
        let orphan = file(
            dir.path(),
            "orphan-semantic",
            &[("p.lora_A.weight", &[1, 8], "F32")],
        );
        let good = file(
            dir.path(),
            "good-semantic",
            &[
                ("p.lora_A.weight", &[1, 8], "F32"),
                ("p.lora_B.weight", &[8, 1], "F32"),
            ],
        );
        let (old_success, _) = super::super::configured_adapter_facts(
            &[spec(good.clone())],
            &[],
            Dtype::Bfloat16,
            projection,
        )
        .unwrap();
        let (batch, retained) = super::super::configured_adapter_facts(
            &[spec(orphan.clone()), spec(good.clone())],
            &[],
            Dtype::Bfloat16,
            projection,
        )
        .unwrap();
        assert_eq!(batch["schema"], "unfused_adapter_semantic_graph_v1");
        assert_eq!(
            batch["files"][1], old_success["files"][0],
            "old successful per-file JSON semantics preserved exactly"
        );
        assert_eq!(retained.len(), 1);
        assert!(super::super::configured_adapter_facts(
            &[spec(orphan)],
            &[],
            Dtype::Bfloat16,
            projection
        )
        .is_err());
        assert!(
            super::super::configured_adapter_facts(&[spec(good)], &[], Dtype::Bfloat16, |_| Ok(
                None
            ))
            .is_err(),
            "unknown target policy not relaxed"
        );
    }

    #[test]
    fn enumerator_abort_keeps_actual_pending_prefix_before_any_callback_delivery() {
        let prefix = AdapterSourcePrefix {
            work: vec![AdapterSourceWork::FileMapAttempt(AdapterFileEpoch(1))],
            ..AdapterSourcePrefix::default()
        };
        let mut calls = 0;
        let mut consume = |_: &AdapterSourceCase| {
            calls += 1;
            Ok(())
        };
        let mut delivery = Delivery {
            consume: &mut consume,
            summary: AdapterSourceSummary::default(),
            next_id: u64::MAX,
        };
        let failure = delivery
            .emit(&prefix, &[], AdapterSourceOutcome::Complete)
            .unwrap_err();
        assert!(matches!(failure.kind, AdapterSourceAbortKind::Enumerator));
        assert_eq!(failure.attempted_case_id, None);
        assert!(matches!(failure.prefix.work.as_slice(),[AdapterSourceWork::FileMapAttempt(AdapterFileEpoch(1))]),
            "ENUMERATOR-PREFIX: actual pending epoch survives ordinal abort, not empty reconstruction");
        drop(delivery);
        assert_eq!(
            calls, 0,
            "overflow abort is BEFORE callback; provenance is not a replay channel"
        );
    }

    #[test]
    fn same_width_f16_bf16_cast_is_conversion_work_not_source_clone() {
        let mut work = empty_work();
        let source = materialized(vec![8, 8], Dtype::Float16);
        let converted = cast(&mut work, source, Dtype::Bfloat16);
        assert_eq!(
            (converted.shape, converted.dtype),
            (vec![8, 8], Dtype::Bfloat16),
            "SAME-WIDTH-CAST dtype equality, not bytewidth, controls the native conversion"
        );
        assert_eq!(work.prepared.len(), 1);
        assert_eq!(
            (
                work.prepared[0].facts.shape.clone(),
                work.prepared[0].facts.dtype,
                work.prepared[0].facts.bytes()
            ),
            (vec![8, 8], Dtype::Bfloat16, 128)
        );
        assert!(
            matches!(
                work.prepared[0].origin,
                AdapterDeclaredOrigin::DeferredCreation
            ),
            "F16->BF16 prepares a typed output descriptor, not another raw/source alias"
        );
        let mut same = empty_work();
        let unchanged = cast(
            &mut same,
            materialized(vec![8, 8], Dtype::Bfloat16),
            Dtype::Bfloat16,
        );
        assert_eq!(unchanged.dtype, Dtype::Bfloat16);
        assert!(
            same.prepared.is_empty(),
            "native same-dtype cast is the alias case; equal-bytewidth alone is not"
        );
    }
}
