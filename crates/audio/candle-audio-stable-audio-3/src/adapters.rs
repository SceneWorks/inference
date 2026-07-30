//! Load-time adapter stacking for Stable Audio 3 — the full eight-type family.
//!
//! Upstream Stable Audio 3 ships a first-class adapter ecosystem, and it is **not** classic LoRA
//! only. Eight canonical types plus one legacy alias are trained and published against these
//! checkpoints, and this module implements all of them:
//!
//! | type | delta | post-transform |
//! |---|---|---|
//! | [`AdapterType::Lora`] | `(α/r)·s·(B @ A)` | none |
//! | [`AdapterType::DoraRows`] | as above | row-normalize, rescale by `magnitude_r` |
//! | [`AdapterType::DoraCols`] | as above | column-normalize, rescale by `magnitude_c` |
//! | [`AdapterType::Bora`] | as above | rows, then columns |
//! | [`AdapterType::LoraXs`] | `(α/r)·s·(U @ M @ Vᵀ)` | none |
//! | [`AdapterType::DoraRowsXs`] | as `LoraXs` | rows |
//! | [`AdapterType::DoraColsXs`] | as `LoraXs` | columns |
//! | [`AdapterType::BoraXs`] | as `LoraXs` | rows, then columns |
//!
//! `dora` is a legacy alias resolved by sniffing which magnitude tensor the file carries, defaulting
//! to [`AdapterType::DoraRows`] — which is also upstream's training default, so it is the type most
//! real artifacts will be.
//!
//! `U` and `V` for the `-xs` half are the top-`r` singular vectors of the **base weight**, not
//! tensors in the adapter file. Candle has no SVD, so [`crate::svd`] provides a deterministic host
//! one; read its module docs before touching anything `-xs`.
//!
//! # Order is load-bearing here, unlike the image lane
//!
//! Classic LoRA deltas commute: `W + δ₁ + δ₂` is order-free, which is why the image providers can
//! fold a stack into a map and not care. **DoRA and BoRA do not commute.** The strength participates
//! *inside* the normalization — `magnitude · (W + s·δ) / ‖W + s·δ‖` is not linear in `δ` — so
//! swapping two adapters in a stack produces genuinely different weights. [`AdapterPlan::build`]
//! therefore preserves [`candle_audio::gen_core::LoadSpec::adapters`] order exactly and
//! [`apply_adapter_ops`] folds strictly sequentially, each adapter seeing the previous one's output
//! as its base.
//!
//! One consequence worth stating because it is a real choice and not a derivation: for a stacked
//! `-xs` adapter, the SVD bases come from the weight **as it enters that adapter**, not from the
//! pristine checkpoint weight. That is the reading under which "fold sequentially" and "attach
//! parametrizations sequentially" are the same operation. It is *not* verified against an upstream
//! artifact, because none exists on this machine — see the honesty note at the bottom of these docs.
//!
//! # Targets
//!
//! The DiT (`model.…`) and the registered conditioner modules (`conditioner.…`). Two structural
//! facts keep the forbidden targets out rather than a check that could be deleted:
//!
//! * **T5Gemma lives in a different file.** The text encoder is `t5gemma-b-b-ul2/model.safetensors`,
//!   a separate mmap with a separate [`candle_nn::VarBuilder`]. [`AdapterBackend`] wraps only the
//!   root checkpoint's backend, so no adapter can reach a T5 tensor by any key it could name.
//! * **SAME is excluded by prefix.** `pretransform.…` is in the root file, so it is excluded
//!   explicitly by [`is_adaptable_target`] and an adapter naming one fails loudly.
//!
//! # Formats, and the security boundary
//!
//! Accepted: a native `.safetensors` file, or a PEFT directory (`adapter_model.safetensors` +
//! `adapter_config.json`). **Refused: every pickle container** — `.ckpt`, `.pt`, `.pth`, `.bin` —
//! with a typed [`candle_audio::gen_core::Error::Unsupported`]. Unpickling is arbitrary code execution, and an
//! adapter path is caller-supplied data; there is no "trusted" version of that. The same rule is
//! why the `svd_bases.pt` shipped beside the `-base` checkpoints is never read here: it is a
//! *startup cache* for a decomposition this crate computes itself, so refusing it costs nothing.
//!
//! # What is NOT claimed
//!
//! **No upstream artifact was available to validate against.** There is no Stable Audio 3 adapter
//! of any type in this machine's cache, and no published community one. Everything here is gated
//! weight-free against adapters this repository *constructs*, in the on-disk format this module
//! declares. That is enough to prove the math, the ordering, the filters, the formats and the
//! refusals; it is **not** evidence that the key spellings below match an upstream file, and no
//! such claim appears anywhere in this crate. Obtaining a real artifact is tracked as `sc-15347`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::{AdapterKind, AdapterSpec};
use candle_audio::{AudioError, Result};

/// Extensions that are pickle containers, refused by [`AdapterSource::classify`].
///
/// Deliberately a denylist of *known* pickle spellings plus a catch-all in the classifier: an
/// unrecognized extension is refused too, so this list existing does not make an unknown format
/// implicitly safe.
pub const PICKLE_EXTENSIONS: &[&str] = &["ckpt", "pt", "pth", "bin"];

/// The PEFT sidecar filenames.
pub const PEFT_WEIGHTS_FILE: &str = "adapter_model.safetensors";
pub const PEFT_CONFIG_FILE: &str = "adapter_config.json";

/// The checkpoint key prefixes an adapter may target.
///
/// `model.` is the DiT; `conditioner.` is the registered conditioner stack, whose one adaptable
/// module on every shipped checkpoint is the learned `seconds_total` NumberConditioner Linear
/// (`conditioner.conditioners.seconds_total.embedder.embedding.1.weight`, `[768, 256]`).
pub const ADAPTABLE_PREFIXES: &[&str] = &["model.", "conditioner."];

/// One of the eight canonical Stable Audio 3 adapter parametrizations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdapterType {
    Lora,
    DoraRows,
    DoraCols,
    Bora,
    LoraXs,
    DoraRowsXs,
    DoraColsXs,
    BoraXs,
}

impl AdapterType {
    /// The canonical spelling, as it appears in adapter metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::DoraRows => "dora-rows",
            Self::DoraCols => "dora-cols",
            Self::Bora => "bora",
            Self::LoraXs => "lora-xs",
            Self::DoraRowsXs => "dora-rows-xs",
            Self::DoraColsXs => "dora-cols-xs",
            Self::BoraXs => "bora-xs",
        }
    }

    /// Every canonical type, in declaration order. Exists so a test can enumerate the family
    /// without restating it — a hand-written list in a test is one that silently stops covering a
    /// type someone adds here.
    pub const ALL: [Self; 8] = [
        Self::Lora,
        Self::DoraRows,
        Self::DoraCols,
        Self::Bora,
        Self::LoraXs,
        Self::DoraRowsXs,
        Self::DoraColsXs,
        Self::BoraXs,
    ];

    /// Does this type reconstruct its delta through the base weight's SVD bases?
    pub fn is_xs(self) -> bool {
        matches!(
            self,
            Self::LoraXs | Self::DoraRowsXs | Self::DoraColsXs | Self::BoraXs
        )
    }

    /// Does this type row-normalize and rescale by `magnitude_r`?
    pub fn uses_row_magnitude(self) -> bool {
        matches!(
            self,
            Self::DoraRows | Self::Bora | Self::DoraRowsXs | Self::BoraXs
        )
    }

    /// Does this type column-normalize and rescale by `magnitude_c`?
    pub fn uses_col_magnitude(self) -> bool {
        matches!(
            self,
            Self::DoraCols | Self::Bora | Self::DoraColsXs | Self::BoraXs
        )
    }
}

/// Resolve a declared type string, including the legacy `dora` / `dora-xs` aliases.
///
/// The alias carries no axis, so it is decided by which magnitude tensor the file actually carries:
/// a `magnitude_c` and no `magnitude_r` means columns, and **anything else — including both, or
/// neither — defaults to rows**, which is upstream's training default. Defaulting rather than
/// failing is deliberate: an alias file is by definition one written before the axis was spelled
/// out, so there is no spelling to consult, and rows is the axis such a file overwhelmingly is.
pub fn resolve_adapter_type(
    declared: &str,
    has_row_magnitude: bool,
    has_col_magnitude: bool,
) -> Result<AdapterType> {
    let normalized = declared.trim().to_ascii_lowercase();
    let resolved = match normalized.as_str() {
        "lora" => AdapterType::Lora,
        "dora-rows" => AdapterType::DoraRows,
        "dora-cols" => AdapterType::DoraCols,
        "bora" => AdapterType::Bora,
        "lora-xs" => AdapterType::LoraXs,
        "dora-rows-xs" => AdapterType::DoraRowsXs,
        "dora-cols-xs" => AdapterType::DoraColsXs,
        "bora-xs" => AdapterType::BoraXs,
        "dora" => {
            if has_col_magnitude && !has_row_magnitude {
                AdapterType::DoraCols
            } else {
                AdapterType::DoraRows
            }
        }
        "dora-xs" => {
            if has_col_magnitude && !has_row_magnitude {
                AdapterType::DoraColsXs
            } else {
                AdapterType::DoraRowsXs
            }
        }
        other => {
            return Err(AudioError::Msg(format!(
                "unknown Stable Audio 3 adapter type {other:?}; expected one of {:?} or the legacy \
                 aliases \"dora\" / \"dora-xs\"",
                AdapterType::ALL.map(AdapterType::as_str)
            )));
        }
    };
    Ok(resolved)
}

/// The per-module factor tensors of one adapter, already validated for shape and finiteness.
#[derive(Clone, Debug)]
pub struct AdapterModule {
    /// `[r, in]` — the down projection. Absent for `-xs`, which carries `m` instead.
    pub a: Option<Tensor>,
    /// `[out, r]` — the up projection. Absent for `-xs`.
    pub b: Option<Tensor>,
    /// `[r, r]` — the `-xs` core, sandwiched between the base weight's own singular bases.
    pub m: Option<Tensor>,
    /// `[out, 1]` — the learned row magnitude.
    pub magnitude_r: Option<Tensor>,
    /// `[1, in]` — the learned column magnitude.
    pub magnitude_c: Option<Tensor>,
}

/// One adapter file, parsed and validated but not yet matched against a checkpoint.
#[derive(Clone, Debug)]
pub struct LoadedAdapter {
    pub path: PathBuf,
    pub kind: AdapterType,
    pub rank: usize,
    pub alpha: f32,
    /// The caller's [`AdapterSpec::scale`].
    pub scale: f32,
    /// Substring filters from the adapter's own metadata, already bracket-expanded.
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    /// Target checkpoint key (**including** the trailing `.weight`) to its factors.
    pub modules: BTreeMap<String, AdapterModule>,
    /// Position in the caller's [`candle_audio::gen_core::LoadSpec::adapters`].
    ///
    /// Assigned by [`plan_for`], which is the only place that can know it: [`load_adapter`] reads
    /// one path and has no view of the stack, so it leaves this `0`. It is carried on the adapter
    /// rather than recomputed while planning because [`plan_for`] builds the plan a second time
    /// over the **zero-scale-filtered** slice, and a positional `enumerate()` there would renumber
    /// the survivors — reporting the live member of `[zero, live]` as index 0.
    pub spec_index: usize,
}

/// Which on-disk shape an adapter path is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterSource {
    /// A single `.safetensors` file carrying this crate's native metadata.
    NativeFile,
    /// A directory with `adapter_model.safetensors` + `adapter_config.json`.
    PeftDirectory,
}

impl AdapterSource {
    /// Classify a caller-supplied adapter path, refusing every unsafe container.
    ///
    /// The pickle refusal is [`candle_audio::gen_core::Error::Unsupported`] rather than a generic message so a
    /// caller can distinguish "this file is the wrong format" from "this file is broken". See the
    /// module docs for why there is no opt-in.
    pub fn classify(path: &Path) -> Result<Self> {
        if path.is_dir() {
            let weights = path.join(PEFT_WEIGHTS_FILE);
            let config = path.join(PEFT_CONFIG_FILE);
            if weights.is_file() && config.is_file() {
                return Ok(Self::PeftDirectory);
            }
            return Err(AudioError::Msg(format!(
                "{} is not a PEFT adapter directory: expected both {PEFT_WEIGHTS_FILE} and \
                 {PEFT_CONFIG_FILE}",
                path.display()
            )));
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if PICKLE_EXTENSIONS.contains(&extension.as_str()) {
            return Err(unsupported(format!(
                "{} is a Python pickle container (.{extension}); Stable Audio 3 adapters must be \
                 .safetensors or a PEFT directory. Unpickling executes arbitrary code, so there is \
                 no opt-in for this — convert the adapter to safetensors first.",
                path.display()
            )));
        }
        if extension == "safetensors" {
            if !path.is_file() {
                return Err(AudioError::Msg(format!(
                    "adapter file does not exist: {}",
                    path.display()
                )));
            }
            return Ok(Self::NativeFile);
        }
        Err(unsupported(format!(
            "{} is not a recognized Stable Audio 3 adapter: expected a .safetensors file or a PEFT \
             directory",
            path.display()
        )))
    }
}

fn unsupported(message: String) -> AudioError {
    // `AudioError` has no `Unsupported` variant; the crate's bridge maps `Msg` to
    // `gen_core::Error::Msg`. Provider entry points that must surface a typed `Unsupported` call
    // `into_unsupported` on the way out — see `crate::model`.
    AudioError::Msg(message)
}

/// Lift an adapter-layer failure into the contract's typed [`candle_audio::gen_core::Error::Unsupported`].
///
/// Every refusal in this module is a statement that the *request* cannot be served, not that the
/// machine failed, so the provider boundary reports them all as `Unsupported`.
pub fn into_unsupported(error: AudioError) -> candle_audio::gen_core::Error {
    match error {
        AudioError::Canceled => candle_audio::gen_core::Error::Canceled,
        other => candle_audio::gen_core::Error::Unsupported(other.to_string()),
    }
}

/// Expand upstream's bracket-range syntax into plain substrings.
///
/// `layers.[0-3].cross_attn` becomes four substrings, one per index. Multiple brackets expand as a
/// cartesian product, so `layers.[0-1].blocks.[0-1]` yields four. A pattern with no bracket is
/// returned unchanged, and a malformed or inverted bracket is an error rather than a silent
/// literal-substring match — a filter that quietly matches nothing is exactly how an adapter
/// no-ops.
pub fn expand_bracket_ranges(pattern: &str) -> Result<Vec<String>> {
    let Some(open) = pattern.find('[') else {
        if pattern.contains(']') {
            return Err(AudioError::Msg(format!(
                "adapter filter {pattern:?} has a closing bracket with no opening bracket"
            )));
        }
        return Ok(vec![pattern.to_string()]);
    };
    let Some(close_offset) = pattern[open..].find(']') else {
        return Err(AudioError::Msg(format!(
            "adapter filter {pattern:?} has an unterminated bracket range"
        )));
    };
    let close = open + close_offset;
    let body = &pattern[open + 1..close];
    let (low, high) = body.split_once('-').ok_or_else(|| {
        AudioError::Msg(format!(
            "adapter filter {pattern:?} has bracket body {body:?}; expected LOW-HIGH"
        ))
    })?;
    let low: usize = low.trim().parse().map_err(|_| {
        AudioError::Msg(format!(
            "adapter filter {pattern:?} has a non-numeric range start {low:?}"
        ))
    })?;
    let high: usize = high.trim().parse().map_err(|_| {
        AudioError::Msg(format!(
            "adapter filter {pattern:?} has a non-numeric range end {high:?}"
        ))
    })?;
    if high < low {
        return Err(AudioError::Msg(format!(
            "adapter filter {pattern:?} has an inverted range [{low}-{high}]"
        )));
    }
    let mut out = Vec::new();
    for index in low..=high {
        let substituted = format!("{}{index}{}", &pattern[..open], &pattern[close + 1..]);
        out.extend(expand_bracket_ranges(&substituted)?);
    }
    Ok(out)
}

/// Is this checkpoint key something an adapter is allowed to target?
///
/// Three conditions, all necessary: it is under [`ADAPTABLE_PREFIXES`] (so `pretransform.…` — SAME —
/// is out), it ends in `.weight`, and it is a 2-D Linear or a 3-D Conv1d. Norm gains, biases and
/// embeddings are 1-D and therefore never targets — **none of the eight types is a bias-diff type**,
/// so there is nothing they could mean.
pub fn is_adaptable_target(key: &str, shape: &[usize]) -> bool {
    if !ADAPTABLE_PREFIXES
        .iter()
        .any(|prefix| key.starts_with(prefix))
    {
        return false;
    }
    if !key.ends_with(".weight") {
        return false;
    }
    matches!(shape.len(), 2 | 3)
}

/// The set of adaptable targets in a checkpoint, read from the safetensors header alone.
///
/// No tensor data is touched. The result is the *only* thing an adapter is matched against, so a
/// module naming a key outside it fails at plan time — before a single weight is materialized.
pub fn adaptable_targets(header: &BTreeMap<String, Vec<usize>>) -> BTreeMap<String, Vec<usize>> {
    header
        .iter()
        .filter(|(key, shape)| is_adaptable_target(key, shape))
        .map(|(key, shape)| (key.clone(), shape.clone()))
        .collect()
}

/// One adapter's contribution to one target weight, resolved and ready to fold.
#[derive(Clone, Debug)]
pub struct AdapterOp {
    pub kind: AdapterType,
    /// `(alpha / rank) * scale`, folded into the reconstruction rather than applied after it.
    pub effective_scale: f64,
    pub rank: usize,
    pub module: AdapterModule,
    /// Which entry of [`candle_audio::gen_core::LoadSpec::adapters`] this came from, for error
    /// messages. This is [`LoadedAdapter::spec_index`] — the position in the caller's **original**
    /// request, not in the zero-scale-filtered slice the surviving plan is built from.
    pub adapter_index: usize,
}

/// The complete resolved stack: for each target key, the ops to fold in request order.
#[derive(Clone, Debug, Default)]
pub struct AdapterPlan {
    by_target: BTreeMap<String, Vec<AdapterOp>>,
}

impl AdapterPlan {
    /// Is there nothing to do? An empty plan is not installed at all, which is what makes a
    /// `scale == 0.0` request byte-identical to an un-adapted load rather than merely close.
    pub fn is_empty(&self) -> bool {
        self.by_target.is_empty()
    }

    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.by_target.keys().map(String::as_str)
    }

    pub fn ops_for(&self, key: &str) -> Option<&[AdapterOp]> {
        self.by_target.get(key).map(Vec::as_slice)
    }

    /// How many folds this plan performs in total, across every target.
    pub fn op_count(&self) -> usize {
        self.by_target.values().map(Vec::len).sum()
    }

    /// Resolve loaded adapters against a checkpoint's adaptable targets, in request order.
    ///
    /// # What this refuses, and why each refusal is loud
    ///
    /// * A module naming a key that is not an adaptable target — `no_target_matched`. This is the
    ///   key-mapping-mismatch case: the alternative is an adapter that loads, runs, and changes
    ///   nothing.
    /// * A module naming a target the adapter's **own** filters exclude. `include`/`exclude` come
    ///   from the adapter's metadata and describe the module set it was trained over, so a module
    ///   outside that set is an internal contradiction in the file, not a caller's subsetting
    ///   choice. Refusing it is what makes "every adapter tensor consumed exactly once" checkable.
    /// * Zero matched targets. Reachable **only** through this function's own public signature,
    ///   with a hand-built module-less [`LoadedAdapter`]: on every shipped path the per-module rule
    ///   above is strictly stronger — an unmatched module returns early, so a loop that completes
    ///   has `matched >= 1` — and [`load_adapter`] never yields a module-less adapter in the first
    ///   place, because `finish_modules` refuses one. It is kept as a refusal rather than a
    ///   `debug_assert!` precisely because `build` is public and its input is caller-constructed;
    ///   `a_module_less_adapter_is_refused_by_build` is the case that holds it live.
    /// * A factor whose shape disagrees with the target weight.
    ///
    /// A `scale == 0.0` adapter is **fully loaded and fully validated** — it reaches here and every
    /// check above runs — and then contributes no ops. Validation is not the thing being skipped;
    /// arithmetic is.
    pub fn build(
        adapters: &[LoadedAdapter],
        targets: &BTreeMap<String, Vec<usize>>,
    ) -> Result<Self> {
        let mut by_target: BTreeMap<String, Vec<AdapterOp>> = BTreeMap::new();
        for adapter in adapters {
            // `adapter.spec_index`, never `enumerate()`: this function is called a second time over
            // the zero-scale-filtered slice, where a positional index would be the wrong one.
            let adapter_index = adapter.spec_index;
            let mut matched = 0usize;
            for (key, module) in &adapter.modules {
                let Some(shape) = targets.get(key) else {
                    return Err(AudioError::Msg(format!(
                        "no_target_matched: adapter {} module {key:?} names a key that is not an \
                         adaptable Stable Audio 3 target. Adaptable targets are 2-D Linear and 3-D \
                         Conv1d `.weight` tensors under {ADAPTABLE_PREFIXES:?}; the checkpoint has \
                         {} of them.",
                        adapter.path.display(),
                        targets.len()
                    )));
                };
                if !passes_filters(key, &adapter.include, &adapter.exclude) {
                    return Err(AudioError::Msg(format!(
                        "no_target_matched: adapter {} carries a module for {key:?} but its own \
                         include/exclude filters exclude that key, so its tensors would never be \
                         consumed",
                        adapter.path.display()
                    )));
                }
                validate_module_against_target(adapter, key, module, shape)?;
                by_target.entry(key.clone()).or_default().push(AdapterOp {
                    kind: adapter.kind,
                    effective_scale: (adapter.alpha as f64 / adapter.rank as f64)
                        * adapter.scale as f64,
                    rank: adapter.rank,
                    module: module.clone(),
                    adapter_index,
                });
                matched += 1;
            }
            if matched == 0 {
                return Err(AudioError::Msg(format!(
                    "no_target_matched: adapter {} matched none of the checkpoint's {} adaptable \
                     targets",
                    adapter.path.display(),
                    targets.len()
                )));
            }
        }
        Ok(Self { by_target })
    }
}

fn passes_filters(key: &str, include: &[String], exclude: &[String]) -> bool {
    if !include.is_empty() && !include.iter().any(|needle| key.contains(needle.as_str())) {
        return false;
    }
    if exclude.iter().any(|needle| key.contains(needle.as_str())) {
        return false;
    }
    true
}

/// The `[out, in]` view of a target weight: Conv1d `[out, in, k]` flattens to `[out, in*k]`.
///
/// The flatten is exactly reversible and the restore is asserted against the original shape, so a
/// Conv1d adapter cannot silently land transposed — which for a `k == 1` kernel (both SA3 Conv1d
/// targets) would otherwise be invisible on shape alone.
fn matrix_shape(shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [out, input] => Ok((*out, *input)),
        [out, input, kernel] => Ok((*out, input * kernel)),
        other => Err(AudioError::Msg(format!(
            "adapter target has rank-{} shape {other:?}; expected a 2-D Linear or 3-D Conv1d",
            other.len()
        ))),
    }
}

fn validate_module_against_target(
    adapter: &LoadedAdapter,
    key: &str,
    module: &AdapterModule,
    shape: &[usize],
) -> Result<()> {
    let (out_features, in_features) = matrix_shape(shape)?;
    let rank = adapter.rank;
    if rank == 0 || rank > out_features.min(in_features) {
        return Err(AudioError::Msg(format!(
            "adapter {} declares rank {rank} for {key:?}, outside 1..={} for a {out_features}x\
             {in_features} weight",
            adapter.path.display(),
            out_features.min(in_features)
        )));
    }
    if adapter.kind.is_xs() {
        let m = module.m.as_ref().ok_or_else(|| {
            AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} has no `lora_M` core",
                adapter.path.display(),
                adapter.kind.as_str()
            ))
        })?;
        expect_shape(adapter, key, "lora_M", m, &[rank, rank])?;
        if module.a.is_some() || module.b.is_some() {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} also carries lora_A/lora_B factors; an -xs \
                 adapter's bases come from the base weight, not from the file",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
    } else {
        let a = module.a.as_ref().ok_or_else(|| {
            AudioError::Msg(format!(
                "adapter {} module {key:?} has no `lora_A` factor",
                adapter.path.display()
            ))
        })?;
        let b = module.b.as_ref().ok_or_else(|| {
            AudioError::Msg(format!(
                "adapter {} module {key:?} has no `lora_B` factor",
                adapter.path.display()
            ))
        })?;
        expect_shape(adapter, key, "lora_A", a, &[rank, in_features])?;
        expect_shape(adapter, key, "lora_B", b, &[out_features, rank])?;
        if module.m.is_some() {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} carries an -xs `lora_M` core",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
    }

    match (adapter.kind.uses_row_magnitude(), &module.magnitude_r) {
        (true, Some(tensor)) => expect_shape(adapter, key, "magnitude_r", tensor, &[out_features])?,
        (true, None) => {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} has no `magnitude_r`",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
        (false, Some(_)) => {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} carries a `magnitude_r` it cannot use",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
        (false, None) => {}
    }
    match (adapter.kind.uses_col_magnitude(), &module.magnitude_c) {
        (true, Some(tensor)) => expect_shape(adapter, key, "magnitude_c", tensor, &[in_features])?,
        (true, None) => {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} has no `magnitude_c`",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
        (false, Some(_)) => {
            return Err(AudioError::Msg(format!(
                "adapter {} is {} but module {key:?} carries a `magnitude_c` it cannot use",
                adapter.path.display(),
                adapter.kind.as_str()
            )));
        }
        (false, None) => {}
    }
    Ok(())
}

fn expect_shape(
    adapter: &LoadedAdapter,
    key: &str,
    factor: &str,
    tensor: &Tensor,
    expected: &[usize],
) -> Result<()> {
    // Magnitudes are accepted as `[n]` or as the keepdim form (`[n, 1]` / `[1, n]`); everything
    // else must match exactly.
    let dims = tensor.dims();
    let squeezed: Vec<usize> = dims.iter().copied().filter(|dim| *dim != 1).collect();
    let expected_squeezed: Vec<usize> = expected.iter().copied().filter(|dim| *dim != 1).collect();
    let ok = if expected.len() == 1 {
        squeezed == expected_squeezed
    } else {
        dims == expected
    };
    if !ok {
        return Err(AudioError::Msg(format!(
            "adapter {} module {key:?} factor {factor} has shape {dims:?}, expected {expected:?}",
            adapter.path.display()
        )));
    }
    Ok(())
}

/// Fold a stack of ops into one base weight, in order, and return the adapted weight.
///
/// `base` is the checkpoint tensor at its persisted shape (2-D or 3-D). Everything here runs in
/// F32; the result is returned at `base`'s dtype and shape, so the caller sees a drop-in
/// replacement.
///
/// # `scale == 0.0`
///
/// Never reaches here: [`AdapterPlan::build`]'s callers drop zero-scale adapters before an op is
/// created, so a zero-scale stack produces an *empty plan* and the wrapper is not installed at all.
/// That is the difference between "multiplies by zero" and "does not run" — the former turns a
/// `NaN` or an `inf` in the adapter into a `NaN` in the weight, and is not bit-identical for a
/// denormal base either.
pub fn apply_adapter_ops(base: &Tensor, ops: &[AdapterOp]) -> Result<Tensor> {
    if ops.is_empty() {
        return Ok(base.clone());
    }
    let original_dtype = base.dtype();
    let original_shape = base.dims().to_vec();
    let (out_features, in_features) = matrix_shape(&original_shape)?;
    let mut weight = base
        .to_dtype(DType::F32)?
        .reshape((out_features, in_features))?;

    for op in ops {
        weight = fold_one(&weight, op, out_features, in_features)?;
    }

    Ok(weight.reshape(original_shape)?.to_dtype(original_dtype)?)
}

fn fold_one(
    weight: &Tensor,
    op: &AdapterOp,
    out_features: usize,
    in_features: usize,
) -> Result<Tensor> {
    let scale = op.effective_scale as f32;
    let delta = if op.kind.is_xs() {
        // U and V are the base weight's own top-`rank` singular vectors — recomputed here rather
        // than read from the file, because they are not in the file. For a *stacked* adapter this
        // is the weight as it enters this op, which is what makes "fold sequentially" and "attach
        // parametrizations sequentially" the same operation.
        let (u, v) = svd_bases(weight, op.rank, out_features, in_features)?;
        let m = op
            .module
            .m
            .as_ref()
            .expect("validated: -xs module carries lora_M")
            .to_dtype(DType::F32)?;
        u.matmul(&m)?.matmul(&v.t()?)?
    } else {
        let a = op
            .module
            .a
            .as_ref()
            .expect("validated: non-xs module carries lora_A")
            .to_dtype(DType::F32)?;
        let b = op
            .module
            .b
            .as_ref()
            .expect("validated: non-xs module carries lora_B")
            .to_dtype(DType::F32)?;
        b.matmul(&a)?
    };
    let mut adapted = (weight + (delta * scale as f64)?)?;

    // The magnitudes rescale the *adapted* weight, so the strength sits inside the normalization —
    // this is precisely why the family does not commute.
    if op.kind.uses_row_magnitude() {
        let magnitude = op
            .module
            .magnitude_r
            .as_ref()
            .expect("validated: row type carries magnitude_r")
            .to_dtype(DType::F32)?
            .reshape((out_features, 1))?;
        let norm = row_norm(&adapted)?;
        adapted = adapted.broadcast_div(&norm)?.broadcast_mul(&magnitude)?;
    }
    if op.kind.uses_col_magnitude() {
        let magnitude = op
            .module
            .magnitude_c
            .as_ref()
            .expect("validated: column type carries magnitude_c")
            .to_dtype(DType::F32)?
            .reshape((1, in_features))?;
        let norm = col_norm(&adapted)?;
        adapted = adapted.broadcast_div(&norm)?.broadcast_mul(&magnitude)?;
    }
    Ok(adapted)
}

/// `[out, 1]` Euclidean norm of each row, floored away from zero.
///
/// The floor is `f32::MIN_POSITIVE` rather than a tuned epsilon: a genuinely zero row makes the
/// normalized direction undefined, and dividing by the smallest positive normal yields a finite
/// (if enormous) value that the subsequent magnitude multiply scales back, whereas dividing by
/// zero yields `NaN` that propagates through the entire checkpoint.
fn row_norm(weight: &Tensor) -> Result<Tensor> {
    let squared = weight.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(squared.clamp(f32::MIN_POSITIVE, f32::INFINITY)?)
}

fn col_norm(weight: &Tensor) -> Result<Tensor> {
    let squared = weight.sqr()?.sum_keepdim(0)?.sqrt()?;
    Ok(squared.clamp(f32::MIN_POSITIVE, f32::INFINITY)?)
}

/// The base weight's top-`rank` singular bases, as `(U [out, rank], V [in, rank])` F32 tensors.
///
/// The decomposition itself is [`crate::svd::jacobi_svd_top_k`] on the host in `f64`; see that
/// module for why it never runs on the accelerator.
pub fn svd_bases(
    weight: &Tensor,
    rank: usize,
    out_features: usize,
    in_features: usize,
) -> Result<(Tensor, Tensor)> {
    let device = weight.device().clone();
    let host: Vec<f32> = weight
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let host: Vec<f64> = host.into_iter().map(f64::from).collect();
    let decomposed = crate::svd::jacobi_svd_top_k(&host, out_features, in_features, rank)
        .map_err(AudioError::Msg)?;
    let u: Vec<f32> = decomposed.u.iter().map(|value| *value as f32).collect();
    let v: Vec<f32> = decomposed.v.iter().map(|value| *value as f32).collect();
    let u = Tensor::from_vec(u, (out_features, rank), &device)?;
    let v = Tensor::from_vec(v, (in_features, rank), &device)?;
    Ok((u, v))
}

/// A [`candle_nn::var_builder::SimpleBackend`] that folds an [`AdapterPlan`] into the weights it
/// serves.
///
/// # Why a backend wrapper rather than a post-load pass
///
/// `stable_audio_3_medium` is 1.45B parameters. Materializing the whole checkpoint to fold into it
/// would cost ~5.8 GB of host memory before the graph is even built, and the no-adapter path must
/// stay a zero-copy mmap. Wrapping the backend makes the fold **per-module and lazy**: each weight
/// is adapted at the moment the model asks for it, temporaries are released immediately, and every
/// tensor the plan does not name is served by the inner backend untouched — the same object it
/// would have been with no adapters at all.
pub struct AdapterBackend {
    inner: Box<dyn candle_nn::var_builder::SimpleBackend>,
    plan: AdapterPlan,
}

impl AdapterBackend {
    /// Wrap `inner`. Callers must not construct this with an empty plan: an empty plan means the
    /// wrapper should not exist, which is what keeps `scale == 0.0` byte-identical.
    pub fn new(inner: Box<dyn candle_nn::var_builder::SimpleBackend>, plan: AdapterPlan) -> Self {
        debug_assert!(
            !plan.is_empty(),
            "an empty adapter plan must not be installed; the un-adapted backend is the identity"
        );
        Self { inner, plan }
    }
}

impl candle_nn::var_builder::SimpleBackend for AdapterBackend {
    fn get(
        &self,
        shape: candle_audio::candle_core::Shape,
        name: &str,
        hints: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_audio::candle_core::Result<Tensor> {
        let base = self.inner.get(shape, name, hints, dtype, device)?;
        match self.plan.ops_for(name) {
            Some(ops) => apply_adapter_ops(&base, ops)
                .map_err(|error| candle_audio::candle_core::Error::Msg(error.to_string())),
            None => Ok(base),
        }
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_audio::candle_core::Result<Tensor> {
        let base = self.inner.get_unchecked(name, dtype, device)?;
        match self.plan.ops_for(name) {
            Some(ops) => apply_adapter_ops(&base, ops)
                .map_err(|error| candle_audio::candle_core::Error::Msg(error.to_string())),
            None => Ok(base),
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

/// Read one adapter file (native or PEFT) at the caller's requested scale.
///
/// `AdapterKind::Lokr` is refused here rather than at the provider boundary so there is exactly one
/// place that decides: no Stable Audio 3 adapter of any published type is a Kronecker
/// factorization, and accepting one would mean guessing a decomposition the family does not have.
pub fn load_adapter(spec: &AdapterSpec, device: &Device) -> Result<LoadedAdapter> {
    validate_spec_shape(spec)?;
    match AdapterSource::classify(&spec.path)? {
        AdapterSource::NativeFile => load_native(&spec.path, spec.scale, device),
        AdapterSource::PeftDirectory => load_peft(&spec.path, spec.scale, device),
    }
}

/// Everything about one [`AdapterSpec`] that can be judged without opening a file.
///
/// Split out from [`load_adapter`] because `load_variant` calls it **before** the snapshot
/// directory is even opened: a `Lokr` request, or one carrying another backend's knob, is a
/// statement about the request that no amount of reading the checkpoint would change, and the
/// caller should hear about it immediately rather than after a snapshot hash.
///
/// The two rejected knobs are deliberate and are *not* repurposed. `pass_scales` is LTX-2.3's
/// per-denoise-stage schedule and `moe_expert` selects one expert of Wan2.2's dual-expert MoE;
/// neither has a Stable Audio 3 meaning, and quietly ignoring a field the caller set is how a knob
/// comes to "work" while doing nothing.
pub fn validate_spec_shape(spec: &AdapterSpec) -> Result<()> {
    if spec.kind != AdapterKind::Lora {
        return Err(unsupported(format!(
            "Stable Audio 3 adapters are the native lora/dora/bora family; {:?} is not one of them",
            spec.kind
        )));
    }
    if !spec.scale.is_finite() {
        return Err(AudioError::Msg(format!(
            "adapter {} has a non-finite scale ({})",
            spec.path.display(),
            spec.scale
        )));
    }
    if spec.pass_scales.is_some() {
        return Err(unsupported(format!(
            "adapter {} sets `pass_scales`, which is LTX-2.3's per-denoise-stage schedule; Stable \
             Audio 3 runs a single-pass denoise and folds adapters at load time",
            spec.path.display()
        )));
    }
    if spec.moe_expert.is_some() {
        return Err(unsupported(format!(
            "adapter {} sets `moe_expert`, which selects one expert of Wan2.2's dual-expert MoE; \
             Stable Audio 3 has a single denoiser",
            spec.path.display()
        )));
    }
    Ok(())
}

/// This crate's native adapter layout.
///
/// Metadata (safetensors `__metadata__`):
///
/// | key | required | meaning |
/// |---|---|---|
/// | `adapter_type` | yes | one of the eight canonical names, or `dora` / `dora-xs` |
/// | `rank` | yes | positive integer, `<= min(out, in)` of every target |
/// | `alpha` | yes | finite float |
/// | `include` | no | comma-separated substrings, bracket ranges allowed |
/// | `exclude` | no | as `include` |
///
/// Tensors are `"{target}.{index}.{factor}"`, where `{target}` is the checkpoint key with its
/// trailing `.weight` removed, `{index}` is the adapter's slot (upstream stacks several per module;
/// a single-adapter file uses `0`), and `{factor}` is one of `lora_A`, `lora_B`, `lora_M`,
/// `magnitude_r`, `magnitude_c`.
pub const NATIVE_FACTORS: &[&str] = &["lora_A", "lora_B", "lora_M", "magnitude_r", "magnitude_c"];

fn load_native(path: &Path, scale: f32, device: &Device) -> Result<LoadedAdapter> {
    let metadata = read_safetensors_metadata(path)?;
    // Safety: the adapter file must remain immutable while it is being read; every tensor is copied
    // into owned storage before this function returns.
    let file = unsafe { MmapedSafetensors::new(path)? };
    let names: Vec<String> = file.tensors().into_iter().map(|(name, _)| name).collect();
    let (modules, has_row, has_col) = collect_native_modules(path, &file, &names, device)?;

    let declared = metadata.get("adapter_type").ok_or_else(|| {
        AudioError::Msg(format!(
            "{} has no `adapter_type` in its safetensors metadata",
            path.display()
        ))
    })?;
    let kind = resolve_adapter_type(declared, has_row, has_col)?;
    let rank = parse_rank(path, metadata.get("rank"))?;
    let alpha = parse_alpha(path, metadata.get("alpha"))?;
    let include = parse_filter(metadata.get("include"))?;
    let exclude = parse_filter(metadata.get("exclude"))?;

    Ok(LoadedAdapter {
        path: path.to_path_buf(),
        kind,
        rank,
        alpha,
        scale,
        include,
        exclude,
        modules,
        // `plan_for` stamps the caller-visible position; one path in isolation has no stack.
        spec_index: 0,
    })
}

type NativeModules = (BTreeMap<String, AdapterModule>, bool, bool);

fn collect_native_modules(
    path: &Path,
    file: &MmapedSafetensors,
    names: &[String],
    device: &Device,
) -> Result<NativeModules> {
    let mut staged: BTreeMap<String, BTreeMap<String, Tensor>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for name in names {
        // `"{stem}.{index}.{factor}"`; the factor is the last segment and the index the one before.
        let (head, factor) = name.rsplit_once('.').ok_or_else(|| {
            AudioError::Msg(format!(
                "{} has unparseable tensor {name:?}",
                path.display()
            ))
        })?;
        if !NATIVE_FACTORS.contains(&factor) {
            return Err(AudioError::Msg(format!(
                "{} carries tensor {name:?}, whose trailing segment {factor:?} is not one of \
                 {NATIVE_FACTORS:?}; every tensor in an adapter must be consumed",
                path.display()
            )));
        }
        let (stem, index) = head.rsplit_once('.').ok_or_else(|| {
            AudioError::Msg(format!(
                "{} has tensor {name:?} without an adapter index segment (expected \
                 \"{{target}}.{{index}}.{{factor}}\")",
                path.display()
            ))
        })?;
        if index.parse::<usize>().is_err() {
            return Err(AudioError::Msg(format!(
                "{} has tensor {name:?} whose index segment {index:?} is not an integer",
                path.display()
            )));
        }
        if !seen.insert(name.clone()) {
            return Err(AudioError::Msg(format!(
                "{} carries {name:?} twice",
                path.display()
            )));
        }
        let target = format!("{stem}.weight");
        let tensor = file.load(name, device)?;
        finite_or_err(path, name, &tensor)?;
        if staged
            .entry(target)
            .or_default()
            .insert(factor.to_string(), tensor)
            .is_some()
        {
            return Err(AudioError::Msg(format!(
                "{} declares factor {factor:?} for {stem:?} more than once (a second adapter index \
                 for the same module is not supported; stack two adapter files instead)",
                path.display()
            )));
        }
    }
    finish_modules(path, staged)
}

fn finish_modules(
    path: &Path,
    staged: BTreeMap<String, BTreeMap<String, Tensor>>,
) -> Result<NativeModules> {
    let mut modules = BTreeMap::new();
    let mut has_row = false;
    let mut has_col = false;
    for (target, mut factors) in staged {
        let module = AdapterModule {
            a: factors.remove("lora_A"),
            b: factors.remove("lora_B"),
            m: factors.remove("lora_M"),
            magnitude_r: factors.remove("magnitude_r"),
            magnitude_c: factors.remove("magnitude_c"),
        };
        has_row |= module.magnitude_r.is_some();
        has_col |= module.magnitude_c.is_some();
        debug_assert!(factors.is_empty());
        modules.insert(target, module);
    }
    if modules.is_empty() {
        return Err(AudioError::Msg(format!(
            "{} contains no adapter modules",
            path.display()
        )));
    }
    Ok((modules, has_row, has_col))
}

fn finite_or_err(path: &Path, name: &str, tensor: &Tensor) -> Result<()> {
    let values: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    if let Some(bad) = values.iter().find(|value| !value.is_finite()) {
        return Err(AudioError::Msg(format!(
            "{} tensor {name:?} contains a non-finite value ({bad})",
            path.display()
        )));
    }
    Ok(())
}

fn load_peft(path: &Path, scale: f32, device: &Device) -> Result<LoadedAdapter> {
    let config_path = path.join(PEFT_CONFIG_FILE);
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| AudioError::Msg(format!("read {}: {e}", config_path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| AudioError::Msg(format!("parse {}: {e}", config_path.display())))?;

    let rank = config
        .get("r")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            AudioError::Msg(format!(
                "{} has no positive integer `r`",
                config_path.display()
            ))
        })? as usize;
    let alpha = config
        .get("lora_alpha")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| AudioError::Msg(format!("{} has no `lora_alpha`", config_path.display())))?
        as f32;
    if !alpha.is_finite() {
        return Err(AudioError::Msg(format!(
            "{} has a non-finite `lora_alpha`",
            config_path.display()
        )));
    }
    let declared = config
        .get("sa3_adapter_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("lora")
        .to_string();
    let include = json_filter(config.get("target_modules"))?;
    let exclude = json_filter(config.get("exclude_modules"))?;

    let weights_path = path.join(PEFT_WEIGHTS_FILE);
    // Safety: as `load_native`.
    let file = unsafe { MmapedSafetensors::new(&weights_path)? };
    let names: Vec<String> = file.tensors().into_iter().map(|(name, _)| name).collect();

    let mut staged: BTreeMap<String, BTreeMap<String, Tensor>> = BTreeMap::new();
    for name in &names {
        let stripped = name
            .strip_prefix("base_model.model.")
            .unwrap_or(name.as_str());
        let (head, tail) = stripped.rsplit_once('.').ok_or_else(|| {
            AudioError::Msg(format!(
                "{} has unparseable tensor {name:?}",
                weights_path.display()
            ))
        })?;
        // PEFT spells factors as `<module>.lora_A.weight`; the magnitudes and the -xs core reuse
        // the same shape so the whole family round-trips through a PEFT directory.
        let (stem, factor) = if tail == "weight" {
            head.rsplit_once('.').ok_or_else(|| {
                AudioError::Msg(format!(
                    "{} has tensor {name:?} with no factor segment",
                    weights_path.display()
                ))
            })?
        } else {
            (head, tail)
        };
        if !NATIVE_FACTORS.contains(&factor) {
            return Err(AudioError::Msg(format!(
                "{} carries tensor {name:?}, whose factor segment {factor:?} is not one of \
                 {NATIVE_FACTORS:?}",
                weights_path.display()
            )));
        }
        let tensor = file.load(name, device)?;
        finite_or_err(&weights_path, name, &tensor)?;
        if staged
            .entry(format!("{stem}.weight"))
            .or_default()
            .insert(factor.to_string(), tensor)
            .is_some()
        {
            return Err(AudioError::Msg(format!(
                "{} declares factor {factor:?} for {stem:?} more than once",
                weights_path.display()
            )));
        }
    }
    let (modules, has_row, has_col) = finish_modules(&weights_path, staged)?;
    let kind = resolve_adapter_type(&declared, has_row, has_col)?;
    if rank == 0 {
        return Err(AudioError::Msg(format!(
            "{} declares rank 0",
            config_path.display()
        )));
    }

    Ok(LoadedAdapter {
        path: path.to_path_buf(),
        kind,
        rank,
        alpha,
        scale,
        include,
        exclude,
        modules,
        // `plan_for` stamps the caller-visible position; one path in isolation has no stack.
        spec_index: 0,
    })
}

fn json_filter(value: Option<&serde_json::Value>) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let raw: Vec<String> = match value {
        serde_json::Value::Null => return Ok(Vec::new()),
        serde_json::Value::String(single) => vec![single.clone()],
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    AudioError::Msg("PEFT module filter entries must be strings".to_string())
                })
            })
            .collect::<Result<Vec<_>>>()?,
        other => {
            return Err(AudioError::Msg(format!(
                "PEFT module filter must be a string or array, got {other}"
            )));
        }
    };
    let mut out = Vec::new();
    for pattern in raw {
        out.extend(expand_bracket_ranges(&pattern)?);
    }
    Ok(out)
}

fn parse_filter(value: Option<&String>) -> Result<Vec<String>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for pattern in value.split(',') {
        let pattern = pattern.trim();
        if pattern.is_empty() {
            continue;
        }
        out.extend(expand_bracket_ranges(pattern)?);
    }
    Ok(out)
}

fn parse_rank(path: &Path, value: Option<&String>) -> Result<usize> {
    let value = value.ok_or_else(|| {
        AudioError::Msg(format!(
            "{} has no `rank` in its safetensors metadata",
            path.display()
        ))
    })?;
    let rank: usize = value.trim().parse().map_err(|_| {
        AudioError::Msg(format!("{} has non-integer rank {value:?}", path.display()))
    })?;
    if rank == 0 {
        return Err(AudioError::Msg(format!(
            "{} declares rank 0",
            path.display()
        )));
    }
    Ok(rank)
}

fn parse_alpha(path: &Path, value: Option<&String>) -> Result<f32> {
    let value = value.ok_or_else(|| {
        AudioError::Msg(format!(
            "{} has no `alpha` in its safetensors metadata",
            path.display()
        ))
    })?;
    let alpha: f32 = value.trim().parse().map_err(|_| {
        AudioError::Msg(format!(
            "{} has non-numeric alpha {value:?}",
            path.display()
        ))
    })?;
    if !alpha.is_finite() {
        return Err(AudioError::Msg(format!(
            "{} has a non-finite alpha {alpha}",
            path.display()
        )));
    }
    Ok(alpha)
}

fn read_safetensors_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let header = crate::weights::safetensors_header_object(path)?;
    let Some(metadata) = header.get("__metadata__") else {
        return Ok(HashMap::new());
    };
    let object = metadata.as_object().ok_or_else(|| {
        AudioError::Msg(format!(
            "{} has a non-object safetensors __metadata__",
            path.display()
        ))
    })?;
    Ok(object
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|text| (key.clone(), text.to_string())))
        .collect())
}

/// Load, validate and resolve a whole `LoadSpec` adapter stack against one checkpoint.
///
/// Returns an **empty plan** when every adapter has `scale == 0.0`, which is the whole
/// bit-identity mechanism: an empty plan is never installed, so the request takes the ordinary
/// mmap path and produces the same bytes as a request with no adapters at all. Every adapter is
/// still fully parsed, shape-checked and matched first — `scale == 0.0` skips *arithmetic*, not
/// *validation*.
pub fn plan_for(
    specs: &[AdapterSpec],
    targets: &BTreeMap<String, Vec<usize>>,
    device: &Device,
) -> Result<AdapterPlan> {
    let mut loaded = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let mut adapter = load_adapter(spec, device)?;
        // Stamped here and carried through the zero-scale filter below, so an op built from the
        // filtered slice still reports the caller's original position.
        adapter.spec_index = index;
        loaded.push(adapter);
    }
    // Resolve the FULL stack — zero-scale members included — against the checkpoint. This call is
    // the validation: a zero-scale adapter with a mistyped key, a wrong rank or a shape that does
    // not fit its target is refused here, exactly as a scale-1.0 one would be. Discarding the
    // result is the point; what survives is that it did not error.
    AdapterPlan::build(&loaded, targets)?;

    let active: Vec<LoadedAdapter> = loaded
        .into_iter()
        .filter(|adapter| adapter.scale != 0.0)
        .collect();
    if active.is_empty() {
        return Ok(AdapterPlan::default());
    }
    AdapterPlan::build(&active, targets)
}
