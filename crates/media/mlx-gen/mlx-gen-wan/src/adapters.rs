//! Wan2.2 LoRA application (sc-2683) — wires the reference `mlx_video/lora/` into the Wan
//! generate path plus the Wan key→module map.
//!
//! **Strategy: weight MERGE** (the reference Wan path), not a forward-time residual. The reference
//! `generate_wan.py` applies LoRA through `load_wan_model` → `load_and_apply_loras` →
//! `apply_loras_to_weights`: for a bf16 expert it folds `ΔW = (B·A)·(alpha/rank)·strength` (computed
//! at the factor dtype, cast to the weight dtype) **into the weight** before the forward; for a
//! quantized expert it dequantizes the targeted layers, merges, and replaces them with bf16 Linears.
//! Its runtime-residual `LoRALinear` class exists but is **not** invoked by the Wan path (that is why
//! LTX — where the reference never wired LoRA — chose a residual instead, sc-2687). Merge is faithful
//! to the production reference (the parity gate is "vs a reference-merged golden"), cheap and exact on
//! the bf16 base, has **zero** per-step / forward cost (the [`WanTransformer`](crate::transformer)
//! is untouched and the no-adapter path is trivially byte-identical), and maps directly onto the MoE
//! high/low split — each expert's weight map is merged independently.
//!
//! **MoE high/low.** The reference forms `_loras_low = (loras)+(loras_low)` and
//! `_loras_high = (loras)+(loras_high)` and merges each onto its expert. Mirrored via
//! [`AdapterSpec::moe_expert`](mlx_gen::AdapterSpec): `None` = a shared file (merged onto **both**
//! experts), `Some(High)`/`Some(Low)` = one expert only. [`merge_wan_adapters`] is called once per
//! expert and selects the shared specs first, then this expert's specific ones (the `(loras)+(loras_*)`
//! order), so a module hit by both accumulates in the reference's order.
//!
//! **Format.** PEFT `lora_A`/`lora_B` and kohya `lora_down`/`lora_up`, optional per-module `.alpha`
//! (default = rank), `diffusion_model.`-prefixed (the real SceneWorks Wan2.2 MoE LoRAs ship PEFT,
//! bf16, rank 64, no alpha). `scale = alpha/rank` (the reference `LoRAWeights.scale`). **LoKr**
//! (sc-2393 — net-new; the reference Wan path is LoRA-only) is parsed by the core `parse_lokr`, its
//! per-module `[out,in]` delta reconstructed via `reconstruct_lokr_delta` (`alpha/rank` folded in),
//! and folded into the weight through the **same in-place merge** as LoRA (`merge_one_lokr`). The
//! kohya `lora_unet_`-**flattened** external form is not part of the reference Wan surface (its
//! `_normalize_wan_lora_key` only strips prefixes + renames dotted paths); such keys resolve to no
//! module and are surfaced (never silently dropped) — adding it would be net-new beyond the fork.
//!
//! **Skips, never errors-on-skip.** Mirrors the reference (`apply_loras_to_weights` counts skipped
//! modules, never raises): a LoRA target absent from this checkpoint is reported, not fatal. The
//! caller errors only if a non-empty spec list matched *nothing* across both experts (a format/prefix
//! misconfiguration).

use std::collections::BTreeMap;

use mlx_rs::ops::{add, matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::{
    is_loha_keys, is_lokr, is_lokr_keys, parse_loha_thirdparty, parse_lokr, parse_lokr_thirdparty,
    resolve_lokr_path,
};
use mlx_gen::adapters::{build_lokr_factors, read_scalar_alpha, AdaptableHost, Adapter};
use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
use mlx_gen::runtime::{AdapterApplyReport, AdapterKind, AdapterSpec, MoeExpert};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

// sc-27342: key normalization, the load-time merge path and the forward-time additive path
// live in sibling modules; the report types and family classification stay here.
mod additive;
mod keys;
mod merge;
pub use additive::*;
pub use keys::*;
pub use merge::*;

/// LoRA key namespace prefixes stripped (longest-first), matching the reference
/// `_normalize_wan_lora_key`. SceneWorks' trained Wan LoRAs use `diffusion_model.`.
const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "model.",
];

/// Outcome of merging one expert's adapters: how many module weights were folded, how many specs
/// applied to this expert, and any LoRA module paths that resolved to no weight (surfaced, never
/// silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct WanLoraReport {
    /// Module weights actually merged (one per resolved target, across all applicable specs).
    pub applied: usize,
    /// Specs that applied to this expert (shared + this-expert-specific).
    pub applicable: usize,
    /// LoRA module paths (normalized) that matched no weight in this checkpoint.
    pub skipped: Vec<String>,
    /// The same outcome broken down per applicable spec, in apply order (sc-27451) — what the
    /// providers fold into the contract's per-adapter `Generator::adapter_apply_reports`.
    pub per_spec: Vec<WanSpecOutcome>,
}

/// One applicable spec's share of a [`WanLoraReport`] (sc-27451).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WanSpecOutcome {
    /// Index of the spec in the `specs` slice the merge/install was given (load order).
    pub spec_index: usize,
    /// Module targets this spec merged/installed on this expert.
    pub applied: usize,
    /// This spec's normalized module paths that matched nothing on this expert.
    pub skipped: Vec<String>,
}

/// Apply one spec through `apply` (a fold or an additive install) and record its per-spec share of
/// `report` — the delta in `applied` / `skipped` that call produced (sc-27451).
fn apply_recorded(
    report: &mut WanLoraReport,
    spec_index: usize,
    apply: impl FnOnce(&mut WanLoraReport) -> Result<()>,
) -> Result<()> {
    let (applied_before, skipped_before) = (report.applied, report.skipped.len());
    report.applicable += 1;
    apply(report)?;
    report.per_spec.push(WanSpecOutcome {
        spec_index,
        applied: report.applied - applied_before,
        skipped: report.skipped[skipped_before..].to_vec(),
    });
    Ok(())
}

/// Fold one or more experts' per-spec outcomes ([`WanLoraReport::per_spec`]) into the contract's
/// per-adapter outcomes, one per
/// `specs` entry in load order (sc-27451): `applied` sums the spec's targets across the experts it
/// reached (a shared spec on a dual-expert model counts both), `skipped` is the sorted, de-duplicated
/// union of its unmatched paths. This is the engine-owned result `Generator::adapter_apply_reports`
/// publishes — `warn_skipped_adapters` remains a secondary stderr log.
pub(crate) fn adapter_apply_reports(
    specs: &[AdapterSpec],
    experts: &[&[WanSpecOutcome]],
) -> Vec<AdapterApplyReport> {
    specs
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let mut applied = 0;
            let mut skipped = Vec::new();
            for outcome in experts
                .iter()
                .flat_map(|outcomes| outcomes.iter())
                .filter(|outcome| outcome.spec_index == index)
            {
                applied += outcome.applied;
                skipped.extend(outcome.skipped.iter().cloned());
            }
            skipped.sort();
            skipped.dedup();
            AdapterApplyReport {
                adapter_path: spec.path.clone(),
                applied,
                skipped,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Role {
    Down, // lora_A / lora_down → A [rank, in]
    Up,   // lora_B / lora_up   → B [out, rank]
    Alpha,
}

#[derive(Default)]
struct LoraParts {
    down: Option<Array>,
    up: Option<Array>,
    alpha: Option<f32>,
}

/// PEFT + kohya factor suffixes (exact match). `lora_A`/`lora_down` are the A (down) factor;
/// `lora_B`/`lora_up` the B (up). Mirrors the reference `load_lora_weights`, which accepts both
/// conventions; the `.alpha` scalar is optional (default rank).
const SUFFIXES: [(&str, Role); 5] = [
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// Emit a single, uniform warning for adapter targets that aren't present in the loaded checkpoint —
/// a *partial* skip, distinct from the hard "matched no module" error the model entries return. The
/// three Wan `Generator` load paths (`model.rs` ×2, `model_vace.rs`) share this so the message can't
/// drift (sc-4031). The structured per-adapter outcome is published through
/// `Generator::adapter_apply_reports` (sc-27451, built by [`adapter_apply_reports`]); this stderr
/// line is kept as a secondary log for runs nobody inspects programmatically. A no-op when nothing
/// was skipped.
pub(crate) fn warn_skipped_adapters(model_id: &str, skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    eprintln!(
        "{model_id}: {} adapter target(s) not present in this checkpoint, skipped: {skipped:?}",
        skipped.len()
    );
}

// ============================================================================================
// Additive (UNMERGED) LoRA/LoKr/LoHa install (sc-10044)
// ============================================================================================
//
// The [`merge_wan_adapters`] path above FOLDS every adapter delta into a dense `Weights` map
// (`W += δ`) before [`WanTransformer::from_weights`] builds the model — which is why a *pre-quantized*
// (packed Q4/Q8) snapshot has nowhere to write the delta and the loader hard-errors (`model.rs:614`).
//
// The additive path below is the runtime alternative: it installs each adapter as a **forward-time
// residual** onto an already-built [`WanTransformer`] (any [`AdaptableHost`]) via
// [`Adapter`]-stacking, so the base weight `W` is used AS-IS — dense bf16 **or** packed Q4/Q8, never
// dequantized or mutated. Each linear then computes `base(x) + Σ scale·(x·A)·B` (LoRA) /
// `base(x) + Σ scale·x·ΔWᵀ` (LoKr/LoHa) — two small matmuls in compute precision, deferred (no out×in
// delta materialized, no per-gen dense reload). This makes LoRA usable on a packed snapshot with zero
// change to the base's quant.
//
// The math is byte-parity with the fold path: for LoRA the fold computes `ΔW = (B·A)·(alpha/rank·s)`
// and forwards `x·(W+ΔW)ᵀ = x·Wᵀ + (alpha/rank·s)·x·(B·A)ᵀ`; the residual computes
// `s·(x·Aᵀ)·(Bᵀ·(alpha/rank)) = (alpha/rank·s)·x·Aᵀ·Bᵀ` — identical (`(B·A)ᵀ = Aᵀ·Bᵀ`). This mirrors
// the core `install_lora_groups` factor convention exactly (`a = downᵀ`, `b = upᵀ`, `alpha/rank`
// folded into `b`, `scale = strength`) so a fold and an additive install of the same file agree
// within Metal matmul tolerance. LoKr/LoHa reconstruct the same `[out,in]` delta the fold uses and
// stack it as an [`Adapter::Lokr`] residual (`scale = strength`, `alpha/rank` already baked into the
// delta), with the base shape recovered from the host (works for a packed base too — see
// [`mlx_gen::adapters::AdaptableLinear::base_shape`]).
//
// Format coverage mirrors [`merge_one`]: PEFT/kohya LoRA, peft LoKr, third-party LyCORIS LoKr/LoHa.
// A target absent from (or outside the adaptable surface of) the host is surfaced (skipped), never
// fatal — matching the fold path's `skipped` reporting.

/// The adapter family a file resolves to, for the packed-snapshot routing (sc-10045/sc-10050).
/// `classify` reads the file's keys/metadata exactly as `install_one_additive` does, so a spec
/// routes the same way it will install: plain LoRA and **LoKr** both apply additively on a packed tier
/// (LoRA via low-rank factors, LoKr via the structured vec-trick, sc-10050); **LoHa** on a packed tier
/// is still deferred (sc-10051), so the loader rejects only LoHa up front rather than dequantizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WanAdapterFamily {
    /// PEFT / kohya LoRA (`lora_A/B` or `lora_down/up`) — additive on any tier.
    Lora,
    /// LoKr (peft `networkType=lokr`, or third-party `lokr_*` keys) — additive on any tier: the
    /// **structured deferred** Kronecker install on a packed base (sc-10050), the materialized-delta
    /// residual / fold on a dense base.
    Lokr,
    /// LoHa (third-party `hada_*` keys) — deferred on a packed tier to sc-10051; dense-only for now.
    Loha,
}

/// Classify one adapter file into a [`WanAdapterFamily`] using the SAME precedence
/// [`install_one_additive`]/[`merge_one`] dispatch on: an explicit `AdapterKind::Lokr` spec kind, a
/// peft LoKr metadata stamp ([`is_lokr`]), third-party LoKr keys ([`is_lokr_keys`]), or LoHa keys
/// ([`is_loha_keys`]) — anything else is plain LoRA. LoKr precedes LoHa (a file is never both). Reads
/// the file once (the caller reuses the load).
fn classify_family(lw: &Weights, kind: AdapterKind) -> WanAdapterFamily {
    if kind == AdapterKind::Lokr || is_lokr(lw) || is_lokr_keys(lw) {
        WanAdapterFamily::Lokr
    } else if is_loha_keys(lw) {
        WanAdapterFamily::Loha
    } else {
        WanAdapterFamily::Lora
    }
}

/// Reject a **LoHa** (LyCORIS Hadamard) adapter on a **pre-quantized (packed Q4/Q8)** Wan snapshot
/// with an explicit, actionable **typed** error (sc-10045; narrowed by sc-10050; finalized by
/// sc-10051, Option A). Plain LoRA AND LoKr now install additively onto a packed base with no dequant
/// — LoRA via its low-rank factors, LoKr via the structured deferred-Kronecker vec-trick
/// (`install_one_lokr_additive`, sc-10050). LoHa is the **only** family that cannot: its delta is
/// `ΔW = (B₁A₁) ⊙ (B₂A₂)`, and the element-wise (Hadamard) product entangles the two low-rank
/// branches so there is **no allocation-free deferred form** — evaluating it requires materializing
/// the full `[out,in]` bf16 delta (~dense-weight-sized, ~28 GB/expert on a 14B Wan tier). On a packed
/// tier that means holding packed-W (~7 GB/expert) **plus** that dense delta, negating the whole
/// low-memory point of the quantized tier. So rather than silently OOM or silently dequantize, the
/// loader **steers the user to the bf16 tier** (where LoHa folds into the dense weight in place) via a
/// typed [`Error::Unsupported`] — which bridges 1:1 to [`mlx_gen::Error::Unsupported`] so the worker
/// can surface actionable guidance instead of an opaque failure (epic 3720, F-008).
///
/// `model_id` is the registry id for the message. Scans every non-empty spec (each file loaded once)
/// and errors on the first LoHa; a LoRA/LoKr set passes through untouched. No-op on a dense base (the
/// caller only calls this when the snapshot is packed).
pub fn reject_loha_on_packed(model_id: &str, specs: &[AdapterSpec]) -> Result<()> {
    for spec in specs {
        let lw = Weights::from_file(&spec.path)?;
        if classify_family(&lw, spec.kind) == WanAdapterFamily::Loha {
            return Err(Error::Unsupported(format!(
                "{model_id}: this is a LoHa (LyCORIS Hadamard) adapter, which cannot run on a \
                 quantized (packed Q4/Q8) Wan tier. LoHa's delta is a Hadamard product of two \
                 low-rank branches, ΔW = (B₁A₁) ⊙ (B₂A₂); the element-wise product entangles the \
                 factors so there is no deferred/additive form — applying it would have to \
                 materialize the full dense weight-sized delta (~28 GB/expert on a 14B tier) on top \
                 of the packed weights, defeating the quantized tier's memory savings. Use the bf16 \
                 tier for this adapter (where LoHa merges into the dense weight in place), or pick a \
                 plain LoRA / LoKr adapter (which apply additively on any tier, sc-10050). Tracked in \
                 sc-10051 (epic 10043). Offending file: {}",
                spec.path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
