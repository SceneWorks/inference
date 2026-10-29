//! Tensor-neutral memory strategy and provider contract (SC-15449).
//!
//! This module is the inference-side authority for what a provider can do. Providers describe
//! structure, lifecycle hooks, formula shape, backend realization, asset facts, and the
//! calibration ABI/fingerprint. Measured coefficients and envelopes remain external evidence, and
//! the caller remains the sole owner of live-budget accounting and least-cost selection.
//!
//! The separation is deliberate: a provider safety gate may reject a shared selection, but the
//! return type cannot substitute a different strategy or numeric tier. Unknown, stale,
//! fingerprint-mismatched, or out-of-envelope evidence is therefore never turned into a claimed
//! optimized fit by provider-local policy.
//!
//! # Why `memory_strategy` and not `memory` (SC-15804)
//!
//! The contract carried an `Image*` prefix through SC-15449 because the image lane adopted it
//! first. Nothing in the four-rung ladder's mechanisms is image-specific: rung 1 sheds a
//! conditioning component, rung 2 bounds decoder scratch, rung 3 bounds attention, and rung 4
//! bounds transformer residency. Those mechanisms can apply to video and audio providers, so the
//! vocabulary is lane-neutral; that applicability does not claim adoption coverage. At the
//! sc-16595 review point (F-200, 2026-08-01 review), audio had no registered memory-strategy
//! contract. sc-16998 separately tracks the first Stable Audio 3 adoption and non-vacuous audio
//! catalog conformance.
//!
//! It is deliberately **not** the bare name `memory`. Five crates in this workspace already have a
//! `memory` module, meaning two different things: `mlx_gen::memory` is the MLX budget interface
//! (`safe_budget_gib`, `clamp_budget_to_cap`, `apply_memory_cap_env`, `MEMORY_CAP_ENV`) that MLX
//! provider files import directly, while `mlx-gen-sam2`'s `memory` is SAM2's *model* memory bank
//! (`MemoryEncoder`, `MemoryAttention`). A third, `mlx_rs::memory`, is the allocator itself
//! (`clear_cache`, `get_peak_memory`). A `gen_core::memory` would land in the same `use` block as
//! the first and read as the second. `memory_strategy` collides with none of them.
//!
//! The **types** stay bare (`MemoryStrategy`, `MemoryProviderContract`, ...): the existing bare
//! `Memory*` names in the workspace (`MemoryEncoder`, `MemoryAttention`, ...) are SAM2/model-bank
//! concepts in other crates and never appear alongside these.
//!
//! # The cost order is ordinal, and it stays backend-neutral (SC-16090)
//!
//! SC-15791 measured rung 4 at ~8.0 s per denoise step on Candle against ~0.309 s on MLX — ~26× for
//! the same rung on the same file — and asked whether the ladder's cost order must therefore become
//! per-backend, or be replaced by a selector that consumes measured per-backend cost. **Neither. The
//! order is unchanged, static, and shared.** Three independent reasons, each sufficient:
//!
//! 1. **The ladder never trades cost against cost.** The shared selector walks
//!    [`MemoryStrategy::ALL`] and takes the *first* rung whose measured peak fits the live budget —
//!    the cheapest sufficient rung. This is the one authoritative least-cost decision path.
//!    Rung 4 is therefore reached only when no cheaper rung is *selectable*, so it never wins over a
//!    cheaper rung that fits: a cost multiplier on it cannot flip that comparison in either
//!    direction. Its alternative is no render — usually a rejection, or an `Unverified` verdict where
//!    a cheaper rung's peak would have fit but its evidence was stale, out-of-envelope or
//!    fingerprint-mismatched (that path exists and is separately tested; it does not restore a
//!    cheaper *render*, only a cheaper non-answer). Nothing in the ladder consults a cost, which is
//!    why [`MemoryStrategyCapability`] carries no cost field.
//! 2. **A cost *order* is ordinal; 26× is a magnitude, and no consumer of the order reads one.** The
//!    order's whole job is "try the cheaper mechanism first", and it discharges that on any backend
//!    where rung 4 remains the most expensive rung. Per-backend *magnitudes* are already carried
//!    where they belong: per cell, as calibration evidence, keyed by backend. (SC-15791 timed no
//!    Candle rung other than 4, so that rung 4 is still last on Candle is the epic's premise and
//!    remains unmeasured; it is not evidence this decision claims.)
//! 3. **The 26× is not a backend cost.** It decomposes entirely into per-window work that is not
//!    per-window work — see [`MemoryWindowMaterialization`]. Forking the shared order would encode a
//!    fixable implementation defect into the contract 20 family stories then have to honour.
//!
//! ## Where the magnitude reaches a TIER decision, and why that is allowed (SC-16104)
//!
//! Reason 1 is a statement about the *ladder*, and it must not be over-read into "the cost never
//! matters anywhere". One layer up, a caller choosing a numeric **tier** can consume this ladder's
//! verdict as a boolean: SceneWorks' capability-downtier collapses "fits, resident" and "fits, but
//! only by engaging rung 4" into the same `Fits`, then keeps the highest-fidelity tier that fits. So
//! under an automatic tier choice, a tier whose *only* fit is rung 4 is preferred over a lower tier
//! that fits resident, and the cost of that rung is what the higher tier is bought with — tens of
//! seconds per step on q8, extrapolating z-image's measured ~1466 ms/block.
//!
//! **That is intended.** Fidelity the user asked for is not a memory lever: the catalog refuses to
//! substitute precision (tier integrity) and refuses to substitute geometry (SC-15807), and refusing
//! to substitute speed-for-fidelity is the same refusal. It does mean this deliberately **declines**
//! SC-15791's recommendation "do not enable rung 4 on q8 until the repack is hoisted" — gating q8
//! would be the substitution the rule forbids — which raises the value of the repack fix (SC-16096)
//! rather than lowering it. What remains open on SC-16104 is disclosure, not policy: nothing tells a
//! caller its render is slow *because* it is streaming blocks to hold the tier.
//!
//! # A rung declares no non-VRAM resource cost (SC-16090)
//!
//! SC-15791 also asked how a rung declares a **host RAM** cost, because its recommended Candle fix
//! (cache the repacked bytes host-side) holds ~3.16 GiB of host memory for the whole request — a
//! derived figure for a *proposed* fix, not a measurement; nothing in the spike watched host RSS —
//! while MLX's mmap realization pays nothing host-side. A rung whose resource cost changes *kind* per
//! backend cannot be expressed as one cost order.
//!
//! **No such axis is added, because the cost belongs to that one candidate fix rather than to the
//! rung.** A realization that materializes windows from device-format bytes on disk holds its host copy
//! as the mapped file — reclaimable page cache, the same kind of resource MLX's mmap is — and makes no
//! anonymous per-request allocation at all. Adding a resource-kind axis for an implementation nobody
//! has to write would be the same mistake as (3) above, one layer down.
//!
//! ## What that leaves undeclared, stated rather than glossed
//!
//! A [`MemoryWindowMaterialization::HostFormatConversion`] realization does have a host cost, and this
//! decision does **not** surface its size. The original packed Candle realization converted each
//! window from MLX-affine source tensors, allocating anonymous host memory per projection: on the
//! order of the block's unpacked codes for q4, and for q8 a **full dense f32 grid**
//! (`repack::dequant_mlx_q8_gs`), which for a 3840 × 15360 projection is a few hundred MiB. Those
//! transients were not held for the request and SC-15791 did not measure them. SC-16096 replaced that
//! Krea/Lens path with content-addressed device-format sidecars; SC-16510 completed the same transition
//! for streamed Z-Image blocks. No current media provider declares `HostFormatConversion`, but the
//! variant remains so a future converting realization must describe itself honestly rather than claim
//! a device-format transfer.
//!
//! **Revisit trigger — deliberately broad enough to catch that case.** The axis is owed if a
//! realization is measured to need host memory proportional to the model that a fit gate would have to
//! account for: whether it is held for the request or transient, reclaimable or not, and whether or not
//! a device-format alternative exists. A provider that crosses that threshold must raise the contract
//! question rather than absorb it. What is *not* owed is an axis built ahead of that measurement.

use crate::{
    weightsmeta::safetensors_path_bytes, AdapterSpec, Error, GenerationRequest, LoadShape,
    Precision, Quant, Result,
};
use sha2::{Digest, Sha256};

/// Current ABI of the provider/evidence calibration handshake.
///
/// Providers also supply a content fingerprint. The ABI changes when this contract changes how
/// formula inputs or lifecycle semantics are interpreted; the fingerprint changes whenever one
/// provider changes tensor layout, quantization floors, execution structure, or another detail that
/// invalidates its measurements.
///
/// ABI 2 added the typed load-shape axis to calibration identities, run contexts, and evidence
/// keys. ABI 3 adds the typed reference-count geometry axis. Decode-quality admission is versioned
/// independently below: adding a semantic policy must not invalidate or imply a refresh of the
/// model-memory calibration matrix.
pub const MEMORY_CALIBRATION_ABI: u32 = 3;

/// ABI for canonical, load-exact structural evidence used to admit an estimate-backed Resident
/// route without misrepresenting that estimate as measured calibration.
pub const MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI: u32 = 1;

/// Current ABI of production-latent tiled-decode quality records.
///
/// This identity is deliberately independent from [`MEMORY_CALIBRATION_ABI`]: decode quality is a
/// semantic admission input, not a memory or performance measurement. Changing its fixture or
/// verdict shape invalidates quality records without pretending that a model-memory matrix was
/// captured or refreshed.
pub const MEMORY_DECODE_QUALITY_ABI: u32 = 2;

/// Canonical evidence digest of the fixed quality record built by this module's tests.
///
/// This pins the *serialization* contract shared by Rust and the Python collector. It moves when
/// [`MemoryDecodeGeometryPolicy::computed_evidence_sha256`] changes shape, or when the fixture it
/// digests changes value — but, unlike before sc-19728, never in response to unrelated repository
/// bytes, because the fixture's stamp is a literal rather than a constant tracking a source
/// closure. See the note on `implementation_fingerprint` in [`MemoryDecodeGeometryPolicy`].
#[cfg(test)]
const MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256: &str =
    "481c5551c57be43823e74c5c9d85907b85fab2ba57dad9ac215ad005d56a34e8";

/// Prefix for the single-line calibration observation protocol consumed by release tooling.
///
/// The payload after this prefix is compact JSON. Keeping the version outside the JSON makes mixed
/// test output cheap to scan without accepting a legacy `SEQ_AB` line by accident.
pub const MEMORY_EVIDENCE_V1_PREFIX: &str = "MEMORY_EVIDENCE_V1 ";
/// Wire schema for [`MemoryEvidenceLogRecord`] payloads. The V1 line prefix is retained for log
/// discovery; version 2 is required because the evidence cell gained family/reference/FPS axes.
pub const MEMORY_EVIDENCE_SCHEMA_VERSION: u32 = 2;

mod strategy;
pub use strategy::*;

mod formula;
pub use formula::*;

mod components;
pub use components::*;

mod facts;
pub use facts::*;

mod contract;
pub use contract::*;

mod safety;
pub use safety::*;

mod selection;
pub use selection::*;

mod decode_quality;
pub use decode_quality::*;

mod run;
pub use run::*;

mod evidence;
pub use evidence::*;

#[cfg(test)]
mod tests;
