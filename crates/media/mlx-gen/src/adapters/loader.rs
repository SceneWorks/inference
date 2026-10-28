//! Adapter-file loaders — read a trained LoRA/LoKr `.safetensors` and install it onto a
//! model tree via [`AdaptableHost`]. Closes out sc-2343's loader piece.
//!
//! **LoKr** is generic and faithfully ported from the fork's `LoKrLoader.apply`: keys are
//! bare module paths (`‹path›.lokr_w1`/`lokr_w2`, full or low-rank `_a`/`_b`) and the file
//! carries `networkType=lokr` + `alpha`/`rank` in safetensors metadata, so the delta and
//! target path are fully determined by the file — no per-model mapping table.
//!
//! **LoRA** here covers two on-disk conventions, both family-agnostic:
//! - **PEFT/diffusers** (`‹prefix›‹path›.lora_A/B.weight` + optional `‹path›.alpha`): dotted module
//!   paths resolve directly via [`AdaptableHost::adaptable_mut`] ([`apply_lora_peft`]).
//! - **kohya / sd-scripts** (`lora_unet_‹path, .→_›.lora_down/up.weight` + optional `.alpha`,
//!   sc-2618): the flattened module path can't be re-split blindly, so it resolves through a
//!   `flattened → dotted` table built from [`AdaptableHost::adaptable_paths`]
//!   ([`apply_lora_kohya`]). kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`, so both feed the
//!   shared `install_lora_groups` and a kohya file yields the identical adapter to its PEFT twin.
//!
//! - **BFL / ComfyUI** (`lora_unet_double_blocks_*` / `diffusion_model.…` / `base_model.model.…`,
//!   sc-2743): a *fused* source linear (`…img_attn.qkv`, `…linear1`) is row-sliced into the model's
//!   *split* targets (`attn.to_q/to_k/to_v`, …) via per-target [`LoraRowSlice`] transforms, with BFL
//!   module renames (`img_in`→`x_embedder`). This is fused→split weight surgery, orthogonal to the
//!   kohya underscore form; the host supplies its table via [`AdaptableHost::bfl_targets`]
//!   ([`apply_lora_bfl`]). Only FLUX.2/FLUX.1 expose one; for other hosts a BFL file's keys surface as
//!   unmatched (loud), never silently dropped.
//!
//! Layout (sc-27427): this module keeps the format dispatch (`apply_adapter_specs*`,
//! `classify_*`), the alpha-control reads and the strict provider install; each format's
//! production code lives in its own submodule (`lokr`, `loha`, `lycoris`, `peft`, `kohya`, `bfl`,
//! `diff`, plus the header-only `configured` facts) and is re-exported here.

use std::collections::{BTreeMap, BTreeSet};

// Specialized unfused source-prefix facts; existing public facts/enums/JSON remain separate.
pub mod source_facts;

// Per-format production modules (sc-27427), each re-exported so the public paths are unchanged.
mod lokr;
pub use lokr::*;
mod lycoris;
pub use lycoris::*;
mod loha;
pub use loha::*;
mod peft;
pub use peft::*;
mod kohya;
pub use kohya::*;
mod bfl;
pub use bfl::*;
mod configured;
pub use configured::*;
mod diff;
pub use diff::*;

use mlx_rs::memory::{get_active_memory, get_memory_limit};
use mlx_rs::ops::add;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use super::{
    build_lokr_factors, reconstruct_loha_delta, reconstruct_lokr_delta,
    reconstruct_lokr_delta_scaled, AdaptableHost, Adapter, DiffPatchPart, LokrFactors,
};
use crate::runtime::{AdapterKind, AdapterSpec};
use crate::weights::Weights;
use crate::{Error, Result};

// The format predicates, factor-suffix tables, rank/alpha parsing, and key-alias resolution are
// backend-neutral string/metadata logic and live in gen-core (sc-3722); this module supplies the
// `Weights`/`Array` half (factor grouping + delta reconstruction). The historical
// `mlx_gen::adapters::loader::{KOHYA_PREFIX, COMMON_LORA_PREFIXES, resolve_lokr_path}` paths stay
// resolvable via these re-exports.
use gen_core::weightsmeta as wmeta;
pub use gen_core::weightsmeta::{resolve_lokr_path, COMMON_LORA_PREFIXES, KOHYA_PREFIX};

/// Bounded alpha CONTROL read for header-only configured iOS graph normalization. This is the sole
/// payload exception: read the required native scalar (at most eight bytes), never a factor/model
/// tensor, Array or eval. Hash only the normalized zero/active branch, not this value or file path.
/// Matches the served shared scalar_alpha first-element semantics, including multi-element alpha
/// controls. No speculative SDXL reshape mode is exposed: its graph does not need this read.
pub fn read_control_scalar_f32(path: &std::path::Path, key: &str) -> Result<f32> {
    if key != "alpha" && !key.ends_with(".alpha") {
        return Err(Error::Unsupported(
            "bounded adapter control reader only accepts alpha keys".into(),
        ));
    }
    let layout = wmeta::safetensors_file_tensor_locations(path).map_err(Error::from)?;
    let location = layout
        .tensors
        .iter()
        .find(|location| location.header.name == key)
        .ok_or_else(|| Error::Msg(format!("missing adapter alpha control {key}")))?;
    let mut file = std::fs::File::open(path)?;
    read_control_scalar_f32_at(&mut file, location, layout.file_len)
}

fn read_control_scalar_f32_at(
    file: &mut std::fs::File,
    location: &gen_core::weightsmeta::SafetensorsTensorLocation,
    file_len: u64,
) -> Result<f32> {
    use gen_core::weightsmeta::Dtype as S;
    use std::io::{Read, Seek, SeekFrom};
    let key = &location.header.name;
    if key != "alpha" && !key.ends_with(".alpha") {
        return Err(Error::Unsupported(
            "bounded adapter control reader only accepts alpha keys".into(),
        ));
    }
    let count = location.header.element_count().map_err(Error::from)?;
    if count == 0 {
        return Err(Error::Msg(format!(
            "adapter alpha control {key} has {count} elements; scalar shape refused"
        )));
    }
    let width = match location.header.dtype {
        S::BOOL | S::U8 | S::I8 | S::F8_E4M3 | S::F8_E8M0 => 1usize,
        S::U16 | S::I16 | S::F16 | S::BF16 => 2,
        S::U32 | S::I32 | S::F32 => 4,
        S::U64 | S::I64 => 8,
        // Match the actual native safetensors reader, not a new tier conversion.
        S::F64 | S::F8_E5M2 => {
            return Err(Error::Unsupported(
                "native MLX cannot load this alpha control dtype".into(),
            ))
        }
    };
    let expected = count
        .checked_mul(width as u64)
        .ok_or_else(|| Error::Msg("adapter alpha control byte count overflow".into()))?;
    let end = location
        .file_offset
        .checked_add(width as u64)
        .ok_or_else(|| Error::Msg("adapter alpha control file offset overflow".into()))?;
    if expected != location.header.data_bytes
        || width as u64 > location.header.data_bytes
        || end > file_len
    {
        return Err(Error::Msg(
            "adapter alpha control shape/offset/length mismatch".into(),
        ));
    }
    let mut bytes = [0u8; 8];
    file.seek(SeekFrom::Start(location.file_offset))?;
    file.read_exact(&mut bytes[..width])?;
    let u16_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let u32_bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    Ok(match location.header.dtype {
        S::BOOL => u8::from(bytes[0] != 0) as f32,
        S::U8 | S::F8_E4M3 | S::F8_E8M0 => bytes[0] as f32,
        S::I8 => (bytes[0] as i8) as f32,
        S::U16 => u16_bits as f32,
        S::I16 => (u16_bits as i16) as f32,
        S::U32 => u32_bits as f32,
        S::I32 => (u32_bits as i32) as f32,
        S::U64 => u64::from_le_bytes(bytes) as f32,
        S::I64 => i64::from_le_bytes(bytes) as f32,
        S::F32 => f32::from_bits(u32_bits),
        S::BF16 => f32::from_bits(u32::from(u16_bits) << 16),
        S::F16 => {
            let sign = u32::from(u16_bits & 0x8000) << 16;
            let exponent = (u16_bits >> 10) & 31;
            let fraction = u32::from(u16_bits & 1023);
            if exponent == 0 {
                let magnitude = fraction as f32 * 2.0_f32.powi(-24);
                if sign != 0 {
                    -magnitude
                } else {
                    magnitude
                }
            } else if exponent == 31 {
                f32::from_bits(sign | 0x7f80_0000 | (fraction << 13))
            } else {
                f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13))
            }
        }
        S::F64 | S::F8_E5M2 => unreachable!("unsupported native dtype rejected before read"),
    })
}

/// Load and install every adapter in `specs` onto `host`, stacking in order. Each spec's file is
/// read, dispatched to the LoKr or PEFT-LoRA loader by its [`AdapterKind`], applied at `spec.scale`,
/// and its [`ApplyReport`] merged into the combined result — unmatched target paths are surfaced,
/// never silently dropped. `lora_strip_prefix` is the per-family namespace stripped from PEFT-LoRA
/// keys (e.g. `"transformer."`); it does not apply to LoKr (whose keys are bare module paths).
///
/// This is the load-time seam (sc-2534): a provider calls it inside `load()` with its model's
/// [`AdaptableHost`] while the model is still mutable. Empty `specs` is a no-op (empty report).
///
/// **Scope (F-035):** this fixed-prefix variant routes PEFT/diffusers LoRA, metadata-stamped LoKr,
/// and keyless third-party LyCORIS LoKr/LoHa. Unlike [`apply_adapter_specs_autoprefix`] it does NOT
/// detect BFL/ComfyUI fused→split or kohya-flattened files — both require walking the host module
/// tree (`bfl_targets()` / `kohya_table()`), which only the autoprefix path does. Callers that may
/// receive BFL or kohya files must use the autoprefix variant; here such a file would resolve no
/// targets and be reported as fully unmatched rather than applied.
/// iOS pre-load quantum label of an adapter (LoRA/LoKr/LoHa/diff-patch) file read.
pub const ADAPTER_LOAD_QUANTUM: &str = "adapter load";

pub fn apply_adapter_specs(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    lora_strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for spec in specs {
        // iOS: header-only floor of the adapter file before it is eagerly read (no-op elsewhere).
        crate::ios_quantum::before_load(ADAPTER_LOAD_QUANTUM, std::slice::from_ref(&spec.path))?;
        let w = Weights::from_file(&spec.path)?;
        let report = match spec.kind {
            AdapterKind::Lokr => {
                // A keys-only third-party LyCORIS LoKr (kohya / ai-toolkit / lycoris-lib) carries
                // `lokr_*` keys but no `networkType` stamp, and its per-module `.alpha` scalars +
                // optional tucker `lokr_t2` are invisible to the peft parser — routing it to
                // `apply_lokr` would silently drop both and install a mis-scaled delta. Detect by
                // keys and route to the third-party applier, mirroring the `Lora` arm below and
                // the autoprefix dispatch (F-012).
                if !is_lokr(&w) && is_lokr_keys(&w) {
                    apply_lokr_thirdparty(host, &w, spec.scale)?
                } else {
                    apply_lokr(host, &w, spec.scale)?
                }
            }
            AdapterKind::Lora => {
                // The file's metadata is authoritative; a kind/metadata mismatch is a caller error
                // (the PEFT-LoRA loader would find no `lora_A/B` keys and apply nothing) — surface it.
                if is_lokr(&w) {
                    return Err(format!(
                        "adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )
                    .into());
                }
                // A third-party LyCORIS LoKr (sc-3642) / LoHa (sc-3643) carries `lokr_*` / `hada_*`
                // keys but no `networkType` stamp, so a caller can't know to set a non-Lora kind —
                // detect + route by keys.
                if is_lokr_keys(&w) {
                    apply_lokr_thirdparty(host, &w, spec.scale)?
                } else if is_loha_keys(&w) {
                    apply_loha_thirdparty(host, &w, spec.scale)?
                } else {
                    apply_lora_peft(host, &w, spec.scale, lora_strip_prefix)?
                }
            }
        };
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

// `COMMON_LORA_PREFIXES` — the LoRA key namespace prefixes diffusers/peft adapter files use, tried
// in order; the first a key begins with is stripped. LoKr files are bare (no prefix); kohya
// `lora_unet_…` files flatten the module dots to underscores and resolve through a separate
// flattened→dotted table ([`apply_lora_kohya`], sc-2618), not this prefix strip. SceneWorks' trained
// LoRAs use `transformer.` (peft `save_lora_weights`) or `diffusion_model.` (sd-scripts export) —
// both observed on real files. Defined in gen-core (`weightsmeta`) and re-exported above.

/// The LoRA namespace prefix present in `w`'s keys, if any (see [`COMMON_LORA_PREFIXES`]).
pub fn detect_lora_prefix(w: &Weights) -> Option<&'static str> {
    wmeta::detect_lora_prefix(w.keys())
}

/// Host-derived lookup tables shared across one [`apply_adapter_specs_autoprefix`] call's specs.
/// Each is a model-tree walk, so it is built lazily and at most once — the first time a spec's
/// classification (or its apply arm) needs it.
#[derive(Default)]
struct HostTables {
    kohya: Option<BTreeMap<String, String>>,
    bfl: Option<Vec<BflTarget>>,
    bfl_lyc: Option<BTreeMap<String, Vec<BflLycorisTarget>>>,
}

impl HostTables {
    fn kohya(&mut self, host: &impl AdaptableHost) -> &BTreeMap<String, String> {
        self.kohya
            .get_or_insert_with(|| kohya_table(&host.adaptable_paths()))
    }

    fn bfl(&mut self, host: &impl AdaptableHost) -> &[BflTarget] {
        self.bfl.get_or_insert_with(|| host.bfl_targets())
    }

    fn bfl_lyc(&mut self, host: &impl AdaptableHost) -> &BTreeMap<String, Vec<BflLycorisTarget>> {
        if self.bfl_lyc.is_none() {
            let map = bfl_lycoris_module_map(self.bfl(host));
            self.bfl_lyc = Some(map);
        }
        self.bfl_lyc.as_ref().unwrap()
    }
}

/// The on-disk format of one adapter file, as resolved by [`classify_adapter_format`] — the single
/// routing truth for [`apply_adapter_specs_autoprefix`]'s dispatch (F-069). Variant order mirrors
/// the detection precedence:
///
/// 1. **Third-party LyCORIS keys** (`lokr_*` without a `networkType` stamp, or `hada_*`) — before
///    BFL-LoRA/kohya, because a kohya-flattened LyCORIS file also carries the `lora_unet_` prefix,
///    which `is_kohya` would otherwise claim and then apply nothing (sc-3642/sc-3643).
/// 2. **BFL/ComfyUI fused→split LoRA naming** — before kohya because it shares the `lora_unet_`
///    prefix (sc-2743). Skipped (with kohya) for a metadata-stamped LoKr, whose keys stay dotted.
/// 3. **kohya-flattened LoRA** (`lora_unet_…`).
/// 4. The spec's declared [`AdapterKind`]: a peft/metadata LoKr (BFL-named or bare) or a
///    PEFT/diffusers LoRA — where a `Lora` declaration against `networkType=lokr` metadata is a
///    caller error, surfaced as [`Self::LoraKindMismatch`].
///
/// Each LyCORIS format carries a BFL twin because a LyCORIS file can ship in BFL/ComfyUI fused
/// naming on a FLUX host (sc-8345); every other host has an empty BFL surface, so the `*Bfl`
/// variants never classify there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterFormat {
    /// Third-party LyCORIS LoKr (`lokr_*` keys, no `networkType` stamp) in BFL/ComfyUI fused naming.
    ThirdpartyLokrBfl,
    /// Third-party LyCORIS LoKr in bare / diffusers / kohya-flattened naming.
    ThirdpartyLokr,
    /// Third-party LyCORIS LoHa (`hada_*` keys) in BFL/ComfyUI fused naming.
    ThirdpartyLohaBfl,
    /// Third-party LyCORIS LoHa in bare / diffusers / kohya-flattened naming.
    ThirdpartyLoha,
    /// BFL/ComfyUI fused→split LoRA (sc-2743).
    BflLora,
    /// kohya / sd-scripts flattened LoRA (`lora_unet_…`).
    KohyaLora,
    /// Metadata-stamped (peft) LoKr in BFL/ComfyUI fused naming (sc-8345).
    PeftLokrBfl,
    /// Declared-`Lokr` fallback: a metadata-stamped or bare dotted-path LoKr.
    PeftLokr,
    /// PEFT/diffusers LoRA — the common fallback.
    PeftLora,
    /// Declared `Lora` but the file's metadata says `networkType=lokr` — a caller error (the
    /// metadata is authoritative), surfaced as a hard error by the dispatch.
    LoraKindMismatch,
}

/// The file routes available to an unfused projection host (no BFL fused-to-split mappings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnfusedAdapterFormat {
    ThirdpartyLokr,
    ThirdpartyLoha,
    KohyaLora,
    PeftLokr,
    PeftLora,
    LoraKindMismatch,
}

/// Header-only counterpart of the actual loader dispatch for an unfused host. No tensor read or
/// model construction is needed; callers must still validate their provider's identity stamps.
pub fn classify_unfused_adapter_format<'a>(
    keys: impl IntoIterator<Item = &'a str>,
    network_type: Option<&str>,
    kind: AdapterKind,
) -> UnfusedAdapterFormat {
    let keys: Vec<&str> = keys.into_iter().collect();
    let stamped = wmeta::is_lokr_network_type(network_type);
    if !stamped && wmeta::keys_contain_lokr(keys.iter().copied()) {
        UnfusedAdapterFormat::ThirdpartyLokr
    } else if wmeta::keys_contain_loha(keys.iter().copied()) {
        UnfusedAdapterFormat::ThirdpartyLoha
    } else if !stamped && wmeta::keys_are_kohya(keys.iter().copied()) {
        UnfusedAdapterFormat::KohyaLora
    } else {
        match kind {
            AdapterKind::Lokr => UnfusedAdapterFormat::PeftLokr,
            AdapterKind::Lora if stamped => UnfusedAdapterFormat::LoraKindMismatch,
            AdapterKind::Lora => UnfusedAdapterFormat::PeftLora,
        }
    }
}

/// A dotted LoRA factor role, shared by the real PEFT loader and header inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoraFactorRole {
    Down,
    Up,
}

/// The actual unfused residual LoRA install's matrix gate (not the separate convolution merge
/// format). Shared with header normalization so rejected factors cannot become admitted facts.
pub fn validate_lora_shapes(path: &str, down: &[i32], up: &[i32]) -> Result<()> {
    if down.len() != 2 || up.len() != 2 {
        return Err(Error::Msg(format!(
            "lora adapter at '{path}' has non-2-D factors (down {:?}, up {:?})",
            down, up
        )));
    }
    Ok(())
}

/// Called ONLY when an alpha fold occurs. A zero stored rank without a fold is not newly refused.
pub fn validate_lora_alpha_rank(path: &str, rank: i32) -> Result<()> {
    if rank == 0 {
        return Err(Error::Msg(format!(
            "lora adapter at '{path}' has zero rank (empty down/up factor)"
        )));
    }
    Ok(())
}

/// Reject ambiguous same-role aliases identically before runtime tensors or header facts are
/// installed. Report keys in lexical order, independent of the source HashMap's iteration order.
/// Separate files are separate stacks and intentionally do not share this claim map.
pub fn claim_lora_factor(
    claimed: &mut BTreeMap<(String, LoraFactorRole), String>,
    path: &str,
    role: LoraFactorRole,
    key: &str,
) -> Result<()> {
    if let Some(previous) = claimed.insert((path.to_owned(), role), key.to_owned()) {
        let (first, second) = if previous.as_str() < key {
            (previous.as_str(), key)
        } else {
            (key, previous.as_str())
        };
        let label = if role == LoraFactorRole::Down {
            "down/A"
        } else {
            "up/B"
        };
        return Err(Error::Msg(format!(
            "LoRA {label} factor for `{path}` is spelled twice (`{first}` and `{second}`); refusing an ambiguous apply"
        )));
    }
    Ok(())
}

pub fn peft_lora_factor<'a>(key: &'a str, prefix: &str) -> Option<(&'a str, LoraFactorRole)> {
    let rest = key.strip_prefix(prefix)?;
    for (suffix, role) in [
        (".lora_A.weight", LoraFactorRole::Down),
        (".lora_A.default.weight", LoraFactorRole::Down),
        (".lora_down.weight", LoraFactorRole::Down),
        (".lora_B.weight", LoraFactorRole::Up),
        (".lora_B.default.weight", LoraFactorRole::Up),
        (".lora_up.weight", LoraFactorRole::Up),
    ] {
        if let Some(path) = rest.strip_suffix(suffix) {
            return Some((path, role));
        }
    }
    None
}

/// Classify one adapter file for the autoprefix dispatch (F-069). A pure function of the file's
/// keys/metadata, the declared `kind`, and the host's kohya/BFL surface; `tables` caches the
/// host-tree walks across specs. See [`AdapterFormat`] for the precedence rationale.
fn classify_adapter_format(
    w: &Weights,
    kind: AdapterKind,
    host: &impl AdaptableHost,
    tables: &mut HostTables,
) -> AdapterFormat {
    // Unfused hosts (including Iris) share this exact header-only classifier with their pre-load
    // inventory. Fused-to-split BFL hosts retain their existing host-specific classification below.
    if tables.bfl(host).is_empty() {
        return match classify_unfused_adapter_format(w.keys(), w.metadata("networkType"), kind) {
            UnfusedAdapterFormat::ThirdpartyLokr => AdapterFormat::ThirdpartyLokr,
            UnfusedAdapterFormat::ThirdpartyLoha => AdapterFormat::ThirdpartyLoha,
            UnfusedAdapterFormat::KohyaLora => AdapterFormat::KohyaLora,
            UnfusedAdapterFormat::PeftLokr => AdapterFormat::PeftLokr,
            UnfusedAdapterFormat::PeftLora => AdapterFormat::PeftLora,
            UnfusedAdapterFormat::LoraKindMismatch => AdapterFormat::LoraKindMismatch,
        };
    }
    // The `networkType=lokr` stamp gates `lokr_*` keys: a stamped (peft) file routes by its stamp
    // (global metadata alpha/rank), a keys-only file by the third-party per-module derivation.
    let is_lokr_meta = is_lokr(w);
    if !is_lokr_meta && is_lokr_keys(w) {
        return if is_bfl_lycoris(w, tables.bfl_lyc(host)) {
            AdapterFormat::ThirdpartyLokrBfl
        } else {
            AdapterFormat::ThirdpartyLokr
        };
    }
    if is_loha_keys(w) {
        return if is_bfl_lycoris(w, tables.bfl_lyc(host)) {
            AdapterFormat::ThirdpartyLohaBfl
        } else {
            AdapterFormat::ThirdpartyLoha
        };
    }
    if !is_lokr_meta {
        if is_bfl(w, tables.bfl(host)) {
            return AdapterFormat::BflLora;
        }
        if is_kohya(w) {
            return AdapterFormat::KohyaLora;
        }
    }
    // No format-discriminating keys left — fall back to the spec's declared kind.
    match kind {
        AdapterKind::Lokr => {
            if is_lokr_meta && is_bfl_lycoris(w, tables.bfl_lyc(host)) {
                AdapterFormat::PeftLokrBfl
            } else {
                AdapterFormat::PeftLokr
            }
        }
        AdapterKind::Lora if is_lokr_meta => AdapterFormat::LoraKindMismatch,
        AdapterKind::Lora => AdapterFormat::PeftLora,
    }
}

/// [`apply_adapter_specs`] with per-file LoRA-prefix **auto-detection** ([`detect_lora_prefix`])
/// instead of a fixed prefix — the common provider path, since LoRA files vary
/// (`transformer.` / `diffusion_model.` / bare) while LoKr keys are bare. The host's key→module map
/// must match the (prefix-stripped) diffusers module paths.
pub fn apply_adapter_specs_autoprefix(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for report in apply_adapter_specs_autoprefix_reported(host, specs)? {
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
    }
    Ok(combined)
}

/// Per-spec form of [`apply_adapter_specs_autoprefix`]. The host is mutated in the same input order
/// and the expensive host lookup tables are still shared across the batch; only the report reduction
/// is deferred so a strict batch caller can retain file attribution.
fn apply_adapter_specs_autoprefix_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<Vec<ApplyReport>> {
    let mut tables = HostTables::default();
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        // Load + classify the file once per spec (F-004); `classify_adapter_format` is the single
        // source of routing truth (F-069) — the arms below only fetch the table the applier needs
        // and apply.
        // iOS: header-only floor of the adapter file before it is eagerly read (no-op elsewhere).
        crate::ios_quantum::before_load(ADAPTER_LOAD_QUANTUM, std::slice::from_ref(&spec.path))?;
        let w = Weights::from_file(&spec.path)?;
        let report = match classify_adapter_format(&w, spec.kind, host, &mut tables) {
            AdapterFormat::ThirdpartyLokrBfl => {
                let map = tables.bfl_lyc(host);
                apply_lokr_thirdparty_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::ThirdpartyLokr => apply_lokr_thirdparty(host, &w, spec.scale)?,
            AdapterFormat::ThirdpartyLohaBfl => {
                let map = tables.bfl_lyc(host);
                apply_loha_thirdparty_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::ThirdpartyLoha => apply_loha_thirdparty(host, &w, spec.scale)?,
            AdapterFormat::BflLora => {
                let targets = tables.bfl(host);
                apply_lora_bfl(host, &w, spec.scale, targets)?
            }
            AdapterFormat::KohyaLora => {
                let table = tables.kohya(host);
                apply_lora_kohya(host, &w, spec.scale, table)?
            }
            AdapterFormat::PeftLokrBfl => {
                let map = tables.bfl_lyc(host);
                apply_lokr_bfl(host, &w, spec.scale, map)?
            }
            AdapterFormat::PeftLokr => apply_lokr(host, &w, spec.scale)?,
            AdapterFormat::PeftLora => {
                apply_lora_peft(host, &w, spec.scale, detect_lora_prefix(&w))?
            }
            AdapterFormat::LoraKindMismatch => {
                return Err(format!(
                    "adapter {} declared Lora but its metadata says networkType=lokr",
                    spec.path.display()
                )
                .into());
            }
        };
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        reports.push(report);
    }
    Ok(reports)
}

/// Provider-facing load-time adapter install: [`apply_adapter_specs_autoprefix`] plus a strict
/// no-silent-drop policy — errors if a non-empty spec list matched nothing, or if any adapter
/// target resolved to no module. `model` names the model in the error (e.g. `"z_image_turbo"`).
/// Both Z-Image and Qwen providers call this; the only per-family piece is the model's
/// `AdaptableHost` key→module map.
pub fn apply_adapters_strict(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    apply_adapters_strict_inner(host, specs, model, 0)
}

/// Diff-patch-aware strict install (sc-13825, MLX parity for candle sc-13726): fold any ComfyUI
/// `.diff`/`.diff_b` **diff-patch** deltas into the host's dense bases first ([`fold_diff_patch_adapters`]),
/// then run the ordinary low-rank residual pass, relaxing the zero-match guard by the count folded. The
/// two passes are disjoint by key suffix, so a mixed lightning file (both `.diff` and `lora_*`) has its
/// weight/bias deltas folded AND its low-rank factors stacked as residuals. Skipped / unmatched
/// diff-patch targets are surfaced on `stderr` **and** returned on
/// [`ApplyReport::diff_patch_unapplied`], so a provider can put a partial install in front of the user
/// instead of leaving it in an engine log (sc-15326); a file that folds nothing **and** matches no
/// low-rank target still errors via the combined guard. Krea's filter-bypass and Krea Realtime's
/// step-distill entry.
///
/// A spec at **scale 0** folds nothing (sc-15265 — a disabled adapter is byte-identical to an
/// uninstalled one), but its resolved targets still relax the guard, so disabling a *pure* diff-patch
/// file does not turn into a "matched nothing" error.
pub fn apply_adapters_strict_with_diff_patch(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<ApplyReport> {
    let mut combined = ApplyReport::default();
    for report in apply_adapters_strict_with_diff_patch_reported(host, specs, model)? {
        combined.applied += report.applied;
        combined.unmatched_paths.extend(report.unmatched_paths);
        combined
            .diff_patch_unapplied
            .extend(report.diff_patch_unapplied);
    }
    Ok(combined)
}

/// Per-spec report form of [`apply_adapters_strict_with_diff_patch`].
///
/// This preserves the batch install contract exactly: every spec's diff-patch deltas fold in input
/// order first, then every spec's low-rank residuals install in input order, and the strict guards
/// evaluate the batch totals. In particular, a wholly unsupported diff-patch file is reported as
/// unapplied without failing the load when another file in the same batch applies; the batch only
/// fails the zero-match guard when nothing across the entire batch lands.
pub fn apply_adapters_strict_with_diff_patch_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
) -> Result<Vec<ApplyReport>> {
    let diff_reports = fold_diff_patch_adapters_reported(host, specs)?;
    let skipped: Vec<&str> = diff_reports
        .iter()
        .flat_map(|report| report.skipped.iter().map(String::as_str))
        .collect();
    if !skipped.is_empty() {
        eprintln!(
            "{model} adapters: {} diff-patch target(s) skipped (shape-incompatible / a weight `.diff` \
             on a packed Linear / no base bias — coupled parts dropped, never half-applied): {:?}",
            skipped.len(),
            skipped
        );
    }
    let unmatched: Vec<&str> = diff_reports
        .iter()
        .flat_map(|report| report.unmatched.iter().map(String::as_str))
        .collect();
    if !unmatched.is_empty() {
        eprintln!(
            "{model} adapters: {} diff-patch target(s) matched no module, skipped: {:?}",
            unmatched.len(),
            unmatched
        );
    }

    let mut reports = apply_adapter_specs_autoprefix_reported(host, specs)?;
    debug_assert_eq!(reports.len(), diff_reports.len());
    for (report, diff) in reports.iter_mut().zip(diff_reports) {
        // Diff-patch deltas that resolved count toward the total install, so the returned report is
        // truthful. A scale-0 delta is included for the same reason the low-rank pass counts a scale-0
        // adapter as applied: the target matched and was accepted, it is simply inert.
        report.applied += diff.folded + diff.disabled;
        // Everything that did NOT land, carried to the provider so a partial install can reach the
        // user with its originating file intact.
        report.diff_patch_unapplied = diff.skipped;
        report.diff_patch_unapplied.extend(diff.unmatched);
    }

    let applied: usize = reports.iter().map(|report| report.applied).sum();
    if !specs.is_empty() && applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — LoKr keys, or a ComfyUI/lightx2v \
             `<module>.diff`/`.diff_b` diff-patch)",
            specs.len()
        )
        .into());
    }
    let low_rank_unmatched: Vec<&str> = reports
        .iter()
        .flat_map(|report| report.unmatched_paths.iter().map(String::as_str))
        .collect();
    if !low_rank_unmatched.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            low_rank_unmatched.len(),
            low_rank_unmatched
        )
        .into());
    }
    Ok(reports)
}

/// Core of [`apply_adapters_strict`]: `pre_applied` is the count of diff-patch targets a prior pass
/// already resolved — folded, or deliberately left unwritten at scale 0 (0 for the plain path). It only
/// relaxes the zero-match guard — a diff-patch-only file whose delta already folded resolves zero
/// low-rank residuals here, and must not read as "matched nothing".
fn apply_adapters_strict_inner(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    model: &str,
    pre_applied: usize,
) -> Result<ApplyReport> {
    let report = apply_adapter_specs_autoprefix(host, specs)?;
    if !specs.is_empty() && report.applied == 0 && pre_applied == 0 {
        return Err(format!(
            "{model} adapters: no target modules matched across {} adapter file(s) — check the \
             format/prefix (expected diffusers/peft LoRA, kohya `lora_unet_` LoRA, BFL/ComfyUI \
             fused→split LoRA — for a host with a BFL surface — LoKr keys, or a ComfyUI/lightx2v \
             `<module>.diff`/`.diff_b` diff-patch)",
            specs.len()
        )
        .into());
    }
    if !report.unmatched_paths.is_empty() {
        return Err(format!(
            "{model} adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )
        .into());
    }
    Ok(report)
}

#[cfg(test)]
mod configured_header_tests;

#[cfg(test)]
mod alpha_control_tests;

#[cfg(test)]
mod tests;
