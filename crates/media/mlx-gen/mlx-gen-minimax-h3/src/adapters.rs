//! MiniMax-H3 LoRA consumption (sc-18724) — the key→module map and, more importantly, the **alpha
//! resolution rule** the lightx2v turbo checkpoints need.
//!
//! **Strategy: forward-time residual** over [`mlx_gen::adapters::AdaptableLinear`], never a merged
//! weight. Every adaptable leaf in this DiT is one, dense or packed alike, so `base(x) + Σ
//! adapter.residual(x)` holds on **every** quant tier with no dequantization: the `q4` DiT folds the
//! turbo LoRA at the same strength the `bf16` one does. Quant tier is a creative choice in this
//! product, so an adapter whose result varied with it would be a real defect.
//!
//! # Key space: **diffusers**, and only diffusers
//!
//! `lightx2v/Minimax-h3-Turbo` publishes each adapter twice. The **diffusers** export is the one this
//! crate consumes:
//!
//! ```text
//! transformer_blocks.{0..49}.attn.{to_q,to_k,to_v,to_out.0}.lora_{A,B}.default.weight
//! transformer_blocks.{0..49}.ff.{net.0.proj,net.2}.lora_{A,B}.default.weight
//! token_refiner.refiner_blocks.{0,1}.…  (the same six leaves)
//! ```
//!
//! 624 tensors → **312 modules**, keyed *exactly* as [`crate::dit`] names its own weights, so the map
//! is a suffix strip and nothing else. Two properties of it are load-bearing:
//!
//! * **The `.default` infix is mandatory.** These are PEFT-with-adapter-name exports (upstream's own
//!   constants), and the shared `mlx_gen::adapters::loader` PEFT path strips only `.lora_A.weight` /
//!   `.lora_B.weight`. The infix is declared per crate — `mlx-gen-ltx`, `mlx-gen-sdxl` and
//!   `candle-gen-anima` all declare it, `mlx-gen-wan` does not — so this crate must.
//! * **The token refiner cannot be stubbed.** 24 of the 624 tensors target
//!   `token_refiner.refiner_blocks.{0,1}`; a stub puts them in `unmatched_paths` and fails the strict
//!   install. sc-17144 already required a real refiner for parity — this is a second, independent
//!   reason.
//!
//! The `_comfyui_` twin of each file is a **different model shape**, not a different spelling, and is
//! [rejected by name][`is_comfyui_key_space`] rather than half-applied: it fuses `to_q/to_k/to_v`
//! into one `attn.qkv_proj` (block-diagonal `B`, concatenated `A`, alpha ×3) and swaps the SwiGLU
//! halves of `mlp.fc1` to `[gate | value]` — the published DiT is `[value | gate]`
//! ([`crate::layout`]). Folding it un-swapped is shape-identical and computes
//! `w2(silu(value)·gate)`: precisely the sc-18740 defect a fully green parity suite did not see. Every
//! comfyui file has a byte-equivalent diffusers twin in the same repo, so rejecting it costs no
//! capability at all.
//!
//! # The alpha trap — why neither existing path can load these files
//!
//! The effective fold is `scale · alpha / rank`, and lightx2v's diffusers exports carry the alpha as a
//! **bare top-level `__metadata__` string** (`{"alpha": "8"}`) with **no `rank` key**, no per-target
//! `.alpha` tensor and no `lora_adapter_metadata` blob. Both in-tree paths mishandle that silently:
//!
//! | path | reads | on `alpha=8, rank=128` | error? |
//! | --- | --- | --- | --- |
//! | shared PEFT ([`mlx_gen::gen_core::weightsmeta`], sc-5513) | `lora_adapter_metadata` | absent ⇒ `alpha = rank` ⇒ scale **1.0** — **16× too strong** | none |
//! | `wmeta::parse_rank_alpha(meta("rank"), meta("alpha"))` (the LoKr path) | `rank`/`alpha` strings | no `rank` ⇒ rank `1.0`, alpha `8` ⇒ scale **8.0** — **128× too strong** | none |
//!
//! [`resolve_alpha`] is the fix, and its fallback is the whole point: **[`DEFAULT_LORA_ALPHA`] = 8,
//! never rank.** Alpha differs *per file inside one repo*, so it is read from the file every time and
//! never hardcoded per family — measured across the published set:
//!
//! | file | `__metadata__` | rank (from shapes) | effective scale at strength 1.0 |
//! | --- | --- | --- | --- |
//! | `…_4step_v1.0_768p_bf16` | `alpha: "128"` | 128 | **1.0** |
//! | `…_8step_v1.0_bf16` | `alpha: "8"` | 128 | **0.0625** |
//! | `…_ref2v_…_4step_v0.1_bf16` | `alpha: "8"` | 128 | **0.0625** |
//! | `…_4step_v0.1` | *(none)* | 128 | **0.0625** (the default alpha) |
//!
//! Rank comes from the **factor shapes** (`lora_A` is `[r, in]`), as upstream's
//! `_validate_lora_state_dict` derives it — never from a metadata string. **Both** declared spellings
//! are cross-checks only, never sources: a `__metadata__["rank"]` string *and* a PEFT
//! `lora_adapter_metadata` `r` / `rank_pattern` that disagrees with the shapes is a hard error, not
//! a silent pick. That is a deliberate divergence from the shared loader, which takes
//! `cfg_rank.unwrap_or(factor_rank)` and so lets a blob `{"r": 8}` over rank-128 factors fold at
//! `8/8 = 1.0` instead of `8/128 = 0.0625` — the same 16× class as the alpha trap below, and just as
//! silent. PEFT writes `r` equal to the factor rank, so rejecting a disagreement rejects nothing
//! legitimate; no published turbo file carries a blob at all.

use std::collections::BTreeMap;

use mlx_rs::Array;

use mlx_gen::adapters::loader::{apply_lokr, is_lokr, is_lokr_keys, ApplyReport};
use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::array::scalar;
use mlx_gen::gen_core::weightsmeta::{LoraAdapterMeta, LORA_ADAPTER_METADATA_KEY};
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::MiniMaxH3DitConfig;

/// The alpha upstream's `inference_minimax_h3.py` assumes when a LoRA declares none —
/// `DEFAULT_LORA_ALPHA = 8`.
///
/// **This is the fallback, and it is deliberately not the rank.** Falling back to the rank (what both
/// in-tree paths do) makes `alpha/rank == 1.0`, which is the *correct* answer for exactly one of the
/// published turbo files and 16× too strong for the rest. `…_4step_v0.1` ships no alpha at all and
/// must fold at `8/128 = 0.0625`.
pub const DEFAULT_LORA_ALPHA: f32 = 8.0;

/// The top-level `__metadata__` key lightx2v stamps the alpha into, as a **string**.
pub const ALPHA_METADATA_KEY: &str = "alpha";

/// The top-level `__metadata__` key a rank would be stamped into. Read only to **cross-check** the
/// factor shapes — never as the source of the rank. See [`resolve_rank`].
pub const RANK_METADATA_KEY: &str = "rank";

/// LoRA key namespace prefixes stripped, longest-first. The published turbo files carry **none** of
/// these (their keys are bare `transformer_blocks.…` / `token_refiner.…`, identical to the
/// checkpoint's own naming); the list exists so a re-export that adds the usual diffusers namespace
/// still resolves. `diffusion_model.` is the ComfyUI namespace and is only ever reached by a file
/// that already passed [`is_comfyui_key_space`].
pub const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
];

/// Which half of a LoRA a key names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// `lora_A` / `lora_down` — the `[rank, in]` down factor.
    Down,
    /// `lora_B` / `lora_up` — the `[out, rank]` up factor.
    Up,
    /// A per-target `.alpha` scalar (kohya / SceneWorks-trainer convention).
    Alpha,
}

/// Factor suffix → role, exact-matched. The **`.default` adapter-name infix** is what the published
/// turbo files use and what the shared PEFT loader does not strip; the bare forms are accepted too so
/// a re-export that drops the adapter name still loads.
const SUFFIXES: [(&str, Role); 9] = [
    (".lora_A.default.weight", Role::Down),
    (".lora_B.default.weight", Role::Up),
    (".lora_down.default.weight", Role::Down),
    (".lora_up.default.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".alpha", Role::Alpha),
];

/// The six adaptable leaves every DiT block and every token-refiner block exposes, in the published
/// diffusers spelling. `to_out.0` and `net.{0.proj,2}` keep their `nn.Sequential` indices.
pub const BLOCK_TARGETS: [&str; 6] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];

/// Outcome of applying the MiniMax-H3 adapter specs.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MiniMaxH3LoraReport {
    /// Target modules that received a residual.
    pub applied: usize,
    /// Adapter module paths that resolved to no module — surfaced, never silently dropped.
    /// [`apply_minimax_h3_adapters`] refuses to return with this non-empty.
    pub unmatched_paths: Vec<String>,
}

/// Every module path a MiniMax-H3 adapter can address, at `cfg`'s geometry: `num_layers` transformer
/// blocks and `num_refiner_layers` refiner blocks × the six [`BLOCK_TARGETS`]. 312 at the shipped
/// geometry (50·6 + 2·6), which is exactly the module count of each published turbo file.
///
/// Derived from the config rather than hardcoded, and asserted to resolve through
/// [`AdaptableHost::adaptable_mut`], so this list and the model tree cannot drift apart.
pub fn adapter_target_paths(cfg: &MiniMaxH3DitConfig) -> Vec<String> {
    let mut v = Vec::new();
    for i in 0..cfg.num_layers {
        v.extend(
            BLOCK_TARGETS
                .iter()
                .map(|t| format!("transformer_blocks.{i}.{t}")),
        );
    }
    for i in 0..cfg.num_refiner_layers {
        v.extend(
            BLOCK_TARGETS
                .iter()
                .map(|t| format!("token_refiner.refiner_blocks.{i}.{t}")),
        );
    }
    v
}

/// Strip a known namespace prefix. The turbo files need no renaming beyond this: their module paths
/// are already the checkpoint's own.
pub fn normalize_minimax_h3_key(key: &str) -> String {
    PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key)
        .to_string()
}

/// `true` if `w` is a lightx2v **`_comfyui_`** export rather than the diffusers one.
///
/// Detected on the **keys** (`attn.qkv_proj` / `attn.out_proj` / `mlp.fc{1,2}` — module names that do
/// not exist in the diffusers layout at all) rather than only on the `target_format` metadata stamp, so
/// a re-export that drops the stamp is still caught. See the module docs for why this is refused and
/// not converted.
pub fn is_comfyui_key_space(w: &Weights) -> bool {
    w.keys().any(|k| {
        k.contains(".attn.qkv_proj.") || k.contains(".attn.out_proj.") || k.contains(".mlp.fc")
    })
}

/// The LoRA rank for one module, from the **factor shapes**: `down` is stored `[rank, in]`, so the
/// rank is its leading dim. Mirrors upstream's `_validate_lora_state_dict`.
///
/// `meta_rank` is a top-level `__metadata__["rank"]`, read only to cross-check. A value that
/// disagrees with the shapes is an error rather than a silent pick — the failure the shared LoKr path
/// makes invisible is exactly a rank taken from (missing) metadata instead of from the tensors.
pub fn resolve_rank(path: &str, down: &Array, meta_rank: Option<&str>) -> Result<f32> {
    if down.shape().len() != 2 {
        return Err(Error::Msg(format!(
            "minimax_h3 adapter '{path}': lora_A must be 2-D [rank, in], got {:?}",
            down.shape()
        )));
    }
    let rank = down.shape()[0];
    if rank <= 0 {
        return Err(Error::Msg(format!(
            "minimax_h3 adapter '{path}': zero rank (empty lora_A factor) — `alpha/rank` would be \
             non-finite and NaN-poison the residual"
        )));
    }
    if let Some(raw) = meta_rank {
        let declared = raw.trim().parse::<f32>().map_err(|_| {
            Error::Msg(format!(
                "minimax_h3 adapter '{path}': __metadata__[\"{RANK_METADATA_KEY}\"] = {raw:?} is \
                 not a number"
            ))
        })?;
        if declared != rank as f32 {
            return Err(Error::Msg(format!(
                "minimax_h3 adapter '{path}': __metadata__[\"{RANK_METADATA_KEY}\"] = {declared} \
                 disagrees with the lora_A factor rank {rank}; the shapes are authoritative, so \
                 this file is inconsistent rather than merely under-specified"
            )));
        }
    }
    Ok(rank as f32)
}

/// The file-level LoRA alpha: top-level `__metadata__["alpha"]`, falling back to
/// [`DEFAULT_LORA_ALPHA`] — **not** to the rank.
///
/// An `alpha` that is present but unparseable or non-finite is an **error**, not a fall-through to the
/// default: a typo'd stamp silently folding at 8 is the same class of bug this function exists to
/// close.
pub fn resolve_alpha(w: &Weights) -> Result<f32> {
    let Some(raw) = w.metadata(ALPHA_METADATA_KEY) else {
        return Ok(DEFAULT_LORA_ALPHA);
    };
    let alpha = raw.trim().parse::<f32>().map_err(|_| {
        Error::Msg(format!(
            "minimax_h3 adapter: __metadata__[\"{ALPHA_METADATA_KEY}\"] = {raw:?} is not a number"
        ))
    })?;
    if !alpha.is_finite() {
        return Err(Error::Msg(format!(
            "minimax_h3 adapter: __metadata__[\"{ALPHA_METADATA_KEY}\"] = {alpha} is not finite"
        )));
    }
    Ok(alpha)
}

/// `alpha/rank` — the fold the residual carries, computed in f64 then narrowed, matching a
/// reference's Python-float arithmetic. The user's `AdapterSpec::scale` multiplies this at the
/// residual, so a strength-1.0 install folds at exactly this number.
pub fn alpha_rank_fold(alpha: f32, rank: f32) -> f32 {
    (alpha as f64 / rank as f64) as f32
}

/// The factors and optional per-target alpha grouped for one module path.
#[derive(Default)]
struct LoraParts {
    down: Option<Array>,
    up: Option<Array>,
    alpha: Option<f32>,
}

/// Read a scalar `.alpha` tensor as f32 regardless of its on-disk dtype (real files ship it f32 or
/// bf16; a direct `as_slice::<f32>()` would panic on a mismatch). `[]`- and `[1]`-shaped both read.
fn read_alpha_tensor(path: &str, a: &Array) -> Result<f32> {
    a.as_dtype(mlx_rs::Dtype::Float32)?
        .as_slice::<f32>()
        .first()
        .copied()
        .ok_or_else(|| Error::Msg(format!("minimax_h3 adapter '{path}': empty .alpha tensor")))
}

/// Install one diffusers-key-space LoRA file's residuals onto `host` at `spec.scale`.
fn apply_one_lora(
    host: &mut impl AdaptableHost,
    w: &Weights,
    spec: &AdapterSpec,
    report: &mut MiniMaxH3LoraReport,
) -> Result<()> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a LoRA factor key — ignore.
        };
        let path = normalize_minimax_h3_key(stem);
        let parts = groups.entry(path.clone()).or_default();
        match role {
            Role::Down => parts.down = Some(w.require(&key)?.clone()),
            Role::Up => parts.up = Some(w.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha_tensor(&path, w.require(&key)?)?),
        }
    }

    // The file-level alpha, read ONCE per file — the whole point of this crate-local loader. The PEFT
    // `lora_adapter_metadata` blob (sc-5513) is honored when present because a peft-saved file is a
    // legitimate input; the turbo files carry none, so they land on the `__metadata__["alpha"]` read
    // and its `DEFAULT_LORA_ALPHA` fallback.
    let blob = LoraAdapterMeta::from_metadata(w.metadata(LORA_ADAPTER_METADATA_KEY));
    let file_alpha = resolve_alpha(w)?;
    let meta_rank = w.metadata(RANK_METADATA_KEY);

    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // An orphan factor (or a bare `.alpha` whose partners targeted nothing) — surfaced, so a
            // malformed file fails loudly instead of half-applying.
            report.unmatched_paths.push(path);
            continue;
        };
        let (blob_alpha, blob_rank) = blob.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = resolve_rank(&path, &down, meta_rank)?;
        // A PEFT blob's `r`/`rank_pattern` goes through the SAME disagreement check as
        // `__metadata__["rank"]`, and for the same reason: the shapes are authoritative. PEFT itself
        // writes `r` equal to the factor rank, so a consistent blob is accepted and an inconsistent
        // one is a malformed file — never a silent override. Honouring it (what the shared loader
        // does at `loader.rs`'s `cfg_rank.unwrap_or(factor_rank)`) would let `{"r": 8}` over rank-128
        // factors fold at `8/8 = 1.0` instead of `8/128 = 0.0625`: exactly the 16x class this module
        // exists to close, with no error. `!=` also covers the F-141 non-finite class without a
        // second guard: `LoraAdapterMeta::from_metadata` already drops a `≤ 0` or NaN `r`, and an
        // INFINITE one — which it keeps — cannot equal the finite, positive shape rank, so it is
        // rejected here instead of folding a non-finite `alpha/rank` into the residual.
        if let Some(declared) = blob_rank {
            if declared != rank {
                return Err(Error::Msg(format!(
                    "minimax_h3 adapter '{path}': lora_adapter_metadata declares rank {declared} \
                     but the lora_A factor is rank {rank}; the shapes are authoritative, so this \
                     file is inconsistent rather than merely under-specified"
                )));
            }
        }
        // Precedence: per-target `.alpha` tensor → the PEFT blob → the file's top-level
        // `__metadata__["alpha"]` → `DEFAULT_LORA_ALPHA`. Only the last two differ from the shared
        // loader, and the last one is the correction: it never falls back to `rank`.
        let alpha = parts.alpha.or(blob_alpha).unwrap_or(file_alpha);
        let segs: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&segs) {
            Some(lin) => {
                // Residual form: a = downᵀ [in, rank], b = upᵀ [rank, out], with `alpha/rank` folded
                // into `b` exactly as the shared `install_lora_groups` does. Factors keep their
                // loaded (bf16) dtype so the residual promotes against the activation like the
                // reference does.
                let a = down.t();
                let fold = alpha_rank_fold(alpha, rank);
                // A **dtype-matched** scalar keeps `b` at the factor's dtype (the reference's
                // weak-float multiply), exactly as `mlx-gen-sdxl`'s and `mlx-gen-wan`'s installs do.
                // The property at stake is FORK PARITY, not host widening: every published turbo
                // file is bf16, so a bare f32 scalar would promote `b` to f32 and run the whole
                // low-rank `(x·A)·B` matmul in f32 where the reference runs it in bf16 — a silent
                // numeric deviation. It could NOT widen the host Linear's output, because
                // `AdaptableLinear::apply_adapters` (sc-15265) narrows every residual to
                // `out.dtype()` before the add; that seam is where widening is prevented.
                let b = up.t().multiply(scalar(fold).as_dtype(up.dtype())?)?;
                lin.push(Adapter::Lora {
                    a,
                    b,
                    scale: spec.scale,
                });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(())
}

/// Install every adapter in `specs` onto a MiniMax-H3 DiT, stacking in order.
///
/// **Strict, like every other family's install:** an unmatched target path, or **any single file**
/// that matched nothing, is an error, never a silent partial fold. The zero-match check is
/// **per-spec** and deliberately not an aggregate over the whole list: an aggregate
/// `report.applied == 0` lets `[good.safetensors, junk.safetensors]` return `Ok`, silently ignoring
/// the junk file. Most wrong-model files are caught anyway — their `.lora_A.weight` keys resolve to
/// no module and trip the unmatched guard below — but a file with **no recognized LoRA suffix at
/// all** (a merged LoRA, a base checkpoint, a textual inversion) contributes neither an `applied`
/// nor an `unmatched_path`, so only a per-spec check can see it. Downstream, the aggregate form
/// reaches a user as "my second LoRA did nothing, and there was no error".
///
/// Routing is by the **file**, not by
/// `spec.kind`: a genuine LoKr (`networkType=lokr`, or bare `lokr_*` keys) goes to the shared
/// LyCORIS seam, and everything else goes to the diffusers LoRA path above. That ordering is what
/// keeps the turbo files off `wmeta::parse_rank_alpha` — they carry an `alpha` string and no
/// `networkType`, so classifying them as LoKr would fold them 128× too strong.
pub fn apply_minimax_h3_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<MiniMaxH3LoraReport> {
    let mut report = MiniMaxH3LoraReport::default();
    for spec in specs {
        let before = report.applied;
        let w = Weights::from_file(&spec.path)?;
        if is_comfyui_key_space(&w) {
            return Err(Error::Msg(format!(
                "minimax_h3 adapter {}: this is a lightx2v `_comfyui_` export (fused \
                 `attn.qkv_proj`, and `mlp.fc1` carrying the SwiGLU halves as [gate | value] where \
                 the published DiT is [value | gate]). Folding it would be shape-valid and \
                 numerically wrong. Use the diffusers twin of the same adapter from the same repo \
                 — the file of the same name WITHOUT `_comfyui_`.",
                spec.path.display()
            )));
        }
        if is_lokr(&w) || is_lokr_keys(&w) {
            let ApplyReport {
                applied,
                unmatched_paths,
                ..
            } = apply_lokr(host, &w, spec.scale)?;
            report.applied += applied;
            report.unmatched_paths.extend(unmatched_paths);
        } else {
            if spec.kind == AdapterKind::Lokr {
                return Err(Error::Msg(format!(
                    "minimax_h3 adapter {}: declared LoKr but the file carries no `lokr_*` factors \
                     and no `networkType=lokr` stamp",
                    spec.path.display()
                )));
            }
            apply_one_lora(host, &w, spec, &mut report)?;
        }
        // Per-spec, NOT aggregate: this file, on its own, must have folded onto something. See the
        // doc comment — an aggregate check passes a junk file riding alongside a good one.
        if report.applied == before {
            return Err(Error::Msg(format!(
                "minimax_h3 adapter {}: no target modules matched in this file — expected the \
                 diffusers key space (`transformer_blocks.{{i}}.…` / `token_refiner.…` with \
                 `.lora_A.default.weight` / `.lora_B.default.weight`)",
                spec.path.display()
            )));
        }
    }
    if !report.unmatched_paths.is_empty() {
        return Err(Error::Msg(format!(
            "minimax_h3 adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// A `Weights` carrying `__metadata__` — built through a real safetensors round-trip, since
    /// `Weights` exposes no metadata setter and the metadata read is the thing under test.
    fn file_with_metadata(dir: &std::path::Path, name: &str, meta: &[(&str, &str)]) -> Weights {
        let path = dir.join(name);
        let map: std::collections::HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let t = Array::zeros::<f32>(&[2, 2]).unwrap();
        Array::save_safetensors(vec![("probe", &t)], Some(&map), &path).unwrap();
        Weights::from_file(&path).unwrap()
    }

    /// The three published `__metadata__` spellings resolve to the three alphas the turbo set
    /// actually ships — and the ABSENT case is 8, not the rank.
    #[test]
    fn alpha_resolution_covers_every_published_spelling() {
        let dir = tempfile::tempdir().unwrap();
        // `…_4step_v1.0_768p_bf16`.
        let w = file_with_metadata(dir.path(), "a.safetensors", &[("alpha", "128")]);
        assert_eq!(resolve_alpha(&w).unwrap(), 128.0);
        // `…_8step_v1.0_bf16` and `…_ref2v_…_4step_v0.1_bf16`.
        let w = file_with_metadata(dir.path(), "b.safetensors", &[("alpha", "8")]);
        assert_eq!(resolve_alpha(&w).unwrap(), 8.0);
        // `…_4step_v0.1` — no alpha at all.
        let w = file_with_metadata(dir.path(), "c.safetensors", &[("format", "pt")]);
        assert_eq!(
            resolve_alpha(&w).unwrap(),
            DEFAULT_LORA_ALPHA,
            "an absent alpha must fall back to upstream's DEFAULT_LORA_ALPHA"
        );
    }

    /// The fold each published file produces at rank 128. **The 0.0625 cases are the point**: a test
    /// that only covered the alpha=128 file would assert the code's own do-nothing default and prove
    /// nothing at all.
    #[test]
    fn published_alphas_produce_the_measured_folds() {
        assert_eq!(alpha_rank_fold(128.0, 128.0), 1.0, "4step_v1.0_768p");
        assert_eq!(alpha_rank_fold(8.0, 128.0), 0.0625, "8step_v1.0");
        assert_eq!(
            alpha_rank_fold(DEFAULT_LORA_ALPHA, 128.0),
            0.0625,
            "4step_v0.1"
        );
        // The two failure modes the story names, stated as the numbers they would produce.
        assert_eq!(
            alpha_rank_fold(128.0, 128.0) / alpha_rank_fold(8.0, 128.0),
            16.0,
            "falling back to alpha = rank folds the 8-step file 16x too strong"
        );
        assert_eq!(
            alpha_rank_fold(8.0, 1.0) / alpha_rank_fold(8.0, 128.0),
            128.0,
            "a rank defaulted to 1.0 (parse_rank_alpha) folds it 128x too strong"
        );
    }

    /// A present-but-malformed alpha must fail loudly rather than fall through to the default —
    /// otherwise the correction is indistinguishable from the bug it replaces.
    #[test]
    fn a_malformed_alpha_is_rejected_not_defaulted() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["", "eight", "nan", "inf"] {
            let w = file_with_metadata(dir.path(), "bad.safetensors", &[("alpha", bad)]);
            assert!(
                resolve_alpha(&w).is_err(),
                "alpha={bad:?} must be rejected, not silently treated as {DEFAULT_LORA_ALPHA}"
            );
        }
    }

    /// Rank comes from the factor shapes. A metadata `rank` is a cross-check only, and a
    /// disagreement is an error rather than a silent pick.
    #[test]
    fn rank_comes_from_the_factor_shapes() {
        let down = Array::zeros::<f32>(&[128, 5376]).unwrap();
        assert_eq!(resolve_rank("p", &down, None).unwrap(), 128.0);
        assert_eq!(resolve_rank("p", &down, Some("128")).unwrap(), 128.0);
        assert!(
            resolve_rank("p", &down, Some("1")).is_err(),
            "a metadata rank that disagrees with the shapes must not be silently preferred"
        );
        assert!(resolve_rank("p", &down, Some("oops")).is_err());
        // Zero rank would make `alpha/rank` non-finite.
        let empty = Array::zeros::<f32>(&[0, 8]).unwrap();
        assert!(resolve_rank("p", &empty, None).is_err());
        // A 1-D factor is malformed.
        let flat = Array::zeros::<f32>(&[8]).unwrap();
        assert!(resolve_rank("p", &flat, None).is_err());
    }

    /// The `.default` infix — the spelling the shared PEFT loader does not strip — plus the bare
    /// forms, all resolve to the checkpoint's own module path.
    #[test]
    fn the_default_infix_and_the_bare_forms_both_strip_to_a_module_path() {
        let cases = [
            (
                "transformer_blocks.0.attn.to_q.lora_A.default.weight",
                "transformer_blocks.0.attn.to_q",
                Role::Down,
            ),
            (
                "transformer_blocks.49.attn.to_out.0.lora_B.default.weight",
                "transformer_blocks.49.attn.to_out.0",
                Role::Up,
            ),
            (
                "token_refiner.refiner_blocks.1.ff.net.0.proj.lora_A.default.weight",
                "token_refiner.refiner_blocks.1.ff.net.0.proj",
                Role::Down,
            ),
            (
                "diffusion_model.transformer_blocks.3.ff.net.2.lora_B.weight",
                "transformer_blocks.3.ff.net.2",
                Role::Up,
            ),
            (
                "transformer.transformer_blocks.3.attn.to_v.lora_down.weight",
                "transformer_blocks.3.attn.to_v",
                Role::Down,
            ),
            (
                "transformer_blocks.3.attn.to_k.alpha",
                "transformer_blocks.3.attn.to_k",
                Role::Alpha,
            ),
        ];
        for (key, want, want_role) in cases {
            let (stem, role) = SUFFIXES
                .iter()
                .find_map(|(suf, r)| key.strip_suffix(suf).map(|s| (s, *r)))
                .unwrap_or_else(|| panic!("no suffix matched {key}"));
            assert_eq!(normalize_minimax_h3_key(stem), want, "key {key}");
            assert_eq!(role, want_role, "key {key}");
        }
        // A base weight is not a factor key.
        assert!(SUFFIXES
            .iter()
            .all(|(suf, _)| "transformer_blocks.0.attn.to_q.weight"
                .strip_suffix(suf)
                .is_none()));
    }

    /// The target list is the published module count and covers BOTH stacks.
    #[test]
    fn target_paths_cover_the_published_three_hundred_and_twelve_modules() {
        let cfg = MiniMaxH3DitConfig::default();
        let paths = adapter_target_paths(&cfg);
        assert_eq!(cfg.num_layers, 50);
        assert_eq!(cfg.num_refiner_layers, 2);
        assert_eq!(
            paths.len(),
            312,
            "50·6 + 2·6 — the turbo files' module count"
        );
        assert_eq!(paths.len() * 2, 624, "each module carries lora_A + lora_B");
        for expect in [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.49.attn.to_out.0",
            "transformer_blocks.49.ff.net.0.proj",
            "transformer_blocks.49.ff.net.2",
            "token_refiner.refiner_blocks.0.attn.to_k",
            "token_refiner.refiner_blocks.1.ff.net.2",
        ] {
            assert!(paths.iter().any(|p| p == expect), "missing {expect}");
        }
        assert_eq!(
            paths
                .iter()
                .filter(|p| p.starts_with("token_refiner."))
                .count(),
            12,
            "24 of the 624 tensors target the refiner — it cannot be stubbed"
        );
        assert!(
            !paths.iter().any(|p| p.contains("adaln")),
            "nothing in the turbo set targets AdaLN, so precompute-and-evict is unaffected"
        );
    }

    /// The `_comfyui_` key space is refused by name, and the diffusers one is not.
    #[test]
    fn the_comfyui_key_space_is_detected_and_the_diffusers_one_is_not() {
        let t = || Array::zeros::<f32>(&[2, 2]).unwrap();
        let mut comfy = Weights::empty();
        comfy.insert("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", t());
        assert!(is_comfyui_key_space(&comfy));
        let mut comfy_mlp = Weights::empty();
        comfy_mlp.insert("diffusion_model.blocks.0.mlp.fc1.lora_A.weight", t());
        assert!(is_comfyui_key_space(&comfy_mlp));
        let mut comfy_out = Weights::empty();
        comfy_out.insert("diffusion_model.blocks.0.attn.out_proj.lora_B.weight", t());
        assert!(is_comfyui_key_space(&comfy_out));

        let mut diffusers = Weights::empty();
        diffusers.insert(
            "transformer_blocks.0.attn.to_out.0.lora_A.default.weight",
            t(),
        );
        diffusers.insert(
            "transformer_blocks.0.ff.net.0.proj.lora_B.default.weight",
            t(),
        );
        assert!(
            !is_comfyui_key_space(&diffusers),
            "the diffusers export must not trip the comfyui guard"
        );
    }

    /// A property of **bf16 itself**, not of the install: scaling a bf16 array by an exact power of
    /// two is lossless, so the 0.0625 fold is not quietly rounded away and the bf16 test fixtures can
    /// assert exact folds.
    ///
    /// This test reimplements the multiply and therefore **cannot** see a change to the install's
    /// fold scalar — that guard is
    /// `turbo_lora::the_installed_fold_keeps_the_bf16_factor_dtype`, which asserts the dtype of the
    /// array `apply_minimax_h3_adapters` actually installed.
    #[test]
    fn a_bf16_array_scaled_by_a_power_of_two_is_exact() {
        let up = Array::from_slice(&[1.0f32, 2.0, 4.0, 8.0], &[2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let fold = alpha_rank_fold(8.0, 128.0);
        let scaled = up
            .multiply(
                Array::from_slice(&[fold], &[1])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            )
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(scaled.as_slice::<f32>(), [0.0625, 0.125, 0.25, 0.5]);
    }
}
