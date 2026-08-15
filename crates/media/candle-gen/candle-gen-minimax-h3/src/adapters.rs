//! MiniMax-H3 LoRA consumption on the **candle/CUDA** lane (sc-18728) — the key→module map, the
//! **alpha resolution rule** the lightx2v turbo checkpoints need, and the **ComfyUI key-space
//! conversion** (sc-19443).
//!
//! The twin of `mlx_gen_minimax_h3::adapters` (not linkable — the MLX crate is not a dependency
//! of this one). Every rule below is the same rule the MLX lane
//! enforces, because a LoRA that folded at a different strength per backend would make the quant
//! tier — a creative choice in this product — decide the picture.
//!
//! **Strategy: forward-time residual** over [`crate::dit::layers::LinearNoBias`], never a merged
//! weight, so `base(x) + Σ adapter.residual(x)` holds unchanged whatever the base is.
//!
//! sc-18728's text asked for a *tier-selected* install — fold on a dense tier, residual on a packed
//! one, after `candle-gen-wan`. That was written before sc-18724 landed, and the MLX lane it names
//! as the parity target chose **residual unconditionally** instead. This lane follows the parity
//! target rather than the story text, for two reasons that are checkable rather than stylistic:
//!
//! * A fold mutates the base weight, so it cannot run on a packed base without dequantizing it —
//!   which is precisely the tier the story says "is the one that matters". Residual-always is the
//!   superset: it is the only form that works on a packed tier, and it is exact on a dense one.
//! * This crate has **no tier loader at all** today — [`crate::model::descriptor`] advertises
//!   `supported_quants: &[]` and `load` refuses `spec.quantize` — so a tier-selected branch would
//!   have a dense arm that runs and a packed arm that nothing can reach. An unreachable branch is
//!   not a capability; it is an untested claim.
//!
//! # Key space: **diffusers**, plus a converted ComfyUI (sc-19443)
//!
//! `lightx2v/Minimax-h3-Turbo` publishes each adapter twice. The **diffusers** export is the native
//! key space:
//!
//! ```text
//! transformer_blocks.{0..49}.attn.{to_q,to_k,to_v,to_out.0}.lora_{A,B}.default.weight
//! transformer_blocks.{0..49}.ff.{net.0.proj,net.2}.lora_{A,B}.default.weight
//! token_refiner.refiner_blocks.{0,1}.…  (the same six leaves)
//! ```
//!
//! 624 tensors → **312 modules**, keyed exactly as [`crate::dit`] names its own weights, so the map
//! is a suffix strip and nothing else. Two properties of it are load-bearing:
//!
//! * **The `.default` infix is mandatory.** These are PEFT-with-adapter-name exports (upstream's own
//!   constants); `candle-gen-anima/src/adapters.rs` is the in-tree precedent for declaring it. The
//!   bare forms are accepted too, so a re-export that drops the adapter name still loads.
//! * **The token refiner cannot be stubbed.** 24 of the 624 tensors target
//!   `token_refiner.refiner_blocks.{0,1}`; a stub puts them in
//!   [`MiniMaxH3LoraReport::unmatched_paths`] and fails the strict install.
//!
//! The `_comfyui_` twin of each file is a **different module shape**, not a different spelling. It
//! is detected on the keys by [`is_comfyui_key_space`] and **converted** by
//! [`convert_comfyui_key_space`] rather than folded as-is: folding it un-converted is shape-valid
//! and computes the wrong thing, which is the sc-18740 defect class a fully green parity suite did
//! not see. See [`convert_comfyui_key_space`] for the three transforms and what each one is worth.
//!
//! # The alpha trap — why neither shared path can load these files
//!
//! The effective fold is `scale · alpha / rank`, and lightx2v's diffusers exports carry the alpha as
//! a **bare top-level `__metadata__` string** (`{"alpha": "8"}`) with **no `rank` key**, no
//! per-target `.alpha` tensor and no `lora_adapter_metadata` blob. A loader that falls back to
//! `alpha = rank` folds the 8-step file **16× too strong**; one that defaults a missing rank to 1.0
//! folds it **128×**. Neither errors.
//!
//! [`resolve_alpha`] is the fix, and its fallback is the whole point: **[`DEFAULT_LORA_ALPHA`] = 8,
//! never rank.** Alpha differs *per file inside one repo*, so it is read from the file every time —
//! measured by sc-18724 across the published set:
//!
//! | file | `__metadata__` | rank (from shapes) | effective scale at strength 1.0 |
//! | --- | --- | --- | --- |
//! | `…_4step_v1.0_768p_bf16` | `alpha: "128"` | 128 | **1.0** |
//! | `…_8step_v1.0_bf16` | `alpha: "8"` | 128 | **0.0625** |
//! | `…_ref2v_…_4step_v0.1_bf16` | `alpha: "8"` | 128 | **0.0625** |
//! | `…_4step_v0.1` | *(none)* | 128 | **0.0625** (the default alpha) |
//!
//! Rank comes from the **factor shapes** (`lora_A` is `[r, in]`), never from a metadata string. Both
//! declared spellings are cross-checks only: a `__metadata__["rank"]` that disagrees with the shapes
//! — or a PEFT `lora_adapter_metadata` `r` / `rank_pattern` that does — is a hard error, not a
//! silent pick.
//!
//! ## The precedence chain, in one place
//!
//! Alpha reaches a target through exactly one ordering, [`resolve_target_alpha`]:
//!
//! ```text
//! per-target `.alpha` tensor  →  PEFT `lora_adapter_metadata`  →  __metadata__["alpha"]  →  DEFAULT_LORA_ALPHA
//! ```
//!
//! **[`convert_comfyui_key_space`] resolves the same chain**, before it splits a fused `qkv_proj`,
//! and emits the split-adjusted result as an explicit per-target `.alpha`. That is not a tidiness
//! point: the conversion's block-diagonal `÷3` is what holds `alpha/rank` fixed across the un-fuse,
//! and a conversion that could only see the in-band `.alpha` tensor let the other three spellings
//! route around it — installing `Ok` and folding attention 3× too strong against the same file's
//! FFN. Since **no published file for this family carries an in-band `.alpha` at all**, that was the
//! dominant path, not an edge case.

use std::collections::{BTreeMap, HashMap};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen::train::lora::LoraAdapterMeta;
use candle_gen::train::merge::{read_adapter, read_scalar};
use candle_gen::{CandleError, Result};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::model::MiniMaxH3Dit;

/// The alpha upstream's `inference_minimax_h3.py` assumes when a LoRA declares none —
/// `DEFAULT_LORA_ALPHA = 8`.
///
/// **This is the fallback, and it is deliberately not the rank.** Falling back to the rank makes
/// `alpha/rank == 1.0`, which is the correct answer for exactly one of the published turbo files and
/// 16× too strong for the rest.
pub const DEFAULT_LORA_ALPHA: f32 = 8.0;

/// The top-level `__metadata__` key lightx2v stamps the alpha into, as a **string**.
pub const ALPHA_METADATA_KEY: &str = "alpha";

/// The top-level `__metadata__` key a rank would be stamped into. Read only to **cross-check** the
/// factor shapes — never as the source of the rank. See [`resolve_rank`].
pub const RANK_METADATA_KEY: &str = "rank";

/// LoRA key namespace prefixes stripped, longest-first. The published turbo files carry **none** of
/// these; the list exists so a re-export that adds the usual diffusers namespace still resolves.
/// `diffusion_model.` is the ComfyUI namespace.
pub const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
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

/// Which half of a LoRA a key names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// `lora_A` / `lora_down` — the `[rank, in]` down factor.
    Down,
    /// `lora_B` / `lora_up` — the `[out, rank]` up factor.
    Up,
    /// A per-target `.alpha` scalar (kohya / ComfyUI convention).
    Alpha,
}

/// Factor suffix → role, exact-matched, longest-first within each family. The **`.default`
/// adapter-name infix** is what the published turbo files use and what the shared PEFT loaders do
/// not strip; the bare forms are accepted too.
pub const SUFFIXES: [(&str, Role); 9] = [
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

/// Outcome of applying the MiniMax-H3 adapter specs.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MiniMaxH3LoraReport {
    /// Target modules that received a residual.
    pub applied: usize,
    /// Adapter module paths that resolved to no module — surfaced, never silently dropped.
    /// [`apply_minimax_h3_adapters`] refuses to return with this non-empty.
    pub unmatched_paths: Vec<String>,
    /// How many of the applied specs arrived in the ComfyUI key space and were converted
    /// (sc-19443). Zero for every published diffusers file.
    pub converted_from_comfyui: usize,
}

/// Every module path a MiniMax-H3 adapter can address, at `cfg`'s geometry: `num_layers`
/// transformer blocks and `num_refiner_layers` refiner blocks × the six [`BLOCK_TARGETS`]. 312 at
/// the shipped geometry (50·6 + 2·6), which is exactly the module count of each published turbo
/// file.
///
/// Derived from the config rather than hardcoded, and asserted to resolve through
/// [`MiniMaxH3Dit::adaptable_mut`], so this list and the model tree cannot drift apart.
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

/// Split a key into `(module_path, role)`, or `None` when it is not a LoRA factor key at all.
pub fn classify_key(key: &str) -> Option<(String, Role)> {
    SUFFIXES.iter().find_map(|(suf, role)| {
        key.strip_suffix(suf)
            .map(|stem| (normalize_minimax_h3_key(stem), *role))
    })
}

/// `true` if `keys` are a lightx2v **`_comfyui_`** export rather than the diffusers one.
///
/// Detected on the **keys** (`attn.qkv_proj` / `attn.out_proj` / `mlp.fc{1,2}` — module names that
/// do not exist in the diffusers layout at all) rather than only on a `target_format` metadata
/// stamp, so a re-export that drops the stamp is still caught.
///
/// **Not relaxed by sc-19443.** A file that reaches the fold path in the wrong key space is a
/// silent-corruption bug; sc-19443 changed what happens *after* detection (convert instead of
/// refuse), never the detection itself.
pub fn is_comfyui_key_space<'a>(keys: impl IntoIterator<Item = &'a str>) -> bool {
    keys.into_iter().any(|k| {
        k.contains(".attn.qkv_proj.") || k.contains(".attn.out_proj.") || k.contains(".mlp.fc")
    })
}

/// The LoRA rank for one module, from the **factor shapes**: `down` is stored `[rank, in]`, so the
/// rank is its leading dim. Mirrors upstream's `_validate_lora_state_dict`.
///
/// `meta_rank` is a top-level `__metadata__["rank"]`, read only to cross-check. A value that
/// disagrees with the shapes is an error rather than a silent pick.
pub fn resolve_rank(path: &str, down: &Tensor, meta_rank: Option<&str>) -> Result<f32> {
    if down.rank() != 2 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 adapter '{path}': lora_A must be 2-D [rank, in], got {:?}",
            down.dims()
        )));
    }
    let rank = down.dims()[0];
    if rank == 0 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 adapter '{path}': zero rank (empty lora_A factor) — `alpha/rank` would be \
             non-finite and NaN-poison the residual"
        )));
    }
    if let Some(raw) = meta_rank {
        let declared = raw.trim().parse::<f32>().map_err(|_| {
            CandleError::Msg(format!(
                "minimax_h3 adapter '{path}': __metadata__[\"{RANK_METADATA_KEY}\"] = {raw:?} is \
                 not a number"
            ))
        })?;
        if declared != rank as f32 {
            return Err(CandleError::Msg(format!(
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
/// An `alpha` that is present but unparseable or non-finite is an **error**, not a fall-through to
/// the default: a typo'd stamp silently folding at 8 is the same class of bug this function exists
/// to close.
pub fn resolve_alpha(meta: &HashMap<String, String>) -> Result<f32> {
    let Some(raw) = meta.get(ALPHA_METADATA_KEY) else {
        return Ok(DEFAULT_LORA_ALPHA);
    };
    let alpha = raw.trim().parse::<f32>().map_err(|_| {
        CandleError::Msg(format!(
            "minimax_h3 adapter: __metadata__[\"{ALPHA_METADATA_KEY}\"] = {raw:?} is not a number"
        ))
    })?;
    if !alpha.is_finite() {
        return Err(CandleError::Msg(format!(
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

/// **The one alpha precedence chain**, used by both the install and the ComfyUI conversion:
///
/// ```text
/// per-target `.alpha` tensor  →  PEFT `lora_adapter_metadata`  →  __metadata__["alpha"]  →  DEFAULT_LORA_ALPHA
/// ```
///
/// Factored out because the *conversion* has to resolve it too, and resolving only part of it there
/// is the sc-19443 review's measured silent-corruption bug: `convert_comfyui_key_space` used to see
/// nothing but the in-band `.alpha` tensor, so a ComfyUI file whose alpha lived in `__metadata__`
/// (the **dominant** spelling for this family — every published lightx2v file stamps it there and
/// ships no `.alpha` tensor at all) emitted no per-target alpha, and the block-diagonal `÷3` that
/// holds `alpha/rank` fixed across the qkv un-fuse never ran. The file then installed `Ok`, with no
/// error, folding attention **3× too strong** against its own FFN.
///
/// `file_alpha` is the caller's already-resolved [`resolve_alpha`] result, so a malformed
/// `__metadata__["alpha"]` is still an error rather than a fall-through to the default.
pub fn resolve_target_alpha(in_band: Option<f32>, blob_alpha: Option<f32>, file_alpha: f32) -> f32 {
    in_band.or(blob_alpha).unwrap_or(file_alpha)
}

// ─── sc-19443: the ComfyUI key space ───────────────────────────────────────────────────────────

/// The ComfyUI container prefix for the 50-block stack. Its diffusers spelling is
/// `transformer_blocks.`; the token refiner keeps its name in both.
const COMFY_BLOCK_CONTAINER: &str = "blocks.";

/// Map a ComfyUI *container* path onto the diffusers one, leaving an already-diffusers path alone.
///
/// The order matters: `token_refiner.refiner_blocks.` and `transformer_blocks.` both *contain*
/// `blocks.`, so they are matched first and returned unchanged. Only a bare leading `blocks.` — the
/// ComfyUI spelling of the trunk — is rewritten.
fn normalize_comfy_container(path: &str) -> String {
    if path.starts_with("transformer_blocks.") || path.starts_with("token_refiner.") {
        return path.to_string();
    }
    match path.strip_prefix(COMFY_BLOCK_CONTAINER) {
        Some(rest) => format!("transformer_blocks.{rest}"),
        None => path.to_string(),
    }
}

/// The four ComfyUI module names a MiniMax-H3 LoRA can name.
///
/// An **enum, not a `&str`**: `split_comfy_leaf` can only ever return one of these four, so a
/// `match` on the string form needed a catch-all arm that no input could reach — an unreachable arm
/// reads like a runtime guard while guarding nothing. With the enum the match is exhaustive by the
/// type system, and a fifth ComfyUI module cannot be added without the compiler naming every site
/// that must handle it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComfyLeaf {
    /// `attn.qkv_proj` — the fused q/k/v projection, split into three diffusers targets.
    Qkv,
    /// `attn.out_proj` → `attn.to_out.0`, a pure rename.
    OutProj,
    /// `mlp.fc1` → `ff.net.0.proj`, renamed **and** SwiGLU-half-swapped.
    Fc1,
    /// `mlp.fc2` → `ff.net.2`, a pure rename.
    Fc2,
}

/// Split `path` into `(container, comfy_leaf)` on the last ComfyUI module name it carries.
fn split_comfy_leaf(path: &str) -> Option<(&str, ComfyLeaf)> {
    for (name, leaf) in [
        ("attn.qkv_proj", ComfyLeaf::Qkv),
        ("attn.out_proj", ComfyLeaf::OutProj),
        ("mlp.fc1", ComfyLeaf::Fc1),
        ("mlp.fc2", ComfyLeaf::Fc2),
    ] {
        if let Some(head) = path.strip_suffix(name) {
            return Some((head.trim_end_matches('.'), leaf));
        }
    }
    None
}

/// Swap the two row halves of an output-side factor — `[gate | value] → [value | gate]`.
///
/// The DiT's `ff.net.0.proj` emits `[value | gate]` ([`crate::layout`]); ComfyUI's `mlp.fc1` emits
/// `[gate | value]`. A LoRA's `lora_B` is the **output-side** factor (`[2·ffn, rank]`), so the swap
/// lands on its rows and `lora_A` is untouched — the input contraction is unchanged.
///
/// Applying the fold without this swap computes `w2(silu(value)·gate)`: shape-identical, plausible
/// output, and the sc-18740 defect that shipped green at cosine 0.73–0.78.
fn swap_row_halves(path: &str, t: &Tensor) -> Result<Tensor> {
    let rows = t.dim(0)?;
    if rows % 2 != 0 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 adapter '{path}': a gated projection factor must have an even row count, \
             got {rows}"
        )));
    }
    let half = rows / 2;
    let gate = t.narrow(0, 0, half)?;
    let value = t.narrow(0, half, half)?;
    Ok(Tensor::cat(&[&value, &gate], 0)?.contiguous()?)
}

/// Whether `up` `[3·out, 3·r]` is **block-diagonal** at block `(out, r)` — i.e. every off-diagonal
/// block is exactly zero.
///
/// This is *measured on the bytes*, never assumed. It is what distinguishes the two legitimate fused
/// forms in [`convert_comfyui_key_space`], and getting it wrong in either direction changes the
/// per-projection rank and therefore the fold.
fn is_block_diagonal(up: &Tensor, out: usize, r: usize) -> Result<bool> {
    for row_block in 0..3 {
        for col_block in 0..3 {
            if row_block == col_block {
                continue;
            }
            let b = up
                .narrow(0, row_block * out, out)?
                .narrow(1, col_block * r, r)?
                .to_dtype(DType::F32)?;
            let m = b.abs()?.max_all()?.to_scalar::<f32>()?;
            if m != 0.0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// **Convert a ComfyUI-key-space MiniMax-H3 LoRA into the diffusers key space** (sc-19443).
///
/// Returns the converted tensor map. SceneWorks accepts user-supplied LoRAs (epics 15404 / 14015)
/// and community adapters are frequently distributed in ComfyUI format only, so this is a real input
/// path rather than a hypothetical. sc-18724 shipped a hard refusal here; the refusal was *correct*
/// but left the user with no path forward, and every transform below is exactly invertible, so the
/// conversion is chosen over confirming the refusal.
///
/// Three transforms, and each is worth stating because each is separately capable of producing a
/// runnable, wrong model:
///
/// 1. **`attn.qkv_proj` → `attn.to_q` / `to_k` / `to_v`.** Two fused forms are legitimate and are
///    told apart by measuring the bytes (`is_block_diagonal`) rather than assumed:
///    * **Block-diagonal** (`A [3r, in]`, `B [3·out, 3r]`, the lightx2v twins' form): split both,
///      giving factors byte-identical to the diffusers twin's, per-projection rank `r`, and an alpha
///      divided by 3. The published `24` becomes `8`, which against rank 128 is the same `0.0625`
///      the twin folds at — `24/384 == 8/128`.
///    * **Shared-`A`** (`A [r, in]`, `B [3·out, r]`, what a LoRA trained natively on the fused
///      module looks like): split `B`'s rows only and reuse `A` for all three, rank `r`, alpha
///      unchanged. Also exact.
///
///      An `A` and `B` whose inner dims disagree, or a `[3·out, 3r]` `B` that is neither
///      block-diagonal nor shared-`A`-shaped, is an error — not a guess.
///
///      **The alpha that gets divided is the fully resolved one**, which is why `meta` is a
///      parameter rather than something the caller applies afterwards. It comes from
///      [`resolve_target_alpha`]: the in-band `.alpha` tensor, else the PEFT
///      `lora_adapter_metadata` blob, else the file's top-level `__metadata__["alpha"]`, else
///      [`DEFAULT_LORA_ALPHA`] — and a per-target `.alpha` is emitted on **every** converted qkv
///      target, unconditionally. Resolving only the in-band spelling here (sc-19443's first cut)
///      left the other three routing around the `÷3` entirely: the file installed `Ok` and folded
///      attention 3× too strong, measured at rel-max-abs `2.007e0` against its own diffusers twin.
///      That is not an edge case for this family — **no published lightx2v file carries an in-band
///      `.alpha` at all**; every one of them stamps the alpha into `__metadata__` (see the module
///      header). The `__metadata__["rank"]` cross-check cannot catch it either, because these files
///      ship no `rank` key.
/// 2. **`mlp.fc1` → `ff.net.0.proj` with the SwiGLU halves swapped** (`swap_row_halves`): the DiT
///    emits `[value | gate]` and ComfyUI `[gate | value]`, and a LoRA's `lora_B` is the output-side
///    factor, so the swap lands on its rows and `lora_A` is untouched.
/// 3. **`attn.out_proj` → `attn.to_out.0`** and **`mlp.fc2` → `ff.net.2`** — pure renames; both
///    projections are unfused and unswapped on both sides.
///
/// The container prefix is normalized too (`blocks.{i}` → `transformer_blocks.{i}`), leaving an
/// export that already spells the trunk or the refiner the diffusers way alone.
///
/// **What this does not do.** It does not claim any *other* key space. A kohya (`lora_unet_`) or BFL
/// export still reaches the strict install and fails there, loudly, naming the paths that matched no
/// module — never a partial fold.
pub fn convert_comfyui_key_space(
    tensors: &HashMap<String, Tensor>,
    meta: &HashMap<String, String>,
) -> Result<HashMap<String, Tensor>> {
    // The file-level alpha sources, resolved ONCE. `resolve_alpha` still errors on a malformed
    // stamp, so a typo'd `__metadata__["alpha"]` fails here rather than folding at the default.
    let file_alpha = resolve_alpha(meta)?;
    let blob = LoraAdapterMeta::from_file_metadata(meta);
    let mut converted: HashMap<String, Tensor> = HashMap::new();
    // Spell a converted key back in the bare PEFT family, whatever family it arrived in.
    let suffix = |role: Role| match role {
        Role::Down => ".lora_A.weight",
        Role::Up => ".lora_B.weight",
        Role::Alpha => ".alpha",
    };

    // The fused `qkv_proj` needs all of its parts together, so collect first and emit after.
    let mut fused: BTreeMap<String, FusedQkv> = BTreeMap::new();

    for (key, t) in tensors {
        let Some((path, role)) = classify_key(key) else {
            // Not a LoRA factor key. The published ComfyUI twins carry nothing else; anything that
            // does is dropped here exactly as it would have been ignored downstream.
            continue;
        };
        let path = normalize_comfy_container(&path);
        let Some((container, leaf)) = split_comfy_leaf(&path) else {
            // Already a diffusers module path inside a ComfyUI-detected file — carry it through.
            converted.insert(format!("{path}{}", suffix(role)), t.clone());
            continue;
        };
        match leaf {
            ComfyLeaf::Qkv => {
                let slot = fused.entry(container.to_string()).or_default();
                match role {
                    Role::Down => slot.down = Some(t.clone()),
                    Role::Up => slot.up = Some(t.clone()),
                    Role::Alpha => slot.alpha = Some(read_scalar(key, "alpha", t)?),
                }
            }
            ComfyLeaf::OutProj => {
                converted.insert(
                    format!("{container}.attn.to_out.0{}", suffix(role)),
                    t.clone(),
                );
            }
            ComfyLeaf::Fc1 => {
                let target = format!("{container}.ff.net.0.proj{}", suffix(role));
                let v = match role {
                    // The output-side factor is the one the SwiGLU swap lands on.
                    Role::Up => swap_row_halves(&target, t)?,
                    Role::Down | Role::Alpha => t.clone(),
                };
                converted.insert(target, v);
            }
            ComfyLeaf::Fc2 => {
                converted.insert(format!("{container}.ff.net.2{}", suffix(role)), t.clone());
            }
        }
    }

    for (container, parts) in fused {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter '{container}.attn.qkv_proj': a fused qkv LoRA needs both \
                 lora_A and lora_B; one is missing"
            )));
        };
        if down.rank() != 2 || up.rank() != 2 {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter '{container}.attn.qkv_proj': factors must be 2-D, got \
                 a={:?} b={:?}",
                down.dims(),
                up.dims()
            )));
        }
        let r_fused = down.dims()[0];
        let (rows, r_up) = (up.dims()[0], up.dims()[1]);
        if r_fused != r_up {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter '{container}.attn.qkv_proj': lora_A is rank {r_fused} but \
                 lora_B is rank {r_up}; the two factors do not compose"
            )));
        }
        if rows == 0 || rows % 3 != 0 {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter '{container}.attn.qkv_proj': lora_B has {rows} rows, which is \
                 not three equal q/k/v projections"
            )));
        }
        let out_dim = rows / 3;
        // Block-diagonal is only *possible* when the fused rank divides by three; when it does, the
        // bytes decide. Shared-`A` is the fallback and is exact for any fused rank.
        let block_diagonal =
            r_fused > 0 && r_fused % 3 == 0 && is_block_diagonal(&up, out_dim, r_fused / 3)?;
        // **Resolve the alpha BEFORE the split, through the whole chain.** The fused module is the
        // one the file's alpha describes, so the value that must be divided is the one that would
        // have folded onto `attn.qkv_proj` — whichever of the four spellings carried it. A PEFT
        // `alpha_pattern` is keyed on the module the file names, which is the fused one.
        let fused_path = format!("{container}.attn.qkv_proj");
        let (blob_alpha, _) = blob
            .as_ref()
            .map_or((None, None), |c| c.effective(&fused_path));
        let fused_alpha = resolve_target_alpha(parts.alpha, blob_alpha, file_alpha);
        for (i, name) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            let stem = format!("{container}.attn.{name}");
            let (a, b) = if block_diagonal {
                let r = r_fused / 3;
                (
                    down.narrow(0, i * r, r)?.contiguous()?,
                    up.narrow(0, i * out_dim, out_dim)?
                        .narrow(1, i * r, r)?
                        .contiguous()?,
                )
            } else {
                (
                    down.clone(),
                    up.narrow(0, i * out_dim, out_dim)?.contiguous()?,
                )
            };
            converted.insert(format!("{stem}.lora_A.weight"), a);
            converted.insert(format!("{stem}.lora_B.weight"), b);
            // **The alpha division is the block-diagonal case's, and only its.** Splitting a
            // `[3r, in]` `A` divides the per-projection rank by three, so the alpha must divide by
            // three to hold `alpha/rank` fixed: the published `24/384` becomes `8/128`, both
            // `0.0625`. The shared-`A` form keeps rank `r`, so its alpha is unchanged — dividing
            // there would fold three times too weak.
            //
            // **Emitted unconditionally**, not `if let Some(parts.alpha)`. The conditional form was
            // the sc-19443 review's blocker: with no in-band `.alpha` the target carried none, and
            // the install downstream re-applied the *undivided* file-level alpha at the *already
            // split* rank — 3× too strong, `Ok`, no error. An explicit per-target alpha on every
            // converted qkv target is what makes the conversion closed: the install's own
            // precedence chain then reads this value first and cannot reach a stale file-level one.
            let scaled = if block_diagonal {
                fused_alpha / 3.0
            } else {
                fused_alpha
            };
            converted.insert(
                format!("{stem}.alpha"),
                Tensor::new(&[scaled], &Device::Cpu)?,
            );
        }
    }
    Ok(converted)
}

/// The three parts of one fused `attn.qkv_proj` module, collected before the split.
#[derive(Default)]
struct FusedQkv {
    down: Option<Tensor>,
    up: Option<Tensor>,
    alpha: Option<f32>,
}

// ─── the install ───────────────────────────────────────────────────────────────────────────────

/// The factors and optional per-target alpha grouped for one module path.
#[derive(Default)]
struct LoraParts {
    down: Option<Tensor>,
    up: Option<Tensor>,
    alpha: Option<f32>,
}

/// Install one diffusers-key-space LoRA file's residuals onto `host` at `scale`.
fn apply_one_lora(
    host: &mut MiniMaxH3Dit,
    tensors: &HashMap<String, Tensor>,
    meta: &HashMap<String, String>,
    scale: f32,
    report: &mut MiniMaxH3LoraReport,
) -> Result<()> {
    let device = host.device().clone();
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for (key, t) in tensors {
        let Some((path, role)) = classify_key(key) else {
            continue; // not a LoRA factor key — ignore.
        };
        let parts = groups.entry(path).or_default();
        match role {
            Role::Down => parts.down = Some(t.clone()),
            Role::Up => parts.up = Some(t.clone()),
            Role::Alpha => parts.alpha = Some(read_scalar(key, "alpha", t)?),
        }
    }

    // The file-level alpha sources, read ONCE per file — the whole point of this module. The PEFT
    // `lora_adapter_metadata` blob (sc-5374) is honored when present because a peft-saved file is a
    // legitimate input and the MLX twin honors it; the published turbo files carry none, so they
    // land on the `__metadata__["alpha"]` read and its `DEFAULT_LORA_ALPHA` fallback.
    let file_alpha = resolve_alpha(meta)?;
    let blob = LoraAdapterMeta::from_file_metadata(meta);
    let meta_rank = meta.get(RANK_METADATA_KEY).map(String::as_str);

    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // An orphan factor (or a bare `.alpha` whose partners targeted nothing) — surfaced, so
            // a malformed file fails loudly instead of half-applying.
            report.unmatched_paths.push(path);
            continue;
        };
        let (blob_alpha, blob_rank) = blob.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = resolve_rank(&path, &down, meta_rank)?;
        // A PEFT blob's `r`/`rank_pattern` goes through the SAME disagreement check as
        // `__metadata__["rank"]`, and for the same reason: the shapes are authoritative. PEFT writes
        // `r` equal to the factor rank, so a consistent blob is accepted and an inconsistent one is
        // a malformed file — never a silent override. The shared loaders' `cfg_rank
        // .unwrap_or(factor_rank)` would let `{"r": 8}` over rank-128 factors fold at `8/8 = 1.0`
        // instead of `8/128 = 0.0625`: the same 16× class this module exists to close, silently.
        if let Some(declared) = blob_rank {
            if declared != rank {
                return Err(CandleError::Msg(format!(
                    "minimax_h3 adapter '{path}': lora_adapter_metadata declares rank {declared} \
                     but the lora_A factor is rank {rank}; the shapes are authoritative, so this \
                     file is inconsistent rather than merely under-specified"
                )));
            }
        }
        // Precedence: per-target `.alpha` tensor → the PEFT blob → the file's top-level
        // `__metadata__["alpha"]` → `DEFAULT_LORA_ALPHA`. The last step is the correction: it never
        // falls back to `rank`. A converted ComfyUI file always carries a per-target `.alpha` on its
        // qkv targets, so the split-adjusted value wins here and the undivided file-level one is
        // unreachable — see [`convert_comfyui_key_space`].
        let alpha = resolve_target_alpha(parts.alpha, blob_alpha, file_alpha);
        let segs: Vec<&str> = path.split('.').collect();
        let Some(lin) = host.adaptable_mut(&segs) else {
            report.unmatched_paths.push(path);
            continue;
        };
        // Residual form: a = downᵀ [in, rank], b = upᵀ [rank, out], with `alpha/rank` folded into
        // `b`. `affine` multiplies **at the tensor's own dtype**, so a bf16 factor stays bf16 and
        // the low-rank `(x·A)·B` runs at the precision the reference runs it at. Widening `b` to f32
        // here would be invisible to every numeric assertion in the suite — every published fold is
        // an exact power of two, so the bf16 and f32 products are bit-identical. Only
        // `the_installed_fold_keeps_the_factor_dtype` can see it.
        let fold = alpha_rank_fold(alpha, rank);
        let a = down.t()?.contiguous()?.to_device(&device)?;
        let b = up
            .t()?
            .contiguous()?
            .affine(fold as f64, 0.0)?
            .to_device(&device)?;
        lin.push_lora(a, b, scale as f64)?;
        report.applied += 1;
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
/// no module and trip the unmatched guard — but a file with **no recognized LoRA suffix at all** (a
/// merged LoRA, a base checkpoint, a textual inversion) contributes neither an `applied` nor an
/// `unmatched_path`, so only a per-spec check can see it. The MLX lane fixed exactly this in review.
///
/// A `_comfyui_` export is **converted** ([`convert_comfyui_key_space`]) rather than refused
/// (sc-19443), and the conversion's own result is re-checked against [`is_comfyui_key_space`] so a
/// conversion that left a fused module behind fails here instead of reaching the fold.
///
/// LoKr is **not** supported on this lane. `AdapterKind::Lokr`, or a file carrying `lokr_*` factors,
/// is refused by name rather than run through the LoRA path — a LoKr's factors do not compose as
/// `(x·A)·B`, so treating one as a LoRA is a different operation rather than a weaker fold.
pub fn apply_minimax_h3_adapters(
    host: &mut MiniMaxH3Dit,
    specs: &[AdapterSpec],
) -> Result<MiniMaxH3LoraReport> {
    let mut report = MiniMaxH3LoraReport::default();
    for spec in specs {
        let before = report.applied;
        let af = read_adapter(&spec.path)?;
        if spec.kind == AdapterKind::Lokr || af.declares_lokr() || has_lokr_keys(&af.tensors) {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter {}: LoKr is not supported on the candle lane — this DiT's \
                 adapter seam installs `scale·((x·A)·B)` residuals only, and a Kronecker delta is \
                 a different operation rather than a weaker one. Use a LoRA export.",
                spec.path.display()
            )));
        }
        let comfy = is_comfyui_key_space(af.tensors.keys().map(String::as_str));
        let tensors = if comfy {
            // `af.meta` is the ORIGINAL file's header: the converted map is rebuilt key by key and
            // carries no metadata of its own, so the alpha the conversion divides has to come from
            // the file the user supplied.
            let converted = convert_comfyui_key_space(&af.tensors, &af.meta)?;
            if is_comfyui_key_space(converted.keys().map(String::as_str)) {
                return Err(CandleError::Msg(format!(
                    "minimax_h3 adapter {}: the ComfyUI conversion left a fused or unrenamed \
                     module behind — refusing rather than folding a half-converted file",
                    spec.path.display()
                )));
            }
            report.converted_from_comfyui += 1;
            converted
        } else {
            af.tensors.clone()
        };
        apply_one_lora(host, &tensors, &af.meta, spec.scale, &mut report)?;
        // Per-spec, NOT aggregate: this file, on its own, must have folded onto something.
        if report.applied == before {
            return Err(CandleError::Msg(format!(
                "minimax_h3 adapter {}: no target modules matched in this file — expected the \
                 diffusers key space (`transformer_blocks.{{i}}.…` / `token_refiner.…` with \
                 `.lora_A.default.weight` / `.lora_B.default.weight`), or a ComfyUI export this \
                 lane can convert",
                spec.path.display()
            )));
        }
    }
    if !report.unmatched_paths.is_empty() {
        return Err(CandleError::Msg(format!(
            "minimax_h3 adapters: {} adapter target(s) matched no module (surfaced, not silently \
             dropped): {:?}",
            report.unmatched_paths.len(),
            report.unmatched_paths
        )));
    }
    Ok(report)
}

/// Whether any key names a LyCORIS LoKr factor — the third-party spelling that carries no
/// `networkType` stamp.
fn has_lokr_keys(tensors: &HashMap<String, Tensor>) -> bool {
    tensors.keys().any(|k| k.contains(".lokr_w"))
}
