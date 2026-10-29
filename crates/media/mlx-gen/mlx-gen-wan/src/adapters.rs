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

/// Normalize a LoRA module path to the Wan checkpoint's naming (the reference
/// `_normalize_wan_lora_key`): strip a known prefix, then the `convert_wan` renames
/// `ffn.0/.2 → ffn.fc1/.fc2`, `text_embedding.0/.2 → text_embedding_0/_1`,
/// `time_embedding.0/.2 → time_embedding_0/_1`, `time_projection.1 → time_projection`,
/// `patch_embedding → patch_embedding_proj`. attn `q/k/v/o` pass through. Both the `.X.` infix and
/// the bare `…X` suffix forms are handled, as the reference does (a LoRA module stem ends at the
/// module, so the suffix forms fire).
///
/// Public because the Krea Realtime provider (`mlx-gen-krea-realtime`), whose DiT reuses this crate's
/// [`WanTransformer`](crate::WanTransformer) verbatim, normalizes Wan-family LoRA keys through it in its
/// own [`AdaptableHost`] before delegating to the inner Wan host
/// (sc-15015) — the same reference-faithful renames the in-crate LoRA install path uses.
pub fn normalize_wan_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // ffn.0 → ffn.fc1, ffn.2 → ffn.fc2
    t = t
        .replace(".ffn.0.", ".ffn.fc1.")
        .replace(".ffn.2.", ".ffn.fc2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.fc1");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.fc2");
    }

    // text_embedding.0/.2 → text_embedding_0/_1
    t = t
        .replace("text_embedding.0.", "text_embedding_0.")
        .replace("text_embedding.2.", "text_embedding_1.");
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}text_embedding_0");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}text_embedding_1");
    }

    // time_embedding.0/.2 → time_embedding_0/_1
    t = t
        .replace("time_embedding.0.", "time_embedding_0.")
        .replace("time_embedding.2.", "time_embedding_1.");
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}time_embedding_0");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}time_embedding_1");
    }

    // time_projection.1 → time_projection
    t = t.replace("time_projection.1.", "time_projection.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}time_projection");
    }

    // patch_embedding → patch_embedding_proj
    if t.contains("patch_embedding") && !t.contains("patch_embedding_proj") {
        t = t.replace("patch_embedding", "patch_embedding_proj");
    }
    t
}

/// Normalize a LoRA module path to the **diffusers** Wan-VACE checkpoint's naming (sc-3439). Unlike
/// [`normalize_wan_key`] (which targets the *native* converted Wan layout), the VACE transformer
/// ([`crate::vace::WanVaceTransformer`]) reads the diffusers tensor names directly, so the target is
/// already diffusers. This mirrors diffusers' own LoRA loader
/// (`_convert_non_diffusers_wan_lora_to_diffusers`): strip a known prefix, rename the **native** Wan
/// module spellings (which third-party trainers like musubi-tuner / diffusion-pipe emit) to their
/// diffusers equivalents, and pass an **already-diffusers** key (incl. every `vace_blocks.*` module,
/// which the diffusers converter does not touch and which only ever ships diffusers-named) through
/// unchanged. Renames (both the `.X.` infix and the bare `…X` suffix forms, as a LoRA module stem
/// ends at the module):
/// - `self_attn.{q,k,v,o}` → `attn1.{to_q,to_k,to_v,to_out.0}`
/// - `cross_attn.{q,k,v,o}` → `attn2.{to_q,to_k,to_v,to_out.0}`
/// - `ffn.0`/`ffn.2` → `ffn.net.0.proj`/`ffn.net.2`
/// - VACE block `before_proj`/`after_proj` → `proj_in`/`proj_out`
/// - `time_projection.1` → `condition_embedder.time_proj`; `head.head` → `proj_out`
/// - `text_embedding.0/.2` → `condition_embedder.text_embedder.linear_1/2`
/// - `time_embedding.0/.2` → `condition_embedder.time_embedder.linear_1/2`
///
/// The i2v `k_img`/`v_img` cross-attn factors (diffusers `add_k_proj`/`add_v_proj`) are intentionally
/// not mapped — the VACE host has no such modules, so they resolve to no weight and are surfaced
/// (skipped), never silently mis-folded.
pub(crate) fn normalize_vace_key(key: &str) -> String {
    let stripped = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key);
    let mut t = stripped.to_string();

    // VACE block hint projections: native before_proj/after_proj → diffusers proj_in/proj_out.
    t = t
        .replace(".before_proj.", ".proj_in.")
        .replace(".after_proj.", ".proj_out.");
    if let Some(h) = t.strip_suffix(".before_proj") {
        t = format!("{h}.proj_in");
    }
    if let Some(h) = t.strip_suffix(".after_proj") {
        t = format!("{h}.proj_out");
    }

    // Self-/cross-attn projections → attn1/attn2.{to_q,to_k,to_v,to_out.0}.
    for (src, dst) in [("self_attn", "attn1"), ("cross_attn", "attn2")] {
        for (n, d) in [
            ("q", "to_q"),
            ("k", "to_k"),
            ("v", "to_v"),
            ("o", "to_out.0"),
        ] {
            t = t.replace(&format!(".{src}.{n}."), &format!(".{dst}.{d}."));
            if let Some(h) = t.strip_suffix(&format!(".{src}.{n}")) {
                t = format!("{h}.{dst}.{d}");
            }
        }
    }

    // ffn.0/.2 → ffn.net.0.proj / ffn.net.2.
    t = t
        .replace(".ffn.0.", ".ffn.net.0.proj.")
        .replace(".ffn.2.", ".ffn.net.2.");
    if let Some(h) = t.strip_suffix(".ffn.0") {
        t = format!("{h}.ffn.net.0.proj");
    }
    if let Some(h) = t.strip_suffix(".ffn.2") {
        t = format!("{h}.ffn.net.2");
    }

    // Global modules (the diffusers converter's "Remaining" branch).
    t = t.replace("time_projection.1.", "condition_embedder.time_proj.");
    if let Some(h) = t.strip_suffix("time_projection.1") {
        t = format!("{h}condition_embedder.time_proj");
    }
    t = t.replace("head.head.", "proj_out.");
    if let Some(h) = t.strip_suffix("head.head") {
        t = format!("{h}proj_out");
    }
    t = t
        .replace(
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        )
        .replace(
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("text_embedding.0") {
        t = format!("{h}condition_embedder.text_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("text_embedding.2") {
        t = format!("{h}condition_embedder.text_embedder.linear_2");
    }
    t = t
        .replace(
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        )
        .replace(
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        );
    if let Some(h) = t.strip_suffix("time_embedding.0") {
        t = format!("{h}condition_embedder.time_embedder.linear_1");
    }
    if let Some(h) = t.strip_suffix("time_embedding.2") {
        t = format!("{h}condition_embedder.time_embedder.linear_2");
    }
    t
}

/// Merge one LoRA file's deltas into the weight map `w` at `spec.scale`, accumulating into `report`.
/// Mirrors `apply_lora_to_linear` per module: `ΔW = (B·A)·(alpha/rank·strength)` at the factor dtype,
/// cast to the weight dtype and added — so the no-LoRA forward and the merged forward share one bf16
/// GEMM (no per-step residual). Multiple files accumulate because each reads the (already-merged)
/// weight back from `w`.
fn merge_one(
    w: &mut Weights,
    spec: &AdapterSpec,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    // Adapter factors are lazy too: read them at load so neither the fold nor the first forward
    // waits on the safetensors reads (sc-24245; the SDXL/LTX adapter loaders' precedent).
    lw.materialize()?;
    if spec.kind == AdapterKind::Lokr || is_lokr(&lw) {
        // LoKr (sc-2393 — net-new; the reference Wan path is LoRA-only) merges through the same
        // in-place weight fold, with the delta reconstructed from Kronecker factors instead of B·A.
        return merge_one_lokr(w, &lw, spec.scale, normalize, report);
    }
    // Third-party LyCORIS (sc-3671): `lokr_*` / `hada_*` keys WITHOUT a `networkType=lokr` stamp
    // (kohya / ai-toolkit / lycoris-lib). `is_lokr` (peft) is handled above, so reaching here means
    // third-party; reconstruct per-module and merge like the peft path.
    if is_lokr_keys(&lw) {
        return merge_one_lokr_thirdparty(w, &lw, spec.scale, normalize, report);
    }
    if is_loha_keys(&lw) {
        return merge_one_loha_thirdparty(w, &lw, spec.scale, normalize, report);
    }

    // Group factors by normalized module path.
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a LoRA factor key (base weight / bundled extra) — ignore.
        };
        let parts = groups.entry(normalize(stem)).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_scalar_alpha(lw.require(&key)?, &key)?),
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without it (kohya / trainer files ship a `.alpha` tensor), in which case the per-target
    // `.alpha` or the factor rank is used exactly as before.
    let cfg = LoraAdapterMeta::from_metadata(lw.metadata(LORA_ADAPTER_METADATA_KEY));
    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            // A down/up whose partner targeted a non-LoRA key — skip the orphan, surface the path.
            report.skipped.push(path);
            continue;
        };
        let wkey = format!("{path}.weight");
        let Some(base) = w.get(&wkey).cloned() else {
            report.skipped.push(path);
            continue;
        };
        // Reject degenerate factors BEFORE deriving the scale (F-058, mirroring the additive twin's
        // guard at `install_one_lora_additive`). A non-2-D factor or a 0-size `lora_A` leading dim
        // would make `rank = 0`, so `alpha/rank = 0/0 = NaN` (or `∞`) silently replaces the ENTIRE base
        // weight with NaN via the merge below — every subsequent render is garbage while the load
        // reports success. Fail this one job with a typed error instead (sc-11129).
        if down.shape().len() != 2 || up.shape().len() != 2 {
            return Err(format!(
                "wan LoRA at '{path}' has non-2-D factors (down {:?}, up {:?})",
                down.shape(),
                up.shape()
            )
            .into());
        }
        if down.shape()[0] == 0 {
            return Err(format!("wan LoRA at '{path}' has zero rank").into());
        }
        // lora_A: [rank, in], lora_B: [out, rank]. delta = B·A → [out, in], the weight's shape.
        // Effective scaling: per-target `.alpha` tensor → `alpha_pattern`/`lora_alpha` blob → factor
        // rank (today's default). The denominator honors the blob `r`/`rank_pattern` when given
        // (always `> 0`), else the stored `down` leading dim (which equals it for a well-formed file).
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let rank = cfg_rank.map(|r| r as f64).unwrap_or(down.shape()[0] as f64);
        let alpha = parts.alpha.or(cfg_alpha).map(|a| a as f64).unwrap_or(rank);
        // (alpha/rank)·strength as a single value, matching the reference's Python-float `scale·strength`.
        let eff = (alpha / rank * spec.scale as f64) as f32;
        let delta = matmul(&up, &down)?;
        // Dtype-matched scalar preserves the factor dtype (the reference's weak `delta * (scale*strength)`).
        let delta = multiply(&delta, &scalar(eff).as_dtype(delta.dtype())?)?;
        let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
        w.insert(wkey, merged);
        report.applied += 1;
    }
    Ok(())
}

/// Merge one LoKr file's deltas into the weight map `w` at `scale` (sc-2393 — net-new; the reference
/// Wan path is LoRA-only). Each module's `[out,in]` delta is reconstructed (f32, `alpha/rank` folded
/// in) from its Kronecker factors via the core `reconstruct_lokr_delta`, scaled by the user strength,
/// and folded into the weight (cast to its dtype) — the same in-place merge as the LoRA path, so the
/// no-adapter forward stays byte-identical and adapters accumulate by reading the merged weight back.
/// A target absent from this checkpoint is surfaced (skipped), never fatal.
fn merge_one_lokr(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let file = parse_lokr(lw)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize(raw_path);
        let wkey = format!("{path}.weight");
        let Some(base) = w.get(&wkey).cloned() else {
            report.skipped.push(path);
            continue;
        };
        // Reconstruct f32 (the SDXL merge precedent, sc-2640) — the merge casts to the weight dtype.
        let delta = file.delta(factors, base.shape(), Dtype::Float32)?;
        let delta = multiply(&delta, &scalar(scale).as_dtype(delta.dtype())?)?;
        let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
        w.insert(wkey, merged);
        report.applied += 1;
    }
    Ok(())
}

/// Build the `flattened-stem → checkpoint-module-path` table from the expert weight map `w` (every
/// `‹path›.weight` key → `‹path›`), so a third-party LyCORIS file's kohya-flattened key resolves to a
/// Wan module (sc-3671). Dotted third-party keys instead go through [`normalize_wan_key`].
fn wan_module_table(w: &Weights) -> BTreeMap<String, String> {
    w.keys()
        .filter_map(|k| k.strip_suffix(".weight"))
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
}

/// Resolve a third-party LoKr/LoHa raw module key to a Wan checkpoint module path: prefer the
/// flattened-stem table (kohya `lora_unet_…`), else the dotted-path `normalize` (the host's rename
/// map — [`normalize_wan_key`] for native Wan, [`normalize_vace_key`] for the diffusers VACE host —
/// which a dotted diffusers third-party file shares with the peft path).
fn resolve_wan_thirdparty(
    raw: &str,
    table: &BTreeMap<String, String>,
    normalize: fn(&str) -> String,
) -> String {
    resolve_lokr_path(raw, table)
        .map(str::to_string)
        .unwrap_or_else(|| normalize(raw))
}

/// Merge one third-party-reconstructed `[out,in]` delta into `w` at the resolved module path
/// (sc-3671): `W += δ·scale` cast to the weight dtype, the same fold as LoRA/peft-LoKr. A path with no
/// weight in this expert is surfaced (skipped), never fatal.
fn merge_wan_thirdparty_delta(
    w: &mut Weights,
    path: String,
    delta_at: impl FnOnce(&[i32]) -> Result<Array>,
    scale: f32,
    report: &mut WanLoraReport,
) -> Result<()> {
    let wkey = format!("{path}.weight");
    let Some(base) = w.get(&wkey).cloned() else {
        report.skipped.push(path);
        return Ok(());
    };
    let delta = delta_at(base.shape())?;
    let delta = multiply(&delta, &scalar(scale).as_dtype(delta.dtype())?)?;
    let merged = add(&base, &delta.as_dtype(base.dtype())?)?;
    w.insert(wkey, merged);
    report.applied += 1;
    Ok(())
}

/// Merge a third-party LyCORIS **LoKr** file (kohya/lycoris keys, per-module `.alpha`, no
/// `networkType` stamp) into `w` at `scale` (sc-3671). Reconstruction reuses the core
/// `ThirdPartyLokr::delta` (f32, lycoris per-module scale baked in); install is the same in-place
/// weight fold as the peft `merge_one_lokr`.
fn merge_one_lokr_thirdparty(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = wan_module_table(w);
    for (raw, g) in &parse_lokr_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        merge_wan_thirdparty_delta(w, path, |bs| g.delta(bs, Dtype::Float32), scale, report)?;
    }
    Ok(())
}

/// Merge a third-party LyCORIS **LoHa** file into `w` at `scale` (sc-3671). As
/// [`merge_one_lokr_thirdparty`] with the Hadamard reconstruction (`ThirdPartyLoha::delta`).
fn merge_one_loha_thirdparty(
    w: &mut Weights,
    lw: &Weights,
    scale: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = wan_module_table(w);
    for (raw, g) in &parse_loha_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        merge_wan_thirdparty_delta(w, path, |bs| g.delta(bs, Dtype::Float32), scale, report)?;
    }
    Ok(())
}

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

/// Merge every adapter in `specs` that targets `expert` into the expert weight map `w` (sc-2683 LoRA /
/// sc-2393 LoKr). Shared
/// specs (`moe_expert == None`) are applied first, then this expert's specific ones (`Some(expert)`),
/// mirroring the reference `(loras)+(loras_high/low)` order so a module hit by both accumulates in
/// the same order. LoRA and LoKr are dispatched per file by metadata / the spec kind; per-key skips
/// are reported, not fatal (the reference warns on skip). Returns the merge report; the caller
/// enforces the "matched nothing across both experts" error.
pub fn merge_wan_adapters(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, expert, normalize_wan_key)
}

/// Merge every adapter in `specs` onto the **diffusers-layout** Wan-VACE transformer weight map `w`
/// (sc-3439). The VACE DiT ([`crate::vace::WanVaceTransformer`]) is a single dense model (no MoE), so
/// it takes only **shared** (untagged) specs — `MoeExpert::High` is passed purely so the untagged
/// pass fires; a `moe_expert`-tagged spec is a misconfiguration the caller
/// ([`crate::model_vace`]) rejects before calling here. Identical merge math + format dispatch as
/// [`merge_wan_adapters`] (PEFT/kohya LoRA, peft LoKr, third-party LyCORIS LoKr/LoHa), differing only
/// in the key→module map: `normalize_vace_key` targets the diffusers `attn1/attn2.{to_*}` +
/// `ffn.net.0.proj`/`net.2` + `vace_blocks.*` host instead of the native Wan layout.
pub fn merge_vace_adapters(w: &mut Weights, specs: &[AdapterSpec]) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, MoeExpert::High, normalize_vace_key)
}

/// Per-expert VACE adapter merge for the **dual-expert (Wan2.2-A14B) VACE-Fun** — the high/low sibling
/// of [`merge_vace_adapters`]. Shared (untagged) specs merge onto this `expert`'s weight map and a spec
/// tagged for the *other* expert is skipped, exactly like [`merge_wan_adapters`]'s `MoeExpert` routing,
/// but on the diffusers-named VACE key surface (`normalize_vace_key`). The caller
/// ([`crate::model_vace`]) merges once per expert (`MoeExpert::High` onto `transformer/`, `Low` onto
/// `transformer_2/`) and enforces the "matched nothing across both experts" error.
pub fn merge_vace_adapters_expert(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    merge_adapters_into(w, specs, expert, normalize_vace_key)
}

/// Shared merge core for both the native Wan ([`merge_wan_adapters`]) and the diffusers VACE
/// ([`merge_vace_adapters`]) hosts — only the `normalize` key→module map differs. Pass 1: shared
/// (untagged) files. Pass 2: this `expert`'s specific files (the reference `(loras)+(loras_*)` order).
fn merge_adapters_into(
    w: &mut Weights,
    specs: &[AdapterSpec],
    expert: MoeExpert,
    normalize: fn(&str) -> String,
) -> Result<WanLoraReport> {
    let mut report = WanLoraReport::default();
    let indexed = || specs.iter().enumerate();
    for (index, spec) in indexed().filter(|(_, s)| s.moe_expert.is_none()) {
        apply_recorded(&mut report, index, |r| merge_one(w, spec, normalize, r))?;
    }
    for (index, spec) in indexed().filter(|(_, s)| s.moe_expert == Some(expert)) {
        apply_recorded(&mut report, index, |r| merge_one(w, spec, normalize, r))?;
    }
    Ok(report)
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

/// Install every adapter in `specs` that targets `expert` onto the native-Wan [`AdaptableHost`]
/// `host` (a [`WanTransformer`](crate::transformer)) as **forward-time residuals** — the unmerged,
/// quant-agnostic sibling of [`merge_wan_adapters`] (sc-10044). The base weights are used AS-IS
/// (dense bf16 or packed Q4/Q8, never dequantized). Shared (`moe_expert == None`) specs install
/// first, then this `expert`'s specific ones, mirroring the reference `(loras)+(loras_*)` order.
/// Returns the same [`WanLoraReport`] the fold path returns; the caller enforces the
/// "matched nothing across both experts" error.
///
/// **Packed-snapshot routing (sc-10045 / sc-10050):** on a pre-quantized base the caller first calls
/// [`reject_loha_on_packed`] so only plain LoRA and LoKr reach here; LoRA installs via its low-rank
/// factors and LoKr via the **structured deferred-Kronecker** vec-trick (`install_one_lokr_additive`,
/// sc-10050) — both with the base staying packed, no `[out,in]` delta materialized. LoHa on a packed
/// tier stays deferred (sc-10051). On a dense base the fold path ([`merge_wan_adapters`]) is used
/// instead, so LoKr/LoHa keep working there unchanged.
pub fn apply_wan_adapters_additive(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    expert: MoeExpert,
) -> Result<WanLoraReport> {
    apply_additive_into(host, specs, expert, normalize_wan_key)
}

/// Shared additive-install core (the forward-time-residual analog of [`merge_adapters_into`]) — Pass 1
/// shared (untagged) files, Pass 2 this `expert`'s specific files. `normalize` maps a file's raw key
/// to the host's module path ([`normalize_wan_key`] for the native Wan DiT).
fn apply_additive_into(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
    expert: MoeExpert,
    normalize: fn(&str) -> String,
) -> Result<WanLoraReport> {
    let mut report = WanLoraReport::default();
    let indexed = || specs.iter().enumerate();
    for (index, spec) in indexed().filter(|(_, s)| s.moe_expert.is_none()) {
        apply_recorded(&mut report, index, |r| {
            install_one_additive(host, spec, normalize, r)
        })?;
    }
    for (index, spec) in indexed().filter(|(_, s)| s.moe_expert == Some(expert)) {
        apply_recorded(&mut report, index, |r| {
            install_one_additive(host, spec, normalize, r)
        })?;
    }
    Ok(report)
}

/// Install one adapter file onto `host` as residuals at `spec.scale` (the additive analog of
/// [`merge_one`]) — same per-file format dispatch (peft/kohya LoRA, peft LoKr, third-party LyCORIS
/// LoKr/LoHa), same normalized module paths, but pushing an [`Adapter`] rather than folding `W += δ`.
fn install_one_additive(
    host: &mut impl AdaptableHost,
    spec: &AdapterSpec,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    // Adapter factors are lazy too: read them at load so neither the fold nor the first forward
    // waits on the safetensors reads (sc-24245; the SDXL/LTX adapter loaders' precedent).
    lw.materialize()?;
    if spec.kind == AdapterKind::Lokr || is_lokr(&lw) {
        return install_one_lokr_additive(host, &lw, spec.scale, normalize, report);
    }
    if is_lokr_keys(&lw) {
        return install_one_lokr_thirdparty_additive(host, &lw, spec.scale, normalize, report);
    }
    if is_loha_keys(&lw) {
        return install_one_loha_thirdparty_additive(host, &lw, spec.scale, normalize, report);
    }
    install_one_lora_additive(host, &lw, spec.scale, normalize, report)
}

/// Push `adapter` onto the host module at the (already-normalized) native path `path`. A path outside
/// the host's adaptable surface (e.g. a non-block target, or a block index beyond this checkpoint) is
/// surfaced in `report.skipped`, never fatal — mirroring the fold path's per-target skip.
fn push_at(
    host: &mut impl AdaptableHost,
    path: String,
    adapter: Adapter,
    report: &mut WanLoraReport,
) {
    let parts: Vec<&str> = path.split('.').collect();
    match host.adaptable_mut(&parts) {
        Some(lin) => {
            lin.push(adapter);
            report.applied += 1;
        }
        None => report.skipped.push(path),
    }
}

/// Install a PEFT/kohya LoRA file as residuals. Mirrors [`merge_one`]'s grouping + the core
/// `install_lora_groups` factor convention: `a = downᵀ [in,rank]`, `b = upᵀ [rank,out]` with
/// `alpha/rank` folded into `b`, and `scale = strength` — so `scale·(x·A)·B` reproduces the fold's
/// `(alpha/rank·strength)·x·(B·A)ᵀ` within tolerance, with `W` untouched (dense OR packed).
fn install_one_lora_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue;
        };
        let parts = groups.entry(normalize(stem)).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_scalar_alpha(lw.require(&key)?, &key)?),
        }
    }

    let cfg = LoraAdapterMeta::from_metadata(lw.metadata(LORA_ADAPTER_METADATA_KEY));
    for (path, parts) in groups {
        let (Some(down), Some(up)) = (parts.down, parts.up) else {
            report.skipped.push(path);
            continue;
        };
        // `down` is [rank, in], `up` is [out, rank]. The residual form is `(x·A)·B` with A = downᵀ
        // [in, rank] and B = upᵀ [rank, out]; the `alpha/rank` scale is folded into B (matching the
        // core `install_lora_groups`), and the user strength stays as the `Adapter::Lora` scale.
        let a = down.t();
        let mut b = up.t();
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err(format!(
                "wan additive LoRA at '{path}' has non-2-D factors (down {:?}, up {:?})",
                down.shape(),
                up.shape()
            )
            .into());
        }
        let (cfg_alpha, cfg_rank) = cfg.as_ref().map_or((None, None), |c| c.effective(&path));
        let factor_rank = a.shape()[1] as f64; // r
        if factor_rank == 0.0 {
            return Err(format!("wan additive LoRA at '{path}' has zero rank").into());
        }
        // Fold `alpha/rank` into B when an alpha is present (per-target `.alpha` → blob), matching the
        // fold path's `eff = alpha/rank·strength` split (the strength stays the residual scale).
        if let Some(alpha) = parts.alpha.or(cfg_alpha) {
            let rank = cfg_rank.map(|r| r as f64).unwrap_or(factor_rank);
            let ratio = (alpha as f64 / rank) as f32;
            b = multiply(&b, &scalar(ratio).as_dtype(b.dtype())?)?;
        }
        push_at(
            host,
            path,
            Adapter::Lora {
                a,
                b,
                scale: strength,
            },
            report,
        );
    }
    Ok(())
}

/// Install a peft LoKr file as residuals (sc-2393/sc-10044/sc-10050). Per target, choose the install
/// by the base's quantization:
/// - **Packed (Q4/Q8)** base (sc-10050 / epic 10043): a **structured, deferred** Kronecker residual
///   ([`Adapter::LokrStructured`]) via the vec-trick — `scale·vec(w1·reshape(x)·w2ᵀ)` — so the full
///   `[out,in]` delta is NEVER materialized and the base stays packed at plain-LoRA memory cost. The
///   `alpha/rank` factor is baked into the structured `w2` (the user strength stays the residual scale).
/// - **Dense** base (unchanged, sc-2393/sc-10044): the fork-parity materialized `[out,in]` delta
///   (bf16, `alpha/rank` folded) stacked as an [`Adapter::Lokr`] — plenty of memory on dense, and the
///   established byte-parity path.
///
/// A non-deferrable LoKr variant (a **tucker/CP** `w2`, i.e. conv-only — the Wan DiT adapter surface
/// is all Linear, so this never fires) falls back to materialization on a **dense** base, or an
/// explicit typed error on a **packed** base (never a silent OOM / wrong result). A target outside the
/// adaptable surface is skipped.
fn install_one_lokr_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let file = parse_lokr(lw)?;
    for (raw_path, factors) in &file.groups {
        let path = normalize(raw_path);
        let parts: Vec<&str> = path.split('.').collect();
        let Some(lin) = host.adaptable_mut(&parts) else {
            report.skipped.push(path);
            continue;
        };
        let base_shape = lin.base_shape();
        if lin.is_quantized() {
            // Structured deferred path (packed base). The FULL scale `(alpha/rank)·strength` is baked
            // into `w2` — the structured residual carries no separate scale, so both the `alpha/rank`
            // fold (as in the dense delta) and the user strength must live in the factors here.
            match build_lokr_factors(
                (file.alpha / file.rank) * strength,
                &base_shape,
                factors.get("lokr_w1"),
                factors.get("lokr_w1_a"),
                factors.get("lokr_w1_b"),
                factors.get("lokr_w2"),
                None, // peft LoKr never carries a tucker factor (conv-only)
                factors.get("lokr_w2_a"),
                factors.get("lokr_w2_b"),
                Dtype::Bfloat16,
            )? {
                Some(structured) => {
                    lin.push(Adapter::LokrStructured {
                        factors: structured,
                    });
                    report.applied += 1;
                }
                None => {
                    return Err(non_deferrable_lokr_error(&path));
                }
            }
        } else {
            // Dense: the established materialized-delta residual (bf16, `alpha/rank` folded in).
            let delta = file.delta(factors, &base_shape, Dtype::Bfloat16)?;
            lin.push(Adapter::Lokr {
                delta,
                scale: strength,
            });
            report.applied += 1;
        }
    }
    Ok(())
}

/// The clear, actionable error for a LoKr variant that is genuinely NOT deferrable via the vec-trick
/// on a **packed** base (only the conv-only tucker/CP `w2` form — which the all-Linear Wan DiT adapter
/// surface never produces). Rather than a silent OOM or a wrong result, point the user at the bf16
/// tier (where it materializes into the dense weight). Dense bases fall back to materialization instead.
fn non_deferrable_lokr_error(path: &str) -> mlx_gen::Error {
    format!(
        "LoKr at '{path}' uses a tucker/CP (conv) factorization that cannot be applied on a \
         quantized (packed Q4/Q8) Wan tier without materializing the full delta — use the bf16 tier \
         (where it merges into the dense weight). This form does not occur for the Wan DiT's Linear \
         adapter targets (sc-10050 / epic 10043)."
    )
    .into()
}

/// The typed [`Error::Unsupported`] for a LoHa target on a packed base (F-060 / sc-10051). LoHa's
/// Hadamard delta has no deferred form, so applying it on a packed base would materialize the full
/// dense-weight-sized delta on top of the packed weights — the OOM the tier's low-memory point exists
/// to avoid. Steer the user to the bf16 tier (where LoHa folds into the dense weight in place) or a
/// plain LoRA/LoKr. Mirrors the file-level `reject_loha_on_packed`, per target.
fn loha_on_packed_target_error(path: &str) -> mlx_gen::Error {
    Error::Unsupported(format!(
        "LoHa at '{path}' (LyCORIS Hadamard) cannot be applied on a quantized (packed Q4/Q8) Wan \
         tier: its delta is a Hadamard product of two low-rank branches with no deferred/additive \
         form, so it would materialize the full dense weight-sized delta on top of the packed weights. \
         Use the bf16 tier for this adapter (LoHa merges into the dense weight in place), or a plain \
         LoRA/LoKr (which apply additively on any tier). Tracked in sc-10051 (epic 10043)."
    ))
}

/// Install a third-party LyCORIS **LoKr** file as residuals (sc-3671/sc-10044). As
/// [`install_one_lokr_additive`] but with per-module lycoris factors + key resolution (kohya-flattened
/// stem table, else the dotted `normalize`). The reconstructed delta bakes in the lycoris per-module
/// scale; the user strength stays the residual scale.
fn install_one_lokr_thirdparty_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = host_module_table(host);
    for (raw, g) in &parse_lokr_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        let parts: Vec<&str> = path.split('.').collect();
        let Some(lin) = host.adaptable_mut(&parts) else {
            report.skipped.push(path);
            continue;
        };
        let base_shape = lin.base_shape();
        if lin.is_quantized() {
            // Structured deferred path on a packed base (sc-10050): the lycoris per-module scale is
            // baked into the small `w2` factor; no `[out,in]` delta is materialized. A tucker/CP `w2`
            // (`lokr_t2`) has no 2-D form (`Ok(None)`) — the all-Linear Wan surface never hits it, but
            // guard it with the same clear error rather than a silent materialize.
            match g.factors(strength, &base_shape, Dtype::Bfloat16)? {
                Some(structured) => {
                    lin.push(Adapter::LokrStructured {
                        factors: structured,
                    });
                    report.applied += 1;
                }
                None => return Err(non_deferrable_lokr_error(&path)),
            }
        } else {
            // Dense: the established materialized-delta residual (lycoris scale baked in).
            let delta = g.delta(&base_shape, Dtype::Bfloat16)?;
            lin.push(Adapter::Lokr {
                delta,
                scale: strength,
            });
            report.applied += 1;
        }
    }
    Ok(())
}

/// Install a third-party LyCORIS **LoHa** file as residuals (sc-3671/sc-10044) — as
/// [`install_one_lokr_thirdparty_additive`] with the Hadamard reconstruction.
fn install_one_loha_thirdparty_additive(
    host: &mut impl AdaptableHost,
    lw: &Weights,
    strength: f32,
    normalize: fn(&str) -> String,
    report: &mut WanLoraReport,
) -> Result<()> {
    let table = host_module_table(host);
    for (raw, g) in &parse_loha_thirdparty(lw)? {
        let path = resolve_wan_thirdparty(raw, &table, normalize);
        install_thirdparty_delta(
            host,
            path,
            |bs| g.delta(bs, Dtype::Bfloat16),
            strength,
            report,
        )?;
    }
    Ok(())
}

/// Reconstruct a third-party `[out,in]` delta at the host target's base shape and stack it as an
/// [`Adapter::Lokr`] residual at `strength` — the additive analog of [`merge_wan_thirdparty_delta`].
/// A path outside the adaptable surface is surfaced (skipped), never fatal.
fn install_thirdparty_delta(
    host: &mut impl AdaptableHost,
    path: String,
    delta_at: impl FnOnce(&[i32]) -> Result<Array>,
    strength: f32,
    report: &mut WanLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    // SC-18319 — resolution here is shape/tier interrogation only (the install happens in `push_at`
    // below), so it goes through the PROBE half of the host surface.
    let base_shape = match host.adaptable_facts(&parts) {
        // Per-target packed-base guard (F-060, sc-10051). LoHa has no allocation-free deferred form —
        // every install materializes the full `[out,in]` bf16 delta. The LoKr additive installers
        // already refuse a materialization on a packed base per target; the LoHa installer did not, so
        // its only safety was the call-site file-level `reject_loha_on_packed` being invoked first. Add
        // the same per-target refusal so a future direct caller cannot silently materialize the
        // ~28 GB/expert dense delta on a packed base (the OOM sc-10051 exists to prevent). Typed
        // `Error::Unsupported` bridges 1:1 to gen_core so the worker surfaces actionable guidance.
        Some(facts) if facts.is_quantized => return Err(loha_on_packed_target_error(&path)),
        Some(facts) => facts.base_shape,
        None => {
            report.skipped.push(path);
            return Ok(());
        }
    };
    let delta = delta_at(&base_shape)?;
    push_at(
        host,
        path,
        Adapter::Lokr {
            delta,
            scale: strength,
        },
        report,
    );
    Ok(())
}

/// Build the `flattened-stem → native-module-path` table from the host's adaptable surface (every
/// [`AdaptableHost::adaptable_paths`] entry, dots→underscores), so a third-party LyCORIS file's
/// kohya-flattened key resolves to a native Wan module (the additive analog of [`wan_module_table`],
/// which reads the weight map — here we read the host, which is already built).
fn host_module_table(host: &impl AdaptableHost) -> BTreeMap<String, String> {
    host.adaptable_paths()
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect()
}

#[cfg(test)]
mod tests;
