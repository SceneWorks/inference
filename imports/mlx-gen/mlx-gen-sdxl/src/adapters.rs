//! SDXL adapter application — LoRA (sc-2639, a faithful Rust port of the vendored SceneWorks
//! `lora.py` merge for the mlx-examples SDXL U-Net) and LoKr (sc-2640).
//!
//! **LoKr** (`<module>.lokr_w1/w2` (+ low-rank `_a`/`_b`), `networkType=lokr` + `alpha`/`rank` meta):
//! the vendored SDXL path *rejects* LoKr, so there is no fork to match — Rust is strictly more
//! capable. The delta is reconstructed with the validated LyCORIS formula (`reconstruct_lokr_delta`,
//! f32 for the f32-everywhere SDXL path) and merged (`W += δ·scale`), chaos-safe like LoRA. Keys
//! resolve through the same kohya table (`lora_unet_<flat>.lokr_*`) or bare/PEFT dotted paths.
//!
//! **LoRA** — two on-disk formats, both **merged into the dense f32 U-Net weights at load** (`W += δ`, NOT a
//! forward-time residual): SDXL's ancestral sampler is chaos-sensitive, and a residual's
//! `W·x + δ·x` differs from the merged `(W+δ)·x` by ~1 ULP, which cascades to a visible whole-image
//! divergence. Merging reproduces the vendored merged-weight forward bit-for-bit.
//!
//! - **kohya** (`lora_unet_<diffusers path, `.`→`_`>.lora_down/up.weight` + optional `.alpha`) — what
//!   `pipe.save_lora_weights()` and most HF community SDXL LoRAs (incl. LCM-LoRA) ship. The
//!   `_`-flattening is ambiguous (diffusers names like `down_blocks`/`transformer_blocks` already
//!   contain `_`), so the flattened stem is resolved against a table built by flattening every
//!   routable module path — the Rust equivalent of the vendored `unet.named_modules()` walk.
//! - **PEFT** (`base_model.model.unet.<dotted path>.lora_A/B.default.weight` + optional `.alpha`) —
//!   what `peft.save_pretrained()` / SceneWorks' `_SdxlLoraBackend` emit. The dotted path resolves
//!   directly. (kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`.)
//!
//! Linear-only. Two coverage modes (see [`LoraCoverage`]):
//! - [`LoraCoverage::Complete`] (sc-2671) — **the `model::load` default** (Michael's
//!   correctness-over-parity call, 2026-06-03): applies SDXL LoRAs in **full**, matching diffusers.
//!   On top of the vendored-reachable set it routes `mid_block.attentions.0` (attention + proj) and
//!   the GEGLU feed-forward (`ff.net.0.proj` row-split into the value/gate halves `linear1`/`linear2`,
//!   `ff.net.2` → `linear3`) of every cross-attention transformer — signal the vendored merge silently
//!   drops. The per-module merge math is the same proven-bit-exact primitive; only the *reachable set*
//!   grows (plus the FF row-split, a bit-exact gather of the `B@A` delta).
//! - [`LoraCoverage::Vendored`] matches the vendored reachable surface **exactly** (515 modules on
//!   LCM-LoRA): down/up attention (`to_q/k/v`, `to_out.0`), `proj_in`/`proj_out`, resnet
//!   `time_emb_proj`. No `mid_block` (the vendored mlx-examples UNet names it `mid_blocks.1.…` so
//!   diffusers keys miss it), no ff/GEGLU, no conv, no text-encoder. `model::load` selects this only
//!   when `SDXL_LORA_VENDORED` is set — the escape hatch for byte-parity with the retired Python path.
//!
//! Either way, conv-shaped and out-of-surface keys are counted as skipped and surfaced in the
//! returned [`SdxlLoraReport`] — never silently dropped. **LoKr stays at the vendored-equivalent
//! surface regardless of coverage** (sc-2671 is LoRA-only; sc-2640 covered LoKr at that surface).

use std::collections::BTreeMap;

use mlx_rs::ops::{matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::loader::is_lokr;
use mlx_gen::adapters::{reconstruct_lokr_delta, AdaptableHost};
use mlx_gen::array::scalar;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::unet::UNet2DConditionModel;

const KOHYA_PREFIX: &str = "lora_unet_";
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// LoKr per-module factor suffixes (each factor is full `lokr_w1`/`lokr_w2` or a low-rank
/// `_a`/`_b` product). Exact-suffix matched; longest-first so `.lokr_w1_a` wins over `.lokr_w1`.
const LOKR_SUFFIXES: [&str; 6] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w1",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w2",
];

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Array>, // A: [rank, in]
    up: Option<Array>,   // B: [out, rank]
    alpha: Option<f32>,
}

/// Outcome of applying the SDXL adapter specs: how many module weights were merged, and how many
/// keys fell outside the routable surface (mid_block / ff / conv / text-encoder — surfaced, not
/// silently dropped). `merged` counts *base Linears updated*, so a GEGLU `ff.net.0.proj` LoRA (when
/// complete coverage row-splits it into `linear1`+`linear2`) contributes 2.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SdxlLoraReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

/// How much of the U-Net LoRA surface to reach.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoraCoverage {
    /// Faithful to the vendored `lora.py` (515 modules on LCM-LoRA): no `mid_block`, no GEGLU FF.
    /// Byte-parity with the retired Python SDXL path — the `model::load` escape hatch
    /// (`SDXL_LORA_VENDORED`), no longer the default.
    Vendored,
    /// Strictly more correct than the vendored path (sc-2671): also routes `mid_block.attentions.0`
    /// and the GEGLU feed-forward of every cross-attention transformer. **The `model::load` default**
    /// — applies SDXL LoRAs in full (matching diffusers).
    Complete,
}

/// A diffusers module path the vendored `lora.py` cannot reach — `mid_block.*` (named `mid_blocks.*`
/// in mlx-examples) or a GEGLU FF leaf (`*.ff.net.*`). Gated out under [`LoraCoverage::Vendored`].
fn is_complete_only(path: &str) -> bool {
    path.starts_with("mid_block") || path.contains(".ff.net.")
}

/// Build the kohya `flattened→dotted` lookup table from a routable-path list.
fn build_table(paths: Vec<String>) -> BTreeMap<String, String> {
    paths
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect()
}

/// Rows `[lo, hi)` of a 2-D array. Bit-exact for slicing a `B@A` LoRA delta: matmul output rows are
/// independent, so `rows(B@A)` equals `(rows(B))@A` to the last bit.
fn rows(a: &Array, lo: i32, hi: i32) -> Result<Array> {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    Ok(a.take_axis(&idx, 0)?)
}

/// Merge `delta` into the dense Linear at `dotted`, counting the merge (or surfacing a miss as
/// skipped). The path is the internal module path (FF already translated to `ff.linearN`).
fn merge_into(
    unet: &mut UNet2DConditionModel,
    dotted: &str,
    delta: &Array,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    match unet.adaptable_mut(&parts) {
        Some(lin) => {
            lin.merge_dense_delta(delta)?;
            report.merged += 1;
        }
        None => report.skipped_keys += 1,
    }
    Ok(())
}

/// Route a computed LoRA `delta` for diffusers `path` into the U-Net, translating the GEGLU FF:
/// `ff.net.0.proj` (a fused `[2·hidden, D]` proj) row-splits into the value half `ff.linear1` (rows
/// `[0:hidden]`) and the gate half `ff.linear2` (`[hidden:2·hidden]`); `ff.net.2` maps to `ff.linear3`.
/// Every other path (attention / proj / time_emb / mid_block) routes 1:1.
fn merge_lora_routed(
    unet: &mut UNet2DConditionModel,
    path: &str,
    delta: &Array,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    if let Some(prefix) = path.strip_suffix(".ff.net.0.proj") {
        let two_h = delta.shape()[0];
        let h = two_h / 2;
        merge_into(
            unet,
            &format!("{prefix}.ff.linear1"),
            &rows(delta, 0, h)?,
            report,
        )?;
        merge_into(
            unet,
            &format!("{prefix}.ff.linear2"),
            &rows(delta, h, two_h)?,
            report,
        )?;
        return Ok(());
    }
    if let Some(prefix) = path.strip_suffix(".ff.net.2") {
        return merge_into(unet, &format!("{prefix}.ff.linear3"), delta, report);
    }
    merge_into(unet, path, delta, report)
}

/// Map one safetensors key to `(diffusers_dotted_path, role)`, or `None` if it targets a module
/// outside the routable surface (mirrors the vendored `_classify_key` returning `(None, None)`).
fn classify_key(key: &str, kohya_to_dotted: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(PEFT_PREFIX) {
        // PEFT: the dotted diffusers path resolves directly. Accept the peft `.default.weight`
        // infix and the bare `.weight` form.
        for (suf, role) in [
            (".lora_A.default.weight", Role::Down),
            (".lora_B.default.weight", Role::Up),
            (".lora_A.weight", Role::Down),
            (".lora_B.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(path) = rem.strip_suffix(suf) {
                return Some((path.to_string(), role));
            }
        }
        return None;
    }
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        // kohya: resolve the flattened stem against the routable-path table.
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return kohya_to_dotted.get(stem).map(|d| (d.clone(), role));
            }
        }
        return None;
    }
    // `lora_te1_`/`lora_te2_`/… text-encoder keys land here — deliberately skipped (UNet-only).
    None
}

/// `δ = (B @ A) · (alpha/rank) · scale`, reproducing the vendored `lora.py` merge bit-for-bit.
///
/// The vendored computes `(b@a)` in the LoRA tensors' on-disk dtype (f16 for community/LCM LoRAs),
/// then `.astype(weight.dtype)` (f32) and `* effective_scale`. On the pmetal NAX build a 16-bit
/// `b@a` (K=rank≤512) would hit the dense GEMM bug, so we run the matmul in **f32** (correct) and
/// round the result back through the source dtype — MLX's f16 matmul equals `round_f16` of the
/// f32-accumulated product, so this is bit-identical to the reference without touching the bug.
pub fn lora_delta(down: &Array, up: &Array, alpha: f32, rank: f32, scale: f32) -> Result<Array> {
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ba = matmul(
        &up.as_dtype(Dtype::Float32)?,
        &down.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching the reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?)
}

fn read_scalar(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.reshape(&[1])?.as_slice::<f32>()[0])
}

/// Merge one LoRA file into `unet` at `scale`, classifying every key (both formats) and folding the
/// complete `(down, up)` pairs into their target weights. Half-pairs and out-of-surface / conv-shaped
/// keys are counted as skipped.
fn merge_one(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    coverage: LoraCoverage,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        match classify_key(&key, kohya_to_dotted) {
            Some((path, Role::Down)) => {
                triples.entry(path).or_default().down = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Up)) => {
                triples.entry(path).or_default().up = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(w.require(&key)?)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            // Half-pair (a down/up whose partner targeted a non-routable module) — skip.
            report.skipped_keys += 1;
            continue;
        };
        // Conv-shaped (4-D) LoRAs are not Linear merges (matches the vendored `ndim != 2` skip).
        if down.ndim() != 2 || up.ndim() != 2 {
            report.skipped_keys += 2;
            continue;
        }
        // Under vendored coverage, keep mid_block/ff out. kohya keys for those never reach here (the
        // table excludes them), but a PEFT key carries its dotted path directly — gate it here so the
        // faithful 515-module merge is byte-identical regardless of on-disk format.
        if coverage == LoraCoverage::Vendored && is_complete_only(&path) {
            report.skipped_keys += 1;
            continue;
        }
        let rank = down.shape()[0] as f32;
        let alpha = t.alpha.unwrap_or(rank);
        let delta = lora_delta(&down, &up, alpha, rank, scale)?;
        // Routes 1:1 for attention/proj/time_emb/mid_block; row-splits the GEGLU `ff.net.0.proj`.
        // A PEFT path naming a genuinely non-routable module surfaces as skipped inside `merge_into`.
        merge_lora_routed(unet, &path, &delta, report)?;
    }
    Ok(())
}

/// Map a LoKr key to `(diffusers_dotted_path, factor_name)`, or `None` if out of surface. kohya
/// `lora_unet_<flat>.lokr_*` resolves the flattened stem via the table; bare/PEFT `<dotted>.lokr_*`
/// (with an optional `base_model.model.unet.` prefix) resolves the dotted path directly.
fn classify_lokr_key(
    key: &str,
    kohya_to_dotted: &BTreeMap<String, String>,
) -> Option<(String, &'static str)> {
    for suf in LOKR_SUFFIXES {
        if let Some(stem) = key.strip_suffix(suf) {
            let factor = &suf[1..]; // drop the leading '.'
            return if let Some(flat) = stem.strip_prefix(KOHYA_PREFIX) {
                kohya_to_dotted.get(flat).map(|d| (d.clone(), factor))
            } else {
                Some((
                    stem.strip_prefix(PEFT_PREFIX).unwrap_or(stem).to_string(),
                    factor,
                ))
            };
        }
    }
    None
}

/// Merge one LoKr file into `unet` at `scale` (sc-2640). The vendored SDXL path *rejects* LoKr, so
/// there is no fork to match — we reconstruct the delta with the validated LyCORIS formula
/// (`reconstruct_lokr_delta`, f32 for the f32-everywhere SDXL path) and **merge** it (`W += δ·scale`),
/// chaos-safe like the LoRA path. `alpha`/`rank` come from the file metadata (alpha defaults to rank).
fn merge_one_lokr(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let rank = w
        .metadata("rank")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);
    let alpha = w
        .metadata("alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(rank);

    let mut grouped: BTreeMap<String, BTreeMap<&'static str, Array>> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        match classify_lokr_key(&key, kohya_to_dotted) {
            Some((path, factor)) => {
                grouped
                    .entry(path)
                    .or_default()
                    .insert(factor, w.require(&key)?.clone());
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, f) in grouped {
        // LoKr stays at the vendored-equivalent surface regardless of coverage (sc-2671 extends only
        // LoRA). mid_block is now routable, so gate it here to keep sc-2640's behaviour unchanged; FF
        // LoKr keys resolve to no internal Linear anyway (`ff.net.*` has no `adaptable_mut` arm).
        if is_complete_only(&path) {
            report.skipped_keys += 1;
            continue;
        }
        let parts: Vec<&str> = path.split('.').collect();
        match unet.adaptable_mut(&parts) {
            Some(lin) => {
                let base_shape = lin.base_shape();
                let delta = reconstruct_lokr_delta(
                    alpha,
                    rank,
                    &base_shape,
                    f.get("lokr_w1"),
                    f.get("lokr_w1_a"),
                    f.get("lokr_w1_b"),
                    f.get("lokr_w2"),
                    f.get("lokr_w2_a"),
                    f.get("lokr_w2_b"),
                    Dtype::Float32,
                )?;
                // The alpha/rank factor is baked into `delta`; apply the user scale on top (scale-0 ⇒
                // δ·0 ⇒ a bit-exact no-op merge).
                let delta = multiply(&delta, scalar(scale))?;
                lin.merge_dense_delta(&delta)?;
                report.merged += 1;
            }
            None => report.skipped_keys += 1,
        }
    }
    Ok(())
}

/// Merge every adapter spec in `specs` into `unet` — LoRA (sc-2639) and LoKr (sc-2640) — at the
/// **vendored-faithful** coverage (515 modules on LCM-LoRA; byte-parity with the retired Python
/// path). The vendored SDXL path supports LoRA only (it *rejects* LoKr); Rust is strictly more
/// capable. NOTE: `model::load` now defaults to [`LoraCoverage::Complete`] (sc-2671) — this faithful
/// entry point is reached only via the `SDXL_LORA_VENDORED` escape hatch. See
/// [`apply_sdxl_adapters_with`]. Errors if a non-empty spec list merges nothing (a real
/// format/prefix misconfiguration — e.g. an original-SD `lora_unet_input_blocks_*` file).
pub fn apply_sdxl_adapters(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
) -> Result<SdxlLoraReport> {
    apply_sdxl_adapters_with(unet, specs, LoraCoverage::Vendored)
}

/// As [`apply_sdxl_adapters`], but with an explicit [`LoraCoverage`]. [`LoraCoverage::Complete`]
/// (sc-2671) reaches `mid_block` + the GEGLU FF for LoRA — strictly more correct than the vendored
/// path, at the cost of byte-parity with it. The SDXL [`crate::model::load`] uses `Complete` by
/// default and falls back to `Vendored` only when `SDXL_LORA_VENDORED` is set.
pub fn apply_sdxl_adapters_with(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
    coverage: LoraCoverage,
) -> Result<SdxlLoraReport> {
    if specs.is_empty() {
        return Ok(SdxlLoraReport::default());
    }
    // LoKr is always merged against the vendored-equivalent surface; LoRA uses the coverage table.
    let vendored_table = build_table(unet.lora_target_paths());
    let complete_table = (coverage == LoraCoverage::Complete)
        .then(|| build_table(unet.lora_target_paths_complete()));
    let lora_table = complete_table.as_ref().unwrap_or(&vendored_table);

    let mut report = SdxlLoraReport::default();
    for spec in specs {
        let w = Weights::from_file(&spec.path)?;
        match spec.kind {
            AdapterKind::Lokr => {
                merge_one_lokr(unet, &w, spec.scale, &vendored_table, &mut report)?
            }
            AdapterKind::Lora => {
                // The file metadata is authoritative — a Lora-declared LoKr file would merge nothing
                // (no `lora_A/B`/`lora_down/up` keys); surface the mismatch loudly.
                if is_lokr(&w) {
                    return Err(Error::Msg(format!(
                        "sdxl: adapter {} declared Lora but its metadata says networkType=lokr",
                        spec.path.display()
                    )));
                }
                merge_one(unet, &w, spec.scale, lora_table, coverage, &mut report)?
            }
        }
    }

    if report.merged == 0 {
        return Err(Error::Msg(format!(
            "sdxl: no adapter target modules matched across {} file(s) — check the format \
             (expected kohya `lora_unet_` with diffusers block naming, PEFT \
             `base_model.model.unet.`, or LoKr `<module>.lokr_w1/w2` + networkType=lokr; \
             original-SD `lora_unet_input_blocks_*` and conv/ff-only adapters are not supported)",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> BTreeMap<String, String> {
        // A tiny routable surface: one attention leaf + a proj.
        [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
            "up_blocks.0.attentions.0.proj_in",
        ]
        .into_iter()
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
    }

    #[test]
    fn classify_kohya_resolves_flattened_stem_incl_to_out_0() {
        let t = table();
        // A kohya `to_q` key resolves through the flattened-stem table to its dotted path.
        let (path, role) = classify_key(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("kohya to_q should resolve");
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // up + alpha roles classify too.
        assert!(matches!(
            classify_key(
                "lora_unet_up_blocks_0_attentions_0_proj_in.lora_up.weight",
                &t
            )
            .unwrap()
            .1,
            Role::Up
        ));
        assert!(matches!(
            classify_key("lora_unet_up_blocks_0_attentions_0_proj_in.alpha", &t)
                .unwrap()
                .1,
            Role::Alpha
        ));
    }

    #[test]
    fn classify_skips_out_of_surface_and_text_encoder_keys() {
        let t = table();
        // mid_block / ff / conv stems aren't in the table → None (skipped, surfaced upstream).
        assert!(classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t
        )
        .is_none());
        assert!(classify_key(
            "lora_unet_down_blocks_1_resnets_0_conv1.lora_down.weight",
            &t
        )
        .is_none());
        // text-encoder LoRA keys are never UNet targets.
        assert!(classify_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &t
        )
        .is_none());
    }

    #[test]
    fn classify_lokr_resolves_factors_via_table_and_bare() {
        let t = table();
        // kohya LoKr factor resolves the flattened stem via the table; longest suffix wins
        // (`.lokr_w1_a` over `.lokr_w1`).
        let (path, factor) =
            classify_lokr_key("lora_unet_up_blocks_0_attentions_0_proj_in.lokr_w1_a", &t)
                .expect("kohya lokr_w1_a should resolve");
        assert_eq!(path, "up_blocks.0.attentions.0.proj_in");
        assert_eq!(factor, "lokr_w1_a");
        assert_eq!(
            classify_lokr_key("lora_unet_up_blocks_0_attentions_0_proj_in.lokr_w2", &t)
                .unwrap()
                .1,
            "lokr_w2"
        );
        // bare / PEFT dotted paths resolve directly (no table); off-surface kohya stems are None.
        assert_eq!(
            classify_lokr_key("base_model.model.unet.foo.bar.lokr_w1", &t).unwrap(),
            ("foo.bar".to_string(), "lokr_w1")
        );
        assert!(
            classify_lokr_key("lora_unet_mid_block_attentions_0_proj_in.lokr_w1", &t).is_none()
        );
    }

    #[test]
    fn is_complete_only_flags_mid_block_and_ff_only() {
        // mid_block (any leaf) and GEGLU FF leaves are complete-only.
        for p in [
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
            "mid_block.attentions.0.proj_in",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2",
        ] {
            assert!(is_complete_only(p), "{p} should be complete-only");
        }
        // The vendored-faithful surface is NOT complete-only.
        for p in [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0",
            "up_blocks.0.attentions.0.proj_in",
            "down_blocks.0.resnets.0.time_emb_proj",
        ] {
            assert!(
                !is_complete_only(p),
                "{p} must stay in the faithful surface"
            );
        }
    }

    #[test]
    fn classify_complete_table_resolves_mid_block_and_ff_stems() {
        // A complete-style table additionally carries mid_block + ff diffusers paths.
        let t: BTreeMap<String, String> = [
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
            "mid_block.attentions.0.proj_in",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2",
        ]
        .into_iter()
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect();
        let (mid, _) = classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("kohya mid_block attn should resolve under the complete table");
        assert_eq!(
            mid,
            "mid_block.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        let (ff, _) = classify_key(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight",
            &t,
        )
        .expect("kohya GEGLU ff.net.0.proj should resolve under the complete table");
        assert_eq!(
            ff,
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj"
        );
        // Those same stems are absent from the faithful table → None (proves the gate works by
        // table construction for kohya keys; PEFT keys are gated by `is_complete_only`).
        assert!(classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &table()
        )
        .is_none());
    }

    #[test]
    fn classify_peft_resolves_dotted_path_with_default_infix() {
        let t = table();
        // PEFT keys carry the dotted diffusers path directly (with the peft `.default.` infix).
        let (path, role) = classify_key(
            "base_model.model.unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
            &t,
        )
        .unwrap();
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // The bare `.weight` form (no `.default`) is also accepted.
        assert!(matches!(
            classify_key("base_model.model.unet.foo.bar.lora_B.weight", &t)
                .unwrap()
                .1,
            Role::Up
        ));
    }
}
