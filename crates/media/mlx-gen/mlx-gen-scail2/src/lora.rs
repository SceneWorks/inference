//! SCAIL-2 **diff-patch** + cross-architecture LoRA install (sc-5684).
//!
//! sc-5451 wired the family-agnostic *residual* LoRA path: a standard `lora_down`/`lora_up` (+`alpha`)
//! file installs onto the DiT as forward-time residuals over the (possibly Q4/Q8) base. That covers
//! every SCAIL-2-native LoRA (the Bias-Aware DPO refinement LoRA, any adapter trained on SCAIL-2). It
//! does **not** cover the **lightx2v cross-architecture step-distill ("lightning") LoRAs**, which add
//! two things the residual loader can't consume:
//!
//!   1. **Diff-patch tensors.** Alongside the low-rank factors the file carries full-rank `.diff`
//!      (weight delta) and `.diff_b` (bias delta) tensors — including on layers the residual host never
//!      exposes as adapter targets: the qk-RMSNorms (`norm_q`/`norm_k`/`norm_k_img.diff`), the affine
//!      `norm3` / `img_emb.proj.{0,4}` LayerNorms, the output `head.head` (a full `.diff` rather than
//!      low-rank factors), and a `.diff_b` on **every** biased projection. This is the ComfyUI "diff
//!      patch" mechanism.
//!   2. **Cross-architecture shape mismatch.** The LoRA targets vanilla **Wan2.1-I2V-14B**, whose
//!      `patch_embedding` has in_dim 36; SCAIL-2's is in_dim **20** (16 z + 4 i2v mask) plus the extra
//!      `patch_embedding_{pose,mask}` stems. So `patch_embedding.diff` (`[5120, 36, 1, 2, 2]`) is
//!      shape-incompatible with SCAIL-2's `[5120, 20, 1, 2, 2]` and must be **deliberately skipped**.
//!      The transformer blocks (dim 5120 q/k/v/o/ffn/k_img/v_img), the dim-5120 globals, and the
//!      `img_emb` stack ARE compatible and DO transfer — only the input patch-embed differs.
//!
//! **Mechanism — split by ROLE, so the file is tier-independent (sc-18198).** A lightning file's two
//! halves want opposite treatment, and each half's targets happen to be exactly the ones that suit it:
//!
//!   * **Full-rank `.diff`/`.diff_b`** are folded directly into the raw [`Weights`] map here, *before*
//!     the DiT is built. This is what reaches the norms and biases the residual host does not expose
//!     as adapter targets. Crucially, every full-rank target — the qk-norms, `norm3`, `head.head`,
//!     `img_emb.proj.{0,4}`, and every Linear bias — stays **dense bf16 even in the pre-quantized
//!     q4/q8 tiers**, because MLX packs only the 2-D block projections. So the fold needs no
//!     dequantization and works identically on every tier.
//!   * **Low-rank `lora_down`/`lora_up` factors** are NOT folded here at all. They target precisely
//!     the projections that DO get packed, and they install as forward-time residuals *after* the
//!     build and quantize — the one form that composes with a packed base, and already how the
//!     Bias-Aware DPO LoRA loads (sc-5451).
//!
//! Nothing in the file ever requires a dense delta to meet a packed weight, so there is no
//! dequantize/requantize step and no tier gate. This replaced a blanket "needs the DENSE (bf16)
//! snapshot" rejection that failed the whole file whenever the snapshot was pre-quantized — which made
//! the bundled `scail2_lightning` toggle unusable on the default (q4) tier. A full-rank delta that
//! genuinely cannot be carried (a `.diff` on a packed target, with no low-rank factor to stand in) is
//! now refused per-target by [`report_outcome`] rather than pre-empted for the whole file.
//!
//! **Shape-aware skipping is loud, never silent.** A target whose weight-delta shape doesn't match
//! the SCAIL-2 base (the in_dim-36 `patch_embedding`) is skipped *as a whole module* — its coupled
//! bias delta `.diff_b` is dropped too, since it was trained jointly with the incompatible weight
//! delta — and surfaced in the report (and a file that matches nothing is a hard error). This is the
//! same "never half-apply a LoRA" contract the strict residual installer enforces.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, Error, Result};
use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

/// LoRA key namespace prefixes stripped (longest-first). The lightx2v files use `diffusion_model.`;
/// SCAIL-2's converted DiT keys are the bare `SCAIL2Model` parameter names (no `ffn.0 → ffn.fc1`
/// rename — unlike the *converted* Wan checkpoint — because SCAIL-2 ships raw I2V module names).
const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "base_model.model.",
    "model.",
];

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
    /// Full-rank weight delta (`.diff`).
    Diff,
    /// Full-rank bias delta (`.diff_b`).
    DiffB,
}

/// Factor / diff suffixes (exact match). `.diff_b` precedes `.diff` so a bias-delta key never mis-binds
/// as a weight delta; the kohya `lora_down`/`lora_up` and PEFT `lora_A`/`lora_B` conventions are both
/// accepted. A key matching none of these (a bundled base weight, say) is ignored.
const SUFFIXES: [(&str, Role); 7] = [
    (".lora_down.weight", Role::Down),
    (".lora_up.weight", Role::Up),
    (".lora_A.weight", Role::Down),
    (".lora_B.weight", Role::Up),
    (".alpha", Role::Alpha),
    (".diff_b", Role::DiffB),
    (".diff", Role::Diff),
];

/// The deltas a diff-patch file carries for one module.
#[derive(Default)]
struct Parts {
    down: Option<Array>,   // lora_A / lora_down → [rank, in]
    up: Option<Array>,     // lora_B / lora_up   → [out, rank]
    alpha: Option<f32>,    // per-target `.alpha` (rare in diff-patch files)
    diff: Option<Array>,   // full-rank weight delta, shape == base weight
    diff_b: Option<Array>, // full-rank bias delta, shape == base bias
}

/// What a diff-patch merge did: counts of merged weights/biases and the targets deliberately skipped
/// (cross-architecture shape mismatch) or absent from the SCAIL-2 checkpoint.
#[derive(Debug, Default)]
pub struct DiffPatchReport {
    pub merged_weights: usize,
    pub merged_biases: usize,
    /// Targets skipped because their weight-delta shape is incompatible with SCAIL-2 (the in_dim-36
    /// vanilla-Wan `patch_embedding`). Surfaced loudly — never silently dropped.
    pub skipped_cross_arch: Vec<String>,
    /// Targets that resolved to no weight in the SCAIL-2 checkpoint (orphan factor / unknown module).
    pub skipped_unmatched: Vec<String>,
    /// Modules whose low-rank factors were deliberately left for the post-quantize residual pass
    /// rather than folded here (sc-18198). This is every low-rank target, on every tier. Counted so the "matched nothing" guard can tell a file that legitimately
    /// carried only low-rank factors from one whose prefixes are misconfigured.
    pub deferred_low_rank: usize,
    /// Modules carrying a full-rank `.diff` whose base weight is PACKED (pre-quantized on disk), so
    /// the dense delta cannot be folded and no low-rank factor exists to carry it. A hard error —
    /// applying the rest would silently ship a partial patch. Empty for the lightx2v lightning file,
    /// whose every full-rank target is dense on every tier (sc-18198).
    pub blocked_packed_diff: Vec<String>,
}

/// `true` if `path` is a diff-patch ("lightning") LoRA — a file carrying any full-rank `.diff` (weight
/// delta) or `.diff_b` (bias delta), which the forward-time residual loader cannot consume.
pub fn has_diff_patch_keys(path: &Path) -> Result<bool> {
    let w = Weights::from_file(path)?;
    let found = w
        .keys()
        .any(|k| k.ends_with(".diff") || k.ends_with(".diff_b"));
    Ok(found)
}

/// `true` if `path` carries anything the post-quantize residual installer could install — i.e. any
/// tensor NOT consumed by the diff-patch fold above.
///
/// Defined by EXCLUSION on purpose. The caller needs this because the residual installer is *strict*:
/// handing it a file whose every key is a full-rank `.diff` resolves zero targets and is reported as
/// "matched nothing", turning a legal pure-diff-patch adapter into a hard error. But enumerating the
/// installable spellings instead would silently drop every format not on the list — the installer also
/// handles **LoKr** (`lokr_w1`/`lokr_w2`, full or `_a`/`_b` low-rank, identified by `networkType`
/// metadata) and **LoHa**, and the SCAIL-2 engine descriptor advertises `supports_lokr`. An
/// allow-list of LoRA factor suffixes would have quietly excluded those from the residual pass — the
/// exact silent drop this module refuses to make anywhere else.
///
/// `.alpha` is excluded alongside the deltas: it is a scalar scaling modifier, never a target on its
/// own, so a file of nothing but `.diff` + `.alpha` correctly has nothing left to install.
pub fn has_residual_installable_keys(path: &Path) -> Result<bool> {
    let w = Weights::from_file(path)?;
    let found = w
        .keys()
        .any(|k| !k.ends_with(".diff") && !k.ends_with(".diff_b") && !k.ends_with(".alpha"));
    Ok(found)
}

/// Cast to f32 for the merge math (the deltas + base are folded in f32, then cast back to the base
/// dtype on write so the snapshot keeps its bf16 footprint).
fn f32a(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// Read a scalar `.alpha` as f32 regardless of on-disk dtype.
fn read_alpha(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.as_slice::<f32>()[0])
}

fn strip_namespace(key: &str) -> &str {
    PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p))
        .unwrap_or(key)
}

/// Fold every diff-patch adapter's **full-rank** `.diff`/`.diff_b` deltas into the SCAIL-2 weight map
/// `w`, in place (sc-5684). Call this on the freshly-loaded `dit.safetensors` weights *before*
/// [`crate::Scail2Dit`] is built and *before* any load-time quantization. Multiple files accumulate
/// (each reads the already-merged weight back from `w`).
///
/// `w` may be a **pre-quantized (q4/q8) tier** as well as the dense bf16 one (sc-18198): a target
/// whose base is packed is detected by its `{stem}.scales` sibling and left alone, and the low-rank
/// factors that target those packed projections are deferred to the post-quantize residual installer
/// by design — this function never folds low-rank factors on any tier. The caller must pass the same
/// specs to that installer; [`report_outcome`] refuses the one case neither pass can carry.
pub fn merge_diff_patch_adapters(
    w: &mut Weights,
    specs: &[&AdapterSpec],
) -> Result<DiffPatchReport> {
    let mut report = DiffPatchReport::default();
    for spec in specs {
        merge_one(w, spec, &mut report)?;
    }
    Ok(report)
}

fn merge_one(w: &mut Weights, spec: &AdapterSpec, report: &mut DiffPatchReport) -> Result<()> {
    let lw = Weights::from_file(&spec.path)?;
    // Group every factor / diff key by its SCAIL-2 module path (namespace prefix stripped).
    let mut groups: BTreeMap<String, Parts> = BTreeMap::new();
    for key in lw.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = SUFFIXES
            .iter()
            .find_map(|(suf, role)| key.strip_suffix(suf).map(|s| (s, *role)))
        else {
            continue; // not a factor / diff key — ignore.
        };
        let parts = groups.entry(strip_namespace(stem).to_string()).or_default();
        match role {
            Role::Down => parts.down = Some(lw.require(&key)?.clone()),
            Role::Up => parts.up = Some(lw.require(&key)?.clone()),
            Role::Alpha => parts.alpha = Some(read_alpha(lw.require(&key)?)?),
            Role::Diff => parts.diff = Some(lw.require(&key)?.clone()),
            Role::DiffB => parts.diff_b = Some(lw.require(&key)?.clone()),
        }
    }
    // The `lora_adapter_metadata` alpha/rank blob (sc-5513) is deliberately NOT read here any more:
    // since sc-18198 this fold handles only full-rank `.diff`/`.diff_b`, which are absolute deltas
    // scaled by `spec.scale` alone — alpha/rank scaling applies to low-rank factors, and those are
    // now installed by the residual loader, which resolves the blob itself.
    for (stem, parts) in groups {
        merge_module(w, &stem, &parts, spec.scale, report)?;
    }
    Ok(())
}

/// Fold one module's deltas into `w`. Weight delta = `strength·diff + (alpha/rank)·strength·(up·down)`
/// (whichever the file carries), bias delta = `strength·diff_b` — accumulated in f32, written back at
/// the base dtype. A weight-delta shape that doesn't match the SCAIL-2 base means a cross-architecture
/// target: skip the whole module (weight AND its coupled bias) and record it.
fn merge_module(
    w: &mut Weights,
    stem: &str,
    parts: &Parts,
    strength: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    let wkey = format!("{stem}.weight");
    let Some(base_w) = w.get(&wkey).cloned() else {
        report.skipped_unmatched.push(stem.to_string());
        return Ok(());
    };
    // A pre-quantized-on-disk tier stores each quantized Linear as packed `{stem}.weight` (u32) plus
    // sibling `{stem}.scales` / `{stem}.biases`. The `.scales` sibling is the reliable marker — the
    // packed weight's own dtype/shape say nothing a dense check could use. NOTE `{stem}.biases` (the
    // quantization zero-points) is a DIFFERENT tensor from `{stem}.bias` (the Linear's bias); only
    // the latter is a `.diff_b` target, and it stays dense on every tier.
    let packed = w.get(&format!("{stem}.scales")).is_some();

    // --- low-rank factors: NEVER folded here (sc-18198) ---
    // They install as forward-time residuals AFTER the DiT is built and quantized, which is the one
    // form that works over a packed base — and is what SCAIL-2 already does for the Bias-Aware DPO
    // LoRA. Folding them here instead would (a) be impossible on a packed tier and (b) make the
    // whole file tier-gated for no reason, since the low-rank factors are the ONLY part of a
    // lightning file that ever targets a quantized Linear. Deferring them uniformly — on dense tiers
    // too — keeps one code path per tier rather than two divergent ones; on an unquantized base a
    // residual and a merged delta are the same arithmetic.
    match (&parts.down, &parts.up) {
        (Some(_), Some(_)) => report.deferred_low_rank += 1,
        (None, None) => {}
        _ => {
            // An orphan low-rank factor (its partner targeted a non-LoRA key) — surface it here
            // rather than leaving the residual pass to infer the mismatch.
            report.skipped_unmatched.push(stem.to_string());
            return Ok(());
        }
    }

    // --- full-rank weight delta (`.diff`, @ strength) ---
    if let Some(diff) = &parts.diff {
        if packed {
            // Nothing can carry this: a dense delta cannot fold into packed u32 weights, and a
            // full-rank `.diff` has no low-rank factor for the residual pass to install instead.
            // Applying the rest of the file would ship a silently partial patch, so this is fatal
            // (raised together for every affected target by `report_outcome`). The lightx2v
            // lightning file never reaches here: its full-rank targets are the qk-norms, `norm3`,
            // `head.head`, `img_emb.proj.{0,4}` and every Linear bias, all of which stay dense
            // BF16 in the q4/q8 tiers — only the 2-D block projections are packed, and those carry
            // low-rank factors.
            report.blocked_packed_diff.push(stem.to_string());
            return Ok(());
        }
        if diff.shape() != base_w.shape() {
            report.skipped_cross_arch.push(stem.to_string());
            return Ok(()); // cross-arch (e.g. patch_embedding in_dim 36 vs 20) — skip whole module.
        }
        let d = multiply(&f32a(diff)?, scalar(strength))?;
        let merged = add(&f32a(&base_w)?, &d)?.as_dtype(base_w.dtype())?;
        w.insert(wkey, merged);
        report.merged_weights += 1;
    }

    // --- bias delta (`.diff_b`, @ strength) ---
    if let Some(diff_b) = &parts.diff_b {
        let bkey = format!("{stem}.bias");
        let Some(base_b) = w.get(&bkey).cloned() else {
            report.skipped_unmatched.push(bkey);
            return Ok(());
        };
        if diff_b.shape() != base_b.shape() {
            report.skipped_cross_arch.push(bkey);
            return Ok(());
        }
        let bd = multiply(&f32a(diff_b)?, scalar(strength))?;
        let merged = add(&f32a(&base_b)?, &bd)?.as_dtype(base_b.dtype())?;
        w.insert(bkey, merged);
        report.merged_biases += 1;
    }
    Ok(())
}

/// Surface a diff-patch merge's skips loudly (the only channel at load time — there is no `Progress`
/// callback yet), and error if the file(s) matched *nothing* (a format/prefix misconfiguration that
/// would otherwise silently no-op). Mirrors the wan loader's `warn_skipped_adapters` + "matched
/// nothing" contract.
pub fn report_outcome(report: &DiffPatchReport, model_id: &str) -> Result<()> {
    if !report.skipped_cross_arch.is_empty() {
        eprintln!(
            "{model_id}: lightx2v diff-patch LoRA — {} cross-architecture target(s) deliberately \
             skipped (shape-incompatible with SCAIL-2, e.g. the in_dim-36 vanilla-Wan2.1-I2V \
             patch_embedding vs SCAIL-2's in_dim 20): {:?}",
            report.skipped_cross_arch.len(),
            report.skipped_cross_arch
        );
    }
    if !report.skipped_unmatched.is_empty() {
        eprintln!(
            "{model_id}: lightx2v diff-patch LoRA — {} target(s) not present in the SCAIL-2 \
             checkpoint, skipped: {:?}",
            report.skipped_unmatched.len(),
            report.skipped_unmatched
        );
    }
    // A full-rank delta on a packed base can be carried by nothing — not the fold (it cannot unpack)
    // and not the residual pass (there is no low-rank factor for that target). Refusing here is the
    // difference between a loud failure and a silently half-applied patch, so it is fatal even though
    // every other target may have applied cleanly (sc-18198).
    if !report.blocked_packed_diff.is_empty() {
        return Err(Error::Msg(format!(
            "{model_id}: {} diff-patch target(s) carry a full-rank `.diff` whose base weight is \
             pre-quantized on disk, which cannot be applied without dequantizing the tier — and \
             they carry no low-rank factor to install as a residual instead. Refusing rather than \
             applying a partial patch. Load the dense `bf16` tier for this adapter, or re-export it \
             with low-rank factors for these targets: {:?}",
            report.blocked_packed_diff.len(),
            report.blocked_packed_diff
        )));
    }
    // `deferred_low_rank` counts targets handed to the post-quantize residual pass. They are real
    // matches, so a file that is mostly low-rank (or whose full-rank targets were all cross-arch
    // skips) must not read as "matched nothing" — the residual installer runs its own strict
    // zero-match guard over exactly those targets.
    if report.merged_weights + report.merged_biases + report.deferred_low_rank == 0 {
        return Err(Error::Msg(format!(
            "{model_id}: the diff-patch LoRA matched no SCAIL-2 module (every target skipped) — \
             likely a format / prefix mismatch, or the wrong base model"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_scail2_lora_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn f32(values: Vec<f32>, shape: &[i32]) -> Array {
        Array::from_slice(&values, shape)
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// A small SCAIL-2-shaped dense weight map: one block projection (Linear + bias), one qk-RMSNorm
    /// (weight only), and a `patch_embedding` Conv stem (5-D weight + bias) standing in for the
    /// cross-architecture target — all the distinct diff-patch module shapes.
    fn synthetic_dit() -> Weights {
        let path = tmp("dit.safetensors");
        let q_w = f32(
            (0..16 * 8).map(|i| i as f32 * 0.01 - 0.3).collect(),
            &[16, 8],
        );
        let q_b = f32((0..16).map(|i| i as f32 * 0.02).collect(), &[16]);
        let norm_w = f32(vec![1.0; 8], &[8]);
        // patch_embedding Conv3d weight [out, in=4, 1, 2, 2] + bias [out].
        let pe_w = f32(
            (0..16 * 4 * 4).map(|i| i as f32 * 0.001).collect(),
            &[16, 4, 1, 2, 2],
        );
        let pe_b = f32((0..16).map(|i| i as f32 * 0.03).collect(), &[16]);
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.weight", &q_w),
                ("blocks.0.self_attn.q.bias", &q_b),
                ("blocks.0.self_attn.norm_q.weight", &norm_w),
                ("patch_embedding.weight", &pe_w),
                ("patch_embedding.bias", &pe_b),
            ],
            None,
            &path,
        )
        .unwrap();
        Weights::from_file(&path).unwrap()
    }

    /// A diff-patch LoRA over the synthetic DiT: the q projection gets low-rank factors + a `.diff_b`
    /// bias delta; norm_q gets a full-rank `.diff`; patch_embedding gets a shape-INCOMPATIBLE `.diff`
    /// (in_dim 6 vs 4) + a (shape-compatible) `.diff_b` — the cross-architecture case.
    fn write_diff_patch(name: &str) -> PathBuf {
        let path = tmp(name);
        let rank = 4;
        let down = f32(
            (0..rank * 8)
                .map(|i| (i as f32 * 0.01).sin() * 0.1)
                .collect(),
            &[rank, 8],
        );
        let up = f32(
            (0..16 * rank)
                .map(|i| (i as f32 * 0.02).cos() * 0.1)
                .collect(),
            &[16, rank],
        );
        let q_diff_b = f32((0..16).map(|i| i as f32 * 0.005).collect(), &[16]);
        let norm_diff = f32((0..8).map(|i| i as f32 * 0.01).collect(), &[8]);
        // in_dim 6 ≠ the base's 4 → must be skipped as cross-architecture.
        let pe_diff = f32(
            (0..16 * 6 * 4).map(|i| i as f32 * 0.001).collect(),
            &[16, 6, 1, 2, 2],
        );
        let pe_diff_b = f32((0..16).map(|i| i as f32 * 0.04).collect(), &[16]);
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    &down,
                ),
                ("diffusion_model.blocks.0.self_attn.q.lora_up.weight", &up),
                ("diffusion_model.blocks.0.self_attn.q.diff_b", &q_diff_b),
                ("diffusion_model.blocks.0.self_attn.norm_q.diff", &norm_diff),
                ("diffusion_model.patch_embedding.diff", &pe_diff),
                ("diffusion_model.patch_embedding.diff_b", &pe_diff_b),
            ],
            None,
            &path,
        )
        .unwrap();
        path
    }

    fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
        AdapterSpec::new(path, scale, mlx_gen::AdapterKind::Lora)
    }

    #[test]
    fn detects_diff_patch_file() {
        let dp = write_diff_patch("detect.safetensors");
        assert!(has_diff_patch_keys(&dp).unwrap());
        // A pure low-rank file (no .diff/.diff_b) is NOT a diff-patch file.
        let plain = tmp("plain.safetensors");
        let down = f32(vec![0.1; 4 * 8], &[4, 8]);
        let up = f32(vec![0.1; 16 * 4], &[16, 4]);
        Array::save_safetensors(
            vec![
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    &down,
                ),
                ("diffusion_model.blocks.0.self_attn.q.lora_up.weight", &up),
            ],
            None,
            &plain,
        )
        .unwrap();
        assert!(!has_diff_patch_keys(&plain).unwrap());
    }

    #[test]
    fn merges_lora_diff_and_diffb_skips_cross_arch() {
        let dp = write_diff_patch("merge.safetensors");
        let mut w = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(dp.clone(), 1.0)]).unwrap();

        // sc-18198: ONLY the full-rank `.diff` folds here. q's low-rank factors are deferred to the
        // post-quantize residual installer (on every tier, not just packed ones), so `norm_q` is the
        // single merged weight and q is counted as deferred instead.
        assert_eq!(report.merged_weights, 1, "norm_q (.diff) only");
        assert_eq!(report.merged_biases, 1, "q.diff_b");
        assert_eq!(report.deferred_low_rank, 1, "q's low-rank factors deferred");
        assert!(report.blocked_packed_diff.is_empty());
        // patch_embedding is the lone cross-architecture skip (its .diff is in_dim 6 vs base 4); its
        // .diff_b is dropped with it even though [16] would have matched.
        assert_eq!(
            report.skipped_cross_arch,
            vec!["patch_embedding".to_string()]
        );
        assert!(report.skipped_unmatched.is_empty());

        // patch_embedding stays bit-identical (skipped entirely — weight AND bias).
        let base = synthetic_dit();
        for k in ["patch_embedding.weight", "patch_embedding.bias"] {
            assert!(
                array_eq(w.require(k).unwrap(), base.require(k).unwrap(), false)
                    .unwrap()
                    .item::<bool>(),
                "{k} must be untouched (cross-arch skip)"
            );
        }

        // q.weight must be left BIT-IDENTICAL: its low-rank factors are the residual installer's job
        // now. Folding them here as well would double-apply them, since the same spec goes to both
        // passes (the low-rank loader ignores `.diff`/`.diff_b`, and this fold ignores `lora_*`).
        assert!(
            array_eq(
                w.require("blocks.0.self_attn.q.weight").unwrap(),
                base.require("blocks.0.self_attn.q.weight").unwrap(),
                false
            )
            .unwrap()
            .item::<bool>(),
            "q.weight must be untouched — low-rank factors install as post-quantize residuals"
        );
        // ...while its `.diff_b` bias delta DID fold (biases stay dense on every tier).
        assert!(
            !array_eq(
                w.require("blocks.0.self_attn.q.bias").unwrap(),
                base.require("blocks.0.self_attn.q.bias").unwrap(),
                false
            )
            .unwrap()
            .item::<bool>(),
            "q.bias must be patched by its .diff_b"
        );
        // norm_q.weight changed (a diff was applied).
        assert!(
            !array_eq(
                w.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                base.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                false
            )
            .unwrap()
            .item::<bool>(),
            "norm_q.weight must be patched by its .diff"
        );
    }

    /// Turn the synthetic DiT's `q` projection into a PRE-QUANTIZED one: packed u32 `weight` plus the
    /// `scales`/`biases` siblings MLX writes. Mirrors the real q4/q8 tiers, where only the 2-D block
    /// projections are packed and every norm / Linear `bias` stays dense bf16. NOTE `q.biases`
    /// (quantization zero-points) is a different tensor from `q.bias` (the Linear bias).
    fn pack_q(w: &mut Weights) {
        let packed = Array::from_slice(&[0u32; 16 * 2], &[16, 2]);
        let scales = f32(vec![0.01; 16 * 2], &[16, 2]);
        let zeros = f32(vec![0.0; 16 * 2], &[16, 2]);
        w.insert("blocks.0.self_attn.q.weight".to_string(), packed);
        w.insert("blocks.0.self_attn.q.scales".to_string(), scales);
        w.insert("blocks.0.self_attn.q.biases".to_string(), zeros);
    }

    /// sc-18198 — the case the old blanket gate rejected outright. On a pre-quantized tier the fold
    /// must still apply everything it can (the dense norms and biases) and leave the packed
    /// projection for the residual pass, rather than failing the whole file.
    #[test]
    fn packed_projection_defers_low_rank_and_still_folds_dense_targets() {
        let dp = write_diff_patch("packed.safetensors");
        let mut w = synthetic_dit();
        pack_q(&mut w);
        let before = {
            let mut b = synthetic_dit();
            pack_q(&mut b);
            b
        };

        let report = merge_diff_patch_adapters(&mut w, &[&spec(dp, 1.0)]).unwrap();

        // Nothing is blocked: q carries only low-rank factors, which the residual pass will install
        // over the packed weight, and its `.diff_b` targets the still-dense `q.bias`.
        assert!(
            report.blocked_packed_diff.is_empty(),
            "no full-rank .diff lands on a packed target in this file: {:?}",
            report.blocked_packed_diff
        );
        assert_eq!(
            report.deferred_low_rank, 1,
            "q deferred to the residual pass"
        );
        assert_eq!(report.merged_weights, 1, "norm_q .diff still folds");
        assert_eq!(
            report.merged_biases, 1,
            "q.diff_b still folds (bias is dense)"
        );
        report_outcome(&report, "scail2").expect("a packed tier must not fail the file");

        // The packed buffer and its quantization siblings are untouched.
        for k in [
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.scales",
            "blocks.0.self_attn.q.biases",
        ] {
            assert!(
                array_eq(w.require(k).unwrap(), before.require(k).unwrap(), false)
                    .unwrap()
                    .item::<bool>(),
                "{k} must be untouched on a packed tier"
            );
        }
        // ...but the dense targets around it were patched.
        assert!(
            !array_eq(
                w.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                before.require("blocks.0.self_attn.norm_q.weight").unwrap(),
                false
            )
            .unwrap()
            .item::<bool>(),
            "norm_q.weight is dense on every tier and must still be patched"
        );
    }

    /// The one case that genuinely cannot be carried: a full-rank `.diff` on a packed base, with no
    /// low-rank factor to install instead. Refusing is the difference between a loud failure and a
    /// silently half-applied patch, so it must be a hard error even though other targets applied.
    #[test]
    fn full_rank_diff_on_a_packed_target_is_a_hard_error() {
        let path = tmp("packed_diff.safetensors");
        // A `.diff` shaped like the DENSE q weight — on a packed base it cannot be folded.
        let q_diff = f32((0..16 * 8).map(|i| i as f32 * 0.001).collect(), &[16, 8]);
        let norm_diff = f32((0..8).map(|i| i as f32 * 0.01).collect(), &[8]);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.q.diff", &q_diff),
                ("diffusion_model.blocks.0.self_attn.norm_q.diff", &norm_diff),
            ],
            None,
            &path,
        )
        .unwrap();

        let mut w = synthetic_dit();
        pack_q(&mut w);
        let report = merge_diff_patch_adapters(&mut w, &[&spec(path, 1.0)]).unwrap();
        assert_eq!(
            report.blocked_packed_diff,
            vec!["blocks.0.self_attn.q".to_string()]
        );
        let err = report_outcome(&report, "scail2")
            .expect_err("a full-rank .diff on a packed base must not apply silently");
        let msg = format!("{err}");
        assert!(
            msg.contains("pre-quantized on disk") && msg.contains("blocks.0.self_attn.q"),
            "error must name the blocking target: {msg}"
        );
    }

    /// The residual pass is opt-in per file, and the predicate is defined by EXCLUSION so it cannot
    /// silently drop a format it was never told about. A first cut enumerated the LoRA factor
    /// suffixes and thereby excluded LoKr/LoHa — which the SCAIL-2 descriptor advertises via
    /// `supports_lokr` — from the residual pass entirely, with no error. This pins all four shapes.
    #[test]
    fn residual_pass_membership_is_by_exclusion_not_an_allow_list() {
        let hybrid = write_diff_patch("lowrank_hybrid.safetensors");
        assert!(has_residual_installable_keys(&hybrid).unwrap());
        assert!(has_diff_patch_keys(&hybrid).unwrap());

        // Pure `.diff` (+ a bare `.alpha`, a scalar modifier and never a target on its own): nothing
        // for the strict installer to resolve, so it must stay OUT or it reads as "matched nothing".
        let pure_diff = tmp("lowrank_pure_diff.safetensors");
        let d = f32((0..8).map(|i| i as f32 * 0.01).collect(), &[8]);
        let a = f32(vec![4.0], &[1]);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.norm_q.diff", &d),
                ("diffusion_model.blocks.0.self_attn.norm_q.alpha", &a),
            ],
            None,
            &pure_diff,
        )
        .unwrap();
        assert!(has_diff_patch_keys(&pure_diff).unwrap());
        assert!(
            !has_residual_installable_keys(&pure_diff).unwrap(),
            "a pure-.diff file (even with an .alpha) must stay out of the residual pass"
        );

        // PEFT spelling, not just the diffusers/ComfyUI one.
        let peft = tmp("lowrank_peft.safetensors");
        let down = f32(vec![0.1; 4 * 8], &[4, 8]);
        let up = f32(vec![0.1; 16 * 4], &[16, 4]);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &down),
                ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &up),
            ],
            None,
            &peft,
        )
        .unwrap();
        assert!(has_residual_installable_keys(&peft).unwrap());
        assert!(!has_diff_patch_keys(&peft).unwrap());

        // LoKr — the regression this test exists for. Its factor names share no suffix with the LoRA
        // spellings, so any allow-list of `lora_*` keys drops it silently.
        let lokr = tmp("lowrank_lokr.safetensors");
        let w1 = f32(vec![0.1; 4 * 4], &[4, 4]);
        let w2 = f32(vec![0.1; 4 * 2], &[4, 2]);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.q.lokr_w1", &w1),
                ("diffusion_model.blocks.0.self_attn.q.lokr_w2", &w2),
            ],
            None,
            &lokr,
        )
        .unwrap();
        assert!(
            has_residual_installable_keys(&lokr).unwrap(),
            "a LoKr adapter must reach the residual installer (supports_lokr is advertised)"
        );
        assert!(!has_diff_patch_keys(&lokr).unwrap());
    }

    #[test]
    fn scale_zero_is_noop() {
        let dp = write_diff_patch("zero.safetensors");
        let mut w = synthetic_dit();
        let base = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(dp, 0.0)]).unwrap();
        // Still "merged" (folded a zero delta), but every touched weight is bit-identical to the base.
        // One weight, not two: q's low-rank factors are deferred rather than folded (sc-18198), so
        // only `norm_q`'s full-rank `.diff` folds here. q.weight is untouched at ANY strength now,
        // which is why the loop below still holds for it.
        assert_eq!(report.merged_weights, 1);
        assert_eq!(report.deferred_low_rank, 1);
        for k in [
            "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.bias",
            "blocks.0.self_attn.norm_q.weight",
        ] {
            assert!(
                all_close(
                    w.require(k).unwrap(),
                    base.require(k).unwrap(),
                    1e-3,
                    1e-3,
                    false
                )
                .unwrap()
                .item::<bool>(),
                "{k} must be ~unchanged at strength 0"
            );
        }
    }

    #[test]
    fn report_errors_when_nothing_matched() {
        // A diff-patch file whose only target isn't in the checkpoint → matched-nothing error.
        let path = tmp("nomatch.safetensors");
        let diff = f32(vec![0.1; 8], &[8]);
        Array::save_safetensors(
            vec![("diffusion_model.blocks.99.unknown.diff", &diff)],
            None,
            &path,
        )
        .unwrap();
        let mut w = synthetic_dit();
        let report = merge_diff_patch_adapters(&mut w, &[&spec(path, 1.0)]).unwrap();
        assert_eq!(report.merged_weights, 0);
        assert_eq!(
            report.skipped_unmatched,
            vec!["blocks.99.unknown".to_string()]
        );
        assert!(report_outcome(&report, "scail2_14b").is_err());
    }
}
