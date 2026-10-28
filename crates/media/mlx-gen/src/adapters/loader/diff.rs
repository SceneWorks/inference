//! Full-rank **diff-patch** adapters (`.diff` / `.diff_b`), folded into the base weights.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

// ---- ComfyUI / lightx2v diff-patch (full-weight/bias deltas), fold-after-build (sc-13825) --------
//
// The ComfyUI "diff patch" mechanism (`comfy/lora.py`): a `‹weight-key›.diff` is a **full-rank weight
// delta** applied `W += strength·diff`, and `‹weight-key›.diff_b` its **bias delta** `b += strength·
// diff_b`. It is the degenerate, already-materialized case of the same additive fold a LoRA/LoKr does,
// with δ handed over raw instead of rebuilt from `B·A` / Kronecker factors, plus a bias channel
// low-rank adapters cannot express. The community Krea "filter-bypass" ships one such tensor,
// `diffusion_model.txtfusion.projector.diff` (`[1, num_text_layers]`), a delta on the 12→1
// `text_fusion.projector` collapse (MLX parity for candle sc-13726).
//
// **Fold-after-build (the MLX seam).** Unlike candle — which folds into the raw weight map before the
// DiT builds — every Krea MLX lane (Resident/Sequential, pose control, the monolithic `KreaPipeline`)
// receives its adapter specs *after* the DiT is constructed, so this folds into the built host's dense
// `AdaptableLinear` bases via [`AdaptableLinear::merge_dense_delta`] / `merge_dense_bias_delta`,
// resolving each stem through the same [`strip_common_lora_prefix`] + host alias
// (`txtfusion`→`text_fusion`, sc-8185) the low-rank pass uses. A `.diff` target must be **dense**: the
// projector (and the other dense front-end projections) stays dense on every tier
// (`TextFusionTransformer::quantize` skips it), so the real bypass folds cleanly on dense, Q4, and Q8.
// A (theoretical) `.diff` aimed at a *quantized* front-end module is surfaced-and-skipped rather than
// unpacked — candle's before-build fold can unpack a packed base, this after-build fold deliberately
// cannot, and says so loudly instead of half-applying.
//
// **Module-coupled shape-aware skip.** A `.diff` whose shape ≠ the base weight (a cross-architecture or
// out-of-surface delta) is skipped as a whole module — its coupled `.diff_b` dropped with it, never a
// half-patch — and surfaced, mirroring ComfyUI's own warn-and-skip and the SCAIL2 diff-patch contract.
//
// ## Two channels that cover every tier, one that does not (sc-15326)
//
// The "a dense delta cannot fold into a packed base" rule above is true only of a **weight** delta into
// a **Linear**. Measured against the real lightx2v `Wan2.1-T2V-14B` cfg-step-distill file (647
// diff-patch keys), taking that rule as the whole story would have folded **7 of 647** on the default Q4
// tier and **407 of 647** on bf16 — the same LoRA rendering differently per tier, on a product where the
// quant tier is a *creative* choice. Two of the three channels in fact fold on **every** tier
// (tier-independent in *coverage* — see the per-channel dtype note below), and this fold uses them:
//
// * **`.diff_b` into a Linear (407 keys).** `QuantizedLinear` packs only the weight; the bias stays a
//   plain dense vector and the forward is `quantized_matmul(…) + b`. So `b += scale·δ_b` folds on Q4,
//   Q8 and bf16 alike — see [`AdaptableLinear::merge_bias_delta`]. Tier-independent in *coverage*:
//   every such key lands on every tier, though the fold dtype still follows the base (a quantize casts
//   the bias to bf16, so an f32-on-disk base folds in f32 dense and bf16 packed).
// * **`.diff` / `.diff_b` into a norm (240 keys).** RMSNorm/LayerNorm parameters are never packed by any
//   family's quantize predicate, so they are dense on every tier. They are not `AdaptableLinear`s at any
//   surface width, so they route through the separate [`AdaptableHost::diff_patch_param_mut`] surface.
// * **`.diff` into a Linear (0 keys in that file).** The only genuinely tier-dependent case, and the
//   only one still skipped-and-surfaced over a packed base.
//
// Result for that file: **647 of 647 on every tier**, with no tier divergence at all.

/// One module's diff-patch deltas: a full-rank weight delta and/or a bias delta.
#[derive(Default)]
pub(super) struct DiffParts {
    pub(super) diff: Option<Array>,   // weight delta, base-shaped
    pub(super) diff_b: Option<Array>, // bias delta, base-shaped
}

/// What a diff-patch fold did: the number of weight/bias deltas folded, and the targets deliberately
/// skipped (shape-incompatible / packed-weight `.diff` / no base bias — coupled parts dropped, never
/// half-applied) or resolved to no module. Surfaced loudly by the caller **and** carried out on
/// [`ApplyReport`] so a provider can tell the user exactly what did not land; a file that folded
/// **nothing** is caught by the combined zero-match guard, not here.
#[derive(Debug, Default)]
pub struct DiffPatchReport {
    /// Count of weight + bias deltas folded (into a dense Linear weight, into any Linear's dense bias
    /// — packed or not — or into a host norm parameter).
    pub folded: usize,
    /// Targets whose weight/bias delta could not fold (shape mismatch, a weight `.diff` aimed at a
    /// packed Linear, or a `.diff_b` with no base bias) — coupled deltas dropped together, never
    /// half-applied.
    pub skipped: Vec<String>,
    /// Diff-patch stems that resolved to no adaptable module on the host.
    pub unmatched: Vec<String>,
    /// Deltas that resolved to a host target and shape-checked clean but were **deliberately not
    /// written**, because their spec's `scale` is 0 (sc-15265). Counted separately from
    /// [`folded`](Self::folded) so the host's arrays stay byte-identical to a never-installed adapter
    /// while the zero-match guard still sees that the file's targets *did* resolve.
    pub disabled: usize,
}

impl DiffPatchReport {
    /// Record one delta that resolved and shape-checked: [`folded`](Self::folded) when the spec is
    /// live, [`disabled`](Self::disabled) when its scale is 0 and nothing was written.
    pub(super) fn record(&mut self, live: bool) {
        if live {
            self.folded += 1;
        } else {
            self.disabled += 1;
        }
    }
}

/// `true` if any of `keys` is a diff-patch delta name (`.diff` / `.diff_b`) — the structural marker of
/// a ComfyUI/lightx2v diff-patch file, which the forward-time residual loader cannot consume. The
/// header-only twin of [`has_diff_patch_keys`] (which needs a fully-loaded [`Weights`]): a caller with
/// just the safetensors key list (e.g. from a [`gen_core::weightsmeta::CheckpointMeta`]) can detect a
/// diff-patch adapter WITHOUT loading its tensors — used by the Krea multi-phase guard (sc-13884) to
/// reject a diff-patch adapter, whose delta folds irreversibly into the dense base and so cannot be
/// toggled off per phase.
pub fn has_diff_patch_key_names<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.ends_with(".diff") || k.ends_with(".diff_b"))
}

/// `true` if any tensor key in `w` is a diff-patch delta (`.diff` / `.diff_b`) — the structural marker
/// of a ComfyUI/lightx2v diff-patch file, which the forward-time residual loader cannot consume.
pub fn has_diff_patch_keys(w: &Weights) -> bool {
    has_diff_patch_key_names(w.keys())
}

/// Fold every ComfyUI/lightx2v **diff-patch** (`.diff` weight / `.diff_b` bias) delta the `specs` carry
/// into `host`'s matched **dense** `AdaptableLinear` bases at each spec's `scale`: `‹stem›.diff` →
/// `W += scale·δ`, `‹stem›.diff_b` → `b += scale·δ_b`. Each stem resolves through the same
/// `strip_common_lora_prefix` + host module alias the low-rank pass uses, so
/// `diffusion_model.txtfusion.projector` reaches `text_fusion.projector`. Non-diff keys (`lora_*` /
/// `lokr_*`) are ignored here — they ride the disjoint forward-time residual pass. A no-op for specs
/// carrying no `.diff`/`.diff_b`. See the module note above for the fold-after-build + coupled-skip
/// contract.
pub fn fold_diff_patch_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<DiffPatchReport> {
    let mut combined = DiffPatchReport::default();
    for report in fold_diff_patch_adapters_reported(host, specs)? {
        combined.folded += report.folded;
        combined.disabled += report.disabled;
        combined.skipped.extend(report.skipped);
        combined.unmatched.extend(report.unmatched);
    }
    Ok(combined)
}

/// Per-spec form of [`fold_diff_patch_adapters`]. All diff-patch specs are still folded in one
/// ordered pass before the low-rank pass; retaining the individual reports does not alter mutation
/// order.
pub(super) fn fold_diff_patch_adapters_reported(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<Vec<DiffPatchReport>> {
    let mut reports = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut report = DiffPatchReport::default();
        // iOS: header-only floor of the adapter file before it is eagerly read (no-op elsewhere).
        crate::ios_quantum::before_load(ADAPTER_LOAD_QUANTUM, std::slice::from_ref(&spec.path))?;
        let w = Weights::from_file(&spec.path)?;
        // Group each module's `.diff`/`.diff_b` by its resolved (prefix-stripped) dotted stem.
        let mut groups: BTreeMap<String, DiffParts> = BTreeMap::new();
        for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
            if let Some(stem) = key.strip_suffix(".diff_b") {
                groups
                    .entry(strip_common_lora_prefix(stem).to_string())
                    .or_default()
                    .diff_b = Some(w.require(&key)?.clone());
            } else if let Some(stem) = key.strip_suffix(".diff") {
                groups
                    .entry(strip_common_lora_prefix(stem).to_string())
                    .or_default()
                    .diff = Some(w.require(&key)?.clone());
            }
        }
        for (stem, parts) in groups {
            fold_one_diff_module(host, &stem, &parts, spec.scale, &mut report)?;
        }
        // Materialize the factors this file contributed at load (sc-24245): left lazy, the first
        // forward's command buffers wait on the safetensors reads — past the GPU watchdog on a
        // cold page cache (see `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        reports.push(report);
    }
    Ok(reports)
}

/// Scale a delta by `scale` in f32 (the fold math runs in f32; `merge_dense_delta`/`_bias_delta` cast
/// back to the base dtype on write).
pub(super) fn scaled_f32(delta: &Array, scale: f32) -> Result<Array> {
    Ok(delta
        .as_dtype(Dtype::Float32)?
        .multiply(crate::array::scalar(scale))?)
}

/// Record a whole module as skipped — the weight delta and, when present, its coupled bias delta
/// (never a half-patch).
pub(super) fn skip_whole_module(stem: &str, parts: &DiffParts, report: &mut DiffPatchReport) {
    report.skipped.push(stem.to_string());
    if parts.diff_b.is_some() {
        report.skipped.push(format!("{stem}.bias"));
    }
}

/// Fold one module's `.diff`/`.diff_b` into the host's base at that dotted stem.
///
/// Routes to whichever of the two host surfaces owns the stem: an [`AdaptableLinear`]
/// ([`AdaptableHost::adaptable_mut`]) first, then a dense non-Linear parameter — a norm
/// ([`AdaptableHost::diff_patch_param_mut`], sc-15326). An unresolved path on both is recorded as
/// unmatched.
///
/// On a Linear: a weight-delta shape ≠ the base, or a weight delta aimed at a **packed** base (this
/// after-build fold does not unpack), skips the **whole** module — its coupled bias dropped too, never a
/// half-patch. The bias delta itself is *not* tier-gated: a quantized Linear keeps its bias dense, so
/// `.diff_b` folds identically on Q4/Q8/bf16 (sc-15326). A `.diff_b` with no shape-matching base bias is
/// surfaced but does not undo the weight fold.
pub(super) fn fold_one_diff_module(
    host: &mut impl AdaptableHost,
    stem: &str,
    parts: &DiffParts,
    scale: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    // sc-15265: a scale-0 install must be byte-identical to no install at all. Resolve and shape-check
    // exactly as usual — so `skipped`/`unmatched` stay truthful and the zero-match guard still sees the
    // targets — but write nothing. (`δ · 0.0` is not reliably inert: it flips `-0.0` to `+0.0`, and
    // `Inf · 0.0` is NaN, which would poison the host at precisely the setting that turns the adapter
    // OFF.)
    let live = scale != 0.0;
    let segs: Vec<&str> = stem.split('.').collect();
    // SC-18319 — every *decision* this function makes (does the path resolve to a Linear at all, is
    // the base packed, do the weight and bias shapes match) is answerable from the PROBE half, so the
    // `&mut` handle is taken only on the branches that actually write. That matters twice over here:
    // the quantized/shape-mismatch arm SKIPS, and a `scale == 0` install writes nothing at all — under
    // the old `&mut`-first resolution both of those still unfused every `FusedQkvProjection` they
    // touched, in exchange for no mutation whatsoever.
    let Some(facts) = host.adaptable_facts(&segs) else {
        return fold_one_diff_param(host, stem, &segs, parts, scale, report);
    };

    if let Some(diff) = &parts.diff {
        // A weight `.diff` is the one genuinely tier-dependent channel: a dense delta cannot fold into
        // a packed weight, and unpacking to make it fit would defeat the tier the user chose. Skip the
        // whole module (coupled bias with it) and surface it.
        if facts.is_quantized || diff.shape() != facts.base_shape.as_slice() {
            skip_whole_module(stem, parts, report);
            return Ok(());
        }
    }
    // Bias delta (`.diff_b`): folds onto `{stem}.bias` when the base carries a shape-matching bias —
    // packed or not, since `QuantizedLinear` never packs its bias.
    let bias_ok = parts
        .diff_b
        .as_ref()
        .is_some_and(|d| facts.bias_shape.as_deref() == Some(d.shape()));

    if let Some(diff) = &parts.diff {
        if live {
            host.adaptable_mut(&segs)
                .expect("resolved through the probe above")
                .merge_dense_delta(&scaled_f32(diff, scale)?)?;
        }
        report.record(live);
    }

    if let Some(diff_b) = &parts.diff_b {
        if bias_ok {
            if live {
                host.adaptable_mut(&segs)
                    .expect("resolved through the probe above")
                    .merge_bias_delta(&scaled_f32(diff_b, scale)?)?;
            }
            report.record(live);
        } else {
            report.skipped.push(format!("{stem}.bias"));
        }
    }
    Ok(())
}

/// Fold one module's deltas through the host's **non-Linear dense parameter** surface — the norms
/// (sc-15326). Reached only when the stem resolves to no [`AdaptableLinear`].
///
/// Unlike the Linear path this is tier-independent by construction: every parameter reachable through
/// [`AdaptableHost::diff_patch_param_mut`] is dense on every quant tier (norm weights/biases are never
/// in a family's quantize predicate), so `p += scale·δ` is the same arithmetic at Q4, Q8 and bf16.
///
/// The two failure modes are split exactly as on the Linear path (sc-15326):
///
/// * A **shape mismatch** on either half skips the **whole** module — a delta that happens to fit one
///   half of some other architecture's module is not a delta for this one, and a half-patched norm is
///   worse than an unpatched one.
/// * A half the **host does not expose** skips only *that* half. Every qk-RMSNorm is weight-only, so a
///   lightning/step-distill file carrying `norm_q.diff_b` next to a perfectly valid `norm_q.diff` still
///   lands the weight delta and surfaces just the bias — mirroring the Linear path, where a `.diff_b`
///   with no shape-matching base bias is surfaced but does not undo the weight fold.
///
/// A stem with neither half present on the host is unmatched.
pub(super) fn fold_one_diff_param(
    host: &mut impl AdaptableHost,
    stem: &str,
    segs: &[&str],
    parts: &DiffParts,
    scale: f32,
    report: &mut DiffPatchReport,
) -> Result<()> {
    // A module the host exposes at all — probed on whichever half the file actually carries, so a
    // weight-only norm (RMSNorm) and a weight+bias norm (affine LayerNorm) both resolve.
    let known = host
        .diff_patch_param_mut(segs, DiffPatchPart::Weight)
        .is_some()
        || host
            .diff_patch_param_mut(segs, DiffPatchPart::Bias)
            .is_some();
    if !known {
        report.unmatched.push(stem.to_string());
        return Ok(());
    }

    let halves = [
        (DiffPatchPart::Weight, parts.diff.as_ref(), stem.to_string()),
        (
            DiffPatchPart::Bias,
            parts.diff_b.as_ref(),
            format!("{stem}.bias"),
        ),
    ];

    // Pre-check both halves BEFORE mutating either, so a shape-mismatched module is skipped whole
    // rather than half-patched — while a half the host simply does not expose only takes itself out.
    let mut absent = [false; 2];
    for (i, (part, delta, _)) in halves.iter().enumerate() {
        let Some(delta) = delta else { continue };
        match host.diff_patch_param_mut(segs, *part) {
            Some(p) if p.shape() == delta.shape() => {}
            // Present but the wrong shape: not a delta for this module at all.
            Some(_) => {
                skip_whole_module(stem, parts, report);
                return Ok(());
            }
            // Not a channel this module has (a `.diff_b` on a weight-only RMSNorm). Surfaced below;
            // the other half still folds.
            None => absent[i] = true,
        }
    }

    for (i, (part, delta, label)) in halves.iter().enumerate() {
        let Some(delta) = delta else { continue };
        if absent[i] {
            report.skipped.push(label.clone());
            continue;
        }
        // Scale 0 (sc-15265): resolved and shape-checked, deliberately not written.
        if scale != 0.0 {
            let scaled = scaled_f32(delta, scale)?;
            let p = host
                .diff_patch_param_mut(segs, *part)
                .expect("shape-checked");
            *p = add(&*p, &scaled.as_dtype(p.dtype())?)?;
        }
        report.record(scale != 0.0);
    }
    Ok(())
}
