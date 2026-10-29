//! The forward-time additive path: adapters installed as residuals on the (packed) host.

use super::*;

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
pub(super) fn apply_additive_into(
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
pub(super) fn install_one_additive(
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
pub(super) fn push_at(
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
pub(super) fn install_one_lora_additive(
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
pub(super) fn install_one_lokr_additive(
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
pub(super) fn non_deferrable_lokr_error(path: &str) -> mlx_gen::Error {
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
pub(super) fn loha_on_packed_target_error(path: &str) -> mlx_gen::Error {
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
pub(super) fn install_one_lokr_thirdparty_additive(
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
pub(super) fn install_one_loha_thirdparty_additive(
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
pub(super) fn install_thirdparty_delta(
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
pub(super) fn host_module_table(host: &impl AdaptableHost) -> BTreeMap<String, String> {
    host.adaptable_paths()
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect()
}
