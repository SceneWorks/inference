//! The load-time merge path: LoRA / LoKr / third-party deltas folded into the base weights.

use super::*;

/// Merge one LoRA file's deltas into the weight map `w` at `spec.scale`, accumulating into `report`.
/// Mirrors `apply_lora_to_linear` per module: `ΔW = (B·A)·(alpha/rank·strength)` at the factor dtype,
/// cast to the weight dtype and added — so the no-LoRA forward and the merged forward share one bf16
/// GEMM (no per-step residual). Multiple files accumulate because each reads the (already-merged)
/// weight back from `w`.
pub(super) fn merge_one(
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
pub(super) fn merge_one_lokr(
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
pub(super) fn wan_module_table(w: &Weights) -> BTreeMap<String, String> {
    w.keys()
        .filter_map(|k| k.strip_suffix(".weight"))
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
}

/// Resolve a third-party LoKr/LoHa raw module key to a Wan checkpoint module path: prefer the
/// flattened-stem table (kohya `lora_unet_…`), else the dotted-path `normalize` (the host's rename
/// map — [`normalize_wan_key`] for native Wan, [`normalize_vace_key`] for the diffusers VACE host —
/// which a dotted diffusers third-party file shares with the peft path).
pub(super) fn resolve_wan_thirdparty(
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
pub(super) fn merge_wan_thirdparty_delta(
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
pub(super) fn merge_one_lokr_thirdparty(
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
pub(super) fn merge_one_loha_thirdparty(
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
pub(super) fn merge_adapters_into(
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
