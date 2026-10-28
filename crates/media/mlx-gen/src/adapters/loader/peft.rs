//! **PEFT / diffusers** LoRA (`‹prefix›‹path›.lora_A/B.weight`) and the shared LoRA group install.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

/// Install a PEFT/diffusers-format LoRA file onto `host`. The down/up factors carry the file's
/// namespace prefix on a **dotted** module path, in either of two interchangeable spellings:
/// - PEFT: `‹prefix›‹path›.lora_A.weight` / `.lora_B.weight`;
/// - diffusers/ComfyUI/ai-toolkit (e.g. the lightx2v Qwen-Image-Lightning LoRAs, sc-2909):
///   `‹prefix›‹path›.lora_down.weight` / `.lora_up.weight` — `lora_down`==`lora_A`, `lora_up`==`lora_B`
///   (identical shapes), differing from the kohya format only in that the path stays dotted (no
///   `lora_unet_` flattening), so it routes here rather than to [`apply_lora_kohya`].
///
/// Both store the down factor as `[r, in]` and the up factor as `[out, r]`; we transpose to the
/// residual form `x·A·B` (`A: [in, r]`, `B: [r, out]`) and fold `alpha/rank` into `B`, matching the
/// fork. `‹prefix›‹path›.alpha` is optional (and may be bare — see below). `strip_prefix` removes a
/// leading namespace such as `"base_model.model."` or `"transformer."`.
pub fn apply_lora_peft(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    strip_prefix: Option<&str>,
) -> Result<ApplyReport> {
    let prefix = strip_prefix.unwrap_or("");
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut claimed = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        // The down/up factors always carry the file's namespace prefix. `lora_A`/`lora_B` (PEFT) and
        // `lora_down`/`lora_up` (diffusers/ComfyUI) are interchangeable spellings of the same role,
        // and a PEFT export that kept its adapter name writes `lora_A.default` (sc-24158 — the
        // candle `install_dotted_adapters` and the kohya loader below already accept it).
        if let Some((path, role)) = peft_lora_factor(&key, prefix) {
            claim_lora_factor(&mut claimed, path, role, &key)?;
            let parts = groups.entry(path.to_string()).or_default();
            let slot = match role {
                LoraFactorRole::Down => &mut parts.a,
                LoraFactorRole::Up => &mut parts.b,
            };
            *slot = Some(w.require(&key)?.clone());
            continue;
        }
        // `alpha` may be prefixed (`<prefix><path>.alpha`) OR bare (`<path>.alpha`): some trainers
        // pair prefixed `lora_A/B` with a bare `alpha` — notably the fork's `QwenLoRAMapping`, whose
        // alpha patterns are bare-only. Resolve to the same `<path>` either way (rather than
        // stripping the A/B prefix off the alpha key and dropping a bare one) so the `alpha/rank`
        // fold is kept; a prefixed and a bare alpha that *disagree* for one path is a hard error (no
        // silent pick). Without this, a prefixed-A/B + bare-alpha file applied at the wrong
        // (unscaled) strength while reporting success (sc-2528 adversarial review).
        if let Some(path) = key
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(".alpha"))
            .or_else(|| key.strip_suffix(".alpha"))
        {
            if let Some(new) = scalar_alpha(w.require(&key)?)? {
                let slot = &mut groups.entry(path.to_string()).or_default().alpha;
                match *slot {
                    Some(existing) if existing != new => {
                        return Err(format!(
                            "LoRA alpha conflict for `{path}`: {existing} vs {new} \
                             (prefixed and bare alpha keys disagree)"
                        )
                        .into());
                    }
                    _ => *slot = Some(new),
                }
            }
        }
    }

    // PEFT/diffusers `save_lora_adapter` files carry no per-target `.alpha` tensor — `lora_alpha`/`r`
    // (+ per-module overrides) live in the `lora_adapter_metadata` header blob (sc-5513). `None` for a
    // file without the blob, in which case the per-target `.alpha` or the factor rank is used exactly
    // as before. (kohya / BFL loaders ship a `.alpha` tensor and pass `None` here.)
    let cfg = wmeta::LoraAdapterMeta::from_metadata(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY));
    install_lora_groups(host, groups, scale, cfg.as_ref())
}

/// Install grouped `(down=A, up=B, alpha)` LoRA factors onto `host`, one residual per resolved module
/// path. Shared by the PEFT/diffusers loader ([`apply_lora_peft`]) and the kohya loader
/// ([`apply_lora_kohya`]): both conventions agree on the math (`A: [r,in]`, `B: [out,r]`, transpose to
/// the residual form `x·A·B`, fold `alpha/rank` into `B`) and differ only in how keys map to `path`.
/// A path with a missing `down` or `up` half is skipped (its partner targeted a non-LoRA key);
/// a path that resolves to no module is surfaced in `unmatched_paths`, never silently dropped.
pub(super) fn install_lora_groups(
    host: &mut impl AdaptableHost,
    groups: BTreeMap<String, LoraParts>,
    scale: f32,
    meta: Option<&wmeta::LoraAdapterMeta>,
) -> Result<ApplyReport> {
    let mut report = ApplyReport::default();
    for (path, parts) in groups {
        let (Some(a_raw), Some(b_raw)) = (parts.a, parts.b) else {
            continue;
        };
        let parents: Vec<&str> = path.split('.').collect();
        match host.adaptable_mut(&parents) {
            Some(lin) => {
                let a = a_raw.t(); // [r, in] -> [in, r]
                let mut b = b_raw.t(); // [out, r] -> [r, out]
                                       // A Linear LoRA's factors are 2-D; a malformed 1-D/scalar factor would panic on the
                                       // `a.shape()[1]` rank read below. Reject it with a typed error up front (F-034).
                validate_lora_shapes(&path, a_raw.shape(), b_raw.shape())?;
                // Effective scaling. Precedence: per-target `.alpha` tensor (kohya / SceneWorks
                // trainer / BFL) → the PEFT/diffusers `lora_adapter_metadata` blob's
                // `alpha_pattern`/`lora_alpha` (sc-5513 — that format ships NO `.alpha` tensor) → no
                // fold (the pre-existing `alpha == rank ⇒ scale 1.0` default). The denominator honors
                // the blob `r`/`rank_pattern` when given (always `> 0`), else the factor's stored
                // leading dim (which equals it for a well-formed PEFT file).
                let (cfg_alpha, cfg_rank) = meta.map_or((None, None), |m| m.effective(&path));
                if let Some(alpha) = parts.alpha.or(cfg_alpha) {
                    let factor_rank = a.shape()[1] as f32; // r
                    validate_lora_alpha_rank(&path, a.shape()[1])?;
                    let rank = cfg_rank.unwrap_or(factor_rank);
                    b = b.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
                }
                lin.push(Adapter::Lora { a, b, scale });
                report.applied += 1;
            }
            None => report.unmatched_paths.push(path),
        }
    }
    Ok(report)
}

#[derive(Default)]
pub(super) struct LoraParts {
    pub(super) a: Option<Array>,
    pub(super) b: Option<Array>,
    pub(super) alpha: Option<f32>,
}
