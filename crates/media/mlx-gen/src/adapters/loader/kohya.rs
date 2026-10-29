//! **kohya / sd-scripts** LoRA (`lora_unet_‹path, .→_›.lora_down/up.weight`), resolved through a
//! flattened → dotted host-path table.
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

// `KOHYA_PREFIX` (`lora_unet_`) is defined in gen-core and re-exported at the top of this module.

/// kohya factor suffixes mapped to a [`LoraParts`] role. `lora_down`==PEFT `lora_A`,
/// `lora_up`==PEFT `lora_B`; the optional `.default` infix is the peft-export form some kohya
/// converters emit. Order is irrelevant (exact-suffix match).
pub(super) const KOHYA_SUFFIXES: [(&str, KohyaRole); 5] = [
    (".lora_down.weight", KohyaRole::Down),
    (".lora_up.weight", KohyaRole::Up),
    (".lora_down.default.weight", KohyaRole::Down),
    (".lora_up.default.weight", KohyaRole::Up),
    (".alpha", KohyaRole::Alpha),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KohyaRole {
    Down,
    Up,
    Alpha,
}

/// The exact kohya suffix normalization used by the loader (including `.default`).
pub fn kohya_factor_key(key: &str) -> Option<(&str, KohyaRole)> {
    let rem = key.strip_prefix(KOHYA_PREFIX)?;
    KOHYA_SUFFIXES
        .iter()
        .find_map(|(suffix, role)| rem.strip_suffix(suffix).map(|stem| (stem, *role)))
}

/// `true` if `w` is a kohya-format LoRA — any key carries the `lora_unet_` prefix. (kohya files are
/// the only convention that flattens the module path; PEFT/diffusers keep dots, LoKr is bare.)
pub fn is_kohya(w: &Weights) -> bool {
    wmeta::keys_are_kohya(w.keys())
}

/// Build the kohya `flattened-stem → dotted-path` lookup from a host's routable target paths
/// (`AdaptableHost::adaptable_paths`). The stem is the dotted path with `.`→`_` (the kohya
/// flattening), WITHOUT the `lora_unet_` prefix. Mirrors the SDXL matcher (sc-2639) and the fork's
/// explicit `lora_unet_…` patterns, generalized over any [`AdaptableHost`].
pub(super) fn kohya_table(paths: &[String]) -> BTreeMap<String, String> {
    wmeta::kohya_table(paths)
}

/// Install a kohya-format LoRA (`lora_unet_<flattened path>.lora_down/up.weight` + optional `.alpha`)
/// onto `host`. The flattened stem is resolved against `table` (built from
/// [`AdaptableHost::adaptable_paths`]) — blind `_`→`.` splitting is impossible because module names
/// contain underscores (`to_out.0`, `feed_forward.w1`, `img_mlp.net.0.proj`). Resolved factors are
/// installed through the same `install_lora_groups` path as PEFT, so a kohya file produces the
/// identical adapter to the equivalent PEFT file.
///
/// `lora_unet_` keys whose stem is NOT in the table (off-surface) are surfaced in `unmatched_paths`
/// so the strict policy fails loudly rather than silently dropping them. The BFL fused→split kohya
/// form (`lora_unet_double_blocks_*`, sc-2743) is routed to [`apply_lora_bfl`] *before* this loader
/// for a host that exposes [`AdaptableHost::bfl_targets`]; reaching here it has no table entry and is
/// likewise surfaced. Keys without the `lora_unet_` prefix (e.g. a bundled text-encoder `lora_te_…`)
/// are not denoiser targets and are ignored, matching the PEFT loader's treatment of out-of-namespace
/// keys.
pub fn apply_lora_kohya(
    host: &mut impl AdaptableHost,
    w: &Weights,
    scale: f32,
    table: &BTreeMap<String, String>,
) -> Result<ApplyReport> {
    let mut groups: BTreeMap<String, LoraParts> = BTreeMap::new();
    let mut claimed = BTreeMap::new();
    let mut unresolved: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let Some((stem, role)) = kohya_factor_key(&key) else {
            continue; // not a recognized denoiser kohya factor (e.g. text-encoder keys).
        };
        let Some(path) = table.get(stem) else {
            unresolved.insert(stem.to_string());
            continue;
        };
        match role {
            KohyaRole::Down => claim_lora_factor(&mut claimed, path, LoraFactorRole::Down, &key)?,
            KohyaRole::Up => claim_lora_factor(&mut claimed, path, LoraFactorRole::Up, &key)?,
            KohyaRole::Alpha => {}
        }
        let parts = groups.entry(path.clone()).or_default();
        match role {
            KohyaRole::Down => parts.a = Some(w.require(&key)?.clone()),
            KohyaRole::Up => parts.b = Some(w.require(&key)?.clone()),
            KohyaRole::Alpha => parts.alpha = scalar_alpha(w.require(&key)?)?,
        }
    }

    // kohya / BFL files carry a per-target `.alpha` tensor, not the `lora_adapter_metadata` blob — no
    // blob to honor here (sc-5513).
    let mut report = install_lora_groups(host, groups, scale, None)?;
    report.unmatched_paths.extend(unresolved);
    Ok(report)
}
