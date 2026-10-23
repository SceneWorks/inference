//! Exact witnesses for order-dependent fused UNet/VAE source aliases.
//!
//! Groups are collected per forward module (a *supergroup*, whose assignments are the product of
//! its groups'). A supergroup combo is *confined* when, against a reference world, the exact
//! pricer's module-output trace ([`q::with_module_trace`]) is unchanged with an equal ledger
//! length, or ends inside the combo's own module (a native shape failure there, or a load/pack
//! failure before any forward). A confined combo then differs from the reference only by its
//! additive loaded bytes, its own module's windows, and where its ledger stops; a stopped ledger
//! is a prefix of the reference's. Every other combo's supergroup, and every group outside one
//! traced module, is enumerated.
//!
//! For a window at prefix position i (in module M, or in none) and any world w: the witness for
//! (M's combo in w, threshold L) keeps every other confined supergroup at its maximum loaded
//! bytes among combos whose ledger reaches L, the smallest stop length >= i. All of w's other
//! combos reach i, hence L; so the witness's exact ledger has w's trajectory, M's windows and at
//! least w's bytes at i. A promoting dtype alias changes a downstream module output, so it is
//! enumerated rather than assumed absent. Classification runs inside one text world, because a
//! module output joins the text context's dtype.
use std::collections::BTreeMap;

use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
use mlx_gen::gen_core::LoadSpec;
use mlx_gen::ios_pricing::IosPhase;
use mlx_gen::Result;

use super::text_alias::Picks;
use super::{fused_header_footprint, Inputs, Regime};
use crate::quantum as q;

/// Heavy witness picks: `resident` maximizes the additive byte channel at every stop threshold
/// (what text-phase windows read); `all` adds every module-local combo.
pub(super) struct HeavyWitnesses {
    pub resident: Vec<Picks>,
    pub all: Vec<Picks>,
}

struct Supergroup {
    module: String,
    combos: Vec<Picks>,
    bytes: Vec<u64>,
}

/// One priced world: its window count, whether any forward phase ran, and its module trace.
struct Run {
    windows: usize,
    forward: bool,
    trace: Vec<q::ModuleOutput>,
}

fn index(rest: &str) -> Option<(u32, &str)> {
    let (number, rest) = rest.split_once('.')?;
    Some((number.parse().ok()?, rest))
}

/// The traced forward module owning a component target, named as [`q::trace_module`] records it.
fn module_of(component: C, name: &str) -> Option<String> {
    let block = |prefix: &str| -> Option<String> {
        let (i, rest) = index(name.strip_prefix(prefix)?)?;
        for kind in ["resnets.", "attentions."] {
            if let Some(rest) = rest.strip_prefix(kind) {
                let (j, _) = index(rest)?;
                return Some(format!("{prefix}{i}.{kind}{j}"));
            }
        }
        ["downsamplers.", "upsamplers."]
            .iter()
            .find(|kind| rest.starts_with(**kind))
            .map(|kind| format!("{prefix}{i}.{}", kind.trim_end_matches('.')))
    };
    let mid = |prefix: &str| -> Option<String> {
        ["resnets.0.", "attentions.0.", "resnets.1."]
            .iter()
            .find(|m| name.starts_with(&format!("{prefix}{m}")))
            .map(|m| format!("{prefix}{}", m.trim_end_matches('.')))
    };
    let starts = |names: &[&str]| names.iter().any(|n| name.starts_with(n));
    match component {
        C::Unet => {
            if starts(&[
                "conv_in.",
                "time_embedding.",
                "add_embedding.",
                "encoder_hid_proj.",
            ]) {
                Some("stem".into())
            } else if starts(&["conv_norm_out.", "conv_out."]) {
                Some("out".into())
            } else {
                block("down_blocks.")
                    .or_else(|| block("up_blocks."))
                    .or_else(|| mid("mid_block."))
            }
        }
        C::Vae => {
            if starts(&["decoder.conv_in.", "post_quant_conv."]) {
                Some("decoder.in".into())
            } else if starts(&["decoder.conv_norm_out.", "decoder.conv_out."]) {
                Some("decoder.out".into())
            } else if starts(&["encoder.conv_in."]) {
                Some("encoder.in".into())
            } else if starts(&["encoder.conv_norm_out.", "encoder.conv_out.", "quant_conv."]) {
                Some("encoder.out".into())
            } else {
                block("decoder.up_blocks.")
                    .or_else(|| block("encoder.down_blocks."))
                    .or_else(|| mid("decoder.mid_block."))
                    .or_else(|| mid("encoder.mid_block."))
            }
        }
        _ => None,
    }
}

/// Every pick list formed by one entry of each list, in list order.
pub(super) fn product(lists: &[Vec<(usize, usize)>]) -> Vec<Picks> {
    let mut result = vec![Vec::new()];
    for list in lists {
        result = result
            .into_iter()
            .flat_map(|picks: Picks| {
                list.iter().map(move |pick| {
                    let mut picks = picks.clone();
                    picks.push(*pick);
                    picks
                })
            })
            .collect();
    }
    result
}

/// Every concatenation of one context with one combo.
fn extend_all(contexts: Vec<Picks>, combos: &[Picks]) -> Vec<Picks> {
    contexts
        .iter()
        .flat_map(|context| {
            combos
                .iter()
                .map(move |combo| context.iter().chain(combo).copied().collect::<Picks>())
        })
        .collect()
}

fn run(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
    world: Picks,
) -> Result<Run> {
    inputs
        .fused
        .as_mut()
        .expect("heavy aliases are fused")
        .world = world;
    let (pricing, trace) = q::with_module_trace(|| super::price_world(inputs, regime, configured));
    let pricing = pricing?;
    Ok(Run {
        windows: pricing.windows.len(),
        forward: pricing.windows.iter().any(|w| {
            matches!(
                w.phase,
                IosPhase::Encode | IosPhase::Denoise | IosPhase::Decode
            )
        }),
        trace,
    })
}

/// Whether a combo of `module` stays confined against the reference run.
fn confined(module: &str, combo: &Run, reference: &Run) -> bool {
    if combo.trace == reference.trace {
        return combo.windows == reference.windows;
    }
    let k = combo.trace.len();
    combo.windows < reference.windows
        && k < reference.trace.len()
        && combo.trace[..] == reference.trace[..k]
        && (reference.trace[k].0 == module || (k == 0 && !combo.forward))
}

fn loaded_bytes(inputs: &Inputs, component: C, picks: &Picks) -> Result<u64> {
    let source = &inputs
        .fused
        .as_ref()
        .expect("heavy aliases are fused")
        .source;
    let headers = source.headers_with(component, picks);
    Ok(match component {
        C::Vae => fused_header_footprint(&headers, inputs.vae.1, true)?.resident,
        _ => fused_header_footprint(&headers, crate::model::DTYPE, false)?.resident,
    })
}

/// Pack-walk projection paths. A projection alias changes shape-only pack outputs and pack
/// scratch, so its bytes are not one regime-independent channel; it is enumerated instead.
fn packed_paths(inputs: &Inputs) -> Result<Vec<String>> {
    let Some(bits) = inputs.quant_bits else {
        return Ok(Vec::new());
    };
    let source = &inputs
        .fused
        .as_ref()
        .expect("heavy aliases are fused")
        .source;
    let mut unet = crate::unet::pricing::UNetFacts::from_headers(
        &source.headers_with(C::Unet, &[]),
        &inputs.arch.unet,
    )?;
    let mut paths = Vec::new();
    unet.quantize_using(bits, &mut |path, _, _| {
        paths.push(format!("{path}."));
        Ok(())
    })?;
    Ok(paths)
}

/// Heavy witnesses within one text world `text` (its picks are carried by every priced world):
/// confinement is decided against that text outcome, since module outputs join context dtypes.
pub(super) fn heavy_witnesses(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&LoadSpec>,
    text: &Picks,
) -> Result<HeavyWitnesses> {
    let empty = || HeavyWitnesses {
        resident: vec![Vec::new()],
        all: vec![Vec::new()],
    };
    let Some(fused) = inputs.fused.as_ref() else {
        return Ok(empty());
    };
    let heavy = (0..fused.source.aliases.len())
        .filter(|&g| matches!(fused.source.aliases[g].component, C::Unet | C::Vae))
        .collect::<Vec<_>>();
    if heavy.is_empty() {
        return Ok(empty());
    }
    let packed = packed_paths(inputs)?;
    let source = &inputs
        .fused
        .as_ref()
        .expect("heavy aliases are fused")
        .source;
    let assignments = |g: usize| -> Vec<(usize, usize)> {
        (0..source.aliases[g].assignments.len())
            .map(|a| (g, a))
            .collect()
    };
    let mut fixed = Vec::new();
    let mut modules: BTreeMap<(String, bool), Vec<usize>> = BTreeMap::new();
    for &group in &heavy {
        let alias = &source.aliases[group];
        let found = alias
            .targets
            .iter()
            .map(|target| module_of(alias.component, target))
            .collect::<Option<Vec<_>>>();
        let projection = alias.component == C::Unet
            && alias
                .targets
                .iter()
                .any(|target| packed.iter().any(|path| target.starts_with(path)));
        match found {
            Some(found) if !projection && found.iter().all(|m| *m == found[0]) => modules
                .entry((found[0].clone(), alias.component == C::Vae))
                .or_default()
                .push(group),
            _ => fixed.push(assignments(group)),
        }
    }
    let mut supers = Vec::new();
    for ((module, vae), groups) in modules {
        let combos = product(&groups.iter().map(|&g| assignments(g)).collect::<Vec<_>>());
        let component = if vae { C::Vae } else { C::Unet };
        let bytes = combos
            .iter()
            .map(|picks| loaded_bytes(inputs, component, picks))
            .collect::<Result<Vec<_>>>()?;
        supers.push(Supergroup {
            module,
            combos,
            bytes,
        });
    }

    // Enumerate any supergroup that is not confined in some enumerated context, then restart.
    let mut enumerated = vec![false; supers.len()];
    let (mut resident, mut all) = 'classify: loop {
        let mut contexts = extend_all(vec![text.clone()], &product(&fixed));
        for (s, group) in supers.iter().enumerate() {
            if enumerated[s] {
                contexts = extend_all(contexts, &group.combos);
            }
        }
        let local = (0..supers.len())
            .filter(|&s| !enumerated[s])
            .collect::<Vec<_>>();
        let (mut resident, mut all) = (Vec::new(), Vec::new());
        for context in contexts {
            let world = |refs: &[usize], replace: Option<(usize, usize)>| -> Picks {
                let mut picks = context.clone();
                for &s in &local {
                    let c = match replace {
                        Some((r, c)) if r == s => c,
                        _ => refs[s],
                    };
                    picks.extend(supers[s].combos[c].iter().copied());
                }
                picks
            };
            // Reference: each local supergroup at its longest-ledger combo. Iterated because a
            // failure is observable only when every earlier module reaches it.
            let mut refs = vec![0usize; supers.len()];
            let runs = loop {
                let mut runs: Vec<Vec<Run>> = (0..supers.len()).map(|_| Vec::new()).collect();
                for &s in &local {
                    for c in 0..supers[s].combos.len() {
                        runs[s].push(run(inputs, regime, configured, world(&refs, Some((s, c))))?);
                    }
                }
                let mut changed = false;
                for &s in &local {
                    let longest = (0..runs[s].len())
                        .max_by_key(|&c| runs[s][c].windows)
                        .expect("a supergroup has a combo");
                    if runs[s][longest].windows > runs[s][refs[s]].windows {
                        refs[s] = longest;
                        changed = true;
                    }
                }
                if !changed {
                    break runs;
                }
            };
            let reference = run(inputs, regime, configured, world(&refs, None))?;
            for &s in &local {
                if runs[s]
                    .iter()
                    .any(|combo| !confined(&supers[s].module, combo, &reference))
                {
                    enumerated[s] = true;
                    continue 'classify;
                }
            }
            let mut thresholds = local
                .iter()
                .flat_map(|&s| runs[s].iter().map(|r| r.windows))
                .chain([reference.windows])
                .collect::<Vec<_>>();
            thresholds.sort_unstable();
            thresholds.dedup();
            for threshold in thresholds {
                let best = |s: usize| {
                    (0..supers[s].combos.len())
                        .filter(|&c| runs[s][c].windows >= threshold)
                        .max_by_key(|&c| supers[s].bytes[c])
                };
                if local.iter().any(|&s| best(s).is_none()) {
                    continue;
                }
                let with = |m: Option<(usize, &Picks)>| -> Picks {
                    let mut picks = context.clone();
                    for &s in &local {
                        match m {
                            Some((local_s, combo)) if local_s == s => {
                                picks.extend(combo.iter().copied())
                            }
                            _ => picks.extend(
                                supers[s].combos[best(s).expect("reachable")]
                                    .iter()
                                    .copied(),
                            ),
                        }
                    }
                    picks
                };
                resident.push(with(None));
                all.push(with(None));
                for &m in &local {
                    for combo in &supers[m].combos {
                        all.push(with(Some((m, combo))));
                    }
                }
            }
        }
        break (resident, all);
    };
    inputs
        .fused
        .as_mut()
        .expect("heavy aliases are fused")
        .world = Vec::new();
    for list in [&mut resident, &mut all] {
        for world in list.iter_mut() {
            world.sort_unstable();
        }
        list.sort();
        list.dedup();
    }
    Ok(HeavyWitnesses { resident, all })
}
