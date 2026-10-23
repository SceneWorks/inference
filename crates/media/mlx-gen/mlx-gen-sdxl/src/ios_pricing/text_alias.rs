//! Exact text-alias witnesses, including worlds whose native execution ends at a shape check.
//!
//! Independent groups contribute additive loaded/retained owners. Layer-local products drive a
//! finite (dtype, width, penultimate dtype) program. A failing local transition contributes its
//! admitted guard but has no successor. Packing uses the same rule: assignments failing at point
//! p are eligible through p, never after p. Symbolic accounting terms after an invalid pack are
//! used only to compute independent deltas; the exact world pricer emits only native prefixes.
use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
use mlx_gen::Result;
use mlx_rs::Dtype;

use super::{fused_header_footprint, Inputs, Regime};
use crate::config::ClipTextConfig;
use crate::text_encoder::{text_pack_accounting, ClipPrice};

pub(super) type Picks = Vec<(usize, usize)>;

/// One encoder outcome: penultimate (dtype, shape), pooled dtype, best retained delta, picks.
type Outcome = ((Dtype, Vec<i32>), Dtype, i128, Picks);

struct WorldFacts {
    retained: u64,
    bytes: u64,
    points: Vec<(String, u64, u64)>,
    failures: Vec<usize>,
    clip: ClipPrice,
}
struct Encoder<'a> {
    component: C,
    cfg: &'a ClipTextConfig,
    groups: Vec<usize>,
    layer: Vec<usize>,
    /// The current assignment of this encoder's encoder-global groups (embeddings, final norm,
    /// projection or multi-layer groups). They are enumerated: every witness carries one.
    fixed: Picks,
}
struct Alternative {
    assignment: usize,
    retained: i128,
    bytes: i128,
    before: Vec<i128>,
    failure: Option<usize>,
}
struct State {
    x: Dtype,
    shape: Vec<i32>,
    penultimate: Option<(Dtype, Vec<i32>)>,
    value: i128,
    picks: Picks,
}
#[derive(Default)]
struct EncoderWitnesses {
    kernels: Vec<Picks>,
    packs: Vec<Picks>,
    outcomes: Vec<Picks>,
    retained_max: Vec<Picks>,
    successful_max: Vec<Picks>,
    bytes_max: Vec<Picks>,
}

fn layer_of(target: &str) -> Option<usize> {
    target
        .strip_prefix("text_model.encoder.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}
fn facts(inputs: &Inputs, encoder: &Encoder<'_>, picks: &[(usize, usize)]) -> Result<WorldFacts> {
    let source = &inputs
        .fused
        .as_ref()
        .expect("text aliases are fused")
        .source;
    let picks = encoder
        .fixed
        .iter()
        .chain(picks)
        .copied()
        .collect::<Picks>();
    let headers = source.headers_with(encoder.component, &picks);
    let mut clip = ClipPrice::from_headers(&headers, "text_model", encoder.cfg)?;
    let (pack, _) = text_pack_accounting(&headers, encoder.cfg, inputs.text_quant_bits, false)?;
    if let Some(bits) = inputs.text_quant_bits {
        clip.pack_witness_terms(bits)?;
    }
    let bytes = fused_header_footprint(&headers, crate::model::DTYPE, false)?.resident;
    Ok(WorldFacts {
        retained: pack.retained,
        bytes,
        points: pack
            .points
            .into_iter()
            .map(|p| (p.path, p.before_resident, p.scratch))
            .collect(),
        failures: pack.failures,
        clip,
    })
}
fn reaches(value: &Alternative, point: usize) -> bool {
    value.failure.is_none_or(|failure| failure >= point)
}
fn best(
    values: &[Alternative],
    point: usize,
    key: impl Fn(&Alternative) -> i128,
) -> Option<(usize, i128)> {
    values
        .iter()
        .filter(|v| reaches(v, point))
        .map(|v| (v.assignment, key(v)))
        .max_by_key(|(_, v)| *v)
}
fn local_products(
    alternatives: &[Vec<Alternative>],
    encoder: &Encoder<'_>,
    groups: &[usize],
    point: usize,
) -> Vec<(Picks, i128)> {
    let mut result = vec![(Vec::new(), 0)];
    for &local in groups {
        let mut next = Vec::new();
        for (picks, delta) in &result {
            for value in alternatives[local].iter().filter(|v| reaches(v, point)) {
                let mut picks = picks.clone();
                picks.push((encoder.groups[local], value.assignment));
                next.push((picks, delta + value.retained));
            }
        }
        result = next;
    }
    result
}
fn recover<T>(result: Result<T>) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if crate::quantum::is_native_shape_failure(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

fn encoder_witnesses(
    inputs: &Inputs,
    regime: &Regime,
    encoder: &Encoder<'_>,
) -> Result<EncoderWitnesses> {
    let base = facts(inputs, encoder, &[])?;
    let source = &inputs
        .fused
        .as_ref()
        .expect("text aliases are fused")
        .source;
    let owns = |local: usize, point: usize| {
        let prefix = format!("{}.", base.points[point].0);
        source.aliases[encoder.groups[local]]
            .targets
            .iter()
            .any(|target| target.starts_with(&prefix))
    };
    let fixed_failure = base
        .failures
        .iter()
        .copied()
        .filter(|&point| !(0..encoder.groups.len()).any(|g| owns(g, point)))
        .min();
    let mut alternatives = Vec::new();
    for (local, &group) in encoder.groups.iter().enumerate() {
        let mut values = Vec::new();
        for assignment in 0..source.aliases[group].assignments.len() {
            let world = facts(inputs, encoder, &[(group, assignment)])?;
            let before = world
                .points
                .iter()
                .zip(&base.points)
                .map(|((path, before, _), (base_path, base_before, _))| {
                    debug_assert_eq!(path, base_path, "pack points are name-invariant");
                    i128::from(*before) - i128::from(*base_before)
                })
                .collect();
            values.push(Alternative {
                assignment,
                retained: i128::from(world.retained) - i128::from(base.retained),
                bytes: i128::from(world.bytes) - i128::from(base.bytes),
                before,
                failure: world
                    .failures
                    .iter()
                    .copied()
                    .filter(|&point| owns(local, point))
                    .min(),
            });
        }
        alternatives.push(values);
    }
    let all = |point: usize, key: &dyn Fn(&Alternative) -> i128| -> Option<Picks> {
        if fixed_failure.is_some_and(|failure| failure < point) {
            return None;
        }
        alternatives
            .iter()
            .enumerate()
            .map(|(local, values)| {
                best(values, point, key).map(|(assignment, _)| (encoder.groups[local], assignment))
            })
            .collect()
    };
    let retained_max = all(usize::MAX, &|v| v.retained);
    // Before packing, even a future shape-failing assignment is a real loaded owner.
    let bytes_max = alternatives
        .iter()
        .enumerate()
        .map(|(local, values)| {
            let value = values
                .iter()
                .max_by_key(|v| v.bytes)
                .expect("realizable group");
            (encoder.groups[local], value.assignment)
        })
        .collect();
    let mut kernels = Vec::new();
    // Keyed by the penultimate (dtype, shape) the native TE1/TE2 concat reads, and pooled dtype.
    let mut outcomes: Vec<Outcome> = Vec::new();
    if let Some(retained_max) = &retained_max {
        kernels.push(retained_max.clone());
        let (b, n) = (regime.conditioning_rows(), inputs.arch.text_tokens);
        let layers = base.clip.layer_count();
        let mut states = match recover(base.clip.embeddings_quantum(b, n).1)? {
            Some((x, shape)) => vec![State {
                x,
                shape,
                penultimate: None,
                value: 0,
                picks: Vec::new(),
            }],
            None => Vec::new(),
        };
        for layer in 0..layers {
            let local = (0..encoder.groups.len())
                .filter(|&g| encoder.layer[g] == layer)
                .collect::<Vec<_>>();
            let mut suffix = Vec::new();
            let mut suffix_value = 0;
            for (group, values) in alternatives.iter().enumerate() {
                if encoder.layer[group] > layer {
                    let (assignment, delta) =
                        best(values, usize::MAX, |v| v.retained).expect("post-pack path exists");
                    suffix.push((encoder.groups[group], assignment));
                    suffix_value += delta;
                }
            }
            let mut window: Option<(i128, Picks)> = None;
            let mut next: Vec<State> = Vec::new();
            for (combo, delta) in local_products(&alternatives, encoder, &local, usize::MAX) {
                let world = facts(inputs, encoder, &combo)?;
                for state in &states {
                    let (scratch, output) =
                        world.clip.layer_quantum(layer, b, n, state.x, &state.shape);
                    let candidate = state.value + delta + i128::from(scratch) + suffix_value;
                    if window.as_ref().is_none_or(|(value, _)| candidate > *value) {
                        window = Some((
                            candidate,
                            state
                                .picks
                                .iter()
                                .chain(&combo)
                                .chain(&suffix)
                                .copied()
                                .collect(),
                        ));
                    }
                    // The guard above counts even when its native operation fails. Such a world
                    // has no later layer, pool, heavy-stage (staged), denoise or decode outcome.
                    let Some((x, shape)) = recover(output)? else {
                        continue;
                    };
                    let penultimate = if layer + 2 == layers {
                        Some((x, shape.clone()))
                    } else {
                        state.penultimate.clone()
                    };
                    let value = state.value + delta;
                    let picks = state.picks.iter().chain(&combo).copied().collect();
                    match next.iter_mut().find(|other| {
                        other.x == x && other.shape == shape && other.penultimate == penultimate
                    }) {
                        Some(other) if other.value >= value => {}
                        Some(other) => {
                            other.value = value;
                            other.picks = picks;
                        }
                        None => next.push(State {
                            x,
                            shape,
                            penultimate,
                            value,
                            picks,
                        }),
                    }
                }
            }
            if let Some((_, picks)) = window {
                kernels.push(picks);
            }
            states = next;
        }
        let mut pooled: Option<(i128, Picks)> = None;
        for state in &states {
            let (scratch, result) = base.clip.pooled_quantum(b, n, state.x, &state.shape);
            let candidate = state.value + i128::from(scratch);
            if pooled.as_ref().is_none_or(|(value, _)| candidate > *value) {
                pooled = Some((candidate, state.picks.clone()));
            }
            let Some(pooled_dtype) = recover(result)? else {
                continue;
            };
            let penultimate = state
                .penultimate
                .clone()
                .unwrap_or_else(|| (state.x, state.shape.clone()));
            match outcomes
                .iter_mut()
                .find(|(p, q, _, _)| *p == penultimate && *q == pooled_dtype)
            {
                Some(outcome) if outcome.2 >= state.value => {}
                Some(outcome) => {
                    outcome.2 = state.value;
                    outcome.3 = state.picks.clone();
                }
                None => {
                    outcomes.push((penultimate, pooled_dtype, state.value, state.picks.clone()))
                }
            }
        }
        if let Some((_, picks)) = pooled {
            kernels.push(picks);
        }
    }
    let successful_max = outcomes
        .iter()
        .max_by_key(|outcome| outcome.2)
        .map(|outcome| outcome.3.clone());
    let mut packs = Vec::new();
    for (index, _) in base.points.iter().enumerate() {
        if fixed_failure.is_some_and(|failure| failure < index) {
            break;
        }
        let local = (0..encoder.groups.len())
            .filter(|&g| owns(g, index))
            .collect::<Vec<_>>();
        let mut others = Vec::new();
        let mut reachable = true;
        for (group, values) in alternatives.iter().enumerate() {
            if !local.contains(&group) {
                if let Some((assignment, _)) = best(values, index, |v| v.before[index]) {
                    others.push((encoder.groups[group], assignment));
                } else {
                    reachable = false;
                    break;
                }
            }
        }
        if !reachable {
            continue;
        }
        let mut point: Option<(u64, Picks)> = None;
        for (combo, _) in local_products(&alternatives, encoder, &local, index) {
            let world = facts(inputs, encoder, &combo)?;
            let value = world.points[index].1 + world.points[index].2;
            if point.as_ref().is_none_or(|(best, _)| value > *best) {
                point = Some((value, combo.iter().chain(&others).copied().collect()));
            }
        }
        if let Some((_, picks)) = point {
            packs.push(picks);
        }
    }
    let with_fixed =
        |picks: Picks| -> Picks { encoder.fixed.iter().copied().chain(picks).collect() };
    Ok(EncoderWitnesses {
        kernels: kernels.into_iter().map(with_fixed).collect(),
        packs: packs.into_iter().map(with_fixed).collect(),
        outcomes: outcomes.into_iter().map(|o| with_fixed(o.3)).collect(),
        retained_max: retained_max.into_iter().map(with_fixed).collect(),
        successful_max: successful_max.into_iter().map(with_fixed).collect(),
        bytes_max: vec![with_fixed(bytes_max)],
    })
}

/// Text witness picks for text-phase windows, and the text worlds every downstream window reads
/// (encoder outcome pairs at their best retained bytes, plus the retained maximum warm heavy
/// loads read). Both are `[[]]` without text aliases.
fn text_witnesses(inputs: &Inputs, regime: &Regime) -> Result<(Vec<Picks>, Vec<Picks>)> {
    let fused = inputs.fused.as_ref().expect("aliases are fused");
    if !fused
        .source
        .aliases
        .iter()
        .any(|alias| matches!(alias.component, C::ClipL | C::ClipBigG))
    {
        return Ok((vec![Vec::new()], vec![Vec::new()]));
    }
    let mut witnesses = Vec::new();
    for (component, cfg) in [
        (C::ClipL, &inputs.arch.te1),
        (C::ClipBigG, &inputs.arch.te2),
    ] {
        let mut groups = Vec::new();
        let mut layer = Vec::new();
        let mut global = Vec::new();
        for (index, alias) in fused.source.aliases.iter().enumerate() {
            if alias.component != component {
                continue;
            }
            let layers = alias
                .targets
                .iter()
                .map(|target| layer_of(target))
                .collect::<Option<Vec<_>>>();
            match layers.as_deref() {
                Some([first, rest @ ..]) if rest.iter().all(|other| other == first) => {
                    groups.push(index);
                    layer.push(*first);
                }
                _ => global.push(
                    (0..alias.assignments.len())
                        .map(|assignment| (index, assignment))
                        .collect::<Vec<_>>(),
                ),
            }
        }
        let mut merged = EncoderWitnesses::default();
        for fixed in super::heavy_alias::product(&global) {
            let encoder = Encoder {
                component,
                cfg,
                groups: groups.clone(),
                layer: layer.clone(),
                fixed,
            };
            let found = encoder_witnesses(inputs, regime, &encoder)?;
            merged.kernels.extend(found.kernels);
            merged.packs.extend(found.packs);
            merged.outcomes.extend(found.outcomes);
            merged.retained_max.extend(found.retained_max);
            merged.successful_max.extend(found.successful_max);
            merged.bytes_max.extend(found.bytes_max);
        }
        witnesses.push(merged);
    }
    let [te1, te2] = [&witnesses[0], &witnesses[1]];
    let pairs = |a: &[Picks], b: &[Picks]| {
        a.iter()
            .flat_map(|first| {
                b.iter()
                    .map(move |second| first.iter().chain(second).copied().collect::<Picks>())
            })
            .collect::<Vec<_>>()
    };
    let mut text = Vec::new();
    text.extend(pairs(&te1.kernels, &te2.retained_max));
    text.extend(pairs(&te1.successful_max, &te2.kernels));
    text.extend(pairs(&te1.packs, &te2.bytes_max));
    text.extend(pairs(&te1.retained_max, &te2.packs));
    // Also witnesses the pre-pack source read/cast window when every path fails before a pack.
    text.extend(pairs(&te1.bytes_max, &te2.bytes_max));
    let mut context = pairs(&te1.outcomes, &te2.outcomes);
    context.extend(pairs(&te1.retained_max, &te2.retained_max));
    Ok((text, context))
}

/// Every witness world whose whole-candidate envelope is the exact maximum over all realizable
/// text and UNet/VAE alias worlds. Text windows read heavy aliases only through additive loaded
/// bytes; heavy windows read text aliases only through retained bytes and encoder outcomes.
pub(super) fn witness_worlds(
    inputs: &mut Inputs,
    regime: &Regime,
    configured: Option<&mlx_gen::gen_core::LoadSpec>,
) -> Result<Vec<Picks>> {
    if inputs
        .fused
        .as_ref()
        .is_none_or(|fused| fused.source.aliases.is_empty())
    {
        return Ok(vec![Vec::new()]);
    }
    let (text, context) = text_witnesses(inputs, regime)?;
    let join = |a: &Picks, b: &Picks| a.iter().chain(b).copied().collect::<Picks>();
    let mut worlds = Vec::new();
    let mut resident = Vec::new();
    for picks in &context {
        let heavy = super::heavy_alias::heavy_witnesses(inputs, regime, configured, picks)?;
        worlds.extend(heavy.all.iter().map(|other| join(picks, other)));
        resident.extend(heavy.resident);
    }
    if context.is_empty() {
        let heavy = super::heavy_alias::heavy_witnesses(inputs, regime, configured, &Vec::new())?;
        resident.extend(heavy.resident);
    }
    resident.sort();
    resident.dedup();
    for picks in &text {
        worlds.extend(resident.iter().map(|other| join(picks, other)));
    }
    for world in &mut worlds {
        world.sort_unstable();
    }
    worlds.sort();
    worlds.dedup();
    Ok(worlds)
}
