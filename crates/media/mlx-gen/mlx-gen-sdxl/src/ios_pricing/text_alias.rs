//! Exact pricing of order-dependent fused text-encoder source aliases.
//!
//! A text alias group's winner changes only its own encoder's component headers. Every SDXL
//! pricing consumer reads those headers through three channels, each proved by the code it names:
//!
//! - **Additive resident terms.** `text_pack_facts` retained/`before_resident` and the fused
//!   footprint are sums of per-target terms; pack-point existence is fixed by `.scales` target
//!   NAMES, so every world has the same pack points. Groups share no target, hence each group's
//!   contribution is its delta against the base world, independent of every other group.
//! - **Local kernels.** A CLIP layer quantum reads only its own layer's facts plus the incoming
//!   activation dtype (`ClipPrice::layer_quantum`); a pack point's scratch reads only its own
//!   projection. Every alias group lies in exactly one encoder layer.
//! - **Encoder outcome.** Everything downstream (UNet/conditioning windows) reads the encoders only
//!   through the penultimate/pooled dtypes and the retained text bytes.
//!
//! So each window's maximum over all realizable worlds is attained by a witness built from one
//! dynamic program over activation dtypes: upstream groups through the state, local groups by
//! their small product, downstream and non-local groups at their own additive maximum. Every
//! witness is a real realizable world priced by the unchanged exact pricer; the caller keeps the
//! whole-candidate envelope. No global product, independent maxima or heuristic bound is used.
use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use super::{fused_header_footprint, Inputs, Regime};
use crate::config::ClipTextConfig;
use crate::text_encoder::{text_pack_facts, ClipPrice};

/// Alias-group assignments selecting one realizable world (`(group, assignment)`).
pub(super) type Picks = Vec<(usize, usize)>;

/// One world's text-encoder facts on the three channels above.
struct WorldFacts {
    retained: u64,
    bytes: u64,
    points: Vec<(String, u64, u64)>,
    clip: ClipPrice,
}

struct Encoder<'a> {
    component: C,
    cfg: &'a ClipTextConfig,
    groups: Vec<usize>,
    layer: Vec<usize>,
}

/// One valid group assignment: (assignment, retained delta, footprint delta, per-point
/// before-resident deltas), each against the base world.
type Alternative = (usize, i128, i128, Vec<i128>);

struct Alternatives {
    // Per encoder group: (assignment, retained delta, bytes delta, per-point before delta).
    valid: Vec<Vec<Alternative>>,
}

/// One state of the activation-dtype program: its incoming dtype, the penultimate hidden dtype
/// once known, the best additive retained delta reaching it, and that path's assignments.
struct State {
    x: Dtype,
    penultimate: Option<Dtype>,
    value: i128,
    picks: Picks,
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
    let headers = source.headers_with(encoder.component, picks);
    let mut clip = ClipPrice::from_headers(&headers, "text_model", encoder.cfg)?;
    let pack = text_pack_facts(&headers, encoder.cfg, inputs.text_quant_bits)?;
    if let Some(bits) = inputs.text_quant_bits {
        clip.quantize(bits)?;
    }
    let bytes = fused_header_footprint(&headers, crate::model::DTYPE, false)?.resident;
    Ok(WorldFacts {
        retained: pack.retained,
        bytes,
        points: pack
            .points
            .into_iter()
            .map(|point| (point.path, point.before_resident, point.scratch))
            .collect(),
        clip,
    })
}

fn best(values: &[Alternative], key: impl Fn(&Alternative) -> i128) -> (usize, i128) {
    values
        .iter()
        .map(|value| (value.0, key(value)))
        .max_by_key(|(_, delta)| *delta)
        .expect("every group keeps a valid assignment")
}

/// Every local assignment product over `groups` (indices into the encoder's group list).
fn local_products(
    alternatives: &Alternatives,
    encoder: &Encoder<'_>,
    groups: &[usize],
) -> Vec<(Picks, i128)> {
    let mut result: Vec<(Picks, i128)> = vec![(Vec::new(), 0)];
    for &local in groups {
        let mut next = Vec::new();
        for (picks, delta) in &result {
            for (assignment, retained, _, _) in &alternatives.valid[local] {
                let mut picks = picks.clone();
                picks.push((encoder.groups[local], *assignment));
                next.push((picks, delta + retained));
            }
        }
        result = next;
    }
    result
}

/// Per encoder: the witness worlds of its own windows (CLIP kernels, pack points, outcomes) and
/// its additive argmax worlds for the retained and footprint channels.
struct EncoderWitnesses {
    kernels: Vec<Picks>,
    packs: Vec<Picks>,
    outcomes: Vec<Picks>,
    retained_max: Picks,
    bytes_max: Picks,
}

fn encoder_witnesses(
    inputs: &Inputs,
    regime: &Regime,
    encoder: &Encoder<'_>,
) -> Result<EncoderWitnesses> {
    let base = facts(inputs, encoder, &[])?;
    let mut alternatives = Alternatives { valid: Vec::new() };
    for &group in &encoder.groups {
        let source = &inputs
            .fused
            .as_ref()
            .expect("text aliases are fused")
            .source;
        let mut valid = Vec::new();
        let mut first_error = None;
        for assignment in 0..source.aliases[group].assignments.len() {
            match facts(inputs, encoder, &[(group, assignment)]) {
                Ok(world) => {
                    let points = world
                        .points
                        .iter()
                        .zip(&base.points)
                        .map(|((path, before, _), (base_path, base_before, _))| {
                            debug_assert_eq!(path, base_path, "pack points are name-invariant");
                            i128::from(*before) - i128::from(*base_before)
                        })
                        .collect();
                    valid.push((
                        assignment,
                        i128::from(world.retained) - i128::from(base.retained),
                        i128::from(world.bytes) - i128::from(base.bytes),
                        points,
                    ));
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if valid.is_empty() {
            // Every native winner of this group fails its own constructor/pack: certain failure.
            return Err(first_error.expect("an alias group has at least one assignment"));
        }
        alternatives.valid.push(valid);
    }
    let all = |key: &dyn Fn(&Alternative) -> i128| -> (Picks, i128) {
        let mut picks = Vec::new();
        let mut total = 0;
        for (local, valid) in alternatives.valid.iter().enumerate() {
            let (assignment, delta) = best(valid, key);
            picks.push((encoder.groups[local], assignment));
            total += delta;
        }
        (picks, total)
    };
    let retained_max = all(&|value| value.1).0;
    let bytes_max = all(&|value| value.2).0;

    let (b, n) = (regime.conditioning_rows(), inputs.arch.text_tokens);
    let layers = base.clip.layer_count();
    let mut kernels = vec![retained_max.clone()];
    let mut states = vec![State {
        x: base.clip.embeddings_quantum(b, n).1,
        penultimate: None,
        value: 0,
        picks: Vec::new(),
    }];
    for layer in 0..layers {
        let local = (0..encoder.groups.len())
            .filter(|&group| encoder.layer[group] == layer)
            .collect::<Vec<_>>();
        let mut suffix = Vec::new();
        let mut suffix_value = 0;
        for (group, valid) in alternatives.valid.iter().enumerate() {
            if encoder.layer[group] > layer {
                let (assignment, delta) = best(valid, |value| value.1);
                suffix.push((encoder.groups[group], assignment));
                suffix_value += delta;
            }
        }
        let mut window: Option<(i128, Picks)> = None;
        let mut next: Vec<State> = Vec::new();
        for (combo, delta) in local_products(&alternatives, encoder, &local) {
            let local_facts;
            let clip = if combo.is_empty() {
                &base.clip
            } else {
                local_facts = facts(inputs, encoder, &combo)?;
                &local_facts.clip
            };
            for state in &states {
                let (scratch, x) = clip.layer_quantum(layer, b, n, state.x);
                let candidate = state.value + delta + i128::from(scratch) + suffix_value;
                if window.as_ref().is_none_or(|(value, _)| candidate > *value) {
                    let mut picks = state.picks.clone();
                    picks.extend(combo.iter().copied());
                    picks.extend(suffix.iter().copied());
                    window = Some((candidate, picks));
                }
                let penultimate = if layer + 2 == layers {
                    Some(x)
                } else {
                    state.penultimate
                };
                let value = state.value + delta;
                match next
                    .iter_mut()
                    .find(|other| other.x == x && other.penultimate == penultimate)
                {
                    Some(other) if other.value >= value => {}
                    Some(other) => {
                        other.value = value;
                        other.picks = state.picks.iter().chain(&combo).copied().collect();
                    }
                    None => next.push(State {
                        x,
                        penultimate,
                        value,
                        picks: state.picks.iter().chain(&combo).copied().collect(),
                    }),
                }
            }
        }
        if let Some((_, picks)) = window {
            kernels.push(picks);
        }
        states = next;
    }
    // The pooled kernel and every downstream window read the final state and retained bytes.
    let mut pooled: Option<(i128, Picks)> = None;
    let mut outcomes: Vec<(Dtype, Dtype, i128, Picks)> = Vec::new();
    for state in &states {
        let (scratch, pooled_dtype) = base.clip.pooled_quantum(b, n, state.x);
        let candidate = state.value + i128::from(scratch);
        if pooled.as_ref().is_none_or(|(value, _)| candidate > *value) {
            pooled = Some((candidate, state.picks.clone()));
        }
        let penultimate = state.penultimate.unwrap_or(state.x);
        match outcomes
            .iter_mut()
            .find(|(p, q, _, _)| *p == penultimate && *q == pooled_dtype)
        {
            Some(outcome) if outcome.2 >= state.value => {}
            Some(outcome) => {
                outcome.2 = state.value;
                outcome.3 = state.picks.clone();
            }
            None => outcomes.push((penultimate, pooled_dtype, state.value, state.picks.clone())),
        }
    }
    if let Some((_, picks)) = pooled {
        kernels.push(picks);
    }
    // Pack points: the local projection's groups by product, every other group at its own
    // additive before-resident maximum for THIS point.
    let mut packs = Vec::new();
    for (index, (path, _, _)) in base.points.iter().enumerate() {
        let prefix = format!("{path}.");
        let source = &inputs
            .fused
            .as_ref()
            .expect("text aliases are fused")
            .source;
        let local = (0..encoder.groups.len())
            .filter(|&group| {
                source.aliases[encoder.groups[group]]
                    .targets
                    .iter()
                    .any(|target| target.starts_with(&prefix))
            })
            .collect::<Vec<_>>();
        let mut others = Vec::new();
        for (group, valid) in alternatives.valid.iter().enumerate() {
            if !local.contains(&group) {
                let (assignment, _) = best(valid, |value| value.3[index]);
                others.push((encoder.groups[group], assignment));
            }
        }
        let mut point: Option<(u64, Picks)> = None;
        for (combo, _) in local_products(&alternatives, encoder, &local) {
            let value = if combo.is_empty() {
                base.points[index].1 + base.points[index].2
            } else {
                let world = facts(inputs, encoder, &combo)?;
                world.points[index].1 + world.points[index].2
            };
            if point.as_ref().is_none_or(|(best, _)| value > *best) {
                point = Some((value, combo.iter().chain(&others).copied().collect()));
            }
        }
        if let Some((_, picks)) = point {
            packs.push(picks);
        }
    }
    Ok(EncoderWitnesses {
        kernels,
        packs,
        outcomes: outcomes.into_iter().map(|outcome| outcome.3).collect(),
        retained_max,
        bytes_max,
    })
}

/// Every witness world whose whole-candidate envelope is the exact maximum over all realizable
/// text-alias worlds. A single empty world when the source has no aliases.
pub(super) fn witness_worlds(inputs: &Inputs, regime: &Regime) -> Result<Vec<Picks>> {
    let Some(fused) = inputs
        .fused
        .as_ref()
        .filter(|fused| !fused.source.aliases.is_empty())
    else {
        return Ok(vec![Vec::new()]);
    };
    let mut encoders = Vec::new();
    for (component, cfg) in [
        (C::ClipL, &inputs.arch.te1),
        (C::ClipBigG, &inputs.arch.te2),
    ] {
        let mut groups = Vec::new();
        let mut layer = Vec::new();
        for (index, alias) in fused.source.aliases.iter().enumerate() {
            if alias.component != component {
                continue;
            }
            let layers = alias
                .targets
                .iter()
                .map(|target| layer_of(target))
                .collect::<Option<Vec<_>>>();
            let Some(&first) = layers.as_ref().and_then(|layers| layers.first()) else {
                return Err(Error::Unsupported(
                    "sdxl fused text alias outside an encoder layer remains unpriced".into(),
                ));
            };
            if layers.iter().flatten().any(|&other| other != first) {
                return Err(Error::Unsupported(
                    "sdxl fused text alias spanning encoder layers remains unpriced".into(),
                ));
            }
            groups.push(index);
            layer.push(first);
        }
        encoders.push(Encoder {
            component,
            cfg,
            groups,
            layer,
        });
    }
    let witnesses = encoders
        .iter()
        .map(|encoder| encoder_witnesses(inputs, regime, encoder))
        .collect::<Result<Vec<_>>>()?;
    let [te1, te2] = [&witnesses[0], &witnesses[1]];
    let join = |a: &Picks, b: &Picks| a.iter().chain(b).copied().collect::<Picks>();
    let mut worlds: Vec<Picks> = Vec::new();
    // Text kernels: the other encoder at its retained maximum (text() is their sum).
    worlds.extend(
        te1.kernels
            .iter()
            .map(|picks| join(picks, &te2.retained_max)),
    );
    worlds.extend(
        te2.kernels
            .iter()
            .map(|picks| join(&te1.retained_max, picks)),
    );
    // Encoder-1 pack points also hold encoder 2's pre-pack footprint; encoder-2 points hold
    // encoder 1's post-pack retained bytes.
    worlds.extend(te1.packs.iter().map(|picks| join(picks, &te2.bytes_max)));
    worlds.extend(te2.packs.iter().map(|picks| join(&te1.retained_max, picks)));
    // Downstream windows: every pair of encoder outcomes, each at its best retained path.
    for first in &te1.outcomes {
        for second in &te2.outcomes {
            worlds.push(join(first, second));
        }
    }
    for world in &mut worlds {
        world.sort_unstable();
    }
    worlds.sort();
    worlds.dedup();
    Ok(worlds)
}
