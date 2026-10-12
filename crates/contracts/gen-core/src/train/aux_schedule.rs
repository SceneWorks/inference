//! Backend-neutral **auxiliary-loss step policy** (epic 2123 E8, sc-2125) — which loss terms a
//! training micro-step trains. Pure policy, so the MLX and Candle trainers share one copy:
//!
//! - [`AuxAlternation`] turns a micro-step into its [`AltKey`]: the step's **optimizer window**
//!   (every micro-step of one gradient-accumulation window shares it, so accumulation never
//!   averages a diffusion and an aux micro-step into one update) and the visit order's **period**
//!   in windows. [`AltKey::claims`] decides, per loss period `every_n`, whether the window is an
//!   aux-only window. Consecutive windows interleave like upstream ai-toolkit-perceptual's
//!   per-optimizer-step alternation (`step_num % 2`: never two aux-only windows in a row), and the
//!   phase drifts by one residue per pass over the data so a fixed visit order cannot lock an
//!   image to one step kind (sc-2124). The key is a pure function of the step, so a resumed run
//!   needs no replay.
//! - [`plan_step`] turns the per-loss [`AuxLossSchedule`]s, the key and the sampled noise level into
//!   a [`StepPlan`]. On an aux-only step the noise level is remapped into the claiming losses'
//!   window, so the claim never depends on the sampled timestep (every micro-step of a window gets
//!   the same kind) and no step is wasted out of window.
//! - [`StepPlan::without_skipped`] drops losses for which the step's image has no usable reference
//!   (e.g. no face found); an aux-only step left with no claiming loss **trains the diffusion loss
//!   instead** rather than taking a zero-gradient step.
//! - [`combine_step_terms`] sums the present terms and refuses a step with none.

use super::AuxLossSchedule;

/// Which loss terms one training micro-step trains, and at what noise level.
#[derive(Clone, Debug, PartialEq)]
pub struct StepPlan {
    /// Whether the diffusion loss contributes. `false` ⇔ an aux-only step: the diffusion term
    /// contributes **zero**.
    pub diffusion: bool,
    /// Indices of the aux losses that contribute this step, ascending.
    pub aux: Vec<usize>,
    /// The aux losses whose alternation claimed this step (a subset of `aux` on an aux-only step;
    /// empty on a diffusion step).
    pub claimed: Vec<usize>,
    /// The noise level the step trains at: the sampled level, or — on an aux-only step — the
    /// sampled level remapped into the claiming losses' window.
    pub noise_level: f32,
}

impl StepPlan {
    /// The plain diffusion step at noise level `t` (every step when no aux loss is enabled).
    pub fn diffusion_only(t: f32) -> Self {
        Self {
            diffusion: true,
            aux: Vec::new(),
            claimed: Vec::new(),
            noise_level: t,
        }
    }

    /// Drop every loss `skipped(i)` reports unusable for this step's image. An aux-only step whose
    /// claiming losses are all skipped trains the diffusion loss instead (at the same noise level);
    /// summed (`every_n == 1`) losses that remain still ride along.
    pub fn without_skipped(mut self, skipped: impl Fn(usize) -> bool) -> Self {
        self.aux.retain(|&i| !skipped(i));
        self.claimed.retain(|&i| !skipped(i));
        if !self.diffusion && self.claimed.is_empty() {
            self.diffusion = true;
        }
        self
    }
}

/// One micro-step's alternation key (see the module docs and [`AltKey::claims`]).
///
/// A bare `u32` converts to the key of that window with no period (`period == 0`): loss `every_n`
/// claims window `w` iff `w % every_n == 0` — the plain upstream counter, for callers that plan an
/// explicit window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AltKey {
    /// The 1-based optimizer window the micro-step belongs to.
    pub window: u32,
    /// Windows per period of the visit order (`lcm(epoch_steps, window_steps) / window_steps`), or
    /// `0` for no period.
    pub period: u32,
}

impl From<u32> for AltKey {
    fn from(window: u32) -> Self {
        Self { window, period: 0 }
    }
}

impl AltKey {
    /// Whether a loss with alternation period `every_n` claims this window (makes it aux-only).
    /// `every_n < 2` never claims (summed losses).
    ///
    /// The window `w` is mapped to a counter `c(w)` and the loss claims iff `c(w) % every_n == 0`,
    /// where, with `m` the period and `n = every_n`:
    ///
    /// - `s` = the smallest `s ≥ 0` with `gcd(m + s, n) = 1` (`0` whenever `gcd(m, n) = 1`);
    /// - `s == 0` ⇒ `c(w) = w`;
    /// - `n == 2`, `s == 1` (an even period) ⇒ `c(w) = w + ⌊(w − 1)/(m + 1)⌋`;
    /// - otherwise ⇒ `c(w) = w + ⌊(w − 1)·s/m⌋`.
    ///
    /// Why: `c` steps by 1 or 2 per window (`s < n` and, when `m < s`, `m ≥ 2`), so two claims are
    /// never adjacent — a step of 2 skips one value, and for `n ≥ 3` two values 2 apart are never
    /// both multiples of `n`, while for `n == 2` the skipped values `j·(m + 2)` are all even, so a
    /// skip makes a diffusion pair, never an aux pair. That is upstream's bound: no two aux-only
    /// windows in a row. And the same slot of the visit order one period later sits `m + s`
    /// counter values on (exactly, for `n ≥ 3`) — a unit modulo `n` — so every slot walks through
    /// every residue within `n` periods; for `n == 2` a slot flips every period except once every
    /// `m + 1` periods, so it gets both kinds within any 3 periods. (Flipping every slot every
    /// period is impossible for an even period without two adjacent aux windows at alternate
    /// period boundaries.)
    pub fn claims(self, every_n: u32) -> bool {
        if every_n < 2 {
            return false;
        }
        let (w, n) = (u64::from(self.window.max(1)), u64::from(every_n));
        if self.period == 0 {
            return w.is_multiple_of(n);
        }
        let m = u64::from(self.period);
        // `s = (1 − m) mod n` makes `m + s ≡ 1 (mod n)`, so the search always finds one.
        let s = (0..n)
            .find(|s| gcd(m + s, n) == 1)
            .unwrap_or((n + 1 - m % n) % n);
        let c = match s {
            0 => w,
            1 if n == 2 => w + (w - 1) / (m + 1),
            _ => w + (w - 1) * s / m,
        };
        c.is_multiple_of(n)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The alternation over a training loop's visit order (see the module docs). Pure: a micro-step's
/// key depends only on the step, so a resumed run continues the same phase without replay.
#[derive(Clone, Copy, Debug)]
pub struct AuxAlternation {
    window: u32,
    period: u32,
}

impl AuxAlternation {
    /// For a loop whose visit order repeats every `epoch_steps` micro-steps (one pass over the
    /// data: `BucketSchedule::epoch_len`, times the expert count for Wan's interleaved experts)
    /// with `window_steps` micro-steps per optimizer window (the gradient accumulation, times the
    /// expert count for Wan).
    pub fn new(epoch_steps: usize, window_steps: u32) -> Self {
        let window = window_steps.max(1);
        let l = epoch_steps.max(1) as u64;
        // lcm(l, window) / window: the windows after which the window ↔ sample layout repeats.
        let period = l / gcd(l, u64::from(window));
        Self {
            window,
            period: u32::try_from(period).unwrap_or(u32::MAX),
        }
    }

    /// The key of 1-based micro-step `step`.
    pub fn key(&self, step: u32) -> AltKey {
        AltKey {
            window: (step.max(1) - 1) / self.window + 1,
            period: self.period,
        }
    }
}

/// The plan for alternation `key` at sampled noise level `raw_t`:
///
/// - a loss **claims** the step when it is enabled and the key [claims](AltKey::claims) its
///   `every_n`;
/// - any claim ⇒ **aux-only** step: the noise level is remapped into the claiming windows'
///   intersection (the first claimer's window when they do not intersect), and the claiming losses
///   whose window holds it contribute, plus every enabled `every_n == 1` loss whose window holds it;
/// - no claim ⇒ diffusion step at `raw_t`, plus every enabled `every_n == 1` loss in window.
pub fn plan_step(schedules: &[AuxLossSchedule], key: impl Into<AltKey>, raw_t: f32) -> StepPlan {
    let key = key.into();
    let claiming: Vec<usize> = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_enabled() && key.claims(s.every_n))
        .map(|(i, _)| i)
        .collect();
    let (diffusion, t) = match claiming.first() {
        None => (true, raw_t),
        Some(&first) => {
            let lo = claiming
                .iter()
                .map(|&i| schedules[i].t_min)
                .fold(f32::MIN, f32::max);
            let hi = claiming
                .iter()
                .map(|&i| schedules[i].t_max)
                .fold(f32::MAX, f32::min);
            let (lo, hi) = if lo <= hi {
                (lo, hi)
            } else {
                (schedules[first].t_min, schedules[first].t_max)
            };
            (false, lo + raw_t.clamp(0.0, 1.0) * (hi - lo))
        }
    };
    let claimed: Vec<usize> = claiming
        .into_iter()
        .filter(|&i| schedules[i].in_window(t))
        .collect();
    let mut aux: Vec<usize> = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_enabled() && s.every_n == 1 && s.in_window(t))
        .map(|(i, _)| i)
        .chain(claimed.iter().copied())
        .collect();
    aux.sort_unstable();
    StepPlan {
        diffusion,
        aux,
        claimed,
        noise_level: t,
    }
}

/// [`plan_step`] with every schedule's window confined to a noise `band` `(lo, hi)` — a Wan MoE
/// expert's own noise range (epic 2123 E8). An expert only ever trains at noise levels inside its
/// band, so an aux-only step lands in `window ∩ band`, and a loss whose window misses the band is
/// disabled for it (its claims fall through to diffusion). `t` is the band-sampled noise level; on
/// an aux-only step its position within the band (`u = (t − lo)/(hi − lo)`) is remapped into the
/// confined window, and a diffusion step keeps `t`. With the full band `(0, 1)` this is exactly
/// [`plan_step`]. Callers then apply [`StepPlan::without_skipped`] for the entry's unusable losses.
pub fn plan_in_band(
    schedules: &[AuxLossSchedule],
    key: impl Into<AltKey>,
    (lo, hi): (f32, f32),
    t: f32,
) -> StepPlan {
    let key = key.into();
    let confined: Vec<AuxLossSchedule> = schedules
        .iter()
        .map(|&s| {
            let (a, b) = (s.t_min.max(lo), s.t_max.min(hi));
            if a <= b {
                AuxLossSchedule {
                    t_min: a,
                    t_max: b,
                    ..s
                }
            } else {
                AuxLossSchedule { weight: 0.0, ..s }
            }
        })
        .collect();
    let u = if hi > lo {
        ((t - lo) / (hi - lo)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Whether the step is claimed depends only on `key`; a diffusion step keeps the sampled `t`.
    let claimed = plan_step(&confined, key, u);
    if claimed.diffusion {
        plan_step(&confined, key, t)
    } else {
        claimed
    }
}

/// Sum a step's present loss terms with `add`; `missing()` is the error for a step with neither
/// (a planning bug — every [`StepPlan`] trains at least one term).
pub fn combine_step_terms<T, E>(
    diffusion: Option<T>,
    aux: Option<T>,
    add: impl FnOnce(T, T) -> Result<T, E>,
    missing: impl FnOnce() -> E,
) -> Result<T, E> {
    match (diffusion, aux) {
        (Some(d), Some(a)) => add(d, a),
        (Some(d), None) => Ok(d),
        (None, Some(a)) => Ok(a),
        (None, None) => Err(missing()),
    }
}

/// Pre-load memory figures of one auxiliary training-time model (epic 2123 E7) — an x0 decoder or
/// a frozen perceptual model — computed from its config before anything loads. Backend-neutral:
/// the MLX and Candle perceptual kits and every trainer's preflight share this one definition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuxModelFootprint {
    /// Resident frozen weights.
    pub param_bytes: u64,
    /// One differentiable forward + backward at the training resolution.
    pub working_set_bytes: u64,
    /// The cached per-image reference.
    pub reference_bytes_per_image: u64,
}

/// The extra training memory the perceptual path adds, in bytes: the decoder (when any loss
/// decodes) plus every enabled loss — weights, working sets (summed: one traced backward holds the
/// step's aux terms together) and `images` cached references each.
pub fn perceptual_footprint_bytes(
    decoder: Option<AuxModelFootprint>,
    losses: &[AuxModelFootprint],
    images: usize,
) -> u64 {
    decoder
        .iter()
        .chain(losses.iter())
        .map(|f| f.param_bytes + f.working_set_bytes + f.reference_bytes_per_image * images as u64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E7: the shared pre-load estimator sums weights, working sets and per-image references.
    /// Mutation: drop the `× images` term ⇒ red.
    #[test]
    fn footprint_sums_models_and_references() {
        let dec = AuxModelFootprint {
            param_bytes: 10,
            working_set_bytes: 100,
            reference_bytes_per_image: 0,
        };
        let loss = AuxModelFootprint {
            param_bytes: 1_000,
            working_set_bytes: 10_000,
            reference_bytes_per_image: 7,
        };
        assert_eq!(
            perceptual_footprint_bytes(Some(dec), &[loss], 3),
            11_110 + 21
        );
        assert_eq!(perceptual_footprint_bytes(None, &[], 3), 0);
    }

    /// E8 (moved from the two Wan trainers): an aux-only step stays inside `window ∩ band`; a band
    /// that misses the window disables the loss (diffusion at the sampled `t`); the full band is
    /// exactly `plan_step`. Mutations: skip the window confinement ⇒ the low-band aux step lands
    /// above the band ⇒ red; remap with the raw `t` instead of its in-band position `u` ⇒ red;
    /// keep the miss-band loss enabled ⇒ red.
    #[test]
    fn plan_in_band_confines_aux_steps_to_the_expert_band() {
        let s = [sched(0.5, 0.6, 0.9, 2)];
        let low = plan_in_band(&s, 2, (0.0, 0.875), 0.4375);
        assert!(!low.diffusion && low.aux == vec![0], "{low:?}");
        // u = 0.5 of the band → the middle of [0.6, 0.875].
        assert!((low.noise_level - 0.7375).abs() < 1e-6, "{low:?}");
        let miss = plan_in_band(&s, 2, (0.95, 1.0), 0.97);
        assert!(miss.diffusion && miss.aux.is_empty(), "{miss:?}");
        assert_eq!(miss.noise_level, 0.97);
        for key in 1..=4 {
            for t in [0.1f32, 0.5, 0.95] {
                assert_eq!(plan_in_band(&s, key, (0.0, 1.0), t), plan_step(&s, key, t));
            }
        }
    }

    fn sched(weight: f32, t_min: f32, t_max: f32, every_n: u32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight,
            t_min,
            t_max,
            every_n,
        }
    }

    /// The two real visit orders a trainer walks: the single-bucket round-robin and a seeded
    /// two-bucket shuffle (each item twice per epoch, a fresh permutation each epoch).
    fn orders(n: usize) -> Vec<(&'static str, crate::train::BucketSchedule)> {
        use crate::train::{BucketSchedule, ResolutionBucket};
        let b = |resolution| ResolutionBucket {
            resolution,
            repeats: 1,
        };
        vec![
            ("round-robin", BucketSchedule::new(n, &[b(512)], 7)),
            ("shuffled", BucketSchedule::new(n, &[b(512), b(768)], 7)),
            (
                "shuffled-seed-2",
                BucketSchedule::new(n, &[b(512), b(768)], 2),
            ),
        ]
    }

    /// One `(epoch, item, diffusion)` per micro-step over `epochs` passes of `schedule`, keyed like
    /// the trainers' `AuxDriver` (`AuxAlternation::new(epoch_len, accum)`).
    fn walk(
        schedule: &crate::train::BucketSchedule,
        accum: u32,
        epochs: usize,
        s: &[AuxLossSchedule],
    ) -> Vec<(usize, usize, bool)> {
        let len = schedule.epoch_len();
        let alt = AuxAlternation::new(len, accum);
        (0..len * epochs)
            .map(|k| {
                let plan = plan_step(s, alt.key(k as u32 + 1), 0.5);
                (k / len, schedule.sample(k).0, plan.diffusion)
            })
            .collect()
    }

    /// One diffusion flag per optimizer window, asserting every micro-step of the window agrees.
    fn window_flags(steps: &[(usize, usize, bool)], accum: u32, what: &str) -> Vec<bool> {
        steps
            .chunks(accum as usize)
            .map(|w| {
                assert!(
                    w.iter().all(|x| x.2 == w[0].2),
                    "{what}: window {w:?} mixes step kinds"
                );
                w[0].2
            })
            .collect()
    }

    fn longest_run(flags: &[bool], kind: bool) -> usize {
        flags
            .split(|&f| f != kind)
            .map(<[bool]>::len)
            .max()
            .unwrap_or(0)
    }

    /// sc-2124 (the A/B defect): consecutive optimizer windows interleave like upstream
    /// ai-toolkit-perceptual (`SDTrainer.calculate_loss`: `step_num % 2 == 0` ⇒ diffusion, else
    /// depth — never two aux-only steps in a row), for N = 2 / 4 / 76 items, `every_n` 2 and 3,
    /// accumulation 1 and 2, round-robin and shuffled bucket orders — instead of whole epochs of
    /// aux-only steps. Diffusion runs stay short too (`≤ 2·every_n − 1` windows) and every
    /// `every_n`-ish window is aux. Mutations: key on the item's own visit count (the old
    /// `window_starts[image]`) ⇒ 76-window aux runs ⇒ red; the even-period drift with
    /// denominator `m` instead of `m + 1` (skips odd values) ⇒ two aux windows in a row ⇒ red.
    #[test]
    fn aux_windows_never_run_back_to_back() {
        for n in [2usize, 4, 76] {
            for every_n in [2u32, 3] {
                let s = [sched(0.1, 0.0, 1.0, every_n)];
                for accum in [1u32, 2] {
                    for (name, schedule) in orders(n) {
                        let what = format!("{name} N={n} every_n={every_n} accum={accum}");
                        let flags = window_flags(&walk(&schedule, accum, 12, &s), accum, &what);
                        assert!(
                            longest_run(&flags, false) <= 1,
                            "{what}: aux run {}",
                            longest_run(&flags, false)
                        );
                        assert!(
                            longest_run(&flags, true) < 2 * every_n as usize,
                            "{what}: diffusion run {}",
                            longest_run(&flags, true)
                        );
                        let aux = flags.iter().filter(|f| !**f).count() as f64;
                        let share = aux / flags.len() as f64;
                        assert!(
                            share >= 1.0 / (2.0 * every_n as f64)
                                && share <= 1.0 / every_n as f64 + 0.02,
                            "{what}: aux share {share}"
                        );
                    }
                }
            }
        }
    }

    /// sc-2124 (b), round-robin: every image gets both a diffusion and an aux step within any
    /// `every_n + 1` consecutive epochs (N = 2 / 4 / 76, `every_n` 2 / 3, accumulation 1 / 2) — no
    /// fixed order locks an image to one kind. (`every_n` epochs is impossible for `every_n = 2`
    /// and an even epoch: flipping every slot every epoch forces two aux windows in a row at
    /// alternate epoch boundaries; the drift costs one slot per epoch one extra epoch.) Mutations:
    /// drop the drift (`c(w) = w`) ⇒ an even N locks ⇒ red; key on the global step ignoring the
    /// period ⇒ red.
    #[test]
    fn every_image_gets_both_kinds_round_robin() {
        for n in [2usize, 4, 76] {
            for every_n in [2u32, 3] {
                let s = [sched(0.1, 0.0, 1.0, every_n)];
                for accum in [1u32, 2] {
                    let (_, schedule) = orders(n).remove(0);
                    let epochs = 4 * (every_n as usize + 1);
                    let steps = walk(&schedule, accum, epochs, &s);
                    let bound = every_n as usize + 1;
                    for item in 0..n {
                        for e0 in 0..=epochs - bound {
                            let got: Vec<bool> = steps
                                .iter()
                                .filter(|x| x.1 == item && (e0..e0 + bound).contains(&x.0))
                                .map(|x| x.2)
                                .collect();
                            assert!(
                                got.contains(&true) && got.contains(&false),
                                "N={n} every_n={every_n} accum={accum}: image {item} epochs \
                                 {e0}..{} got {got:?}",
                                e0 + bound
                            );
                        }
                    }
                }
            }
        }
    }

    /// sc-2124 (b), seeded bucket shuffle: each sample lands on a random slot of the interleaved
    /// window pattern, so every image gets both kinds early (here within `3·every_n` epochs, for
    /// two seeds, N = 2 / 4 / 76, `every_n` 2 / 3, accumulation 1 / 2). Mutation: claim on every
    /// window ⇒ no diffusion step ⇒ red.
    #[test]
    fn every_image_gets_both_kinds_shuffled_buckets() {
        for n in [2usize, 4, 76] {
            for every_n in [2u32, 3] {
                let s = [sched(0.1, 0.0, 1.0, every_n)];
                for accum in [1u32, 2] {
                    for (name, schedule) in orders(n).into_iter().skip(1) {
                        let steps = walk(&schedule, accum, 3 * every_n as usize, &s);
                        for item in 0..n {
                            let got: Vec<bool> =
                                steps.iter().filter(|x| x.1 == item).map(|x| x.2).collect();
                            assert!(
                                got.contains(&true) && got.contains(&false),
                                "{name} N={n} every_n={every_n} accum={accum}: image {item} \
                                 got {got:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The general claim behind the two tests above, for every period `m ≤ 96` and `every_n` 2..=5:
    /// no two claimed windows are adjacent, and every slot of the period is claimed and unclaimed
    /// within any `every_n` (`3` for `every_n = 2` with an even period) consecutive periods.
    /// Mutations: the even-period drift over `m` (not `m + 1`) ⇒ adjacent claims ⇒ red; `s = 0`
    /// always ⇒ a period sharing a factor with `every_n` locks its slots ⇒ red.
    #[test]
    fn claims_interleave_and_cycle_every_slot() {
        for m in 1u32..=96 {
            for n in 2u32..=5 {
                let bound = if n == 2 && m % 2 == 0 { 3 } else { n };
                let periods = 3 * bound + 3;
                let claim = |w: u32| {
                    AltKey {
                        window: w,
                        period: m,
                    }
                    .claims(n)
                };
                for w in 1..m * periods {
                    assert!(
                        !(claim(w) && claim(w + 1)),
                        "m={m} n={n}: windows {w} and {} both aux",
                        w + 1
                    );
                }
                for q in 1..=m {
                    for p0 in 0..=periods - bound {
                        let got: Vec<bool> = (p0..p0 + bound).map(|p| claim(p * m + q)).collect();
                        assert!(
                            got.contains(&true) && got.contains(&false),
                            "m={m} n={n}: slot {q} periods {p0}.. got {got:?}"
                        );
                    }
                }
            }
        }
    }

    /// The plan is keyed per optimizer window — with accumulation 2 (and Wan's `accum · experts`
    /// window) every micro-step of a window shares the diffusion flag (never an averaged
    /// diffusion + aux update), and windows still alternate. Mutation: key on the micro-step
    /// (`window: step`) ⇒ red.
    #[test]
    fn accumulation_windows_share_one_step_kind() {
        let s = [sched(0.1, 0.0, 1.0, 2)];
        for n in [1usize, 2, 3, 4] {
            for window in [2u32, 4] {
                let alt = AuxAlternation::new(n, window);
                let flags: Vec<bool> = (1..=16 * window)
                    .map(|step| plan_step(&s, alt.key(step), 0.5).diffusion)
                    .collect();
                for w in flags.chunks(window as usize) {
                    assert!(
                        w.iter().all(|f| *f == w[0]),
                        "N={n} window={window}: {w:?} mixes step kinds ({flags:?})"
                    );
                }
                assert!(
                    flags.contains(&true) && flags.contains(&false),
                    "N={n}: {flags:?}"
                );
            }
        }
    }

    /// Alternation with a single image: diffusion, aux, diffusion, … (strict alternation), and on an
    /// aux step the diffusion term is off. Mutation: `diffusion: true` ⇒ red.
    #[test]
    fn strict_alternation_pattern() {
        let s = [sched(0.1, 0.0, 1.0, 2)];
        let pattern: Vec<bool> = (1..=6)
            .map(|key| plan_step(&s, key, 0.5).diffusion)
            .collect();
        assert_eq!(pattern, vec![true, false, true, false, true, false]);
        for key in 1..=6 {
            let p = plan_step(&s, key, 0.5);
            assert_eq!(p.aux.is_empty(), p.diffusion, "key {key}: {p:?}");
        }
    }

    /// An aux-only step trains inside its window (the sampled level is remapped, so the claim never
    /// depends on the timestep); a diffusion step keeps the sampled level. Mutation: drop the remap
    /// (`t = raw_t`) ⇒ an out-of-window claimed step trains no aux term ⇒ red.
    #[test]
    fn aux_steps_are_remapped_into_the_window() {
        let s = [sched(0.1, 0.2, 0.6, 2)];
        for raw in [0.0f32, 0.05, 0.5, 0.95, 1.0] {
            let p = plan_step(&s, 2, raw);
            assert!(!p.diffusion && p.aux == vec![0], "{raw}: {p:?}");
            assert!((0.2..=0.6).contains(&p.noise_level), "{raw}: {p:?}");
            let d = plan_step(&s, 1, raw);
            assert_eq!(d, StepPlan::diffusion_only(raw));
        }
    }

    #[test]
    fn sum_mode_off_and_mixed_losses() {
        let sum = [sched(0.1, 0.0, 0.5, 1)];
        let p = plan_step(&sum, 7, 0.3);
        assert!(p.diffusion && p.aux == vec![0]);
        assert_eq!(plan_step(&sum, 7, 0.8), StepPlan::diffusion_only(0.8));
        let off = [AuxLossSchedule::OFF];
        for key in 1..=4 {
            assert_eq!(plan_step(&off, key, 0.5), StepPlan::diffusion_only(0.5));
        }
        let both = [sched(0.1, 0.0, 1.0, 2), sched(0.2, 0.0, 1.0, 1)];
        let p = plan_step(&both, 2, 0.5);
        assert!(!p.diffusion && p.aux == vec![0, 1] && p.claimed == vec![0]);
    }

    /// A skipped image on a claimed step falls back to the diffusion loss (documented choice);
    /// summed losses still ride along. Mutation: drop the fallback ⇒ an empty aux-only step ⇒ red.
    #[test]
    fn skipped_claimers_fall_back_to_diffusion() {
        let s = [sched(0.1, 0.0, 1.0, 2), sched(0.2, 0.0, 1.0, 1)];
        let p = plan_step(&s, 2, 0.5).without_skipped(|i| i == 0);
        assert!(p.diffusion, "{p:?}");
        assert_eq!(p.aux, vec![1]);
        let p = plan_step(&s[..1], 2, 0.5).without_skipped(|_| true);
        assert!(p.diffusion && p.aux.is_empty(), "{p:?}");
        // Not skipped ⇒ unchanged.
        let p = plan_step(&s, 2, 0.5);
        assert_eq!(p.clone().without_skipped(|_| false), p);
    }

    #[test]
    fn combine_requires_a_term() {
        let add = |a: i32, b: i32| -> Result<i32, String> { Ok(a + b) };
        assert_eq!(
            combine_step_terms(Some(1), Some(2), add, || "x".to_string()),
            Ok(3)
        );
        assert_eq!(
            combine_step_terms(Some(1), None, add, || "x".to_string()),
            Ok(1)
        );
        assert_eq!(
            combine_step_terms(None, Some(2), add, || "x".to_string()),
            Ok(2)
        );
        assert!(combine_step_terms::<i32, String>(None, None, add, || "x".to_string()).is_err());
    }
}
