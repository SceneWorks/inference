//! Backend-neutral **auxiliary-loss step policy** (epic 2123 E8, sc-2125) — which loss terms a
//! training micro-step trains. Pure policy, so the MLX and Candle trainers share one copy:
//!
//! - [`AuxAlternation`] turns `(micro-step, image)` into the step's **alternation key**: how many
//!   optimizer-update windows the window's first image has started (its own visit count when
//!   `gradient_accumulation == 1`), shared by the window's other micro-steps. Keying on the image (not the global step)
//!   means every image alternates between diffusion and aux steps whatever order the trainer visits
//!   images in — round-robin, shuffled, or bucketed — and the period can never lock an image to one
//!   step kind. Keying per window means gradient accumulation never averages a diffusion and an aux
//!   micro-step into one update.
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

/// Per-image alternation keys (see the module docs). Feed it every micro-step in order.
#[derive(Clone, Debug)]
pub struct AuxAlternation {
    window_starts: Vec<u32>,
    accum: u32,
    window_key: u32,
}

impl AuxAlternation {
    /// For a dataset of `images` items and `gradient_accumulation` micro-steps per update.
    pub fn new(images: usize, gradient_accumulation: u32) -> Self {
        Self {
            window_starts: vec![0; images],
            accum: gradient_accumulation.max(1),
            window_key: 0,
        }
    }

    /// Record 1-based micro-step `step` visiting dataset item `image` and return the step's key: at
    /// the first micro-step of an optimizer window, the number of windows `image` has now started
    /// (1 the first time); the window's later micro-steps reuse it. Call for every micro-step, in order
    /// (replay the skipped prefix on resume).
    pub fn key(&mut self, step: u32, image: usize) -> u32 {
        if (step.max(1) - 1).is_multiple_of(self.accum) {
            let starts = &mut self.window_starts[image];
            *starts += 1;
            self.window_key = *starts;
        }
        self.window_key
    }
}

/// The plan for alternation `key` at sampled noise level `raw_t`:
///
/// - a loss **claims** the step when it is enabled, `every_n ≥ 2` and `key % every_n == 0`;
/// - any claim ⇒ **aux-only** step: the noise level is remapped into the claiming windows'
///   intersection (the first claimer's window when they do not intersect), and the claiming losses
///   whose window holds it contribute, plus every enabled `every_n == 1` loss whose window holds it;
/// - no claim ⇒ diffusion step at `raw_t`, plus every enabled `every_n == 1` loss in window.
pub fn plan_step(schedules: &[AuxLossSchedule], key: u32, raw_t: f32) -> StepPlan {
    let claiming: Vec<usize> = schedules
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_enabled() && s.every_n >= 2 && key.is_multiple_of(s.every_n))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(weight: f32, t_min: f32, t_max: f32, every_n: u32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight,
            t_min,
            t_max,
            every_n,
        }
    }

    /// Each image's step kinds over `steps` micro-steps visiting `order(step)`.
    fn kinds(
        n: usize,
        accum: u32,
        steps: u32,
        s: &[AuxLossSchedule],
        order: impl Fn(u32) -> usize,
    ) -> (Vec<Vec<bool>>, Vec<bool>) {
        let mut alt = AuxAlternation::new(n, accum);
        let mut per_image = vec![Vec::new(); n];
        let mut flags = Vec::new();
        for step in 1..=steps {
            let image = order(step);
            let plan = plan_step(s, alt.key(step, image), 0.5);
            per_image[image].push(plan.diffusion);
            flags.push(plan.diffusion);
        }
        (per_image, flags)
    }

    /// Review blocker: with round-robin image order and `every_n` dividing N, a global-step key
    /// locks each image to one step kind. Keying on the image's own visits gives every image both a
    /// diffusion and an aux step within 2·N steps — for round-robin and for a shuffled order.
    /// Mutation: key on the global step (`self.window_key = step`) ⇒ N = 2 / N = 4 lock ⇒ red.
    #[test]
    fn every_image_gets_both_step_kinds_for_any_order() {
        let s = [sched(0.1, 0.0, 1.0, 2)];
        for n in [2usize, 4] {
            let steps = 2 * n as u32;
            let shuffled = |step: u32| {
                // A fixed per-epoch permutation (reversed on odd epochs).
                let epoch = (step - 1) / n as u32;
                let pos = ((step - 1) % n as u32) as usize;
                if epoch.is_multiple_of(2) {
                    pos
                } else {
                    n - 1 - pos
                }
            };
            for (name, per_image) in [
                (
                    "round-robin",
                    kinds(n, 1, steps, &s, |st| ((st - 1) as usize) % n).0,
                ),
                ("shuffled", kinds(n, 1, steps, &s, shuffled).0),
            ] {
                for (i, k) in per_image.iter().enumerate() {
                    assert!(
                        k.contains(&true) && k.contains(&false),
                        "{name} N={n}: image {i} got {k:?}"
                    );
                }
            }
        }
    }

    /// Review major: the plan is keyed per optimizer-update window — with accumulation 2 both
    /// micro-steps of a window share the diffusion flag (never an averaged diffusion+aux update),
    /// and windows still alternate. Mutation: update `window_key` on every micro-step ⇒ red.
    #[test]
    fn accumulation_windows_share_one_step_kind() {
        let s = [sched(0.1, 0.0, 1.0, 2)];
        for n in [1usize, 2, 3, 4] {
            let (_, flags) = kinds(n, 2, 16, &s, |st| ((st - 1) as usize) % n);
            for w in flags.chunks(2) {
                assert_eq!(
                    w[0], w[1],
                    "N={n}: window {w:?} mixes step kinds ({flags:?})"
                );
            }
            assert!(
                flags.contains(&true) && flags.contains(&false),
                "N={n}: {flags:?}"
            );
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
