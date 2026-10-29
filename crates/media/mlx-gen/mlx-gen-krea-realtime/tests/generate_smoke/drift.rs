//! The S18 scene-drift metric (descriptor, trend and excursion scoring) and its weights-free
//! gates.

use super::*;

// ---------------------------------------------------------------------------------------------
// sc-15127 (S18) — long-clip coherence under the bounded KV window
// ---------------------------------------------------------------------------------------------
//
// The question is NOT "does the clip change over time" — a video is supposed to. It is "does the
// clip's *established scene* survive the KV window rolling out from under it". So the metric is
// built to be blind to ordinary motion and sensitive to a one-way excursion:
//
//   * The descriptor is a set of **scene-level** statistics that ordinary camera/subject motion
//     perturbs in a bounded, oscillating way, and that a degenerating AR loop moves one way. It spans
//     three axes deliberately, because a single-axis descriptor has large blind spots (see
//     [`the_drift_metric_catches_the_plausible_ar_failure_shapes`]): **tone** (luma mean, contrast),
//     **colour** (saturation and the two opponent-colour means, which a hue rotation moves and a
//     luma/saturation descriptor cannot see), and **space** (the spread of a 5x5 grid of block luma
//     means, which a localized subject collapse moves and every global moment cannot see).
//   * The **baseline** is the pre-roll segment: the output frames produced before the first KV
//     eviction. Those frames saw the full history, so their frame-to-frame fluctuation IS the
//     ordinary-motion scale for this clip, measured on this clip.
//   * The verdict statistic is **trend AND excursion**, both gated. The OLS trend catches a ramp; it
//     is structurally blind to a *plateau* (the loop settling into a degenerate attractor, which is
//     the most likely AR failure), which is exactly what the excursion catches. The excursion's own
//     failure mode — an oscillating scene whose baseline sits at an extreme of its cycle — is handled
//     by suppressing it when it is small relative to the pre-roll fluctuation scale (`z`), not by
//     discarding the statistic.
//   * The **per-roll-bucket** table is printed alongside, so a non-linear collapse is visible rather
//     than averaged into a slope.
//
// **What is NOT available as a control, and why the budget is absolute.** A bounded window over a long
// clip evicts *by definition* — so there is no such thing as a zero-roll run at the shipped window and
// the shipped clip length. The checkpoint's global window (row E) is *not* that null: changing
// `local_attn_size` changes the attention mask, hence the sampling trajectory, hence the video. Rows
// are not the same clip; they are the same *prompt and seed* under different attention. Row E is
// retained as an out-of-regime reference (it is what the Mac bound replaces) and is explicitly **not**
// a motion floor. What remains valid:
//   * a **within-regime three-dose response**: rows A/D/F use the same bounded local-attention path
//     at windows 6/15/30 (13/10/5 rolls). The decision statistic is a paired-seed slope, not an
//     endpoint ranking.
//   * a **within-regime zero-eviction rate finding** (row Z): the shipped window on a clip short
//     enough that it never rolls. Its short-segment slope is retained, but the recorded value is
//     higher than row A's long-clip rate and over-predicts row A when extrapolated, so it does not
//     establish a rate floor.
//   * an **absolute budget** ([`DRIFT_BUDGET`]), justified against the metric's measured floor and its
//     measured response to the plausible failure shapes, and gated in CI on both.

/// Grid resolution for the spatial component of [`scene_descriptor`]: `SPATIAL_GRID`^2.
pub(super) const SPATIAL_GRID: usize = 5;

/// Number of components in [`scene_descriptor`].
pub(super) const N_DESC: usize = 6;

/// Per-frame scene descriptor in 0..255 units.
///
/// Three axes, because each covers a failure shape the others are blind to:
/// * **tone** — `luma-mean`, `contrast` (luma std).
/// * **colour** — `saturation` (max−min chroma), and the two opponent means `R−G` and `B−(R+G)/2`.
///   A hue rotation at constant luma and constant saturation moves *only* the opponent pair; without
///   them the descriptor cannot see a colour cast developing.
/// * **space** — `spatial-sd`, the standard deviation of a `SPATIAL_GRID`^2 grid of block luma means.
///   A subject collapsing in the centre while the surround compensates leaves every *global* moment
///   unchanged; it moves this one hard.
///
/// Deliberately still statistics, not a per-pixel comparison against frame 0 — that measures motion,
/// which is exactly the signal that must NOT drive this conclusion.
pub(super) fn scene_descriptor(f: &Image) -> [f64; N_DESC] {
    let w = (f.width as usize).max(1);
    let h = (f.height as usize).max(1);
    let n = (f.pixels.len() / 3).max(1);
    let (mut sum, mut sumsq, mut sat, mut rg, mut by) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut blk = [0.0f64; SPATIAL_GRID * SPATIAL_GRID];
    let mut blk_n = [0usize; SPATIAL_GRID * SPATIAL_GRID];
    for (i, px) in f.pixels.chunks_exact(3).enumerate() {
        let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        sum += y;
        sumsq += y * y;
        sat += r.max(g).max(b) - r.min(g).min(b);
        rg += r - g;
        by += b - 0.5 * (r + g);
        let bx = (i % w) * SPATIAL_GRID / w;
        let byi = (i / w).min(h - 1) * SPATIAL_GRID / h;
        let bi = byi.min(SPATIAL_GRID - 1) * SPATIAL_GRID + bx.min(SPATIAL_GRID - 1);
        blk[bi] += y;
        blk_n[bi] += 1;
    }
    let mean = sum / n as f64;
    let var = (sumsq / n as f64 - mean * mean).max(0.0);
    let block_means: Vec<f64> = blk
        .iter()
        .zip(blk_n.iter())
        .filter(|(_, &c)| c > 0)
        .map(|(&s, &c)| s / c as f64)
        .collect();
    [
        mean,
        var.sqrt(),
        sat / n as f64,
        rg / n as f64,
        by / n as f64,
        std_f64(&block_means),
    ]
}

/// Names of the [`scene_descriptor`] components, for the printed table.
pub(super) const DESCRIPTOR_NAMES: [&str; N_DESC] = [
    "luma-mean",
    "contrast",
    "saturation",
    "opp-R-G",
    "opp-B-Y",
    "spatial-sd",
];

/// The absolute one-way drift budget, in 0..255 units, for both the trend and the z-gated excursion.
///
/// **Absolute, not derived from a control row** — see the module comment: no zero-roll run exists at
/// the shipped window and the shipped clip length, so there is no valid within-regime floor to subtract.
/// The number is instead pinned from both sides, and both sides are gated in CI by
/// [`the_drift_metric_separates_drift_from_ordinary_motion`] and
/// [`the_drift_metric_catches_the_plausible_ar_failure_shapes`]:
/// * **from below** — the metric reads a violently moving clip and a violently jittery clip at
///   ≤ 5/255;
/// * **from above** — every plausible AR failure shape (linear wash-out ramp, plateauing wash-out,
///   localized subject collapse, hue rotation) scores above it, several of them by 2–6×.
///
/// **Both sides are synthetic, and that is a real limit.** An earlier revision of this comment also
/// claimed the within-regime zero-eviction row `Z` "bounds this content's own rate". It does not, and
/// the sweep now computes the comparison instead of asserting it ([`S18Sweep::rate_floor_clause`],
/// reported in the verdict): row Z's post segment is only 12 output frames, its measured slope
/// (29.63/100f) is *higher* than the shipped row's over 156 frames (15.56/100f), and extrapolating it
/// across the long clip predicts ~46/255 — roughly 1.9× more drift than row A actually shows. A
/// 12-frame OLS slope is dominated by ordinary frame-to-frame motion and does not extrapolate. Row Z
/// also cannot be made longer: the shipped window evicts as soon as a clip passes 6 latent frames, so
/// **no longer zero-eviction row exists at the shipped window**. There is therefore no measured
/// same-content floor, and "past the budget" means "past an absolute number bracketed by synthetic
/// controls", not "past a measured same-content baseline".
///
/// Those numbers are 640×384. sc-17324 measured row Z at **832×480** — the shipping bucket — for the
/// first time and it comes out the same way, harder: Z runs at 38.29/100f against row A's 18.82,
/// A/Z = 0.49. So the missing floor is not an artifact of the smaller bucket. See
/// [`MEASURED_832_SC17324`] and [`the_sc17324_832_sweep_reports_its_rate_comparison`].
///
/// The two sides are measured, not asserted, and they bracket a **narrow** range:
/// * floor — motion and jitter both score **2.81/255**;
/// * ceiling — the *weakest* of the four failure shapes (a 180° hue rotation at constant luma and
///   constant chroma magnitude, whose scene-mean chroma partially cancels across pixels) scores
///   **11.37/255**. Everything else scores 21–80.
///
/// So the budget must lie in `(2.81, 11.37)`. 8/255 ≈ 3.1% of the range sits at 2.85× the measured
/// floor and 0.70× the weakest detection, and is about where a *sustained one-way* shift in scene
/// luma, colour cast or spatial structure starts to read as the picture changing rather than the
/// subject moving. Both margins are asserted in the two gates above, so this cannot silently rot into
/// a number that admits gross drift or rejects ordinary motion.
pub(super) const DRIFT_BUDGET: f64 = 8.0;

/// Minimum `|z|` for an excursion to count. Below this the tail's offset from the baseline is inside
/// the clip's own pre-roll fluctuation, i.e. it is where a moving scene happens to be in its cycle, and
/// gating on it would flag every real clip. (Measured on the synthetic motion control, whose raw
/// excursion is large and whose `z` is not.)
pub(super) const EXCURSION_Z_MIN: f64 = 3.0;

pub(super) fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

pub(super) fn std_f64(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean_f64(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Ordinary-least-squares slope of `y` against its own index, scaled to **units per 100 samples**.
pub(super) fn slope_per_100(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let n = y.len() as f64;
    let xbar = (n - 1.0) / 2.0;
    let ybar = mean_f64(y);
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - xbar;
        num += dx * (v - ybar);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * num / den
    }
}

/// The drift summary for one descriptor component of one clip.
#[derive(Clone)]
pub(super) struct ComponentDrift {
    /// The total *one-way* movement the rolling portion of the clip accumulates, in 0..255 units: the
    /// OLS trend over the post-roll segment × that segment's length.
    ///
    /// Catches a ramp. **Structurally blind to a plateau** — a loop that washes out by frame 54 and
    /// then holds has a small trend and a large [`excursion`](Self::excursion), which is why both are
    /// gated.
    pub(super) trend: f64,
    /// Signed offset of the clip's final segment from the pre-roll baseline, in 0..255 units.
    ///
    /// Catches a plateau. Its own failure mode is an oscillating (i.e. ordinary, moving) scene whose
    /// baseline window happens to sit at an extreme of its cycle: that shows a large offset while going
    /// nowhere — measured at +11.7/255 on the synthetic motion control. That is handled by
    /// [`gated_excursion`](Self::gated_excursion), not by discarding the statistic.
    pub(super) excursion: f64,
    /// [`excursion`](Self::excursion) in units of the pre-roll frame-to-frame fluctuation.
    pub(super) z: f64,
    /// OLS slope over the post-roll segment, per 100 output frames. **Length-independent**, so this is
    /// the statistic that can be compared across rows of different clip length (row Z).
    pub(super) slope_post: f64,
    /// OLS slope over the pre-roll baseline segment, per 100 output frames.
    pub(super) slope_pre: f64,
    /// Per-bucket signed excursion from the baseline (bucket = one window-roll's worth of frames).
    /// Printed so a non-linear collapse is visible rather than averaged into a single slope.
    pub(super) buckets: Vec<f64>,
}

impl ComponentDrift {
    /// [`excursion`](Self::excursion), suppressed when it is inside the clip's own pre-roll
    /// fluctuation scale (see [`EXCURSION_Z_MIN`]).
    pub(super) fn gated_excursion(&self) -> f64 {
        if self.z.abs() < EXCURSION_Z_MIN {
            0.0
        } else {
            self.excursion
        }
    }
}

/// Score one clip's drift. `pre_len` is the number of leading output frames generated *before* the
/// first KV eviction (the baseline), `bucket` the frames one window roll contributes.
pub(super) fn score_drift(series: &[f64], pre_len: usize, bucket: usize) -> ComponentDrift {
    let pre_len = pre_len.min(series.len()).max(1);
    let pre = &series[..pre_len];
    let base = mean_f64(pre);
    // Floor the motion scale so a pathologically still baseline cannot manufacture an infinite z.
    // 0.25/255 is well under any perceptible change and under this decode's own noise floor.
    let scale = std_f64(pre).max(0.25);
    let tail_start = series.len().saturating_sub(pre_len).max(pre_len);
    let tail = &series[tail_start.min(series.len().saturating_sub(1))..];
    let excursion = mean_f64(tail) - base;
    let mut buckets = Vec::new();
    let bucket = bucket.max(1);
    let mut i = pre_len;
    while i < series.len() {
        let end = (i + bucket).min(series.len());
        buckets.push(mean_f64(&series[i..end]) - base);
        i = end;
    }
    let post = &series[pre_len.min(series.len())..];
    let slope_post = slope_per_100(post);
    ComponentDrift {
        trend: slope_post * post.len() as f64 / 100.0,
        excursion,
        z: excursion / scale,
        slope_post,
        slope_pre: slope_per_100(pre),
        buckets,
    }
}

/// Score a whole clip: one [`ComponentDrift`] per [`scene_descriptor`] component.
pub(super) fn score_clip(
    frames: &[Image],
    pre_len: usize,
    bucket: usize,
) -> [ComponentDrift; N_DESC] {
    let d: Vec<[f64; N_DESC]> = frames.iter().map(scene_descriptor).collect();
    std::array::from_fn(|c| {
        let series: Vec<f64> = d.iter().map(|x| x[c]).collect();
        score_drift(&series, pre_len, bucket)
    })
}

/// Largest absolute one-way component trend, in 0..255 units.
pub(super) fn worst_trend(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter().map(|c| c.trend.abs()).fold(0.0, f64::max)
}

/// Largest absolute **z-gated** component excursion, in 0..255 units.
pub(super) fn worst_excursion(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter()
        .map(|c| c.gated_excursion().abs())
        .fold(0.0, f64::max)
}

/// Largest absolute component slope over the rolling portion, per 100 output frames. This is the only
/// statistic comparable across clips of *different length* (row Z is 24 output frames, row A is 180).
pub(super) fn worst_slope(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter().map(|c| c.slope_post.abs()).fold(0.0, f64::max)
}

/// The clip's headline drift figure: worst trend **or** worst gated excursion, whichever is larger.
/// Gating on the max is what makes the budget a "trend AND excursion" gate.
pub(super) fn worst_drift(d: &[ComponentDrift; N_DESC]) -> f64 {
    worst_trend(d).max(worst_excursion(d))
}

/// Which [`DESCRIPTOR_NAMES`] component produced [`worst_drift`]. Recorded per cell so "what mode is
/// this clip failing in?" is answerable from the table instead of from prose.
pub(super) fn worst_component(d: &[ComponentDrift; N_DESC]) -> &'static str {
    let mut best = (0usize, -1.0f64);
    for (i, c) in d.iter().enumerate() {
        let v = c.trend.abs().max(c.gated_excursion().abs());
        if v > best.1 {
            best = (i, v);
        }
    }
    DESCRIPTOR_NAMES[best.0]
}

pub(super) fn print_drift(label: &str, d: &[ComponentDrift; N_DESC]) {
    for (name, c) in DESCRIPTOR_NAMES.iter().zip(d.iter()) {
        println!(
            "    {label} {name:<11}: TREND {:+7.2}/255 | excursion {:+7.2} (z {:+6.2}, gated \
             {:+7.2}), slope pre {:+7.3} -> post {:+7.3} per 100f",
            c.trend,
            c.excursion,
            c.z,
            c.gated_excursion(),
            c.slope_pre,
            c.slope_post
        );
        let b: Vec<String> = c.buckets.iter().map(|v| format!("{v:+.2}")).collect();
        println!("      per-roll buckets: [{}]", b.join(", "));
    }
}

/// Apply a per-frame pixel transform, for building the synthetic drift stimuli.
pub(super) fn map_frames(
    n: usize,
    w: usize,
    h: usize,
    mut f: impl FnMut(usize, f64, &mut Image),
) -> Vec<Image> {
    (0..n)
        .map(|i| {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let mut img = smooth_frame(w, h, i);
            f(i, t, &mut img);
            img
        })
        .collect()
}

/// **CI gate for the metric itself.** A metric that cannot tell drift from motion is not evidence, so
/// this drives [`score_clip`] on two synthetic clips with the *same* total excursion budget: one that
/// oscillates (ordinary motion) and one that ramps one way (drift). The gate is that the drift score
/// separates them — it is red if the descriptor, the baseline, or the bucketing stops discriminating.
///
/// It also pins the **lower** side of [`DRIFT_BUDGET`]: motion and noise must both land well under it,
/// or the budget is not a budget.
#[test]
fn the_drift_metric_separates_drift_from_ordinary_motion() {
    const W: usize = 128;
    const H: usize = 96;
    const N: usize = 180;
    const PRE: usize = 24;
    const BUCKET: usize = 12;

    // Ordinary motion: a moving scene whose global statistics oscillate. Amplitude is deliberately
    // LARGE (the frames swing right across the range) so this is not passing by being static.
    let motion: Vec<Image> = (0..N).map(|i| smooth_frame(W, H, i)).collect();
    let m = score_clip(&motion, PRE, BUCKET);
    print_drift("motion", &m);

    // Drift: the same clip with a one-way wash-out applied — luma pulled up and saturation pulled
    // down, both proportional to the frame index, i.e. exactly the bounded-window AR failure mode.
    let drift = map_frames(N, W, H, |_, t, f| {
        for px in f.pixels.chunks_exact_mut(3) {
            let g = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            for c in px.iter_mut() {
                // Pull each channel toward a rising grey: desaturates and brightens together.
                let v = *c as f64;
                *c = (v + t * 0.85 * (g + 60.0 - v)).clamp(0.0, 255.0) as u8;
            }
        }
    });
    let d = score_clip(&drift, PRE, BUCKET);
    print_drift("drift", &d);

    let m_worst = worst_drift(&m);
    let d_worst = worst_drift(&d);
    assert!(
        m_worst * 2.5 < DRIFT_BUDGET,
        "the ordinary-motion clip scored {m_worst:.2}/255 against a {DRIFT_BUDGET:.2} budget — that \
         is under a 2.5x margin, so the budget is not comfortably above the metric's own motion \
         floor and a no-drift verdict would be resting on noise"
    );
    assert!(
        d_worst > 25.0,
        "the one-way wash-out clip scored only {d_worst:.2}/255 — the metric cannot see the drift \
         mode it exists to detect, so a green long-clip run would prove nothing"
    );
    assert!(
        d_worst > m_worst * 4.0,
        "drift {d_worst:.2}/255 vs motion {m_worst:.2}/255 — the separation is too small for the \
         long-clip verdict to rest on"
    );
    // The RAW head->tail offset is explicitly not gated: on this very stimulus it reads the moving clip
    // as drifting harder than several plausible real-drift budgets. That is what `EXCURSION_Z_MIN`
    // exists for, and this pins both halves — the raw offset is big, and the gated one is not.
    let m_offset = m.iter().map(|c| c.excursion.abs()).fold(0.0, f64::max);
    assert!(
        m_offset > worst_trend(&m) * 3.0,
        "the moving control's raw offset ({m_offset:.2}/255) is no longer much larger than its trend \
         ({:.2}) — this stimulus has stopped demonstrating why the raw offset is unusable, so the \
         z-gate is no longer evidenced",
        worst_trend(&m)
    );
    assert!(
        worst_excursion(&m) < DRIFT_BUDGET,
        "the z-gate let the moving control's offset through at {:.2}/255 — EXCURSION_Z_MIN is too \
         low to keep ordinary motion out of the verdict",
        worst_excursion(&m)
    );
    // A second, INDEPENDENT way the metric could be fooled: per-frame noise. A jittery clip has huge
    // frame-to-frame variance and goes nowhere, and a metric that reported that as drift would flag
    // every real clip. Same stimulus, plus a large zero-mean per-frame brightness jitter.
    let mut lcg = 0x2545_F491_4F6C_DD1Du64;
    let jitter = map_frames(N, W, H, |_, _, f| {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((lcg >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 60.0;
        for c in f.pixels.iter_mut() {
            *c = (*c as f64 + j).clamp(0.0, 255.0) as u8;
        }
    });
    let j = score_clip(&jitter, PRE, BUCKET);
    print_drift("jitter", &j);
    let j_worst = worst_drift(&j);
    assert!(
        j_worst < DRIFT_BUDGET,
        "a zero-mean per-frame jitter scored {j_worst:.2}/255 — the metric is reading noise as \
         drift, so it cannot support a drift verdict either"
    );
    // ...and that jitter really was violent, so this control is not passing by being tame.
    let j_delta = mean_f64(
        &frame_stats(&jitter)
            .iter()
            .skip(1)
            .map(|s| s.delta)
            .collect::<Vec<_>>(),
    );
    let m_delta = mean_f64(
        &frame_stats(&motion)
            .iter()
            .skip(1)
            .map(|s| s.delta)
            .collect::<Vec<_>>(),
    );
    assert!(
        j_delta > m_delta * 2.0,
        "the jitter control moves {j_delta:.2}/255 per frame against the plain motion clip's \
         {m_delta:.2} — it is not the noisy stimulus it claims to be"
    );
}

/// **CI gate for the metric's blind spots.** The linear full-amplitude ramp above is the shape an OLS
/// trend is *maximally* good at; passing on it is not a sensitivity floor. This drives [`score_clip`]
/// on three failure shapes an autoregressive loop is at least as likely to produce, each of which
/// defeats a single-statistic, single-axis descriptor:
///
/// 1. **Plateauing wash-out** — the loop settles into a degenerate attractor: brightness ramps for a
///    third of the clip and then holds. The OLS trend over the post-roll segment is structurally small;
///    only the excursion sees it. (Measured at TREND ≈ 11.9 against excursion ≈ 32.)
/// 2. **Localized subject collapse** — the centre of the frame goes dark while the surround brightens
///    to hold the global mean. Every *global* moment is unmoved; only the spatial component sees it.
/// 3. **Hue rotation at constant luma and saturation** — a colour cast develops. Luma, contrast and
///    max−min saturation are all unmoved; only the opponent-colour components see it.
///
/// Each must exceed [`DRIFT_BUDGET`], and — because "it scored high on *something*" is not enough —
/// each is also asserted to be caught by the *specific* statistic/axis that exists for it, so the
/// coverage cannot silently rot away.
#[test]
fn the_drift_metric_catches_the_plausible_ar_failure_shapes() {
    const W: usize = 128;
    const H: usize = 96;
    const N: usize = 180;
    const PRE: usize = 24;
    const BUCKET: usize = 12;

    // 1. Plateau: +40/255 wash-out that starts at the first eviction (frame `PRE` — a real AR drift
    //    cannot begin before the window first rolls), is fully in place by frame 54, and then holds.
    let plateau = map_frames(N, W, H, |i, _, f| {
        let a = 40.0 * ((i.saturating_sub(PRE) as f64 / (54 - PRE) as f64).min(1.0));
        for c in f.pixels.iter_mut() {
            *c = (*c as f64 + a).clamp(0.0, 255.0) as u8;
        }
    });
    let p = score_clip(&plateau, PRE, BUCKET);
    print_drift("plateau", &p);
    assert!(
        worst_drift(&p) > DRIFT_BUDGET,
        "a permanent +40/255 wash-out that plateaus scored {:.2}/255, inside the {DRIFT_BUDGET:.2} \
         budget — the metric is blind to the most likely AR failure shape",
        worst_drift(&p)
    );
    assert!(
        worst_excursion(&p) > worst_trend(&p),
        "the plateau was caught by the TREND ({:.2}) not the excursion ({:.2}) — this stimulus has \
         stopped demonstrating why the excursion is gated, so MAJOR-4 is no longer evidenced",
        worst_trend(&p),
        worst_excursion(&p)
    );

    // 1b. STEP plateau: the same +40/255 wash-out, but the loop falls into the degenerate attractor
    //     within one AR chunk of the first eviction and then holds. This is where an OLS trend is
    //     *genuinely* blind — a step early in the segment leaves almost no slope — so it is the
    //     stimulus that makes the excursion gate load-bearing rather than merely redundant.
    let step = map_frames(N, W, H, |i, _, f| {
        if i >= PRE + 4 {
            for c in f.pixels.iter_mut() {
                *c = (*c as f64 + 40.0).clamp(0.0, 255.0) as u8;
            }
        }
    });
    let st = score_clip(&step, PRE, BUCKET);
    print_drift("step", &st);
    assert!(
        worst_trend(&st) < DRIFT_BUDGET,
        "the step-onset plateau scored {:.2}/255 on the TREND alone — this stimulus no longer \
         demonstrates the OLS blind spot, so it cannot evidence why the excursion is gated",
        worst_trend(&st)
    );
    assert!(
        worst_drift(&st) > DRIFT_BUDGET,
        "a +40/255 wash-out that lands one chunk after the first eviction and holds scored {:.2}/255 \
         — the gate is blind to a degenerate attractor, which is the most likely AR failure",
        worst_drift(&st)
    );

    // 2. Localized collapse: the central 12% of the frame fades to black while the surround brightens
    //    to keep the GLOBAL luma mean where it started. Statistically invisible to global moments.
    let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
    let r2 = 0.12 * (W * H) as f64 / std::f64::consts::PI;
    let localized = map_frames(N, W, H, |_, t, f| {
        let (mut inside, mut outside) = (0usize, 0usize);
        let mut lost = 0.0f64;
        for (i, px) in f.pixels.chunks_exact(3).enumerate() {
            let (x, y) = ((i % W) as f64 - cx, (i / W) as f64 - cy);
            let luma = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            if x * x + y * y <= r2 {
                inside += 1;
                lost += t * luma;
            } else {
                outside += 1;
            }
        }
        let comp = lost / outside.max(1) as f64;
        let _ = inside;
        for (i, px) in f.pixels.chunks_exact_mut(3).enumerate() {
            let (x, y) = ((i % W) as f64 - cx, (i / W) as f64 - cy);
            if x * x + y * y <= r2 {
                for c in px.iter_mut() {
                    *c = (*c as f64 * (1.0 - t)).clamp(0.0, 255.0) as u8;
                }
            } else {
                for c in px.iter_mut() {
                    *c = (*c as f64 + comp).clamp(0.0, 255.0) as u8;
                }
            }
        }
    });
    let l = score_clip(&localized, PRE, BUCKET);
    print_drift("localized", &l);
    assert!(
        worst_drift(&l) > DRIFT_BUDGET,
        "a localized subject collapse that holds the global mean scored {:.2}/255, inside the \
         {DRIFT_BUDGET:.2} budget — global moments alone cannot see subject-identity loss",
        worst_drift(&l)
    );
    // ...and it is the SPATIAL component that caught it, which is why that component exists.
    let spatial = l[5].trend.abs().max(l[5].gated_excursion().abs());
    assert!(
        spatial > DRIFT_BUDGET,
        "the spatial component scored only {spatial:.2}/255 on a localized collapse — the block-luma \
         spread is not doing the job it was added for"
    );

    // 3. Hue rotation at constant luma AND constant chroma magnitude: rotate the (Cb, Cr) opponent
    //    pair through 180 degrees over the clip. Luma is preserved exactly by reconstruction and the
    //    chroma *magnitude* is preserved by the rotation, so neither luma-mean, contrast, saturation
    //    nor the spatial spread can see it — only the opponent pair, which ends sign-reversed.
    let hue = map_frames(N, W, H, |_, t, f| {
        let (s, c) = (std::f64::consts::PI * t).sin_cos();
        for px in f.pixels.chunks_exact_mut(3) {
            let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let (cb, cr) = (b - y, r - y);
            let (cb2, cr2) = (cb * c - cr * s, cb * s + cr * c);
            let (r2, b2) = (y + cr2, y + cb2);
            let g2 = (y - 0.299 * r2 - 0.114 * b2) / 0.587;
            px[0] = r2.clamp(0.0, 255.0) as u8;
            px[1] = g2.clamp(0.0, 255.0) as u8;
            px[2] = b2.clamp(0.0, 255.0) as u8;
        }
    });
    let hu = score_clip(&hue, PRE, BUCKET);
    print_drift("hue", &hu);
    assert!(
        worst_drift(&hu) > DRIFT_BUDGET,
        "a 180-degree hue rotation at constant luma scored {:.2}/255, inside the {DRIFT_BUDGET:.2} \
         budget — the descriptor cannot see a colour cast",
        worst_drift(&hu)
    );
    let opponent = hu[3..5]
        .iter()
        .map(|c| c.trend.abs().max(c.gated_excursion().abs()))
        .fold(0.0, f64::max);
    assert!(
        opponent > DRIFT_BUDGET,
        "the opponent-colour components scored only {opponent:.2}/255 on a hue rotation — they are \
         not doing the job they were added for"
    );

    // The budget's UPPER pin: the weakest of these four shapes bounds how high `DRIFT_BUDGET` may be
    // set. Assert the margin explicitly, so raising the budget past what the metric can actually
    // detect turns this gate red instead of silently widening the blind spot.
    let weakest = [&p, &st, &l, &hu]
        .into_iter()
        .map(worst_drift)
        .fold(f64::INFINITY, f64::min);
    assert!(
        weakest > DRIFT_BUDGET * 1.3,
        "the weakest plausible failure shape scores {weakest:.2}/255 against a {DRIFT_BUDGET:.2} \
         budget — under a 1.3x margin, so the budget is too close to the metric's detection limit \
         for a no-drift verdict to mean anything"
    );

    // No descriptor component may be INERT: a channel that returns the same number on every stimulus
    // is contributing nothing and would silently narrow the coverage this gate claims. Compare each
    // component's trend across the four stimuli and require it to actually move somewhere.
    let motion: Vec<Image> = (0..N).map(|i| smooth_frame(W, H, i)).collect();
    let mo = score_clip(&motion, PRE, BUCKET);
    for (c, name) in DESCRIPTOR_NAMES.iter().enumerate().map(|(i, n)| (i, *n)) {
        let vals = [
            mo[c].trend,
            p[c].trend,
            l[c].trend,
            hu[c].trend,
            mo[c].excursion,
            p[c].excursion,
            l[c].excursion,
            hu[c].excursion,
        ];
        let span = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - vals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            span > 1.0,
            "descriptor component `{name}` moves only {span:.4}/255 across four different failure \
             shapes — it is inert and the descriptor is narrower than it claims"
        );
    }
}
