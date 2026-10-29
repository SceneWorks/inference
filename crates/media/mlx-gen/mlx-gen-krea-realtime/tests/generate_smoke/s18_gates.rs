//! Weights-free gates on the S18 verdict rule, the KV-tier comparison and the crate prose.

use super::*;

/// **CI gate for the S18 decision rule.** The real-weight run happens on a gated host; the rule it
/// applies to those numbers must not be un-tested. Drives [`S18Sweep::verdict`] over every outcome it
/// distinguishes, including the two power-related ones added after review.
#[test]
fn the_s18_verdict_rule_distinguishes_its_outcomes() {
    // Build a sweep with `n` seeds per row, each row's values spread by `spread` around its mean.
    let sweep = |means: [f64; 6], spread: f64, n: usize| {
        let mut cells = Vec::new();
        for (i, row) in ['A', 'B', 'C', 'D', 'F', 'E'].iter().enumerate() {
            for k in 0..n {
                let off = if n < 2 {
                    0.0
                } else {
                    spread * (k as f64 / (n - 1) as f64 - 0.5)
                };
                // Dose rows have independent per-seed noise. Reusing the same offset on every row
                // would cancel it from each matched-seed slope and fabricate zero uncertainty.
                let row_noise = if *row == 'F' { -off } else { off };
                cells.push(S18Cell {
                    row: *row,
                    seed: k as u64,
                    latent_frames: 45,
                    rolls: match *row {
                        'D' => 10,
                        'F' => 5,
                        'E' => 0,
                        _ => 13,
                    },
                    reported_drift: means[i] + row_noise,
                    trend: means[i] + row_noise,
                    excursion: 0.0,
                    // A 100-output-frame post segment, so slope == trend numerically. Sweeps built by
                    // this closure have no row Z, so the rate-floor clause is inert in them; cases 2g
                    // and 2h below add one explicitly, and the RECORDED 640x384 data has one.
                    slope: means[i] + row_noise,
                    peak_bytes: 1,
                    clip_mean: 0.0,
                    head_motion: 2.0,
                    tail_motion: 2.0,
                    ar_wall_ms: 0,
                    component: "luma-mean",
                });
            }
        }
        S18Sweep {
            bucket: "test".into(),
            kv: KV_TIER_BF16.to_string(),
            cells,
        }
    };

    // Add the zero-eviction row Z that `sweep` does not build, so the A-vs-Z rate-floor comparison
    // has something to compare. Shaped like the recorded 640x384 row Z: a 6-latent-frame clip — the
    // shipped window, so it never evicts and `structural_checks`' `rolls == 0` holds — whose
    // `100 * trend / slope` post segment is ~12 output frames against row A's 100. Row A's mean slope
    // is its mean drift in these sweeps (slope == trend in the closure above), so the caller picks
    // which side of row A row Z lands on purely by choosing slopes.
    let with_z = |mut s: S18Sweep, slopes: [f64; 3]| {
        for (k, slope) in slopes.into_iter().enumerate() {
            s.cells.push(S18Cell {
                row: 'Z',
                seed: k as u64,
                latent_frames: 6,
                rolls: 0,
                reported_drift: 7.2,
                trend: 7.2,
                excursion: 0.0,
                slope,
                peak_bytes: 1,
                clip_mean: 0.0,
                head_motion: 2.0,
                tail_motion: 2.0,
                ar_wall_ms: 0,
                component: "luma-mean",
            });
        }
        s
    };

    // 1. Coherent: every bounded row sits far under the budget, with a resolvable margin.
    let v = sweep([2.0, 2.1, 1.9, 2.2, 2.0, 1.8], 0.4, 3)
        .verdict()
        .expect("a flat, replicated sweep must yield a verdict");
    assert!(v.starts_with("coherent"), "got: {v}");

    // 2. Drift, repaired by the anchor: the shipped row is far past the budget and the sinks pull it
    //    most of the way back.
    let v = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a repaired-drift sweep must yield a verdict");
    assert!(v.starts_with("drift is real"), "got: {v}");

    // 2b. Drift with a significant NEGATIVE linear dose response: fewer rolls are worse. This rejects
    //     a positive linear/monotone roll-count contribution over the measured range, but must not
    //     claim that every possible cache-window mechanism is impossible or reach for a sink.
    let v = sweep([40.0, 8.0, 6.0, 42.0, 45.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unattributed drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("does NOT support a positive linear bounded-window dose response"),
        "got: {v}"
    );
    assert!(
        v.contains("does not exclude a non-linear"),
        "the conclusion must retain the model-form limitation: {v}"
    );
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");
    assert!(v.contains("three-dose within-regime fit"), "got: {v}");
    // ...and it must NOT lean on row E. Row E is out of regime (a different attention mask), n=1 and
    // has no variance estimate; citing it as attribution evidence is the same inference this rule
    // refuses everywhere else, with the sign flipped.
    assert!(
        !v.contains("global"),
        "the out-of-regime, n=1 global reference row must not appear in the attribution sentence: {v}"
    );
    // ...and with a row Z present this arm must still carry the rate clause. It is the ONE arm that
    // returns early, so it interpolates `{rate}` itself rather than taking the shared append below
    // it; without this case nothing in the suite would notice that copy being deleted as redundant.
    let v = with_z(
        sweep([40.0, 8.0, 6.0, 42.0, 45.0, 2.0], 2.0, 3),
        [60.0, 61.0, 62.0],
    )
    .verdict()
    .expect("a negative dose response with a measured Z is still a conclusion");
    assert!(
        v.contains("sink anchor is not indicated. The within-regime zero-eviction row Z"),
        "the early-returning attribution arm must emit the rate clause in place, before its own \
         remaining sentences: {v}"
    );

    // 2c. A shallow three-dose slope inside its between-seed scatter. The symmetric rule must refuse,
    //     in both directions, and report the practical 2*SEM magnitude floor without calling it a
    //     confidence interval.
    let v = sweep([40.0, 8.0, 6.0, 40.5, 41.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unresolvable attribution is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );
    assert!(!v.contains("NOT attributable"), "got: {v}");
    assert!(v.contains("NEITHER direction may be asserted"), "got: {v}");
    assert!(v.contains("practical 2*SEM magnitude floor"), "got: {v}");

    // 2d. **The reviewer's own probe.** Row D is genuinely ~40% BETTER than row A, but the scatter
    //     swamps it. The old rule returned an unqualified "the dose-response runs the wrong way /
    //     eviction is not the mechanism" on exactly this data. It must not.
    let probe = |row: char, vals: [f64; 3]| {
        vals.into_iter().enumerate().map(move |(k, t)| S18Cell {
            row,
            seed: k as u64,
            latent_frames: 45,
            rolls: match row {
                'D' => 10,
                'F' => 5,
                _ => 13,
            },
            reported_drift: t,
            trend: t,
            excursion: 0.0,
            slope: t,
            peak_bytes: 1,
            clip_mean: 0.0,
            head_motion: 2.0,
            tail_motion: 2.0,
            ar_wall_ms: 0,
            component: "luma-mean",
        })
    };
    let mut cells: Vec<S18Cell> = probe('A', [20.0, 40.0, 22.0]).collect();
    cells.extend(probe('D', [10.0, 26.0, 13.0]));
    cells.extend(probe('F', [30.0, 5.0, 10.0]));
    cells.extend(probe('B', [18.0, 30.0, 20.0]));
    cells.extend(probe('C', [17.0, 28.0, 19.0]));
    let v = S18Sweep {
        bucket: "reviewer-probe".into(),
        kv: KV_TIER_BF16.to_string(),
        cells,
    }
    .verdict()
    .expect("the probe must still yield a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        !v.contains("NOT attributable"),
        "the reviewer's probe has row D 40% BETTER than row A and must never return a window \
         falsification — got: {v}"
    );
    assert!(
        !v.contains("runs the wrong way"),
        "the reviewer's probe must never return a wrong-way dose-response — got: {v}"
    );
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );

    // 2e. And the OTHER direction must be assertable when the evidence clears the heuristic: a wider
    //     window materially better than the shipped one supports a positive bounded-window response.
    let v = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an implicating dose-response is still a conclusion");
    assert!(
        v.contains("supports a positive bounded-window dose response"),
        "got: {v}"
    );

    // 2f. The middle D dose must materially participate in the three-point fit. Hold the A/F
    //     endpoints fixed and move only D across the curve: an endpoint-only implementation would
    //     return the same classification for both sweeps and this mutation gate would fail.
    let negative_middle = sweep([40.0, 8.0, 6.0, 0.0, 40.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a negative middle-dose slope is still a conclusion");
    let positive_middle = sweep([40.0, 8.0, 6.0, 100.0, 40.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a positive middle-dose slope is still a conclusion");
    assert!(
        negative_middle.contains("does NOT support a positive linear bounded-window dose response"),
        "moving only D low must make the fitted slope negative: {negative_middle}"
    );
    assert!(
        positive_middle.contains("supports a positive bounded-window dose response"),
        "moving only D high must make the fitted slope positive: {positive_middle}"
    );

    // 2g. **Drift, no row F, but a measured row Z.** The A-vs-Z rate-floor comparison is independent
    //     of the A/D/F dose ladder, so an `Unmeasured` attribution must not take the row-Z evidence
    //     down with it. This is the shape a fresh 832x480 sweep takes on the current host, where row
    //     F does not fit — the committed `MEASURED_832` still carries an F, but new runs will not
    //     (sc-17841).
    let no_f = || {
        let mut s = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
        s.cells.retain(|c| c.row != 'F');
        s
    };
    // Row A's mean slope is 40.00/100f, so a Z above it is the recorded finding: Z is HIGHER,
    // therefore it is not a rate floor.
    let v = with_z(no_f(), [60.0, 61.0, 62.0])
        .verdict()
        .expect("a drift sweep with no F but a measured Z is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("zero-eviction row Z does NOT establish a rate floor here"),
        "the A-vs-Z rate comparison must survive an Unmeasured window attribution — this is the \
         sc-17841 regression, and without the fix the clause is discarded with the empty \
         attribution clause: {v}"
    );
    assert!(
        v.contains("which is HIGHER than the shipped row's"),
        "row Z above row A must report the HIGHER branch, with its numbers: {v}"
    );
    assert!(
        v.contains("budget). The within-regime"),
        "with no attribution clause the rate sentence must be joined to the head by a full stop, \
         not run on after `budget)`: {v}"
    );
    // ...and the clause must be an addition, not a replacement: the sink conclusion still lands.
    assert!(v.contains("repairs it to"), "got: {v}");
    // The OTHER rate-floor branch — a row Z genuinely below row A — would mean a same-content floor
    // does exist, which is the outcome the recorded sweeps' narrowed wording is contingent on. It has
    // no coverage anywhere else in the suite, and it is reachable from exactly this shape.
    let v = with_z(no_f(), [10.0, 11.0, 12.0])
        .verdict()
        .expect("a drift sweep with a low row Z is still a conclusion");
    assert!(
        v.contains("Z is lower, so the shipped row's rate does exceed the zero-eviction rate"),
        "row Z below row A must report the LOWER branch: {v}"
    );
    // Negative control: the same sweep with NO row Z must not manufacture a rate clause, and the
    // sentence-joining full stop above must not leak into the empty case. The missing stop after
    // `budget)` here is pre-existing wording this change deliberately leaves alone — it is pinned so
    // the join stays conditional on there being a clause to join.
    let v = no_f()
        .verdict()
        .expect("a drift sweep with neither F nor Z is still a conclusion");
    assert!(
        !v.contains("zero-eviction row Z"),
        "an unmeasured row Z must produce no rate clause at all: {v}"
    );
    assert!(
        v.contains("budget) And a first-chunk"),
        "with no rate clause the head must not gain a trailing full stop: {v}"
    );

    // 2h. The rate clause must ALSO still be emitted on the attribution arms that already carried it,
    //     now that it is appended once rather than interpolated per-arm.
    let v = with_z(
        sweep([40.0, 8.0, 6.0, 40.5, 41.0, 2.0], 2.0, 3),
        [60.0, 61.0, 62.0],
    )
    .verdict()
    .expect("an unresolvable attribution with a measured Z is still a conclusion");
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );
    assert!(
        v.contains("practical floor. The within-regime zero-eviction row Z"),
        "the rate clause must follow the attribution clause without losing its sentence break: {v}"
    );

    // 3. Drift the sink does NOT repair — a real finding, but it must never read as "ship a sink".
    let v = sweep([40.0, 38.0, 39.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unrepaired drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("does NOT repair it"), "got: {v}");
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 3b. Drift where the sink comparison is swamped by seed scatter — must say so, and must still
    //     refuse to wire a sink.
    let v = sweep([40.0, 22.0, 26.0, 30.0, 20.0, 2.0], 12.0, 3)
        .verdict()
        .expect("an unresolvable repair comparison is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("NOT resolvable"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 4. Incoherent evidence: the shipped row looks clean but the WIDER window looks worse. That
    //    ordering is impossible under the drift hypothesis, so the run proves nothing either way.
    let e = sweep([3.0, 3.0, 3.0, 25.0, 40.0, 2.0], 0.4, 3)
        .verdict()
        .expect_err("contradictory rows must not produce a no-drift verdict");
    assert!(e.contains("dose-response"), "got: {e}");

    // 5. UNDERPOWERED, near the budget: one seed per row and a shipped row close to the budget. The
    //    rule must refuse rather than pick a side — this is the guard the review demanded.
    let e = sweep([13.0, 8.0, 9.0, 12.5, 12.0, 2.0], 0.0, 1)
        .verdict()
        .expect_err("a single-seed sweep near the budget must not produce a verdict");
    assert!(e.contains("UNDERPOWERED"), "got: {e}");
    assert!(e.contains("NO variance estimate"), "got: {e}");

    // 5b. ...and the same means WITH enough seeds still refuse while the shipped row's own scatter
    //     swamps its margin against the budget.
    let e = sweep([13.0, 8.0, 9.0, 12.5, 12.0, 2.0], 12.0, 3)
        .verdict()
        .expect_err("a sweep whose seed scatter swamps its margin must not produce a verdict");
    assert!(e.contains("between-seed spread"), "got: {e}");

    // 6. Underpowered but unambiguously clean: one seed, but everything is far under the budget. That
    //    is reportable — as a narrowed claim that says so.
    let v = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.0, 1)
        .verdict()
        .expect("a single-seed but grossly clean sweep must yield a narrowed verdict");
    assert!(v.starts_with("underpowered but no gross drift"), "got: {v}");
    assert!(v.contains("does NOT rank the configs"), "got: {v}");

    // 7. And the sweep must have been long enough / the reference must really be a reference.
    let mut short = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in short.cells.iter_mut().filter(|c| c.row == 'A') {
        c.rolls = 4;
    }
    assert!(short.verdict().unwrap_err().contains("not a long clip"));
    let mut bad_ref = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in bad_ref.cells.iter_mut().filter(|c| c.row == 'E') {
        c.rolls = 3;
    }
    assert!(bad_ref.verdict().unwrap_err().contains("not a reference"));
    let mut unmatched = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    unmatched.cells.retain(|c| !(c.row == 'F' && c.seed == 2));
    assert!(
        unmatched
            .verdict()
            .unwrap_err()
            .contains("same matched seeds"),
        "dropping one F seed must invalidate the paired slope"
    );
    let mut wrong_dose = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    for c in wrong_dose.cells.iter_mut().filter(|c| c.row == 'F') {
        c.rolls = 10;
    }
    assert!(
        wrong_dose
            .verdict()
            .unwrap_err()
            .contains("exact 45-latent-frame, 13/10/5-roll"),
        "using the wrong F window must invalidate the three-dose fit"
    );
    let mut short_f = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    for c in short_f.cells.iter_mut().filter(|c| c.row == 'F') {
        c.latent_frames = 36;
        c.rolls = 2;
    }
    assert!(
        short_f
            .verdict()
            .unwrap_err()
            .contains("exact 45-latent-frame, 13/10/5-roll"),
        "a shorter F clip must not masquerade as the planned dose"
    );
    let mut duplicate_seed = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    let duplicate = *duplicate_seed
        .cells
        .iter()
        .find(|c| c.row == 'F' && c.seed == 0)
        .unwrap();
    duplicate_seed.cells.push(duplicate);
    assert!(
        duplicate_seed
            .verdict()
            .unwrap_err()
            .contains("duplicate seed cells"),
        "duplicate evidence must not be silently reduced to the first matching cell"
    );
    let mut non_finite = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    non_finite
        .cells
        .iter_mut()
        .find(|c| c.row == 'F' && c.seed == 0)
        .unwrap()
        .trend = f64::INFINITY;
    assert!(
        non_finite
            .verdict()
            .unwrap_err()
            .contains("non-finite drift statistic"),
        "non-finite evidence must never fall through to an attribution verdict"
    );
    // 8. No shipped row at all — refuse, do not index off the end.
    let empty = S18Sweep {
        bucket: "test".into(),
        kv: KV_TIER_BF16.to_string(),
        cells: vec![],
    };
    assert!(empty.verdict().unwrap_err().contains("nothing to conclude"));
}

/// **The two WITHDRAWN sc-15127 claims must not survive in crate *prose*.**
///
/// Every other S18 gate in this file inspects the *generated verdict string*. That is exactly why two
/// stale doc blocks in `src/t2v.rs` — the module header and the `mac_ar_config` ship-decision doc —
/// asserted both withdrawn claims through two full review rounds: no gate reads hand-written prose, so
/// prose is where a retracted conclusion hides. This test closes that hole by reading the doc-bearing
/// sources themselves.
///
/// Doc comments wrap, so the check normalises `//!` / `///` markers and runs of whitespace to single
/// spaces before matching — a banned phrase split across two lines is still a banned phrase.
#[test]
fn the_withdrawn_s18_claims_do_not_survive_in_crate_prose() {
    /// `//!` / `///` markers and line wrapping removed, whitespace collapsed, lowercased.
    fn normalise(src: &str) -> String {
        src.split_whitespace()
            .filter(|w| *w != "//!" && *w != "///")
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    // Both phrases are WITHDRAWN findings, not style nits — see sc-15571's "WITHDRAWN" section.
    //
    //   "not attributable to the bounded ..." — the enlarged A/D/F 13/10/5-roll ladder remains
    //     UNRESOLVED in both buckets, not falsified: its matched-seed slopes (+0.571/255 at 640×384,
    //     −0.278/255 at 832×480) are inside their predeclared 2·SEM heuristics (1.897 and 1.678).
    //   "saturation run-away" — `saturation` wins NO row-A cell in either bucket (`S18Cell::component`
    //     records the winner per cell); the headline mode is colour-cast/tone/structure.
    const BANNED: &[(&str, &str)] = &[
        (
            "not attributable to the bounded",
            "the enlarged A/D/F 13/10/5-roll ladder remains UNRESOLVED in both buckets, not \
             falsified — its matched-seed slopes (+0.571/255 at 640×384, −0.278/255 at 832×480) \
             remain inside their predeclared 2·SEM heuristics (1.897 and 1.678)",
        ),
        (
            "saturation run-away",
            "`saturation` wins NO row-A cell in either bucket (see the per-cell `S18Cell::component` \
             winners) — the headline mode is a colour-cast/tone/structure drift, alongside which a \
             saturation rise is only separately observable",
        ),
    ];

    for (file, src) in [
        ("src/t2v.rs", include_str!("../../src/t2v.rs")),
        ("src/lib.rs", include_str!("../../src/lib.rs")),
    ] {
        let prose = normalise(src);
        for (phrase, why) in BANNED {
            assert!(
                !prose.contains(&phrase.to_lowercase()),
                "{file} contains the WITHDRAWN sc-15127 claim \"{phrase}\".\n\n\
                 Why it is banned: {why}.\n\n\
                 Do NOT delete this assertion to make the build pass — it exists because both of \
                 these phrases survived two review rounds in hand-written crate prose while every \
                 other S18 gate only inspected the generated verdict string. Rewrite the prose to \
                 match the verdict the recorded sweep actually supports (see \
                 `generate_t2v_from_components`' doc and sc-15571's WITHDRAWN section). If a future \
                 measurement genuinely re-establishes one of these claims, retire the entry here in \
                 the same change that records the data."
            );
        }
    }
}

/// **CI gate for the sc-17807 KV-tier A/B rule.** The quantized arm is measured on a gated host; the
/// rule applied to those numbers must not itself be untested, and it must be able to return every
/// failure it claims to gate. A comparison that can only pass is not a comparison.
///
/// Drives [`S18Sweep::kv_tier_comparison`] over each outcome, with hand-built cells so no weights are
/// needed. The baseline's scatter here is deliberately of the same order as the recorded sweep's
/// (±4-5/255 on row A at 640×384), NOT a fabricated tight one — a demonstration of symmetry on data
/// unlike the real data would prove nothing about the real comparison.
#[test]
fn the_kv_tier_comparison_can_fail_the_quantized_arm() {
    fn cell(row: char, seed: u64, drift: f64, peak: usize) -> S18Cell {
        S18Cell {
            row,
            seed,
            latent_frames: 45,
            rolls: 13,
            reported_drift: drift,
            trend: drift,
            excursion: 0.0,
            slope: 1.0,
            peak_bytes: peak,
            clip_mean: 1.0,
            head_motion: 12.0,
            tail_motion: 8.0,
            ar_wall_ms: 300_000,
            component: "luma-mean",
        }
    }
    let sweep = |kv: &str, cells: Vec<S18Cell>| S18Sweep {
        bucket: "640x384".into(),
        kv: kv.into(),
        cells,
    };
    // Row A at the recorded 640x384 scatter: 23.04 / 31.05 / 28.43, spread (2*SEM) ~4.72. Any
    // unpaired comparison against this cannot resolve less than ~6.7/255.
    const BASE: [f64; 3] = [23.0413, 31.0541, 28.4337];
    let seeds = [7u64, 11, 23];
    let arm_cells = |shift: f64, peak: usize| {
        seeds
            .iter()
            .zip(BASE)
            .map(|(&seed, d)| cell('A', seed, d + shift, peak))
            .collect::<Vec<_>>()
    };
    let bf16 = sweep(KV_TIER_BF16, arm_cells(0.0, 16_000_000_000));

    // 1. A near-identical arm is not a regression, and reports the saving and the wall time.
    let same = sweep("q8/g64", arm_cells(0.1, 12_000_000_000));
    let ok = same
        .kv_tier_comparison(&bf16)
        .expect("+0.10/255 is under the material floor");
    assert!(ok.contains("not resolvably worse"), "{ok}");
    // A constant shift gives the paired deltas zero scatter, so +0.10 clears its own 2*SEM. It must
    // still not be called a regression — and it must be REPORTED rather than swallowed, which is
    // what distinguishes a material floor from simply raising the threshold.
    assert!(ok.contains("resolvable but IMMATERIAL"), "{ok}");
    assert!(ok.contains("0.10"), "the magnitude must survive: {ok}");
    assert!(
        ok.contains("0.75x"),
        "the memory saving must be reported: {ok}"
    );
    assert!(ok.contains("AR wall"), "{ok}");

    // 1b. ...and a difference that is material but NOT statistically resolvable is also not a
    //     regression. Both conditions are required, and this is the one that fails the other way.
    let noisy_baseline = sweep(
        KV_TIER_BF16,
        vec![
            cell('A', 7, 20.0, 16_000_000_000),
            cell('A', 11, 30.0, 16_000_000_000),
            cell('A', 23, 25.0, 16_000_000_000),
        ],
    );
    let scattered = sweep(
        "q8/g64",
        vec![
            cell('A', 7, 27.0, 12_000_000_000),
            cell('A', 11, 26.0, 12_000_000_000),
            cell('A', 23, 32.0, 12_000_000_000),
        ],
    );
    let text = scattered
        .kv_tier_comparison(&noisy_baseline)
        .expect("a large but unresolved delta is not a regression");
    assert!(text.contains("unresolved"), "{text}");
    assert!(text.contains("can only exclude"), "{text}");

    // 2. THE POWER RESULT. A uniform +3.0/255 regression is FAR inside the unpaired 2*SEM (~6.7)
    //    and would pass an unpaired comparison — the paired form catches it, because both arms ran
    //    the same seeds and the per-seed delta is a constant with zero scatter. This is the whole
    //    reason the statistic is paired; if it regresses to unpaired, this assertion goes red.
    let worse = sweep("q8/g64", arm_cells(3.0, 12_000_000_000));
    let err = worse
        .kv_tier_comparison(&bf16)
        .expect_err("a uniform +3.00/255 paired regression must not pass");
    assert!(err.contains("RESOLVABLY WORSE"), "{err}");
    assert!(
        err.contains("paired"),
        "the paired form must be the one that caught it: {err}"
    );
    assert!(err.contains("not a win"), "{err}");

    // 3. Correctly signed: a uniform improvement is reported as better, not as worse.
    let better = sweep("q8/g64", arm_cells(-3.0, 12_000_000_000));
    let text = better
        .kv_tier_comparison(&bf16)
        .expect("better is not worse");
    assert!(text.contains("better by"), "{text}");

    // 4. MEMORY IS GATED, not decorated. An arm with identical drift but a HIGHER peak has bought
    //    nothing, and the tier exists to buy memory — so it must fail.
    let no_saving = sweep("q8/g64", arm_cells(0.0, 17_600_000_000));
    let err = no_saving
        .kv_tier_comparison(&bf16)
        .expect_err("a tier that raises the peak has bought nothing");
    assert!(
        err.contains("did NOT lower the measured MLX active peak"),
        "{err}"
    );
    assert!(err.contains("1.10x"), "{err}");
    // Equal peaks are also not a saving.
    assert!(sweep("q8/g64", arm_cells(0.0, 16_000_000_000))
        .kv_tier_comparison(&bf16)
        .is_err());

    // 5. One seed per row is UNRANKABLE, and the HEADLINE must say so. Reporting "not resolvably
    //    worse" off an arm with no variance estimate asserts a conclusion the data cannot support —
    //    the row detail said UNRANKABLE while the headline said agreement.
    let single = sweep("q8/g64", vec![cell('A', 7, 40.0, 12_000_000_000)]);
    let text = single
        .kv_tier_comparison(&bf16)
        .expect("an unrankable row is an absent comparison, not a failure");
    assert!(text.contains("NO ROW WAS RANKABLE"), "{text}");
    assert!(text.contains("This is not agreement"), "{text}");
    assert!(!text.contains("is not resolvably worse"), "{text}");

    // 6. Unmatched seeds fall back to the unpaired form rather than silently comparing nothing.
    let unmatched = sweep(
        "q8/g64",
        vec![
            cell('A', 101, 23.0, 12_000_000_000),
            cell('A', 102, 31.0, 12_000_000_000),
            cell('A', 103, 28.0, 12_000_000_000),
        ],
    );
    let text = unmatched
        .kv_tier_comparison(&bf16)
        .expect("unmatched seeds still compare, less powerfully");
    assert!(text.contains("unpaired"), "{text}");

    // 7. Malformed comparisons are refused rather than answered.
    assert!(bf16
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("not an A/B"));
    let other_bucket = S18Sweep {
        bucket: "832x480".into(),
        kv: "q8/g64".into(),
        cells: arm_cells(0.0, 12_000_000_000),
    };
    assert!(other_bucket
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("across geometries"));
    let no_overlap = sweep("q8/g64", vec![cell('D', 7, 20.0, 12_000_000_000)]);
    assert!(no_overlap
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("nothing to compare"));
    // A "baseline" that is itself quantized is not the shipped reference.
    let q4 = sweep("q4/g64", arm_cells(0.0, 8_000_000_000));
    assert!(same
        .kv_tier_comparison(&q4)
        .unwrap_err()
        .contains("not the shipped"));
}

/// **CI gate on the recorded sc-17807 KV-tier A/B — and it is a RECORDED FAILURE.**
///
/// The measured answer to the story's question ("a cheaper cache that increases drift is not a
/// win, and this sweep is the instrument that can say so") is that **Q8 KV costs coherence**. Both
/// arms were measured on one host — rows A/B/C/D x seeds 7/11/23 at 640x384, 45 latent frames, q4
/// weights — and the comparison returns `Err`:
///
/// | row | sink | per-seed Δ (q8 − bf16) | mean | 2*SEM | |
/// |---|---|---|---|---|---|
/// | A | 0 | +0.89, +7.36, −2.52 | +1.91 | 5.79 | unresolved |
/// | B | 1 | +2.12, +2.15, −0.67 | +1.20 | 1.87 | unresolved |
/// | C | 3 | +2.44, +3.59, +2.34 | **+2.79** | 0.80 | **RESOLVABLY WORSE** |
/// | D | 0 | −0.62, +3.34, +3.87 | +2.20 | 2.83 | unresolved |
///
/// **Read it as one effect, not one bad row.** Every row's mean is positive, 9 of the 12 paired
/// deltas are positive, and row C is simply the row whose paired scatter (2*SEM 0.80) is tight
/// enough to *resolve* something the others cannot — their bounds only exclude regressions above
/// 1.87–5.79/255. The consistent reading is a drift cost of roughly **+2/255** across the bounded
/// rows, resolvable where the power exists. That it lands on the largest-sink row is *consistent
/// with* — but does not establish — the mechanism you would guess: sink KV is quantized once and
/// then attended for the whole clip, so its error never ages out the way a rolling tail's does.
/// Distinguishing "an effect everywhere, resolved only on C" from "an effect concentrated on the
/// sink" needs more seeds on A/B/D, not more prose.
///
/// The memory saving is real and simultaneous: peak 0.86x (A), 0.85x (B), 0.82x (C), 0.76x (D).
/// So this is a **measured trade**, not a free win — which is exactly why
/// [`KreaArConfig::kv_cache_quant`](mlx_gen_krea_realtime::KreaArConfig::kv_cache_quant) defaults
/// to `None` and why nothing in the tree turns it on.
///
/// Provenance: `tests/fixtures/s18_kv_ab_{bf16,q8}_cells.tsv`, the verbatim `S18CELL` lines from
/// twelve gated runs. Row F is absent by decision — see
/// [`long_clip_coherence_under_the_bounded_window`].
///
/// This test asserts the **failure**. If a future change makes the A/B pass, that is either a real
/// improvement or a weakened rule, and either way it must be re-argued here rather than silently
/// flipping green.
#[test]
fn the_recorded_kv_tier_ab_records_a_resolvable_q8_regression() {
    let arm_of = |src: &str, want: &str| {
        let s = s18_sweep_from_accumulated(src, None).expect("the recorded arm must parse");
        assert_eq!(s.kv, want, "the recorded arm carries its tier");
        assert_eq!(s.cells.len(), 12, "4 bounded rows x 3 seeds");
        s
    };
    let bf16 = arm_of(
        include_str!("../fixtures/s18_kv_ab_bf16_cells.tsv"),
        KV_TIER_BF16,
    );
    let q8 = arm_of(include_str!("../fixtures/s18_kv_ab_q8_cells.tsv"), "q8/g64");

    // Both arms are structurally valid sweeps under the sweep's OWN rule before either is compared.
    for (label, arm) in [("bf16", &bf16), ("q8", &q8)] {
        arm.structural_checks()
            .unwrap_or_else(|e| panic!("{label} arm: {e}"));
        arm.verdict()
            .unwrap_or_else(|e| panic!("{label} arm verdict: {e}"));
    }

    let err = q8
        .kv_tier_comparison(&bf16)
        .expect_err("the recorded A/B is a REGRESSION — see this test's doc before changing it");
    assert!(err.contains("RESOLVABLY WORSE"), "{err}");
    assert!(
        err.contains("C +2.79/255"),
        "row C's effect size is the published one: {err}"
    );
    // The memory saving is real and must stay reported alongside the regression — the finding is a
    // trade, and an `Err` that dropped the saving would misrepresent it as a pure loss.
    for saving in ["0.86x", "0.85x", "0.82x", "0.76x"] {
        assert!(err.contains(saving), "missing peak ratio {saving}: {err}");
    }

    // Every row leans worse. This is what makes "one effect, resolved only where the power is"
    // the honest reading rather than "row C is an outlier" — if a re-measurement ever makes the
    // other rows negative, that reading has to change.
    let mut positive = 0;
    for row in ['A', 'B', 'C', 'D'] {
        let (a, b) = (q8.mean(row).unwrap(), bf16.mean(row).unwrap());
        assert!(a > b, "row {row}: q8 {a:.2} is not above bf16 {b:.2}");
        positive += q8
            .paired_deltas(&bf16, row)
            .expect("matched seeds")
            .iter()
            .filter(|d| **d > 0.0)
            .count();
    }
    assert_eq!(positive, 9, "9 of 12 paired deltas positive");

    // The knob this evidence is about must still be OFF by default. A measured coherence cost that
    // ships enabled would be the worst possible outcome of having measured it.
    assert_eq!(
        KreaRealtimeConfig::krea_realtime_14b().ar.kv_cache_quant,
        None
    );
}

/// **A mis-set KV tier must fail loudly, not silently measure the other arm.**
///
/// The failure this closes costs hours and produces evidence that looks fine: `KREA_S18_KV_BITS=q8`
/// (a plausible typo — the *label* is `q8`, the *value* is `8`) parsed with `.ok()` yields `None`,
/// so the sweep runs the bf16 cache, labels its cells `bf16`, and the A/B then compares the baseline
/// against itself and reports agreement. Every downstream guard in this file would be green.
///
/// Serial by construction: it mutates process env, so it must not interleave with another test that
/// reads it. `RUST_TEST_THREADS=1` is forced repo-wide (`.cargo/config.toml`), which is what makes
/// that safe here.
#[test]
fn a_malformed_kv_tier_env_is_refused_rather_than_silently_bf16() {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            std::env::remove_var("KREA_S18_KV_BITS");
            std::env::remove_var("KREA_S18_KV_GROUP_SIZE");
        }
    }
    let _guard = Guard;

    std::env::remove_var("KREA_S18_KV_BITS");
    std::env::remove_var("KREA_S18_KV_GROUP_SIZE");
    assert_eq!(
        parse_kv_tier_env().unwrap(),
        None,
        "unset is the shipped cache"
    );

    // Empty is how a shell spells "unset for this invocation" — accepted, and it means bf16.
    std::env::set_var("KREA_S18_KV_BITS", "");
    assert_eq!(parse_kv_tier_env().unwrap(), None);

    // The typo that motivates all of this.
    std::env::set_var("KREA_S18_KV_BITS", "q8");
    let err = parse_kv_tier_env().expect_err("`q8` is the label, not the value");
    assert!(err.contains("KREA_S18_KV_BITS"), "{err}");

    // A width MLX cannot express is refused here rather than deep inside the first chunk forward.
    std::env::set_var("KREA_S18_KV_BITS", "7");
    assert!(parse_kv_tier_env()
        .unwrap_err()
        .contains("affine quantization width"));

    // A valid tier resolves, group size included, and labels itself.
    std::env::set_var("KREA_S18_KV_BITS", "8");
    std::env::set_var("KREA_S18_KV_GROUP_SIZE", "32");
    let q = parse_kv_tier_env().unwrap().expect("a valid tier");
    assert_eq!(
        q,
        KvCacheQuant {
            bits: 8,
            group_size: 32
        }
    );
    assert_eq!(kv_tier_label(Some(q)), "q8/g32");
    assert_eq!(kv_tier_label(None), KV_TIER_BF16);

    // A half-set dispatch (group size, no width) is a mistake, not a bf16 request.
    std::env::remove_var("KREA_S18_KV_BITS");
    assert!(parse_kv_tier_env()
        .unwrap_err()
        .contains("KREA_S18_KV_GROUP_SIZE is set"));
}

/// Accumulated evidence from two KV tiers must be refused, not pooled — the sc-17807 analogue of the
/// duplicate-cell and mixed-bucket guards. Pooling would read as "more seeds" while silently
/// averaging two different models, and it is one `cat` away.
#[test]
fn accumulated_s18_evidence_refuses_to_pool_two_kv_tiers() {
    let bf16 = "S18CELL\tA\t7\t640x384\t45\t13\t23.0413\t19.9907\t23.0413\t12.8146\t15625190652\t\
                1.8230\t13.5111\t8.4549\tspatial-sd";
    let q8 = "S18CELL\tA\t11\t640x384\t45\t13\t23.5000\t20.0000\t23.5000\t12.9000\t9900000000\t\
              1.8000\t13.5000\t8.4000\tspatial-sd\tq8/g64";

    // Each tier on its own resolves, and carries the right label.
    assert_eq!(
        s18_sweep_from_accumulated(bf16, None)
            .expect("bf16 alone")
            .kv,
        KV_TIER_BF16,
        "a pre-sc-17807 line has no tier field and is bf16 by construction"
    );
    assert_eq!(
        s18_sweep_from_accumulated(q8, None).expect("q8 alone").kv,
        "q8/g64"
    );

    // Concatenated, they are refused.
    let mixed = format!("{bf16}\n{q8}");
    let err = s18_sweep_from_accumulated(&mixed, None)
        .expect_err("two tiers in one file must not be pooled");
    assert!(err.contains("mixes KV cache tiers"), "{err}");
    assert!(err.contains("two different models"), "{err}");
}

/// **The per-token KV figure quoted in crate prose must be the computed one (sc-17807).**
///
/// The same hole [`the_withdrawn_s18_claims_do_not_survive_in_crate_prose`] closes, for a *number*
/// rather than a claim: a 546 kB per-token cost sat in this file's own S18 doc block and in the S18
/// memory preflight's prose, 1.5× below the real figure, and nothing read it. A per-token cost that is
/// 1.5× low is not cosmetic — it is what makes a row look affordable on a host that it then takes
/// down, which is exactly what sc-17324 spent two runner crashes discovering.
///
/// The ban targets the **assertive** spellings — the withdrawn number quoted as *the* per-token cost,
/// with or without an approximation sign (the exact needles are assembled below). A sentence that names
/// the withdrawn number as withdrawn — as the two correction notes above do — is the point of the
/// correction, not a relapse, so it deliberately stays legal.
///
/// So: the stale figure is banned from the doc-bearing sources, and the figure that replaced it is
/// asserted to be the one [`KreaRealtimeConfig::kv_bytes_per_token`] computes — prose and code cannot
/// drift apart without this going red.
#[test]
fn the_kv_per_token_figure_in_crate_prose_is_the_computed_one() {
    fn normalise(src: &str) -> String {
        src.split_whitespace()
            .filter(|w| *w != "//!" && *w != "///" && *w != "//" && *w != "#")
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let per_token = cfg.kv_bytes_per_token().expect("the shipped bf16 cache");
    assert_eq!(
        per_token, 819_200,
        "800 KiB — 2 x 40 layers x 5120 x 2 bytes"
    );

    for (file, src) in [
        (
            "tests/generate_smoke.rs + tests/generate_smoke/*.rs",
            SUITE_SOURCE,
        ),
        ("src/causal.rs", include_str!("../../src/causal.rs")),
        ("src/config.rs", include_str!("../../src/config.rs")),
        ("src/t2v.rs", include_str!("../../src/t2v.rs")),
        (
            "scripts/ci/s18_memory_preflight.py",
            include_str!("../../../../../../scripts/ci/s18_memory_preflight.py"),
        ),
    ] {
        let prose = normalise(src);
        // The needles are ASSEMBLED rather than written as literals: this scan reads its own source
        // file, so writing a banned spelling out here would make the test flag its own assertion. (The
        // same reason `the_withdrawn_s18_claims_do_not_survive_in_crate_prose` scans only `src/`.)
        let n = 546;
        for stale in [
            format!("{n} kb/token"),
            format!("\u{2248}{n}"),
            format!("~{n}"),
        ] {
            assert!(
                !prose.contains(&stale),
                "{file} still quotes the WITHDRAWN `{stale}` KV cost. The derived figure is \
                 {per_token} bytes/token (800 KiB) — see KreaRealtimeConfig::kv_bytes_per_token, \
                 measured against the allocator by kv_cache_residency_at_the_production_geometry. \
                 Do not delete this assertion to make the build pass: a 1.5x-low per-token figure is \
                 what makes an unaffordable sweep row look affordable (sc-17324)."
            );
        }
        // ...and the corrected one is actually present where the cost is discussed, rather than the
        // stale figure having simply been deleted.
        assert!(
            prose.contains("800 kib") || prose.contains("819_200") || prose.contains("819,200"),
            "{file} no longer states the per-token KV cost at all — the correction was a deletion, \
             not a replacement"
        );
    }
}
