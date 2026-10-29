//! The recorded S18 evidence: its parser and the gates that hold the documented conclusion to
//! the recorded data.

use super::*;

// --- The recorded measurement -----------------------------------------------------------------
//
// The gated real-weight sweep runs on one host; the numbers it produced are recorded here so the
// documented conclusion is gated by the data rather than by prose. Regenerate with:
//
//   KREA_REALTIME_SNAPSHOT_DIR=... KREA_SMOKE_W=640 KREA_SMOKE_H=384 KREA_S18_SEEDS=7,11,23 \
//     cargo test -p mlx-gen-krea-realtime --test integration generate_smoke:: -- --ignored --nocapture \
//     long_clip_coherence_under_the_bounded_window
//
// and paste the `S18CELL` lines below.

/// The prefix of the verdict the crate documentation claims. Changing the documented conclusion means
/// changing this, which means [`the_recorded_s18_sweep_is_what_the_docs_claim`] re-checks it against
/// the recorded data.
pub(super) const RECORDED_VERDICT_PREFIX: &str = "drift is real";

#[derive(Debug)]
pub(super) struct S18Evidence {
    pub(super) row: char,
    pub(super) seed: u64,
    pub(super) bucket: String,
    /// The KV storage tier the emitting sweep ran at (sc-17807). `None` for the pre-sc-17807 lines,
    /// which are all bf16 by construction — the knob did not exist when they were measured.
    pub(super) kv: Option<String>,
    /// AR-loop wall time in milliseconds. `None` for the pre-sc-17807 lines, which predate the field.
    pub(super) ar_wall_ms: Option<u64>,
    pub(super) latent_frames: usize,
    pub(super) rolls: usize,
    pub(super) drift: f64,
    pub(super) trend: f64,
    pub(super) excursion: f64,
    pub(super) slope: f64,
    pub(super) peak_bytes: usize,
    pub(super) clip_mean: f64,
    pub(super) head_motion: f64,
    pub(super) tail_motion: f64,
    pub(super) component: Option<String>,
}

pub(super) fn parse_s18_evidence(src: &str) -> Result<Vec<S18Evidence>, String> {
    fn number<T: std::str::FromStr>(field: &str, name: &str, line: usize) -> Result<T, String> {
        field
            .parse()
            .map_err(|_| format!("line {line}: invalid {name} `{field}`"))
    }

    src.lines()
        .enumerate()
        .map(|(index, line)| {
            let line_number = index + 1;
            let fields: Vec<&str> = line.split('\t').collect();
            if !(14..=17).contains(&fields.len()) || fields[0] != "S18CELL" {
                return Err(format!(
                    "line {line_number}: expected a 14- to 17-field S18CELL record, got `{line}`"
                ));
            }
            let mut row_chars = fields[1].chars();
            let row = row_chars
                .next()
                .filter(|_| row_chars.next().is_none())
                .ok_or_else(|| format!("line {line_number}: invalid row `{}`", fields[1]))?;
            Ok(S18Evidence {
                row,
                seed: number(fields[2], "seed", line_number)?,
                bucket: fields[3].to_string(),
                kv: fields.get(15).map(|value| (*value).to_string()),
                ar_wall_ms: fields
                    .get(16)
                    .map(|value| number(value, "ar_wall_ms", line_number))
                    .transpose()?,
                latent_frames: number(fields[4], "latent_frames", line_number)?,
                rolls: number(fields[5], "rolls", line_number)?,
                drift: number(fields[6], "drift", line_number)?,
                trend: number(fields[7], "trend", line_number)?,
                excursion: number(fields[8], "excursion", line_number)?,
                slope: number(fields[9], "slope", line_number)?,
                peak_bytes: number(fields[10], "peak_bytes", line_number)?,
                clip_mean: number(fields[11], "clip_mean", line_number)?,
                head_motion: number(fields[12], "head_motion", line_number)?,
                tail_motion: number(fields[13], "tail_motion", line_number)?,
                component: fields.get(14).map(|value| (*value).to_string()),
            })
        })
        .collect()
}

/// **CI gate on the sc-17841 fix, against real cells rather than synthetic ones.**
///
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`] proves the rule emits the rate clause on an
/// `Unmeasured` attribution using constructed sweeps. This proves it on the measurement that
/// motivated the fix, and pins the finding itself so it stops living in a story description: at the
/// **shipping** bucket, as at 640×384, the zero-eviction row runs FASTER than the shipped row, so
/// there is still no measured same-content rate floor anywhere.
///
/// Before sc-17841 the verdict for exactly these cells contained no rate sentence at all — the
/// clause was computed and then discarded with the empty attribution clause — so three cells of real
/// GPU time produced a number that had to be worked out by hand. This test is what makes that
/// unable to happen again quietly.
#[test]
fn the_sc17324_832_sweep_reports_its_rate_comparison() {
    // The committed raw lines and the typed table must not drift apart, exactly as
    // `the_recorded_s18_sweep_is_what_the_docs_claim` requires of the sc-15127 record.
    let evidence = parse_s18_evidence(include_str!("../fixtures/s18_sc17324_832_cells.tsv"))
        .expect("the committed sc-17324 S18CELL evidence must parse");
    assert_eq!(
        evidence.len(),
        MEASURED_832_SC17324.len(),
        "the committed sc-17324 evidence and its table have different cell counts"
    );
    for raw in &evidence {
        assert_eq!(
            raw.bucket, "832x480",
            "this record is the 832x480 re-measurement"
        );
        let cell = MEASURED_832_SC17324
            .iter()
            .find(|c| c.row == raw.row && c.seed == raw.seed)
            .unwrap_or_else(|| {
                panic!(
                    "row {} seed {} is missing from the table",
                    raw.row, raw.seed
                )
            });
        assert_eq!(
            (
                raw.latent_frames,
                raw.rolls,
                raw.peak_bytes,
                raw.drift,
                raw.trend,
                raw.excursion,
                raw.slope,
                raw.clip_mean,
                raw.head_motion,
                raw.tail_motion,
                raw.component.as_deref(),
            ),
            (
                cell.latent_frames,
                cell.rolls,
                cell.peak_bytes,
                cell.reported_drift,
                cell.trend,
                cell.excursion,
                cell.slope,
                cell.clip_mean,
                cell.head_motion,
                cell.tail_motion,
                Some(cell.component),
            ),
            "row {} seed {} differs between the committed lines and the table",
            raw.row,
            raw.seed
        );
        assert!(
            DESCRIPTOR_NAMES.contains(&cell.component),
            "row {} seed {} records component `{}`, which is not a descriptor component",
            raw.row,
            raw.seed,
            cell.component
        );
    }

    let sweep = S18Sweep {
        bucket: "832x480".into(),
        // sc-17324's record predates the sc-17807 KV knob, so it is the shipped bf16 cache.
        kv: KV_TIER_BF16.to_string(),
        cells: MEASURED_832_SC17324.to_vec(),
    };
    println!("{}", sweep.summary());
    // The shape that motivated sc-17841: no D and no F, so the ladder cannot be fitted at all.
    assert!(
        matches!(sweep.window_attribution(), WindowAttribution::Unmeasured),
        "this record has no A/D/F ladder, so its window attribution must be Unmeasured — if that          changed, the sweep is no longer the case sc-17841 was about"
    );

    // The finding, pinned. `mean_slope` is the per-100-output-frame rate, the only statistic
    // comparable across row Z's 12-frame post segment and row A's 156.
    let (a, z) = (
        sweep.mean_slope('A').expect("row A is recorded"),
        sweep.mean_slope('Z').expect("row Z is recorded"),
    );
    assert!(
        (a - 18.8191).abs() < 0.0001 && (z - 38.2933).abs() < 0.0001,
        "the sc-17324 rate statistics drifted: A {a:.4}/100f, Z {z:.4}/100f; expected 18.8191 and          38.2933"
    );
    assert!(
        z > a,
        "row Z now runs SLOWER than row A at 832x480 ({z:.2} vs {a:.2}/100f) — a same-content rate          floor WOULD then exist at the shipping bucket, and both this crate's prose and sc-15571's          narrowed wording would need revisiting"
    );

    // ...and the verdict must actually SAY so. This is the sc-17841 regression gate on real data:
    // before the fix this string contained no rate sentence, because the attribution is Unmeasured.
    let v = sweep
        .verdict()
        .expect("the sc-17324 sweep supports a verdict");
    println!("  VERDICT: {v}");
    assert!(
        v.contains("zero-eviction row Z does NOT establish a rate floor here"),
        "the measured row-Z evidence must reach the verdict rather than being discarded with the          empty attribution clause (sc-17841): {v}"
    );
    assert!(
        v.contains("which is HIGHER than the shipped row's"),
        "row Z is faster than row A here, so the verdict must report the HIGHER branch: {v}"
    );
}

/// **CI gate on the recorded measurement.** The numbers the crate documentation cites live in
/// [`MEASURED_640`] / [`MEASURED_832`], and this applies the *same* [`S18Sweep::verdict`] rule to them
/// that the gated GPU run applies to fresh ones. Editing the docs' conclusion without re-measuring, or
/// pasting in a sweep that does not actually support it, turns this red.
///
/// It also gates the thing the review caught: **both buckets are recorded and both are checked.** The
/// 832×480 bucket is the crate default and a shipping bucket; a result there that disagrees with
/// 640×384 cannot be set aside.
#[test]
fn the_recorded_s18_sweep_is_what_the_docs_claim() {
    for (bucket, cells) in [("640x384", MEASURED_640), ("832x480", MEASURED_832)] {
        assert!(
            !cells.is_empty(),
            "the {bucket} sweep is not recorded — the documented conclusion has no data behind it"
        );
        let sweep = S18Sweep {
            bucket: bucket.to_string(),
            kv: KV_TIER_BF16.to_string(),
            cells: cells.to_vec(),
        };
        println!("{}", sweep.summary());
        let v = sweep.verdict();
        println!("  {bucket}: {v:?}");
        assert_eq!(
            v.as_ref().map(|s| s.starts_with(RECORDED_VERDICT_PREFIX)),
            Ok(true),
            "the recorded {bucket} sweep yields `{v:?}`, not the documented \
             `{RECORDED_VERDICT_PREFIX}...` — src/t2v.rs and src/lib.rs are claiming something the \
             data does not say"
        );
        // The enlarged three-dose A/D/F ladder still does not resolve the attribution in either
        // bucket. Gate that wording so an unsupported positive or negative causal claim cannot return.
        let v = v.expect("verdict asserted above");
        assert!(
            v.contains("attribution to the bounded KV window is NOT resolvable"),
            "the recorded {bucket} sweep no longer returns an UNRESOLVED attribution — the docs and \
             sc-15571 say it is unresolved, so one of them is now wrong: {v}"
        );
        assert!(
            !v.contains("NOT attributable"),
            "the recorded {bucket} sweep returned a window falsification — that claim was withdrawn \
             in review and must not come back without a slope that clears the predeclared 2*SEM \
             heuristic: {v}"
        );

        // Bind the published exact statistics, not only the broad UNRESOLVED classification. This
        // makes A, D and F all evidence-bearing: changing any dose or silently replacing the
        // three-point OLS with an endpoint contrast changes at least one pinned number.
        let slopes = sweep.window_dose_slopes();
        let slope = mean_f64(&slopes);
        let uncertainty = 2.0 * std_f64(&slopes) / (slopes.len() as f64).sqrt();
        let practical_floor = 8.0 * (slope.abs() + uncertainty);
        let (expected_slope, expected_uncertainty, expected_floor) = match bucket {
            "640x384" => (0.5714, 1.8970, 19.75),
            "832x480" => (-0.2780, 1.6781, 15.65),
            _ => unreachable!(),
        };
        assert!(
            (slope - expected_slope).abs() < 0.0001
                && (uncertainty - expected_uncertainty).abs() < 0.0001
                && (practical_floor - expected_floor).abs() < 0.01,
            "{bucket} exact A/D/F statistics drifted: slope {slope:+.4}, 2*SEM \
             {uncertainty:.4}, practical floor {practical_floor:.2}; expected \
             {expected_slope:+.4}, {expected_uncertainty:.4}, {expected_floor:.2}"
        );
        // Every cell must name the descriptor component that actually scored it, and it must be a
        // real one. This is what makes "which channel failed?" checkable from the table.
        for c in &sweep.cells {
            assert!(
                DESCRIPTOR_NAMES.contains(&c.component),
                "{bucket} row {} seed {} records component `{}`, which is not a descriptor component",
                c.row,
                c.seed,
                c.component
            );
        }
        // ...and specifically: the corroborating `report_artifacts` saturation statistic is NOT the
        // channel that scored row A. Recording the winners is what lets the docs say that instead of
        // implying the two are the same mode.
        let a_components: Vec<&str> = sweep
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.component)
            .collect();
        assert!(
            !a_components.contains(&"saturation"),
            "{bucket} row A is now scored by the `saturation` component — the docs say the \
             corroborating saturation statistic is a DIFFERENT channel from the one that scored row \
             A, so that wording must be revisited: {a_components:?}"
        );
        // Replication is the other half of what the review demanded: a recorded bucket with one seed
        // per row cannot support a ranking claim, so it must not be recorded as if it could.
        for row in ['A', 'B', 'C', 'D', 'F'] {
            // ...and they must be DIFFERENT seeds. Three cells from one seed is determinism, not
            // replication, and would give a spread of zero that makes every comparison look resolved.
            let mut seeds: Vec<u64> = sweep
                .cells
                .iter()
                .filter(|c| c.row == row)
                .map(|c| c.seed)
                .collect();
            let n = seeds.len();
            seeds.sort_unstable();
            seeds.dedup();
            assert_eq!(
                seeds.len(),
                n,
                "{bucket} row {row} records a repeated seed — that is determinism, not replication"
            );
            if !sweep.of(row).is_empty() {
                assert!(
                    sweep.of(row).len() >= 3,
                    "{bucket} row {row} is recorded with only {} seed(s) — the documented \
                     conclusion would rest on an unreplicated cell",
                    sweep.of(row).len()
                );
            }
        }
    }

    // The within-regime rate comparison the DRIFT_BUDGET doc used to assert and never compute. It is
    // recorded here as a FINDING, not a floor: row Z's 12-frame slope is higher than row A's over 156
    // frames, so it does not bound row A and the budget rests on the synthetic controls alone. If a
    // future measurement makes Z genuinely lower, this goes red and the doc's narrowed wording — and
    // sc-15571's — must be revisited, because then a same-content floor WOULD exist.
    let sweep = S18Sweep {
        bucket: "640x384".into(),
        kv: KV_TIER_BF16.to_string(),
        cells: MEASURED_640.to_vec(),
    };
    let (z, a) = (
        sweep.mean_slope('Z').expect("row Z is recorded at 640x384"),
        sweep.mean_slope('A').expect("row A is recorded at 640x384"),
    );
    println!("  rate floor: {}", sweep.rate_floor_clause());
    assert!(
        z > a,
        "row Z's zero-eviction rate is now {z:.2}/100f against row A's {a:.2} — Z is LOWER, so it \
         does bound row A's rate after all and the docs' \"no measured same-content rate floor\" \
         wording is stale"
    );
    // Row Z's post segment really is short — 12 output frames against row A's 156 — which is the
    // stated reason its slope does not extrapolate. Pin it so the argument cannot rot.
    let z_post = sweep.mean_post_len('Z').expect("row Z has a slope");
    let a_post = sweep.mean_post_len('A').expect("row A has a slope");
    assert!(
        z_post < a_post / 4.0,
        "row Z's post segment is {z_post:.0} output frames against row A's {a_post:.0} — the docs' \
         \"a 12-frame OLS slope does not extrapolate across 156\" argument no longer describes the \
         data"
    );

    // The committed fixture is the durable copy of the exact S18CELL lines emitted by the preserved
    // real-weight logs. Parse every field and compare it to the hand-shaped tables used by the verdict
    // so neither representation can drift independently.
    let evidence = parse_s18_evidence(include_str!("../fixtures/s18_recorded_cells.tsv"))
        .expect("the committed S18CELL evidence must parse");
    let cells: Vec<&S18Cell> = MEASURED_640.iter().chain(MEASURED_832).collect();
    assert_eq!(
        evidence.len(),
        34,
        "the committed S18CELL evidence changed cell count; reconcile it with the source logs"
    );
    assert_eq!(
        evidence.len(),
        cells.len(),
        "the committed S18CELL evidence and MEASURED tables have different cell counts"
    );

    let mut seen = Vec::new();
    for raw in &evidence {
        let table = match raw.bucket.as_str() {
            "640x384" => MEASURED_640,
            "832x480" => MEASURED_832,
            other => panic!("fixture contains an unrecognised S18 bucket `{other}`"),
        };
        let matches: Vec<&S18Cell> = table
            .iter()
            .filter(|cell| cell.row == raw.row && cell.seed == raw.seed)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "{} row {} seed {} must identify exactly one MEASURED cell, found {}",
            raw.bucket,
            raw.row,
            raw.seed,
            matches.len()
        );
        let cell = matches[0];
        assert_eq!(
            (raw.latent_frames, raw.rolls, raw.peak_bytes),
            (cell.latent_frames, cell.rolls, cell.peak_bytes),
            "{} row {} seed {} integer fields differ from the committed S18CELL evidence",
            raw.bucket,
            raw.row,
            raw.seed
        );
        assert_eq!(
            [
                raw.drift,
                raw.trend,
                raw.excursion,
                raw.slope,
                raw.clip_mean,
                raw.head_motion,
                raw.tail_motion,
            ],
            [
                cell.reported_drift,
                cell.trend,
                cell.excursion,
                cell.slope,
                cell.clip_mean,
                cell.head_motion,
                cell.tail_motion,
            ],
            "{} row {} seed {} numeric fields differ from the committed S18CELL evidence",
            raw.bucket,
            raw.row,
            raw.seed
        );
        assert_eq!(
            cell.reported_drift,
            cell.drift(),
            "{} row {} seed {} records a drift that is not max(trend, excursion)",
            raw.bucket,
            raw.row,
            raw.seed
        );
        if raw.row == 'F' {
            assert!(
                raw.component.is_some(),
                "{} row F seed {} must retain the current 15-field S18CELL schema from the exact \
                 sc-15585 log; dropping the winning component weakens the evidence",
                raw.bucket,
                raw.seed
            );
        }
        if let Some(component) = &raw.component {
            assert_eq!(
                component, cell.component,
                "{} row {} seed {} component differs from the committed S18CELL evidence",
                raw.bucket, raw.row, raw.seed
            );
        }
        seen.push((raw.bucket.as_str(), raw.row, raw.seed));
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        evidence.len(),
        "the committed S18CELL evidence contains a duplicate bucket/row/seed tuple"
    );

    assert!(
        evidence
            .iter()
            .all(|cell| cell.tail_motion.is_finite() && cell.tail_motion > 0.0),
        "every recorded S18 cell must retain its positive finite tail-motion freeze measurement"
    );
    let min_tail = evidence
        .iter()
        .map(|cell| cell.tail_motion)
        .fold(f64::INFINITY, f64::min);
    let max_tail = evidence
        .iter()
        .map(|cell| cell.tail_motion)
        .fold(f64::NEG_INFINITY, f64::max);
    let documented = format!(
        "tail motion {min_tail:.1}–{max_tail:.1}/255 per frame across all {} recorded cells",
        evidence.len()
    );
    let prose = include_str!("../../src/t2v.rs")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        prose.contains(&documented),
        "src/t2v.rs must derive its freeze-evidence count/range from the recorded sweep: expected \
         `{documented}`"
    );

    let row_e = evidence
        .iter()
        .find(|cell| cell.bucket == "640x384" && cell.row == 'E')
        .expect("the 640x384 global reference row is committed");
    let row_e_gib = row_e.peak_bytes as f64 / 1024_f64.powi(3);
    let row_e_doc = format!("{row_e_gib:.2} GiB MLX peak");
    let recorded_prose = SUITE_SOURCE
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let row_e_sentence = format!("Row E is `n = 1`: its {row_e_doc}");
    assert!(
        recorded_prose.contains(&row_e_sentence),
        "the recorded-sweep prose must state Row E memory in binary GiB from its recorded byte \
         count: expected `{row_e_sentence}`"
    );
}
