//! S18 sweep types: cells, rows, the accumulated-evidence ingest and the KV-tier labels, with
//! the ingest's weights-free gates.

use super::*;

/// One measured row of the S18 sweep, as printed by the gated real-weight run.
pub(super) struct S18Row {
    pub(super) label: String,
    pub(super) latent_frames: usize,
    /// The verdict statistic: worst trend or worst z-gated excursion (see [`worst_drift`]).
    pub(super) drift: f64,
    pub(super) trend: f64,
    pub(super) excursion: f64,
    /// Worst per-100-frame slope — the only length-comparable statistic, used for row Z.
    pub(super) slope: f64,
    /// Which descriptor component produced `drift` (see [`worst_component`]).
    pub(super) component: &'static str,
    pub(super) rolls: usize,
    /// MLX active peak for this row's AR loop — the sink/window memory cost, measured.
    pub(super) peak: usize,
    /// Mean % of pixels at or above 250 in the brightest channel — the independent quality metric.
    pub(super) clip_mean: f64,
    /// Mean |Δ| per frame over the first / last third: the freeze check.
    pub(super) head_motion: f64,
    pub(super) tail_motion: f64,
    pub(super) row: char,
    pub(super) seed: u64,
    /// AR-loop wall time in milliseconds — the throughput half of sc-17807's feasibility question.
    /// The story warned that dequantizing per read "may spend back the win in bandwidth"; without a
    /// recorded time the artifacts cannot answer that, only the live log can.
    pub(super) ar_wall_ms: u64,
}

/// One row of the S18 sweep. Rows share a prompt, seed and geometry, but **not** a clip: changing
/// `local_attn_size` changes the attention mask and therefore the sampling trajectory. Rows D, F and Z
/// are within-regime controls (same bounded local-attention path); row E is an out-of-regime reference.
pub(super) struct DriftRun {
    pub(super) row: char,
    pub(super) label: &'static str,
    pub(super) window: i64,
    pub(super) sink: usize,
    /// Latent frames for this row. Row Z is deliberately short — short enough that the shipped window
    /// never evicts — so only its *slope* is comparable with the long rows.
    pub(super) latent_frames: Option<usize>,
}

/// Number of AR chunks that append past a bounded read window. A negative window is the separate
/// global-attention regime and never evicts.
pub(super) fn eviction_rolls(latent_frames: usize, frames_per_block: usize, window: i64) -> usize {
    if window < 0 {
        return 0;
    }
    (0..latent_frames.div_ceil(frames_per_block))
        .filter(|chunk| frames_per_block * (chunk + 1) > window as usize)
        .count()
}

/// **The re-aggregation ingest must refuse duplicated evidence, not pool it.**
///
/// This is the gate for [`s18_sweep_from_accumulated`], which is otherwise reachable only through an
/// `#[ignore]`d entry point and would ship uncovered.
///
/// The duplicate case is the one that matters and it is not hypothetical: the documented workflow is
/// `cat piece-1.tsv piece-2.tsv > all.tsv`, every piece downloads under the same `s18-cells.tsv`
/// name, so including one twice is a slip rather than an act of malice. It must not degrade
/// *quietly*, because pooling a duplicate does not add noise — it removes it. `spread` is
/// `2*SD/sqrt(n)` over the sample SD, so the same three cells listed twice keep the mean and shrink
/// the interval by ~3.2x, which is the wrong direction: re-pasting evidence would buy confidence.
/// The assertion below pins exactly that, by showing the pooled sweep would have crossed from
/// unresolvable to resolved.
#[test]
fn accumulated_s18_evidence_rejects_duplicated_cells() {
    // A row A whose three seeds straddle the budget widely enough that 2*SEM exceeds the margin:
    // unresolvable at n = 3, and resolvable if the same cells are counted twice.
    let cell = |seed: u64, drift: f64| {
        format!("S18CELL\tA\t{seed}\t832x480\t45\t13\t{drift:.4}\t{drift:.4}\t0.0000\t10.0000\t18719499004\t2.0000\t14.0000\t14.0000\topp-B-Y")
    };
    // mean 20.0 against the 8.0 budget, so margin 12.0, and sample SD 12.0. At n = 3 the spread is
    // 2*12/sqrt(3) = 13.86 > 12 (unresolvable); listed twice, n = 6 and SD 10.73 give 8.76 < 12
    // (resolved). The duplicate does not add a single new measurement, and buys the answer.
    let once = [cell(7, 8.0), cell(11, 20.0), cell(23, 32.0)].join("\n");

    let sweep = s18_sweep_from_accumulated(&once, None).expect("three distinct cells parse");
    assert_eq!(sweep.cells.len(), 3, "one cell per (row, seed)");
    assert_eq!(sweep.bucket, "832x480", "bucket inferred from the evidence");
    let single = sweep.verdict();

    // Pooling the duplicate would CHANGE THE ANSWER — this is the whole reason the guard exists,
    // and asserting it keeps the guard from being weakened into a no-op later.
    let doubled_cells = {
        let evidence = parse_s18_evidence(&[once.clone(), once.clone()].join("\n")).expect("parse");
        evidence
            .iter()
            .map(|c| S18Cell {
                row: c.row,
                seed: c.seed,
                latent_frames: c.latent_frames,
                rolls: c.rolls,
                reported_drift: c.drift,
                trend: c.trend,
                excursion: c.excursion,
                slope: c.slope,
                peak_bytes: c.peak_bytes,
                clip_mean: c.clip_mean,
                head_motion: c.head_motion,
                tail_motion: c.tail_motion,
                ar_wall_ms: c.ar_wall_ms.unwrap_or(0),
                component: "",
            })
            .collect::<Vec<_>>()
    };
    let pooled = S18Sweep {
        bucket: "832x480".to_string(),
        kv: KV_TIER_BF16.to_string(),
        cells: doubled_cells,
    };
    // The claim is specifically that the duplicate defeats the POWER gate — not that it produces a
    // clean verdict (this fixture has no sink rows, so the pooled sweep stops on that instead).
    // Crossing from "cannot resolve which side of the budget" to "resolved" is the damage.
    let single_err = single.expect_err("n = 3 must be underpowered for this fixture");
    assert!(
        single_err.contains("UNDERPOWERED"),
        "fixture must start underpowered or the assertion below proves nothing: {single_err}"
    );
    let pooled_verdict = pooled.verdict();
    let still_underpowered = pooled_verdict
        .as_ref()
        .err()
        .is_some_and(|e| e.contains("UNDERPOWERED"));
    assert!(
        !still_underpowered,
        "fixture must be one where pooling DEFEATS the power gate, or the assertion below proves \
         nothing: {pooled_verdict:?}"
    );

    // ...and the ingest must not let that happen.
    let twice = [once.clone(), once.clone()].join("\n");
    let err = s18_sweep_from_accumulated(&twice, None)
        .expect_err("the same file concatenated twice must be rejected");
    assert!(
        err.contains("duplicate (row, seed)") && err.contains("A/seed 7"),
        "the rejection must name the offending cells: {err}"
    );

    // Duplicate detection is per bucket: the same (row, seed) at two geometries is two measurements.
    let cross_bucket = format!(
        "{}\n{}",
        cell(7, 2.0),
        cell(7, 2.0).replace("832x480", "640x384")
    );
    assert!(
        s18_sweep_from_accumulated(&cross_bucket, Some("832x480")).is_ok(),
        "the same seed at another bucket is not a duplicate"
    );
    let mixed = s18_sweep_from_accumulated(&cross_bucket, None)
        .expect_err("mixed buckets with no selection must be refused");
    assert!(mixed.contains("mixes buckets"), "got: {mixed}");

    // Artifact noise must survive: `VERDICT:` lines, blanks and indented cells.
    let noisy = format!(
        "  VERDICT: drift is real (whatever)\n\n  {}\n",
        once.replace('\n', "\n  ")
    );
    assert_eq!(
        s18_sweep_from_accumulated(&noisy, None)
            .expect("artifact noise is filtered, indented cells are kept")
            .cells
            .len(),
        3,
        "indented S18CELL lines must not be silently dropped"
    );
    assert!(
        s18_sweep_from_accumulated("  VERDICT: nothing here\n", None)
            .expect_err("a file with no cells is not evidence")
            .contains("no S18CELL lines"),
    );
}

/// **Structural checks must not depend on the sweep being verdict-complete.**
///
/// sc-17655 made partial dispatches the normal way to drive this sweep, and these three checks used
/// to live inside [`S18Sweep::verdict`] — reachable only when rows A/D/F were all present. A piece
/// that mis-parameterises row E or Z has to fail on its own.
#[test]
fn s18_structural_checks_run_without_a_complete_ladder() {
    let cell = |row: char, seed: u64, rolls: usize, latent: usize| S18Cell {
        row,
        seed,
        latent_frames: latent,
        rolls,
        reported_drift: 20.0,
        trend: 20.0,
        excursion: 0.0,
        slope: 10.0,
        peak_bytes: 1,
        clip_mean: 2.0,
        head_motion: 14.0,
        tail_motion: 14.0,
        ar_wall_ms: 0,
        component: "",
    };
    let sweep = |cells: Vec<S18Cell>| S18Sweep {
        bucket: "832x480".to_string(),
        kv: KV_TIER_BF16.to_string(),
        cells,
    };

    // A row E that evicted is not a global reference — caught with no A, D or F in sight.
    let e_only = sweep(vec![cell('E', 7, 3, 45)]);
    assert!(
        e_only.validate_window_dose_ladder().is_ok(),
        "no ladder here"
    );
    assert!(e_only
        .structural_checks()
        .expect_err("a rolled row E must be rejected")
        .contains("not a reference"),);
    // A row Z that evicted is not a zero-eviction floor.
    assert!(sweep(vec![cell('Z', 7, 1, 6)])
        .structural_checks()
        .expect_err("an evicting row Z must be rejected")
        .contains("zero-eviction floor"));
    // Too short a clip is rejected on a row-A-only piece.
    assert!(sweep(vec![cell('A', 7, 2, 9)])
        .structural_checks()
        .expect_err("a short clip must be rejected")
        .contains("not a long clip"));
    // Well-formed pieces pass.
    assert!(sweep(vec![cell('A', 7, 13, 45), cell('Z', 11, 0, 6)])
        .structural_checks()
        .is_ok());
}

/// Rebuild an [`S18Sweep`] from accumulated `S18CELL` lines.
///
/// Split out of [`s18_verdict_from_accumulated_cells`] so the ingest is testable with no weights,
/// no environment and no files — see [`accumulated_s18_evidence_rejects_duplicated_cells`]. The
/// entry point itself is `#[ignore]`d, so without this split none of the parsing, bucket selection
/// or duplicate rejection below would be covered by anything.
pub(super) fn s18_sweep_from_accumulated(
    src: &str,
    want_bucket: Option<&str>,
) -> Result<S18Sweep, String> {
    // The uploaded artifact interleaves `  VERDICT:` lines with the cells, and concatenating
    // several leaves blank lines — `parse_s18_evidence` rejects both, so filter to cells first.
    // `trim_start` because a hand-assembled file is the expected input and indented cells are the
    // obvious way to lose one silently.
    let cells_only: String = src
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("S18CELL"))
        .collect::<Vec<_>>()
        .join("\n");
    if cells_only.is_empty() {
        return Err("contains no S18CELL lines — it is not S18 sweep evidence".to_string());
    }
    let evidence = parse_s18_evidence(&cells_only)?;

    let mut buckets: Vec<String> = evidence.iter().map(|c| c.bucket.clone()).collect();
    buckets.sort();
    buckets.dedup();
    let bucket = match want_bucket {
        Some(want) => {
            if !buckets.iter().any(|b| b == want) {
                return Err(format!(
                    "bucket `{want}` is not present (buckets: {buckets:?})"
                ));
            }
            want.to_string()
        }
        None => {
            if buckets.len() != 1 {
                return Err(format!(
                    "mixes buckets {buckets:?} — set KREA_S18_BUCKET to choose one, because a \
                     verdict is per geometry and pooling them would compare different clips"
                ));
            }
            buckets[0].clone()
        }
    };

    let selected: Vec<&S18Evidence> = evidence.iter().filter(|c| c.bucket == bucket).collect();

    // sc-17807 — the KV storage tier is part of what a cell measured, so a file that mixes tiers is
    // refused rather than pooled. Pooling would look like "more seeds" while actually comparing two
    // different models: quantizing the cache perturbs the very coherence this sweep measures, which
    // is why the knob ships opt-in and why its arm is recorded separately from the bf16 baseline.
    let mut tiers: Vec<String> = selected
        .iter()
        .map(|c| c.kv.clone().unwrap_or_else(|| KV_TIER_BF16.to_string()))
        .collect();
    tiers.sort();
    tiers.dedup();
    let kv = match tiers.as_slice() {
        [only] => only.clone(),
        _ => {
            return Err(format!(
                "mixes KV cache tiers {tiers:?} at {bucket} — a verdict is per tier, and pooling \
                 them would compare two different models while looking like added seeds. Split the \
                 evidence and re-aggregate each tier on its own."
            ))
        }
    };

    // Duplicate (row, seed) cells are REJECTED here rather than left to the verdict rule.
    // `validate_window_dose_ladder` does reject them — but only once rows A, D and F are all
    // present; it returns early otherwise, and a piecewise sweep routinely holds one or two rows,
    // which is precisely when this function is used. Pooling is not a harmless double count:
    // `spread` is 2*SD/sqrt(n) over the SAMPLE SD, so including the same evidence twice shrinks
    // the interval by ~3.2x at n = 3 and can flip an UNDERPOWERED sweep into a confident verdict.
    // Concatenating one `s18-cells.tsv` twice is a single `cat` away, so the ingest owns this.
    let mut seen: Vec<(char, u64)> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();
    for cell in &selected {
        let key = (cell.row, cell.seed);
        if seen.contains(&key) {
            duplicates.push(format!("{}/seed {}", cell.row, cell.seed));
        } else {
            seen.push(key);
        }
    }
    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        return Err(format!(
            "duplicate (row, seed) cells at {bucket}: {}. The same configuration appears more than \
             once — most likely a piece was included twice. Pooling duplicates SHRINKS the \
             between-seed spread and manufactures confidence, so the input must be de-duplicated \
             rather than accepted.",
            duplicates.join(", ")
        ));
    }

    let cells: Vec<S18Cell> = selected
        .iter()
        .map(|c| S18Cell {
            row: c.row,
            seed: c.seed,
            latent_frames: c.latent_frames,
            rolls: c.rolls,
            reported_drift: c.drift,
            trend: c.trend,
            excursion: c.excursion,
            slope: c.slope,
            peak_bytes: c.peak_bytes,
            clip_mean: c.clip_mean,
            head_motion: c.head_motion,
            tail_motion: c.tail_motion,
            // `None` for a pre-sc-17807 line, which has no time in it. Reported as "not recorded"
            // rather than as a zero that would read like an instantaneous run.
            ar_wall_ms: c.ar_wall_ms.unwrap_or(0),
            // `S18Cell` holds `&'static str` so the recorded tables can be `const`. Rather than
            // leak the parsed string, bind it to the matching [`DESCRIPTOR_NAMES`] entry; anything
            // else (including the historical rows that predate the field) becomes "". Nothing in
            // `verdict`/`summary` reads this — only the recorded-evidence tests do — so an unknown
            // descriptor is worth neither a leak nor a hard error here.
            component: c
                .component
                .as_deref()
                .and_then(|name| DESCRIPTOR_NAMES.iter().find(|known| **known == name))
                .copied()
                .unwrap_or(""),
        })
        .collect();

    Ok(S18Sweep { bucket, kv, cells })
}

/// One measured (row, seed) cell of the S18 sweep.
#[derive(Clone, Copy, Debug)]
pub(super) struct S18Cell {
    /// `A` shipped · `B` sink 1 · `C` sink 3 · `D` wide15 · `F` wide30 · `E` global reference ·
    /// `Z` zero-roll.
    pub(super) row: char,
    pub(super) seed: u64,
    pub(super) latent_frames: usize,
    pub(super) rolls: usize,
    /// The `S18CELL` verdict value as printed by the original run.
    pub(super) reported_drift: f64,
    /// Worst absolute one-way component trend, 0..255.
    pub(super) trend: f64,
    /// Worst absolute z-gated component excursion, 0..255.
    pub(super) excursion: f64,
    /// Worst absolute component slope over the post-roll segment, per 100 output frames. Recorded
    /// because it is the **only** statistic comparable across rows of different clip length, which is
    /// the entire point of row Z — see [`S18Sweep::rate_floor_clause`]. Same component as
    /// [`trend`](Self::trend) (they differ by the constant post-segment length), so the segment length
    /// is recoverable as `100 * trend / slope`.
    pub(super) slope: f64,
    /// MLX active peak bytes for this cell's AR loop.
    pub(super) peak_bytes: usize,
    /// Mean percentage of pixels at or above 250 in the brightest channel.
    pub(super) clip_mean: f64,
    /// Mean absolute RGB frame-to-frame delta over the first third, in 0..255 units.
    pub(super) head_motion: f64,
    /// Mean absolute RGB frame-to-frame delta over the last third of the clip, in 0..255 units.
    ///
    /// This is the sweep's freeze check. It is retained in the recorded cells so the crate prose's
    /// count and rounded range are derived from the raw `S18CELL` evidence rather than hand-copied.
    pub(super) tail_motion: f64,
    /// AR-loop wall time in milliseconds. Recorded because sc-17807's feasibility question has a
    /// throughput half — dequantizing the read window per layer could have spent the memory win back
    /// in bandwidth, and a cell with no time in it cannot say either way. `0` for the historical
    /// cells, which predate the field.
    pub(super) ar_wall_ms: u64,
    /// Which [`DESCRIPTOR_NAMES`] component actually produced [`drift`](Self::drift) for this cell.
    ///
    /// Recorded so the reader can check *which channel* scored a row rather than taking the earlier
    /// "saturation" framing of the mode on trust. It is not the same thing as the corroborating
    /// `report_artifacts` saturation statistic, and this field is what makes that checkable.
    pub(super) component: &'static str,
}

impl S18Cell {
    /// Retain every verdict and freeze-evidence field emitted by the live real-weight sweep.
    pub(super) fn from_measured(row: &S18Row) -> Self {
        Self {
            row: row.row,
            seed: row.seed,
            latent_frames: row.latent_frames,
            rolls: row.rolls,
            reported_drift: row.drift,
            trend: row.trend,
            excursion: row.excursion,
            slope: row.slope,
            peak_bytes: row.peak,
            clip_mean: row.clip_mean,
            head_motion: row.head_motion,
            tail_motion: row.tail_motion,
            ar_wall_ms: row.ar_wall_ms,
            component: row.component,
        }
    }

    /// The gated statistic: trend AND excursion must be inside budget, i.e. their max must be.
    pub(super) fn drift(&self) -> f64 {
        self.trend.abs().max(self.excursion.abs())
    }

    /// Output frames in this cell's post-roll segment, recovered from the trend/slope pair.
    pub(super) fn post_len(&self) -> Option<f64> {
        (self.slope.abs() > 1e-9).then(|| 100.0 * self.trend.abs() / self.slope.abs())
    }
}

#[test]
fn an_s18_cell_retains_tail_motion_from_the_live_sweep() {
    let measured = S18Row {
        label: "test".into(),
        latent_frames: 45,
        drift: 1.0,
        trend: 2.0,
        excursion: 3.0,
        slope: 4.0,
        component: "luma-mean",
        rolls: 5,
        peak: 6,
        clip_mean: 7.0,
        head_motion: 8.0,
        tail_motion: 12.3456,
        ar_wall_ms: 0,
        row: 'A',
        seed: 9,
    };
    let cell = S18Cell::from_measured(&measured);
    assert_eq!(
        cell.tail_motion, measured.tail_motion,
        "the live S18 result-to-record conversion must not discard the freeze measurement"
    );
}

#[test]
fn the_s18_bounded_dose_ladder_spans_thirteen_ten_and_five_rolls() {
    let (latent_frames, frames_per_block) = (45, 3);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 6), 13);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 15), 10);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 30), 5);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, -1), 0);
}

/// Outcome of the within-regime A/D/F window dose-response. The decision statistic is the mean,
/// across matched seeds, of the OLS slope of drift against roll count at windows 6/15/30.
///
/// **Symmetric by construction.** An earlier form fired whenever `D + combined_spread >= A`, i.e.
/// whenever the wider window failed to *beat* the shipped one, and then asserted an unqualified
/// falsification ("eviction is not the mechanism"). That is a non-inferiority test wearing a
/// falsification's clothes: noisier data makes it *easier* to satisfy, which is the same inversion the
/// max−min → 2·SEM change fixed elsewhere in this file. It is also inconsistent with the sink
/// comparison, which correctly applies a *resolvability* standard to exactly this shape of question.
///
/// So: neither direction may be asserted unless the fitted slope clears its between-seed 2·SEM, and
/// when it does not, the outcome names the effect size the enlarged roll span **can** exclude.
pub(super) enum WindowAttribution {
    /// One or more of A/D/F was not measured — the three-dose response was not run.
    Unmeasured,
    /// The fitted slope is inside its 2·SEM. Nothing may be concluded in either direction.
    Unresolved {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
    /// The fitted slope is negative beyond the predeclared 2·SEM heuristic. This rejects a positive
    /// linear/monotone roll-count contribution over the measured 5–13-roll range; it does not exclude
    /// every possible cache-window mechanism.
    NoPositiveLinearDoseResponse {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
    /// The fitted slope is positive beyond the predeclared 2·SEM heuristic: more evictions predict
    /// more drift over the measured bounded-window dose range.
    PositiveLinearDoseResponse {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
}

/// A measured S18 sweep at one geometry bucket, and the decision rule applied to it. Split out of the
/// real-weight driver so the **rule** is gated in CI rather than only exercised on the gated GPU run.
#[derive(Debug)]
pub(super) struct S18Sweep {
    pub(super) bucket: String,
    /// The **KV cache storage tier** this sweep ran at (sc-17807) — `"bf16"` for the shipped cache,
    /// `"q8/g64"` and friends for a quantized one. Carried on the sweep rather than on each cell
    /// because one dispatch runs one tier, and pooling two tiers would compare different models
    /// while looking like added seeds. `s18_sweep_from_accumulated` refuses a mixed file for exactly
    /// that reason.
    pub(super) kv: String,
    pub(super) cells: Vec<S18Cell>,
}

/// The label a bf16-KV sweep carries — the shipped default, and what every cell recorded before
/// sc-17807 implicitly is.
pub(super) const KV_TIER_BF16: &str = "bf16";

/// Smallest drift difference the sc-17807 KV-tier A/B will call a regression, in 0..255 units.
///
/// Statistical resolvability is necessary but not sufficient here. The paired statistic
/// ([`S18Sweep::paired_deltas`]) is powerful precisely because it cancels content variance — which
/// means three paired deltas can land with a sample SD near zero, and then *any* consistent shift
/// clears its own 2·SEM. A 0.1/255 "regression" resolved that way is an artifact of `n = 3`, not a
/// finding, so the rule also requires the effect to be big enough to act on.
///
/// One eighth of [`DRIFT_BUDGET`] — one unit on the 0..255 scale the descriptors live on, i.e. the
/// coarsest resolution at which a difference in these statistics means anything at all.
///
/// **Declared with the tension visible**, because a floor chosen after seeing the data is worthless:
/// the sc-17807 calibration cell (row A, seed 7, 640×384) measured q8 at +0.89/255 against bf16, so
/// this floor sits *above* the one observation that existed when it was written. A resolvable
/// difference under it is REPORTED as "resolvable but IMMATERIAL" with its magnitude, never
/// silently folded into agreement.
pub(super) const KV_TIER_MATERIAL_FLOOR: f64 = DRIFT_BUDGET / 8.0;

/// The `S18CELL` / summary label for a KV storage tier: `bf16`, or `q<bits>/g<group>`.
pub(super) fn kv_tier_label(quant: Option<KvCacheQuant>) -> String {
    match quant {
        None => KV_TIER_BF16.to_string(),
        Some(q) => format!("q{}/g{}", q.bits, q.group_size),
    }
}

/// Read the sc-17807 KV storage tier from `KREA_S18_KV_BITS` / `KREA_S18_KV_GROUP_SIZE`.
///
/// **Strict on purpose.** Unset (or empty — how a shell spells "unset for this invocation") is the
/// shipped bf16 cache; anything else must parse to a tier MLX can express. The tempting
/// `.parse().ok()` form swallows a typo into `None`, which here does not mean "no tier" but "run the
/// bf16 arm for two hours while the log, the recorded cells and the A/B all say q8" — a mislabelled
/// arm compared against itself. Split out so the parse is gated without weights.
pub(super) fn parse_kv_tier_env() -> std::result::Result<Option<KvCacheQuant>, String> {
    fn field(name: &str, default: i32) -> std::result::Result<Option<i32>, String> {
        match std::env::var(name) {
            Err(_) => Ok(None),
            Ok(raw) if raw.trim().is_empty() => Ok(None),
            Ok(raw) => raw
                .trim()
                .parse::<i32>()
                .map(Some)
                .map_err(|_| format!("{name}=`{raw}` is not an integer (default {default})")),
        }
    }
    let group_size = field("KREA_S18_KV_GROUP_SIZE", 64)?.unwrap_or(64);
    let Some(bits) = field("KREA_S18_KV_BITS", 0)? else {
        // A group size with no width selects nothing and is almost certainly a half-set dispatch.
        if std::env::var("KREA_S18_KV_GROUP_SIZE").is_ok_and(|v| !v.trim().is_empty()) {
            return Err(
                "KREA_S18_KV_GROUP_SIZE is set but KREA_S18_KV_BITS is not — that selects the bf16 \
                 cache, which is not what a half-set quantization dispatch means"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let quant = KvCacheQuant { bits, group_size };
    quant.validate().map_err(|e| format!("{e}"))?;
    Ok(Some(quant))
}
