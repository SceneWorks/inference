//! The S18 sweep verdict: per-row statistics, window attribution and the KV-tier comparison.

use super::*;

impl S18Sweep {
    pub(super) fn of(&self, row: char) -> Vec<f64> {
        self.cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.drift())
            .collect()
    }

    pub(super) fn mean(&self, row: char) -> Option<f64> {
        let v = self.of(row);
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// The between-seed uncertainty heuristic on a row's **mean**: twice the standard error.
    /// This is the sweep's predeclared resolvability rule, not a Student-t 95% confidence interval
    /// (with three seeds, that would need a materially larger multiplier). `None` if the row has fewer
    /// than two seeds — in which case there is **no variance estimate at all**, which the verdict must
    /// say out loud.
    ///
    /// Deliberately not the max−min range: that *grows* with the number of seeds, so a rule gated on
    /// it would get harder to satisfy the more evidence you collected. This shrinks as 1/√n, which is
    /// what makes "add seeds until the comparison resolves" a real option.
    pub(super) fn spread(&self, row: char) -> Option<f64> {
        let v = self.of(row);
        (v.len() >= 2).then(|| 2.0 * std_f64(&v) / (v.len() as f64).sqrt())
    }

    /// Mean of a row's worst per-100-frame slope — the length-comparable rate statistic.
    pub(super) fn mean_slope(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.slope.abs())
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// Mean post-roll segment length of a row, in output frames.
    pub(super) fn mean_post_len(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .filter_map(|c| c.post_len())
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// The **within-regime rate** comparison the [`DRIFT_BUDGET`] doc used to assert and never compute:
    /// row Z is the same attention regime with *zero* evictions, so if a per-frame rate floor exists
    /// for this content, Z is where it lives, and [`S18Cell::slope`] is the only statistic comparable
    /// across Z's short clip and A's long one.
    ///
    /// It is **reported, not gated**, because on the recorded data it does not separate: see the
    /// returned text. A short-segment OLS slope is dominated by ordinary frame-to-frame motion, so Z's
    /// slope is *larger* than A's, and extrapolating it over A's post segment over-predicts A's own
    /// measured drift. Row Z cannot be lengthened either — the shipped 6-latent-frame window evicts as
    /// soon as the clip passes 6 latent frames, so a longer zero-eviction row does not exist at the
    /// shipped window. The consequence is stated rather than hidden: this sweep has **no measured
    /// same-content rate floor**, and [`DRIFT_BUDGET`] is bracketed by the synthetic motion/jitter and
    /// failure-shape controls alone.
    pub(super) fn rate_floor_clause(&self) -> String {
        let (Some(a_slope), Some(z_slope)) = (self.mean_slope('A'), self.mean_slope('Z')) else {
            return String::new();
        };
        let a_post = self.mean_post_len('A').unwrap_or(0.0);
        let z_post = self.mean_post_len('Z').unwrap_or(0.0);
        let a_trend = self.mean_slope('A').unwrap_or(0.0) * a_post / 100.0;
        let predicted = z_slope * a_post / 100.0;
        if z_slope < a_slope {
            format!(
                " The within-regime zero-eviction row Z does NOT establish a rate floor here: over \
                 its {z_post:.0}-output-frame post segment it runs at {z_slope:.2}/100f against the \
                 shipped row's {a_slope:.2}/100f over {a_post:.0} — Z is lower, so the shipped row's \
                 rate does exceed the zero-eviction rate."
            )
        } else {
            format!(
                " The within-regime zero-eviction row Z does NOT establish a rate floor here, and \
                 the sweep says so rather than omitting it: over its {z_post:.0}-output-frame post \
                 segment Z runs at {z_slope:.2}/100f, which is HIGHER than the shipped row's \
                 {a_slope:.2}/100f over {a_post:.0} frames. Extrapolated across the long clip that \
                 rate predicts {predicted:.1}/255 of drift — {:.1}x what row A actually measures \
                 ({a_trend:.1}/255) — so the extrapolation over-predicts and a {z_post:.0}-frame OLS \
                 slope (dominated by ordinary frame-to-frame motion) is not extrapolable. Row Z \
                 cannot be lengthened: the shipped window evicts as soon as the clip passes 6 latent \
                 frames, so no longer zero-eviction row exists at the shipped window. This sweep \
                 therefore has NO measured same-content rate floor, and the budget is bracketed by \
                 the synthetic motion/jitter and failure-shape controls alone.",
                predicted / a_trend.max(1e-6)
            )
        }
    }

    /// Per-seed OLS slopes of drift against roll count over the A/D/F ladder. Matching by seed prevents
    /// ordinary seed-to-seed content variation from masquerading as a dose response.
    pub(super) fn window_dose_slopes(&self) -> Vec<f64> {
        let mut seeds: Vec<u64> = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.seed)
            .collect();
        seeds.sort_unstable();
        seeds.dedup();
        seeds
            .into_iter()
            .filter_map(|seed| {
                let points: Vec<(f64, f64)> = ['A', 'D', 'F']
                    .iter()
                    .filter_map(|&row| {
                        self.cells
                            .iter()
                            .find(|c| c.row == row && c.seed == seed)
                            .map(|c| (c.rolls as f64, c.drift()))
                    })
                    .collect();
                if points.len() != 3 {
                    return None;
                }
                let x_mean = points.iter().map(|p| p.0).sum::<f64>() / 3.0;
                let y_mean = points.iter().map(|p| p.1).sum::<f64>() / 3.0;
                let denom = points.iter().map(|p| (p.0 - x_mean).powi(2)).sum::<f64>();
                (denom > 0.0).then(|| {
                    points
                        .iter()
                        .map(|p| (p.0 - x_mean) * (p.1 - y_mean))
                        .sum::<f64>()
                        / denom
                })
            })
            .collect()
    }

    /// The slope is a paired-seed statistic over the exact three-dose experiment: 45 latent frames
    /// with A/D/F at 13/10/5 rolls. Refuse malformed evidence instead of silently accepting duplicate
    /// seeds, a shorter F clip, a different window width, a non-finite statistic, or fitting duplicate
    /// roll counts.
    pub(super) fn validate_window_dose_ladder(&self) -> std::result::Result<(), String> {
        if ['A', 'D', 'F'].iter().any(|&row| self.of(row).is_empty()) {
            return Ok(());
        }
        let seeds = |row| {
            let mut values: Vec<u64> = self
                .cells
                .iter()
                .filter(|c| c.row == row)
                .map(|c| c.seed)
                .collect();
            values.sort_unstable();
            values
        };
        let (a_seeds, d_seeds, f_seeds) = (seeds('A'), seeds('D'), seeds('F'));
        let has_duplicate = |values: &[u64]| values.windows(2).any(|pair| pair[0] == pair[1]);
        if has_duplicate(&a_seeds) || has_duplicate(&d_seeds) || has_duplicate(&f_seeds) {
            return Err(format!(
                "the A/D/F dose ladder contains duplicate seed cells: \
                 A={a_seeds:?}, D={d_seeds:?}, F={f_seeds:?}"
            ));
        }
        if a_seeds != d_seeds || a_seeds != f_seeds {
            return Err(format!(
                "the A/D/F dose ladder does not contain the same matched seeds: \
                 A={a_seeds:?}, D={d_seeds:?}, F={f_seeds:?}"
            ));
        }
        for cell in self
            .cells
            .iter()
            .filter(|c| matches!(c.row, 'A' | 'D' | 'F'))
        {
            let expected_rolls = match cell.row {
                'A' => 13,
                'D' => 10,
                'F' => 5,
                _ => unreachable!(),
            };
            if cell.latent_frames != 45 || cell.rolls != expected_rolls {
                return Err(format!(
                    "the A/D/F dose ladder must be the exact 45-latent-frame, 13/10/5-roll \
                     experiment: row {} seed {} has {} latent frames and {} rolls",
                    cell.row, cell.seed, cell.latent_frames, cell.rolls
                ));
            }
            if !cell.reported_drift.is_finite()
                || !cell.trend.is_finite()
                || !cell.excursion.is_finite()
            {
                return Err(format!(
                    "the A/D/F dose ladder contains a non-finite drift statistic: row {} seed {}",
                    cell.row, cell.seed
                ));
            }
        }
        let slopes = self.window_dose_slopes();
        if slopes.len() != a_seeds.len() || slopes.iter().any(|slope| !slope.is_finite()) {
            return Err(format!(
                "the A/D/F dose ladder did not produce one finite paired slope per seed: \
                 {} slopes for {} matched seeds",
                slopes.len(),
                a_seeds.len()
            ));
        }
        Ok(())
    }

    /// Classify the within-regime window dose-response. See [`WindowAttribution`] for why this is
    /// two-sided.
    pub(super) fn window_attribution(&self) -> WindowAttribution {
        let (Some(_d), Some(f)) = (self.mean('D'), self.mean('F')) else {
            return WindowAttribution::Unmeasured;
        };
        let f_rolls = self
            .cells
            .iter()
            .find(|c| c.row == 'F')
            .map(|c| c.rolls)
            .unwrap_or(0);
        let slopes = self.window_dose_slopes();
        let n = slopes.len();
        let slope = if slopes.is_empty() {
            0.0
        } else {
            mean_f64(&slopes)
        };
        let unc = if n >= 2 {
            2.0 * std_f64(&slopes) / (n as f64).sqrt()
        } else {
            f64::INFINITY
        };
        if slope.abs() <= unc {
            WindowAttribution::Unresolved {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        } else if slope < 0.0 {
            WindowAttribution::NoPositiveLinearDoseResponse {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        } else {
            WindowAttribution::PositiveLinearDoseResponse {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        }
    }

    /// The row with the smaller mean of `B`/`C`, i.e. the sink anchor that did best.
    pub(super) fn best_sink(&self) -> Option<(char, f64)> {
        ['B', 'C']
            .iter()
            .filter_map(|&r| self.mean(r).map(|m| (r, m)))
            .fold(None, |acc: Option<(char, f64)>, x| match acc {
                Some(a) if a.1 <= x.1 => Some(a),
                _ => Some(x),
            })
    }

    pub(super) fn summary(&self) -> String {
        let f = |r: char| match (self.mean(r), self.spread(r)) {
            (None, _) => "-".to_string(),
            (Some(m), None) => format!("{m:.2} (n=1)"),
            (Some(m), Some(s)) => format!("{m:.2} ±{s:.2} (n={}, 2*SEM)", self.of(r).len()),
        };
        format!(
            "{} KV {} — A shipped {} | B sink1 {} | C sink3 {} | D wide15 {} | F wide30 {} | E global-ref {} | budget \
             {DRIFT_BUDGET:.2}/255 (absolute)",
            self.bucket,
            self.kv,
            f('A'),
            f('B'),
            f('C'),
            f('D'),
            f('F'),
            f('E'),
        )
    }

    /// Per-seed `arm − baseline` drift deltas for a row, matched by seed — `None` when the two arms
    /// share fewer than two seeds on this row.
    ///
    /// Pairing is what makes this A/B powerful enough to be worth running. Both arms drive the same
    /// seeds, so the same init noise produces the same content; the unpaired difference of means
    /// carries the full seed-to-seed content variance, which at the recorded 640×384 scatter cannot
    /// resolve anything smaller than ~6.7/255 on row A — 84% of the whole [`DRIFT_BUDGET`]. The
    /// paired form cancels it, and it is the same matched-seed reasoning
    /// [`S18Sweep::window_dose_slopes`] already applies to the dose ladder.
    pub(super) fn paired_deltas(&self, baseline: &S18Sweep, row: char) -> Option<Vec<f64>> {
        let mut deltas = Vec::new();
        for cell in self.cells.iter().filter(|c| c.row == row) {
            if let Some(base) = baseline
                .cells
                .iter()
                .find(|b| b.row == row && b.seed == cell.seed)
            {
                deltas.push(cell.drift() - base.drift());
            }
        }
        (deltas.len() >= 2).then_some(deltas)
    }

    /// Mean AR-loop wall time (ms) for a row, or `None` when no cell recorded one (the pre-sc-17807
    /// cells store `0`, which would read as an instantaneous run rather than as absent).
    pub(super) fn mean_wall_ms(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row && c.ar_wall_ms > 0)
            .map(|c| c.ar_wall_ms as f64)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// Mean MLX active peak (bytes) for a row, or `None` if it was not measured.
    pub(super) fn mean_peak(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.peak_bytes as f64)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// **The sc-17807 A/B: is a quantized KV cache worse than the bf16 baseline?**
    ///
    /// `self` is the quantized arm, `baseline` the bf16 one at the same geometry. Three things are
    /// gated, and all three can fail:
    ///
    ///   1. **Drift**, per row, on the same gated statistic [`S18Sweep::verdict`] uses. The primary
    ///      form is the **matched-seed paired delta** (`arm(seed) − base(seed)`), following this
    ///      file's own dose-ladder precedent: both arms run the same seeds, so the same init noise
    ///      produces the same content, and pairing cancels the content variance that otherwise
    ///      dominates. It matters — against the recorded 640×384 scatter the unpaired form cannot
    ///      resolve anything smaller than ~6.7/255 on row A, i.e. 84% of the whole
    ///      [`DRIFT_BUDGET`]. The unpaired difference-of-means (`sqrt(a² + b²)` on the two arms'
    ///      2·SEM) is the fallback when the seeds do not match.
    ///   2. **Memory**, per row. A quantized cache whose measured peak is not *below* the baseline's
    ///      has bought nothing, and the entire story is the saving — so this is an `Err`, not a
    ///      footnote. There is a real mechanism for it to fail (a dequantized read window that stops
    ///      being freed per layer would make the packed path *worse* at peak than the dense one).
    ///   3. **Rankability.** If no row resolves in either direction, the answer is "this A/B did not
    ///      resolve", not "not resolvably worse" — the headline is what a reader takes away, and it
    ///      must not assert a conclusion the seeds cannot support. The minimum detectable effect is
    ///      reported so an underpowered result reads as underpowered.
    ///
    /// **What this rule is NOT.** It is not a drift pass/fail: [`S18Sweep::verdict`] is applied to
    /// the arm separately (that is item 3's "using the existing verdict rule"), but at these
    /// geometries row A already scores far outside [`DRIFT_BUDGET`] in *both* arms, so `verdict`
    /// takes its drift-attribution branch and returns `Ok` for either. It answers "does the recorded
    /// sweep still say what the docs say", not "is q8 acceptable". This comparison is what answers
    /// the second question, and only relative to bf16.
    pub(super) fn kv_tier_comparison(
        &self,
        baseline: &S18Sweep,
    ) -> std::result::Result<String, String> {
        if self.bucket != baseline.bucket {
            return Err(format!(
                "cannot compare KV tiers across geometries ({} vs {}) — the clips differ",
                self.bucket, baseline.bucket
            ));
        }
        if self.kv == baseline.kv {
            return Err(format!(
                "both arms ran the same KV tier ({}) — that is a repeat, not an A/B",
                self.kv
            ));
        }
        if baseline.kv != KV_TIER_BF16 {
            return Err(format!(
                "the baseline arm is `{}`, not the shipped `{KV_TIER_BF16}` cache — the comparison \
                 is defined against what ships",
                baseline.kv
            ));
        }

        let mut lines: Vec<String> = Vec::new();
        let mut worse: Vec<String> = Vec::new();
        let (mut compared, mut rankable) = (0usize, 0usize);
        for row in ['A', 'B', 'C', 'D', 'F'] {
            let (Some(arm), Some(base)) = (self.mean(row), baseline.mean(row)) else {
                continue;
            };
            compared += 1;
            // Matched-seed paired deltas when the two arms ran the same seeds — the powerful form,
            // and the one this file already uses for the dose ladder. `None` when the seeds do not
            // line up, in which case the unpaired difference of means stands in.
            let paired = self.paired_deltas(baseline, row);
            let (delta, unc, form) = match &paired {
                Some(d) if d.len() >= 2 => (
                    mean_f64(d),
                    Some(2.0 * std_f64(d) / (d.len() as f64).sqrt()),
                    format!("paired, n={}", d.len()),
                ),
                _ => (
                    arm - base,
                    match (self.spread(row), baseline.spread(row)) {
                        (Some(a), Some(b)) => Some((a * a + b * b).sqrt()),
                        _ => None,
                    },
                    "unpaired".to_string(),
                ),
            };
            let verdict = match unc {
                None => "UNRANKABLE (an arm has one seed — no variance estimate, so this row is \
                     ABSENT from the comparison rather than agreeing with it)"
                    .to_string(),
                Some(u) => {
                    rankable += 1;
                    // TWO conditions, reported separately. Statistical resolvability alone is not
                    // enough: three paired deltas can have a sample SD near zero by chance, and a
                    // 0.1/255 "regression" resolved that way is an artifact of a small sample, not a
                    // finding. Materiality alone is not enough either — that is just eyeballing.
                    let resolved = delta.abs() > u;
                    let material = delta.abs() >= KV_TIER_MATERIAL_FLOOR;
                    match (resolved, material, delta > 0.0) {
                        (false, _, _) => format!(
                            "unresolved ({form}; |Δ| {:.2} <= 2*SEM {u:.2}, so this A/B can only \
                             exclude a regression larger than {:.2}/255)",
                            delta.abs(),
                            u.max(KV_TIER_MATERIAL_FLOOR)
                        ),
                        (true, false, _) => format!(
                            "resolvable but IMMATERIAL ({form}; Δ {delta:+.2} beyond 2*SEM \
                             {u:.2}, but under the {KV_TIER_MATERIAL_FLOOR:.2}/255 floor)"
                        ),
                        (true, true, true) => {
                            worse.push(format!("{row} +{delta:.2}/255 ({form}, 2*SEM {u:.2})"));
                            format!("WORSE by {delta:.2} beyond 2*SEM {u:.2} ({form})")
                        }
                        (true, true, false) => {
                            format!("better by {:.2} beyond 2*SEM {u:.2} ({form})", -delta)
                        }
                    }
                }
            };
            lines.push(format!(
                "row {row}: {} {arm:.2} vs {KV_TIER_BF16} {base:.2} — {verdict}",
                self.kv
            ));
        }
        if compared == 0 {
            return Err(format!(
                "no bounded row is present in both arms at {} — there is nothing to compare",
                self.bucket
            ));
        }
        // The memory half — GATED, not reported. A quantized cache that does not lower the measured
        // peak has bought nothing, and "the KV dominates AR video memory" is the whole story.
        let mut mem: Vec<String> = Vec::new();
        let mut no_saving: Vec<String> = Vec::new();
        for row in ['A', 'B', 'C', 'D', 'F'] {
            if let (Some(arm), Some(base)) = (self.mean_peak(row), baseline.mean_peak(row)) {
                mem.push(format!(
                    "{row} {:.2}->{:.2} GiB ({:.2}x)",
                    base / 1024f64.powi(3),
                    arm / 1024f64.powi(3),
                    arm / base
                ));
                if arm >= base {
                    no_saving.push(format!("{row} {:.2}x", arm / base));
                }
            }
        }
        // The throughput half — reported, because the story's feasibility question explicitly asks
        // whether dequantizing per read spends the memory win back in bandwidth.
        let mut wall: Vec<String> = Vec::new();
        for row in ['A', 'B', 'C', 'D', 'F'] {
            if let (Some(arm), Some(base)) = (self.mean_wall_ms(row), baseline.mean_wall_ms(row)) {
                wall.push(format!(
                    "{row} {:.0}->{:.0} s ({:.2}x)",
                    base / 1000.0,
                    arm / 1000.0,
                    arm / base
                ));
            }
        }
        if !no_saving.is_empty() {
            return Err(format!(
                "KV tier `{}` did NOT lower the measured MLX active peak at {} on {} — the tier \
                 exists to buy memory, so a coherence result without a saving beside it is not a \
                 pass. Peaks: {}.\n  {}",
                self.kv,
                self.bucket,
                no_saving.join(", "),
                mem.join(", "),
                lines.join("\n  ")
            ));
        }

        if !worse.is_empty() {
            return Err(format!(
                "KV tier `{}` is RESOLVABLY WORSE than the shipped bf16 cache at {} on {}. A \
                 cheaper cache that costs long-clip coherence is not a win, whatever it saves \
                 ({}).\n  {}",
                self.kv,
                self.bucket,
                worse.join(", "),
                if mem.is_empty() {
                    "no peaks recorded".to_string()
                } else {
                    mem.join(", ")
                },
                lines.join("\n  ")
            ));
        }
        // Rankability decides the HEADLINE, which is the line a reader takes away. A comparison in
        // which nothing resolved has not shown agreement; it has shown that these seeds cannot tell.
        let headline = if rankable == 0 {
            format!(
                "NO ROW WAS RANKABLE — KV tier `{}` vs the shipped bf16 cache at {} is UNDECIDED \
                 over {compared} bounded row(s) (an arm has one seed, so there is no variance \
                 estimate). This is not agreement. Add seeds (KREA_S18_SEEDS) to both arms",
                self.kv, self.bucket
            )
        } else {
            format!(
                "KV tier `{}` is not resolvably worse than the shipped bf16 cache at {} over \
                 {rankable} rankable of {compared} bounded row(s)",
                self.kv, self.bucket
            )
        };
        Ok(format!(
            "{headline}. MLX active peak: {}. AR wall: {}.\n  {}",
            if mem.is_empty() {
                "not recorded".to_string()
            } else {
                mem.join(", ")
            },
            if wall.is_empty() {
                "not recorded".to_string()
            } else {
                wall.join(", ")
            },
            lines.join("\n  ")
        ))
    }

    /// Checks that a measurement is **structurally** what it claims to be, independent of whether
    /// there is enough of it to conclude anything.
    ///
    /// These three were inside [`S18Sweep::verdict`], which meant they only ran on a sweep complete
    /// enough to reach a verdict. Since sc-17655 the sweep is dispatched in row-sized pieces and a
    /// piece is *usually* not verdict-complete, so a mis-parameterised row would have sailed through
    /// green with a "partial sweep" line and only surfaced at re-aggregation — after the GPU time was
    /// already spent. They are cheap and they are about the rows themselves, so both the live sweep
    /// and the re-aggregation entry point now run them unconditionally.
    ///
    /// The shipped-row check is skipped when row A is absent, which is legitimate for a piece; row A
    /// being *required* is [`S18Sweep::verdict`]'s concern, not this one's.
    pub(super) fn structural_checks(&self) -> std::result::Result<(), String> {
        if let Some(shipped_rolls) = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.rolls)
            .max()
        {
            if shipped_rolls < 10 {
                return Err(format!(
                    "the shipped row only rolled the window {shipped_rolls} times — that is not a \
                     long clip, so it cannot answer the long-clip question. Raise \
                     KREA_S18_LATENT_FRAMES."
                ));
            }
        }
        if self.cells.iter().any(|c| c.row == 'E' && c.rolls != 0) {
            return Err(
                "the global reference row rolled the window — it is not a reference".into(),
            );
        }
        if self.cells.iter().any(|c| c.row == 'Z' && c.rolls != 0) {
            return Err(
                "the zero-roll row Z evicted — its clip is too long to be a zero-eviction floor"
                    .into(),
            );
        }
        Ok(())
    }

    /// The verdict, or the reason this sweep cannot support one.
    ///
    /// The rule is deliberately conservative about **statistical power**: a single seed per row gives
    /// no variance estimate, and this sweep's config-to-config differences are of the same order as its
    /// seed-to-seed scatter. So the rule refuses to *rank configs* unless the margin it would rank them
    /// by exceeds the measured scatter; what it will still do with one seed is state whether gross
    /// drift was observed, which is a much weaker claim and is labelled as such.
    pub(super) fn verdict(&self) -> std::result::Result<String, String> {
        let shipped_rolls = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.rolls)
            .max()
            .ok_or_else(|| "no shipped (A) row was measured — nothing to conclude".to_string())?;
        self.structural_checks()?;
        self.validate_window_dose_ladder()?;
        let a = self.mean('A').expect("row A measured above");
        let margin = (a - DRIFT_BUDGET).abs();
        // Which side of the budget the SHIPPED row is on is decided by the shipped row's own
        // between-seed spread, not by the noisiest row in the sweep.
        let a_spread = self.spread('A');
        let resolvable = match a_spread {
            None => false,
            Some(s) => s < margin,
        };

        if !resolvable {
            let power = match a_spread {
                None => "one seed per row — NO variance estimate".to_string(),
                Some(s) => format!("between-seed spread {s:.2}/255 vs a {margin:.2}/255 margin"),
            };
            // Underpowered to rank, but a *gross* result is still reportable: if every measured
            // bounded row is far inside the budget, "no gross drift observed" survives the scatter.
            let worst_bounded = ['A', 'B', 'C', 'D', 'F']
                .iter()
                .filter_map(|&r| self.mean(r))
                .fold(0.0f64, f64::max);
            if worst_bounded <= DRIFT_BUDGET / 2.0 {
                return Ok(format!(
                    "underpowered but no gross drift — every bounded row sits at or below \
                     {worst_bounded:.2}/255 against a {DRIFT_BUDGET:.2} budget over \
                     {shipped_rolls} rolls ({power}). This does NOT rank the configs and does not \
                     license a claim about any drift mode outside the measured descriptor."
                ));
            }
            return Err(format!(
                "UNDERPOWERED: the shipped row is {a:.2}/255 against a {DRIFT_BUDGET:.2} budget \
                 ({power}), and the bounded rows reach {worst_bounded:.2}. The sweep cannot resolve \
                 which side of the budget the shipped config is on. Add seeds \
                 (KREA_S18_SEEDS) before concluding."
            ));
        }

        if a <= DRIFT_BUDGET {
            // No drift: the within-regime dose-response must agree — a wider window (half the rolls)
            // cannot be materially worse, or "no drift" was an artifact of an insensitive metric.
            for (row, dose) in [('D', self.mean('D')), ('F', self.mean('F'))] {
                if let Some(d) = dose {
                    if d > DRIFT_BUDGET {
                        return Err(format!(
                            "the shipped window scored {a:.2}/255 (inside the {DRIFT_BUDGET:.2} \
                             budget) but wider bounded row {row} scored {d:.2} — the within-regime \
                             dose-response disagrees, so this run does not support a no-drift verdict"
                        ));
                    }
                }
            }
            Ok(format!(
                "coherent — the shipped bounded window holds the clip over {shipped_rolls} window \
                 rolls at {a:.2}/255 against a {DRIFT_BUDGET:.2}/255 absolute budget, with the \
                 between-seed scatter under the margin. No sink anchor warranted for the drift \
                 modes this descriptor covers."
            ))
        } else {
            // Drift is real. **Attribution first**: sc-15127 asks whether the *bounded window* costs
            // coherence, and "the clip drifts" does not answer that. The within-regime dose-response
            // (rows A/D/F, the same local-attention path at wider windows and fewer rolls) is what
            // decides it, and it is judged SYMMETRICALLY — see `WindowAttribution`.
            //
            // sc-15585 enlarges the bounded ladder from A/D (13 -> 10 rolls) to A/D/F
            // (13 -> 10 -> 5 rolls). The decision statistic is the matched-seed OLS slope over all
            // three doses, not a two-row endpoint difference.
            let head = format!(
                "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a {DRIFT_BUDGET:.2} \
                 budget)"
            );
            let rate = self.rate_floor_clause();
            let attribution = self.window_attribution();
            let d_rolls = self
                .cells
                .iter()
                .find(|c| c.row == 'D')
                .map(|c| c.rolls)
                .unwrap_or(0);
            let attrib_clause = match attribution {
                WindowAttribution::Unmeasured => String::new(),
                WindowAttribution::NoPositiveLinearDoseResponse {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => {
                    let sinks = ['B', 'C']
                        .iter()
                        .filter_map(|&r| self.mean(r).map(|m| format!("{r} {m:.2}")))
                        .collect::<Vec<_>>()
                        .join(", ");
                    // Row E is deliberately NOT cited here: it is out of regime (a different
                    // attention mask), n=1, and has no variance estimate. It is a reference, and a
                    // reference does not get to carry an attribution claim.
                    //
                    // This arm returns early, so it is the one place that interpolates `{rate}`
                    // itself — mid-paragraph, because its own sentences continue past it. Every
                    // other arm gets the rate clause appended once below (sc-17841); keep this
                    // `{rate}` when editing the wording here.
                    return Ok(format!(
                        "{head}, but it does NOT support a positive linear bounded-window dose \
                         response: the three-dose within-regime fit runs the wrong way — A/D/F span \
                         {shipped_rolls}/{d_rolls}/{f_rolls} rolls across {n} matched seeds and fit \
                         {slope:+.3}/255 per roll, clear of the predeclared 2*SEM heuristic \
                         {unc:.3}. Row F scores {f:.2}/255 against shipped A's {a:.2}; fewer \
                         evictions predict less coherence, excluding a positive linear/monotone \
                         roll-count contribution over the measured 5-13-roll range. This does not \
                         exclude a non-linear or otherwise different cache-window mechanism, and a \
                         first-chunk sink anchor is not indicated.{rate} No sink is wired. Sink \
                         rows for the record: {sinks}. The drift itself needs its own investigation."
                    ));
                }
                WindowAttribution::Unresolved {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => format!(
                    ", the attribution to the bounded KV window is NOT resolvable at this sample \
                     size — the three-dose A/D/F fit over {n} matched seeds and \
                     {shipped_rolls}/{d_rolls}/{f_rolls} rolls is {slope:+.3}/255 per roll, inside the \
                     predeclared 2*SEM heuristic of {unc:.3}, so NEITHER direction may be asserted. \
                     Row F scores \
                     {f:.2}/255 against shipped A's {a:.2}. Across the enlarged {}-roll span, this \
                     design's practical 2*SEM magnitude floor is {:.2}/255 \
                     ({:.0}% of the shipped row's drift); anything smaller remains below the \
                     practical floor.",
                    shipped_rolls.saturating_sub(f_rolls),
                    (slope.abs() + unc) * shipped_rolls.saturating_sub(f_rolls) as f64,
                    100.0 * (slope.abs() + unc) * shipped_rolls.saturating_sub(f_rolls) as f64
                        / a.max(1e-6)
                ),
                WindowAttribution::PositiveLinearDoseResponse {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => format!(
                    ", and it supports a positive bounded-window dose response: the three-dose A/D/F \
                     fit over {n} matched seeds and {shipped_rolls}/{d_rolls}/{f_rolls} rolls is \
                     {slope:+.3}/255 per roll, clear of the predeclared 2*SEM heuristic {unc:.3}. Row \
                     F scores {f:.2}/255 against shipped A's {a:.2}; more evictions predict more drift \
                     across the measured bounded-window dose range. This is evidence for a linear \
                     roll-count contribution in that range, not proof that every drift mechanism is \
                     cache-window driven."
                ),
            };
            // The rate floor is an A-vs-Z comparison — "does the shipped window's drift rate exceed
            // the zero-eviction rate on the same content" — and is INDEPENDENT of the A/D/F dose
            // ladder. It used to be interpolated inside the attribution arms only, so a sweep whose
            // attribution is `Unmeasured` discarded the measured row-Z evidence along with the empty
            // attribution clause and never printed it. That is the shape new sweeps now take at the
            // shipping bucket: row F needs ~49 GiB at 832x480 against `nax-macos-2`'s ~101 GiB
            // (the committed `MEASURED_832` record predates that host and still carries one), so a
            // fresh 832x480 sweep there has no row F at all, and sc-17324's three row-Z cells of
            // real GPU time went unreported (sc-17841). Appended once here so no attribution outcome
            // can drop it again.
            let attrib_and_rate = if rate.is_empty() {
                attrib_clause
            } else if attrib_clause.is_empty() {
                // `head` ends on `)`, not a full stop, and the rate clause is a sentence.
                format!(".{rate}")
            } else {
                format!("{attrib_clause}{rate}")
            };
            // The remaining question is whether the anchored rows repair it — a SECOND comparison
            // with its own power problem, so it gets its own resolvability check.
            let Some((row, best_sink)) = self.best_sink() else {
                return Err(format!(
                    "the shipped window drifted {a:.2}/255 past the {DRIFT_BUDGET:.2} budget but no \
                     sink row was measured — run rows B and C before concluding"
                ));
            };
            let threshold = a * 0.6;
            // The repair claim is a comparison between two noisy means, so the gap must clear the
            // scatter of BOTH rows before it can be called either way.
            let combined = self.spread('A').unwrap_or(f64::INFINITY)
                + self.spread(row).unwrap_or(f64::INFINITY);
            if (threshold - best_sink).abs() < combined {
                return Ok(format!(
                    "{head}{attrib_and_rate} And the sink anchor's effect on it is NOT resolvable at \
                     this sample size either: the best sink row ({row}) is {best_sink:.2}/255 \
                     against a {threshold:.2} repair threshold, a gap of {:.2} inside a combined \
                     between-seed scatter of {combined:.2}. No sink is wired — permanently-resident \
                     KV must not be bought on an unresolved comparison.",
                    (threshold - best_sink).abs()
                ));
            }
            if best_sink >= threshold {
                return Ok(format!(
                    "{head}{attrib_and_rate} And a first-chunk sink anchor does NOT repair it: the \
                     best sink row ({row}) only reached {best_sink:.2}/255 against a {threshold:.2} \
                     repair threshold. No sink is wired; sc-15127 needs a different anchor."
                ));
            }
            Ok(format!(
                "{head}{attrib_and_rate} And a first-chunk sink anchor repairs it to \
                 {best_sink:.2}/255 (row {row})."
            ))
        }
    }
}
