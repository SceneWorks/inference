use super::*;

#[test]
fn public_decode_peaks_are_the_planners_full_output_cases() {
    assert_eq!(
        conservative_video_decode_peak_bytes_for_vae(WanVae::VAE_TILING, 64, 64, 9),
        Some(241_975_296)
    );
    assert_eq!(
        conservative_video_decode_peak_bytes_for_vae(Wan22Vae::VAE_TILING, 64, 64, 9),
        Some(126_812_160)
    );
    assert_eq!(
        conservative_video_decode_peak_bytes_for_vae(VaeTiling::QWEN_IMAGE, 64, 64, 9),
        None
    );
    assert_eq!(
        conservative_video_decode_peak_bytes_for_vae(WanVae::VAE_TILING, 0, 64, 9),
        None
    );
}

#[test]
fn full_output_peak_bounds_every_valid_tiling_shape() {
    use mlx_gen::tiling::{SpatialTiling, TemporalTiling};

    let (width, height, frames) = (1280, 720, 121);
    for vae in [WanVae::VAE_TILING, Wan22Vae::VAE_TILING] {
        let full =
            conservative_video_decode_peak_bytes_for_vae(vae, width, height, frames).unwrap();
        for tiling in [
            TilingConfig::spatial_only(768, 64),
            TilingConfig::spatial_only(192, 64),
            TilingConfig::temporal_only(96, 24),
            TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: 384,
                    overlap_px: 64,
                }),
                temporal: Some(TemporalTiling {
                    tile_frames: 32,
                    overlap_frames: 8,
                }),
            },
        ] {
            let tiled = video_decode_peak_bytes_for_vae_and_tiling(
                vae,
                width,
                height,
                frames,
                Some(&tiling),
            )
            .unwrap();
            assert!(tiled <= full, "{vae:?} {tiling:?}: {tiled} > {full}");
        }
    }
}

#[test]
fn selected_vae22_plan_preserves_exact_edge_and_observable_spatial_carrier() {
    let (tiling, profile) = selected_vae22_plan_with_budget(480, 480, 1, 768, 64, 100.0).unwrap();
    assert_eq!(
        tiling
            .spatial
            .map(|spatial| (spatial.tile_px, spatial.overlap_px)),
        Some((768, 64)),
        "the admitted carrier stays observable even below its cap"
    );
    assert_eq!(tiling.temporal.map(|tile| tile.tile_frames), None);
    assert_eq!(profile.resident_decoder_bytes_included(), 0);
    assert_eq!(
        profile.working_set_bytes(),
        video_decode_peak_bytes_for_vae_and_tiling(
            Wan22Vae::VAE_TILING,
            480,
            480,
            1,
            Some(&tiling),
        )
        .unwrap()
    );

    let (tiling, _) = selected_vae22_plan_with_budget(1280, 704, 97, 448, 64, 1_000.0).unwrap();
    assert_eq!(
        tiling
            .spatial
            .map(|spatial| (spatial.tile_px, spatial.overlap_px)),
        Some((448, 64)),
        "the full-spatial candidate must not bypass an admitted lower cap"
    );
}

#[test]
fn selected_vae22_plan_uses_real_temporal_candidates_and_refuses_overbudget() {
    // At this output/edge, 96 frames projects to ~67.5 GiB while 64 projects to ~47.2 GiB.
    // The injected 60 GiB budget therefore selects the exact measured 64/24 temporal row.
    let (tiling, profile) = selected_vae22_plan_with_budget(1280, 704, 193, 448, 64, 60.0).unwrap();
    assert_eq!(
        tiling
            .spatial
            .map(|spatial| (spatial.tile_px, spatial.overlap_px)),
        Some((448, 64))
    );
    assert_eq!(
        tiling
            .temporal
            .map(|temporal| (temporal.tile_frames, temporal.overlap_frames)),
        Some((64, 24))
    );
    let exact = video_decode_peak_bytes_for_vae_and_tiling(
        Wan22Vae::VAE_TILING,
        1280,
        704,
        193,
        Some(&tiling),
    )
    .unwrap();
    assert_eq!(profile.working_set_bytes(), exact);
    assert!(exact as f64 / (1024.0 * 1024.0 * 1024.0) <= 60.0);

    assert!(selected_vae22_plan_with_budget(1280, 704, 193, 448, 64, 6.0).is_err());
}
use std::collections::{BTreeMap, BTreeSet};

const WAN_OVERLAP_EVIDENCE: [(&str, &str); 4] = [
    (
        "z16-640",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../docs/migration/evidence/sc-15445/z16-640.log"
        )),
    ),
    (
        "z16-832",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../docs/migration/evidence/sc-15445/z16-832.log"
        )),
    ),
    (
        "z48-640",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../docs/migration/evidence/sc-15445/z48-640.log"
        )),
    ),
    (
        "z48-832",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../docs/migration/evidence/sc-15445/z48-832.log"
        )),
    ),
];
const WAN_OVERLAP_Z48_832_CLEAN_QUALITY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../docs/migration/evidence/sc-15445/z48-832-quality-clean.log"
));

fn evidence_result(log: &'static str) -> &'static str {
    let mut results = log.lines().filter_map(|line| line.strip_prefix("RESULT "));
    let result = results
        .next()
        .expect("evidence log must contain one RESULT");
    assert!(
        results.next().is_none(),
        "evidence log must contain exactly one RESULT"
    );
    result
}

fn evidence_fields(log: &'static str) -> BTreeMap<&'static str, &'static str> {
    evidence_result(log)
        .split_ascii_whitespace()
        .map(|field| {
            field
                .split_once('=')
                .expect("every RESULT field must be key=value")
        })
        .collect()
}

fn evidence_f64(fields: &BTreeMap<&str, &str>, key: &str) -> f64 {
    fields
        .get(key)
        .unwrap_or_else(|| panic!("evidence RESULT missing {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("evidence RESULT {key} must be numeric"))
}

fn evidence_i32(fields: &BTreeMap<&str, &str>, key: &str) -> i32 {
    fields
        .get(key)
        .unwrap_or_else(|| panic!("evidence RESULT missing {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("evidence RESULT {key} must be an integer"))
}

fn evidence_pair(fields: &BTreeMap<&str, &str>, key: &str) -> (usize, usize) {
    let value = fields
        .get(key)
        .unwrap_or_else(|| panic!("evidence RESULT missing {key}"));
    let (old, half) = value
        .split_once("=>")
        .unwrap_or_else(|| panic!("evidence RESULT {key} must be old=>half"));
    (
        old.parse()
            .unwrap_or_else(|_| panic!("evidence RESULT {key} old must be an integer")),
        half.parse()
            .unwrap_or_else(|_| panic!("evidence RESULT {key} half must be an integer")),
    )
}

fn evidence_range(fields: &BTreeMap<&str, &str>, key: &str) -> (f64, f64) {
    let value = fields
        .get(key)
        .unwrap_or_else(|| panic!("evidence RESULT missing {key}"));
    let (min, max) = value
        .split_once("..")
        .unwrap_or_else(|| panic!("evidence RESULT {key} must be min..max"));
    (
        min.parse()
            .unwrap_or_else(|_| panic!("evidence RESULT {key} min must be numeric")),
        max.parse()
            .unwrap_or_else(|_| panic!("evidence RESULT {key} max must be numeric")),
    )
}

fn evidence_result_checksum() -> u64 {
    WAN_OVERLAP_EVIDENCE
        .iter()
        .map(|(_, log)| evidence_result(log))
        .chain(std::iter::once(evidence_result(
            WAN_OVERLAP_Z48_832_CLEAN_QUALITY,
        )))
        .flat_map(|result| result.bytes().chain(std::iter::once(b'\n')))
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

/// Pin the actual callback order for all four curated samplers. Heun and DPM++ SDE add
/// intermediate evaluations, but their sigma streams still cross the expert boundary exactly
/// once and never return to the high expert.
#[test]
fn every_wan_curated_sampler_crosses_the_expert_boundary_once_per_eval() {
    use mlx_gen::gen_core::sampling::{sampler_by_name, CpuLatentOps, FlowModelSampling};
    use mlx_gen::TimestepConvention;

    let sigmas = compute_sigmas(6, 5.0, 1000);
    let boundary = 875.0_f32;
    let ops = CpuLatentOps;
    let model = FlowModelSampling::new(TimestepConvention::Sigma);
    for name in ["euler_ancestral", "heun", "dpmpp_sde", "ddim"] {
        let sampler = sampler_by_name::<CpuLatentOps>(name).expect("curated sampler");
        let mut routed_high = Vec::new();
        let mut denoise = |_x: &Vec<f32>, sigma: f32| {
            routed_high.push((sigma * 1000.0).trunc() >= boundary);
            Ok(vec![0.0])
        };
        sampler
            .sample(&ops, &model, &mut denoise, vec![1.0], &sigmas, 42)
            .unwrap();

        assert!(
            routed_high.contains(&true),
            "{name} must evaluate the high expert"
        );
        assert!(
            routed_high.contains(&false),
            "{name} must evaluate the low expert"
        );
        let transitions = routed_high
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        assert_eq!(
            transitions, 1,
            "{name} must make exactly one high-to-low swap"
        );
        assert!(
            routed_high.windows(2).all(|pair| pair[0] || !pair[1]),
            "{name} must never route low-to-high"
        );
    }
}

// ── sc-12736: the A14B expert-swap residency contract (mlx Pillar-1). ────────────────────────

/// [`crossing_index`] is the prefix/suffix boundary the resident loop and the sequential swap both
/// split on — it must be **exactly** the per-step `t ≥ boundary` choice. Pinned against a
/// hand-rolled per-step scan over a monotonically-decreasing integer-timestep schedule (T2V's 875
/// boundary), including the two corners (never crossed / crossed at step 0).
#[test]
fn crossing_index_equals_the_per_step_boundary_choice() {
    let boundary = 875.0_f32; // 0.875 · 1000 (T2V)
    let timesteps = [1000.0, 950.0, 900.0, 875.0, 874.0, 500.0, 100.0, 0.0];
    let k = crossing_index(&timesteps, boundary);
    // Per-step witness: the first index whose `t < boundary` (t == boundary stays HIGH).
    let expected = timesteps.iter().position(|&t| t < boundary).unwrap();
    assert_eq!(k, expected);
    assert_eq!(k, 4, "875 (== boundary) is HIGH; 874 is the first LOW step");
    for (i, &t) in timesteps.iter().enumerate() {
        let is_high_by_split = i < k;
        let is_high_by_step = t >= boundary;
        assert_eq!(
            is_high_by_split, is_high_by_step,
            "step {i} (t={t}) disagrees between the split and the per-step rule"
        );
    }
    // Never crossed → all high (k == len); crossed at step 0 → all low (k == 0).
    assert_eq!(crossing_index(&[1000.0, 900.0, 880.0], boundary), 3);
    assert_eq!(crossing_index(&[800.0, 700.0], boundary), 0);
}

/// A liveness witness for the expert-swap residency tests (mirrors candle's `LiveTracker`): bumps a
/// shared live-counter on construction and drops it on `Drop`, recording the peak concurrency and an
/// ordered load/drop/note log.
struct LiveTracker {
    live: std::cell::Cell<usize>,
    peak: std::cell::Cell<usize>,
    log: std::cell::RefCell<Vec<&'static str>>,
}
impl LiveTracker {
    fn new() -> Self {
        Self {
            live: std::cell::Cell::new(0),
            peak: std::cell::Cell::new(0),
            log: std::cell::RefCell::new(Vec::new()),
        }
    }
    fn born(&self, tag: &'static str) {
        self.live.set(self.live.get() + 1);
        if self.live.get() > self.peak.get() {
            self.peak.set(self.live.get());
        }
        self.log.borrow_mut().push(tag);
    }
    fn died(&self, tag: &'static str) {
        self.live.set(self.live.get() - 1);
        self.log.borrow_mut().push(tag);
    }
    fn note(&self, tag: &'static str) {
        self.log.borrow_mut().push(tag);
    }
}

/// Stands in for a loaded 14B expert: its lifetime on the live-counter is exactly the expert's
/// residency window inside [`staged_expert_swap`].
struct ExpertWitness<'a> {
    tracker: &'a LiveTracker,
    drop_tag: &'static str,
}
impl<'a> ExpertWitness<'a> {
    fn new(tracker: &'a LiveTracker, born_tag: &'static str, drop_tag: &'static str) -> Self {
        tracker.born(born_tag);
        Self { tracker, drop_tag }
    }
}
impl Drop for ExpertWitness<'_> {
    fn drop(&mut self) {
        self.tracker.died(self.drop_tag);
    }
}

/// The Pillar-1 invariant (sc-12736): the two experts are **never co-resident**. Driven through the
/// production [`staged_expert_swap`] with `0 < k < steps` (a genuine swap), the peak live-expert
/// count is 1 and the high expert drops (and is evicted) before the low expert loads — a drop-order
/// witness, structural rather than a VRAM read (MLX's buffer cache makes a live memory probe blind
/// to the drop until `clear_cache`).
#[test]
fn expert_swap_is_never_co_resident_and_high_drops_before_low_loads() {
    let tracker = LiveTracker::new();
    let mut state = ();
    let out = staged_expert_swap(
        3, // k: 0 < k < steps → both experts own steps → a genuine swap
        8, // steps
        &mut state,
        |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
        |_w, _st| Ok(()),
        |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
        |_w, _st| Ok(()),
        || {
            tracker.note("evict");
            Ok(())
        },
    );
    assert!(out.is_ok());
    assert_eq!(
        tracker.peak.get(),
        1,
        "the two 14B experts must NEVER be co-resident (the whole Pillar-1 win)"
    );
    assert_eq!(
        *tracker.log.borrow(),
        vec![
            "load-high",
            "drop-high",
            "evict", // high freed + cache flushed …
            "load-low",
            "drop-low",
            "evict", // … BEFORE low ever loads
        ],
        "high must be dropped AND evicted before the low expert loads (evict-then-load)"
    );
}

/// Mutation-check (sc-12736 acceptance): force both experts resident and confirm the
/// never-co-resident assertion regresses — proving the passing test above is not a default-value
/// false green. This is the exact both-resident behavior the sequential path removes (the resident
/// `denoise_moe` holds both for the whole loop): binding `high` and `low` in one scope co-resides
/// them, and the SAME liveness witness now reports peak concurrency 2, so `peak == 1` goes RED.
#[test]
fn forcing_both_experts_resident_regresses_the_never_co_resident_assertion() {
    let tracker = LiveTracker::new();
    {
        // MUTATION: the inactive expert is NOT dropped before the next loads.
        let _high = ExpertWitness::new(&tracker, "load-high", "drop-high");
        let _low = ExpertWitness::new(&tracker, "load-low", "drop-low");
        tracker.note("both-resident-denoise");
    }
    assert_eq!(
        tracker.peak.get(),
        2,
        "the forced-both-resident mutation co-resides the two experts"
    );
    assert!(
        tracker.peak.get() > 1,
        "the never-co-resident assertion (peak == 1) MUST fail under the both-resident mutation — \
             the passing test genuinely discriminates co-residence, it is not a false green"
    );
}

/// [`staged_expert_swap`] skips loading the expert whose step range is empty (memory-optimal single
/// crossing): `k == 0` loads only the low expert, `k == steps` only the high expert.
#[test]
fn expert_swap_skips_the_expert_that_owns_no_steps() {
    // k == 0 → all-low: the high loader is never called.
    let low_only = LiveTracker::new();
    let mut st = ();
    staged_expert_swap(
        0,
        8,
        &mut st,
        |_st| Ok(ExpertWitness::new(&low_only, "load-high", "drop-high")),
        |_w, _st| Ok(()),
        |_st| Ok(ExpertWitness::new(&low_only, "load-low", "drop-low")),
        |_w, _st| Ok(()),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(
        *low_only.log.borrow(),
        vec!["load-low", "drop-low"],
        "k == 0 must load ONLY the low expert"
    );

    // k == steps → all-high: the low loader is never called.
    let high_only = LiveTracker::new();
    let mut st = ();
    staged_expert_swap(
        8,
        8,
        &mut st,
        |_st| Ok(ExpertWitness::new(&high_only, "load-high", "drop-high")),
        |_w, _st| Ok(()),
        |_st| Ok(ExpertWitness::new(&high_only, "load-low", "drop-low")),
        |_w, _st| Ok(()),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(
        *high_only.log.borrow(),
        vec!["load-high", "drop-high"],
        "k == steps must load ONLY the high expert"
    );
}

/// The evict runs after each expert is **used and dropped**, before the next loads (sc-12736): the
/// MLX free-after-drop discipline — `clear_cache` returns the dropped expert's pages to the OS once
/// its arrays are freed and before the incoming expert allocates.
#[test]
fn expert_swap_evicts_after_each_expert_drops_and_before_the_next_loads() {
    let tracker = LiveTracker::new();
    let mut st = ();
    staged_expert_swap(
        3,
        8,
        &mut st,
        |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
        |_w, _st| {
            tracker.note("use-high");
            Ok(())
        },
        |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
        |_w, _st| {
            tracker.note("use-low");
            Ok(())
        },
        || {
            tracker.note("evict");
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        *tracker.log.borrow(),
        vec![
            "load-high",
            "use-high",
            "drop-high",
            "evict", //
            "load-low",
            "use-low",
            "drop-low",
            "evict",
        ],
        "each expert must be used, dropped, then evicted before the next loads"
    );
}

/// A load failure on the low expert still drops the (already-used-and-evicted) high expert via the
/// explicit drop before the `?` — no leak, peak stays 1, and the error propagates.
#[test]
fn expert_swap_propagates_a_low_load_failure_after_dropping_high() {
    let tracker = LiveTracker::new();
    let mut st = ();
    let out: Result<()> = staged_expert_swap(
        3,
        8,
        &mut st,
        |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
        |_w, _st| Ok(()),
        |_st| Err(Error::Msg("low expert OOM".into())),
        |_w: &ExpertWitness, _st| Ok(()),
        || Ok(()),
    );
    assert!(matches!(out, Err(Error::Msg(_))));
    assert_eq!(
        *tracker.log.borrow(),
        vec!["load-high", "drop-high"],
        "high must have dropped before the low load was even attempted"
    );
    assert_eq!(tracker.peak.get(), 1);
}

/// **sc-12349: the z16 selector tiles past the write bound**, no matter how much memory the machine
/// has. Wan z16 writes 96 channels at full output resolution, so at 720p only 24 output frames fit
/// under `MAX_WRITABLE_ELEMS` — while the shipped `frame_num` is 81.
///
/// On MLX 0.31.2 taking the single pass past the cap decoded silently-wrong pixels (the conv3d
/// corruption). **sc-12748: on 0.32.0 that single pass is now correct** (#3524), so this cap is
/// defense-in-depth rather than a correctness necessity — but the selector still tiles past it (a
/// cheap, always-correct default), which this pins on an unlimited budget (the sharpest form).
#[test]
fn z16_tiles_past_the_write_bound_on_an_unlimited_budget() {
    let (h, w, f) = (720i32, 1280i32, 81i32);
    assert!(
        (f as i64) > WanVae::VAE_TILING.writable_frame_cap(h, w),
        "test precondition: 81 frames at 720p must exceed the z16 write cap ({})",
        WanVae::VAE_TILING.writable_frame_cap(h, w)
    );

    let cfg = plan_z16_tiling(h, w, f, f64::INFINITY)
        .expect("an unlimited budget must still yield a plan, not a budget error")
        .expect(
            "past the write bound the z16 decode MUST tile even with infinite memory — None here \
                 is the silently-wrong single pass sc-12349 exists to prevent",
        );

    // The chosen tile must itself be writable. z16's smallest temporal candidate is 32 frames,
    // above the 24-frame cap, so spatial tiling has to carry it — which is exactly why the bound
    // is checked against the whole tile volume rather than the frame count alone.
    let tf = cfg
        .temporal
        .map(|t| t.tile_frames as i64)
        .unwrap_or(f as i64);
    let (th, tw) = cfg
        .spatial
        .map(|s| {
            (
                (s.tile_px as i64).min(h as i64),
                (s.tile_px as i64).min(w as i64),
            )
        })
        .unwrap_or((h as i64, w as i64));
    let write = WanVae::VAE_TILING.full_res_channels as i64 * tf * th * tw;
    assert!(
        write <= mlx_gen::tiling::MAX_WRITABLE_ELEMS,
        "the selected z16 tile writes {write} elements, past the bound — plan {cfg:?}"
    );
}

#[test]
fn denoise_peak_estimate_matches_5b_measurements() {
    // sc-4986 anchors (real 5B, dim 3072, batch 2 / CFG on). bf16 model.safetensors ≈ 11.5 GiB
    // resident; measured total denoise peak at L tokens. The estimate must land within ~2 GiB.
    let weights = (11.5 * 1024.0 * 1024.0 * 1024.0) as u64;
    for (tokens, measured_peak) in [(1760usize, 11.2_f64), (16720, 17.5), (32560, 24.9)] {
        let est = estimated_denoise_peak_gib(weights, tokens, 3072, true);
        assert!(
            (est - measured_peak).abs() < 2.0,
            "L={tokens}: estimate {est:.1} GiB vs measured {measured_peak:.1} GiB"
        );
    }
}

#[test]
fn denoise_peak_scales_with_cfg_and_tokens() {
    let w = 10u64 << 30; // 10 GiB
                         // CFG doubles the activation term.
    let on = estimated_denoise_peak_gib(w, 32560, 3072, true);
    let off = estimated_denoise_peak_gib(w, 32560, 3072, false);
    assert!(
        on - 10.0 > 1.9 * (off - 10.0),
        "CFG should ~2× the activation term"
    );
    // Monotonic in tokens.
    assert!(estimated_denoise_peak_gib(w, 40000, 3072, true) > on);
}

#[test]
fn guard_rejects_over_budget_and_passes_under() {
    use mlx_rs::memory::set_memory_limit;
    // Pin a deterministic budget (32 GiB) so the threshold is exercised on any machine, then
    // restore. set_memory_limit returns the previous value.
    let prev = set_memory_limit(32 << 30);
    // A 14B-class resident (two bf16 experts ≈ 56 GiB) blows the 32 GiB budget on weights alone.
    let res = preflight_denoise_memory_guard("wan_test", 56 << 30, 1024, 5120, true);
    // A tiny model + small request fits comfortably (10 GiB + ~1.5 GiB acts < 27 GiB safe).
    let ok = preflight_denoise_memory_guard("wan_test", 10 << 30, 5280, 3072, true);
    set_memory_limit(prev);
    assert!(
        res.is_err(),
        "56 GiB resident must be rejected under a 32 GiB budget"
    );
    assert!(ok.is_ok(), "11.5 GiB peak must pass under a 32 GiB budget");
}

#[test]
fn resolve_sampler_knobs_falls_back_to_defaults_then_request() {
    // Unset request fields take the config defaults; an unset sampler → UniPC; the seed is some
    // value (drawn fresh). This is the byte-identical inline block the four generate paths used.
    let req = GenerationRequest {
        prompt: "x".into(),
        ..Default::default()
    };
    let (steps, shift, kind, _seed) = resolve_sampler_knobs(&req, 40, 5.0);
    assert_eq!(steps, 40);
    assert_eq!(shift, 5.0);
    assert_eq!(kind, SolverKind::UniPC);

    // Explicit request fields win over the defaults, and the sampler name maps through.
    let req = GenerationRequest {
        prompt: "x".into(),
        steps: Some(12),
        scheduler_shift: Some(3.5),
        sampler: Some("euler".into()),
        seed: Some(99),
        ..Default::default()
    };
    let (steps, shift, kind, seed) = resolve_sampler_knobs(&req, 40, 5.0);
    assert_eq!((steps, shift, kind, seed), (12, 3.5, SolverKind::Euler, 99));
}

// --- sc-4998: memory-budgeted z48 vae22 decode tiling ---------------------------------------

/// The estimated peak of the chosen tiling, recomputed from a returned config + output dims (the
/// largest tile spans `min(tile_px, dim)` on each spatial axis and `min(tile_frames, f)` frames).
fn chosen_peak_gib(cfg: &TilingConfig, h: i64, w: i64, f: i64, bf16: bool) -> f64 {
    let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
    let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
    let tile_f = cfg
        .temporal
        .map(|t| (t.tile_frames as i64).min(f))
        .unwrap_or(f);
    estimated_vae22_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w, bf16)
}

#[test]
fn vae22_decode_peak_matches_wedge_anchors() {
    // sc-4998 f32 anchors (real 5B z48 vae22, M5 Max): the model must reproduce both within ~10 %.
    // A: 1024×576×97 video, 512 px / 64-frame tile → 60 GB.
    let a = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 512, 512, false);
    assert!((a - 60.0).abs() < 6.0, "anchor A estimate {a:.1} GiB vs 60");
    // B: 1280×704×145 video, 256 px / 32-frame tile → 12.6 GB.
    let b = estimated_vae22_decode_peak_gib(145, 704, 1280, 32, 256, 256, false);
    assert!(
        (b - 12.6).abs() < 2.0,
        "anchor B estimate {b:.1} GiB vs 12.6"
    );
    // sc-5039 bf16 anchors (real-weight, 1024×576×97): 768 px/64 f → 79.7 GB, 640 px/48 f →
    // 55.1 GB. bf16 must estimate below f32 and stay **conservative** (never below the measured
    // peak — the guard must not under-shoot). Tile dims use the selector's nominal frame count.
    let bf16_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, true);
    let f32_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, false);
    assert!(bf16_768 < f32_768, "bf16 peak must be below f32");
    assert!(
        bf16_768 >= 79.7,
        "bf16 768/64 estimate {bf16_768:.1} under-shoots 79.7"
    );
    let bf16_640 = estimated_vae22_decode_peak_gib(97, 576, 1024, 48, 576, 640, true);
    assert!(
        bf16_640 >= 55.1,
        "bf16 640/48 estimate {bf16_640:.1} under-shoots 55.1"
    );
}

#[test]
fn vae22_tiling_single_pass_when_small() {
    // A short, low-res clip fits a single-pass decode comfortably → no tiling.
    let plan = plan_vae22_tiling(256, 256, 33, 40.0, false).unwrap();
    assert!(
        plan.is_none(),
        "small clip should not need tiling: {plan:?}"
    );
}

#[test]
fn vae22_tiling_bounds_moderate_res_peak() {
    // The regression: 1024×576×97 on a 64 GiB machine. The px-threshold `auto` chose 512 px tiles
    // → ~60 GB. The budgeted plan must tile, keep the peak under the safe budget, and crucially
    // below the 60 GB blow-up that OOMs a 64 GiB Mac.
    let safe = 64.0 * 0.85; // 54.4 GiB
    let cfg = plan_vae22_tiling(576, 1024, 97, safe, false)
        .unwrap()
        .expect("moderate res must tile");
    let peak = chosen_peak_gib(&cfg, 576, 1024, 97, false);
    assert!(
        peak <= safe,
        "chosen peak {peak:.1} GiB over safe {safe:.1}"
    );
    assert!(
        peak < 60.0,
        "chosen peak {peak:.1} GiB not below the 60 GB blow-up"
    );
}

#[test]
fn vae22_bf16_tiling_stays_under_budget_and_below_f32_peak() {
    // sc-5039: the bf16 plan must keep its chosen tile under the safe budget (the 3100→3400
    // coefficient fix — 3100 let a tile measure 55.1 GB, over the 54.4 GB line), and the bf16
    // peak of a *given* tile is strictly below the f32 peak of the same tile (the headroom that
    // is bf16's only real win — no wall-clock benefit). No claim that it fits a *bigger* tile:
    // at this resolution the candidate grid lands bf16 on the same 384/full-97 tile as f32.
    let safe = 64.0 * 0.85; // 54.4 GiB
    let bf16 = plan_vae22_tiling(576, 1024, 97, safe, true)
        .unwrap()
        .expect("bf16 still needs tiling at 64 GiB");
    let bf16_peak = chosen_peak_gib(&bf16, 576, 1024, 97, true);
    assert!(
        bf16_peak <= safe,
        "bf16 chosen peak {bf16_peak:.1} GiB over safe {safe:.1}"
    );
    // Same tile, bf16 vs f32: bf16 must be the lighter estimate.
    let f32_same = chosen_peak_gib(&bf16, 576, 1024, 97, false);
    assert!(
        bf16_peak < f32_same,
        "bf16 peak {bf16_peak:.1} not below f32 {f32_same:.1} for the same tile"
    );
}

#[test]
fn vae22_tiling_bounds_peak_across_output_sizes() {
    // The px-threshold `auto`'s real defect was *non-monotonic* peak: a moderate 1024×576×97
    // decode spiked to 60 GB while the *larger* 1280×704×145 sat at 12.6 GB — so the dangerous
    // peak hid at a routine resolution. The budgeted plan must hold every size under the safe
    // budget (and below that 60 GB spike), regardless of how output size grows.
    let safe = 64.0 * 0.85; // 54.4 GiB
    for (h, w, f) in [
        (576i64, 1024i64, 49i64),
        (576, 1024, 97),
        (576, 1024, 145),
        (704, 1280, 145),
        (1088, 1920, 97),
    ] {
        let peak = match plan_vae22_tiling(h as i32, w as i32, f as i32, safe, false).unwrap() {
            Some(cfg) => chosen_peak_gib(&cfg, h, w, f, false),
            None => estimated_vae22_decode_peak_gib(f, h, w, f, h, w, false), // single-pass fit
        };
        assert!(
            peak <= safe && peak < 60.0,
            "{w}×{h}×{f}: peak {peak:.1} GiB not bounded under safe {safe:.1} / 60 GB spike"
        );
    }
}

#[test]
fn vae22_tiling_errors_when_unfittable() {
    // A huge video under a tiny budget: even the smallest tile (and the unavoidable output
    // accumulators) cannot fit → a catchable error, not an OOM/abort.
    let err = plan_vae22_tiling(1088, 1920, 241, 8.0, false);
    assert!(err.is_err(), "over-budget decode must error, got {err:?}");
}

// --- sc-6894 F-009: z16 Wan 2.1 VAE decode budgeting ----------------------------------------

/// Re-derive a z16 plan's peak the way the selector sizes its largest tile.
fn z16_chosen_peak(cfg: &TilingConfig, h: i64, w: i64, f: i64) -> f64 {
    let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
    let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
    let tile_f = cfg
        .temporal
        .map(|t| (t.tile_frames as i64).min(f))
        .unwrap_or(f);
    estimated_z16_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w)
}

#[test]
fn z16_decode_peak_matches_sweep_anchors() {
    // Real-weight anchors from `vae16_decode_sweep.rs` (128 GB M-series, f32). The model must be
    // CONSERVATIVE (never below the measured peak — an under-shoot is an OOM) and within ~10 %.
    // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_gib)
    let anchors = [
        (16, 512, 512, 16, 512, 512, 25.39),   // single-pass
        (16, 768, 768, 16, 768, 768, 56.35),   // single-pass
        (32, 512, 512, 32, 512, 512, 50.12),   // single-pass (temporal scaling == spatial)
        (16, 768, 768, 16, 384, 384, 14.46),   // tiled @384 px
        (16, 1024, 1024, 16, 512, 512, 25.66), // tiled @512 px
    ];
    for (of, oh, ow, tf, th, tw, measured) in anchors {
        let est = estimated_z16_decode_peak_gib(of, oh, ow, tf, th, tw);
        assert!(
            est >= measured,
            "z16 model {est:.2} GiB UNDER-shoots measured {measured} (OOM risk) for tile \
                 [{tf},{th},{tw}] of [{of},{oh},{ow}]"
        );
        assert!(
            est <= measured * 1.10,
            "z16 model {est:.2} GiB over-conservative vs measured {measured} (>10 %)"
        );
    }
}

#[test]
fn z16_tiling_single_pass_when_small() {
    // A short, low-res z16 clip fits a single-pass decode → no tiling.
    let plan = plan_z16_tiling(256, 256, 16, 60.0).unwrap();
    assert!(plan.is_none(), "small z16 clip should not tile: {plan:?}");
}

#[test]
fn z16_tiling_bounds_moderate_res_peak() {
    // 1280×720×80 on a 64 GiB machine: single-pass z16 would peak ~450 GB. The budgeted plan must
    // tile and keep the recomputed peak under the safe budget (the bounded/catchable guarantee).
    let safe = 64.0 * 0.85; // 54.4 GiB
    let cfg = plan_z16_tiling(720, 1280, 80, safe)
        .unwrap()
        .expect("moderate-res z16 must tile");
    let peak = z16_chosen_peak(&cfg, 720, 1280, 80);
    assert!(
        peak <= safe,
        "z16 chosen peak {peak:.1} GiB over safe {safe:.1}"
    );
}

/// sc-15445 decision gate: Wan product selectors restore the historical candidate overlap while
/// Krea/SCAIL-2's z16 selector retains half-tile overlap. The budgets deliberately select the
/// 32-frame row measured at 640×384×81, so changing either policy is independently red.
#[test]
fn wan_product_and_quality_selectors_keep_the_measured_overlap_split() {
    let wan_z16 = plan_z16_tiling(384, 640, 84, 10.0).unwrap().unwrap();
    let quality_z16 =
        plan_z16_tiling_with_overlap(384, 640, 84, 10.0, TemporalOverlapPolicy::HalfTile)
            .unwrap()
            .unwrap();
    assert_eq!(
        wan_z16.temporal.map(|t| (t.tile_frames, t.overlap_frames)),
        Some((32, 8)),
    );
    assert_eq!(
        quality_z16
            .temporal
            .map(|t| (t.tile_frames, t.overlap_frames)),
        Some((32, 16)),
    );
    assert_eq!(
        wan_z16.spatial.map(|s| (s.tile_px, s.overlap_px)),
        quality_z16.spatial.map(|s| (s.tile_px, s.overlap_px)),
        "the policy split must change only temporal overlap",
    );

    let wan_z48 = plan_vae22_tiling(384, 640, 81, 4.5, true).unwrap().unwrap();
    assert_eq!(
        wan_z48.temporal.map(|t| (t.tile_frames, t.overlap_frames)),
        Some((32, 8)),
    );

    // Parse the committed raw RESULT records rather than restating their numbers in this test.
    // The checksum makes any evidence-row edit explicit; the assertions below independently
    // gate the four-cell scope, reproduce every recorded call count from live tiling geometry,
    // and enforce the cost/quality thresholds behind Candidate.
    let expected_cells = [
        "Z16:640x384:81",
        "Z16:832x480:121",
        "Z48:640x384:81",
        "Z48:832x480:121",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let mut actual_cells = BTreeSet::new();
    for (_, log) in WAN_OVERLAP_EVIDENCE {
        let fields = evidence_fields(log);
        let family = fields["family"];
        let cell = format!("{family}:{}:{}", fields["bucket"], fields["product_frames"]);
        assert!(actual_cells.insert(cell.clone()), "duplicate evidence cell");

        let vae = match family {
            "Z16" => WanVae::VAE_TILING,
            "Z48" => Wan22Vae::VAE_TILING,
            other => panic!("unexpected evidence family {other}"),
        };
        let (width, height) = fields["bucket"]
            .split_once('x')
            .map(|(w, h)| {
                (
                    w.parse::<i32>().expect("numeric evidence width"),
                    h.parse::<i32>().expect("numeric evidence height"),
                )
            })
            .expect("evidence bucket must be WIDTHxHEIGHT");
        let decoded_frames = evidence_i32(&fields, "decoded_frames");
        let latent_frames = if vae.causal_temporal {
            (decoded_frames - 1) / vae.temporal_scale + 1
        } else {
            decoded_frames / vae.temporal_scale
        };
        let tile_frames = evidence_i32(&fields, "tile");
        let candidate_overlap = evidence_i32(&fields, "old_overlap");
        let half_overlap = evidence_i32(&fields, "current_overlap");
        let iterations = |overlap_frames| {
            let plan = TilingConfig {
                spatial: Some(mlx_gen::tiling::SpatialTiling {
                    tile_px: 192,
                    overlap_px: 64,
                }),
                temporal: Some(mlx_gen::tiling::TemporalTiling {
                    tile_frames,
                    overlap_frames,
                }),
            }
            .plan(
                vae,
                latent_frames,
                height / vae.spatial_scale,
                width / vae.spatial_scale,
            );
            (plan.t.len(), plan.t.len() * plan.h.len() * plan.w.len())
        };
        let candidate_iters = iterations(candidate_overlap);
        let half_iters = iterations(half_overlap);
        assert_eq!(
            (candidate_iters.0, half_iters.0),
            evidence_pair(&fields, "temporal_iterations"),
            "{cell}: temporal calls must reproduce from the live planner",
        );
        assert_eq!(
            (candidate_iters.1, half_iters.1),
            evidence_pair(&fields, "all_iterations"),
            "{cell}: all tile calls must reproduce from the live planner",
        );

        let old_s = evidence_f64(&fields, "old_seconds_median");
        let half_s = evidence_f64(&fields, "current_seconds_median");
        let recorded_ratio = evidence_f64(&fields, "wall_ratio");
        assert!(
            (half_s / old_s - recorded_ratio).abs() <= 0.0001,
            "{cell}: recorded wall ratio must agree with medians",
        );
        assert!(
            half_s / old_s >= 1.20,
            "{cell}: half-tile wall cost stopped being material",
        );
        let (_, old_max) = evidence_range(&fields, "old_seconds_range");
        let (half_min, _) = evidence_range(&fields, "current_seconds_range");
        assert!(half_min > old_max, "{cell}: repeated timing ranges overlap",);

        let old_err = evidence_f64(&fields, "old_mae255");
        let half_err = evidence_f64(&fields, "current_mae255");
        assert!(
            half_err < old_err && old_err - half_err <= 0.61,
            "{cell}: recorded quality trade no longer supports the measured decision",
        );
        let old_peak = evidence_f64(&fields, "old_peak_gib");
        let half_peak = evidence_f64(&fields, "current_peak_gib");
        assert!(
            (half_peak / old_peak - 1.0f64).abs() <= 0.05,
            "{cell}: overlap unexpectedly changed peak memory",
        );
    }
    assert_eq!(
        actual_cells, expected_cells,
        "evidence must contain exactly both shipping buckets for both Wan VAE families",
    );
    assert_eq!(
        evidence_result_checksum(),
        0x5717_3a67_4348_cabb,
        "sc-15445 RESULT records changed; audit the new evidence and update this checksum",
    );

    // The final timed row's post-timing quality phase overlapped another Metal task. Its
    // uncontended quality-only rerun must reproduce every metric used by the decision.
    let timed = evidence_fields(WAN_OVERLAP_EVIDENCE[3].1);
    let clean = evidence_fields(WAN_OVERLAP_Z48_832_CLEAN_QUALITY);
    for key in [
        "old_mae255",
        "current_mae255",
        "reference_clip_mean",
        "reference_clip_worst",
        "old_clip_mean",
        "old_clip_worst",
        "current_clip_mean",
        "current_clip_worst",
    ] {
        assert_eq!(
            timed[key], clean[key],
            "clean z48 832 quality rerun changed {key}",
        );
    }
}

#[test]
fn z16_tiling_errors_when_unfittable() {
    // 4K × 240 frames under an 8 GiB budget: the output accumulators alone blow it → a catchable
    // error before the decode, not a SIGKILL.
    let err = plan_z16_tiling(2160, 3840, 240, 8.0);
    assert!(
        err.is_err(),
        "over-budget z16 decode must error, got {err:?}"
    );
}

#[test]
fn vae22_tiling_budgeted_reads_free_memory() {
    use mlx_rs::memory::set_memory_limit;
    // Exercise the public wrapper end-to-end. Pin a 64 GiB limit; with a low baseline residency the
    // free-aware budget (≈ 64 × 0.85) still forces the 1024×576×97 decode to tile. Restore after.
    std::env::remove_var(WAN_VAE_BUDGET_ENV); // ensure the free-probe path, not an ambient override
    let prev = set_memory_limit(64 << 30);
    let plan = auto_tiling_budgeted(576, 1024, 97, false);
    set_memory_limit(prev);
    let cfg = plan.unwrap().expect("moderate res tiles at 64 GiB free");
    assert!(cfg.spatial.is_some() || cfg.temporal.is_some());
}

// --- sc-12737: free-aware VAE-decode budget (contract-aligned with candle sc-12734) ----------

#[test]
fn free_aware_budget_is_free_times_frac_clamped() {
    // The pure arithmetic: `(limit − resident) × frac`. With 24 GiB resident under a 96 GiB limit,
    // free = 72; at 0.85 → 61.2 GiB, strictly below the total-based 96×0.85 the old code used.
    let frac = WAN_VAE_BUDGET_SAFE_FRAC;
    let (limit, resident) = (96.0, 24.0);
    let free = limit - resident;
    assert!((free_aware_budget_gib(free, frac) - free * frac).abs() < 1e-9);
    assert!(
        free_aware_budget_gib(free, frac) < limit * frac,
        "resident memory must shrink the budget vs the total-based 0.85×limit"
    );
    // Never negative even if resident somehow exceeds the limit (defensive clamp).
    assert_eq!(free_aware_budget_gib(-5.0, frac), 0.0);
}

#[test]
fn free_aware_resolver_uses_probe_and_env_override_wins() {
    // With N GiB resident the probe reports (limit − resident); the resolver returns free × frac,
    // strictly below the total-based budget for the same limit.
    let frac = WAN_VAE_BUDGET_SAFE_FRAC;
    let (limit, resident) = (96.0, 30.0);
    let stub_free = move || Some(limit - resident);
    let free_budget = resolve_free_aware_budget("WAN_VAE_TEST_BUDGET_UNSET", frac, 16.0, stub_free);
    assert!((free_budget - (limit - resident) * frac).abs() < 1e-9);
    assert!(
        free_budget < limit * frac,
        "free-aware budget must be below the total-based budget"
    );

    // The env override wins over the live probe (the deterministic worker/test injection point).
    // Capture then remove before asserting so a failure can't leak the var into other tests.
    std::env::set_var("WAN_VAE_TEST_BUDGET", "42.5");
    let overridden = resolve_free_aware_budget("WAN_VAE_TEST_BUDGET", frac, 16.0, || Some(1.0));
    std::env::remove_var("WAN_VAE_TEST_BUDGET");
    assert_eq!(overridden, 42.5);

    // No env + no probe (limit disabled → None) → the conservative default.
    assert_eq!(
        resolve_free_aware_budget("WAN_VAE_TEST_BUDGET_UNSET", frac, 16.0, || None),
        16.0
    );
}

#[test]
fn wan_vae_budget_honors_env_override() {
    // The shared resolver both tilers use must let WAN_VAE_BUDGET_GIB pin the budget (the same knob
    // candle honors). Capture then remove before asserting so a failure can't leak the var.
    std::env::set_var(WAN_VAE_BUDGET_ENV, "42.5");
    let got = wan_vae_safe_budget_gib();
    std::env::remove_var(WAN_VAE_BUDGET_ENV);
    assert_eq!(got, 42.5);
}

#[test]
fn free_aware_budget_tracks_resident_memory_end_to_end() {
    // The discriminating test: prove the LIVE budget shrinks as resident memory grows — i.e. it
    // budgets against FREE (limit − active), not TOTAL. A revert to `get_memory_limit() × 0.85`
    // would leave the budget unchanged when active rises AND equal to 48×0.85 exactly — both
    // assertions below would then fail.
    use mlx_rs::memory::{get_active_memory, set_memory_limit};
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    std::env::remove_var(WAN_VAE_BUDGET_ENV); // force the free-probe path, not an ambient override
    let prev = set_memory_limit(48 << 30); // pin a deterministic 48 GiB ceiling
    let total_based = 48.0 * WAN_VAE_BUDGET_SAFE_FRAC; // what the old get_memory_limit()×0.85 gave

    let a_before = get_active_memory();
    let b_before = wan_vae_safe_budget_gib();

    // Raise resident memory by a live, materialized 1 GiB f32 array (256Mi elems, < i32::MAX).
    let resident = Array::zeros::<f32>(&[268_435_456]).unwrap();
    mlx_rs::transforms::eval([&resident]).unwrap();

    let a_after = get_active_memory();
    let b_after = wan_vae_safe_budget_gib();

    set_memory_limit(prev);

    let delta_active_gib = a_after.saturating_sub(a_before) as f64 / GIB;
    assert!(
        delta_active_gib > 0.5,
        "test precondition: the live array must raise active memory (Δ {delta_active_gib:.2} GiB)"
    );
    // Free-aware ⇒ the budget drops by ≈ Δactive × frac. A total-based budget would not move.
    let expected_drop = delta_active_gib * WAN_VAE_BUDGET_SAFE_FRAC;
    let actual_drop = b_before - b_after;
    assert!(
            (actual_drop - expected_drop).abs() < 0.25,
            "budget must track FREE memory: dropped {actual_drop:.3} GiB, expected ~{expected_drop:.3} \
             (Δactive {delta_active_gib:.3} × {WAN_VAE_BUDGET_SAFE_FRAC}). Total-based would not move."
        );
    // And it must be strictly below the total-based 48×0.85 while memory is resident (active > 0).
    assert!(
        b_after < total_based,
        "free-aware budget {b_after:.2} must be below the total-based {total_based:.2} while \
             memory is resident"
    );
    drop(resident);
}

#[test]
fn align_dim_rounds_down_to_tile() {
    // patch 2 × vae_stride 8 = 16-px grid.
    assert_eq!(align_dim(1280, 2, 8), 1280);
    assert_eq!(align_dim(1281, 2, 8), 1280);
    assert_eq!(align_dim(1295, 2, 8), 1280);
    assert_eq!(align_dim(1296, 2, 8), 1296);
}

#[test]
fn latent_shape_and_seq_len_match_reference_formulas() {
    // 49 frames, 512×512, z16, stride (4,8,8), patch (1,2,2).
    let ls = latent_shape(49, 512, 512, 16, (4, 8, 8)).unwrap();
    assert_eq!(ls, [16, 13, 64, 64]); // (49-1)/4+1=13, 512/8=64
    let sl = seq_len(ls, (1, 2, 2));
    // ceil(64*64/(2*2) * 13) = 1024 * 13 = 13312
    assert_eq!(sl, 13312);
}

#[test]
fn latent_shape_rejects_zero_frames() {
    // frames == 0 must be a clean error, not a usize underflow → huge t_lat (F-007).
    assert!(latent_shape(0, 512, 512, 16, (4, 8, 8)).is_err());
    assert!(latent_shape(1, 512, 512, 16, (4, 8, 8)).is_ok());
}

#[test]
fn cfg_combine_is_uncond_plus_gs_delta() {
    let cond = Array::from_slice(&[2.0f32, 4.0], &[2]);
    let uncond = Array::from_slice(&[1.0f32, 1.0], &[2]);
    let got = cfg_combine(&cond, &uncond, 3.0).unwrap();
    // 1 + 3*(2-1) = 4 ; 1 + 3*(4-1) = 10
    assert_eq!(got.as_slice::<f32>(), &[4.0, 10.0]);
}

#[test]
fn py_round_is_half_to_even() {
    assert_eq!(py_round(19.2), 19);
    assert_eq!(py_round(16.0), 16);
    assert_eq!(py_round(0.5), 0); // half → even (down)
    assert_eq!(py_round(1.5), 2); // half → even (up)
    assert_eq!(py_round(2.5), 2); // half → even (down)
    assert_eq!(py_round(2.500001), 3); // just over half → up
}

/// sc-12308 — these two tests replace `best_output_size_caps_area_and_aligns`, which pinned the
/// silent refit `best_output_size(1280, 720, 16, 16, 704*1280) == (1264, 704)`. Both the
/// function and that behaviour are gone: an over-cap request is now rejected, and `1280×720` is
/// not over the 14B family's real cap at all.
#[test]
fn over_area_is_rejected_not_refit() {
    let req = |w, h| GenerationRequest {
        prompt: "x".into(),
        width: w,
        height: h,
        ..Default::default()
    };

    // The canonical 720p is AT the 14B cap (`>` check) and must pass untouched — this is the
    // exact geometry the old refit turned into 1264×704, off every advertised bucket.
    assert!(reject_over_area("t2v", &req(1280, 720), 16, 16, crate::config::MAX_AREA_14B).is_ok());
    assert!(reject_over_area("t2v", &req(720, 1280), 16, 16, crate::config::MAX_AREA_14B).is_ok());

    // Genuinely over-envelope (sc-9028's 1280×1280) is rejected, with a message naming the cap.
    let err = reject_over_area("t2v", &req(1280, 1280), 16, 16, crate::config::MAX_AREA_14B)
        .expect_err("over-area must be rejected");
    assert!(err.to_string().contains("max area"), "actionable: {err}");

    // The 5B's own budget is smaller, and ITS 720p is 704-tall: 1280×704 is exactly at cap.
    assert!(reject_over_area("5b", &req(1280, 704), 32, 32, crate::config::MAX_AREA_5B).is_ok());
    // A 1280×720 ask does NOT trip the 5B's area cap — its 32-px grid aligns 720 down to 704
    // first, landing at 901,120. That silent 720→704 is the STRIDE floor, a separate constraint
    // the manifest handles by advertising 1280x704 for this model; it is not an area rejection.
    assert!(reject_over_area("5b", &req(1280, 720), 32, 32, crate::config::MAX_AREA_5B).is_ok());
    // What the 5B's cap does catch is genuinely over-envelope geometry.
    assert!(reject_over_area("5b", &req(1280, 1280), 32, 32, crate::config::MAX_AREA_5B).is_err());
    // …and the 5B must not silently inherit the 14B family's larger budget.
    const { assert!(crate::config::MAX_AREA_5B < crate::config::MAX_AREA_14B) };

    // `max_area == 0` means uncapped.
    assert!(reject_over_area("x", &req(4096, 4096), 16, 16, 0).is_ok());
}

#[test]
fn over_area_is_judged_on_the_aligned_geometry() {
    let req = |w, h| GenerationRequest {
        prompt: "x".into(),
        width: w,
        height: h,
        ..Default::default()
    };
    // An off-grid request is aligned DOWN before rendering, so it must be judged on what it
    // becomes, not what was typed: 1288×724 → 1280×720 = 921,600, exactly at the cap. Judging
    // the raw 932,512 would reject a request that renders perfectly legally.
    assert_eq!((align_dim(1288, 1, 16), align_dim(724, 1, 16)), (1280, 720));
    assert!(reject_over_area("t2v", &req(1288, 724), 16, 16, crate::config::MAX_AREA_14B).is_ok());
}

#[test]
fn build_i2v_mask_is_one_at_first_latent_frame() {
    // [4, T_lat=2, 1, 1]: channel-major, temporal index 0 → 1.0, index 1 → 0.0.
    let m = build_i2v_mask(2, 1, 1);
    assert_eq!(m.shape(), &[4, 2, 1, 1]);
    assert_eq!(m.as_slice::<f32>(), &[1., 0., 1., 0., 1., 0., 1., 0.]);
}

#[test]
fn build_ti2v_mask_freezes_first_frame() {
    // z=2, T_lat=2, h=w=2, patch (1,2,2) → grid (2,1,1) → L=2 tokens.
    let (mask, tokens) = build_ti2v_mask(&[(0, 1.0)], 2, 2, 2, 2, (1, 2, 2));
    assert_eq!(mask.shape(), &[2, 2, 2, 2]);
    // Per channel (8 vals): temporal 0 → 0.0 (4 spatial), temporal 1 → 1.0 (4 spatial).
    assert_eq!(
        mask.as_slice::<f32>(),
        &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]
    );
    // Token mask: first (t'=0) token frozen (0), second (t'=1) active (1).
    assert_eq!(tokens.shape(), &[1, 2]);
    assert_eq!(tokens.as_slice::<f32>(), &[0., 1.]);
}

#[test]
fn build_ti2v_mask_pins_first_and_last() {
    // first_last_frame: z=1, T_lat=3, h=w=2, patch (1,2,2) → grid (3,1,1) → 3 tokens.
    let (mask, tokens) = build_ti2v_mask(&[(0, 1.0), (2, 1.0)], 1, 3, 2, 2, (1, 2, 2));
    assert_eq!(mask.shape(), &[1, 3, 2, 2]);
    // temporal 0 → 0 (4), temporal 1 → 1 (4), temporal 2 → 0 (4).
    assert_eq!(
        mask.as_slice::<f32>(),
        &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.]
    );
    // token mask: frame0 + frame2 tokens 0, frame1 token 1.
    assert_eq!(tokens.as_slice::<f32>(), &[0., 1., 0.]);
}

/// sc-19571: the per-pin strength must land in BOTH masks as `1 − strength`, with each pin
/// carrying its OWN weight — the defect this replaced built a hard 0/1 mask from the indices
/// alone, so every value below is one a `strength`-blind builder cannot produce. Deliberately
/// **non-default**: 1.0 (the old hard pin) appears nowhere as a strength here.
#[test]
fn build_ti2v_mask_threads_per_pin_strength() {
    // z=1, T_lat=3, h=w=2, patch (1,2,2) → grid (3,1,1) → 3 tokens. First pinned at 0.25,
    // last at 0.75 — different from each other, so a builder that reused one strength for
    // every pin is also caught.
    let (mask, tokens) = build_ti2v_mask(&[(0, 0.25), (2, 0.75)], 1, 3, 2, 2, (1, 2, 2));
    assert_eq!(
        mask.as_slice::<f32>(),
        &[0.75, 0.75, 0.75, 0.75, 1., 1., 1., 1., 0.25, 0.25, 0.25, 0.25],
        "latent mask must be 1 − strength at each pin, per pin"
    );
    assert_eq!(
        tokens.as_slice::<f32>(),
        &[0.75, 1., 0.25],
        "token mask must carry the same per-pin 1 − strength"
    );
    // strength 0.0 is a no-op pin: the frame denoises exactly as if it were never pinned.
    let (free, free_tokens) = build_ti2v_mask(&[(0, 0.0)], 1, 2, 1, 1, (1, 1, 1));
    assert_eq!(free.as_slice::<f32>(), &[1., 1.]);
    assert_eq!(free_tokens.as_slice::<f32>(), &[1., 1.]);

    // **Cross-lane pin.** These are the literal inputs and expected values in candle's
    // `candle_gen_wan::pipeline::ti2v_tests::mask_and_keyframe_scatter_pin_first_and_last`, so
    // the two backends are gated against the same numbers rather than each against its own idea
    // of the construction — the divergence that let one lane honor the knob while the other
    // silently dropped it is exactly what this asserts away.
    let (mask, tokens) = build_ti2v_mask(&[(0, 1.0), (2, 0.25)], 1, 3, 2, 2, (1, 2, 2));
    assert_eq!(
        mask.as_slice::<f32>(),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75]
    );
    assert_eq!(tokens.as_slice::<f32>(), &[0.0, 1.0, 0.75]);
}

#[test]
fn build_ti2v_keyframe_z_scatters_frames() {
    // z=1, h=w=1, T_lat=3; place A=[7] @0 and B=[9] @2; frame1 = 0.
    let a = Array::from_slice(&[7.0f32], &[1, 1, 1, 1]);
    let b = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
    let z = build_ti2v_keyframe_z(&[(a, 0), (b, 2)], 1, 3, 1, 1).unwrap();
    assert_eq!(z.shape(), &[1, 3, 1, 1]);
    assert_eq!(z.as_slice::<f32>(), &[7.0, 0.0, 9.0]);
}

#[test]
fn ti2v_blend_init_freezes_first_frame() {
    // z=1,T=2,h=w=1: mask 0 at t=0, 1 at t=1. z_img=[9] (frame0), noise=[5,7].
    let z_img = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
    let mask = Array::from_slice(&[0.0f32, 1.0], &[1, 2, 1, 1]);
    let noise = Array::from_slice(&[5.0f32, 7.0], &[1, 2, 1, 1]);
    let out = ti2v_blend_init(&z_img, &mask, &noise).unwrap();
    // frame0 = z_img (9), frame1 = noise (7).
    assert_eq!(out.as_slice::<f32>(), &[9.0, 7.0]);
}

/// sc-20686 E8: the encode working-set model covers every point `encode_footprint_harness`
/// measured (production-width synthetic weights, f32, post-release live == live + cache) and
/// stays within 30 % above it.
#[test]
fn encode_peak_model_covers_the_measured_encodes() {
    // (vae, frames, height, width, measured live bytes)
    let measured = [
        (WanVae::VAE_TILING, 1, 64, 128, 51_478_544_u64),
        (WanVae::VAE_TILING, 5, 64, 128, 176_734_228),
        (WanVae::VAE_TILING, 9, 64, 128, 199_360_532),
        (WanVae::VAE_TILING, 1, 128, 128, 102_334_480),
        (WanVae::VAE_TILING, 5, 128, 128, 352_256_020),
        (WanVae::VAE_TILING, 13, 128, 128, 397_541_396),
        (Wan22Vae::VAE_TILING, 1, 64, 128, 23_289_876),
        (Wan22Vae::VAE_TILING, 5, 64, 128, 78_393_368),
        (Wan22Vae::VAE_TILING, 9, 64, 128, 78_786_584),
        (Wan22Vae::VAE_TILING, 1, 128, 128, 46_333_972),
        (Wan22Vae::VAE_TILING, 5, 128, 128, 156_287_000),
        (Wan22Vae::VAE_TILING, 13, 128, 128, 157_859_864),
    ];
    for (vae, frames, height, width, live) in measured {
        let modelled = video_encode_peak_bytes_for_vae(vae, width, height, frames).unwrap();
        assert!(
            modelled >= live && (modelled as f64) <= live as f64 * 1.3,
            "{vae:?} {frames}x{height}x{width}: modelled {modelled} vs measured {live}"
        );
    }
    assert_eq!(
        video_encode_peak_bytes_for_vae(WanVae::VAE_TILING, 64, 64, 0),
        None
    );
}
