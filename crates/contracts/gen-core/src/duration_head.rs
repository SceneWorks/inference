//! Tensor-free machinery for the LTX-2.5 `DurationHead` (sc-18774): the pinned architecture
//! hyperparameters, and the caption-driven auto-duration resolution algorithm (predict →
//! exponentiate [done by the head itself] → clamp → snap to the VAE's causal `n % 8 == 1` temporal
//! grid). Ported from the reference `Lightricks/LTX-2` @ `d151147788a9284cca791edc6ce898007e727fe6`
//! (v1.2.0):
//!
//! * `packages/ltx-core/src/ltx_core/duration_head/duration_head.py` — `DurationHead`/
//!   `AttentionPooler.__init__` defaults (the checkpoint's own `config.duration_head` metadata
//!   section ships **empty** — confirmed sc-18756, `docs/reference/sc-18756-headers/model_patches/
//!   ltx-2.5-duration-head-bf16.safetensors.json` — so these are pinned explicitly here rather than
//!   read from the file).
//! * `packages/ltx-pipelines/src/ltx_pipelines/utils/types.py::AutoDuration` — the `[min, max]`
//!   seconds range + its 1s/20s defaults.
//! * `packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py::{snap_frames_to_grid,
//!   seconds_to_clamped_num_frames}` and `utils/blocks.py::DurationPredictor.__call__` — the
//!   clamp-then-snap resolution.
//!
//! Shared (like [`crate::ltx_checkpoint`]) so mlx-gen-ltx and candle-gen-ltx resolve an identical
//! frame count from an identical prediction: this module owns none of the tensor math (backend
//! specific — linear projections + a small cross-attention pooler), only the numbers, hyperparameters
//! and the opt-in resolution seam both backends must agree on.

use crate::{Error, Result};

/// `DurationHead`/`AttentionPooler` hyperparameters (reference constructor defaults, pinned — see
/// module docs for why these are not read from the checkpoint).
pub mod hparams {
    /// `AttentionPooler`/`DurationHead` shared hidden dim (`pooler_hidden_dim`).
    pub const POOLER_HIDDEN_DIM: usize = 256;
    /// Number of learnable pooling queries (`AttentionPooler.num_queries`).
    pub const NUM_QUERIES: usize = 1;
    /// Attention heads in the pooler's cross-attention (`num_pooler_heads` / `AttentionPooler.
    /// num_heads`).
    pub const NUM_POOLER_HEADS: usize = 4;
    /// Hidden width of the output MLP (`DurationHead.mlp_hidden` out_features).
    pub const MLP_HIDDEN: usize = 256;
    /// Video connector output dim this head projects from — matches the checkpoint's OWN
    /// `config.transformer.cross_attention_dim` (confirmed 4096, sc-18756 §2.5), not a coincidence:
    /// the duration-head file re-declares the transformer's cross-attention dims precisely because
    /// this head consumes that connector's output.
    pub const VIDEO_CROSS_ATTENTION_DIM: usize = 4096;
    /// Audio connector output dim (matches `config.transformer.audio_cross_attention_dim`, confirmed
    /// 2048, sc-18756 §2.5).
    pub const AUDIO_CROSS_ATTENTION_DIM: usize = 2048;
}

/// The VAE's causal temporal grid stride (upstream `SpatioTemporalScaleFactors.time`; unchanged
/// 2.3→2.5 per sc-18756 §2.1/epic 18755 §1.2): a valid engine frame count satisfies
/// `(frames - 1) % TEMPORAL_GRID == 0`.
pub const TEMPORAL_GRID: u32 = 8;

/// `[min_seconds, max_seconds]` an auto-duration request clamps its prediction to (upstream
/// `ltx_pipelines.utils.types.AutoDuration`). [`Default`] mirrors upstream's `AutoDuration()`:
/// 1s–20s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDurationRange {
    pub min_seconds: f32,
    pub max_seconds: f32,
}

impl Default for AutoDurationRange {
    fn default() -> Self {
        Self {
            min_seconds: 1.0,
            max_seconds: 20.0,
        }
    }
}

impl AutoDurationRange {
    /// Validate a caller-supplied range. Upstream's `AutoDuration` is a bare dataclass with no
    /// validation; we add it here because a request-supplied range is untrusted input reaching a
    /// snap/clamp routine, and a non-finite or inverted range should fail loudly at the request
    /// boundary rather than produce a nonsensical frame count.
    pub fn new(min_seconds: f32, max_seconds: f32) -> Result<Self> {
        if !(min_seconds.is_finite() && max_seconds.is_finite()) {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration range must be finite, got [{min_seconds}, {max_seconds}]"
            )));
        }
        if min_seconds <= 0.0 || max_seconds <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration range must be positive, got [{min_seconds}, {max_seconds}]"
            )));
        }
        if min_seconds > max_seconds {
            return Err(Error::Msg(format!(
                "ltx duration_head: auto-duration min_seconds {min_seconds} > max_seconds {max_seconds}"
            )));
        }
        Ok(Self {
            min_seconds,
            max_seconds,
        })
    }
}

/// Round `frames` DOWN to the nearest `k * time_scale + 1` (upstream `snap_frames_to_grid`).
pub fn snap_frames_to_grid(frames: u32, time_scale: u32) -> Result<u32> {
    if frames < 1 {
        return Err(Error::Msg(format!(
            "ltx duration_head: frames must be >= 1, got {frames}"
        )));
    }
    if time_scale == 0 {
        return Err(Error::Msg(
            "ltx duration_head: time_scale must be > 0".into(),
        ));
    }
    Ok(((frames - 1) / time_scale) * time_scale + 1)
}

/// Convert a duration in seconds to a frame count on the VAE's temporal grid, clamped to
/// `[min_frames, max_frames]` (upstream `seconds_to_clamped_num_frames`). Snapping floors to the
/// grid, which can undershoot `min_frames`; when that happens the result snaps UP to the next grid
/// point instead (upstream's own fixup), capped at `max_frames`.
///
/// **One deliberate divergence from a literal port**, at the very end: when `[min_frames,
/// max_frames]` is so narrow that no grid point exists at or above `min_frames` within it (a
/// pathological but reachable case — an explicit, oddly-precise `min_seconds`/`max_seconds` pair —
/// upstream's own fixup formula can then return a frame count that is *inside* the window but is
/// NOT itself `8k + 1`, i.e. it can fail the very invariant the snap exists to guarantee). Per the
/// story: "The snap is where an unclamped or unsnapped prediction becomes a hard engine error
/// downstream" — so this port adds one final defensive re-snap to the largest valid grid point at
/// or below `max_frames`, which makes the `n % 8 == 1` contract unconditional rather than "usually
/// true". This never fires for any window with realistic width (`max_seconds − min_seconds` on the
/// order of the default 19s, or explicit ranges a caller would plausibly pass); it only changes
/// behavior in the adversarial narrow-window edge case, and even then only enough to restore engine
/// validity.
pub fn seconds_to_clamped_num_frames(
    seconds: f32,
    frame_rate: f32,
    min_frames: u32,
    max_frames: u32,
    time_scale: u32,
) -> Result<u32> {
    if !seconds.is_finite() {
        return Err(Error::Msg(format!(
            "ltx duration_head: seconds must be finite, got {seconds}"
        )));
    }
    if !(frame_rate.is_finite() && frame_rate > 0.0) {
        return Err(Error::Msg(format!(
            "ltx duration_head: frame_rate must be positive, got {frame_rate}"
        )));
    }
    if min_frames < 1 || min_frames > max_frames {
        return Err(Error::Msg(format!(
            "ltx duration_head: invalid frame bounds [{min_frames}, {max_frames}]"
        )));
    }
    let raw = (seconds * frame_rate).round();
    // Clamp in float space (matches upstream's clamp-before-snap order); saturating `as u32` casts
    // are safe in Rust regardless, but this keeps the clamp explicit and readable.
    let raw_frames = raw.clamp(min_frames as f32, max_frames as f32) as u32;
    let mut frames = snap_frames_to_grid(raw_frames, time_scale)?;
    if frames < min_frames {
        let ceil_div = (min_frames - 1).div_ceil(time_scale);
        frames = (ceil_div * time_scale + 1).min(max_frames);
    }
    // Final defensive guard — see doc comment above.
    if (frames - 1) % time_scale != 0 || frames > max_frames {
        frames = snap_frames_to_grid(max_frames, time_scale)?;
    }
    Ok(frames)
}

/// Resolve a `DurationHead` prediction (seconds) into a frame count (upstream
/// `DurationPredictor.__call__`): `min_frames`/`max_frames` are `round(seconds * frame_rate)` of the
/// range bounds (floored at 1 frame — upstream does not guard this, but a `min_seconds` so tiny or
/// an `frame_rate` so low that it rounds to 0 frames would otherwise make
/// [`seconds_to_clamped_num_frames`] reject it outright; flooring at 1 is strictly more permissive,
/// never less, than upstream for any input that upstream itself accepts), then
/// [`seconds_to_clamped_num_frames`].
pub fn resolve_auto_duration_frames(
    predicted_seconds: f32,
    frame_rate: f32,
    range: AutoDurationRange,
    time_scale: u32,
) -> Result<u32> {
    if !(frame_rate.is_finite() && frame_rate > 0.0) {
        return Err(Error::Msg(format!(
            "ltx duration_head: frame_rate must be positive, got {frame_rate}"
        )));
    }
    let min_frames = ((range.min_seconds * frame_rate).round()).max(1.0) as u32;
    let max_frames = ((range.max_seconds * frame_rate).round()).max(min_frames as f32) as u32;
    seconds_to_clamped_num_frames(predicted_seconds, frame_rate, min_frames, max_frames, time_scale)
}

/// The engine-boundary opt-in seam (sc-18774 acceptance: "surface it as an explicit opt-in ...; a
/// request with an explicit duration must never be silently overridden by the prediction").
///
/// Resolution order, mirroring upstream's `_resolve_num_frames` precedence ("an explicit
/// `--num-frames` wins ... otherwise `--auto-duration` if given, else unset"):
///
/// 1. `explicit_frames` is `Some` → returned as-is; `predict_seconds` is **never called** — an
///    explicit request always wins over prediction, proven by the
///    [`explicit_duration_wins_and_never_predicts`] test below (a spy `predict_seconds` that panics
///    if invoked).
/// 2. `explicit_frames` is `None` and `auto_duration` is `Some(range)` → `predict_seconds` is
///    called exactly once (the reachability contract: opting in must actually reach the head, not
///    merely declare support for it — see [`opt_in_reaches_the_predict_hook`]), and its seconds
///    prediction is resolved to a frame count via [`resolve_auto_duration_frames`].
/// 3. Neither is set → `Ok(None)`: auto-duration is **explicit opt-in only**; a caller that never
///    sets it must never have `predict_seconds` invoked (see
///    [`neither_flag_never_calls_predict`]) — a defaulted-off flag that fires anyway would defeat
///    the whole point of proving reachability separately from declaration.
///
/// `predict_seconds` is injected so callers (and this module's tests) can supply the real
/// `DurationHead::forward` or a spy independently of this resolution logic — the same seam pattern
/// as `mlx-gen-ltx`/`candle-gen-ltx`'s `image_crf::condition_image_for_checkpoint`.
pub fn resolve_request_num_frames(
    explicit_frames: Option<u32>,
    auto_duration: Option<AutoDurationRange>,
    frame_rate: f32,
    time_scale: u32,
    predict_seconds: &mut dyn FnMut() -> Result<f32>,
) -> Result<Option<u32>> {
    if let Some(frames) = explicit_frames {
        return Ok(Some(frames));
    }
    let Some(range) = auto_duration else {
        return Ok(None);
    };
    let seconds = predict_seconds()?;
    let frames = resolve_auto_duration_frames(seconds, frame_rate, range, time_scale)?;
    Ok(Some(frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------------
    // snap_frames_to_grid
    // ---------------------------------------------------------------------------------------

    #[test]
    fn snap_matches_reference_values() {
        // Pinned against the upstream formula `((frames - 1) // 8) * 8 + 1`.
        for (frames, want) in [(1, 1), (8, 1), (9, 9), (16, 9), (17, 17), (24, 17), (25, 25)] {
            assert_eq!(snap_frames_to_grid(frames, 8).unwrap(), want, "frames={frames}");
        }
    }

    #[test]
    fn snap_rejects_zero_frames_and_zero_time_scale() {
        assert!(snap_frames_to_grid(0, 8).is_err());
        assert!(snap_frames_to_grid(9, 0).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // seconds_to_clamped_num_frames / resolve_auto_duration_frames — pinned against a Python
    // re-implementation of upstream's exact algorithm (see the story's implementation notes).
    // ---------------------------------------------------------------------------------------

    #[test]
    fn resolve_matches_reference_values_fps24_default_range() {
        let range = AutoDurationRange::default(); // 1..20s, matches upstream AutoDuration()
        for (seconds, want) in [
            (0.01_f32, 25_u32),
            (1.0, 25),
            (5.0, 113),
            (12.3456, 289),
            (20.0, 473),
            (1000.0, 473),
        ] {
            let got = resolve_auto_duration_frames(seconds, 24.0, range, TEMPORAL_GRID).unwrap();
            assert_eq!(got, want, "seconds={seconds}");
        }
    }

    #[test]
    fn resolve_matches_reference_values_fps25_narrow_range() {
        let range = AutoDurationRange::new(2.0, 10.0).unwrap();
        for (seconds, want) in [
            (0.5_f32, 57_u32),
            (2.0, 57),
            (6.7, 161),
            (10.0, 249),
            (50.0, 249),
        ] {
            let got = resolve_auto_duration_frames(seconds, 25.0, range, TEMPORAL_GRID).unwrap();
            assert_eq!(got, want, "seconds={seconds}");
        }
    }

    /// Acceptance: "A test proves the clamp and the `n % 8 == 1` snap both hold across the range,
    /// including at the min and max bounds." A prediction far below `min_seconds` clamps UP to the
    /// min bound and snaps to a valid grid point; far above `max_seconds` clamps DOWN and snaps.
    #[test]
    fn clamp_and_snap_hold_at_min_and_max_bounds() {
        let range = AutoDurationRange::default();
        let below_min = resolve_auto_duration_frames(0.01, 24.0, range, TEMPORAL_GRID).unwrap();
        let above_max = resolve_auto_duration_frames(1_000_000.0, 24.0, range, TEMPORAL_GRID).unwrap();
        assert_eq!((below_min - 1) % TEMPORAL_GRID, 0);
        assert_eq!((above_max - 1) % TEMPORAL_GRID, 0);
        // Clamped toward (not past) the requested bounds: below_min sits at/near the snapped-up
        // min_frames (round(1.0*24)=24 -> next grid point 25); above_max sits at/near the
        // snapped-down max_frames (round(20.0*24)=480 -> floor to grid 473).
        assert_eq!(below_min, 25);
        assert_eq!(above_max, 473);
    }

    /// Property sweep: for a wide range of fps / range / prediction combinations, the result is
    /// ALWAYS on the `n % 8 == 1` grid — this is the hard invariant the story calls out ("The snap
    /// is where an unclamped or unsnapped prediction becomes a hard engine error downstream").
    #[test]
    fn n_mod_8_eq_1_holds_across_a_wide_sweep() {
        for fps_i in [8, 12, 16, 24, 25, 30, 60] {
            let fps = fps_i as f32;
            let mut seconds = 0.0_f32;
            while seconds <= 80.0 {
                let range = AutoDurationRange::new(0.5, 45.0).unwrap();
                let frames =
                    resolve_auto_duration_frames(seconds, fps, range, TEMPORAL_GRID).unwrap();
                assert_eq!(
                    (frames - 1) % TEMPORAL_GRID,
                    0,
                    "seconds={seconds} fps={fps} -> frames={frames}"
                );
                assert!(frames >= 1);
                seconds += 0.37; // odd step so it doesn't land on nice round numbers only
            }
        }
    }

    /// Even the pathological narrow-window case (documented in
    /// [`seconds_to_clamped_num_frames`]'s doc comment) still returns a valid grid point — the
    /// defensive final guard, not a literal port of upstream's fixup alone.
    #[test]
    fn narrow_window_still_returns_a_valid_grid_point() {
        let range = AutoDurationRange::new(2.2826559577605403, 2.6795361998095593).unwrap();
        let frames = resolve_auto_duration_frames(56.634_225, 8.0, range, TEMPORAL_GRID).unwrap();
        assert_eq!((frames - 1) % TEMPORAL_GRID, 0, "frames={frames}");
        assert!(frames >= 1);
    }

    #[test]
    fn seconds_to_clamped_num_frames_rejects_bad_inputs() {
        assert!(seconds_to_clamped_num_frames(f32::NAN, 24.0, 1, 100, 8).is_err());
        assert!(seconds_to_clamped_num_frames(5.0, 0.0, 1, 100, 8).is_err());
        assert!(seconds_to_clamped_num_frames(5.0, -1.0, 1, 100, 8).is_err());
        assert!(seconds_to_clamped_num_frames(5.0, 24.0, 0, 100, 8).is_err());
        assert!(seconds_to_clamped_num_frames(5.0, 24.0, 100, 1, 8).is_err());
    }

    #[test]
    fn auto_duration_range_rejects_bad_inputs() {
        assert!(AutoDurationRange::new(f32::NAN, 10.0).is_err());
        assert!(AutoDurationRange::new(1.0, f32::INFINITY).is_err());
        assert!(AutoDurationRange::new(0.0, 10.0).is_err());
        assert!(AutoDurationRange::new(-1.0, 10.0).is_err());
        assert!(AutoDurationRange::new(10.0, 1.0).is_err());
        assert!(AutoDurationRange::new(1.0, 10.0).is_ok());
    }

    #[test]
    fn resolve_auto_duration_frames_rejects_bad_frame_rate() {
        let range = AutoDurationRange::default();
        assert!(resolve_auto_duration_frames(5.0, 0.0, range, TEMPORAL_GRID).is_err());
        assert!(resolve_auto_duration_frames(5.0, f32::NAN, range, TEMPORAL_GRID).is_err());
    }

    // ---------------------------------------------------------------------------------------
    // resolve_request_num_frames — the engine-boundary opt-in seam.
    // ---------------------------------------------------------------------------------------

    /// Acceptance: "An explicit duration wins over auto-duration, proven by test." The spy panics
    /// if called at all, so a passing test proves the prediction path is never reached.
    #[test]
    fn explicit_duration_wins_and_never_predicts() {
        let mut predict = || -> Result<f32> { panic!("predict_seconds must not be called when explicit_frames is Some") };
        let got = resolve_request_num_frames(
            Some(121),
            Some(AutoDurationRange::default()),
            24.0,
            TEMPORAL_GRID,
            &mut predict,
        )
        .unwrap();
        assert_eq!(got, Some(121));
    }

    /// Acceptance: "Reachability: a request that opts in actually reaches the head — declaration is
    /// not enforcement, and a defaulted-off flag that never fires would pass a naive test." A spy
    /// that counts its own invocations proves the opt-in path genuinely calls into the injected
    /// predictor (which, at the engine boundary, is the real `DurationHead::forward`) exactly once.
    #[test]
    fn opt_in_reaches_the_predict_hook() {
        let mut calls = 0u32;
        let mut predict = || -> Result<f32> {
            calls += 1;
            Ok(5.0)
        };
        let got = resolve_request_num_frames(
            None,
            Some(AutoDurationRange::default()),
            24.0,
            TEMPORAL_GRID,
            &mut predict,
        )
        .unwrap();
        assert_eq!(calls, 1, "predict_seconds must be called exactly once");
        assert_eq!(got, Some(113)); // resolve_auto_duration_frames(5.0, 24.0, default) = 113, pinned above
    }

    /// The complementary reachability proof: when neither `explicit_frames` nor `auto_duration` is
    /// set, the predictor must NEVER fire (auto-duration is explicit opt-in only) — this is exactly
    /// the "defaulted-off flag that never fires" case the acceptance criterion warns a naive test
    /// would let slip through, asserted directly rather than left implicit.
    #[test]
    fn neither_flag_never_calls_predict() {
        let mut predict =
            || -> Result<f32> { panic!("predict_seconds must not be called when opted out") };
        let got = resolve_request_num_frames(None, None, 24.0, TEMPORAL_GRID, &mut predict).unwrap();
        assert_eq!(got, None);
    }

    /// A prediction failure propagates rather than silently falling back to "no opinion" or a
    /// default duration.
    #[test]
    fn predict_failure_propagates() {
        let mut predict = || -> Result<f32> { Err(Error::Msg("boom".into())) };
        let err = resolve_request_num_frames(
            None,
            Some(AutoDurationRange::default()),
            24.0,
            TEMPORAL_GRID,
            &mut predict,
        )
        .expect_err("predict failure must propagate");
        assert!(err.to_string().contains("boom"));
    }
}
