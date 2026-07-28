//! The reproducible, level-matched stimulus set for the SA3 medium-vs-small listening panel
//! (`sc-15178`).
//!
//! This is a **generator, not a gate**. It renders the pinned stimulus set that
//! `docs/migration/SC_15178_SA3_LISTENING_PROTOCOL.md` specifies, level-matches every take to a
//! common integrated loudness, writes the WAVs a human panel actually listens to, and emits a
//! manifest recording exactly what was rendered and what gain was applied to each take. The real
//! render case is `#[ignore]`d and needs three multi-GB snapshots; nothing here is a pass/fail
//! quality bar.
//!
//! # Why this file exists at all
//!
//! `tests/variant_quality.rs` establishes what medium's registration *can* claim — both domains,
//! and a checkpoint distinct from each specialist — and states in its own header what it cannot:
//! that medium sounds better. No objective metric that fits inside a `cargo test` can carry that
//! claim for this pair. `tests/dtype_policy.rs` has the numbers: on the registered eight-step
//! Pingpong sampler, medium's F16 render sits at waveform cosine **0.222** against its own F32
//! render *at the same seed*, while two F32 renders at *adjacent seeds* sit at **0.005** — a ~48x
//! ratio. Cross-checkpoint cosines (0.037–0.282, `variant_quality.rs`) sit in the same band as an
//! unrelated take. Any agreement threshold wide enough to admit honest output also admits noise.
//!
//! So the perceptual question is answered by listeners, and the only part of that a machine can
//! own is making the stimuli **reproducible and fair**. That is this file's whole job.
//!
//! # What "fair" means here, concretely
//!
//! **Level matching is the load-bearing control.** An uncorrected loudness difference alone
//! produces a preference — louder reliably reads as "better" in an unlevelled comparison — so a
//! panel run on unmatched takes measures gain staging, not quality. Matching is done on BS.1770-4
//! **gated integrated loudness** via [`candle_audio::harness::MetricSet`], **not** RMS: RMS is
//! frequency-blind, and two takes at identical RMS but different spectral centres differ in
//! perceived loudness by many LU. `level_matching_collapses_a_loudness_gap_that_rms_matching_leaves`
//! measures that gap rather than asserting it from prose.
//!
//! The common target sits at or below the quietest take, so **every applied gain is attenuating**
//! and no take can be pushed into clipping. It is also pulled down by the set's worst
//! **peak-to-loudness ratio**, because loudness normalization alone does not bound the peak — see
//! [`common_target`], which records the measured take that proved it. Both properties hold for the
//! whole set at once, so level matching is exact rather than per-take.
//!
//! # Running it
//!
//! ```text
//! SA3_MEDIUM_SNAPSHOT=...      \
//! SA3_SMALL_MUSIC_SNAPSHOT=... \
//! SA3_SMALL_SFX_SNAPSHOT=...   \
//! SA3_LISTENING_WAV_DIR=/path/to/stimuli \
//!   cargo test -p candle-audio-stable-audio-3 --features metal --release \
//!     --test listening_stimuli -- --ignored --nocapture
//! ```
//!
//! Each variant is loaded **once** and renders every one of its takes in that single process:
//! constructing a generator costs ~43 s of cold start against a 10.4 GB (medium) or 3.45 GB
//! (specialist) snapshot, so a per-take load would dominate the run for no benefit. This mirrors
//! `tests/variant_quality.rs`.
//!
//! The output directory receives one WAV per take plus `manifest.json`. Feed that manifest to
//! `scripts/audio/sa3_listening_blind.py assign` to produce the blinded playlist and the private
//! key; the checkpoint identity of a take never reaches a listener.

use std::path::PathBuf;
use std::time::Duration;

use candle_audio_stable_audio_3::candle_audio;
use candle_audio_stable_audio_3::candle_audio::harness::MetricSet;
use candle_audio_stable_audio_3::gen_core::{
    AudioParams, AudioTrack, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::Variant;

/// The two domains medium serves, each paired with the specialist it is compared against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Domain {
    Music,
    Sfx,
}

impl Domain {
    fn label(self) -> &'static str {
        match self {
            Domain::Music => "music",
            Domain::Sfx => "sfx",
        }
    }

    /// The specialist medium is contrasted against in this domain, and its snapshot env var.
    fn specialist(self) -> (Variant, &'static str) {
        match self {
            Domain::Music => (Variant::SmallMusic, "SA3_SMALL_MUSIC_SNAPSHOT"),
            Domain::Sfx => (Variant::SmallSfx, "SA3_SMALL_SFX_SNAPSHOT"),
        }
    }
}

struct Stimulus {
    /// Stable, opaque-ish id used in filenames and in the manifest. Carries the domain and an
    /// index only — never the checkpoint, which is what the blinding script strips.
    id: &'static str,
    domain: Domain,
    prompt: &'static str,
    /// `true` when this prompt appears in **no** committed test constant anywhere in this crate.
    ///
    /// The protocol requires at least half the set to be held out from anything already used as a
    /// gate operating point, so the panel is not scored entirely on prompts the crate's floors were
    /// calibrated against. The fraction is asserted here
    /// ([`MIN_HELD_OUT_FRACTION`]); the *truth* of each flag is a cross-file property and is
    /// asserted by `scripts/tests/test_sa3_listening_blind.py`, which re-derives it by scanning
    /// every other test source in the crate.
    held_out: bool,
}

/// The pinned stimulus set: three music prompts, three SFX prompts, four of the six held out.
///
/// # Why the held-out four are *not* `demo_cond` prompts
///
/// The obvious design — draw the whole set from the checkpoints' own shipped `demo_cond` lists — is
/// unavailable, and finding that out is part of this story's result. `tests/provider.rs` commits
/// the SA3 `demo_cond` prompts as side-ratio calibration constants (`SFX_SWEEP_PROMPTS`,
/// `MUSIC_SWEEP_PROMPTS`, `MEDIUM_SWEEP_PROMPTS`, `MUSIC_BASE_SWEEP_PROMPTS`), and those sweeps are
/// what the shipped floors were *tuned on*, so every prompt in them is a gate operating point.
///
/// Counted against the six on-disk snapshots' `model_config.json`, the pool is **11 distinct
/// prompts across 18 per-snapshot entries**, and the crate commits all of them: medium ships **2**
/// (both in `MEDIUM_SWEEP_PROMPTS`); `small-music`, `small-sfx`, `small-music-base` and
/// `small-sfx-base` ship 4 each; `medium-base` ships none. Two of those lists collapse into others.
/// `small-sfx-base` ships the *same four as `small-music-base`* — music prompts, not SFX ones — and
/// `small-music-base` in turn **shares 3 of its 4 with `small-music`**, differing only in that its
/// Latin-jazz entry ends "…and a whispered melodic female voice". 18 − 4 − 3 = 11.
///
/// Neither count is asserted: [`DEMO_COND_POOL`] commits the pool per snapshot,
/// `the_committed_demo_cond_pool_matches_the_shipped_configs` re-reads the six `model_config.json`
/// files it was taken from, and `scripts/tests/test_sa3_listening_blind.py` recomputes 11 / 18 / 3
/// from that constant and fails if this sentence or the protocol document states anything else. An
/// earlier revision of this comment said "14 distinct", which is the per-snapshot sum with only
/// `small-sfx-base` deduplicated — a derived number nothing derived.
///
/// Medium's count is worth naming explicitly, because
/// `MEDIUM_SWEEP_PROMPTS`'s own doc comment reads like a *selection* ("two shipped music `demo_cond`
/// prompts from medium's own `model_config.json`"). It is not a selection — two is all medium has.
///
/// There is therefore no held-out prompt anywhere in the `demo_cond` pool, and a set drawn entirely
/// from it would score the panel on the crate's own tuning data.
///
/// So the four held-out entries are authored for this panel, in the same idiom and domains, and
/// appear **in no crate source**. That is the scope `scripts/tests/test_sa3_listening_blind.py`
/// actually scans (`crates/**/*.rs`) and the scope "held out" means here — deliberately not a
/// whole-repository claim, since the protocol document lists all six prompts, as would any write-up
/// of the result.
///
/// The two anchors are deliberate and are the crate's existing gate prompts. They tie the panel to
/// the operating point the real-weight lanes measure at, so a panel result and a gate measurement
/// are about the same renders.
///
/// `sfx-3` ("Dog barking next to a waterfall") is the load-bearing case and is not optional.
/// Medium's stereo side ratio on it collapses to ~1.2e-4 at two of five seeds
/// (`tests/provider.rs`'s two-domain sweep, sc-14545). Sparse SFX is where the two checkpoints are
/// most likely to differ in *character* rather than in fidelity — exactly the difference a
/// preference test can detect and an agreement metric provably cannot. `sfx-1` and `sfx-2` are
/// authored to be sparse for the same reason, so the load-bearing condition is not carried by a
/// single prompt.
const STIMULI: &[Stimulus] = &[
    Stimulus {
        id: "music-1",
        domain: Domain::Music,
        prompt: "Slow downtempo trip-hop with dusty vinyl crackle, muted trumpet and a walking \
                 upright bass",
        held_out: true,
    },
    Stimulus {
        id: "music-2",
        domain: Domain::Music,
        prompt:
            "Bright Afrobeat groove with interlocking guitars, talking drum and a horn section \
                 riff",
        held_out: true,
    },
    Stimulus {
        id: "music-3",
        domain: Domain::Music,
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
        held_out: false,
    },
    Stimulus {
        id: "sfx-1",
        domain: Domain::Sfx,
        prompt: "A single wooden door creaking open in a large empty stone hall",
        held_out: true,
    },
    Stimulus {
        id: "sfx-2",
        domain: Domain::Sfx,
        prompt: "Distant thunder rolling over a quiet field with light rain on leaves",
        held_out: true,
    },
    Stimulus {
        id: "sfx-3",
        domain: Domain::Sfx,
        prompt: "Dog barking next to a waterfall",
        held_out: false,
    },
];

/// The complete shipped `demo_cond` pool, transcribed from the six pinned snapshots'
/// `model_config.json` (revisions in `docs/migration/SC_14534_SA3_REFERENCE_PARITY.md`).
///
/// It exists so the counts in [`STIMULI`]'s doc comment and in §3 of the protocol document are
/// *computed* from a committed table rather than counted by hand. `medium-base` carries no
/// `demo_cond` key at all and is listed with an empty slice so the six-snapshot scope is explicit
/// rather than implied by omission.
///
/// `the_committed_demo_cond_pool_matches_the_shipped_configs` re-reads the snapshots and fails if
/// this table drifts from them; `scripts/tests/test_sa3_listening_blind.py` derives the distinct
/// count, the per-snapshot total and the `small-music` overlap from it, checks the prose against
/// those, and checks that every entry is committed in one of `tests/provider.rs`'s sweep constants.
const DEMO_COND_POOL: &[(&str, &[&str])] = &[
    (
        "stable-audio-3-medium",
        &[
            "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
            "A tropical house track with upbeat melodies, a driving bassline, and cheery vibes",
        ],
    ),
    ("stable-audio-3-medium-base", &[]),
    (
        "stable-audio-3-small-music",
        &[
            "A beautiful piano arpeggio grows into a grand cinematic climax",
            "Elegant and sophisticated Latin jazz piece with a Cuban base",
            "Amen break 174 BPM",
            "lofi house loop",
        ],
    ),
    (
        "stable-audio-3-small-music-base",
        &[
            "A beautiful piano arpeggio grows into a grand cinematic climax",
            "Elegant and sophisticated Latin jazz piece with a Cuban base and a whispered melodic \
             female voice",
            "Amen break 174 BPM",
            "lofi house loop",
        ],
    ),
    (
        "stable-audio-3-small-sfx",
        &[
            "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
            "Dog barking next to a waterfall",
            "Sparkling fantasy energy swirl, mystical shimmer, rising magical burst",
            "Running footsteps on pavement, fast pace, urban street environment, energetic motion \
             sound",
        ],
    ),
    (
        "stable-audio-3-small-sfx-base",
        &[
            "A beautiful piano arpeggio grows into a grand cinematic climax",
            "Elegant and sophisticated Latin jazz piece with a Cuban base and a whispered melodic \
             female voice",
            "Amen break 174 BPM",
            "lofi house loop",
        ],
    ),
];

/// The snapshot env vars behind [`DEMO_COND_POOL`], in the same order.
const DEMO_COND_POOL_ENVS: &[&str] = &[
    "SA3_MEDIUM_SNAPSHOT",
    "SA3_MEDIUM_BASE_SNAPSHOT",
    "SA3_SMALL_MUSIC_SNAPSHOT",
    "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
    "SA3_SMALL_SFX_SNAPSHOT",
    "SA3_SMALL_SFX_BASE_SNAPSHOT",
];

/// At least this fraction of the set must be held out from committed test constants.
const MIN_HELD_OUT_FRACTION: f64 = 0.5;

/// Two seeds per (prompt, checkpoint), deliberately disjoint from the crate's gate seeds
/// (`42 / 7 / 2026`): a panel scored on the exact draws a threshold was calibrated against would
/// inherit that calibration's luck. The pair also supplies the protocol's within-listener replicate
/// — 6 prompts x 2 seeds = the 12 contrast ABX trials each listener takes.
const SEEDS: &[u64] = &[15_178, 15_377];

/// Fixed render controls. 30 s / 8 steps / Pingpong is the operating point every real-weight gate
/// in this crate measures at, so the panel is not listening to a configuration nothing else covers.
const SECONDS: f32 = 30.0;
const STEPS: u32 = 8;
const SAMPLE_RATE: u32 = 44_100;

/// Upper bound on the common loudness target — see [`common_target`] for the other two terms.
///
/// -23 LUFS is the EBU R128 programme target. Capping here rather than simply matching to the
/// quietest take keeps the set at a sane absolute level even if a future stimulus renders hot.
const TARGET_CEILING_LUFS: f32 = -23.0;

/// Post-match tolerance, in LU, both against the target and pairwise across the whole set.
///
/// 0.5 LU is roughly half the smallest loudness step listeners reliably report, so a residual at
/// this bound cannot be the thing a preference is built on. It is asserted after re-measurement,
/// not assumed from the applied gain: BS.1770-4's -70 LUFS absolute gate is fixed in absolute
/// terms, so a uniform gain does not move gated integrated loudness by exactly the gain in dB.
const MAX_LUFS_DELTA: f32 = 0.5;

/// Post-match true-peak ceiling, in dBTP. Inter-sample overs are a real, audible artefact of D/A
/// reconstruction, and one clipping take in a preference panel is a confound.
const MAX_TRUE_PEAK_DBTP: f32 = -1.0;

/// Slack subtracted when deriving the target from the peak constraint, absorbing the residual the
/// loudness loop is allowed to leave (it stops inside `MAX_LUFS_DELTA / 10`).
const TRUE_PEAK_SAFETY_MARGIN_DB: f32 = 0.1;

/// At most this many correction passes when converging on the target loudness. One pass is exact
/// to well under the tolerance for programme-level material; the second exists for the gating
/// non-linearity, and the loop asserts convergence rather than silently accepting a miss.
const MAX_LEVEL_MATCH_PASSES: usize = 3;

/// The one loudness target the whole set is matched to.
///
/// # Why this is not just "the quietest take"
///
/// Loudness normalization alone does **not** bound the peak, and on this set it demonstrably fails
/// to. Measured on Metal at 30 s / 8 steps, `small_sfx` on the *"Distant thunder rolling over a
/// quiet field"* prompt renders at a **peak-to-loudness ratio of ~23.2 dB** — sparse, transient,
/// almost all of it near silence. Matched to the set's quietest integrated loudness (-23.1 LUFS) its
/// true peak lands at **+0.031 dBTP**: above full scale, from a take whose sample values were all
/// inside [-1, 1]. That is what 4x-oversampled true peak measures and sample peak misses.
///
/// The alternatives are worse. A limiter would alter the audio being judged. Per-take attenuation
/// would break the level matching, which is the control this whole protocol rests on. So the
/// **target itself** is peak-constrained: it is pulled down by the set's worst peak-to-loudness
/// ratio, so after matching, every take clears the ceiling and the set stays matched to a single
/// number. The whole set gets quieter together, which costs nothing — a listener sets playback level
/// once, and only the *relative* match matters.
///
/// The three terms:
///
/// 1. the quietest take, so every applied gain attenuates and no take can be pushed into clipping;
/// 2. [`TARGET_CEILING_LUFS`], the programme-level cap;
/// 3. `MAX_TRUE_PEAK_DBTP - max(true_peak - integrated_lufs)`, the peak constraint. True peak scales
///    with the applied gain in dB, so take `i` ends at `target + PLR_i`; bounding by the worst
///    `PLR` bounds all of them at once.
fn common_target(measurements: &[MetricSet]) -> f32 {
    let quietest = measurements
        .iter()
        .map(|m| m.integrated_lufs)
        .fold(f32::INFINITY, f32::min);
    let worst_peak_to_loudness = measurements
        .iter()
        .map(|m| m.true_peak_dbtp - m.integrated_lufs)
        .fold(f32::NEG_INFINITY, f32::max);
    quietest
        .min(TARGET_CEILING_LUFS)
        .min(MAX_TRUE_PEAK_DBTP - worst_peak_to_loudness - TRUE_PEAK_SAFETY_MARGIN_DB)
}

fn snapshot(env: &str) -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    ))
}

fn request(prompt: &str, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        seed: Some(seed),
        steps: Some(STEPS),
        sampler: Some("pingpong".into()),
        audio: Some(AudioParams {
            target_duration: Some(SECONDS),
            sample_rate: Some(SAMPLE_RATE),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn measure(track: &AudioTrack) -> MetricSet {
    MetricSet::measure(track, Duration::ZERO).expect("loudness measurement over a valid track")
}

/// What was done to one take to bring it onto the common target, recorded per take in the manifest
/// so the level matching is auditable after the fact rather than trusted.
struct LevelMatch {
    pre_lufs: f32,
    post_lufs: f32,
    post_true_peak_dbtp: f32,
    gain_linear: f32,
}

impl LevelMatch {
    fn gain_db(&self) -> f32 {
        20.0 * self.gain_linear.max(f32::MIN_POSITIVE).log10()
    }
}

/// Scale `track` onto `target_lufs`, re-measuring after each pass, and return what was applied.
///
/// The loop is the point. Applying `10^((target - measured)/20)` once and declaring victory would
/// be assuming the gated meter is linear in the gain, which it is not: BS.1770-4's absolute -70
/// LUFS gate does not move with the signal, so blocks near the floor can enter or leave the gated
/// mean. The residual is tiny for programme material, but "tiny" is a measurement, so it is
/// measured.
fn level_match(track: &mut AudioTrack, target_lufs: f32) -> LevelMatch {
    let initial = measure(track);
    let mut total_gain = 1.0f32;
    let mut latest = initial.clone();

    for _ in 0..MAX_LEVEL_MATCH_PASSES {
        let error = target_lufs - latest.integrated_lufs;
        if error.abs() < MAX_LUFS_DELTA * 0.1 {
            break;
        }
        let gain = 10f32.powf(error / 20.0);
        for sample in &mut track.samples {
            *sample *= gain;
        }
        total_gain *= gain;
        latest = measure(track);
    }

    // Convergence is asserted *here*, not left to callers. Returning a non-converged track and
    // trusting whoever called to notice is how a silently mis-levelled take reaches a listener, and
    // level matching is the control the whole protocol rests on.
    let residual = (latest.integrated_lufs - target_lufs).abs();
    assert!(
        residual < MAX_LUFS_DELTA,
        "level_match did not converge: {residual:.3} LU from the {target_lufs:.3} LUFS target after \
         {MAX_LEVEL_MATCH_PASSES} passes, at or above the {MAX_LUFS_DELTA} LU tolerance"
    );

    LevelMatch {
        pre_lufs: initial.integrated_lufs,
        post_lufs: latest.integrated_lufs,
        post_true_peak_dbtp: latest.true_peak_dbtp,
        gain_linear: total_gain,
    }
}

/// One rendered take before level matching.
struct Take {
    stimulus: usize,
    variant: Variant,
    seed: u64,
    track: AudioTrack,
}

impl Take {
    fn file_stem(&self) -> String {
        format!(
            "{}_{}_seed{}",
            STIMULI[self.stimulus].id,
            self.variant.model_id(),
            self.seed
        )
    }
}

fn assert_valid(label: &str, track: &AudioTrack) {
    assert_eq!(track.sample_rate, SAMPLE_RATE, "{label}: sample rate");
    assert_eq!(track.channels, 2, "{label}: channel count");
    assert_eq!(
        track.samples.len(),
        (SECONDS as f64 * f64::from(SAMPLE_RATE)).floor() as usize * 2,
        "{label}: frame count"
    );
    assert!(
        track
            .samples
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)),
        "{label}: non-finite or unclamped PCM"
    );
    let energy: f64 = track.samples.iter().map(|s| f64::from(*s).powi(2)).sum();
    assert!(
        (energy / track.samples.len() as f64).sqrt() > 1e-4,
        "{label}: output is silent"
    );
}

/// Load one variant once and render every `(stimulus, seed)` pair it owns.
fn render_all(variant: Variant, env: &str, wanted: &[usize]) -> Vec<Take> {
    let generator = candle_audio_stable_audio_3::provider_registry()
        .expect("provider registry")
        .load(variant.model_id(), &LoadSpec::new(snapshot(env)))
        .expect("strict registered variant-bound load");

    let mut takes = Vec::with_capacity(wanted.len() * SEEDS.len());
    for &stimulus in wanted {
        for &seed in SEEDS {
            let track = match generator
                .generate(&request(STIMULI[stimulus].prompt, seed), &mut |_| {})
                .expect("connected generation")
            {
                GenerationOutput::Audio(track) => track,
                other => panic!("expected audio, got {other:?}"),
            };
            let take = Take {
                stimulus,
                variant,
                seed,
                track,
            };
            assert_valid(&take.file_stem(), &take.track);
            takes.push(take);
        }
    }
    takes
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render the pinned set, level-match it, and write the WAVs plus the manifest the blinding script
/// consumes.
///
/// Everything this asserts is a property of the *stimuli*, not of either checkpoint's quality:
/// the takes are valid audio, the set is at one loudness, and nothing clips. A quality claim is
/// produced by the panel in `docs/migration/SC_15178_SA3_LISTENING_PROTOCOL.md`, which is executed
/// by human listeners and tracked separately as **sc-15377**. This target must never grow an
/// assertion that one checkpoint beats another.
#[test]
#[ignore = "requires the pinned medium, small-music and small-sfx snapshots and an output dir"]
fn generate_the_level_matched_listening_stimulus_set() {
    let out_dir = PathBuf::from(
        std::env::var("SA3_LISTENING_WAV_DIR")
            .expect("set SA3_LISTENING_WAV_DIR to the stimulus output directory"),
    );
    std::fs::create_dir_all(&out_dir).expect("create the stimulus output directory");

    let held_out = STIMULI.iter().filter(|s| s.held_out).count();
    assert!(
        held_out as f64 / STIMULI.len() as f64 >= MIN_HELD_OUT_FRACTION,
        "only {held_out} of {} stimuli are held out from committed test constants; the protocol \
         requires at least half, or the panel is scored on the prompts the gates were tuned on",
        STIMULI.len()
    );

    // Medium serves both domains, so it renders every stimulus; each specialist renders only its
    // own. Three loads total rather than twenty-four.
    let all: Vec<usize> = (0..STIMULI.len()).collect();
    let mut takes = render_all(Variant::Medium, "SA3_MEDIUM_SNAPSHOT", &all);
    for domain in [Domain::Music, Domain::Sfx] {
        let (specialist, env) = domain.specialist();
        let owned: Vec<usize> = (0..STIMULI.len())
            .filter(|&i| STIMULI[i].domain == domain)
            .collect();
        takes.extend(render_all(specialist, env, &owned));
    }

    // One target for the whole set: every gain attenuates, and the worst peak-to-loudness ratio in
    // the set is what pulls the target down far enough that nothing clips. See `common_target`.
    let measurements: Vec<MetricSet> = takes.iter().map(|take| measure(&take.track)).collect();
    let target = common_target(&measurements);
    let quietest = measurements
        .iter()
        .map(|m| m.integrated_lufs)
        .fold(f32::INFINITY, f32::min);
    let worst_plr = measurements
        .iter()
        .map(|m| m.true_peak_dbtp - m.integrated_lufs)
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "listening set: {} takes, quietest {quietest:.3} LUFS, worst peak-to-loudness ratio \
         {worst_plr:.3} dB, common target {target:.3} LUFS",
        takes.len()
    );

    let mut entries = Vec::with_capacity(takes.len());
    let mut post_lufs = Vec::with_capacity(takes.len());
    for take in &mut takes {
        let matched = level_match(&mut take.track, target);
        assert!(
            matched.gain_db() <= 0.001,
            "{}: level match applied {:+.3} dB of make-up gain; the target sits at or below the \
             quietest take so every gain attenuates and no take can be pushed into clipping",
            take.file_stem(),
            matched.gain_db()
        );
        assert!(
            (matched.post_lufs - target).abs() < MAX_LUFS_DELTA,
            "{}: post-match loudness {:.3} LUFS is {:.3} LU off the {target:.3} LUFS target",
            take.file_stem(),
            matched.post_lufs,
            matched.post_lufs - target
        );
        assert!(
            matched.post_true_peak_dbtp <= MAX_TRUE_PEAK_DBTP,
            "{}: post-match true peak {:.3} dBTP exceeds {MAX_TRUE_PEAK_DBTP} dBTP; the common \
             target is supposed to be pulled down by the set's worst peak-to-loudness ratio so \
             this cannot happen — see common_target",
            take.file_stem(),
            matched.post_true_peak_dbtp
        );
        assert_valid(&take.file_stem(), &take.track);

        let name = format!("{}.wav", take.file_stem());
        candle_audio::wav::write_wav_pcm16(&out_dir.join(&name), &take.track).expect("write WAV");
        eprintln!(
            "{}: pre={:.3} LUFS gain={:+.3} dB post={:.3} LUFS peak={:.3} dBTP",
            take.file_stem(),
            matched.pre_lufs,
            matched.gain_db(),
            matched.post_lufs,
            matched.post_true_peak_dbtp
        );

        post_lufs.push(matched.post_lufs);
        let stimulus = &STIMULI[take.stimulus];
        entries.push(format!(
            "    {{\"file\": {}, \"stimulus\": {}, \"domain\": {}, \"prompt\": {}, \
             \"held_out\": {}, \"variant\": {}, \"seed\": {}, \"seconds\": {SECONDS}, \
             \"steps\": {STEPS}, \"sampler\": \"pingpong\", \"sample_rate\": {SAMPLE_RATE}, \
             \"pre_lufs\": {:.4}, \"gain_linear\": {:.9}, \"gain_db\": {:.4}, \
             \"post_lufs\": {:.4}, \"post_true_peak_dbtp\": {:.4}, \"pcm_sha256\": {}}}",
            json_string(&name),
            json_string(stimulus.id),
            json_string(stimulus.domain.label()),
            json_string(stimulus.prompt),
            stimulus.held_out,
            json_string(take.variant.model_id()),
            take.seed,
            matched.pre_lufs,
            matched.gain_linear,
            matched.gain_db(),
            matched.post_lufs,
            matched.post_true_peak_dbtp,
            json_string(&measure(&take.track).pcm_sha256),
        ));
    }

    // The pairwise assertion is what actually rules out "the louder one won". Matching each take to
    // a target individually could still leave two takes half a tolerance apart in opposite
    // directions, which is the comparison a listener makes.
    let widest = post_lufs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - post_lufs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    eprintln!("widest pairwise post-match loudness delta: {widest:.4} LU");
    assert!(
        widest < MAX_LUFS_DELTA,
        "widest pairwise post-match loudness delta {widest:.4} LU is at or above the \
         {MAX_LUFS_DELTA} LU tolerance; an uncorrected loudness difference alone produces a \
         preference, so the panel must not run on this set"
    );

    let manifest = format!(
        "{{\n  \"story\": \"sc-15178\",\n  \"protocol\": \
         \"docs/migration/SC_15178_SA3_LISTENING_PROTOCOL.md\",\n  \
         \"target_lufs\": {target:.4},\n  \"quietest_take_lufs\": {quietest:.4},\n  \
         \"worst_peak_to_loudness_db\": {worst_plr:.4},\n  \
         \"max_lufs_delta\": {MAX_LUFS_DELTA},\n  \
         \"max_true_peak_dbtp\": {MAX_TRUE_PEAK_DBTP},\n  \
         \"widest_pairwise_lufs_delta\": {widest:.4},\n  \"takes\": [\n{}\n  ]\n}}\n",
        entries.join(",\n")
    );
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest).expect("write the stimulus manifest");
    eprintln!("stimulus manifest: {}", manifest_path.display());
}

/// Re-read the six pinned `model_config.json` files and check [`DEMO_COND_POOL`] against them.
///
/// This is what makes the pool a transcription rather than a memory. It needs the snapshots only
/// for their configs — a few kilobytes each, no weights are loaded — but it still needs all six
/// paths, so it is `#[ignore]`d with the crate's other snapshot-dependent cases and runs in the
/// real-weight lanes. The counts derived *from* the pool are checked weight-free, in Python.
#[test]
#[ignore = "requires the six pinned snapshots' model_config.json"]
fn the_committed_demo_cond_pool_matches_the_shipped_configs() {
    assert_eq!(
        DEMO_COND_POOL.len(),
        DEMO_COND_POOL_ENVS.len(),
        "the pool and its env-var list must stay aligned"
    );
    for ((model_id, committed), env) in DEMO_COND_POOL.iter().zip(DEMO_COND_POOL_ENVS) {
        let path = PathBuf::from(
            std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned snapshot")),
        )
        .join("model_config.json");
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        // `training.demo.demo_cond` is absent entirely on `medium-base`; absent and empty are the
        // same fact here, and the committed table records it as an empty slice.
        let shipped: Vec<String> = config["training"]["demo"]["demo_cond"]
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| {
                        entry["prompt"]
                            .as_str()
                            .unwrap_or_else(|| {
                                panic!("{model_id}: a demo_cond entry carries no prompt")
                            })
                            .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();
        eprintln!("{model_id}: {} shipped demo_cond prompt(s)", shipped.len());
        assert_eq!(
            shipped,
            committed.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            "{model_id}: DEMO_COND_POOL no longer matches {}",
            path.display()
        );
    }
}

/// The level-matching control, without weights: an equal-**RMS** pair is *not* an equal-loudness
/// pair, and the LUFS match closes the gap RMS matching leaves open.
///
/// This is one of the target's three non-`#[ignore]`d cases, and it exists because the protocol's
/// central control would otherwise be prose. Two tones at identical RMS — one at 60 Hz, one at
/// 3 kHz — sit many LU apart under BS.1770-4's K-weighting, which high-passes the low end and
/// boosts the presence region. A panel level-matched on RMS would therefore hear a loudness
/// difference and report a preference for it. The test measures that gap (so the claim is a
/// number, not an assertion), then matches both onto a common target and requires the gap to
/// collapse inside [`MAX_LUFS_DELTA`], with no clipping.
///
/// It is named in the `Test Stable Audio 3 weight-free quality gates` step of
/// `.github/workflows/ci.yml`; `scripts/tests/test_sa3_ci_target_coverage.py` fails the build if it
/// ever is not.
#[test]
fn level_matching_collapses_a_loudness_gap_that_rms_matching_leaves() {
    /// Three seconds is comfortably more than BS.1770-4's 400 ms gating block, so the integrated
    /// measurement is a real gated mean rather than the short-signal fallback.
    fn tone(frequency: f32, rms: f32) -> AudioTrack {
        let frames = (SAMPLE_RATE * 3) as usize;
        // A sine of amplitude `a` has RMS `a / sqrt(2)`.
        let amplitude = rms * std::f32::consts::SQRT_2;
        let mut samples = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            let phase = std::f32::consts::TAU * frequency * frame as f32 / SAMPLE_RATE as f32;
            let value = amplitude * phase.sin();
            samples.push(value);
            samples.push(value);
        }
        AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            stems: Vec::new(),
        }
    }

    let mut low = tone(60.0, 0.2);
    let mut high = tone(3_000.0, 0.2);

    // The construction has to actually be RMS-matched, or the gap measured below is about
    // something else.
    let rms = |track: &AudioTrack| -> f64 {
        (track
            .samples
            .iter()
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>()
            / track.samples.len() as f64)
            .sqrt()
    };
    assert!(
        (rms(&low) - rms(&high)).abs() < 1e-6,
        "the control pair is not RMS-matched: {} vs {}",
        rms(&low),
        rms(&high)
    );

    let gap = (measure(&low).integrated_lufs - measure(&high).integrated_lufs).abs();
    eprintln!(
        "RMS-matched 60 Hz vs 3 kHz: {:.3} LUFS vs {:.3} LUFS ({gap:.3} LU apart)",
        measure(&low).integrated_lufs,
        measure(&high).integrated_lufs
    );
    assert!(
        gap > MAX_LUFS_DELTA,
        "an RMS-matched pair sits only {gap:.3} LU apart, so this control cannot show that RMS \
         matching is insufficient — the protocol's reason for specifying LUFS would be unevidenced"
    );

    let target = common_target(&[measure(&low), measure(&high)]);
    let low_match = level_match(&mut low, target);
    let high_match = level_match(&mut high, target);
    let matched_gap = (low_match.post_lufs - high_match.post_lufs).abs();
    eprintln!(
        "after LUFS matching to {target:.3} LUFS: {:.3} vs {:.3} ({matched_gap:.3} LU apart), \
         gains {:+.3} dB / {:+.3} dB",
        low_match.post_lufs,
        high_match.post_lufs,
        low_match.gain_db(),
        high_match.gain_db()
    );

    assert!(
        matched_gap < MAX_LUFS_DELTA,
        "LUFS matching left the pair {matched_gap:.3} LU apart, at or above the {MAX_LUFS_DELTA} \
         LU tolerance the generator asserts"
    );
    assert!(
        (low_match.post_lufs - target).abs() < MAX_LUFS_DELTA
            && (high_match.post_lufs - target).abs() < MAX_LUFS_DELTA,
        "level_match did not converge onto the target"
    );
    // Attenuation-only, exactly as the real set requires.
    assert!(
        low_match.gain_db() <= 0.001 && high_match.gain_db() <= 0.001,
        "level matching applied make-up gain ({:+.3} dB / {:+.3} dB); the target is the quieter of \
         the pair, so both gains must attenuate",
        low_match.gain_db(),
        high_match.gain_db()
    );
    assert!(
        low_match.post_true_peak_dbtp <= MAX_TRUE_PEAK_DBTP
            && high_match.post_true_peak_dbtp <= MAX_TRUE_PEAK_DBTP,
        "level matching produced a take above the {MAX_TRUE_PEAK_DBTP} dBTP ceiling"
    );
}

/// The peak constraint's control, without weights — a replay of the measurement that produced it.
///
/// The first real run of the generator above failed, and the failure is the reason
/// [`common_target`] has a third term. Matching the set to its quietest integrated loudness
/// (-23.149 LUFS) left `small_sfx`'s *"Distant thunder rolling over a quiet field"* take at
/// **+0.031 dBTP** — over full scale, from PCM whose sample values were all inside [-1, 1], because
/// 4x-oversampled true peak catches inter-sample overs that sample peak does not. That take's
/// peak-to-loudness ratio is ~23.2 dB.
///
/// The numbers below are that measurement. The test asserts both directions: a loudness-only target
/// **fails** the ceiling (so the constraint is not solving an imaginary problem), and
/// [`common_target`] clears it (so the fix works). Without the first assertion this would be a
/// test of a value that was never in danger.
#[test]
fn the_common_target_is_pulled_down_by_the_sets_worst_peak_to_loudness_ratio() {
    /// Only the two loudness fields participate in [`common_target`]; the rest are inert here.
    fn observed(integrated_lufs: f32, true_peak_dbtp: f32) -> MetricSet {
        MetricSet {
            latency: Duration::ZERO,
            peak_rss_bytes: None,
            duration_secs: SECONDS as f64,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            clipped_samples: 0,
            integrated_lufs,
            true_peak_dbtp,
            pcm_sha256: String::new(),
            prompt_adherence: None,
            lyric_alignment: None,
        }
    }

    // Three takes measured on Metal at 30 s / 8 steps, in the run that found this. Each pair is
    // `(integrated LUFS, true peak dBTP)` before matching.
    //
    // `transient` is `sfx-2 / small_sfx / seed 15377`, the sparse thunder take: it was the set's
    // quietest, so it took ~0 dB of gain and landed at +0.031 dBTP — the observed failure.
    let transient = observed(-23.149, 0.031);
    let sparse_music = observed(-21.108, -2.805);
    let dense_music = observed(-11.413, 0.049);
    let set = [transient.clone(), sparse_music, dense_music];

    let worst_plr = transient.true_peak_dbtp - transient.integrated_lufs;
    assert!(
        worst_plr > 20.0,
        "the fixture no longer carries the high peak-to-loudness take this control is about \
         ({worst_plr:.3} dB)"
    );

    // What the generator did before this constraint existed: loudness-only.
    let loudness_only = transient.integrated_lufs.min(TARGET_CEILING_LUFS);
    let unconstrained_peak = loudness_only + worst_plr;
    eprintln!(
        "loudness-only target {loudness_only:.3} LUFS leaves the transient take at \
         {unconstrained_peak:.3} dBTP"
    );
    assert!(
        unconstrained_peak > MAX_TRUE_PEAK_DBTP,
        "a loudness-only target already clears the {MAX_TRUE_PEAK_DBTP} dBTP ceiling here, so this \
         control proves nothing about the peak constraint"
    );

    let target = common_target(&set);
    eprintln!(
        "peak-constrained target {target:.3} LUFS ({:.3} dB below loudness-only)",
        target - loudness_only
    );
    assert!(
        target <= loudness_only,
        "the peak constraint must only ever lower the target, never raise it"
    );
    for measured in &set {
        let after = measured.true_peak_dbtp + (target - measured.integrated_lufs);
        assert!(
            after <= MAX_TRUE_PEAK_DBTP,
            "matching to {target:.3} LUFS leaves a take at {after:.3} dBTP, above the \
             {MAX_TRUE_PEAK_DBTP} dBTP ceiling"
        );
    }

    // The programme-level cap must still bind when no take is peak-constrained, or the ceiling
    // constant is dead code.
    let tame = [observed(-14.0, -11.0), observed(-16.0, -13.0)];
    assert!(
        (common_target(&tame) - TARGET_CEILING_LUFS).abs() < 1e-4,
        "with no high-PLR take the target should fall back to the {TARGET_CEILING_LUFS} LUFS cap"
    );
}

// ---------------------------------------------------------------------------
// The screening practice contrast (protocol §6, screening criterion 3).
// ---------------------------------------------------------------------------

/// Duration of each practice take, in seconds. Six ABX trials on 4 s takes is a ~3 minute block.
const PRACTICE_SECONDS: f32 = 4.0;

/// Seed for the practice source's noise bed. Disjoint from [`SEEDS`] and from the crate's gate
/// seeds, though nothing depends on that here — the practice material is never rendered by a model.
const PRACTICE_NOISE_SEED: u64 = 15_178_004;

/// Nominal cutoff of the practice low-pass, in Hz. The protocol quotes this number.
const PRACTICE_CUTOFF_HZ: f32 = 4_000.0;

/// Amplitude of the noise bed before the transient envelope.
const PRACTICE_AMPLITUDE: f32 = 0.6;

/// Per-sample decay of the transient envelope, and how often it re-triggers, in frames.
const PRACTICE_DECAY: f32 = 0.999_75;
const PRACTICE_RETRIGGER_FRAMES: usize = 11_025;

/// Half (centre outward) of a 101-tap linear-phase windowed-sinc low-pass: Blackman window,
/// −6 dB at 4 kHz, flat to 3 kHz, below −49 dB by 5 kHz. Committed as literals rather than
/// designed at runtime because designing it needs `sin`, and `sin` is a platform libm detail — the
/// digests below would then differ per target. Everything on the digest path here is `+`, `-` and
/// `*` on `f32`, which IEEE-754 pins exactly. The response *check* in the test may use `sin`/`cos`
/// freely: it asserts a tolerance, not a hash.
#[rustfmt::skip]
const PRACTICE_LOWPASS_HALF_TAPS: &[f32] = &[
    1.8141413e-01, 1.7147432e-01, 1.4367366e-01, 1.03564925e-01,
    5.8865767e-02, 1.7602999e-02, -1.372459e-02, -3.1484712e-02,
    -3.5432357e-02, -2.8339086e-02, -1.4908495e-02, -3.3818043e-04,
    1.1047597e-02, 1.6734675e-02, 1.6327875e-02, 1.1237069e-02,
    3.9227908e-03, -3.015015e-03, -7.6179365e-03, -9.015534e-03,
    -7.465476e-03, -4.049527e-03, -1.8110452e-04, 2.8899254e-03,
    4.422586e-03, 4.3028463e-03, 2.943159e-03, 1.0396474e-03,
    -7.067358e-04, -1.8088807e-03, -2.095281e-03, -1.6881531e-03,
    -8.930223e-04, -5.6162324e-05, 5.55905e-04, 8.205542e-04,
    7.59546e-04, 4.91795e-04, 1.666423e-04, -9.455298e-05,
    -2.3067648e-04, -2.4292208e-04, -1.7435457e-04, -8.069043e-05,
    -5.533947e-06, 3.1911775e-05, 3.553106e-05, 2.1860295e-05,
    7.5354137e-06, 7.9058015e-07, 1.93534e-20,
];

/// SHA-256 over the practice takes' PCM, in the canonical little-endian `f32` serialization
/// [`candle_audio::harness::pcm_sha256`] defines. These are the pinned screening material.
const PRACTICE_FULL_BAND_PCM_SHA256: &str =
    "59b1ad3a7f7a16e5107cab50d8c5499ab62f868951fe0daaadec540c971a2036";
const PRACTICE_LOW_PASSED_PCM_SHA256: &str =
    "a885449d672a73b54065be121a44fa0fa7eec6d8d65eb792346c6461f9f92f60";

/// The practice source: a deterministic broadband bed with re-triggered transients.
///
/// Deliberately **synthetic and model-free**. The protocol requires the practice contrast to use no
/// panel stimulus and no panel checkpoint; generating it from a committed PRNG rather than from any
/// checkpoint makes that structural instead of a promise, and it means the block exists without
/// provisioning weights. Broadband noise plus hard transients is also the easiest possible material
/// for a bandwidth ABX, which is what criterion 3 wants: it screens for understanding the task, not
/// for acuity.
fn practice_source() -> AudioTrack {
    let frames = (PRACTICE_SECONDS * SAMPLE_RATE as f32) as usize;
    let mut state = PRACTICE_NOISE_SEED;
    let mut next = || {
        // SplitMix64, the same stream `scripts/audio/sa3_listening_blind.py` commits, so the whole
        // protocol has one PRNG rather than two.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    let mut samples = Vec::with_capacity(frames * 2);
    let mut envelope = 1.0f32;
    for frame in 0..frames {
        if frame % PRACTICE_RETRIGGER_FRAMES == 0 {
            envelope = 1.0;
        }
        // 24 bits into `f32` is exact, and the scale is a power of two, so this mapping is the
        // same bit pattern on every target — which is what lets the digests below be committed.
        let unit = ((next() >> 40) as f32) * (1.0 / 8_388_608.0) - 1.0;
        let value = unit * PRACTICE_AMPLITUDE * envelope;
        samples.push(value);
        samples.push(value);
        envelope *= PRACTICE_DECAY;
    }
    AudioTrack {
        samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        stems: Vec::new(),
    }
}

/// Mirror [`PRACTICE_LOWPASS_HALF_TAPS`] into the full symmetric kernel.
fn practice_taps() -> Vec<f32> {
    let half = PRACTICE_LOWPASS_HALF_TAPS;
    let mut taps = Vec::with_capacity(half.len() * 2 - 1);
    taps.extend(half.iter().rev().copied());
    taps.extend(half.iter().skip(1).copied());
    taps
}

/// Apply the committed low-pass to an interleaved stereo track, per channel, delay-compensated.
///
/// Fixed accumulation order, `f32` throughout, no transcendental calls — so the output is
/// bit-identical on every target and the committed digest means something.
fn practice_low_pass(track: &AudioTrack) -> AudioTrack {
    let taps = practice_taps();
    let centre = taps.len() / 2;
    let channels = track.channels as usize;
    let frames = track.samples.len() / channels;
    let mut samples = vec![0.0f32; track.samples.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            let mut acc = 0.0f32;
            for (tap, coefficient) in taps.iter().enumerate() {
                let source = frame as isize + tap as isize - centre as isize;
                if source >= 0 && (source as usize) < frames {
                    acc += coefficient * track.samples[source as usize * channels + channel];
                }
            }
            samples[frame * channels + channel] = acc;
        }
    }
    AudioTrack {
        samples,
        sample_rate: track.sample_rate,
        channels: track.channels,
        stems: Vec::new(),
    }
}

/// Magnitude response of the committed kernel at `frequency`, in dB. Test-side only.
fn practice_response_db(frequency: f32) -> f32 {
    let taps = practice_taps();
    let centre = (taps.len() - 1) as f64 / 2.0;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (tap, coefficient) in taps.iter().enumerate() {
        let phase = -std::f64::consts::TAU * f64::from(frequency) * (tap as f64 - centre)
            / f64::from(SAMPLE_RATE);
        re += f64::from(*coefficient) * phase.cos();
        im += f64::from(*coefficient) * phase.sin();
    }
    (20.0 * (re.hypot(im) + 1e-30).log10()) as f32
}

/// Generate and pin the screening practice contrast, and check it is the contrast §6 describes.
///
/// # Why this case exists
///
/// Screening criterion 3 — six trials of a take at full bandwidth against the same take low-passed
/// at 4 kHz, 5 of 6 to pass — is the protocol's **only performance-based inclusion criterion**. It
/// was specified without naming which take, and no committed generator produced the block, so the
/// one criterion that can exclude a listener rested on material chosen on the day. In a protocol
/// whose thesis is that nothing is decided per session, that was the last thing decided per
/// session.
///
/// The material is therefore committed: a deterministic source, a committed kernel, and a digest
/// for each take. Nothing about it can vary between sessions, and nothing about it touches the
/// panel — no panel prompt, no panel checkpoint, no model at all.
///
/// The case asserts, in order: that the kernel really is the low-pass §6 quotes (passband flat,
/// −6 dB at the cutoff, deep stopband an octave up); that the source is valid, non-silent,
/// unclipped audio; that the contrast is audible in the only machine-checkable sense — the takes
/// differ, and the low-passed one has lost most of its energy; and that both digests are the
/// committed ones. Setting `SA3_LISTENING_WAV_DIR` also writes the two WAVs, so the block is
/// produced by the same committed code that pins it.
#[test]
fn the_screening_practice_contrast_is_pinned_material_not_a_run_time_choice() {
    // 1. The kernel is the filter the protocol names.
    let passband = practice_response_db(1_000.0);
    let cutoff = practice_response_db(PRACTICE_CUTOFF_HZ);
    let octave_up = practice_response_db(PRACTICE_CUTOFF_HZ * 2.0);
    eprintln!(
        "practice low-pass: {passband:.2} dB at 1 kHz, {cutoff:.2} dB at {PRACTICE_CUTOFF_HZ} Hz, \
         {octave_up:.2} dB an octave up"
    );
    assert!(
        passband.abs() < 0.5,
        "the practice kernel is not flat in the passband ({passband:.2} dB at 1 kHz)"
    );
    assert!(
        (-7.0..=-5.0).contains(&cutoff),
        "the practice kernel's {PRACTICE_CUTOFF_HZ} Hz point is {cutoff:.2} dB; §6 calls this a \
         {PRACTICE_CUTOFF_HZ} Hz low-pass, so the cutoff must sit at the usual −6 dB"
    );
    assert!(
        octave_up < -40.0,
        "the practice kernel only reaches {octave_up:.2} dB an octave above the cutoff; a listener \
         could fail this block for lack of a difference rather than for lack of understanding"
    );

    // 2. Both takes are valid, unclipped, non-silent audio.
    let full = practice_source();
    let low_passed = practice_low_pass(&full);
    let frames = (PRACTICE_SECONDS * SAMPLE_RATE as f32) as usize;
    for (label, track) in [("practice-full", &full), ("practice-lowpass", &low_passed)] {
        assert_eq!(track.sample_rate, SAMPLE_RATE, "{label}: sample rate");
        assert_eq!(track.channels, 2, "{label}: channel count");
        assert_eq!(track.samples.len(), frames * 2, "{label}: frame count");
        assert!(
            track
                .samples
                .iter()
                .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)),
            "{label}: non-finite or unclamped PCM"
        );
        let energy: f64 = track.samples.iter().map(|s| f64::from(*s).powi(2)).sum();
        let rms = (energy / track.samples.len() as f64).sqrt();
        eprintln!("{label}: rms {rms:.6}");
        assert!(rms > 1e-3, "{label}: output is effectively silent");
    }

    // 3. The contrast is real. Broadband noise low-passed at 4 kHz of a 22.05 kHz band loses most
    //    of its power; if it did not, the block would be a coin flip and criterion 3 would exclude
    //    listeners at random.
    let power = |track: &AudioTrack| -> f64 {
        track
            .samples
            .iter()
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>()
    };
    let retained = power(&low_passed) / power(&full);
    eprintln!(
        "practice low-pass retains {:.1}% of the source power",
        retained * 100.0
    );
    assert!(
        retained < 0.35,
        "the low-passed take retains {:.1}% of the source power; the two takes are too close for a \
         6-trial screening block to be the near-certain pass §6 assumes",
        retained * 100.0
    );
    assert_ne!(
        full.samples, low_passed.samples,
        "the filter is a no-op; there is no contrast to practise on"
    );

    // 4. The material is the committed material.
    let full_digest = candle_audio::harness::pcm_sha256(&full.samples);
    let low_passed_digest = candle_audio::harness::pcm_sha256(&low_passed.samples);
    eprintln!("practice-full    pcm_sha256 {full_digest}");
    eprintln!("practice-lowpass pcm_sha256 {low_passed_digest}");
    assert_ne!(
        full_digest, low_passed_digest,
        "both practice takes hash the same, so one of them is not what it claims to be"
    );
    assert_eq!(
        full_digest, PRACTICE_FULL_BAND_PCM_SHA256,
        "the full-bandwidth practice take is not the committed one; screening material may not \
         change between sessions"
    );
    assert_eq!(
        low_passed_digest, PRACTICE_LOW_PASSED_PCM_SHA256,
        "the low-passed practice take is not the committed one; screening material may not change \
         between sessions"
    );

    if let Ok(dir) = std::env::var("SA3_LISTENING_WAV_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create the stimulus output directory");
        candle_audio::wav::write_wav_pcm16(&dir.join("practice_full.wav"), &full)
            .expect("write the full-bandwidth practice take");
        candle_audio::wav::write_wav_pcm16(&dir.join("practice_lowpass.wav"), &low_passed)
            .expect("write the low-passed practice take");
        eprintln!("practice block written to {}", dir.display());
    }
}
