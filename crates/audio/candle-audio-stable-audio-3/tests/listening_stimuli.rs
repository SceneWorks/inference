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
/// **every shipped `demo_cond` prompt of every SA3 variant** as a side-ratio calibration constant
/// (`SFX_SWEEP_PROMPTS`, `MUSIC_SWEEP_PROMPTS`, `MEDIUM_SWEEP_PROMPTS`,
/// `MUSIC_BASE_SWEEP_PROMPTS`). Those sweeps are what the shipped floors were *tuned on*. There is
/// therefore no held-out prompt anywhere in the `demo_cond` pool, and a set drawn entirely from it
/// would score the panel on the crate's own tuning data.
///
/// So the four held-out entries are authored for this panel, in the same idiom and domains, and
/// appear nowhere else in the repository.
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

/// The level-matching control, without weights: an equal-**RMS** pair is *not* an equal-loudness
/// pair, and the LUFS match closes the gap RMS matching leaves open.
///
/// This is the only non-`#[ignore]`d case in this target, and it exists because the protocol's
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
