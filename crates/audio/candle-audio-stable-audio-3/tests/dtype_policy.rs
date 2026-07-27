//! Compute-dtype policy: what it resolves to, and what half precision actually costs (`sc-14545`).
//!
//! Upstream runs `model_half=True` when a CUDA device is selected and forces fp32 everywhere else
//! (`stable_audio_3/model.py`). Before sc-14545 this crate ignored that and hard-coded
//! `DType::F32, DType::F32` at the one call site, on every backend. The story asked for the upstream
//! policy *and* for evidence that fp16 does not degrade output past a parity bound. This file is the
//! evidence, and the evidence is why the shipped policy is F32.
//!
//! # 1. Waveform agreement cannot carry the claim, and this file proves that with numbers
//!
//! The registered sampler is eight-step Pingpong: every step re-injects fresh noise and the
//! trajectory is chaotic in the rounding. Measured on Metal at 30 s / 8 steps, medium's F16 render
//! sits at **cosine 0.222** against its own F32 render *at the same seed*, while two F32 renders at
//! adjacent seeds sit at **cosine 0.005**. Half precision therefore does not perturb the sample, it
//! selects a different one — the same thing a seed change does. An MR-STFT or SNR bound against the
//! F32 render would have to be loosened until it admitted an unrelated take, and that is not a
//! statement about quality.
//!
//! # 2. What can be measured is distributional, and it leans the wrong way
//!
//! Three seeds at each dtype, same prompt, same duration, same step count:
//!
//! | statistic | F32 range | F16 range |
//! |---|---|---|
//! | rms | 0.057639 … 0.067437 | 0.069209 … 0.075448 |
//! | peak | 0.519312 … 0.619903 | 0.525879 … 0.603027 |
//! | hf emphasis | 0.121821 … 0.150911 | 0.095454 … 0.123614 |
//! | side ratio | 0.502560 … 0.650129 | 0.373487 … 0.951624 |
//!
//! F16 is louder on 3/3 seeds, duller on 3/3, and its stereo spread exceeds twice the F32 envelope.
//! Louder-and-duller is what a decoder losing precision looks like. It is **not conclusive** — a
//! different draw legitimately has a different brightness, and three seeds cannot separate "fp16
//! degrades" from "fp16 rolled different dice". But an ambiguous measurement is not a licence to
//! ship the change, especially on CUDA, which is the only backend the policy would apply to and the
//! one no hardware was available to measure on. So the seam ships resolving to F32 everywhere, this
//! file keeps the fp16 path executable so it cannot rot, and the split policy worth trying next
//! (half the DiT, keep the SAME autoencoder at F32) is filed with these numbers.
//!
//! # 3. This runs on Metal, not CUDA
//!
//! F16 is IEEE binary16 on both backends and the graph is identical, so a Metal F16-vs-F32
//! comparison exercises the arithmetic a CUDA half-cast would select. It does not cover a
//! CUDA-specific kernel difference — which is precisely why the numbers above are not treated as
//! sufficient to enable it there.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device};
use candle_audio_stable_audio_3::gen_core::WeightsSource;
use candle_audio_stable_audio_3::pipeline::{
    ComputeDTypes, StableAudio3Pipeline, SynthesisParameters,
};
use candle_audio_stable_audio_3::sampler::SamplerKind;
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_audio_stable_audio_3::Variant;

/// Prompt, duration and step count every measurement in this file uses.
///
/// 30 s / 8 steps is the configuration the shipped render gates operate at, so a bound derived here
/// is a bound at a configuration something else also enforces.
const PROMPT: &str = "Meditative lo-fi ambient piano jazz, soft acoustic drum kit";
const SECONDS: f32 = 30.0;
const STEPS: usize = 8;

/// Seeds rendered at both dtypes. Three is the minimum that yields an envelope rather than a point.
const SEEDS: &[u64] = &[42, 7, 2_026];

/// How far outside the F32 seed envelope an F16 statistic may fall, as a fraction of that envelope's
/// own width.
///
/// # What this value is bracketed by
///
/// It cannot go **below ≈ 2.044**. Measured on Metal at 30 s / 8 steps, the widest real excursion is
/// `side_ratio`, where F16 reaches 0.951624 against an F32 envelope of [0.502560, 0.650129] — 2.043
/// envelope widths above the top. A slack under that rejects a measurement this crate committed as
/// honest, which would make the gate a bound on which dice came up rather than on quality.
///
/// It cannot go **above ≈ 2.79** without the gate ceasing to reject a dulled high end, which is the
/// characteristic fp16 decoder failure. At 2.79 the `hf_emphasis` rejection point reaches a third of
/// the committed F32 minimum, and past it a decoder that lost two thirds of its high end passes.
///
/// `2.5` sits inside that window with room on both sides.
/// [`the_envelope_check_rejects_the_degradations_it_is_named_for`] asserts *both* edges against the
/// committed measurements, so this constant cannot be loosened or tightened out of the window
/// without a weight-free test failing. sc-14545 originally shipped `3.0`, which was outside it: at
/// `3.0` a halved level and a two-thirds-dulled high end were both admitted.
///
/// # What the gate actually rejects at 2.5
///
/// Applied to the committed F32 envelopes, a statistic is rejected below these fractions of that
/// envelope's own minimum:
///
/// | statistic | F32 minimum | rejection point | as a fraction |
/// |---|---:|---:|---:|
/// | rms | 0.057639 | 0.033144 | 57.5% |
/// | peak | 0.519312 | 0.267834 | 51.6% |
/// | hf emphasis | 0.121821 | 0.049096 | 40.3% |
/// | side ratio | 0.502560 | 0.133637 | 26.6% |
///
/// So this is a **gross-degradation** gate, not a fine parity bound: it catches a decoder that
/// halves the level, loses 60% of its high end, or narrows the image to a quarter. It does not catch
/// a 10% dulling. That is the honest reach of a three-seed envelope on a sampler that re-injects
/// noise at every step, and it is why the shipped policy rests on the measured table in this file's
/// module documentation rather than on this check alone.
const ENVELOPE_SLACK: f64 = 2.5;

/// The F32 seed envelope committed in this file's module documentation, measured on Metal at
/// 30 s / 8 steps over [`SEEDS`].
///
/// Two `Character`s carrying the per-statistic minima and maxima. They are not two real takes — no
/// single seed produced all four minima — but [`envelope`] reduces any set of takes to exactly these
/// per-statistic bounds, so as a *reference envelope* this pair is indistinguishable from the three
/// measured takes and is reproducible from the committed table.
const COMMITTED_F32_ENVELOPE: [Character; 2] = [
    Character {
        rms: 0.057639,
        peak: 0.519312,
        high_frequency_emphasis: 0.121821,
        side_ratio: 0.502560,
    },
    Character {
        rms: 0.067437,
        peak: 0.619903,
        high_frequency_emphasis: 0.150911,
        side_ratio: 0.650129,
    },
];

/// The F16 envelope committed in this file's module documentation, same run, same seeds.
///
/// This is the gate's *passing* control: a real half-precision measurement that the shipped slack
/// admits. Its `side_ratio` maximum is the binding observation behind [`ENVELOPE_SLACK`]'s lower
/// bracket.
const COMMITTED_F16_ENVELOPE: [Character; 2] = [
    Character {
        rms: 0.069209,
        peak: 0.525879,
        high_frequency_emphasis: 0.095454,
        side_ratio: 0.373487,
    },
    Character {
        rms: 0.075448,
        peak: 0.603027,
        high_frequency_emphasis: 0.123614,
        side_ratio: 0.951624,
    },
];

fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    )
}

fn device() -> Device {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).expect("Metal device")
    }
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    {
        Device::new_cuda(0).expect("CUDA device")
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Device::Cpu
    }
}

/// Every statistic compared across dtypes, all defined here so the gate has no hidden dependency.
#[derive(Debug, Clone, Copy)]
struct Character {
    /// Overall level.
    rms: f64,
    /// Peak level — catches clipping and collapse that RMS averages away.
    peak: f64,
    /// `rms(x[n] - x[n-1]) / rms(x)`. A monotone brightness proxy: it rises with spectral centroid
    /// and falls when a decoder loses its high end, which is the characteristic fp16 failure.
    high_frequency_emphasis: f64,
    /// `rms(side) / rms(mid)` — the stereo image, which a degraded decoder narrows.
    side_ratio: f64,
}

fn rms(values: &[f32]) -> f64 {
    (values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

fn character(interleaved: &[f32]) -> Character {
    let left = interleaved
        .chunks_exact(2)
        .map(|frame| frame[0])
        .collect::<Vec<_>>();
    let right = interleaved
        .chunks_exact(2)
        .map(|frame| frame[1])
        .collect::<Vec<_>>();
    let mid = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l + r) * 0.5)
        .collect::<Vec<_>>();
    let side = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l - r) * 0.5)
        .collect::<Vec<_>>();
    let difference = mid.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    let mid_rms = rms(&mid);
    Character {
        rms: rms(interleaved),
        peak: interleaved
            .iter()
            .fold(0.0f64, |max, s| max.max(s.abs() as f64)),
        high_frequency_emphasis: rms(&difference) / mid_rms.max(f64::MIN_POSITIVE),
        side_ratio: rms(&side) / mid_rms.max(f64::MIN_POSITIVE),
    }
}

impl Character {
    const NAMES: [&'static str; 4] = ["rms", "peak", "hf_emphasis", "side_ratio"];

    fn values(&self) -> [f64; 4] {
        [
            self.rms,
            self.peak,
            self.high_frequency_emphasis,
            self.side_ratio,
        ]
    }
}

fn parameters(seed: u64) -> SynthesisParameters {
    SynthesisParameters {
        duration_secs: SECONDS,
        steps: STEPS,
        sampler: SamplerKind::Pingpong,
        guidance: Default::default(),
        seed,
    }
}

/// Render every seed once through one graph loaded at `root`.
fn render_all(variant: Variant, env: &str, device: &Device, root: DType) -> Vec<Vec<f32>> {
    let layout = SnapshotLayout::from_weights(&WeightsSource::Dir(snapshot(env))).unwrap();
    let pipeline = StableAudio3Pipeline::from_layout_with_dtypes(
        &layout,
        variant.geometry(),
        device,
        ComputeDTypes {
            root,
            text: DType::F32,
        },
    )
    .unwrap_or_else(|error| panic!("load {} at root {root:?}: {error}", variant.model_id()));
    assert_eq!(pipeline.dtypes().root, root);
    SEEDS
        .iter()
        .map(|&seed| {
            pipeline
                .synthesize(
                    PROMPT,
                    None,
                    parameters(seed),
                    &mut |_, _| {},
                    &mut || {},
                    &|| false,
                )
                .unwrap_or_else(|error| panic!("synthesize at root {root:?}: {error}"))
        })
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&a, &b) in left.iter().zip(right) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

/// `[minimum, maximum]` of one statistic over a set of takes.
fn envelope(takes: &[Character], index: usize) -> (f64, f64) {
    takes
        .iter()
        .map(|take| take.values()[index])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        })
}

fn assert_within_envelope(label: &str, reference: &[Character], candidate: &[Character]) {
    for index in 0..Character::NAMES.len() {
        let (low, high) = envelope(reference, index);
        let width = (high - low).max(f64::MIN_POSITIVE);
        let slack = width * ENVELOPE_SLACK;
        let (candidate_low, candidate_high) = envelope(candidate, index);
        eprintln!(
            "{label} {:>12}: reference [{low:.6}, {high:.6}] candidate [{candidate_low:.6}, \
             {candidate_high:.6}] allowed [{:.6}, {:.6}]",
            Character::NAMES[index],
            low - slack,
            high + slack
        );
        assert!(
            candidate_low >= low - slack && candidate_high <= high + slack,
            "{label}: {} left the F32 seed envelope [{low}, {high}] widened by {slack}: \
             [{candidate_low}, {candidate_high}]",
            Character::NAMES[index]
        );
    }
}

/// The policy itself, without weights.
///
/// Every backend resolves to F32, on the measured evidence in this file's module documentation. If
/// that ever changes, this assertion is the one that has to change with it — deliberately, rather
/// than as a side effect.
#[test]
fn compute_policy_resolves_to_full_precision_on_every_backend() {
    let expected = ComputeDTypes {
        root: DType::F32,
        text: DType::F32,
    };
    assert_eq!(ComputeDTypes::for_device(&Device::Cpu), expected);
    #[cfg(feature = "metal")]
    assert_eq!(
        ComputeDTypes::for_device(&Device::new_metal(0).expect("Metal device")),
        expected
    );
    #[cfg(feature = "cuda")]
    assert_eq!(
        ComputeDTypes::for_device(&Device::new_cuda(0).expect("CUDA device")),
        expected,
        "upstream half-casts on CUDA; sc-14545 measured that and did not adopt it"
    );
}

/// The envelope machinery's own controls, driven by the committed measurements.
///
/// A gate that cannot fail is not a gate — and a control built from invented numbers proves nothing
/// about the *shipped* gate, because the gate's reach depends entirely on how wide the real
/// reference envelope is relative to its own level. So every number here comes from
/// [`COMMITTED_F32_ENVELOPE`] / [`COMMITTED_F16_ENVELOPE`], which are the measurements in this
/// file's module documentation.
///
/// Four things are asserted, all at the shipped [`ENVELOPE_SLACK`]:
///
/// 1. The measured F16 envelope is **admitted**. That is the passing control, and it is a real
///    half-precision render rather than a synthetic "different draw".
/// 2. Each of the three degradations half precision produces in a decoder — a halved level, a
///    high end cut to a third, an image narrowed to a tenth — is **rejected**.
/// 3. A milder version of each of those three is **admitted**. This is what pins the constant: it
///    fails if the slack is tightened past the point where the gate would start rejecting honest
///    variation.
/// 4. The rejection points documented on [`ENVELOPE_SLACK`] are the ones the gate actually has.
///
/// Together (2) and (3) bracket `ENVELOPE_SLACK` into roughly `[2.09, 2.79)`, and the F16 control
/// independently requires it to be at least `2.05`. The `3.0` this file originally shipped is
/// outside that window, and this test fails at `3.0`.
#[test]
fn the_envelope_check_rejects_the_degradations_it_is_named_for() {
    // (1) The real F16 measurement is what the gate must admit.
    assert_within_envelope(
        "control/measured F16",
        &COMMITTED_F32_ENVELOPE,
        &COMMITTED_F16_ENVELOPE,
    );

    // Every degradation starts from the committed F32 *minimum* of each statistic and scales one of
    // them down. Starting at the minimum is the adversarial choice: it is the value closest to the
    // gate's lower edge, so it is the easiest place for a degradation to slip through.
    let baseline: [f64; 4] =
        std::array::from_fn(|index| envelope(&COMMITTED_F32_ENVELOPE, index).0);

    for (index, label, rejected_fraction, admitted_fraction) in [
        (0usize, "level collapse", 0.50, 0.65),
        (2, "dulled high end", 1.0 / 3.0, 0.50),
        (3, "narrowed image", 0.10, 0.40),
    ] {
        // (4) The documented rejection point, recomputed from the committed envelope.
        let (low, high) = envelope(&COMMITTED_F32_ENVELOPE, index);
        let rejection_point = low - (high - low) * ENVELOPE_SLACK;
        eprintln!(
            "envelope gate {:>12}: F32 min {low:.6}, rejects below {rejection_point:.6} ({:.1}% of \
             the minimum)",
            Character::NAMES[index],
            100.0 * rejection_point / low
        );

        // (2) The named degradation must be rejected.
        assert!(
            rejects(baseline, index, low * rejected_fraction),
            "the envelope check must reject a {label} ({:.0}% of the F32 minimum, {:.6}); its \
             rejection point is {rejection_point:.6}, so at this slack it gates nothing",
            100.0 * rejected_fraction,
            low * rejected_fraction
        );

        // (3) A milder version of the same degradation must be admitted, or the slack is so tight
        // that the gate would flag honest seed-to-seed variation.
        assert!(
            !rejects(baseline, index, low * admitted_fraction),
            "the envelope check must ADMIT {:.0}% of the F32 minimum for {}; rejecting it would \
             make the gate tighter than the seed spread it is measured against",
            100.0 * admitted_fraction,
            Character::NAMES[index]
        );
    }

    // (4) again, as a committed table rather than a log line, so the doc on `ENVELOPE_SLACK` cannot
    // drift away from the constant.
    let documented = [0.57503, 0.51575, 0.40302, 0.26591];
    for (index, expected) in documented.into_iter().enumerate() {
        let (low, high) = envelope(&COMMITTED_F32_ENVELOPE, index);
        let fraction = (low - (high - low) * ENVELOPE_SLACK) / low;
        assert!(
            (fraction - expected).abs() < 5e-4,
            "{} rejects below {:.5} of its F32 minimum, but ENVELOPE_SLACK's doc says {expected:.5}",
            Character::NAMES[index],
            fraction
        );
    }
}

/// Does the envelope check reject a candidate that is the committed F32 minima with statistic
/// `index` replaced by `value`?
fn rejects(baseline: [f64; 4], index: usize, value: f64) -> bool {
    let mut values = baseline;
    values[index] = value;
    let candidate = Character {
        rms: values[0],
        peak: values[1],
        high_frequency_emphasis: values[2],
        side_ratio: values[3],
    };
    std::panic::catch_unwind(|| {
        assert_within_envelope("control/candidate", &COMMITTED_F32_ENVELOPE, &[candidate]);
    })
    .is_err()
}

/// Render the same requests at F32 and at F16 on one device and measure what actually changes.
///
/// This is the measurement behind the shipped policy, kept executable for two reasons. It keeps the
/// fp16 path alive — the dtype boundary threaded through `dit.rs`, `sampler.rs` and the guidance
/// math has no production caller while the policy resolves to F32, so without this test it would rot
/// silently and the follow-up story would start from scratch. And it re-derives the numbers the
/// policy rests on, so a future change that makes fp16 viable is a change to committed measurements
/// rather than to an argument.
///
/// Three things are asserted:
///
/// 1. The fp16 graph still produces structurally valid audio — exact framing, finite, clamped,
///    non-silent. A dtype-threading regression fails here first.
/// 2. Half precision selects a *different sample* rather than perturbing one. If that ever stops
///    being true, the envelope check below is the wrong instrument and a direct parity bound should
///    replace it; the assertion says so.
/// 3. Every fp16 statistic lands inside the fp32 seed envelope widened by [`ENVELOPE_SLACK`].
#[test]
#[ignore = "real 10.4 GB weights; set SA3_MEDIUM_SNAPSHOT"]
fn half_precision_draws_from_the_same_distribution_as_full_precision() {
    let device = device();
    let full = render_all(Variant::Medium, "SA3_MEDIUM_SNAPSHOT", &device, DType::F32);
    let half = render_all(Variant::Medium, "SA3_MEDIUM_SNAPSHOT", &device, DType::F16);

    let expected_len = (SECONDS as f64 * 44_100.0).floor() as usize * 2;
    for (index, take) in half.iter().enumerate() {
        assert_eq!(
            take.len(),
            expected_len,
            "half-precision seed {} lost exact framing",
            SEEDS[index]
        );
        assert!(
            take.iter()
                .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)),
            "half-precision seed {} produced non-finite or unclamped PCM",
            SEEDS[index]
        );
    }

    let same_seed_cosine = cosine(&full[0], &half[0]);
    let seed_change_cosine = cosine(&full[0], &full[1]);
    eprintln!(
        "medium @ {SECONDS}s/{STEPS} steps: F16-vs-F32 same-seed cosine={same_seed_cosine:.9}, \
         F32-vs-F32 seed-change cosine={seed_change_cosine:.9}"
    );
    assert!(
        same_seed_cosine.abs() < 0.9,
        "F16 tracked F32 waveform-for-waveform (cosine {same_seed_cosine}); if that ever becomes \
         true this file's premise — that half precision selects a different sample — is wrong and \
         a direct parity bound should replace the envelope check"
    );

    let full_character = full.iter().map(|take| character(take)).collect::<Vec<_>>();
    let half_character = half.iter().map(|take| character(take)).collect::<Vec<_>>();
    for (seeds, label, takes) in [
        (SEEDS, "F32", &full_character),
        (SEEDS, "F16", &half_character),
    ] {
        for (seed, take) in seeds.iter().zip(takes) {
            eprintln!(
                "medium {label} seed={seed:<6} rms={:.9} peak={:.9} hf_emphasis={:.9} \
                 side_ratio={:.9}",
                take.rms, take.peak, take.high_frequency_emphasis, take.side_ratio
            );
        }
    }
    assert_within_envelope("medium F16 vs F32", &full_character, &half_character);
}
