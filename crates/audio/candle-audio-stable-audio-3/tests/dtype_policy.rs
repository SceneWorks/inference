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
/// Measured on Metal at 30 s / 8 steps, the widest excursion is `side_ratio`: F16 reaches 0.951624
/// against an F32 envelope of [0.502560, 0.650129], i.e. 2.04 envelope widths above the top. `3.0`
/// clears that with margin.
///
/// This is deliberately loose, and the looseness is the finding rather than a concession: at three
/// seeds the fp16 spread genuinely is wider than the fp32 spread, because fp16 draws different
/// samples rather than perturbing the same one. A tighter bound here would be a bound on which dice
/// came up. What keeps the check honest is not its tightness but
/// [`the_envelope_check_rejects_the_degradations_it_is_named_for`], which constructs a level
/// collapse, a dulled high end and a narrowed image and asserts each is rejected at this exact
/// slack.
const ENVELOPE_SLACK: f64 = 3.0;

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

/// The envelope machinery's own controls, without weights.
///
/// A gate that cannot fail is not a gate. These construct the three degradations half precision
/// actually produces in a decoder and assert the envelope check rejects each one, plus a passing
/// control that is merely a different draw from the same distribution.
#[test]
fn the_envelope_check_rejects_the_degradations_it_is_named_for() {
    let reference = [
        Character {
            rms: 0.100,
            peak: 0.90,
            high_frequency_emphasis: 0.30,
            side_ratio: 0.50,
        },
        Character {
            rms: 0.110,
            peak: 0.95,
            high_frequency_emphasis: 0.33,
            side_ratio: 0.60,
        },
    ];
    // Inside the envelope: a different draw with the same character.
    assert_within_envelope(
        "control/passing",
        &reference,
        &[Character {
            rms: 0.105,
            peak: 0.92,
            high_frequency_emphasis: 0.31,
            side_ratio: 0.55,
        }],
    );

    let degradations = [
        (
            "level collapse",
            Character {
                rms: 0.050,
                peak: 0.90,
                high_frequency_emphasis: 0.30,
                side_ratio: 0.50,
            },
        ),
        (
            "dulled high end",
            Character {
                rms: 0.100,
                peak: 0.90,
                high_frequency_emphasis: 0.10,
                side_ratio: 0.50,
            },
        ),
        (
            "narrowed image",
            Character {
                rms: 0.100,
                peak: 0.90,
                high_frequency_emphasis: 0.30,
                side_ratio: 0.05,
            },
        ),
    ];
    for (label, degraded) in degradations {
        let rejected = std::panic::catch_unwind(|| {
            assert_within_envelope("control/degraded", &reference, &[degraded]);
        })
        .is_err();
        assert!(
            rejected,
            "the envelope check must reject a {label}, otherwise it gates nothing"
        );
    }
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
