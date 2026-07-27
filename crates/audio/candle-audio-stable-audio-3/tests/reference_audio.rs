//! Audio→audio restyle gates for every registered Stable Audio 3 checkpoint (sc-14547).
//!
//! # What this target exists to catch
//!
//! Two same-named `strength` parameters with **opposite** meanings meet on this seam:
//!
//! * the contract's `Conditioning::ReferenceAudio.strength`, which this workspace defines as
//!   **retention** (it mirrors the mflux-derived img2img strength, where a higher value starts the
//!   loop later and therefore preserves *more* of the source);
//! * Stable Audio 3's sampler `strength`, which is upstream's `init_noise_level` — `1.0` is pure
//!   noise and `0.0` returns the source untouched.
//!
//! A silent inversion between them produces a feature that runs, emits plausible audio, and does the
//! opposite of what the caller asked. Nothing about the output's *shape* or *quality* would reveal
//! it. The inversion has **two** distinct shapes, and each needs its own weight-free gate:
//!
//! 1. the *conversion* is wrong — `reference_noise_level` returns something other than the
//!    complement. Gated by `contract_strength_is_retention_and_a_flipped_sign_fails_here`, which
//!    drives the shipped conversion into the shipped schedule builder and the shipped init mix and
//!    asserts contract `1.0` returns the prepared source bit-for-bit while contract `0.0` returns
//!    the sampler noise bit-for-bit;
//! 2. the conversion is right but the *wrong field is handed to the pipeline* — the provider builds
//!    its `ReferenceAudio` with `strength` where `noise_level` belongs. This is the mistake an
//!    adversarial review actually landed, and (1) is blind to it: the conversion is still correct,
//!    it is simply not the value that travels. Gated by
//!    `the_request_surface_hands_the_pipeline_the_converted_noise_level`, which builds a real
//!    `GenerationRequest`, calls the shipped `resolve_reference_audio` and the shipped
//!    `reference_audio_for`, and compares the whole constructed struct.
//!
//! Past `reference_audio_for` the value goes into `Pipeline::synthesize_with_reference`, which needs
//! weights; that span is covered by
//! `real_reference_restyle_is_bounded_and_ordered_on_all_six_variants`, which requires the measured
//! source correlation at contract `1.0` to exceed the one at contract `0.0` by a wide margin.
//!
//! The remaining weight-free cases pin the preprocessing contract (resample, target sizing, channel
//! conformance), the geometry seam that decides sizing from the *requested* duration rather than the
//! source's extent, and the typed rejections.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_stable_audio_3::gen_core::{
    self, AudioEditMode, AudioParams, AudioTrack, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::pipeline::{prepare_reference_pcm, ReferenceAudio, SAMPLE_RATE};
use candle_audio_stable_audio_3::sampler::{
    adapt_sample_size_for_max, build_schedule, initialized_start, DistributionShift, Schedule,
};
use candle_audio_stable_audio_3::{
    descriptor_for, load_variant, reference_audio_for, reference_noise_level,
    resolve_reference_audio, synthesis_parameters, Variant, DEFAULT_REFERENCE_STRENGTH,
};

const CHANNELS: usize = 2;
/// The `DEFAULT_DURATION_PADDING` the sampler applies on top of the requested duration.
const DURATION_PADDING_SECS: f64 = 6.0;

// ---------------------------------------------------------------------------------------------
// Weight-free gates
// ---------------------------------------------------------------------------------------------

fn track(samples: Vec<f32>, sample_rate: u32, channels: u16) -> AudioTrack {
    AudioTrack {
        samples,
        sample_rate,
        channels,
        stems: Vec::new(),
    }
}

/// A deterministic, structured source clip: a decaying 220 Hz saw with a slow stereo offset.
///
/// Structured rather than random on purpose — a correlation floor measured against white noise
/// would be indistinguishable from a correlation floor measured against silence.
fn source_clip(seconds: f32, rate: u32, channels: u16) -> AudioTrack {
    let frames = (seconds as f64 * rate as f64) as usize;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for frame in 0..frames {
        let t = frame as f64 / rate as f64;
        let saw = 2.0 * ((220.0 * t) % 1.0) - 1.0;
        let envelope = (1.0 - (t / seconds as f64)).max(0.0);
        for channel in 0..channels {
            let phase = 1.0 + 0.05 * channel as f64;
            let value = 0.4 * envelope * (2.0 * ((220.0 * phase * t) % 1.0) - 1.0) + 0.05 * saw;
            samples.push(value as f32);
        }
    }
    track(samples, rate, channels)
}

fn reference_request(
    prompt: &str,
    duration: f32,
    reference: Option<(AudioTrack, Option<f32>)>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        seed: Some(7),
        steps: Some(4),
        audio: Some(AudioParams {
            target_duration: Some(duration),
            sample_rate: Some(SAMPLE_RATE),
            ..Default::default()
        }),
        conditioning: reference
            .map(|(audio, strength)| vec![Conditioning::ReferenceAudio { audio, strength }])
            .unwrap_or_default(),
        ..Default::default()
    }
}

#[test]
fn all_six_descriptors_advertise_reference_audio_conditioning() {
    for variant in Variant::ALL {
        let descriptor = descriptor_for(variant);
        assert_eq!(
            descriptor.capabilities.conditioning,
            vec![ConditioningKind::ReferenceAudio],
            "{} must advertise exactly ReferenceAudio",
            descriptor.id
        );
        // The advertisement is what makes the generic floor stop rejecting the variant, so prove
        // the floor now admits a well-formed reference on every id.
        let request = reference_request(
            "a restyled version of this clip",
            5.0,
            Some((source_clip(1.0, SAMPLE_RATE, 2), Some(0.5))),
        );
        descriptor
            .capabilities
            .validate_request_audio(descriptor.id, &request)
            .unwrap_or_else(|error| {
                panic!("{} rejected a valid reference: {error}", descriptor.id)
            });
    }
}

fn shared_row(schedule: &Schedule) -> Vec<f32> {
    match schedule {
        Schedule::Shared(values) => values.clone(),
        Schedule::PerExample(rows) => rows[0].clone(),
    }
}

/// The sign gate, weight-free, at the contract endpoint.
///
/// Every value here flows through the shipped conversion (`reference_noise_level`), the shipped
/// schedule builder, and the shipped init mix. Flipping the mapping to `init_noise_level = strength`
/// swaps the two endpoint assertions, so this test cannot pass under the inversion.
#[test]
fn contract_strength_is_retention_and_a_flipped_sign_fails_here() {
    // The mapping itself is the complement, and is demonstrably not the identity. Asserting only
    // the endpoints would admit `|1 - 2s|`, which agrees at both ends and is wrong in between.
    assert_eq!(reference_noise_level(Some(0.0)), 1.0);
    assert_eq!(reference_noise_level(Some(1.0)), 0.0);
    assert_eq!(reference_noise_level(Some(0.25)), 0.75);
    assert_eq!(reference_noise_level(Some(0.75)), 0.25);
    assert_ne!(
        reference_noise_level(Some(0.25)),
        0.25,
        "the conversion must not be the identity"
    );
    assert_eq!(
        reference_noise_level(None),
        1.0 - DEFAULT_REFERENCE_STRENGTH,
        "an omitted strength resolves to the documented default before conversion"
    );

    let device = Device::Cpu;
    let init = Tensor::arange(0f32, 8.0, &device)
        .unwrap()
        .reshape((1, 2, 4))
        .unwrap();
    let noise = Tensor::full(9f32, (1, 2, 4), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let flat = |tensor: &Tensor| tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Contract strength 1.0 == full retention == init noise level 0.0: the prepared source is
    // returned bit-for-bit and the DiT never runs.
    let level = reference_noise_level(Some(1.0));
    let schedule = build_schedule(4, level, &DistributionShift::Identity, None, 4).unwrap();
    assert_eq!(shared_row(&schedule)[0], 0.0);
    let start = initialized_start(&noise, Some(&init), level, &schedule).unwrap();
    assert!(
        start.skip_model,
        "full retention must short-circuit the DiT entirely"
    );
    assert_eq!(flat(&start.latents), flat(&init));

    // Contract strength 0.0 == no retention == init noise level 1.0: pure generation.
    let level = reference_noise_level(Some(0.0));
    let schedule = build_schedule(4, level, &DistributionShift::Identity, None, 4).unwrap();
    assert_eq!(shared_row(&schedule)[0], 1.0);
    let start = initialized_start(&noise, Some(&init), level, &schedule).unwrap();
    assert!(!start.skip_model);
    assert_eq!(flat(&start.latents), flat(&noise));

    // And the interior is the documented mix, in the retention direction: raising the contract
    // strength must move the start point monotonically towards the source.
    let mut distance_to_source = Vec::new();
    for strength in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let level = reference_noise_level(Some(strength));
        let schedule = build_schedule(4, level, &DistributionShift::Identity, None, 4).unwrap();
        let start = initialized_start(&noise, Some(&init), level, &schedule).unwrap();
        let distance = flat(&start.latents)
            .iter()
            .zip(flat(&init))
            .map(|(a, b)| ((a - b) as f64).abs())
            .sum::<f64>();
        distance_to_source.push(distance);
    }
    assert!(
        distance_to_source
            .windows(2)
            .all(|pair| pair[1] < pair[0] - 1e-9),
        "distance to the source must fall strictly as retention rises, got {distance_to_source:?}"
    );
}

/// The other half of the sign gate: the converted value is the one that actually travels.
///
/// `contract_strength_is_retention_and_a_flipped_sign_fails_here` pins the *conversion*. It is
/// deliberately blind to the mistake this case exists for: building the pipeline's `ReferenceAudio`
/// out of the resolved **`strength`** instead of the resolved **`noise_level`**. Under that
/// substitution the conversion is still perfectly correct and simply never reaches the sampler —
/// the sampler receives retention where it expects an init noise level, i.e. exactly the inversion,
/// arrived at by a different route. An adversarial review landed precisely that mutation and the
/// whole weight-free suite stayed green, because no case here ever entered the request surface.
///
/// So this drives the shipped request path — a real `GenerationRequest` carrying
/// `Conditioning::ReferenceAudio`, the shipped `resolve_reference_audio`, and the shipped
/// `reference_audio_for` that `StableAudio3Generator::generate` itself calls — and compares the
/// **whole constructed struct**, so it is the field selection that fails and not a derived quantity.
///
/// Every strength here is chosen so `strength != noise_level`; at `0.5` the two coincide and the
/// substitution would be invisible.
#[test]
fn the_request_surface_hands_the_pipeline_the_converted_noise_level() {
    // (explicit strength, expected retention in force, expected init noise level)
    let expectations = [
        (Some(1.0f32), 1.0f32, 0.0f32),
        (Some(0.0), 0.0, 1.0),
        (Some(0.25), 0.25, 0.75),
        (Some(0.75), 0.75, 0.25),
        (None, DEFAULT_REFERENCE_STRENGTH, 0.9),
    ];
    for (strength, retention, level) in expectations {
        assert_ne!(
            retention, level,
            "a case where retention equals the noise level cannot discriminate the two fields"
        );
        let clip = source_clip(0.5, 48_000, 1);
        let request = reference_request("restyle this clip", 5.0, Some((clip.clone(), strength)));

        let resolved = resolve_reference_audio(&request).unwrap_or_else(|| {
            panic!("a request carrying ReferenceAudio must resolve ({strength:?})")
        });
        assert_eq!(
            resolved.strength, retention,
            "{strength:?}: the retention in force, explicit or defaulted"
        );
        assert_eq!(
            resolved.noise_level, level,
            "{strength:?}: the sampler-facing init noise level is the complement"
        );

        // The struct `generate` hands `Pipeline::synthesize_with_reference`, field for field. The
        // scalar fields are asserted individually first so a failure names the wrong value instead
        // of dumping the whole source buffer; the struct equality after them is what would catch a
        // *new* field being added unconverted.
        let handed =
            reference_audio_for(&request).expect("the same request builds a pipeline reference");
        assert_eq!(
            handed.noise_level, level,
            "{strength:?}: the pipeline must receive the converted init noise level {level}, not \
             the contract retention {retention}"
        );
        assert_eq!(handed.sample_rate, 48_000, "{strength:?}: source rate");
        assert_eq!(handed.channels, 1, "{strength:?}: source channels");
        assert!(
            handed.samples == clip.samples.as_slice(),
            "{strength:?}: the source PCM must be passed through untouched"
        );
        assert_eq!(
            handed,
            ReferenceAudio {
                samples: &clip.samples,
                sample_rate: 48_000,
                channels: 1,
                noise_level: level,
            }
        );
    }

    // A request with no reference resolves to nothing, so the text-to-audio path is untouched.
    let plain = reference_request("just generate", 5.0, None);
    assert!(resolve_reference_audio(&plain).is_none());
    assert!(reference_audio_for(&plain).is_none());
}

#[test]
fn prepared_reference_pcm_is_resampled_channel_conformed_and_target_sized() {
    // 48 kHz mono — the exact case the rest of this audio lane emits, and the 160:147 ratio the
    // decision to resample rather than reject was taken on.
    let source = source_clip(0.5, 48_000, 1);
    let target_frames = 44_100; // one second of the model's own timeline
    let prepared = prepare_reference_pcm(&source.samples, 48_000, 1, target_frames).unwrap();
    assert_eq!(prepared.len(), target_frames * CHANNELS);

    // Mono duplicates into both channels, across the whole target including the padded tail.
    //
    // Note what this does *not* prove. The spec says "conform channels after padding", and that is
    // what `prepare_reference_pcm` does, but the ordering is **not observable and is not gated
    // here**: the pad value is zero, and duplicating a zero commutes with padding with zeros (as
    // does keeping the first two of four zeros). Conform-then-pad produces byte-identical output,
    // which a reviewer confirmed empirically by rewriting the function and watching every case in
    // this file still pass. The spec bullet is satisfied by construction, not by a test, and it is
    // recorded that way rather than dressed up — inventing a contrived way to make the two orders
    // differ would gate an artefact, not the contract. What is asserted below is the channel
    // conformance *result*, which is real.
    for frame in 0..target_frames {
        assert_eq!(
            prepared[frame * CHANNELS],
            prepared[frame * CHANNELS + 1],
            "mono source must duplicate at frame {frame}"
        );
    }

    // 0.5 s at 48 kHz resamples to ~22050 frames at 44.1 kHz; everything past that is zero.
    let resampled_frames = 22_050;
    assert!(
        prepared[..resampled_frames * CHANNELS]
            .iter()
            .any(|value| value.abs() > 1e-4),
        "the resampled prefix must carry the source"
    );
    assert!(
        prepared[(resampled_frames + 16) * CHANNELS..]
            .iter()
            .all(|value| *value == 0.0),
        "the tail past the source must be right-zero-padded"
    );

    // Stereo passes through unchanged at the model's own rate.
    let stereo = source_clip(0.25, SAMPLE_RATE, 2);
    let frames = stereo.samples.len() / CHANNELS;
    let prepared = prepare_reference_pcm(&stereo.samples, SAMPLE_RATE, 2, frames).unwrap();
    assert_eq!(prepared, stereo.samples);

    // More than two channels keeps the first two and drops the rest.
    let quad: Vec<f32> = (0..16).map(|value| value as f32).collect();
    let prepared = prepare_reference_pcm(&quad, SAMPLE_RATE, 4, 4).unwrap();
    assert_eq!(prepared, vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0]);

    // A source longer than the target is trimmed from offset 0.
    let long = source_clip(2.0, SAMPLE_RATE, 2);
    let prepared = prepare_reference_pcm(&long.samples, SAMPLE_RATE, 2, 1_000).unwrap();
    assert_eq!(prepared.len(), 1_000 * CHANNELS);
    assert_eq!(prepared[..], long.samples[..1_000 * CHANNELS]);

    // Malformed clips are refused here too, not only at the request boundary.
    assert!(prepare_reference_pcm(&[], SAMPLE_RATE, 2, 8).is_err());
    assert!(prepare_reference_pcm(&[0.0, 1.0, 2.0], SAMPLE_RATE, 2, 8).is_err());
    assert!(prepare_reference_pcm(&[0.0, f32::NAN], SAMPLE_RATE, 2, 8).is_err());
    assert!(prepare_reference_pcm(&[0.0, 1.0], 0, 2, 8).is_err());
    assert!(prepare_reference_pcm(&[0.0, 1.0], SAMPLE_RATE, 0, 8).is_err());
}

/// The sizing geometry this path runs on comes from the **requested** duration, never from how long
/// the caller's clip is.
///
/// # What the previous version of this case got wrong
///
/// It asserted `prepare_reference_pcm(..., sample_size).len() == sample_size * CHANNELS` across
/// three source lengths. That cannot fail: `prepare_reference_pcm` returns exactly
/// `target_frames * CHANNELS` unconditionally, so sweeping the *source* length proved nothing about
/// source extent. The seam that could actually get this wrong is upstream of it — the resolution of
/// `SynthesisParameters::duration_secs`, which is the single number
/// `Pipeline::synthesize_with_reference` turns into the adapted sample size, the length
/// `prepare_reference_pcm` conforms to, and the attention mask. A source-extent leak *there* would
/// be completely invisible in `prepare_reference_pcm`'s own output. So that is what is asserted,
/// plus the geometry arithmetic behind it, two-sided.
#[test]
fn sizing_geometry_comes_from_the_requested_duration_not_the_source_extent() {
    // 1. The provider's duration resolution ignores the clip, however long it is.
    for source_seconds in [0.25f32, 10.0, 60.0] {
        for variant in Variant::ALL {
            let request = reference_request(
                "restyle this",
                10.0,
                Some((source_clip(source_seconds, SAMPLE_RATE, 2), Some(0.5))),
            );
            let parameters = synthesis_parameters(variant, &request);
            assert_eq!(
                parameters.duration_secs,
                10.0,
                "{}: a {source_seconds}s source must not move the requested duration",
                variant.model_id()
            );
        }
    }
    // And it is not simply pinned to a constant: the requested duration is what moves it, and an
    // absent one falls back to the variant's default rather than to anything about the source.
    let mut longer = reference_request(
        "restyle this",
        25.0,
        Some((source_clip(1.0, SAMPLE_RATE, 2), Some(0.5))),
    );
    assert_eq!(
        synthesis_parameters(Variant::SmallMusic, &longer).duration_secs,
        25.0
    );
    longer.audio.as_mut().unwrap().target_duration = None;
    assert_eq!(
        synthesis_parameters(Variant::SmallMusic, &longer).duration_secs,
        Variant::SmallMusic.default_duration_secs(),
        "an omitted target duration falls back to the variant default, not to the source"
    );

    // 2. The geometry that duration produces is exact, not merely "at least".
    //
    // Derived independently of `adapt_sample_size_for_max`: at 4096 samples per latent frame,
    // 10 s = 441_000 samples rounds up to 108 latent frames, and the 6 s headroom is
    // floor(6 * 44_100 / 4096) = 64 more. 108 + 64 = 172.
    let max_sample_size = 5_292_032; // the smalls' own ceiling
    let geometry =
        adapt_sample_size_for_max(max_sample_size, &[Some(10.0)], DURATION_PADDING_SECS).unwrap();
    assert_eq!(geometry.valid_lengths[0], 172);
    assert_eq!(geometry.effective_lengths.as_ref().unwrap()[0], 108);
    assert!(
        geometry.latent_length > geometry.valid_lengths[0],
        "the valid length must not be clamped by the adapted length, or the equality above is \
         measuring the clamp instead of the arithmetic"
    );

    // 3. Both terms are load-bearing, asserted by moving each one on its own.
    let doubled =
        adapt_sample_size_for_max(max_sample_size, &[Some(20.0)], DURATION_PADDING_SECS).unwrap();
    assert!(
        doubled.valid_lengths[0] > geometry.valid_lengths[0]
            && doubled.sample_size > geometry.sample_size,
        "a longer requested duration must grow the geometry"
    );
    let unpadded = adapt_sample_size_for_max(max_sample_size, &[Some(10.0)], 0.0).unwrap();
    assert_eq!(
        unpadded.valid_lengths[0], 108,
        "without headroom the valid length is the requested duration alone"
    );

    // 4. And the prepared buffer is sized by that geometry, at the model's own timeline.
    let source = source_clip(0.25, SAMPLE_RATE, 2);
    let prepared =
        prepare_reference_pcm(&source.samples, SAMPLE_RATE, 2, geometry.sample_size).unwrap();
    assert_eq!(prepared.len(), geometry.sample_size * CHANNELS);
}

#[test]
fn reference_validation_rejects_every_malformed_clip_on_every_variant() {
    for variant in Variant::ALL {
        let descriptor = descriptor_for(variant);
        let id = descriptor.id;
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/nonexistent/sa3")));
        // The generator cannot be constructed without a snapshot, so validation is exercised
        // through the same function the generator's `validate` delegates to.
        assert!(load_variant(variant, &spec).is_err());

        let valid = reference_request(
            "restyle this",
            5.0,
            Some((source_clip(1.0, 48_000, 1), Some(0.4))),
        );
        assert!(
            validate(variant, &valid).is_ok(),
            "{id} must accept a well-formed 48 kHz mono reference"
        );

        // Two references. Typed `Unsupported`, deliberately the same type as the `AudioEdit`
        // combination below: both are statements about what this family can do, as opposed to the
        // malformed-clip rejections further down, which are about the caller's data and are `Msg`.
        // The split is asserted by type on both sides so it cannot drift.
        let mut invalid = valid.clone();
        invalid.conditioning.push(Conditioning::ReferenceAudio {
            audio: source_clip(1.0, SAMPLE_RATE, 2),
            strength: None,
        });
        assert!(
            matches!(
                validate(variant, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ),
            "{id}: two references must be typed Unsupported, got {:?}",
            validate(variant, &invalid)
        );

        // Reference plus an audio edit.
        let mut invalid = valid.clone();
        invalid.conditioning.push(Conditioning::AudioEdit {
            audio: source_clip(1.0, SAMPLE_RATE, 2),
            mode: AudioEditMode::Cover,
            region: None,
            strength: None,
        });
        assert!(
            matches!(
                validate(variant, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ),
            "{id}: reference + audio edit must be typed Unsupported"
        );

        // Malformed PCM and metadata.
        for (name, audio) in [
            ("empty", track(Vec::new(), SAMPLE_RATE, 2)),
            (
                "non-finite",
                track(vec![0.1, f32::INFINITY], SAMPLE_RATE, 2),
            ),
            ("zero rate", track(vec![0.1, 0.2], 0, 2)),
            ("zero channels", track(vec![0.1, 0.2], SAMPLE_RATE, 0)),
            ("ragged", track(vec![0.1, 0.2, 0.3], SAMPLE_RATE, 2)),
        ] {
            let invalid = reference_request("restyle this", 5.0, Some((audio, Some(0.4))));
            assert!(
                matches!(validate(variant, &invalid), Err(gen_core::Error::Msg(_))),
                "{id}: a {name} reference clip must be rejected as Msg (bad caller data), got {:?}",
                validate(variant, &invalid)
            );
        }

        // Strength range. gen-core enforces finiteness only, so the range is this family's job.
        for strength in [-0.1f32, 1.1, 40.0] {
            let invalid = reference_request(
                "restyle this",
                5.0,
                Some((source_clip(1.0, SAMPLE_RATE, 2), Some(strength))),
            );
            assert!(
                validate(variant, &invalid).is_err(),
                "{id}: strength {strength} must be rejected"
            );
        }
        for strength in [0.0f32, 0.5, 1.0] {
            let valid = reference_request(
                "restyle this",
                5.0,
                Some((source_clip(1.0, SAMPLE_RATE, 2), Some(strength))),
            );
            assert!(
                validate(variant, &valid).is_ok(),
                "{id}: strength {strength} is inside the documented range"
            );
        }
    }
}

/// Validation without a snapshot: the descriptor floor plus this family's own reference gates.
fn validate(variant: Variant, request: &GenerationRequest) -> gen_core::Result<()> {
    candle_audio_stable_audio_3::model::validate_request_for(variant, request)
}

// ---------------------------------------------------------------------------------------------
// Real-weight gates
// ---------------------------------------------------------------------------------------------

struct Case {
    variant: Variant,
    env: &'static str,
    prompt: &'static str,
}

const CASES: &[Case] = &[
    Case {
        variant: Variant::SmallMusic,
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        prompt: "warm cinematic post-rock with bowed strings and restrained drums",
    },
    Case {
        variant: Variant::SmallSfx,
        env: "SA3_SMALL_SFX_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    },
    Case {
        variant: Variant::Medium,
        env: "SA3_MEDIUM_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
    },
    Case {
        variant: Variant::SmallMusicBase,
        env: "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        prompt: "A beautiful piano arpeggio grows into a grand cinematic climax",
    },
    Case {
        variant: Variant::SmallSfxBase,
        env: "SA3_SMALL_SFX_BASE_SNAPSHOT",
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
    },
    Case {
        variant: Variant::MediumBase,
        env: "SA3_MEDIUM_BASE_SNAPSHOT",
        prompt: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit",
    },
];

fn snapshot(env: &str) -> WeightsSource {
    WeightsSource::Dir(PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    ))
}

/// Zero-lag Pearson correlation of two equal-length interleaved buffers, mono-summed.
fn correlation(left: &[f32], right: &[f32]) -> f64 {
    let frames = left.len().min(right.len()) / CHANNELS;
    let mono = |buffer: &[f32]| -> Vec<f64> {
        (0..frames)
            .map(|frame| {
                (buffer[frame * CHANNELS] as f64 + buffer[frame * CHANNELS + 1] as f64) / 2.0
            })
            .collect()
    };
    let a = mono(left);
    let b = mono(right);
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut numerator = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(&b) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        numerator += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    numerator / (var_a.sqrt() * var_b.sqrt()).max(f64::MIN_POSITIVE)
}

fn audio(output: GenerationOutput) -> AudioTrack {
    match output {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio output, got {other:?}"),
    }
}

/// Both bounds, on every registered checkpoint, plus the sign gate through the full graph.
///
/// The comparison target is the **prepared** source — `prepare_reference_pcm`'s own output at the
/// requested duration — so the 48 kHz→44.1 kHz conversion is inside the measurement rather than an
/// unattributed delta on the side of it.
///
/// What is asserted, and why each is not redundant:
///
/// * contract `1.0` (full retention) must correlate with the prepared source far above contract
///   `0.0` (pure generation). Under a flipped sign these two swap, so this is the sign gate;
/// * contract `1.0` must still not be a bit copy — it is a SAME encode/decode round trip;
/// * contract `0.0` must sit near zero correlation, i.e. the reference genuinely stops mattering;
/// * the aggregate trend over the sweep must rise with retention. Deliberately an *aggregate*
///   comparison of the top half against the bottom half rather than a strict per-pair ordering:
///   each render is one stochastic draw, and requiring five stochastic draws to be perfectly
///   ordered would be a flake, not a gate.
///
/// # The floors are measured, not chosen
///
/// Source correlation on Metal, 5 s / 4 steps, seed 7, this exact 48 kHz mono source
/// (`cargo test --release --features metal`, M-series):
///
/// | id | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
/// |---|---|---|---|---|---|
/// | `small_music` | -0.003934 | 0.574467 | 0.909590 | 0.951326 | 0.966623 |
/// | `small_sfx` | 0.005339 | 0.718412 | 0.937722 | 0.952103 | 0.966623 |
/// | `medium` | 0.003340 | 0.903839 | 0.945623 | 0.962782 | 0.985503 |
/// | `small_music_base` | 0.006965 | 0.766585 | 0.942801 | 0.959396 | 0.966623 |
/// | `small_sfx_base` | 0.000483 | 0.649653 | 0.920254 | 0.956970 | 0.966623 |
/// | `medium_base` | -0.006359 | 0.905855 | 0.961553 | 0.977967 | 0.985503 |
///
/// Two things in that table are worth stating because they are load-bearing rather than incidental.
/// The `1.0` column is *identical* across all four SAME-S ids and across both SAME-L ids, and is
/// independent of the prompt: at full retention the DiT is skipped entirely, so what is measured is
/// a pure autoencoder round trip of the same prepared buffer. And the `0.0` column sits within
/// `0.007` of zero on every id, which is what makes the `0.2` divergence floor a floor rather than a
/// fitted threshold.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_reference_restyle_is_bounded_and_ordered_on_all_six_variants() {
    let seconds = 5.0f32;
    let frames = (seconds as f64 * SAMPLE_RATE as f64) as usize;
    let source = source_clip(seconds, 48_000, 1);
    let prepared = prepare_reference_pcm(&source.samples, 48_000, 1, frames).unwrap();

    for case in CASES {
        let spec = LoadSpec::new(snapshot(case.env));
        let generator = load_variant(case.variant, &spec).expect("load pinned snapshot");
        let id = generator.descriptor().id;

        let mut measured = Vec::new();
        for strength in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let mut request =
                reference_request(case.prompt, seconds, Some((source.clone(), Some(strength))));
            request.steps = Some(4);
            let output = audio(
                generator
                    .generate(&request, &mut |_| {})
                    .unwrap_or_else(|error| panic!("{id} @ strength {strength}: {error}")),
            );
            assert_eq!(output.sample_rate, SAMPLE_RATE);
            assert_eq!(output.channels as usize, CHANNELS);
            assert_eq!(
                output.samples.len(),
                frames * CHANNELS,
                "{id}: the requested duration, not the source extent, sets the output length"
            );
            assert!(
                output.samples.iter().all(|value| value.is_finite()),
                "{id} @ strength {strength} emitted non-finite PCM"
            );
            let value = correlation(&output.samples, &prepared);
            println!("{id} strength={strength} source_correlation={value:.6}");
            measured.push((strength, value));
        }

        let full = measured.last().unwrap().1;
        let none = measured.first().unwrap().1;
        assert!(
            full > 0.5,
            "{id}: full retention must keep measurable source structure, got {full:.6}"
        );
        assert!(
            full > none + 0.3,
            "{id}: retention must be the direction of `strength` — full {full:.6} vs none {none:.6}"
        );
        assert!(
            none.abs() < 0.2,
            "{id}: zero retention must be effectively independent of the source, got {none:.6}"
        );
        assert!(
            full < 0.999_9,
            "{id}: full retention is a SAME round trip, not a bit copy, got {full:.6}"
        );
        let high = measured[3..].iter().map(|(_, v)| *v).sum::<f64>() / 2.0;
        let low = measured[..2].iter().map(|(_, v)| *v).sum::<f64>() / 2.0;
        assert!(
            high > low,
            "{id}: aggregate source retention must rise with strength — high {high:.6} vs low {low:.6}"
        );
    }
}

/// The frozen draw order: the sampler's initial noise is the request stream's **first** draw, and
/// the source encode's draws come after it.
///
/// This is the only observation that separates the two orders. Encoding first would move every
/// subsequent draw, so a text-only request and a reference request at the same seed would no longer
/// share their initial latents — silently, and only visible as "the same seed sounds different once
/// you attach a clip".
///
/// # The discrimination is not uniform across the six, and that is the point of the last assertion
///
/// Measured here: **SAME-S consumes zero draws on encode**. Its `draws_after_source_encode` is
/// therefore `1`, identical to `draws_after_initial_noise`, and on those four checkpoints the
/// ordering assertion is *vacuous* — swapping the two operations would not move a single count. Only
/// medium's **SAME-L** encode draws, so only medium and medium-base can falsify the invariant.
///
/// Running all six and additionally requiring that *at least one* of them reports a drawing encode
/// is what keeps this from being a green test that proves nothing: if a future change made every
/// encode deterministic, the final assertion fails and says so, rather than the whole case quietly
/// degrading into a tautology.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_initial_sampler_noise_precedes_the_source_encode() {
    use candle_audio_stable_audio_3::pipeline::{
        ReferenceAudio, StableAudio3Pipeline, SynthesisParameters,
    };
    use candle_audio_stable_audio_3::weights::SnapshotLayout;
    use candle_audio_stable_audio_3::{resolve_device, DevicePolicy};

    let seconds = 3.0f32;
    let frames = (seconds as f64 * SAMPLE_RATE as f64) as usize;
    let source = source_clip(seconds, SAMPLE_RATE, 2);
    let device = resolve_device(DevicePolicy::Default).unwrap();
    let mut drawing_encoders = 0usize;

    for case in CASES {
        let root = match snapshot(case.env) {
            WeightsSource::Dir(path) => path,
            WeightsSource::File(path) => panic!("expected a directory, got {}", path.display()),
        };
        let layout = SnapshotLayout::from_dir(&root).unwrap();
        let pipeline =
            StableAudio3Pipeline::from_layout(&layout, case.variant.geometry(), &device).unwrap();
        let id = case.variant.model_id();
        let parameters = SynthesisParameters {
            duration_secs: seconds,
            steps: 2,
            sampler: case.variant.recommended_sampler(),
            guidance: Default::default(),
            seed: 11,
        };
        let (samples, order) = pipeline
            .synthesize_with_reference_traced(
                "restyle this clip",
                None,
                parameters,
                Some(ReferenceAudio {
                    samples: &source.samples,
                    sample_rate: SAMPLE_RATE,
                    channels: 2,
                    noise_level: reference_noise_level(Some(0.5)),
                }),
                &mut |_, _| {},
                &mut || {},
                &|| false,
            )
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(samples.len(), frames * CHANNELS);
        let order = order.expect("a reference render reports its draw order");
        assert_eq!(
            order.draws_after_initial_noise, 1,
            "{id}: the sampler's initial noise must be the request stream's first draw"
        );
        assert!(
            order.draws_after_source_encode >= order.draws_after_initial_noise,
            "{id}: the source encode cannot run before the initial draw"
        );
        if order.draws_after_source_encode > order.draws_after_initial_noise {
            drawing_encoders += 1;
        }
        println!(
            "{id}: draws after initial noise = {}, after source encode = {}",
            order.draws_after_initial_noise, order.draws_after_source_encode
        );
    }

    assert!(
        drawing_encoders > 0,
        "no checkpoint's encode consumed a draw, so the ordering assertion above discriminated \
         nothing on any of them — the SAME-L encoder's eval-time noise is what makes this case a \
         gate rather than a tautology"
    );
}
