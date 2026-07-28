//! Inpaint / repaint / extend gates for every registered Stable Audio 3 checkpoint (sc-14548).
//!
//! # What this target exists to catch
//!
//! The DiT half of this feature was already wired before the story started: `DitInputs` takes a
//! `[batch, 257, time]` local conditioning ordered `[inpaint_mask, inpaint_masked_input]`, every
//! block carries its `LocalConditioning` MLP, and the batch-2 CFG forward already repeats the tensor
//! on both branches. What was missing was the **producer** — the pipeline handed the DiT a zero
//! tensor. So everything gated here is on the path from a `GenerationRequest` to that tensor, plus
//! the PCM stitch that follows the decode.
//!
//! Almost every way of getting this wrong is silent. The output is the right length, the right
//! sample rate, finite, and audibly plausible under all of them:
//!
//! * **mask polarity** — a `1 = edit` mask hands the model the region it should keep and blanks the
//!   surroundings, i.e. it edits the complement of what was asked;
//! * **channel order** — `[masked_input, mask]` has the identical shape and the model still runs;
//! * **mask built at the un-adapted size** — the mask is constructed over the *adapted* sample size
//!   (which carries the 6 s headroom and the 8192 alignment) and only then resized to latent
//!   resolution. Building it over `duration * 44_100` and resizing compresses the whole timeline and
//!   moves the edit window, while every "the outside is preserved" assertion still passes, because
//!   the stitch is a separate mechanism;
//! * **mask built pre-resample** — applying the caller's region seconds against the caller's own
//!   sample rate and then resampling the audio underneath the mask shifts the window by the rate
//!   ratio. A 48 kHz source is the case the rest of this audio lane actually emits;
//! * **rounding vs. derived latent indices** — the latent window is `[ceil(start/4096),
//!   ceil(end/4096))` *because* the audio-resolution mask is nearest-resized, and nearest resizing
//!   takes `src[t*4096]`. Computing latent indices directly by rounding the seconds agrees at some
//!   boundaries and is off by one frame at others;
//! * **ones in the padding** — upstream zeroes the local conditioner past `seconds_total`, and the
//!   model was trained that way; leaving ones there is an out-of-distribution input;
//! * **unmasked source latents** — showing the DiT the source inside the edit window is showing it
//!   the answer;
//! * **a missing local tensor on the negative CFG branch** — guidance varies the prompt and only the
//!   prompt;
//! * **an ignored `strength`** — gen-core carries `AudioEdit.strength` as a first-class float and
//!   this family has nothing for it to modulate, so accepting and discarding it is the
//!   "appears to work, does nothing" failure;
//! * **a silent first-`AudioEdit` win** — `GenerationRequest::audio_edit()` is first-match-only, so
//!   a second region would be dropped without a word.
//!
//! Each of those has a case below, and each case was verified to fail under its own mutation rather
//! than merely to pass under the correct code; `docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md`
//! carries the table.
//!
//! # What is *not* claimed
//!
//! There is **no frozen-PyTorch inpaint oracle** on this machine, and none is vendored in the
//! repository: upstream lives in an external checkout per
//! `docs/migration/SC_14534_SA3_REFERENCE_PARITY.md`, which is not present. So nothing here is a
//! cross-framework parity claim. What the real-weight cases below assert is *internal consistency*
//! against the shipped preprocessing (exact outside-region preservation, material interior change,
//! alias byte-equality, seam continuity), and what the weight-free cases assert is that the mask and
//! local conditioning are the ones this crate's own configs and the landed DiT declare.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_stable_audio_3::dit::batch_cfg_local_conditioning;
use candle_audio_stable_audio_3::gen_core::{
    self, AudioEditMode, AudioParams, AudioTrack, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, TimeRegion, WeightsSource,
};
use candle_audio_stable_audio_3::pipeline::{
    conditioning_is_forwarded, edit_geometry, edit_geometry_matches_request, edit_keep_mask,
    edit_local_conditioning, edit_local_conditioning_is_present, edit_region_latents,
    edit_region_samples, edit_retained_latent_count, prepare_reference_pcm, resampled_frame_count,
    stitch_outside_region, tensor_has_nonzero, AudioEdit, EditGeometry, SAMPLE_RATE,
};
use candle_audio_stable_audio_3::sampler::{
    adapt_sample_size_for_max, SampleGeometry, DEFAULT_DURATION_PADDING, LATENT_DOWNSAMPLING,
};
use candle_audio_stable_audio_3::{
    audio_edit_for, load_variant, resolve_audio_edit, synthesis_parameters, Variant,
};

const CHANNELS: usize = 2;
const LATENT_CHANNELS: usize = 256;
/// The two small post-trained checkpoints' own ceiling, spelled out so the pinned geometry below is
/// derived from a number this file states rather than from the code under test.
const SMALL_SAMPLE_SIZE: usize = 5_292_032;

// ---------------------------------------------------------------------------------------------
// Fixtures
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
/// Structured rather than random so a correlation or continuity measurement against it means
/// something; white noise would make a preservation floor indistinguishable from a silence floor.
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

/// A second deterministic clip, unmistakably different material at the same length and rate.
///
/// Exists for the source-sensitivity half of the real-weight inpaint case: two edits that differ
/// **only** in their source must not render the same interior. A rising 147 Hz triangle against
/// `source_clip`'s decaying 220 Hz saw differs in fundamental, timbre and envelope direction, so the
/// two encode to genuinely different latents rather than to a scaled version of the same ones.
fn alternate_source_clip(seconds: f32, rate: u32, channels: u16) -> AudioTrack {
    let frames = (seconds as f64 * rate as f64) as usize;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for frame in 0..frames {
        let t = frame as f64 / rate as f64;
        let envelope = (t / seconds as f64).min(1.0);
        for channel in 0..channels {
            let phase = 1.0 + 0.11 * channel as f64;
            let triangle = 4.0 * ((147.0 * phase * t) % 1.0 - 0.5).abs() - 1.0;
            samples.push((0.45 * envelope * triangle) as f32);
        }
    }
    track(samples, rate, channels)
}

fn edit_request(
    prompt: &str,
    audio: AudioTrack,
    mode: AudioEditMode,
    region: Option<TimeRegion>,
    target_duration: Option<f32>,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        seed: Some(7),
        steps: Some(4),
        audio: Some(AudioParams {
            target_duration,
            sample_rate: Some(SAMPLE_RATE),
            ..Default::default()
        }),
        conditioning: vec![Conditioning::AudioEdit {
            audio,
            mode,
            region,
            strength: None,
        }],
        ..Default::default()
    }
}

fn region(start: f32, end: Option<f32>) -> Option<TimeRegion> {
    Some(TimeRegion {
        start_secs: start,
        end_secs: end,
    })
}

/// Validation without a snapshot: the descriptor floor plus this family's own edit gates.
fn validate(variant: Variant, request: &GenerationRequest) -> gen_core::Result<()> {
    candle_audio_stable_audio_3::model::validate_request_for(variant, request)
}

fn small_geometry(duration: f32) -> SampleGeometry {
    adapt_sample_size_for_max(
        SMALL_SAMPLE_SIZE,
        &[Some(duration as f64)],
        DEFAULT_DURATION_PADDING,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------------------------
// Weight-free gates
// ---------------------------------------------------------------------------------------------

/// The advertised surface: three modes on all six ids, and `Cover` deliberately absent.
///
/// The negative half is what makes this a gate rather than a restatement of the descriptor. Stable
/// Audio 3's six configs pin their **complete** conditioner surface — `global = [seconds_total]`,
/// `local = [inpaint_mask, inpaint_masked_input]` — so there is no cover conditioner to map
/// `AudioEditMode::Cover` onto, and gen-core's own doc for that mode names ACE-Step. Whole-clip
/// restyle is not lost: it is `Conditioning::ReferenceAudio` (sc-14547), advertised alongside, with
/// the retention `strength` this surface does not have. So `Cover` must come back typed
/// `Unsupported` from the generic allowlist, on every id.
#[test]
fn all_six_descriptors_advertise_three_edit_modes_and_refuse_cover() {
    for variant in Variant::ALL {
        let descriptor = candle_audio_stable_audio_3::descriptor_for(variant);
        let id = descriptor.id;
        assert_eq!(
            descriptor.capabilities.audio_edit_modes,
            vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend
            ],
            "{id}: exactly the three modes this family implements, in a stable order"
        );
        assert!(
            descriptor
                .capabilities
                .conditioning
                .contains(&ConditioningKind::AudioEdit),
            "{id}: advertising the kind is what makes the generic floor stop rejecting it"
        );
        // Whole-clip restyle still exists, by the name the `AudioEdit` docs point callers at.
        assert!(
            descriptor
                .capabilities
                .conditioning
                .contains(&ConditioningKind::ReferenceAudio),
            "{id}: dropping Cover costs no capability only because ReferenceAudio is advertised"
        );

        let source = source_clip(4.0, SAMPLE_RATE, 2);
        for mode in [
            AudioEditMode::Inpaint,
            AudioEditMode::Repaint,
            AudioEditMode::Extend,
        ] {
            let (span, duration) = match mode {
                AudioEditMode::Extend => (region(4.0, Some(7.0)), Some(7.0)),
                _ => (region(1.0, Some(2.0)), Some(4.0)),
            };
            let request = edit_request("fill this in", source.clone(), mode, span, duration);
            validate(variant, &request)
                .unwrap_or_else(|error| panic!("{id}: {mode:?} must be accepted: {error}"));
        }

        let cover = edit_request(
            "restyle the whole thing",
            source.clone(),
            AudioEditMode::Cover,
            None,
            Some(4.0),
        );
        assert!(
            matches!(
                validate(variant, &cover),
                Err(gen_core::Error::Unsupported(_))
            ),
            "{id}: Cover must be typed Unsupported, got {:?}",
            validate(variant, &cover)
        );
    }
}

/// The pre-resample length arithmetic must agree with the resampler it predicts.
///
/// `resampled_frame_count` reproduces `candle_audio::dsp::resample`'s own output-length rule,
/// because the geometry (adapted sample size, latent length, output length, and therefore the whole
/// mask) has to be resolved *before* the buffer is resampled. A reproduction is a copy, and a copy
/// can drift, so it is checked against the real thing rather than against a restatement of the
/// formula.
#[test]
fn resampled_frame_count_agrees_with_the_shared_resampler() {
    for rate in [8_000u32, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000] {
        for source_frames in [1usize, 7, 1_000, 44_100, 48_000, 130_913] {
            let samples = vec![0.25f32; source_frames * CHANNELS];
            let resampled = candle_audio_stable_audio_3::candle_audio::dsp::resample(
                &samples,
                rate,
                SAMPLE_RATE,
                2,
            )
            .unwrap();
            assert_eq!(
                resampled.len() / CHANNELS,
                resampled_frame_count(source_frames, rate),
                "rate {rate}, {source_frames} frames"
            );
        }
    }
    // And it is not the identity: a 48 kHz second is more than 44_100 frames before conversion and
    // exactly 44_100 after, which is the whole reason the region seconds must be applied post
    // resample.
    assert_eq!(resampled_frame_count(48_000, 48_000), 44_100);
    assert_eq!(resampled_frame_count(44_100, 44_100), 44_100);
}

/// Request → resolved region, the single site where a mode plus an `Option<TimeRegion>` becomes a
/// concrete `[start, end)` and an output length.
///
/// This is the carry-forward from sc-14547, applied before the same defect could recur: a value that
/// is computed correctly and never travels is invisible to a gate on the computation. So the whole
/// chain is driven here — a real `GenerationRequest`, the shipped `resolve_audio_edit`, the shipped
/// `audio_edit_for` that `StableAudio3Generator::generate` itself calls, and the shipped
/// `synthesis_parameters` that turns the result into the duration everything downstream is sized
/// from. Nothing on this path needs weights, and before the resolution was extracted out of
/// `generate`, all of it did.
#[test]
fn the_request_surface_hands_the_pipeline_the_resolved_region() {
    // 1. Inpaint: the output is exactly as long as the source, and the region is the caller's.
    let source = source_clip(6.0, SAMPLE_RATE, 2);
    let request = edit_request(
        "a cymbal swell here",
        source.clone(),
        AudioEditMode::Inpaint,
        region(2.0, Some(4.0)),
        None,
    );
    let resolved = resolve_audio_edit(&request).expect("an AudioEdit request resolves");
    assert_eq!(resolved.source_duration_secs, 6.0);
    assert_eq!(resolved.start_secs, 2.0);
    assert_eq!(resolved.end_secs, 4.0);
    assert_eq!(
        resolved.output_duration_secs, 6.0,
        "an inpaint's output is the source's own length"
    );
    let handed = audio_edit_for(&request).expect("the same request builds a pipeline edit");
    assert_eq!(
        handed,
        AudioEdit {
            samples: &source.samples,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            start_secs: 2.0,
            end_secs: 4.0,
        },
        "field for field, so a swapped or dropped endpoint fails here and not downstream"
    );
    for variant in Variant::ALL {
        assert_eq!(
            synthesis_parameters(variant, &request).duration_secs,
            6.0,
            "{}: the edit decides the output length",
            variant.model_id()
        );
    }

    // 2. `end_secs = None` means the end of the clip, not zero and not the variant default.
    let open = edit_request(
        "rework the tail",
        source.clone(),
        AudioEditMode::Repaint,
        region(2.0, None),
        None,
    );
    let handed = audio_edit_for(&open).expect("an open-ended region resolves");
    assert_eq!(handed.start_secs, 2.0);
    assert_eq!(
        handed.end_secs, 6.0,
        "an absent end resolves to the source end"
    );

    // 3. Extend: the region's end is the new total length, and it is what sizes the render — even
    //    with no `target_duration` at all, which is where the variant's 120 s / 380 s default would
    //    otherwise take over.
    let extend = edit_request(
        "continue the phrase",
        source.clone(),
        AudioEditMode::Extend,
        region(6.0, Some(10.0)),
        None,
    );
    let resolved = resolve_audio_edit(&extend).expect("an extend resolves");
    assert_eq!(resolved.output_duration_secs, 10.0);
    for variant in Variant::ALL {
        let duration = synthesis_parameters(variant, &extend).duration_secs;
        assert_eq!(
            duration,
            10.0,
            "{}: an extend with no target_duration must render its region's end, not the {}s \
             variant default",
            variant.model_id(),
            variant.default_duration_secs()
        );
        assert_ne!(
            duration,
            variant.default_duration_secs(),
            "{}: and the two must be genuinely different, or the assertion above proves nothing",
            variant.model_id()
        );
    }

    // 4. A 48 kHz source — what ACE-Step and MOSS-SFX emit one step earlier in this same product —
    //    resolves its *duration* on the post-resample timeline while its region seconds pass through
    //    untouched. Seconds are seconds on either timeline; frames are not, which is what the
    //    geometry case below pins.
    let off_rate = source_clip(6.0, 48_000, 1);
    let request = edit_request(
        "a cymbal swell here",
        off_rate.clone(),
        AudioEditMode::Inpaint,
        region(2.0, Some(4.0)),
        None,
    );
    let resolved = resolve_audio_edit(&request).expect("an off-rate source resolves");
    assert_eq!(
        resolved.source_duration_secs, 6.0,
        "6 s at 48 kHz is 6 s at 44.1 kHz, measured through the resampler's own length rule"
    );
    let handed = audio_edit_for(&request).expect("and builds a pipeline edit");
    assert_eq!(
        handed.sample_rate, 48_000,
        "the source rate is passed through"
    );
    assert_eq!(handed.start_secs, 2.0);
    assert_eq!(handed.end_secs, 4.0);

    // 5. A request with no edit resolves to nothing, so the text-to-audio and restyle paths are
    //    untouched and `synthesis_parameters` still reads `target_duration`.
    let plain = GenerationRequest {
        prompt: "just generate".into(),
        audio: Some(AudioParams {
            target_duration: Some(12.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(resolve_audio_edit(&plain).is_none());
    assert!(audio_edit_for(&plain).is_none());
    assert_eq!(
        synthesis_parameters(Variant::SmallMusic, &plain).duration_secs,
        12.0
    );
}

/// A resolved edit must actually reach the pipeline.
///
/// The sc-14547 carry-forward in its sharpest form. `StableAudio3Generator::generate` forwards the
/// output of `audio_edit_for` and `reference_audio_for`; substituting either for `None` is one token
/// and deletes the whole feature *silently* — the render comes back the right length, the right rate,
/// finite, and plausible, just with no source in it. `generate` needs multi-gigabyte weights, so the
/// PR lane cannot evaluate that forward.
///
/// What is gated here is the **rule** (`conditioning_is_forwarded`) plus the two resolutions that
/// feed it, which together mean a dropped forward is refused at runtime instead of degrading.
///
/// Round-2 review found that the rule alone left the *next* token unguarded: the guard read the
/// `edit` local, and the argument list of the call one line below it was a second, unchecked place
/// to write `None`. The resolved values now travel as a `pipeline::ForwardedConditioning` receipt
/// carrying the request booleans with them, and `synthesize_conditioned` re-checks the pair on the
/// far side of the boundary — which **moves** the seam rather than removing it. Nulling a *field*
/// on the receipt is refused there; the destructured `edit` that `synthesize_conditioned` forwards
/// to `synthesize_traced` one line later is still a one-token site, caught with weights by
/// `real_inpaint_…`'s bit-exact-outside assertion (row 7's catcher, recorded as row 13).
/// All of those call sites need weights; `docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md` carries
/// the per-site table with the lane that catches each, every row verified by running the mutation.
#[test]
fn a_resolved_edit_must_be_the_one_the_pipeline_is_handed() {
    // Agreement in all four honest combinations.
    assert!(conditioning_is_forwarded(false, false, false, false).is_ok());
    assert!(conditioning_is_forwarded(true, true, false, false).is_ok());
    assert!(conditioning_is_forwarded(false, false, true, true).is_ok());
    // Every disagreement fails closed, in both directions on both halves.
    assert!(conditioning_is_forwarded(false, false, true, false).is_err());
    assert!(conditioning_is_forwarded(false, false, false, true).is_err());
    assert!(conditioning_is_forwarded(true, false, false, false).is_err());
    assert!(conditioning_is_forwarded(false, true, false, false).is_err());
    // And the combination is refused a second time, here, independently of `validate`.
    assert!(conditioning_is_forwarded(true, true, true, true).is_err());

    // Driven from the request surface, so the booleans `generate` computes are the ones asserted.
    let source = source_clip(4.0, SAMPLE_RATE, 2);
    let edited = edit_request(
        "fill this in",
        source.clone(),
        AudioEditMode::Inpaint,
        region(1.0, Some(2.0)),
        Some(4.0),
    );
    let has_edit = |request: &GenerationRequest| {
        request
            .conditioning
            .iter()
            .any(|item| matches!(item, Conditioning::AudioEdit { .. }))
    };
    assert!(has_edit(&edited));
    assert!(conditioning_is_forwarded(
        false,
        false,
        has_edit(&edited),
        audio_edit_for(&edited).is_some()
    )
    .is_ok());
    // The deletion the guard exists for.
    assert!(conditioning_is_forwarded(
        false,
        false,
        has_edit(&edited),
        None::<AudioEdit>.is_some()
    )
    .is_err());

    let plain = GenerationRequest {
        prompt: "just generate".into(),
        ..Default::default()
    };
    assert!(!has_edit(&plain));
    assert!(conditioning_is_forwarded(
        false,
        false,
        has_edit(&plain),
        audio_edit_for(&plain).is_some()
    )
    .is_ok());
}

/// The pinned alignment examples, and the three ways of getting the window arithmetic wrong.
///
/// The two examples are stated on the story and reproduce exactly here. They are derived
/// independently in the comments rather than read out of the code under test, so this is arithmetic
/// against arithmetic and not a snapshot of whatever the function currently returns.
#[test]
fn the_edit_geometry_reproduces_the_pinned_alignment_examples() {
    // --- 10 s inpaint of [2, 7) -----------------------------------------------------------------
    //
    // adapted: (10 + 6) * 44_100 = 705_600 -> ceil to 4096 = 708_608 -> ceil to 8192 = 712_704.
    // latent:  712_704 / 4096 = 174.
    // edit:    int(2 * 44_100) = 88_200 -> ceil(88_200/4096) = 22;
    //          int(7 * 44_100) = 308_700 -> ceil(308_700/4096) = 76.
    // effective boundary: int(10 * 44_100) = 441_000 -> latent ceil = 108.
    let geometry = small_geometry(10.0);
    assert_eq!(geometry.sample_size, 712_704);
    assert_eq!(geometry.latent_length, 174);
    let source = source_clip(10.0, SAMPLE_RATE, 2);
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();
    assert_eq!(
        resolved,
        EditGeometry {
            adapted_size: 712_704,
            latent_length: 174,
            effective_samples: 441_000,
            start_sample: 88_200,
            end_sample: 308_700,
            start_latent: 22,
            end_latent: 76,
        }
    );
    // The mask spans the **adapted** size, not the requested duration's 441_000 samples. A mask
    // built at the un-adapted size and resized would map the same seconds onto different latents.
    assert_eq!(edit_keep_mask(&resolved).len(), 712_704);
    assert_ne!(
        resolved.adapted_size, resolved.effective_samples,
        "the two must differ, or 'built at the adapted size' is untestable here"
    );

    // --- 10 s extended to 18 s ------------------------------------------------------------------
    //
    // adapted: (18 + 6) * 44_100 = 1_058_400 -> ceil 4096 = 1_060_864 -> ceil 8192 = 1_064_960.
    // latent:  1_064_960 / 4096 = 260.
    // source boundary: int(10 * 44_100) = 441_000 -> ceil/4096 = 108.
    // effective boundary: int(18 * 44_100) = 793_800 -> ceil/4096 = 194.
    let geometry = small_geometry(18.0);
    assert_eq!(geometry.sample_size, 1_064_960);
    assert_eq!(geometry.latent_length, 260);
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 10.0,
        end_secs: 18.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 18.0).unwrap();
    assert_eq!(
        resolved,
        EditGeometry {
            adapted_size: 1_064_960,
            latent_length: 260,
            effective_samples: 793_800,
            start_sample: 441_000,
            end_sample: 793_800,
            start_latent: 108,
            end_latent: 194,
        }
    );

    // --- the latent window is *derived*, not rounded ---------------------------------------------
    //
    // The mask is built at audio resolution and nearest-resized, and nearest resizing takes
    // `src[t*4096]`, so latent `t` is zeroed exactly when `t*4096` lands in `[start, end)` — i.e.
    // on `[ceil(start/4096), ceil(end/4096))`. Rounding the seconds straight to latent indices
    // agrees at some boundaries and disagrees at others, which is what makes it a live mutation
    // rather than a style preference. 3 s = 132_300 samples = 32.30 latents: ceil is 33, round is
    // 32, floor is 32. 7 s = 308_700 = 75.37: ceil 76, round 75.
    let (start_sample, end_sample) = edit_region_samples(3.0, 7.0);
    assert_eq!((start_sample, end_sample), (132_300, 308_700));
    assert_eq!(edit_region_latents(start_sample, end_sample), (33, 76));
    let rounded = |samples: usize| (samples as f64 / LATENT_DOWNSAMPLING as f64).round() as usize;
    assert_ne!(rounded(start_sample), 33, "rounding the start differs here");
    assert_ne!(rounded(end_sample), 76, "and so does rounding the end");
    // Exactly on a latent boundary the three agree, which is why a single well-chosen region is not
    // enough evidence on its own.
    assert_eq!(edit_region_latents(4_096, 8_192), (1, 2));

    // --- the region is on the post-resample timeline ---------------------------------------------
    //
    // Same seconds, a 48 kHz mono source instead of a 44.1 kHz stereo one: the sample and latent
    // indices must be identical. Under the "apply the caller's seconds at the caller's own rate"
    // mutation the 48 kHz case lands at int(2 * 48_000) = 96_000 -> latent 24 instead of 22.
    let off_rate = source_clip(10.0, 48_000, 1);
    let native = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let converted = AudioEdit {
        samples: &off_rate.samples,
        sample_rate: 48_000,
        channels: 1,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let geometry = small_geometry(10.0);
    assert_eq!(
        edit_geometry(&converted, &geometry, 10.0).unwrap(),
        edit_geometry(&native, &geometry, 10.0).unwrap(),
        "a source's own sample rate must not move the edit window"
    );

    // --- a region that spans no latent frame is refused, not silently no-op'd -------------------
    let hairline = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        // 4096 samples is ~0.0929 s; this span sits strictly inside one latent frame.
        start_secs: 0.01,
        end_secs: 0.02,
    };
    assert_eq!(edit_region_latents(441, 882), (1, 1));
    assert!(
        edit_geometry(&hairline, &geometry, 10.0).is_err(),
        "a sub-latent-frame region masks nothing and must be refused"
    );
    // Inverted and empty regions too.
    for (start, end) in [(3.0f32, 2.0f32), (2.0, 2.0), (-1.0, 2.0)] {
        let bad = AudioEdit {
            samples: &source.samples,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            start_secs: start,
            end_secs: end,
        };
        assert!(
            edit_geometry(&bad, &geometry, 10.0).is_err(),
            "[{start},{end})"
        );
    }
}

/// The keep mask's polarity, its two zeroed spans, and the latent window they produce.
#[test]
fn the_keep_mask_zeroes_the_region_and_the_padding_and_nothing_else() {
    let geometry = small_geometry(10.0);
    let source = source_clip(10.0, SAMPLE_RATE, 2);
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();
    let mask = edit_keep_mask(&resolved);
    assert_eq!(mask.len(), 712_704);

    // Polarity: ones KEEP. Under a flipped mask the two `all` assertions below swap, so this cannot
    // pass inverted.
    assert!(
        mask[..88_200].iter().all(|value| *value == 1.0),
        "before the region the source is kept"
    );
    assert!(
        mask[88_200..308_700].iter().all(|value| *value == 0.0),
        "inside the region the source is erased"
    );
    assert!(
        mask[308_700..441_000].iter().all(|value| *value == 1.0),
        "after the region and before seconds_total the source is kept again"
    );
    // Training parity: everything past `seconds_total` is zero even though the attention mask's
    // 6 s headroom still marks part of it valid. Leaving ones there is an input the model never saw
    // in training, and it is invisible in every output-length or preservation assertion.
    assert!(
        mask[441_000..].iter().all(|value| *value == 0.0),
        "the padding past seconds_total is zeroed for training parity"
    );

    // And the region is genuinely bounded on both sides, so an "erase everything" mask fails.
    assert_eq!(
        mask.iter().filter(|value| **value == 0.0).count(),
        (308_700 - 88_200) + (712_704 - 441_000)
    );
}

/// The `[1, 257, latent_length]` local conditioning: channel order, polarity, and masked latents.
///
/// Weight-free because [`edit_local_conditioning`] takes the encoded source as a plain tensor rather
/// than encoding it. That is the point of the seam: inside the SAME-encoding synthesis method the
/// entire construction would have been reachable only from a real-weight lane, which is exactly how
/// sc-14547's three sign inversions each stayed green.
///
/// The synthetic latents are `channel * 1000 + position`, distinct in both axes, so a transposed or
/// mis-sliced concat produces different numbers rather than a plausible-looking tensor.
#[test]
fn the_local_conditioning_is_mask_first_then_the_masked_source() {
    let device = Device::Cpu;
    let geometry = small_geometry(10.0);
    let source = source_clip(10.0, SAMPLE_RATE, 2);
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();
    let length = resolved.latent_length;
    let values: Vec<f32> = (0..LATENT_CHANNELS)
        .flat_map(|channel| (0..length).map(move |position| (channel * 1_000 + position) as f32))
        .collect();
    let latents = Tensor::from_vec(values, (1, LATENT_CHANNELS, length), &device).unwrap();

    let local = edit_local_conditioning(&resolved, &latents, &device, DType::F32).unwrap();
    assert_eq!(local.dims(), &[1, 1 + LATENT_CHANNELS, length]);
    let local = local.to_vec3::<f32>().unwrap();
    let latents = latents.to_vec3::<f32>().unwrap();

    // Channel 0 is the mask, in the order `local_add_cond_ids` declares. Concatenating the other way
    // round has the identical shape and the model still runs, so the discriminator is the *content*:
    // channel 0 must be binary and channel 1 must not be.
    let mask = &local[0][0];
    assert!(
        mask.iter().all(|value| *value == 0.0 || *value == 1.0),
        "channel 0 must be the binary inpaint mask, not a latent row"
    );
    assert!(
        local[0][1]
            .iter()
            .any(|value| *value != 0.0 && *value != 1.0),
        "channel 1 must be a latent row, not the mask — the concat order is [mask, masked_input]"
    );

    // The zeroed latent positions are exactly the edit window union the padding, derived above.
    let zeros: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, value)| **value == 0.0)
        .map(|(index, _)| index)
        .collect();
    let expected: Vec<usize> = (22..76).chain(108..174).collect();
    assert_eq!(
        zeros, expected,
        "edit window [22,76) plus padding [108,174)"
    );

    // `masked_input = source * mask`, per channel: the source survives outside the window and is
    // exactly zero inside it. Handing the DiT the *unmasked* source inside the window shows it the
    // answer, and nothing about the shape or the mask channel would reveal it.
    for channel in 0..LATENT_CHANNELS {
        for position in 0..length {
            let expected = if mask[position] == 1.0 {
                latents[0][channel][position]
            } else {
                0.0
            };
            assert_eq!(
                local[0][channel + 1][position],
                expected,
                "channel {channel} position {position}"
            );
        }
    }
    // Two-sided: some masked value is genuinely nonzero, so an all-zero masked_input fails.
    assert!(
        local[0][1..].iter().flatten().any(|value| *value != 0.0),
        "the kept latents must survive the multiply"
    );

    // A latents tensor of the wrong length is refused rather than broadcast into place.
    let wrong = Tensor::zeros((1, LATENT_CHANNELS, length + 1), DType::F32, &device).unwrap();
    assert!(edit_local_conditioning(&resolved, &wrong, &device, DType::F32).is_err());
}

/// The local conditioning must still be non-zero where it **crosses** into the DiT.
///
/// # The mutation this is written against
///
/// `local = edit_local_conditioning(...)` → `let _ = edit_local_conditioning(...)` at the one call
/// site inside `StableAudio3Pipeline::synthesize_traced`. One token, and it leaves `local` as the
/// zero tensor the text-only path allocates, so every inpaint, repaint and extend runs as plain
/// text-to-audio inside the region while still replacing the region and still preserving the
/// outside.
///
/// The case above pins the construction; this pins the *handoff*, which is a different seam — the
/// sc-14547 finding, applied. And the reason it cannot be a divergence floor instead: an
/// unconditioned interior diverges **more** from the source and **more** between seeds than a
/// conditioned one, so both real-weight floors are wrong-signed against this mutation and tightening
/// them makes it easier to pass, not harder.
///
/// Two halves, because the rule and the expectation are separately falsifiable:
///
/// * the rule, over all four boolean combinations, the way sc-14547's `reference_halves_agree` is
///   gated — including the reverse direction, so a text-only request that somehow carried a non-zero
///   local conditioner is refused too;
/// * the expectation, `edit_retained_latent_count`, cross-checked against what
///   `edit_local_conditioning` actually builds over a spread of regions. That function is a second,
///   independent derivation from the resolved indices; if it were computed by re-running the mask
///   construction it would move with the mutation instead of catching it, so it is checked against
///   the real tensor here rather than assumed to agree.
///
/// The degenerate row is included on purpose: a region covering the whole effective span retains
/// nothing, hands the DiT an all-zero conditioner legitimately, and is therefore the one geometry on
/// which the guard cannot see the zeroing mutation. Named, not hidden.
#[test]
fn the_local_conditioning_handoff_must_still_be_present_where_it_crosses_into_the_dit() {
    // The rule, all four combinations.
    assert!(edit_local_conditioning_is_present(false, false).is_ok());
    assert!(edit_local_conditioning_is_present(true, true).is_ok());
    assert!(
        edit_local_conditioning_is_present(true, false).is_err(),
        "an edit that retains source latents but hands the DiT zeros is the whole point of this \
         guard"
    );
    assert!(
        edit_local_conditioning_is_present(false, true).is_err(),
        "and the reverse: nothing but an edit may put a non-zero local conditioner on the wire"
    );

    // The expectation, against the tensor the pipeline actually builds.
    let device = Device::Cpu;
    let geometry = small_geometry(10.0);
    let source = source_clip(10.0, SAMPLE_RATE, 2);
    let mut saw_retaining = false;
    let mut saw_empty = false;
    for (start, end) in [(2.0f32, 7.0f32), (0.0, 4.0), (9.0, 10.0), (0.0, 16.0)] {
        let edit = AudioEdit {
            samples: &source.samples,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            start_secs: start,
            end_secs: end,
        };
        let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();
        let length = resolved.latent_length;
        let values: Vec<f32> = (0..LATENT_CHANNELS)
            .flat_map(|channel| {
                (0..length).map(move |position| (channel * 1_000 + position) as f32)
            })
            .collect();
        let latents = Tensor::from_vec(values, (1, LATENT_CHANNELS, length), &device).unwrap();
        let local = edit_local_conditioning(&resolved, &latents, &device, DType::F32).unwrap();
        // The *shipped* predicate, not a restatement of it. `tensor_has_nonzero` is what
        // `synthesize_traced` calls, and that call site needs weights; computing `observed` here
        // with a hand-rolled `any(|v| *v != 0.0)` would move with any mutation of the real one
        // instead of catching it, so the real one is exported `#[doc(hidden)]` and driven directly.
        let observed = tensor_has_nonzero(&local).unwrap();
        let expected = edit_retained_latent_count(&resolved) > 0;
        assert_eq!(
            expected, observed,
            "[{start}s, {end}s): the retained-latent count and the built tensor must agree on \
             whether the DiT is conditioned at all"
        );
        assert!(edit_local_conditioning_is_present(expected, observed).is_ok());
        saw_retaining |= expected;
        saw_empty |= !expected;
    }
    // Two-sided: the spread above covers both answers, so neither branch of the equality is
    // untested and the assertion is not "true == true" four times.
    assert!(
        saw_retaining && saw_empty,
        "the region spread must produce both a conditioning edit and the degenerate whole-span one"
    );

    // The rows above are all non-negative — the mask is 0/1 and the synthetic latents count upward
    // — so the `.abs()` inside the predicate is invisible on them. Drive the cancelling case
    // directly: a tensor that sums to zero elementwise-signed is still *present*, and a bare
    // `sum_all` would call the DiT unconditioned on a source that happens to be symmetric.
    let cancelling = Tensor::from_vec(vec![3.0f32, -3.0, 1.5, -1.5], (1, 2, 2), &device).unwrap();
    assert!(
        tensor_has_nonzero(&cancelling).unwrap(),
        "a local conditioning whose elements cancel to a zero sum is still present; the presence \
         reduction must be over |x|"
    );
}

/// The resolved edit geometry must describe the geometry the request is actually sampled at.
///
/// # The mutation this is written against
///
/// The `duration_secs` argument at `edit_geometry(&edit, &geometry, parameters.duration_secs)`
/// inside `synthesize_traced`. One token, weights-only, and it sets `effective_samples` — the
/// training-parity padding boundary the keep mask zeroes from. Substituting the source's own extent,
/// or `edit.end_secs`, moves where the local conditioner stops describing the source; the output
/// keeps its length, its rate and its bit-exact outside.
///
/// The comparison is against `SampleGeometry::effective_lengths`, which `adapt_sample_size` derives
/// from the same duration by a genuinely different route — `ceil` to a 4096 multiple, in latent
/// units, against this path's `trunc` in audio samples — so it is a cross-check rather than a copy.
/// Its granularity is that latent frame: a duration wrong by under 93 ms is **not** caught here, and
/// `the_edit_geometry_reproduces_the_pinned_alignment_examples` is what pins the sample-resolution
/// value itself.
#[test]
fn the_resolved_edit_geometry_must_match_the_geometry_the_request_is_sampled_at() {
    let geometry = small_geometry(10.0);
    let source = source_clip(10.0, SAMPLE_RATE, 2);
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: SAMPLE_RATE,
        channels: 2,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();
    assert!(
        edit_geometry_matches_request(&resolved, &geometry).is_ok(),
        "the shipped pairing must pass"
    );

    // A duration argument that drifted: resolved against 10 s, sampled at 18 s and vice versa.
    let other = small_geometry(18.0);
    assert!(
        edit_geometry_matches_request(&resolved, &other).is_err(),
        "a resolution taken at the wrong duration must be refused"
    );
    let long = edit_geometry(&edit, &other, 18.0).unwrap();
    assert!(edit_geometry_matches_request(&long, &other).is_ok());
    assert!(
        edit_geometry_matches_request(&long, &geometry).is_err(),
        "and in the other direction"
    );

    // The same sampling geometry, resolved against a different *duration* only — the exact shape of
    // the call-site mutation, with `edit` and `geometry` both untouched.
    let drifted = edit_geometry(&edit, &geometry, 7.0).unwrap();
    assert_eq!(drifted.adapted_size, resolved.adapted_size);
    assert_ne!(drifted.effective_samples, resolved.effective_samples);
    assert!(
        edit_geometry_matches_request(&drifted, &geometry).is_err(),
        "substituting the region's end for the output duration must be refused"
    );
}

/// Guidance varies the prompt, and only the prompt: both CFG branches see the same local tensor.
///
/// Extracted from `StableAudio3Dit::forward_guided` — which needs multi-gigabyte weights — precisely
/// so this is checkable here. The mistake it excludes is the intuitive one: "unconditional means no
/// conditioning", i.e. zeroing the negative half. That keeps every shape, keeps the model running,
/// and drives the guided delta with a branch that never saw the source, so the regenerated region
/// fights its surroundings instead of continuing them.
#[test]
fn the_negative_cfg_branch_receives_the_same_local_conditioning() {
    let device = Device::Cpu;
    let values: Vec<f32> = (0..(257 * 5)).map(|index| index as f32 + 0.5).collect();
    let local = Tensor::from_vec(values, (1usize, 257usize, 5usize), &device).unwrap();
    let batched = batch_cfg_local_conditioning(&local).unwrap();
    assert_eq!(batched.dims(), &[2, 257, 5]);
    let batched = batched.to_vec3::<f32>().unwrap();
    let original = local.to_vec3::<f32>().unwrap();
    assert_eq!(batched[0], original[0], "the conditional branch");
    assert_eq!(
        batched[1], original[0],
        "the unconditional branch sees the identical tensor, not a zeroed one"
    );
    // Two-sided: the tensor is not itself zero, so a `zeros_like` mutation cannot pass.
    assert!(batched[1].iter().flatten().any(|value| *value != 0.0));
}

/// Outside the region the output is the prepared source **bit for bit**.
///
/// The story's original acceptance said "preserved to a tight numeric bound"; that is amended to
/// exact equality, and the reason is mechanical. Frozen upstream regenerates and decodes the whole
/// clip and pastes nothing back, so the preservation here is not something the model does — it is an
/// explicit stitch. A numeric tolerance would pass a stitch that had drifted by a frame or dropped a
/// channel, which is the entire failure class this exists to exclude. Any future crossfade must
/// therefore live wholly inside `[start, end)`.
#[test]
fn the_stitch_restores_the_prepared_source_outside_the_region_exactly() {
    let geometry = small_geometry(10.0);
    let source = source_clip(10.0, 48_000, 1);
    let prepared = prepare_reference_pcm(&source.samples, 48_000, 1, geometry.sample_size).unwrap();
    let edit = AudioEdit {
        samples: &source.samples,
        sample_rate: 48_000,
        channels: 1,
        start_secs: 2.0,
        end_secs: 7.0,
    };
    let resolved = edit_geometry(&edit, &geometry, 10.0).unwrap();

    // A "render" that agrees with the source nowhere, so preservation cannot be an accident.
    let frames = 441_000usize;
    let rendered: Vec<f32> = (0..frames * CHANNELS)
        .map(|index| -1.0 - (index % 97) as f32)
        .collect();
    let stitched = stitch_outside_region(&rendered, &prepared, &resolved).unwrap();
    assert_eq!(stitched.len(), rendered.len());

    for frame in 0..frames {
        let inside = (88_200..308_700).contains(&frame);
        for channel in 0..CHANNELS {
            let index = frame * CHANNELS + channel;
            if inside {
                assert_eq!(
                    stitched[index], rendered[index],
                    "frame {frame} is inside the region and must come from the render"
                );
            } else {
                assert_eq!(
                    stitched[index], prepared[index],
                    "frame {frame} is outside the region and must be the prepared source exactly"
                );
            }
        }
    }
    // Two-sided: the interior really did survive, so a stitch that overwrote everything fails.
    assert_ne!(
        stitched[88_200 * CHANNELS],
        prepared[88_200 * CHANNELS],
        "the region itself must not be overwritten by the source"
    );
    // Stereo is not collapsed: the prepared buffer's two channels are written independently.
    let stereo_source = source_clip(10.0, SAMPLE_RATE, 2);
    let prepared_stereo =
        prepare_reference_pcm(&stereo_source.samples, SAMPLE_RATE, 2, geometry.sample_size)
            .unwrap();
    let stitched = stitch_outside_region(&rendered, &prepared_stereo, &resolved).unwrap();
    assert!(
        (0..1_000).any(|frame| stitched[frame * CHANNELS] != stitched[frame * CHANNELS + 1]),
        "a stereo source must keep its two channels distinct through the stitch"
    );

    // A prepared buffer shorter than the render is refused rather than silently truncating the
    // preserved span.
    assert!(stitch_outside_region(&rendered, &prepared[..64], &resolved).is_err());
    assert!(stitch_outside_region(&rendered[..3], &prepared, &resolved).is_err());
}

/// The whole typed-rejection surface, on every variant.
///
/// Two things here are not routine coverage.
///
/// **`strength` must be refused, and `None` must succeed.** Asserting only that `Some(0.5)` errors
/// would be satisfied by a family that rejects every edit; asserting only that `None` succeeds would
/// be satisfied by one that ignores the field. The shipped anti-pattern this avoids is
/// `chatterbox/src/model.rs`, which destructures `ReferenceAudio { audio, .. }` and discards the
/// strength silently.
///
/// **The `ReferenceAudio` + `AudioEdit` refusal is newly reachable.** sc-14547 shipped it as
/// declared defence in depth: `AudioEdit` was not advertised, so the generic allowlist refused the
/// item on its own and the specific check never fired. Advertising the kind turns it on, so it is
/// asserted here from the edit side as well as from the reference side, and both must produce the
/// same message.
#[test]
fn audio_edit_validation_rejects_every_malformed_request_on_every_variant() {
    for variant in Variant::ALL {
        let id = variant.model_id();
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/nonexistent/sa3")));
        assert!(load_variant(variant, &spec).is_err());
        let source = source_clip(6.0, SAMPLE_RATE, 2);

        let valid = edit_request(
            "a cymbal swell",
            source.clone(),
            AudioEditMode::Inpaint,
            region(2.0, Some(4.0)),
            Some(6.0),
        );
        assert!(
            validate(variant, &valid).is_ok(),
            "{id}: a well-formed inpaint must be accepted: {:?}",
            validate(variant, &valid)
        );

        // --- strength: refused, and its absence is not ------------------------------------------
        for strength in [0.0f32, 0.5, 1.0] {
            let mut invalid = valid.clone();
            invalid.conditioning = vec![Conditioning::AudioEdit {
                audio: source.clone(),
                mode: AudioEditMode::Inpaint,
                region: region(2.0, Some(4.0)),
                strength: Some(strength),
            }];
            let error = validate(variant, &invalid);
            assert!(
                matches!(error, Err(gen_core::Error::Unsupported(_))),
                "{id}: audio edit strength {strength} must be typed Unsupported, got {error:?}"
            );
            assert!(
                format!("{error:?}").contains("ReferenceAudio"),
                "{id}: the refusal must name the knob that does exist, got {error:?}"
            );
        }

        // --- arity: two edits are refused, never first-match-wins --------------------------------
        let mut invalid = valid.clone();
        invalid.conditioning.push(Conditioning::AudioEdit {
            audio: source.clone(),
            mode: AudioEditMode::Inpaint,
            region: region(4.5, Some(5.0)),
            strength: None,
        });
        assert!(
            matches!(
                validate(variant, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ),
            "{id}: a second edit region must be refused, not silently dropped by audio_edit()'s \
             first-match resolution, got {:?}",
            validate(variant, &invalid)
        );

        // --- the combination, from the edit side -------------------------------------------------
        let mut invalid = valid.clone();
        invalid.conditioning.push(Conditioning::ReferenceAudio {
            audio: source.clone(),
            strength: Some(0.4),
        });
        let from_edit = validate(variant, &invalid);
        assert!(
            matches!(from_edit, Err(gen_core::Error::Unsupported(_))),
            "{id}: reference + edit must be typed Unsupported, got {from_edit:?}"
        );
        // And from the reference side, with the two conditioning items in the other order: one
        // hoisted check, so the caller sees one message either way.
        let mut reversed = valid.clone();
        reversed.conditioning.insert(
            0,
            Conditioning::ReferenceAudio {
                audio: source.clone(),
                strength: Some(0.4),
            },
        );
        assert_eq!(
            format!("{:?}", validate(variant, &reversed)),
            format!("{from_edit:?}"),
            "{id}: the combination refusal must not depend on conditioning order"
        );

        // --- regions ------------------------------------------------------------------------------
        let cases: [(&str, AudioEditMode, Option<TimeRegion>, Option<f32>); 8] = [
            ("no region", AudioEditMode::Inpaint, None, Some(6.0)),
            (
                "start past the source",
                AudioEditMode::Inpaint,
                region(6.5, Some(7.0)),
                Some(6.0),
            ),
            (
                "end past the source",
                AudioEditMode::Repaint,
                region(2.0, Some(9.0)),
                Some(6.0),
            ),
            (
                "sub-latent-frame span",
                AudioEditMode::Inpaint,
                region(2.0, Some(2.01)),
                Some(6.0),
            ),
            (
                "extend starting before the source end",
                AudioEditMode::Extend,
                region(4.0, Some(9.0)),
                Some(9.0),
            ),
            (
                "extend with no new length",
                AudioEditMode::Extend,
                region(6.0, None),
                None,
            ),
            (
                "target_duration conflicting with an inpaint",
                AudioEditMode::Inpaint,
                region(2.0, Some(4.0)),
                Some(30.0),
            ),
            (
                "target_duration conflicting with an extend",
                AudioEditMode::Extend,
                region(6.0, Some(9.0)),
                Some(12.0),
            ),
        ];
        for (name, mode, span, duration) in cases {
            let invalid = edit_request("edit this", source.clone(), mode, span, duration);
            assert!(
                validate(variant, &invalid).is_err(),
                "{id}: {name} must be rejected"
            );
        }

        // An extend past the variant's advertised cap is refused even though `target_duration` —
        // which the generic floor caps — is absent.
        let over = edit_request(
            "keep going forever",
            source.clone(),
            AudioEditMode::Extend,
            region(6.0, Some(variant.max_duration_secs() + 10.0)),
            None,
        );
        assert!(
            validate(variant, &over).is_err(),
            "{id}: an extend past the {}s cap must be rejected",
            variant.max_duration_secs()
        );

        // --- malformed source clips: caller data, so `Msg` ---------------------------------------
        for (name, audio) in [
            ("empty", track(Vec::new(), SAMPLE_RATE, 2)),
            (
                "non-finite",
                track(vec![0.1, f32::INFINITY], SAMPLE_RATE, 2),
            ),
            ("zero rate", track(vec![0.1; 441_000], 0, 2)),
            ("zero channels", track(vec![0.1; 441_000], SAMPLE_RATE, 0)),
            ("ragged", track(vec![0.1, 0.2, 0.3], SAMPLE_RATE, 2)),
        ] {
            let invalid = edit_request(
                "edit this",
                audio,
                AudioEditMode::Inpaint,
                region(0.5, Some(1.0)),
                None,
            );
            assert!(
                matches!(validate(variant, &invalid), Err(gen_core::Error::Msg(_))),
                "{id}: a {name} source clip must be rejected as Msg (bad caller data), got {:?}",
                validate(variant, &invalid)
            );
        }
    }
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

fn audio(output: GenerationOutput) -> AudioTrack {
    match output {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio output, got {other:?}"),
    }
}

/// Floor for the source-sensitivity measurement, as a fraction of the interior's source energy.
///
/// Set **after** the measurement, from the low end of the measured spread: the six ids land at
/// `1.318`–`1.813` times the interior's source energy (see the case's own table), and this sits
/// `1.88x` below the tightest of them — the same relative distance the other two floors take from
/// their own low mode. It is an order of magnitude above the `0.08` the other two floors use because
/// the quantity is an order of magnitude larger, not because it was tuned.
///
/// Its job is grading, not gating. The gate against a dropped local-conditioning handoff is the
/// byte-inequality beside it, which that mutation fails at *any* threshold because the two renders
/// become identical.
const CONDITIONING_DIVERGENCE_FLOOR: f64 = 0.70;

/// Mean absolute value of an interleaved span, both channels.
fn energy(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples
        .iter()
        .map(|value| (*value as f64).abs())
        .sum::<f64>()
        / samples.len() as f64
}

/// Mean absolute difference of two equal-length interleaved spans.
fn divergence(left: &[f32], right: &[f32]) -> f64 {
    let count = left.len().min(right.len());
    if count == 0 {
        return 0.0;
    }
    (0..count)
        .map(|index| (left[index] as f64 - right[index] as f64).abs())
        .sum::<f64>()
        / count as f64
}

/// The acceptance, both halves, on every registered checkpoint.
///
/// Outside the region the assertion is **exact equality**, not a bound: that span is written by
/// `stitch_outside_region` from the same buffer this test prepares, so anything other than equality
/// is a bug rather than a tolerance question. A tight numeric bound would pass a stitch that had
/// slipped a frame, which is the failure the amended acceptance names.
///
/// Preservation alone passes for a no-op, so the interior is asserted to have changed in the same
/// pass. **Three** measurements, because they answer three different questions and two of them are
/// blind to the failure the third exists for:
///
/// * `source_divergence` — the region against the prepared source. This is the obvious measurement
///   and it is the *weak* one: a full SAME encode/decode round trip already diverges from its input
///   (sc-14547 measured source correlation `0.966`–`0.986` at full retention, not `1.0`), so a
///   region that was merely copied through the autoencoder would still score well above zero. The
///   floor below is therefore not "is it nonzero" but "is it past what a round trip alone explains",
///   and it is set from the **measured minimum across all six ids**, not chosen.
/// * `seed_divergence` — the same region rendered at two seeds. This is the one that cannot be
///   satisfied by a copy: a stitch that overwrote the whole clip, an unmasked local conditioning
///   that handed the model its own answer, or any other degeneracy that returns the source produces
///   a *seed-independent* interior. Nothing about it is fitted — a copy scores identically zero.
/// * `conditioning_divergence` — the same region rendered from a **different source clip** at the
///   identical seed. This is the only one of the three that observes whether the DiT is conditioned
///   on the source at all, and the only one correctly signed against that failure. Both measurements
///   above get **larger** when the interior stops being conditioned — an unconditioned region
///   wanders further from the source and further between seeds — so no tightening of either can
///   catch a dropped local-conditioning handoff, and tightening them makes it *easier* to pass. Two
///   renders that agree on prompt, seed, region, duration and geometry and differ only in their
///   source clip are bit-identical if the source never reaches the model, so the assertion beside
///   the floor is a **byte-inequality**, which that mutation fails at any threshold. The weight-free
///   half of the same seam is `pipeline::edit_local_conditioning_is_present`.
///
/// # The floors are measured, not chosen
///
/// Metal, `--release`, 6 s / 4 steps, 48 kHz mono source, region `[2 s, 4 s)`, seeds 7 and 4242.
/// Interior source energy is `0.124632` on every row (the same prepared buffer), so the floor
/// `0.08 * energy` is `0.009971`.
///
/// | id | source divergence | seed divergence | conditioning divergence |
/// |---|---|---|---|
/// | `stable_audio_3_small_music` | 0.049025 | 0.060849 | 0.164266 |
/// | `stable_audio_3_small_sfx` | 0.040199 | 0.034961 | 0.181044 |
/// | `stable_audio_3_medium` | 0.023272 | 0.018142 | 0.165492 |
/// | `stable_audio_3_small_music_base` | 0.060284 | 0.043346 | 0.166105 |
/// | `stable_audio_3_small_sfx_base` | 0.072861 | 0.068967 | 0.225991 |
/// | `stable_audio_3_medium_base` | 0.022110 | 0.018060 | 0.178970 |
///
/// The conditioning column is `1.318`–`1.813` times the source energy, i.e. **an order of magnitude
/// above** either of the other two: swapping the source clip changes the interior far more than
/// changing the seed does, which is the shape "the region is conditioned on the source" has. Under
/// the zeroed-handoff mutation that column is exactly `0.000000` on every row.
///
/// The spread is bimodal by autoencoder family — the two SAME-L ids sit at roughly `0.018`–`0.023`
/// and the four SAME-S ids at `0.035`–`0.073` — so the floor is taken from the **low** mode, at
/// `1.8x` below its tightest member (`medium_base`'s `0.018048`) rather than fitted to the average.
///
/// The first calibration attempt asserted `source_divergence > 0.25 * source_energy` — a number
/// picked before any measurement, and the only measurement in the case. `medium` failed it at
/// `0.023271` against a `0.031158` floor **while genuinely regenerating the region**. That is
/// recorded rather than quietly rewritten: it is exactly the "thresholds must be derived, not
/// eyeballed, and taken from the low mode of a bimodal sweep" hazard the story names, and the fix
/// was both to re-derive the floor and to add `seed_divergence`, which a copy fails at *any*
/// threshold because a copy scores zero.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_inpaint_preserves_the_outside_exactly_and_changes_the_inside() {
    let seconds = 6.0f32;
    let frames = (seconds as f64 * SAMPLE_RATE as f64) as usize;
    let source = source_clip(seconds, 48_000, 1);
    let other = alternate_source_clip(seconds, 48_000, 1);
    let prepared = prepare_reference_pcm(&source.samples, 48_000, 1, frames).unwrap();
    let (start_frame, end_frame) = edit_region_samples(2.0, 4.0);

    // Every id is measured before anything is asserted, so one failing checkpoint does not hide the
    // other five's numbers — which is precisely what made the first floor hard to calibrate.
    let mut measured = Vec::new();
    for case in CASES {
        let spec = LoadSpec::new(snapshot(case.env));
        let generator = load_variant(case.variant, &spec).expect("load pinned snapshot");
        let id = generator.descriptor().id;
        let render = |seed: u64, clip: &AudioTrack| {
            let mut request = edit_request(
                case.prompt,
                clip.clone(),
                AudioEditMode::Inpaint,
                region(2.0, Some(4.0)),
                Some(seconds),
            );
            request.seed = Some(seed);
            audio(
                generator
                    .generate(&request, &mut |_| {})
                    .unwrap_or_else(|error| panic!("{id} @ seed {seed}: {error}")),
            )
        };
        let output = render(7, &source);
        let alternate = render(4_242, &source);
        // Same seed, same prompt, same region, same duration — a different source clip and nothing
        // else. See the doc comment: this is the half that observes conditioning.
        let other_source = render(7, &other);
        assert_eq!(output.sample_rate, SAMPLE_RATE);
        assert_eq!(output.channels as usize, CHANNELS);
        assert_eq!(
            output.samples.len(),
            frames * CHANNELS,
            "{id}: an inpaint's output is exactly as long as its source"
        );
        assert!(
            output.samples.iter().all(|value| value.is_finite()),
            "{id}: emitted non-finite PCM"
        );

        // Half one: outside the region, bit for bit — on **both** renders, so the preservation is a
        // property of the stitch and not of one lucky draw.
        for frame in (0..frames).filter(|frame| !(start_frame..end_frame).contains(frame)) {
            for channel in 0..CHANNELS {
                let index = frame * CHANNELS + channel;
                assert_eq!(
                    output.samples[index], prepared[index],
                    "{id}: frame {frame} channel {channel} is outside [2s, 4s) and must be the \
                     prepared source exactly"
                );
                assert_eq!(
                    alternate.samples[index], prepared[index],
                    "{id}: the same must hold at another seed"
                );
            }
        }

        let inside = start_frame * CHANNELS..end_frame * CHANNELS;
        let source_divergence =
            divergence(&output.samples[inside.clone()], &prepared[inside.clone()]);
        let seed_divergence = divergence(
            &output.samples[inside.clone()],
            &alternate.samples[inside.clone()],
        );
        // The conditioning half. Two renders that agree on every input except the source clip; if
        // the source never reaches the DiT they are bit-identical, because the request stream, the
        // draw counts, the prompt and the geometry are all identical between them.
        let conditioning_divergence = divergence(
            &output.samples[inside.clone()],
            &other_source.samples[inside.clone()],
        );
        // Deliberately `assert!` on a comparison rather than `assert_ne!`: the operands are
        // 176,400-sample slices and the macro would dump both of them into the failure output.
        assert!(
            output.samples[inside.clone()] != other_source.samples[inside.clone()],
            "{id}: two inpaints differing only in their source rendered the *identical* interior, \
             which is what an unconditioned region looks like"
        );
        let reference = energy(&prepared[inside]);
        println!(
            "{id} source_divergence={source_divergence:.6} seed_divergence={seed_divergence:.6} \
             conditioning_divergence={conditioning_divergence:.6} source_energy={reference:.6}"
        );
        measured.push((
            id,
            source_divergence,
            seed_divergence,
            conditioning_divergence,
            reference,
        ));
    }

    for (id, source_divergence, seed_divergence, conditioning_divergence, reference) in &measured {
        // The weak measurement, with a floor taken from the low end of the measured spread rather
        // than from an intuition. It is stated against the source's own energy so it means the same
        // thing whatever the material is.
        assert!(
            *source_divergence > 0.08 * reference,
            "{id}: the region must differ from the source — divergence {source_divergence:.6} \
             against source energy {reference:.6}"
        );
        // The strong one: a region that was copied rather than generated is seed-independent, and
        // scores identically zero here however loud the material is. No fitted threshold can hide
        // that, which is why this exists alongside the measurement above rather than instead of it.
        assert!(
            *seed_divergence > 0.08 * reference,
            "{id}: the region must be *generated* — two seeds produced interiors differing by only \
             {seed_divergence:.6}, which is what a copied region looks like"
        );
        // The correctly-signed one. Both floors above get *larger* when the interior stops being
        // conditioned on the source, so neither can see a zeroed local conditioner; this one goes to
        // exactly zero. The byte-inequality above is the threshold-free form and is the actual gate
        // against the mutation; this floor is the graded version, taken from the measured spread.
        assert!(
            *conditioning_divergence > CONDITIONING_DIVERGENCE_FLOOR * reference,
            "{id}: swapping the source changed the interior by only \
             {conditioning_divergence:.6} against source energy {reference:.6} — the region is \
             barely conditioned on the clip being edited"
        );
    }
}

/// `Repaint` is an alias of `Inpaint` on this family, so the same request and seed must be
/// byte-identical.
///
/// gen-core documents the two as different (silence-substituted vs context-conditioned), but that
/// distinction is written against ACE-Step's two native tasks; upstream Stable Audio 3 has one
/// inpaint mechanism. `pipeline::AudioEdit` carries no mode field, so this is structural — and it is
/// asserted anyway, because a structural argument does not survive somebody adding one.
///
/// # What this case is structurally blind to, and why it must never be cited as a fallback catcher
///
/// It compares **two renders of the same code**. Any mutation on the shared edit path is applied
/// identically to both, so the equality holds and the `energy > 1e-4` control still passes: the case
/// is blind, by construction, to every uniform mutation — not merely to the ones tried so far.
/// This was confirmed by running it, not reasoned: under `edit.channels` → `1` it passes **GREEN**
/// on all six ids, while `real_extend_…` under the same mutation is RED ("the first 10 s must be
/// the prepared source bit for bit"). Cite `real_inpaint_…` or `real_extend_…` as the catcher for a
/// shared-path mutation; this case only gates the alias itself.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_repaint_is_byte_identical_to_inpaint() {
    let seconds = 6.0f32;
    let source = source_clip(seconds, SAMPLE_RATE, 2);
    for case in CASES {
        let spec = LoadSpec::new(snapshot(case.env));
        let generator = load_variant(case.variant, &spec).expect("load pinned snapshot");
        let id = generator.descriptor().id;
        let render = |mode| {
            let request = edit_request(
                case.prompt,
                source.clone(),
                mode,
                region(2.0, Some(4.0)),
                Some(seconds),
            );
            audio(
                generator
                    .generate(&request, &mut |_| {})
                    .unwrap_or_else(|error| panic!("{id} {mode:?}: {error}")),
            )
            .samples
        };
        let inpaint = render(AudioEditMode::Inpaint);
        let repaint = render(AudioEditMode::Repaint);
        assert_eq!(
            inpaint, repaint,
            "{id}: Repaint and Inpaint are the same upstream path and must not diverge by a bit"
        );
        // Not vacuous: the render is not silence, so two all-zero buffers cannot pass.
        assert!(
            energy(&inpaint) > 1e-4,
            "{id}: the render must not be silent"
        );
    }
}

/// The story's pinned extend: a 10 s source becomes 18 s, keeps its first 10 s exactly, and does not
/// snap at the seam.
///
/// Three assertions, and the third is the one that is easy to omit. Preserving the prefix is the
/// stitch's job and is exact. Producing *something* in the tail is not enough — silence would pass a
/// preservation-only test. And a tail that is loud but discontinuous at 10 s is the specific failure
/// an extend has.
///
/// The seam predicate, stated whole rather than as "measured": `seam <= typical`, where `typical` is
/// this fixture's own 99.9th-percentile frame-to-frame step over the second before the seam. The
/// *yardstick* is measured; the multiplier is `1.0` and that is a choice, made because a join no
/// sharper than the material's own steepest transition is not audible as an edit. The earlier
/// `(typical * 8.0).max(0.05)` was not a weaker version of this — at `typical = 0.163181` it bounded
/// the seam at `1.305`, past the largest step reachable on this fixture at all, so it admitted every
/// tail including a fully discontinuous one.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_extend_keeps_the_source_prefix_and_bridges_the_seam() {
    let source_seconds = 10.0f32;
    let total_seconds = 18.0f32;
    let source_frames = 441_000usize;
    let total_frames = 793_800usize;
    let source = source_clip(source_seconds, SAMPLE_RATE, 2);
    let prepared = prepare_reference_pcm(&source.samples, SAMPLE_RATE, 2, total_frames).unwrap();

    for case in CASES {
        let spec = LoadSpec::new(snapshot(case.env));
        let generator = load_variant(case.variant, &spec).expect("load pinned snapshot");
        let id = generator.descriptor().id;
        let request = edit_request(
            case.prompt,
            source.clone(),
            AudioEditMode::Extend,
            region(source_seconds, Some(total_seconds)),
            Some(total_seconds),
        );
        let output = audio(
            generator
                .generate(&request, &mut |_| {})
                .unwrap_or_else(|error| panic!("{id}: {error}")),
        );
        assert_eq!(
            output.samples.len(),
            total_frames * CHANNELS,
            "{id}: 18 s of stereo at 44.1 kHz"
        );

        // 1. The prefix, exactly.
        assert_eq!(
            &output.samples[..source_frames * CHANNELS],
            &prepared[..source_frames * CHANNELS],
            "{id}: the first 10 s must be the prepared source bit for bit"
        );

        // 2. The tail is real audio, not silence and not a continuation of the zero padding the
        //    prepared buffer carries there.
        let tail = &output.samples[source_frames * CHANNELS..];
        let tail_energy = energy(tail);
        assert!(
            tail_energy > 1e-3,
            "{id}: the appended tail must carry audio, got mean |x| {tail_energy:.6}"
        );
        assert!(
            energy(&prepared[source_frames * CHANNELS..]) == 0.0,
            "{id}: the prepared buffer is silent past the source, so the tail came from the model"
        );

        // 3. The seam is not a discontinuity. The predicate is stated in full because the
        //    multiplier is part of it: the seam step must be **no larger than** the material's own
        //    99.9th-percentile frame-to-frame step over the second before the seam. Multiplier
        //    `1.0`, no additive floor.
        //
        //    What that protects against: a tail that starts at a level unrelated to where the
        //    source left off — the click an extend produces when the model regenerates the tail
        //    without conditioning on the prefix, or when the stitch boundary is off by a frame. The
        //    source's own steepest transition is the right yardstick because a join sharper than
        //    anything the material itself does is audible as an edit; a join no sharper than that
        //    is not.
        //
        //    The 8x multiplier and 0.05 floor this replaces were **unfalsifiable**: `typical` is
        //    0.163181 on this fixture, so the bound was 1.305, above the largest step reachable at
        //    all here — `source_clip`'s envelope decays to zero at the 10 s mark, so the seam step
        //    is bounded by the tail's own opening amplitude. Any tail passed, discontinuous or not.
        //    At `1.0` the measured seam steps (0.034-0.046) sit ~4x inside the bound, which is
        //    headroom against checkpoint variation rather than slack fitted to the numbers.
        //
        //    `typical` is itself asserted to be a meaningful number, so a future fixture whose
        //    waveform was nearly flat would fail loudly here instead of silently tightening this
        //    into a flake.
        let step = |buffer: &[f32], frame: usize| -> f64 {
            (0..CHANNELS)
                .map(|channel| {
                    (buffer[frame * CHANNELS + channel] as f64
                        - buffer[(frame - 1) * CHANNELS + channel] as f64)
                        .abs()
                })
                .fold(0.0, f64::max)
        };
        let mut steps: Vec<f64> = ((source_frames - SAMPLE_RATE as usize)..source_frames)
            .map(|frame| step(&output.samples, frame))
            .collect();
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let typical = steps[steps.len() * 999 / 1000];
        let seam = step(&output.samples, source_frames);
        println!("{id} seam_step={seam:.6} typical_step={typical:.6} tail_energy={tail_energy:.6}");
        assert!(
            typical > 0.02,
            "{id}: the fixture's own 99.9th-percentile step is {typical:.6}; a bound derived from \
             material that flat would be a flake, not a gate"
        );
        assert!(
            seam <= typical,
            "{id}: the 10 s seam jumps {seam:.6}, past the material's own steepest {typical:.6} \
             transition"
        );
    }
}

/// The draw order, on the edit path: the sampler's initial noise is the request stream's **first**
/// draw and the source encode's draws come after it.
///
/// # How much this discriminates, stated honestly
///
/// **SAME-S consumes zero draws on encode** (measured on sc-14547 across all six pinned snapshots).
/// So on the four small ids `draws_after_source_encode` equals `draws_after_initial_noise` and
/// swapping the two operations would not move a count — the ordering assertion is *vacuous* there.
/// Only medium's SAME-L encode draws, so only `medium` and `medium_base` can falsify it. Running all
/// six and separately requiring at least one drawing encode is what keeps this from decaying into a
/// tautology if a future change made every encode deterministic.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn real_edit_initial_sampler_noise_precedes_the_source_encode() {
    use candle_audio_stable_audio_3::pipeline::{StableAudio3Pipeline, SynthesisParameters};
    use candle_audio_stable_audio_3::weights::SnapshotLayout;
    use candle_audio_stable_audio_3::{resolve_device, DevicePolicy};

    let seconds = 4.0f32;
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
            .synthesize_with_edit_traced(
                "fill this region",
                None,
                parameters,
                AudioEdit {
                    samples: &source.samples,
                    sample_rate: SAMPLE_RATE,
                    channels: 2,
                    start_secs: 1.0,
                    end_secs: 2.0,
                },
                &mut |_, _| {},
                &mut || {},
                &|| false,
            )
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(
            samples.len(),
            (seconds as f64 * SAMPLE_RATE as f64) as usize * CHANNELS
        );
        let order = order.expect("an edit render reports its draw order");
        assert_eq!(
            order.draws_after_initial_noise, 1,
            "{id}: the sampler's initial noise must be the request stream's first draw"
        );
        assert!(order.draws_after_source_encode >= order.draws_after_initial_noise);
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
        "no checkpoint's encode consumed a draw, so the ordering assertion discriminated nothing \
         on any of them — the SAME-L encoder's eval-time noise is what makes this a gate"
    );
}
