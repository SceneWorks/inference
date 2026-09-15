//! Bounded Stable Audio 3 reference-preparation regression (sc-16601).
//!
//! This is a separate test binary so the shared resampler's process-global work counter cannot be
//! perturbed by another reference test running concurrently.

use candle_audio_stable_audio_3::candle_audio::dsp::{
    resample, resample_output_frames, resample_test_support,
};
use candle_audio_stable_audio_3::pipeline::{prepare_reference_pcm, CHANNELS, SAMPLE_RATE};

#[test]
fn reference_preparation_uses_a_bounded_global_window_and_matches_whole_clip_slices() {
    const SOURCE_RATE: u32 = 48_000;
    const SOURCE_FRAMES: usize = SOURCE_RATE as usize;
    const TARGET_FRAMES: usize = 4_096;
    let source: Vec<f32> = (0..SOURCE_FRAMES * 2)
        .map(|index| ((index * 37 % 1_009) as f32 - 504.0) / 504.0)
        .collect();
    let full_frames = resample_output_frames(SOURCE_FRAMES, SOURCE_RATE, SAMPLE_RATE).unwrap();
    assert!(full_frames > TARGET_FRAMES * 10);

    resample_test_support::reset();
    let prepared =
        prepare_reference_pcm(&source, SOURCE_RATE, 2, TARGET_FRAMES).expect("bounded prepare");
    let (output_frames, source_frame_work) = resample_test_support::work();
    assert_eq!(output_frames, TARGET_FRAMES);
    assert!(source_frame_work > 0);
    assert!(
        source_frame_work <= TARGET_FRAMES * 215 * 2,
        "the 48 kHz -> 44.1 kHz window must touch at most its 215-tap support per requested frame and retained input channel, saw {source_frame_work}"
    );
    assert_eq!(prepared.len(), TARGET_FRAMES * CHANNELS);

    // Global phase and both clip boundaries come from the complete source: the bounded result must
    // be bit-identical to slicing the historical whole-buffer resampler, not merely close.
    resample_test_support::reset();
    let whole = resample(&source, SOURCE_RATE, SAMPLE_RATE, 2).unwrap();
    let (_, whole_source_frame_work) = resample_test_support::work();
    assert!(
        source_frame_work * 10 < whole_source_frame_work,
        "bounded source work {source_frame_work} must be more than 10x below whole-clip work {whole_source_frame_work}"
    );
    assert_eq!(prepared, whole[..TARGET_FRAMES * CHANNELS]);

    // A short off-rate mono clip exercises duplication, the complete clip's trailing FIR boundary,
    // and right padding together. Every retained bit still comes from the whole-buffer result.
    let mono: Vec<f32> = (0..1_000)
        .map(|index| ((index * 19 % 263) as f32 - 131.0) / 131.0)
        .collect();
    let mono_target = 2_000usize;
    let whole_mono = resample(&mono, 48_000, SAMPLE_RATE, 1).unwrap();
    let mut expected_mono = vec![0.0f32; mono_target * CHANNELS];
    for (frame, &sample) in whole_mono.iter().enumerate() {
        expected_mono[frame * CHANNELS] = sample;
        expected_mono[frame * CHANNELS + 1] = sample;
    }
    assert_eq!(
        prepare_reference_pcm(&mono, 48_000, 1, mono_target).unwrap(),
        expected_mono
    );

    // The same compatibility boundary holds for >2 channels while conformance retains only the
    // first two. No target-sized source-channel padding allocation is observable in the result.
    let quad_frames = 1_000usize;
    let quad: Vec<f32> = (0..quad_frames * 4)
        .map(|index| ((index * 13 % 257) as f32 - 128.0) / 128.0)
        .collect();
    let target = 1_500usize;
    let whole_quad = resample(&quad, 32_000, SAMPLE_RATE, 4).unwrap();
    let available = (whole_quad.len() / 4).min(target);
    let mut expected = vec![0.0f32; target * CHANNELS];
    for frame in 0..available {
        expected[frame * CHANNELS] = whole_quad[frame * 4];
        expected[frame * CHANNELS + 1] = whole_quad[frame * 4 + 1];
    }
    assert_eq!(
        prepare_reference_pcm(&quad, 32_000, 4, target).unwrap(),
        expected
    );
}
