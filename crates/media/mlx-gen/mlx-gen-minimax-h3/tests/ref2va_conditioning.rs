//! sc-17149: **`ref2va` multi-modal reference conditioning, end to end through the engine
//! plumbing.**
//!
//! The story's crux acceptance criterion is that all three reference modalities *bind*,
//! individually and combined. That is the same hard bar `fl2va_conditioning.rs` documents, for the
//! same reason: **a pipeline that silently ignores a reference still produces plausible video.**
//! Shape, norm, variance and finiteness are all unchanged, so "the render succeeded" passes on a
//! wholly unconditioned model.
//!
//! # What binding means here, stated as separable claims
//!
//! | claim | what breaks it | gated by |
//! |---|---|---|
//! | an **image** reference's pixels reach its conditioning latent | encoding a constant, or the mean | `an_image_reference_binds_its_own_pixels` |
//! | a **clip** reference's pixels reach its conditioning latent, and its frames are distinguishable | encoding only frame 0 | `a_clip_reference_binds_every_frame` |
//! | an **audio** reference reaches the packed sequence as clean condition rows | dropping the audio prepend | `an_audio_reference_binds_as_clean_condition_rows` |
//! | the three **combine** without any of them being dropped | a layout that reserves rows for two of three | `the_three_modalities_combine_without_dropping_one` |
//! | reference rows reach the **model's input** | dropping the prepend | `the_model_sees_every_reference_in_its_rows` |
//! | reference rows **survive** the loop | stepping the whole block instead of its tail | `references_are_never_denoised` |
//! | **order is semantic** | sorting the reference list by modality | `reordering_the_same_references_is_a_different_request` |
//!
//! # The metric is relative max-abs-diff, and deliberately nothing else
//!
//! Norm, cosine and checksum have each been blind to a real defect in this epic — cosine is
//! scale-invariant, norms barely move under a half-swap (89 vs 85 while every block was
//! 0.86–0.99 wrong), and a checksum answers a question nobody asked. Every claim below is gated on
//! [`common::rel`]'s peak-relative difference, with a **paired control**: two different references
//! must differ by more than [`MUTATION_FLOOR`], and the *same* reference twice must differ by less
//! than [`DETERMINISM_CEILING`]. Without the second half a per-call RNG draw would satisfy the
//! first.
//!
//! # Why this runs at fixture scale
//!
//! The real render is ~655 s at ~53 GB and needs the 66 GB `transformer_ref`. Everything here
//! drives the **real** code — `mlx_gen_minimax_h3::reference`, `conditioning`, `PackedLayout`,
//! `prepend_condition_rows`, `denoise_av` — against the committed tiny VAE with a recording
//! `JointVelocity` standing in for the DiT. Checkpoint selection and the two-checkpoint memory
//! property are gated separately, on real weights, by `ref2va_checkpoint.rs`.

use crate::common;

use common::{encode_fixture_config, rel, ENCODE_FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::media::{AudioTrack, Image};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_minimax_h3::audio_config::AUDIO_SAMPLE_RATE;
use mlx_gen_minimax_h3::conditioning::{
    encode_reference_condition, keyframe_condition_rows, reference_audio_rows,
    reference_clip_to_vae_pixels, KeyframeNoise,
};
use mlx_gen_minimax_h3::dit::positions::ReferenceLatentGeometry;
use mlx_gen_minimax_h3::pipeline::{prepend_condition_audio_rows, prepend_condition_rows};
use mlx_gen_minimax_h3::reference::{
    AudioReference, Ref2VaReference, Ref2VaReferences, ReferenceKind, VideoReference,
};
use mlx_gen_minimax_h3::{
    denoise_av, JointGeometry, JointSchedule, JointStep, JointVelocity, MiniMaxH3VideoVae,
    PackedLayout, RowClass, REFERENCE_AUDIO_TIMESTEP, SMALLEST_LEGAL_FRAMES, TEXT_TAG,
};

/// A mutation must move a tensor by at least this much to count as gated.
const MUTATION_FLOOR: f32 = 1e-2;

/// Two runs that should agree must agree to at least this.
const DETERMINISM_CEILING: f32 = 1e-6;

/// The DiT patch — one latent frame of a 6x6 latent is 9 rows of 96 features.
const PATCH: [i32; 3] = [1, 2, 2];

/// Reference edge the fixture VAE compresses 4x into a 6x6 latent.
const REF_EDGE: u32 = 24;

/// Audio latent channels the fixture works in.
const AUDIO_FEATURES: i32 = 32;

/// Audio channels the soundtrack is packed channel-major over.
const AUDIO_CHANNELS: i32 = 2;

fn vae() -> MiniMaxH3VideoVae {
    let mut w = Weights::from_file(ENCODE_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "const."] {
        w.remove_prefix(prefix);
    }
    MiniMaxH3VideoVae::from_weights(&mut w, &encode_fixture_config(3), Dtype::Float32).unwrap()
}

/// A probe image that **cannot be confused with an unconditioned result**.
///
/// Deliberately not a flat fill and not the canvas mean: a reference at the mean of the latent
/// distribution encodes to roughly what an unconditioned block looks like, so a test built on one
/// reports PASS whether or not the reference was used. `seed` selects between structurally
/// different, high-contrast patterns, so "different reference" is a large, unambiguous change.
fn probe_image(edge: u32, seed: u32) -> Image {
    let mut pixels = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            let on = match seed % 3 {
                0 => ((x + y) / 3) % 2 == 0,
                1 => (x.max(y) / 4) % 2 == 0,
                _ => ((x * 2 + y / 2) / 5) % 2 == 0,
            };
            let (r, g, b) = if on { (250, 20, 90) } else { (10, 200, 240) };
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    Image {
        width: edge,
        height: edge,
        pixels,
    }
}

/// A synthetic soundtrack whose samples are structurally distinguishable per `seed`.
fn probe_audio(seed: u32) -> AudioTrack {
    AudioTrack {
        samples: (0..512)
            .map(|i| (i as f32 * (0.05 + seed as f32 * 0.03)).sin())
            .collect(),
        // The engine ships no resampler, so `Ref2VaReferences::new` admits only this rate.
        sample_rate: AUDIO_SAMPLE_RATE,
        channels: 1,
        stems: Vec::new(),
    }
}

/// Already-normalized reference audio latents, standing in for the audio VAE's posterior mean.
///
/// The audio VAE **encoder** has its own parity gate; what is under test here is whether encoded
/// soundtrack rows reach the packed sequence and survive it, which is a property of the layout and
/// the prepend rather than of the encoder's arithmetic. `seed` makes two soundtracks
/// distinguishable.
fn audio_latents(num_latents: i32, seed: u32) -> Array {
    let n = (AUDIO_CHANNELS * num_latents * AUDIO_FEATURES) as usize;
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.017) + seed as f32 * 1.7).sin())
        .collect();
    Array::from_slice(&v, &[AUDIO_CHANNELS, num_latents, AUDIO_FEATURES])
}

fn geometry() -> JointGeometry {
    JointGeometry::new(SMALLEST_LEGAL_FRAMES, 6, 6).unwrap()
}

/// The encoded latent geometry an image reference of `REF_EDGE` produces through the fixture VAE.
fn image_geometry() -> ReferenceLatentGeometry {
    ReferenceLatentGeometry {
        kind: ReferenceKind::Image,
        num_latent_frames: 1,
        latent_height: 6,
        latent_width: 6,
        num_audio_latents: 0,
    }
}

fn clip_geometry(num_latent_frames: i32, num_audio_latents: i32) -> ReferenceLatentGeometry {
    ReferenceLatentGeometry {
        kind: ReferenceKind::Video,
        num_latent_frames,
        latent_height: 6,
        latent_width: 6,
        num_audio_latents,
    }
}

fn audio_geometry(num_audio_latents: i32) -> ReferenceLatentGeometry {
    ReferenceLatentGeometry {
        kind: ReferenceKind::Audio,
        num_latent_frames: 0,
        latent_height: 0,
        latent_width: 0,
        num_audio_latents,
    }
}

fn layout(refs: &[ReferenceLatentGeometry], num_text: usize) -> PackedLayout {
    PackedLayout::build_ref2va(
        geometry(),
        PATCH,
        &vec![TEXT_TAG; num_text],
        AUDIO_CHANNELS,
        refs,
    )
    .unwrap()
}

/// One image reference's conditioning rows, through the real conditioning path.
fn image_rows(seed: u32) -> Array {
    let v = vae();
    let pixels = reference_clip_to_vae_pixels(&[probe_image(REF_EDGE, seed)]).unwrap();
    let condition = encode_reference_condition(&v, &pixels, &KeyframeNoise::Seeded, 0).unwrap();
    keyframe_condition_rows(&condition, PATCH, &KeyframeNoise::Seeded, 0).unwrap()
}

/// One clip reference's conditioning rows, through the real conditioning path.
fn clip_rows(seeds: &[u32]) -> Array {
    let v = vae();
    let frames: Vec<Image> = seeds.iter().map(|&s| probe_image(REF_EDGE, s)).collect();
    let pixels = reference_clip_to_vae_pixels(&frames).unwrap();
    let condition = encode_reference_condition(&v, &pixels, &KeyframeNoise::Seeded, 0).unwrap();
    keyframe_condition_rows(&condition, PATCH, &KeyframeNoise::Seeded, 0).unwrap()
}

/// A `JointVelocity` that records exactly what the model was handed at each step.
///
/// The velocity is `0.5·rows + 1`, **not** zero: a zero velocity makes the Euler update collapse to
/// `prev = sample`, so the loop is inert and "the references did not move" holds trivially.
struct Recorder {
    seen_video: Vec<Array>,
    seen_audio: Vec<Array>,
    calls: usize,
}

impl JointVelocity for Recorder {
    fn forward(&mut self, step: &JointStep<'_>) -> mlx_gen::Result<(Array, Array)> {
        self.calls += 1;
        self.seen_video.push(step.video_rows.clone());
        self.seen_audio.push(step.audio_rows.clone());
        let velocity = |x: &Array| -> mlx_gen::Result<Array> {
            Ok(x.multiply(Array::from_f32(0.5))?
                .add(Array::from_f32(1.0))?)
        };
        Ok((velocity(step.video_rows)?, velocity(step.audio_rows)?))
    }
}

/// Run the joint loop; returns `(video at step 0, audio at step 0, final video, final audio)`.
fn run(
    layout: &PackedLayout,
    video_rows: &Array,
    audio_rows: &Array,
) -> (Array, Array, Array, Array) {
    let schedule = JointSchedule::new(3).unwrap();
    let adaln = mlx_gen_minimax_h3::adaln_schedule(&schedule).unwrap();
    let mut rec = Recorder {
        seen_video: Vec::new(),
        seen_audio: Vec::new(),
        calls: 0,
    };
    let (video, audio) = denoise_av(
        &mut rec,
        layout,
        &schedule,
        &adaln,
        video_rows,
        audio_rows,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();
    assert!(rec.calls > 0, "the loop must have evaluated the model");
    (
        rec.seen_video[0].clone(),
        rec.seen_audio[0].clone(),
        video,
        audio,
    )
}

fn generated_video(layout: &PackedLayout) -> Array {
    let g = layout.geometry();
    let rows = g.num_latent_frames * layout.rows_per_frame();
    let n = (rows * 96) as usize;
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).cos()).collect();
    Array::from_slice(&v, &[1, rows, 96])
}

fn generated_audio(layout: &PackedLayout) -> Array {
    let g = layout.geometry();
    let rows = g.num_audio_latents * AUDIO_CHANNELS;
    let n = (rows * AUDIO_FEATURES) as usize;
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    Array::from_slice(&v, &[1, rows, AUDIO_FEATURES])
}

/// `[1, start..end, features]` — the crate's own axis slice, so the tests cut rows the same way the
/// engine does.
fn slice(x: &Array, start: i32, end: i32) -> Array {
    mlx_gen_minimax_h3::tensor::slice_axis(x, 1, start, end).unwrap()
}

// ---------------------------------------------------------------------------------------------
// 1. Each modality binds individually
// ---------------------------------------------------------------------------------------------

/// **Claim 1 — an image reference's own pixels reach its conditioning latent.**
///
/// Two structurally different probes must produce conditioning rows that differ by more than
/// [`MUTATION_FLOOR`]; the *same* probe twice must agree to [`DETERMINISM_CEILING`]. The second
/// half is what makes the first attributable to the image rather than to a per-call RNG draw.
#[test]
fn an_image_reference_binds_its_own_pixels() {
    let a = image_rows(0);
    let b = image_rows(1);
    let again = image_rows(0);

    let (peak_same, _) = rel(&a, &again);
    assert!(
        peak_same < DETERMINISM_CEILING,
        "the same reference twice must encode identically, got {peak_same:e}"
    );
    let (peak_diff, _) = rel(&a, &b);
    assert!(
        peak_diff > MUTATION_FLOOR,
        "two different references must encode differently, got {peak_diff:e}"
    );
}

/// **Claim 2 — a clip reference binds every frame, not just its first.**
///
/// Two clips that share frame 0 and differ only from frame 1 onwards must still produce different
/// conditioning. An implementation that encoded only the first frame — a plausible shortcut, since
/// an image reference *is* a one-frame clip — would pass every shape check and fail here.
#[test]
fn a_clip_reference_binds_every_frame() {
    // Same first frame, different tail.
    let a = clip_rows(&[0, 1, 1, 1, 1, 1]);
    let b = clip_rows(&[0, 2, 2, 2, 2, 2]);
    let again = clip_rows(&[0, 1, 1, 1, 1, 1]);

    let (peak_same, _) = rel(&a, &again);
    assert!(
        peak_same < DETERMINISM_CEILING,
        "the same clip twice must encode identically, got {peak_same:e}"
    );
    let (peak_diff, _) = rel(&a, &b);
    assert!(
        peak_diff > MUTATION_FLOOR,
        "clips sharing only their first frame must encode differently, got {peak_diff:e}"
    );

    // …and a clip is genuinely more rows than a single image, so the temporal axis is not collapsed.
    let single = image_rows(0);
    assert!(
        a.shape()[1] > single.shape()[1],
        "a {}-row clip must occupy more rows than a {}-row still",
        a.shape()[1],
        single.shape()[1]
    );
}

/// **Claim 3 — an audio reference reaches the packed sequence as CLEAN condition rows.**
///
/// Three separable facts, because the audio path is the one modality with no visual analogue:
/// the rows are reserved by the layout, they carry [`RowClass::ConditionAudio`], and that class
/// sits at [`REFERENCE_AUDIO_TIMESTEP`] (`1.0`) rather than the `0.999` visual anchors are held at.
#[test]
fn an_audio_reference_binds_as_clean_condition_rows() {
    let latents = 4;
    let l = layout(&[image_geometry(), audio_geometry(latents)], 5);

    let reserved = l.num_condition_audio_rows();
    assert_eq!(
        reserved,
        latents * AUDIO_CHANNELS,
        "the layout must reserve one row per latent per channel"
    );

    // The rows really are the encoded soundtrack, and the prepend agrees with the layout.
    let rows = reference_audio_rows(&audio_latents(latents, 0)).unwrap();
    assert_eq!(rows.shape(), &[1, reserved, AUDIO_FEATURES]);
    let packed = prepend_condition_audio_rows(&l, Some(&rows), &generated_audio(&l)).unwrap();
    assert_eq!(
        packed.shape()[1],
        reserved + l.geometry().num_audio_latents * AUDIO_CHANNELS
    );

    // A block of the wrong height is REFUSED rather than concatenated into a runnable, silently
    // misaligned sequence.
    let wrong = reference_audio_rows(&audio_latents(latents + 1, 0)).unwrap();
    assert!(prepend_condition_audio_rows(&l, Some(&wrong), &generated_audio(&l)).is_err());
    assert!(
        prepend_condition_audio_rows(&l, None, &generated_audio(&l)).is_err(),
        "a layout reserving reference audio must not accept a missing block"
    );

    // The class and its timestep.
    let classes: Vec<i32> = l.row_classes().as_slice::<i32>().to_vec();
    let audio_idx = l.audio_indices();
    for &row in audio_idx.iter().take(reserved as usize) {
        assert_eq!(
            classes[row as usize],
            RowClass::ConditionAudio as i32,
            "leading audio rows are reference soundtrack"
        );
    }
    for &row in audio_idx.iter().skip(reserved as usize) {
        assert_eq!(classes[row as usize], RowClass::Audio as i32);
    }
    let t = PackedLayout::row_timesteps(0.3, 0.6);
    assert_eq!(
        t[RowClass::ConditionAudio as usize],
        REFERENCE_AUDIO_TIMESTEP,
        "a reference soundtrack rides CLEAN at 1.0, not at the visual 0.999"
    );
    assert_ne!(
        t[RowClass::ConditionAudio as usize],
        t[RowClass::ConditionVideo as usize],
        "the two conditioning classes are genuinely different timesteps"
    );
}

// ---------------------------------------------------------------------------------------------
// 2. Combined
// ---------------------------------------------------------------------------------------------

/// **All three modalities combine, and the sequence accounts for every one of them.**
///
/// The row count is checked against the sum of the parts rather than against a recorded constant,
/// and each single-modality layout is shown to be strictly shorter — so a layout that quietly
/// dropped one modality would fail on arithmetic rather than on a golden number nobody can read.
#[test]
fn the_three_modalities_combine_without_dropping_one() {
    let text = 5;
    let latents = 4;
    let img = image_geometry();
    let clip = clip_geometry(2, 0);
    let aud = audio_geometry(latents);

    let only_image = layout(&[img], text);
    let only_clip = layout(&[clip], text);
    let combined = layout(&[img, clip, aud], text);

    // Video conditioning rows are additive across the two visual modalities.
    assert_eq!(
        combined.num_condition_video_rows(),
        only_image.num_condition_video_rows() + only_clip.num_condition_video_rows(),
        "the combined layout must reserve both visual references"
    );
    assert_eq!(
        combined.num_condition_audio_rows(),
        latents * AUDIO_CHANNELS
    );

    // Each single-modality layout is strictly shorter than the combined one.
    for (name, l) in [("image", &only_image), ("clip", &only_clip)] {
        assert!(
            l.seq_len() < combined.seq_len(),
            "{name}-only ({}) must be shorter than combined ({})",
            l.seq_len(),
            combined.seq_len()
        );
    }

    // The whole packed sequence is exactly text + references + generated audio + generated video.
    let g = combined.geometry();
    let expected = text as i32
        + combined.num_condition_video_rows()
        + combined.num_condition_audio_rows()
        + g.num_audio_latents * AUDIO_CHANNELS
        + g.num_latent_frames * combined.rows_per_frame();
    assert_eq!(combined.seq_len(), expected);

    // And a clip that carries its OWN soundtrack contributes audio rows without being an audio
    // reference — the label/cap distinction, expressed in rows.
    let sounded = layout(&[img, clip_geometry(2, latents)], text);
    assert_eq!(sounded.num_condition_audio_rows(), latents * AUDIO_CHANNELS);
    assert_eq!(
        sounded.num_condition_video_rows(),
        combined.num_condition_video_rows()
    );
}

/// The three reference modalities each bind through the **request** surface, individually and
/// combined — the acceptance criterion stated in the caller's own vocabulary.
#[test]
fn every_modality_is_constructible_individually_and_combined() {
    let img = || Ref2VaReference::Image(probe_image(REF_EDGE, 0));
    let clip = |sound: bool| {
        Ref2VaReference::Video(VideoReference {
            frames: vec![probe_image(REF_EDGE, 1); 30],
            fps: 24.0,
            audio: sound.then(|| probe_audio(0)),
        })
    };
    let aud = || {
        Ref2VaReference::Audio(AudioReference {
            audio: probe_audio(1),
        })
    };

    // Individually — audio only ever paired, which is the model's own rule.
    Ref2VaReferences::new(vec![img()]).expect("image alone");
    Ref2VaReferences::new(vec![clip(false)]).expect("clip alone");
    Ref2VaReferences::new(vec![clip(true)]).expect("clip with its own soundtrack");
    Ref2VaReferences::new(vec![img(), aud()]).expect("audio paired with an image");

    // Combined, at the caps.
    let mut all = vec![img(); 9];
    all.extend(vec![clip(true); 2]);
    all.push(aud());
    let refs = Ref2VaReferences::new(all).expect("9 + 2 + 1 = 12");
    assert_eq!(refs.len(), 12);
    assert_eq!(refs.counts().images, 9);
    assert_eq!(refs.counts().videos, 2);
    assert_eq!(
        refs.audio_count(),
        1,
        "only standalone audio counts to the cap"
    );
    assert_eq!(
        refs.audio_label_count(),
        3,
        "two clip soundtracks plus the standalone one each take an <Audio j> label"
    );
}

// ---------------------------------------------------------------------------------------------
// 3. The references reach the model, and survive it
// ---------------------------------------------------------------------------------------------

/// **Claim 4 — the model is handed every reference, in its own rows.**
///
/// The leading video rows the model sees *are* the conditioning block, the leading audio rows *are*
/// the soundtrack block, and swapping a reference moves what the model sees. The negative control
/// is a layout with one fewer reference, which must carry strictly fewer rows — so "the model saw
/// the references" is not satisfied by a sequence that would have those rows anyway.
#[test]
fn the_model_sees_every_reference_in_its_rows() {
    let latents = 4;
    let l = layout(&[image_geometry(), audio_geometry(latents)], 5);
    let cond_v = image_rows(0);
    let cond_a = reference_audio_rows(&audio_latents(latents, 0)).unwrap();

    let video = prepend_condition_rows(&l, Some(&cond_v), &generated_video(&l)).unwrap();
    let audio = prepend_condition_audio_rows(&l, Some(&cond_a), &generated_audio(&l)).unwrap();
    let (seen_v, seen_a, _, _) = run(&l, &video, &audio);

    let n_v = l.num_condition_video_rows();
    let n_a = l.num_condition_audio_rows();
    let lead = |x: &Array, n: i32| slice(x, 0, n);

    let (peak, _) = rel(&lead(&seen_v, n_v), &cond_v);
    assert!(
        peak < DETERMINISM_CEILING,
        "the model's leading video rows must BE the conditioning block, got {peak:e}"
    );
    let (peak, _) = rel(&lead(&seen_a, n_a), &cond_a);
    assert!(
        peak < DETERMINISM_CEILING,
        "the model's leading audio rows must BE the soundtrack block, got {peak:e}"
    );

    // Swapping the image reference moves what the model is handed.
    let other = image_rows(1);
    let video2 = prepend_condition_rows(&l, Some(&other), &generated_video(&l)).unwrap();
    let (seen_v2, _, _, _) = run(&l, &video2, &audio);
    let (peak, _) = rel(&lead(&seen_v, n_v), &lead(&seen_v2, n_v));
    assert!(
        peak > MUTATION_FLOOR,
        "a different reference must change the model's input, got {peak:e}"
    );

    // Negative control: dropping the audio reference is strictly fewer rows.
    let fewer = layout(&[image_geometry()], 5);
    assert_eq!(fewer.num_condition_audio_rows(), 0);
    assert!(fewer.seq_len() < l.seq_len());
}

/// **Claim 5 — reference rows ride through the loop untouched while the generated tail moves.**
///
/// Both halves matter. "The references did not move" is trivially true of an inert loop, so the
/// generated tail is asserted to move by more than [`MUTATION_FLOOR`] in the same run.
#[test]
fn references_are_never_denoised() {
    let latents = 4;
    let l = layout(&[image_geometry(), audio_geometry(latents)], 5);
    let cond_v = image_rows(0);
    let cond_a = reference_audio_rows(&audio_latents(latents, 0)).unwrap();
    let video = prepend_condition_rows(&l, Some(&cond_v), &generated_video(&l)).unwrap();
    let audio = prepend_condition_audio_rows(&l, Some(&cond_a), &generated_audio(&l)).unwrap();

    let (_, _, final_v, final_a) = run(&l, &video, &audio);
    let (n_v, n_a) = (l.num_condition_video_rows(), l.num_condition_audio_rows());

    let (peak, _) = rel(&slice(&final_v, 0, n_v), &cond_v);
    assert!(
        peak < DETERMINISM_CEILING,
        "reference video rows must survive the loop unchanged, got {peak:e}"
    );
    let (peak, _) = rel(&slice(&final_a, 0, n_a), &cond_a);
    assert!(
        peak < DETERMINISM_CEILING,
        "reference audio rows must survive the loop unchanged, got {peak:e}"
    );

    // …and the loop is NOT inert: the generated tail moved.
    let tail_before = slice(&video, n_v, video.shape()[1]);
    let tail_after = slice(&final_v, n_v, final_v.shape()[1]);
    let (peak, _) = rel(&tail_after, &tail_before);
    assert!(
        peak > MUTATION_FLOOR,
        "the generated tail must actually be denoised, got {peak:e} — otherwise 'the references \
         did not move' proves nothing"
    );
}

// ---------------------------------------------------------------------------------------------
// 4. Order is semantic
// ---------------------------------------------------------------------------------------------

/// **Reordering the same references is a different request.**
///
/// The rotary clock is shared and each block advances it, so the *same* set of references in a
/// different order lands on different position ids. A layout that sorted by modality — the obvious
/// implementation, and what `fl2va` legitimately does for its two fixed slots — would produce
/// identical position ids here and this is what catches it.
#[test]
fn reordering_the_same_references_is_a_different_request() {
    let text = 5;
    let img = image_geometry();
    let clip = clip_geometry(2, 0);
    let aud = audio_geometry(4);

    let a = layout(&[img, clip, aud], text);
    let b = layout(&[aud, clip, img], text);

    // Same rows — this is the same multiset of references.
    assert_eq!(a.seq_len(), b.seq_len());
    assert_eq!(a.num_condition_video_rows(), b.num_condition_video_rows());
    assert_eq!(a.num_condition_audio_rows(), b.num_condition_audio_rows());

    // …and genuinely different positions.
    let (peak, _) = rel(a.position_ids(), b.position_ids());
    assert!(
        peak > MUTATION_FLOOR,
        "reordering references must change the rotary layout, got {peak:e}"
    );

    // The same order twice is identical, so the difference above is the order and not noise.
    let a2 = layout(&[img, clip, aud], text);
    let (peak, _) = rel(a.position_ids(), a2.position_ids());
    assert!(peak < DETERMINISM_CEILING, "layout must be deterministic");
}

/// The shared rotary clock **advances per reference**: each block starts where the previous one
/// ended, and the generated rows start after all of them.
///
/// Pinned as an inequality chain rather than against recorded constants, because the interesting
/// failure is a clock that never advances — every reference stacked at one time, which is a
/// perfectly runnable sequence.
#[test]
fn the_rotary_clock_advances_across_reference_blocks() {
    let text = 5;
    let l = layout(
        &[image_geometry(), image_geometry(), image_geometry()],
        text,
    );
    let pos: Vec<f32> = l.position_ids().as_slice::<f32>().to_vec();
    let time_of = |row: usize| pos[row * 3];

    let rows_each = l.num_condition_video_rows() / 3;
    assert!(rows_each > 0);
    let v = l.video_indices();
    let t0 = time_of(v[0] as usize);
    let t1 = time_of(v[rows_each as usize] as usize);
    let t2 = time_of(v[(2 * rows_each) as usize] as usize);

    assert!(
        (t0 - text as f32).abs() < 1e-4,
        "the first reference starts where the text rows end, got {t0}"
    );
    // An image takes exactly ONE integer slot — not a latent frame's 5/3 span.
    assert!(
        (t1 - t0 - 1.0).abs() < 1e-4,
        "an image reference advances the clock by 1.0, got {}",
        t1 - t0
    );
    assert!((t2 - t1 - 1.0).abs() < 1e-4);
    // …and the generated rows start after every reference.
    let generated = time_of(v[l.num_condition_video_rows() as usize] as usize);
    assert!(
        generated > t2,
        "the generated video must start after the last reference ({generated} vs {t2})"
    );
}
