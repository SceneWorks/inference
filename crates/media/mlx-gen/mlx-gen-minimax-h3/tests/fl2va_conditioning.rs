//! sc-17148: **`fl2va` keyframe conditioning, end to end through the engine plumbing.**
//!
//! The story's crux acceptance criterion is that *conditioning actually binds* — "a test that
//! would fail if the supplied image were ignored". That is a harder bar than it sounds, because a
//! pipeline that silently drops a keyframe **still produces plausible video**. Nothing about the
//! output shape, its norm, its variance or its finiteness changes. So a test that only checks the
//! render succeeded is a test that passes on a completely unconditioned model.
//!
//! # What binding means here, stated as three separable claims
//!
//! | claim | what breaks it | gated by |
//! |---|---|---|
//! | the keyframe's **pixels** reach the conditioning latent | encoding a constant, or the mean | `the_conditioning_latent_depends_on_the_keyframe_pixels` |
//! | the conditioning latent reaches the **model's input** | dropping the prepend, or a layout with no anchors | `the_model_sees_the_keyframe_in_its_video_rows` |
//! | the anchors **survive** the loop | stepping the whole video block instead of its tail | `the_anchors_are_never_denoised` |
//!
//! Each is checked against a **different keyframe** as its mutation, so "the image is ignored" is a
//! measurable red rather than an untested assumption. The probe images are deliberately not
//! anything the unconditioned path could coincidentally produce — see [`probe_image`].
//!
//! # Why this runs at fixture scale
//!
//! The real render is ~655 s and 52.81 GB per shape, so gating four shapes on it in CI is not
//! available. Everything below drives the **real** conditioning code —
//! `mlx_gen_minimax_h3::conditioning`, `PackedLayout`, `prepend_condition_rows`, `denoise_av` —
//! against the committed tiny VAE, with a recording `JointVelocity` standing in for the 33 B DiT.
//! The DiT's own arithmetic is gated by `dit_parity.rs`; what is under test here is whether the
//! keyframe reaches it.

use crate::common;

use common::{encode_fixture_config, rel, ENCODE_FIXTURE};

use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_minimax_h3::conditioning::{build_condition_rows, KeyframeNoise};
use mlx_gen_minimax_h3::dit::positions::KeyframeAnchor;
use mlx_gen_minimax_h3::keyframe::{
    anchors_for, fit_keyframes, keyframe_to_vae_pixels, resolve_keyframe_canvas,
};
use mlx_gen_minimax_h3::pipeline::prepend_condition_rows;
use mlx_gen_minimax_h3::{
    denoise_av, resolve_geometry, JointGeometry, JointSchedule, JointStep, JointVelocity,
    MiniMaxH3VideoVae, PackedLayout, RowClass, SMALLEST_LEGAL_FRAMES, SPATIAL_STRIDE, TEXT_TAG,
    VIDEO_TAG,
};

/// A mutation must move a tensor by at least this much to count as gated.
const MUTATION_FLOOR: f32 = 1e-2;

/// The DiT patch. `[1, 2, 2]`, so one latent frame of a 6×6 latent is 9 rows of 96 features.
const PATCH: [i32; 3] = [1, 2, 2];

/// Keyframe canvas the fixture VAE compresses 4× into a 6×6 latent.
const KEYFRAME_EDGE: u32 = 24;

fn vae() -> MiniMaxH3VideoVae {
    let mut w = Weights::from_file(ENCODE_FIXTURE).unwrap();
    for prefix in ["src.", "in.", "out.", "const."] {
        w.remove_prefix(prefix);
    }
    MiniMaxH3VideoVae::from_weights(&mut w, &encode_fixture_config(3), Dtype::Float32).unwrap()
}

/// A probe image that **cannot be confused with an unconditioned result**.
///
/// Deliberately not a flat fill and not the canvas mean: a keyframe at the mean of the latent
/// distribution encodes to something close to what an unconditioned anchor would look like, so a
/// test built on one reports PASS whether or not the image was used. `seed` selects between two
/// structurally different, high-contrast patterns — diagonal bars and concentric blocks — so
/// "different keyframe" is a large, unambiguous change.
fn probe_image(edge: u32, seed: u32) -> Image {
    let mut pixels = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            let on = if seed == 0 {
                ((x + y) / 3) % 2 == 0
            } else {
                (x.max(y) / 4) % 2 == 0
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

/// Conditioning rows for a set of anchors, from the real conditioning path.
fn condition_rows(anchors: &[KeyframeAnchor], seeds: &[u32]) -> Option<Array> {
    let v = vae();
    let images: Vec<Image> = seeds
        .iter()
        .map(|&s| probe_image(KEYFRAME_EDGE, s))
        .collect();
    let refs: Vec<&Image> = images.iter().collect();
    let fitted = fit_keyframes(&refs, KEYFRAME_EDGE, KEYFRAME_EDGE).unwrap();
    let pixels: Vec<Array> = fitted
        .iter()
        .map(|i| keyframe_to_vae_pixels(i).unwrap())
        .collect();
    build_condition_rows(&v, &pixels, anchors, PATCH, &KeyframeNoise::Seeded).unwrap()
}

fn geometry() -> JointGeometry {
    // 6×6 latent — what the fixture VAE produces from a 24×24 keyframe at its 4× compression.
    JointGeometry::new(SMALLEST_LEGAL_FRAMES, 6, 6).unwrap()
}

fn layout(anchors: &[KeyframeAnchor], num_text: usize) -> PackedLayout {
    PackedLayout::build(geometry(), PATCH, &vec![TEXT_TAG; num_text], 2, anchors).unwrap()
}

/// A `JointVelocity` that records exactly what the model was handed at each step.
///
/// The velocity is `0.5·rows + 1` rather than zero. A **zero** velocity makes the Euler update
/// `prev = ratio·sample + (1 - ratio)·(sample + σ·0)` collapse to `sample`, i.e. the loop becomes
/// inert and "the anchors did not move" holds trivially — which is exactly the false green
/// `the_anchors_are_never_denoised` guards against. The `+ 1` keeps it non-zero even where the
/// incoming rows are.
struct Recorder {
    seen_video: Vec<Array>,
    calls: usize,
}

impl JointVelocity for Recorder {
    fn forward(&mut self, step: &JointStep<'_>) -> mlx_gen::Result<(Array, Array)> {
        self.calls += 1;
        self.seen_video.push(step.video_rows.clone());
        let velocity = |x: &Array| -> mlx_gen::Result<Array> {
            Ok(x.multiply(Array::from_f32(0.5))?
                .add(Array::from_f32(1.0))?)
        };
        Ok((velocity(step.video_rows)?, velocity(step.audio_rows)?))
    }
}

/// Run the joint loop and return `(what the model saw at step 0, the final video rows)`.
fn run(layout: &PackedLayout, video_rows: &Array) -> (Array, Array) {
    let g = layout.geometry();
    let audio =
        mlx_rs::ops::zeros_dtype(&[1, g.num_audio_latents * 2, 32], Dtype::Float32).unwrap();
    let schedule = JointSchedule::new(3).unwrap();
    let adaln = mlx_gen_minimax_h3::adaln_schedule(&schedule).unwrap();
    let mut rec = Recorder {
        seen_video: Vec::new(),
        calls: 0,
    };
    let (video, _) = denoise_av(
        &mut rec,
        layout,
        &schedule,
        &adaln,
        video_rows,
        &audio,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();
    assert!(rec.calls > 0, "the loop must have evaluated the model");
    (rec.seen_video[0].clone(), video)
}

// ---------------------------------------------------------------------------------------------
// 1. All four conditioning shapes
// ---------------------------------------------------------------------------------------------

/// **All four shapes build, and no two of them are the same sequence.**
///
/// 0 images, first-only, last-only and first+last differ in row count, in which rows are
/// conditioning, and — for the two one-image shapes, which have *identical* row counts — in the
/// rotary time their single anchor sits at. That last pair is the one a naive implementation gets
/// wrong: treating "one image" as "the first frame" produces a perfectly runnable last-frame
/// request that silently anchors the wrong end.
#[test]
fn all_four_conditioning_shapes_build_and_differ() {
    use KeyframeAnchor::{First, Last};
    let shapes: [(&str, Vec<KeyframeAnchor>); 4] = [
        ("0 images (t2va)", vec![]),
        ("first only", vec![First]),
        ("last only", vec![Last]),
        ("first + last", vec![First, Last]),
    ];

    let rows_per_frame = layout(&[], 4).rows_per_frame();
    let mut seq_lens = Vec::new();
    for (name, anchors) in &shapes {
        let l = layout(anchors, 4);
        assert_eq!(
            l.num_condition_video_rows(),
            anchors.len() as i32 * rows_per_frame,
            "{name}: one anchor reserves one latent frame of rows"
        );
        // The generated tail is the same size in every shape — conditioning ADDS rows, it does not
        // consume any of the ones being generated.
        assert_eq!(
            l.generated_video_rows().len() as i32,
            l.geometry().num_latent_frames * rows_per_frame,
            "{name}: the generated video is unchanged in size"
        );
        seq_lens.push(l.seq_len());
        println!(
            "{name}: seq_len {} condition_rows {}",
            l.seq_len(),
            l.num_condition_video_rows()
        );
    }
    // 0 < first == last < first+last.
    assert!(seq_lens[0] < seq_lens[1]);
    assert_eq!(
        seq_lens[1], seq_lens[2],
        "one image is one image either way"
    );
    assert!(seq_lens[2] < seq_lens[3]);

    // **The two one-image shapes are NOT the same request.** Same row count, different anchor
    // times — which is the only thing distinguishing them.
    let first = layout(&[First], 4);
    let last = layout(&[Last], 4);
    let anchor_time = |l: &PackedLayout| {
        let ids = l.position_ids().as_slice::<f32>().to_vec();
        let row = l.video_indices()[0] as usize;
        ids[row * 3]
    };
    let (tf, tl) = (anchor_time(&first), anchor_time(&last));
    println!("first anchor t = {tf}, last anchor t = {tl}");
    assert!(
        tl > tf,
        "the last-frame anchor must sit LATER on the rotary clock than the first-frame one \
         ({tl} vs {tf}); equal times would make the two shapes indistinguishable to the model"
    );

    // …and `anchors_for` produces exactly these four from the two presence bits.
    assert_eq!(anchors_for(false, false), shapes[0].1);
    assert_eq!(anchors_for(true, false), shapes[1].1);
    assert_eq!(anchors_for(false, true), shapes[2].1);
    assert_eq!(anchors_for(true, true), shapes[3].1);
}

/// Every shape produces conditioning rows of the size its layout reserved, and the two-image shape
/// really carries two distinct anchors rather than one image twice.
#[test]
fn every_shape_produces_the_rows_its_layout_reserves() {
    use KeyframeAnchor::{First, Last};
    for (anchors, seeds) in [
        (vec![], vec![]),
        (vec![First], vec![0]),
        (vec![Last], vec![0]),
        (vec![First, Last], vec![0, 1]),
    ] {
        let l = layout(&anchors, 4);
        let rows = condition_rows(&anchors, &seeds);
        match (&rows, anchors.len()) {
            (None, 0) => {}
            (Some(r), n) => {
                assert_eq!(r.shape()[0], 1);
                assert_eq!(
                    r.shape()[1],
                    n as i32 * l.rows_per_frame(),
                    "{n} anchors must produce {n} latent frames of rows"
                );
                assert_eq!(r.shape()[2], 96, "24 channels x 1x2x2 patch");
            }
            (None, n) => panic!("{n} anchors produced no rows"),
        }
        // The prepend agrees with the layout, which is the check that catches a mis-sized block.
        let video = mlx_rs::ops::zeros_dtype(
            &[1, l.generated_video_rows().len() as i32, 96],
            Dtype::Float32,
        )
        .unwrap();
        let packed = prepend_condition_rows(&l, rows.as_ref(), &video).unwrap();
        assert_eq!(packed.shape()[1], l.video_indices().len() as i32);
    }

    // Two keyframes, two DIFFERENT images: the two anchor blocks must not be equal.
    let two = condition_rows(&[First, Last], &[0, 1]).unwrap();
    let rpf = layout(&[First, Last], 4).rows_per_frame();
    let a = mlx_gen_minimax_h3::tensor::slice_axis(&two, 1, 0, rpf).unwrap();
    let b = mlx_gen_minimax_h3::tensor::slice_axis(&two, 1, rpf, 2 * rpf).unwrap();
    let (delta, _) = rel(&a, &b);
    println!("two distinct keyframes differ by peak-rel {delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "the two anchor blocks are identical ({delta:.3e}) — the second keyframe is being ignored"
    );
}

// ---------------------------------------------------------------------------------------------
// 2. BINDING — the three claims
// ---------------------------------------------------------------------------------------------

/// **Claim 1: the keyframe's pixels reach the conditioning latent.**
///
/// Two structurally different probe images must produce measurably different conditioning rows. An
/// implementation that encoded a constant, the canvas mean, or pure noise would produce the same
/// rows for both and fail here.
#[test]
fn the_conditioning_latent_depends_on_the_keyframe_pixels() {
    let a = condition_rows(&[KeyframeAnchor::First], &[0]).unwrap();
    let b = condition_rows(&[KeyframeAnchor::First], &[1]).unwrap();
    assert_eq!(a.shape(), b.shape());

    let (delta, mean_delta) = rel(&a, &b);
    println!("two keyframes: peak-rel {delta:.3e} mean-rel {mean_delta:.3e}");
    assert!(
        delta > MUTATION_FLOOR,
        "two different keyframes produced the same conditioning ({delta:.3e}); the pixels are \
         being ignored somewhere between the image and the latent"
    );

    // The same keyframe twice is deterministic — so the difference above is the IMAGE, not the
    // RNG. Without this, a per-call random draw would pass the assertion above for the wrong
    // reason.
    let a2 = condition_rows(&[KeyframeAnchor::First], &[0]).unwrap();
    let (same, _) = rel(&a, &a2);
    assert!(
        same < 1e-6,
        "the same keyframe encoded differently twice ({same:.3e}); the conditioning seed is not \
         fixed, so the difference measured above is not attributable to the image"
    );
}

/// **Claim 2: the conditioning latent reaches the model's input.**
///
/// Runs the real joint loop with a recording model and asserts the video row block it was handed
/// begins with exactly the conditioning rows. The mutation is a different keyframe: what the model
/// sees must change with it.
#[test]
fn the_model_sees_the_keyframe_in_its_video_rows() {
    let anchors = vec![KeyframeAnchor::First];
    let l = layout(&anchors, 4);
    let n = l.num_condition_video_rows();
    assert!(n > 0);

    let generated = mlx_rs::ops::zeros_dtype(
        &[1, l.generated_video_rows().len() as i32, 96],
        Dtype::Float32,
    )
    .unwrap();

    let cond_a = condition_rows(&anchors, &[0]).unwrap();
    let rows_a = prepend_condition_rows(&l, Some(&cond_a), &generated).unwrap();
    let (seen_a, _) = run(&l, &rows_a);

    // The leading rows of what the model saw ARE the conditioning block.
    let lead_a = mlx_gen_minimax_h3::tensor::slice_axis(&seen_a, 1, 0, n).unwrap();
    let (residual, _) = rel(&lead_a, &cond_a);
    assert!(
        residual < 1e-6,
        "the model's leading video rows are not the conditioning block ({residual:.3e})"
    );

    // …and the rest is the generated block, which here is all zeros — so the conditioning is not
    // merely present, it is the ONLY non-zero part of the video stream.
    let tail = mlx_gen_minimax_h3::tensor::slice_axis(&seen_a, 1, n, seen_a.shape()[1]).unwrap();
    let tail_max: f32 = tail.abs().unwrap().max(None).unwrap().item();
    assert_eq!(tail_max, 0.0, "the generated tail should still be zero");
    let lead_max: f32 = lead_a.abs().unwrap().max(None).unwrap().item();
    assert!(
        lead_max > 1e-3,
        "the conditioning rows the model saw are ~zero ({lead_max:.3e}) — indistinguishable from \
         no conditioning at all"
    );

    // **The mutation.** A different keyframe must change what the model sees.
    let cond_b = condition_rows(&anchors, &[1]).unwrap();
    let rows_b = prepend_condition_rows(&l, Some(&cond_b), &generated).unwrap();
    let (seen_b, _) = run(&l, &rows_b);
    let (delta, _) = rel(&seen_a, &seen_b);
    println!("model input changed by peak-rel {delta:.3e} when the keyframe changed");
    assert!(
        delta > MUTATION_FLOOR,
        "swapping the keyframe changed the model's input by {delta:.3e} — the image is not \
         reaching the DiT"
    );

    // **The negative control.** With no keyframe the model sees no conditioning rows at all, so a
    // layout/latent pair that forgot the anchors is a different shape, not a subtle difference.
    let l0 = layout(&[], 4);
    let g0 = mlx_rs::ops::zeros_dtype(
        &[1, l0.generated_video_rows().len() as i32, 96],
        Dtype::Float32,
    )
    .unwrap();
    let (seen0, _) = run(&l0, &g0);
    assert_eq!(l0.num_condition_video_rows(), 0);
    assert_eq!(
        seen0.shape()[1] + n,
        seen_a.shape()[1],
        "the keyframe run must carry exactly {n} more video rows than the t2va one"
    );
}

/// **Claim 3: the anchors survive the loop untouched.**
///
/// `video_indices` is the conditioning rows *followed by* the target rows, and the scheduler writes
/// only the tail. If it wrote the whole block, the anchors would be denoised away over the run —
/// the conditioning would be destroyed by the very loop that is supposed to honour it, and the
/// output would still be plausible video.
#[test]
fn the_anchors_are_never_denoised() {
    let anchors = vec![KeyframeAnchor::First, KeyframeAnchor::Last];
    let l = layout(&anchors, 4);
    let n = l.num_condition_video_rows();
    let cond = condition_rows(&anchors, &[0, 1]).unwrap();

    // A NON-zero generated block, so the loop has something real to step and the anchors are not
    // trivially preserved by everything being zero.
    let generated = mlx_rs::random::normal::<f32>(
        &[1, l.generated_video_rows().len() as i32, 96],
        None,
        None,
        Some(&mlx_rs::random::key(7).unwrap()),
    )
    .unwrap();
    let rows = prepend_condition_rows(&l, Some(&cond), &generated).unwrap();
    let (_, final_rows) = run(&l, &rows);

    let before = mlx_gen_minimax_h3::tensor::slice_axis(&rows, 1, 0, n).unwrap();
    let after = mlx_gen_minimax_h3::tensor::slice_axis(&final_rows, 1, 0, n).unwrap();
    let (drift, _) = rel(&after, &before);
    println!("anchor drift across the whole loop: peak-rel {drift:.3e}");
    assert!(
        drift < 1e-6,
        "the keyframe anchors moved by {drift:.3e} across the loop; they must ride through \
         unchanged"
    );

    // The generated tail, meanwhile, DID move — otherwise "unchanged" would be a property of the
    // loop doing nothing rather than of the anchors being excluded.
    let tail_before = mlx_gen_minimax_h3::tensor::slice_axis(&rows, 1, n, rows.shape()[1]).unwrap();
    let tail_after =
        mlx_gen_minimax_h3::tensor::slice_axis(&final_rows, 1, n, final_rows.shape()[1]).unwrap();
    let (moved, _) = rel(&tail_after, &tail_before);
    println!("generated tail moved by peak-rel {moved:.3e}");
    assert!(
        moved > MUTATION_FLOOR,
        "the generated rows did not move either ({moved:.3e}); this test cannot distinguish \
         'anchors excluded' from 'the loop is inert'"
    );

    // The anchors sit in their own row class, at their own timestep.
    let classes = l.row_classes().as_slice::<i32>().to_vec();
    for &row in &l.video_indices()[..n as usize] {
        assert_eq!(classes[row as usize], RowClass::ConditionVideo as i32);
    }
    for &row in l.generated_video_rows() {
        assert_eq!(classes[row as usize], RowClass::Video as i32);
    }
    let t = PackedLayout::row_timesteps(0.3, 0.6);
    assert_eq!(
        t[RowClass::ConditionVideo as usize],
        mlx_gen_minimax_h3::KEYFRAME_NOISE_AUG,
        "the anchors are held at 0.999, not at the video timestep"
    );
}

// ---------------------------------------------------------------------------------------------
// 3. Geometry — the keyframe-derived canvas on the stride-32 lattice
// ---------------------------------------------------------------------------------------------

/// **A keyframe-derived canvas is always a legal render canvas.**
///
/// The canvas comes from the keyframe's aspect ratio, and it has to land on the same stride-32
/// lattice `resolve_geometry` enforces — otherwise a keyframe would imply a request the engine then
/// refuses. Checked over a spread of real-world aspect ratios rather than the three in the
/// reference's own table.
#[test]
fn keyframe_derived_canvases_are_on_the_stride_32_lattice() {
    let ratios: [(u32, u32); 12] = [
        (1920, 1080),
        (1080, 1920),
        (512, 512),
        (3840, 2160),
        (1280, 720),
        (720, 1280),
        (2048, 512),
        (512, 2048),
        (1024, 768),
        (768, 1024),
        (832, 1216),
        (1000, 999),
    ];
    for (w, h) in ratios {
        let (cw, ch) = resolve_keyframe_canvas(w, h).unwrap();
        assert_eq!(cw % SPATIAL_STRIDE, 0, "{w}x{h} -> width {cw}");
        assert_eq!(ch % SPATIAL_STRIDE, 0, "{w}x{h} -> height {ch}");
        // …and it really is a canvas the engine will render: same gate, not a re-derivation.
        let g = resolve_geometry(cw, ch, SMALLEST_LEGAL_FRAMES)
            .unwrap_or_else(|e| panic!("{w}x{h} resolved to {cw}x{ch}, which is illegal: {e}"));
        assert_eq!(g.width, cw);
        assert_eq!(g.height, ch);
        println!("{w}x{h} -> {cw}x{ch}");
    }
}

/// Off-lattice geometry is still **rejected, not refitted** — the keyframe path did not weaken it.
///
/// This is the regression that would be easiest to introduce here: a keyframe implies a canvas, and
/// the tempting shortcut is to snap the request to it. `resolve_geometry` must keep refusing.
#[test]
fn the_keyframe_path_does_not_soften_the_geometry_gate() {
    // A canvas one pixel off the stride.
    assert!(resolve_geometry(1345, 768, SMALLEST_LEGAL_FRAMES).is_err());
    assert!(resolve_geometry(1344, 769, SMALLEST_LEGAL_FRAMES).is_err());
    // 16-aligned but not 32-aligned — the case `SIZE_MULTIPLE` exists for.
    assert!(resolve_geometry(1360, 768, SMALLEST_LEGAL_FRAMES).is_err());
    // An off-lattice frame count.
    assert!(resolve_geometry(1344, 768, 125).is_err());
    assert!(resolve_geometry(1344, 768, 123).is_err());
    // And a ratio outside 1:4..4:1 has no canvas at all rather than a clamped one.
    assert!(resolve_keyframe_canvas(5000, 1000).is_err());
    assert!(resolve_keyframe_canvas(1000, 5000).is_err());
    // The boundary is legal, so the rejection is a bound and not a blanket refusal.
    resolve_keyframe_canvas(4000, 1000).unwrap();
}

/// A keyframe is fitted to the request's canvas before **either** path sees it, so the vision tower
/// and the VAE encode cannot disagree about what the image is.
#[test]
fn keyframes_are_fitted_to_the_canvas_before_encoding() {
    let odd = probe_image(37, 0);
    let refs = [&odd];
    let fitted = fit_keyframes(&refs, KEYFRAME_EDGE, KEYFRAME_EDGE).unwrap();
    assert_eq!(fitted.len(), 1);
    assert_eq!(fitted[0].width, KEYFRAME_EDGE);
    assert_eq!(fitted[0].height, KEYFRAME_EDGE);

    // …and the VAE consumes exactly that, as a single frame.
    let pixels = keyframe_to_vae_pixels(&fitted[0]).unwrap();
    assert_eq!(
        pixels.shape(),
        &[1, 3, 1, KEYFRAME_EDGE as i32, KEYFRAME_EDGE as i32]
    );
    let latent = vae().encode(&pixels).unwrap();
    assert_eq!(
        latent.mean().shape()[2],
        1,
        "one keyframe, one latent frame"
    );
    assert_eq!(
        latent.mean().shape()[3],
        KEYFRAME_EDGE as i32 / encode_fixture_config(3).patch_size
    );
}

/// The vision-block rows a keyframe contributes to the PROMPT stream are tagged **video**, not
/// text — which is what makes them address a different block of the AdaLN modulation table.
#[test]
fn vision_rows_are_tagged_video_in_the_prompt_stream() {
    // A presentation of 3 label rows, 5 vision rows, 4 prompt rows.
    let mut tags = vec![TEXT_TAG; 3];
    tags.extend(vec![VIDEO_TAG; 5]);
    tags.extend(vec![TEXT_TAG; 4]);
    let l = PackedLayout::build(geometry(), PATCH, &tags, 2, &[KeyframeAnchor::First]).unwrap();

    let got = l.token_tags().as_slice::<i32>().to_vec();
    for (i, &row) in l.text_indices().iter().enumerate() {
        assert_eq!(
            got[row as usize], tags[i],
            "text row {i} lost its modality tag"
        );
    }
    // The vision rows really are tagged differently from the label rows around them.
    assert_eq!(got[l.text_indices()[0] as usize], TEXT_TAG);
    assert_eq!(got[l.text_indices()[4] as usize], VIDEO_TAG);
    assert_ne!(TEXT_TAG, VIDEO_TAG);
}
