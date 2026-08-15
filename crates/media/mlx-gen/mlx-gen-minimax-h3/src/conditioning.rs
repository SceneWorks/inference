//! **`fl2va` keyframe conditioning** (sc-17148): turning prepared keyframe pixels into the
//! conditioning rows that lead the packed video stream.
//!
//! ```text
//! keyframe pixels ─► VAE encode ─► posterior ─► sample(seed 42) ─► fp16 ─► normalize
//!                                                                            │
//!                                       ┌────────────────────────────────────┘
//!                                       ▼
//!                        scale_noise(t = 0.999) ─► patchify ─► condition rows
//! ```
//!
//! Everything here is a policy the checkpoint was trained under and none of it is in any shipped
//! config file. All four constants below are `MiniMaxH3ModularPipeline` **properties** on
//! `diffusers@main`, invisible to anything reading `vae/config.json` or the tensor index.
//!
//! # The anchor is noised, and holding it clean is off-distribution
//!
//! [`KEYFRAME_NOISE_AUG`] is `0.999`, not `1.0`. In MiniMax's convention `t = 1` is **clean**, so
//! the anchor is `0.999 · latent + 0.001 · noise` — very slightly noised, because that is how the
//! released model was trained. Conditioning on an exactly clean anchor is the obvious
//! implementation and it is wrong.
//!
//! The same number reappears as the anchors' per-row timestep, `max(video_t, 0.999)`
//! ([`crate::denoise::PackedLayout::row_timesteps`]), so the DiT is told the anchors sit at the
//! same `t` they were actually mixed to.
//!
//! # The posterior seed is 42, and it is NOT the request seed
//!
//! [`KEYFRAME_ENCODE_SEED`] is fixed. The reference builds a fresh CPU generator per condition, so
//! the same keyframe always encodes to the same anchor **regardless of the request's seed**, and
//! the request generator is not advanced by it. A port that draws this from the request RNG makes
//! conditioning vary per seed — plausible-looking, and never reproducible against the reference.
//!
//! [`KeyframeNoise`] makes that choice explicit rather than implicit in a `random::key` call: this
//! crate's RNG is MLX's, not torch's, so the *stream* cannot match the reference bit for bit
//! anyway. What can and must match is the **structure** — a dedicated, request-independent stream
//! at a pinned seed — and [`KeyframeNoise::Provided`] exists so parity tests can inject the
//! reference's own draw and take the RNG out of the comparison entirely.
//!
//! # The fp16 round-trip is load-bearing
//!
//! The reference does `latents.to(torch.float16).float()` **before** normalizing. That is a
//! deliberate quantization to ~11 mantissa bits, not an artifact of a dtype it happened to be in;
//! skipping it changes the anchor. [`fp16_round_trip`] is it.

use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

use crate::denoise::schedule::KEYFRAME_NOISE_AUG;
use crate::dit::positions::KeyframeAnchor;
use crate::pipeline::patchify_video_latents;
use crate::vae::MiniMaxH3VideoVae;
use crate::vae_encoder::DiagonalGaussian;

/// The **fixed** seed the keyframe posterior is sampled under, independent of the request seed.
pub const KEYFRAME_ENCODE_SEED: u64 = 42;

/// Re-exported so a caller conditioning a keyframe does not have to reach into the scheduler for
/// the `t` the anchor is held at. See the module docs.
pub const KEYFRAME_NOISE_AUG_T: f32 = KEYFRAME_NOISE_AUG;

/// Quantize through fp16 and back, as the reference does before normalizing a conditioning latent.
pub fn fp16_round_trip(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float16)?.as_dtype(Dtype::Float32)?)
}

/// Where the noise for a keyframe's posterior sample and its augmentation comes from.
///
/// Two variants because the two use cases have genuinely different requirements: production wants
/// a deterministic, request-independent draw, and parity wants the reference's exact tensor.
#[derive(Debug, Clone)]
pub enum KeyframeNoise {
    /// Draw from a dedicated MLX stream keyed on [`KEYFRAME_ENCODE_SEED`] plus the keyframe's
    /// index, so two keyframes in one request do not share a draw and **no** draw is taken from
    /// the request's generator.
    Seeded,
    /// Use the supplied tensors verbatim: `(posterior_noise, augmentation_noise)`. Parity tests
    /// inject the reference's own draws so the comparison measures the arithmetic rather than the
    /// RNG.
    Provided(Box<(Array, Array)>),
}

impl KeyframeNoise {
    fn draw(&self, shape: &[i32], index: usize) -> Result<(Array, Array)> {
        match self {
            Self::Seeded => {
                // Two independent sub-streams per keyframe: the posterior sample and the
                // augmentation are separate draws in the reference too.
                let base = KEYFRAME_ENCODE_SEED.wrapping_add((index as u64).wrapping_mul(2));
                let k0 = mlx_rs::random::key(base)?;
                let k1 = mlx_rs::random::key(base.wrapping_add(1))?;
                Ok((
                    mlx_rs::random::normal::<f32>(shape, None, None, Some(&k0))?,
                    mlx_rs::random::normal::<f32>(shape, None, None, Some(&k1))?,
                ))
            }
            Self::Provided(pair) => {
                let (posterior, aug) = pair.as_ref();
                for (what, t) in [("posterior", posterior), ("augmentation", aug)] {
                    if t.shape() != shape {
                        return Err(Error::Msg(format!(
                            "minimax-h3 conditioning: provided {what} noise is {:?}, expected \
                             {shape:?}",
                            t.shape()
                        )));
                    }
                }
                Ok((posterior.clone(), aug.clone()))
            }
        }
    }
}

/// Mix a clean latent towards noise at `timestep`, in MiniMax's convention where **`t = 1` is
/// clean**: `t · sample + (1 - t) · noise`.
///
/// This is `MiniMaxH3Scheduler.scale_noise`, and the convention is the opposite of the usual
/// flow-matching one. It is applied at face value — `timestep` is **not** looked up in the
/// schedule's `timesteps`.
pub fn scale_noise(sample: &Array, timestep: f32, noise: &Array) -> Result<Array> {
    if sample.shape() != noise.shape() {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: noise {:?} does not match the sample {:?}",
            noise.shape(),
            sample.shape()
        )));
    }
    let t = Array::from_f32(timestep);
    let one_minus = Array::from_f32(1.0 - timestep);
    Ok(sample.multiply(&t)?.add(&noise.multiply(&one_minus)?)?)
}

/// One keyframe's conditioning latent: encode, sample, fp16, normalize.
///
/// `pixels` is `[1, 3, 1, H, W]` from [`crate::keyframe::keyframe_to_vae_pixels`]. The result is
/// `[1, latent_channels, 1, H/16, W/16]`, normalized by the VAE's per-channel statistics.
pub fn encode_keyframe_condition(
    vae: &MiniMaxH3VideoVae,
    pixels: &Array,
    noise: &KeyframeNoise,
    index: usize,
) -> Result<Array> {
    let s = pixels.shape();
    if s.len() != 5 || s[2] != 1 {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: a keyframe is a SINGLE frame, [1, 3, 1, H, W]; got {s:?}"
        )));
    }
    let posterior: DiagonalGaussian = vae.encode(pixels)?;
    let (sample_noise, _) = noise.draw(posterior.mean().shape(), index)?;
    let sampled = posterior.sample_with(&sample_noise)?;
    vae.normalize(&fp16_round_trip(&sampled)?)
}

/// The rows one keyframe contributes to the packed video stream: noise-augmented to
/// [`KEYFRAME_NOISE_AUG_T`], then patchified.
///
/// Mixing **before** the patchify is the same arithmetic as after, since patchify only permutes —
/// the reference does it in this order and so does this.
pub fn keyframe_condition_rows(
    condition: &Array,
    patch: [i32; 3],
    noise: &KeyframeNoise,
    index: usize,
) -> Result<Array> {
    let (_, aug) = noise.draw(condition.shape(), index)?;
    let noised = scale_noise(condition, KEYFRAME_NOISE_AUG_T, &aug)?;
    patchify_video_latents(&noised, patch)
}

/// Check that a keyframe set is one of the four legal `fl2va` shapes.
///
/// Returns `None` for the zero-keyframe (`t2va`) shape and `Some(n)` for the rest. Split out of
/// [`build_condition_rows`] so the arity rules are testable without loading a VAE — a test that
/// re-implements them would prove nothing about the code that runs.
///
/// `anchors` and `pixels` are **positional**; letting them diverge mislabels which end of the clip
/// an image anchors, and both orders produce a runnable sequence.
pub fn validate_condition_arity(
    num_pixels: usize,
    anchors: &[KeyframeAnchor],
) -> Result<Option<usize>> {
    if anchors.len() != num_pixels {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: {} anchors for {num_pixels} keyframes — they are positional",
            anchors.len()
        )));
    }
    if num_pixels == 0 {
        return Ok(None);
    }
    if num_pixels > 2 {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: fl2va accepts at most a first and a last frame, got \
             {num_pixels}"
        )));
    }
    // The same end cannot be anchored twice: `[First, First]` is arity-legal and semantically
    // impossible, and it would write two anchor rows at the same rotary time.
    if anchors.len() == 2 && anchors[0] == anchors[1] {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: both keyframes anchor {:?}; fl2va's two slots are the FIRST \
             and the LAST frame",
            anchors[0]
        )));
    }
    if anchors.len() == 2 && anchors[0] != KeyframeAnchor::First {
        return Err(Error::Msg(
            "minimax-h3 conditioning: with two keyframes the first must be the FIRST-frame \
             anchor; packed order is (first, last)"
                .into(),
        ));
    }
    Ok(Some(num_pixels))
}

/// Every keyframe's conditioning rows, concatenated in packed order — the block that **leads** the
/// video row stream.
///
/// `anchors` and `pixels` are positional and must agree: `anchors[i]` describes `pixels[i]`. The
/// reference builds both from the same ordered `(("first", image), ("last", last_image))` filter,
/// so a caller that lets them diverge has mislabelled which end of the clip an image anchors —
/// which is silent, because both orders produce a runnable sequence.
pub fn build_condition_rows(
    vae: &MiniMaxH3VideoVae,
    pixels: &[Array],
    anchors: &[KeyframeAnchor],
    patch: [i32; 3],
    noise: &KeyframeNoise,
) -> Result<Option<Array>> {
    if validate_condition_arity(pixels.len(), anchors)?.is_none() {
        return Ok(None);
    }
    let mut rows = Vec::with_capacity(pixels.len());
    for (i, p) in pixels.iter().enumerate() {
        let condition = encode_keyframe_condition(vae, p, noise, i)?;
        rows.push(keyframe_condition_rows(&condition, patch, noise, i)?);
    }
    // Row axis 1: `patchify_video_latents` emits `[1, rows, features]`.
    Ok(Some(mlx_rs::ops::concatenate_axis(&rows, 1)?))
}

/// A `ref2va` reference clip's pixels in the video VAE's space — `[1, 3, F, H, W]`.
///
/// The multi-frame sibling of [`crate::keyframe::keyframe_to_vae_pixels`], and it applies the same
/// `x/255` + ImageNet normalization, because a reference takes the same two preprocessing paths a
/// keyframe does (the VAE's here, the vision tower's `0.5/0.5` separately).
///
/// **Every frame must share one size.** A ragged clip would encode to a latent whose geometry does
/// not match what the layout reserved, and the mismatch first surfaces as a shape error deep inside
/// the DiT.
pub fn reference_clip_to_vae_pixels(frames: &[mlx_gen::media::Image]) -> Result<Array> {
    let Some(first) = frames.first() else {
        return Err(Error::Msg(
            "minimax-h3 conditioning: a reference clip has no frames".into(),
        ));
    };
    let (w, h) = (first.width as i32, first.height as i32);
    let mut per_frame = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        if f.width as i32 != w || f.height as i32 != h {
            return Err(Error::Msg(format!(
                "minimax-h3 conditioning: reference clip frame {i} is {}x{}, but frame 0 is {w}x{h} \
                 — every frame of a clip shares one size",
                f.width, f.height
            )));
        }
        // `[1, 3, 1, H, W]` each; concatenating on the temporal axis gives `[1, 3, F, H, W]`.
        per_frame.push(crate::keyframe::keyframe_to_vae_pixels(f)?);
    }
    Ok(mlx_rs::ops::concatenate_axis(&per_frame, 2)?)
}

/// One `ref2va` reference's conditioning latent: encode, sample, fp16, normalize.
///
/// The multi-frame sibling of [`encode_keyframe_condition`] and deliberately the *same* arithmetic
/// — the reference implementation calls one `encode_vae_condition` for a keyframe, a reference image
/// and a reference clip alike, so this differs from the keyframe entry point only in accepting more
/// than one frame.
///
/// `pixels` is `[1, 3, F, H, W]`. The result is `[1, latent_channels, F', H/16, W/16]` at **this
/// reference's own resolution** — references are not fitted to the generated canvas.
pub fn encode_reference_condition(
    vae: &MiniMaxH3VideoVae,
    pixels: &Array,
    noise: &KeyframeNoise,
    index: usize,
) -> Result<Array> {
    let s = pixels.shape();
    if s.len() != 5 || s[0] != 1 || s[1] != 3 || s[2] < 1 {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: a reference is [1, 3, F, H, W] with F >= 1, got {s:?}"
        )));
    }
    let posterior: DiagonalGaussian = vae.encode(pixels)?;
    let (sample_noise, _) = noise.draw(posterior.mean().shape(), index)?;
    let sampled = posterior.sample_with(&sample_noise)?;
    vae.normalize(&fp16_round_trip(&sampled)?)
}

/// Snap a reference clip's frame count **down** to the `17n + 5` lattice the video VAE encodes
/// without padding.
///
/// Down, not up: padding a short reference would condition on frames that do not exist. This only
/// bites when the reference is shorter than the target, whose own count already has that form.
///
/// The floor is `n = 1`, i.e. **22** frames — not `5`. `n = 0` would ask the VAE for a chunkless
/// clip, so the `max(1)` below is a floor rather than a clamp to the lattice's zeroth point, and the
/// caller pairs this with `.min(frames.len())` so a shorter clip is never over-read.
pub fn snap_reference_frames_down(num_frames: usize) -> usize {
    // `17n + 5` in the VAE's own terms: `frames_per_chunk = 17`, `latents_per_chunk = 5`.
    const FRAMES_PER_CHUNK: usize = 17;
    const LATENTS_PER_CHUNK: usize = 5;
    let n = num_frames.saturating_sub(LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK;
    n.max(1) * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK
}

/// The **clean** reference-soundtrack rows of one encoded soundtrack, reshaped channel-major.
///
/// `normalized` is the audio VAE posterior's **mean**, already normalized by the component's
/// `latents_mean` / `latents_std`, shaped `[channels, num_audio_latents, audio_latent_channels]`.
/// The result is `[1, channels · num_audio_latents, audio_latent_channels]`.
///
/// # Two things here are deliberate and neither is obvious
///
/// * **The posterior is never sampled.** A soundtrack takes the distribution's *mode*, unlike a
///   visual condition which draws a sample at [`KEYFRAME_ENCODE_SEED`]. Sampling it would make a
///   reference soundtrack vary per call for no reason the model was trained with.
/// * **The rows ride clean, at `t = 1.0`.** They are *not* noise-augmented to
///   [`KEYFRAME_NOISE_AUG_T`] the way visual anchors are — that is what
///   [`crate::denoise::RowClass::ConditionAudio`] exists to express, and applying the visual
///   augmentation here would be the plausible wrong answer.
pub fn reference_audio_rows(normalized: &Array) -> Result<Array> {
    let s = normalized.shape();
    if s.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 conditioning: reference audio latents are [channels, latents, features], \
             got {s:?}"
        )));
    }
    Ok(normalized.reshape(&[1, s[0] * s[1], s[2]])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four constants, pinned. None of them is in any shipped config file, so nothing else
    /// can catch a drift.
    #[test]
    fn the_conditioning_constants_are_the_references() {
        assert_eq!(KEYFRAME_ENCODE_SEED, 42);
        assert_eq!(KEYFRAME_NOISE_AUG_T, 0.999);
        // `t = 1` is CLEAN in this convention, so the augmentation is a 0.1% noise admixture and
        // NOT a 99.9% one. A port that reads the convention backwards destroys the anchor.
        const {
            assert!(
                KEYFRAME_NOISE_AUG_T > 0.99,
                "0.999 is nearly-clean, not nearly-noise"
            )
        };
    }

    /// `scale_noise` is a lerp in the `t = 1 is clean` convention.
    #[test]
    fn scale_noise_lerps_towards_noise_as_t_falls() {
        let x = Array::from_slice(&[10.0f32, 20.0], &[2]);
        let n = Array::from_slice(&[0.0f32, 0.0], &[2]);
        // t = 1 is a no-op: the sample is clean.
        assert_eq!(
            scale_noise(&x, 1.0, &n).unwrap().as_slice::<f32>(),
            &[10.0, 20.0]
        );
        // t = 0 is pure noise.
        assert_eq!(
            scale_noise(&x, 0.0, &n).unwrap().as_slice::<f32>(),
            &[0.0, 0.0]
        );
        // t = 0.999 keeps 99.9% of the signal.
        let aug = scale_noise(&x, KEYFRAME_NOISE_AUG_T, &n).unwrap();
        let v = aug.as_slice::<f32>();
        assert!((v[0] - 9.99).abs() < 1e-4, "{v:?}");
        // …and it is NOT a no-op, which is the whole point of the constant.
        assert!(
            (v[0] - 10.0).abs() > 1e-4,
            "0.999 must actually move the anchor"
        );
        assert!(scale_noise(&x, 0.5, &Array::from_slice(&[0.0f32], &[1])).is_err());
    }

    /// The fp16 round-trip really quantizes — a value needing more than 11 mantissa bits moves.
    #[test]
    fn the_fp16_round_trip_is_not_a_no_op() {
        let x = Array::from_slice(&[1.000_061_f32, 0.1, 65_600.0], &[3]);
        let r = fp16_round_trip(&x).unwrap();
        let v = r.as_slice::<f32>();
        assert_ne!(v[0], 1.000_061, "fp16 cannot represent this exactly");
        assert!((v[1] - 0.1).abs() > 0.0, "0.1 is not exact in fp16 either");
        assert!(v[2].is_infinite(), "65600 overflows fp16's range");
        // A value fp16 CAN hold is unchanged, so the round-trip is not lossy noise.
        let exact = Array::from_slice(&[0.5f32, 2.0, -8.0], &[3]);
        assert_eq!(
            fp16_round_trip(&exact).unwrap().as_slice::<f32>(),
            &[0.5, 2.0, -8.0]
        );
    }

    /// The seeded stream is deterministic, independent of any request seed, and **different per
    /// keyframe** — two anchors must not share a draw.
    #[test]
    fn the_keyframe_noise_stream_is_fixed_and_per_index() {
        let shape = [1, 4, 1, 2, 2];
        let (a0, b0) = KeyframeNoise::Seeded.draw(&shape, 0).unwrap();
        let (a0b, _) = KeyframeNoise::Seeded.draw(&shape, 0).unwrap();
        assert_eq!(
            a0.as_slice::<f32>(),
            a0b.as_slice::<f32>(),
            "the same index must always draw the same tensor"
        );
        // The posterior and augmentation draws are independent.
        assert_ne!(a0.as_slice::<f32>(), b0.as_slice::<f32>());
        // …and keyframe 1 differs from keyframe 0.
        let (a1, _) = KeyframeNoise::Seeded.draw(&shape, 1).unwrap();
        assert_ne!(
            a0.as_slice::<f32>(),
            a1.as_slice::<f32>(),
            "two keyframes must not share a posterior draw"
        );

        // Provided noise is passed through verbatim, and a mismatched shape is rejected rather
        // than broadcast.
        let ones = Array::from_slice(&[1.0f32; 16], &shape);
        let zeros = Array::from_slice(&[0.0f32; 16], &shape);
        let provided = KeyframeNoise::Provided(Box::new((ones.clone(), zeros.clone())));
        let (p, q) = provided.draw(&shape, 0).unwrap();
        assert_eq!(p.as_slice::<f32>(), ones.as_slice::<f32>());
        assert_eq!(q.as_slice::<f32>(), zeros.as_slice::<f32>());
        assert!(provided.draw(&[1, 4, 1, 2, 3], 0).is_err());
    }

    /// **All four legal shapes are accepted and everything else is rejected.** This is the
    /// function `build_condition_rows` actually calls, not a restatement of it.
    #[test]
    fn the_four_conditioning_shapes_are_the_only_legal_ones() {
        use KeyframeAnchor::{First, Last};
        // 0 images — `t2va`, no conditioning rows at all.
        assert_eq!(validate_condition_arity(0, &[]).unwrap(), None);
        // first-only, last-only, first+last.
        assert_eq!(validate_condition_arity(1, &[First]).unwrap(), Some(1));
        assert_eq!(validate_condition_arity(1, &[Last]).unwrap(), Some(1));
        assert_eq!(
            validate_condition_arity(2, &[First, Last]).unwrap(),
            Some(2)
        );

        // Arity mismatches.
        assert!(validate_condition_arity(1, &[]).is_err());
        assert!(validate_condition_arity(0, &[First]).is_err());
        assert!(validate_condition_arity(2, &[First]).is_err());
        // More than two keyframes is not a shape fl2va has.
        assert!(validate_condition_arity(3, &[First, Last, First]).is_err());
        // The same end twice is arity-legal and semantically impossible — two anchor rows at the
        // identical rotary time.
        assert!(validate_condition_arity(2, &[First, First]).is_err());
        assert!(validate_condition_arity(2, &[Last, Last]).is_err());
        // Reversed packed order is rejected rather than silently mislabelling the two ends.
        assert!(validate_condition_arity(2, &[Last, First]).is_err());
    }
}
