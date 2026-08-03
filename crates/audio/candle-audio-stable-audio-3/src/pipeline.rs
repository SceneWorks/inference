//! Connected Stable Audio 3 inference pipeline, shared by every registered variant.

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};

use crate::config::{ConditionerConfig, DiffusionObjective, ModelConfig};
use crate::dit::{Guidance, StableAudio3Dit};
use crate::model::VariantShape;
use crate::same::{
    SameAutoencoder, SameChunkingParameters, SameChunkingPolicy, SameDecodeChunkNoise,
};
use crate::sampler::{
    build_schedule, default_sample_geometry, inference_shift, padding_mask,
    sample_dit_initialized_with_interval_and_cancel, GuidanceInterval, InjectedNoise, NoiseSource,
    ProgressCallback, SampleGeometry, SamplerKind, SeededNoise, LATENT_DOWNSAMPLING,
};
use crate::t5gemma::T5GemmaConditioner;
use crate::weights::{SnapshotKind, SnapshotLayout};

pub const SAMPLE_RATE: u32 = 44_100;
pub const CHANNELS: usize = 2;
pub const LATENT_CHANNELS: usize = 256;

#[derive(Clone, Copy)]
enum ReferenceInputValidation {
    Required,
    Complete,
}

/// The upstream API default step count, which is also the post-trained checkpoints' own
/// `training.demo.demo_steps`.
pub const DEFAULT_STEPS: usize = 8;
/// The upstream API default guidance, which is also the post-trained checkpoints' own
/// `training.demo.demo_cfg_scales` (`[1]`). At `1.0` the DiT takes the batch-1 branch and a negative
/// prompt has no effect at all — see [`crate::dit::StableAudio3Dit::forward_guided`].
pub const DEFAULT_GUIDANCE: f64 = 1.0;

/// The `-base` checkpoints' default step count (sc-14546).
///
/// **This is a deliberate product choice, not an upstream API default.** Upstream's Python and CLI
/// entry points default to [`DEFAULT_STEPS`] / [`DEFAULT_GUIDANCE`] for every checkpoint; only
/// Stability's Gradio app varies them per model. The number is not invented here either: each base
/// `model_config.json` ships `training.demo.demo_steps = 50` and
/// `training.demo.demo_cfg_scales = [2, 4, 7]`, against `8` and `[1]` in all three post-trained
/// configs, so the checkpoints themselves declare the operating point they were demoed at.
/// `tests/base_guidance.rs` reads those two fields straight out of the pinned snapshots and
/// asserts they still agree with these constants.
///
/// The cost is real and is not hidden: at [`BASE_DEFAULT_GUIDANCE`] every step is a batch-2 CFG
/// forward, so a default base render is `50 x 2 = 100` DiT forwards against the post-trained
/// default's `8 x 1 = 8` — 12.5x the example-work per second of audio.
pub const BASE_DEFAULT_STEPS: usize = 50;

/// The `-base` checkpoints' default guidance (sc-14546).
///
/// The largest of the base configs' own `training.demo.demo_cfg_scales`, and the value Stability's
/// Gradio app presents for these checkpoints. See [`BASE_DEFAULT_STEPS`] for why this is a product
/// choice rather than an upstream default.
pub const BASE_DEFAULT_GUIDANCE: f64 = 7.0;

/// What the strict wrapper authenticates a snapshot against: one variant's architecture bound to
/// the identity fields its own `model_config.json` declares.
///
/// # Provenance and gate value are different fields, deliberately
///
/// [`Self::hub_repo`] is **provenance**: the repository the checkpoint was published from. It feeds
/// error messages, the weight-license source URLs, and the CI snapshot manifest.
///
/// [`Self::expected_conditioner_repo`] is the **gate value**: the `repo_id` the snapshot's own
/// conditioner config declares. For the three post-trained checkpoints the two coincide, which is
/// why sc-14544/sc-14545 could carry a single field. They do **not** coincide for the `-base`
/// checkpoints: every base `model_config.json` declares its *post-trained sibling's* repository
/// (`stable-audio-3-small-music-base` declares `stabilityai/stable-audio-3-small-music`). Collapsing
/// the two back into one field rejects every base snapshot ever provisioned, and "fixing" that by
/// dropping the check would open all three post-trained ids to their base siblings.
///
/// The declared `repo_id` is never resolved over the network. It is compared as a string and
/// nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantGeometry {
    pub shape: VariantShape,
    /// Provenance only: the repository that published this checkpoint.
    pub hub_repo: &'static str,
    /// The conditioner `repo_id` this checkpoint's `model_config.json` must declare.
    pub expected_conditioner_repo: &'static str,
    /// The `diffusion_objective` this checkpoint must declare.
    ///
    /// This is the **only universal** base/post-trained discriminator: medium and medium-base agree
    /// on tensor inventory, `sample_size`, root byte length *and* conditioner `repo_id`, and differ
    /// on nothing else in the entire config.
    pub expected_objective: DiffusionObjective,
}

/// Compute dtypes for one loaded graph.
///
/// # Why this is F32 everywhere, and what was measured to decide that
///
/// Upstream runs `model_half=True` when a CUDA device is selected and forces fp32 otherwise
/// (`stable_audio_3/model.py`). sc-14545 built the machinery to mirror that — the graph now runs
/// end to end at F16, which it could not before — and then measured it, because "mirror upstream"
/// is not evidence that the output survives.
///
/// Measured on Metal at 30 s / 8 steps on `stable_audio_3_medium`, three seeds at each dtype
/// (`tests/dtype_policy.rs`):
///
/// | statistic | F32 range | F16 range |
/// |---|---|---|
/// | rms | 0.057639 … 0.067437 | 0.069209 … 0.075448 |
/// | peak | 0.519312 … 0.619903 | 0.525879 … 0.603027 |
/// | hf emphasis | 0.121821 … 0.150911 | 0.095454 … 0.123614 |
/// | side ratio | 0.502560 … 0.650129 | 0.373487 … 0.951624 |
///
/// F16 is louder on all three seeds, duller on all three, and its stereo image spreads past twice
/// the fp32 envelope. Those are the three signatures of a decoder losing precision. They are *not*
/// conclusive, because F16 does not perturb the sample — at a fixed seed the F16 and F32 waveforms
/// sit at cosine `0.222`, against `0.005` for two F32 renders at adjacent seeds — so half precision
/// selects a different draw and a different draw legitimately has a different brightness. Three
/// seeds cannot separate those two explanations.
///
/// What is not ambiguous is the decision that follows. The only backend fp16 would apply to is
/// CUDA, no CUDA hardware was available to measure the policy on the backend it would ship to, and
/// the one measurement that could be taken points at degradation rather than away from it. Adopting
/// it would also re-open the two already-merged small providers on that backend. So the shipped
/// policy is F32 on every backend, the seam stays typed and tested so the fp16 path cannot rot, and
/// the split policy worth trying next — half the 1.45B DiT, keep the 852M SAME autoencoder at F32,
/// which would isolate whether the dullness comes from the decoder — is filed as its own story with
/// these numbers attached.
///
/// The text side was never part of that decision. sc-14537 pinned BF16 weights on disk and F32
/// compute on every backend. CPU keeps the raw encoder output F32; Metal and CUDA apply one BF16
/// rounding at the raw-embedding boundary before F32 conditioning. `tests/text_oracle.rs` gates
/// that policy against the frozen Transformers 5.8.0 oracle. Switching T5Gemma to BF16 *compute*
/// would move a surface with a numeric parity gate behind it for no memory win worth having: the
/// encoder is 281 MB of the exact 10,443,755,936-byte medium artifact pin set.
///
/// # The F16 path did not load on CUDA at all until sc-14545's second fix cycle
///
/// The first real-weight CUDA run of this seam failed at `load SA3 SAME` with
/// `DriverError(CUDA_ERROR_INVALID_VALUE)` while F32 loaded cleanly. The cause was not the dtype
/// threading: SAME's `bottleneck.noise_scaling_factor` is persisted as an empty `[1, 0, 1]` buffer,
/// and casting a zero-element tensor to F16 launches a kernel with a zero grid dimension, which the
/// driver rejects. The same zero-element buffer broke the Metal embedded-SAME-L lane by a different
/// mechanism. Both are fixed in [`crate::weights`], where the reasoning is recorded in full. Worth
/// stating here because it bounds what "the graph runs end to end at F16" meant before that fix: it
/// was demonstrated on Metal, and on CUDA it was not demonstrated at all.
///
/// # What is pinned F32 independently of `root`, and what is not
///
/// Three things do not follow `root`:
///
/// * The **DiT's block normalizations**. `config.rs` rejects any Stable Audio 3 DiT config whose
///   `norm_kwargs.force_fp32` is not `true`, and [`crate::transformer`]'s `Norm::forward` upcasts
///   to F32 when that flag is set, so the DiT's RMS statistics are F32 at any `root`.
/// * **RoPE**. `inv_freq` is requested at `DType::F32` explicitly rather than inheriting the
///   `VarBuilder` dtype, and the position outer product is evaluated against it in F32.
/// * **The sigma schedule**. `sampler.rs` materializes it in F32 and keeps it F32 for every value
///   handed to the model; only the solver arithmetic runs at the latents' dtype.
///
/// **`same.rs` is not on that list, and that is load-bearing.** The SAME autoencoder builds its
/// blocks with `NormConfig { eps: 1e-3, ..Default::default() }`, whose `force_fp32` is `false`, and
/// the QK norms inside its attention are constructed with `force_fp32: false` as well. Medium's
/// SAME-L config sets `dyt: true`, and `Norm::forward` returns through the `DynamicTanh` branch
/// *before* the `force_fp32` upcast is reached, so DyT would not honour the flag even if it were
/// set. At `root = F16` the whole autoencoder — normalization statistics included — runs in half
/// precision.
///
/// That is the concrete mechanism behind the split policy filed as the follow-up (sc-15151). "Half
/// the DiT, keep the SAME autoencoder at F32" is not merely a memory trade: it is the only
/// arrangement in which SAME's statistics stay F32. Adopting F16 wholesale would mean either
/// accepting half-precision DyT statistics through an 852M-parameter decoder or threading a second
/// dtype into `same.rs`, and neither was measured here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeDTypes {
    /// DiT + SAME + learned conditioner.
    pub root: DType,
    /// T5Gemma compute dtype.
    pub text: DType,
}

impl ComputeDTypes {
    /// Resolve the compute policy for a selected device.
    ///
    /// Keyed on the *selected* device rather than on host CUDA availability. Upstream keys its own
    /// half-cast on whether a CUDA device exists anywhere on the host, which silently half-casts a
    /// CPU run on a CUDA box; that bug is deliberately not reproduced, so this stays a `match` on
    /// the device even while every arm currently agrees.
    pub fn for_device(device: &Device) -> Self {
        Self {
            root: match device {
                Device::Cpu | Device::Metal(_) | Device::Cuda(_) => DType::F32,
            },
            text: DType::F32,
        }
    }
}

/// A caller-supplied source clip for the audio→audio restyle path (sc-14547).
///
/// # `noise_level` is the sampler's orientation, deliberately not the contract's
///
/// This field is upstream's `init_noise_level`: `1.0` replaces the source with pure noise (i.e. the
/// ordinary text-to-audio path) and `0.0` returns the prepared source without a single DiT forward.
/// It is the **complement** of the backend-neutral
/// [`gen_core::Conditioning::ReferenceAudio`](candle_audio::gen_core::Conditioning::ReferenceAudio)
/// `strength`, which this workspace defines as *retention* — higher strength preserves more of the
/// source. The one conversion lives in [`crate::model::reference_noise_level`] so the two
/// same-named parameters can never meet unconverted; a flipped sign here produces a feature that
/// runs, emits plausible audio, and does nothing the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceAudio<'a> {
    /// Interleaved source PCM, `channels` values per frame.
    pub samples: &'a [f32],
    pub sample_rate: u32,
    pub channels: u16,
    /// Init **noise** level in `[0,1]` — see the type docs.
    pub noise_level: f32,
}

/// The sampler `strength` a request implies, reference attached or not (sc-14547).
///
/// # Why this is a function and not two lines inside the synthesis path
///
/// This is the *last* seam on the sign's journey. [`crate::model::reference_noise_level`] converts
/// the contract's retention into an init noise level and [`crate::model::reference_audio_for`]
/// selects the converted field; this decides which value the sampler is actually handed, and it is
/// the point at which the reference and text-only paths converge on one scalar.
///
/// Inside [`StableAudio3Pipeline::synthesize_with_reference_traced`] it was reachable only with
/// multi-gigabyte weights, so `1.0 - reference.noise_level` there — a complete, user-visible sign
/// inversion on all six ids — left the whole weight-free suite green. Extracted here it is gated
/// weight-free by `tests/reference_audio.rs`
/// `the_pipeline_hands_the_sampler_the_converted_noise_level_as_strength`, which feeds it the output
/// of `reference_audio_for` so the whole request → sampler-strength chain is one assertion.
///
/// # Why the synthesis path holds no `strength` binding
///
/// The sampler cross-checks the scalar it is handed against the schedule's first sigma
/// ([`crate::sampler::initialized_start`]). That check only discriminates when the two sides are
/// separate expressions. Two earlier shapes defeated it and were each silent weight-free:
///
/// * forwarding a pre-computed `strength` **argument** into the weights-only helper, and
/// * binding `let strength = sampler_strength_for(reference)` **inside** the helper.
///
/// In both, [`build_schedule`] and the DiT sample call read one expression, so an edit to it moved
/// both sides of the cross-check together and they still agreed. So `StableAudio3Pipeline::sample`
/// now takes `Option<&ReferenceAudio>` and calls this function *inline at each of the two consumer
/// sites*; the text-only replay path passes `None` rather than spelling `1.0` out a second time.
///
/// What that buys, stated exactly and measured rather than reasoned: editing **either one** of those
/// two call sites alone — inverting it, or swapping its `reference` for `None` — leaves it
/// disagreeing with the other, and `initialized_start` bails with `init noise strength must equal
/// every schedule's first sigma`. The only such edits that survive the bail are the ones that do not
/// change the value at that operating point (inverting at noise level `0.5`, `None`-substituting at
/// noise level `1.0`), and those are behaviour-neutral.
///
/// It does **not** stop a coordinated edit to both sites at once, which agrees with itself, passes
/// the bail, and is green weight-free; only the real-weight lane sees that. Editing this function's
/// body instead is caught weight-free by the gate above.
///
/// The `None` arm is the text-to-audio path: `1.0` is pure noise, i.e. no source influence, which is
/// exactly what "no reference" means in the sampler's orientation.
pub fn sampler_strength_for(reference: Option<&ReferenceAudio<'_>>) -> f32 {
    reference.map_or(1.0, |reference| reference.noise_level)
}

/// The two halves of a restyle must be forwarded together into the synthesis path (sc-14547).
///
/// `StableAudio3Pipeline::sample` receives the SAME-encoded source (`init_latents`) and the
/// reference itself as separate arguments. The reference is what
/// [`sampler_strength_for`] reads, so dropping *it* alone at the call site still encodes the source
/// and then discards it under strength `1.0`: a silent degradation to plain text-to-audio. Dropping
/// `init_latents` alone is the converse. Both are one-token edits inside a weights-only method,
/// which is exactly the shape that has been missed before, so the disagreement is rejected here
/// rather than left to be inferred.
///
/// The predicate is a free function so this rejection is exercised without weights by
/// `tests/reference_audio.rs`
/// `the_pipeline_hands_the_sampler_the_converted_noise_level_as_strength`. That gates the rule; it
/// does not gate the call site inside `sample`, which still needs the real-weight lane.
pub fn reference_halves_agree(init_latents_present: bool, reference_present: bool) -> Result<()> {
    if init_latents_present != reference_present {
        return Err(AudioError::Msg(format!(
            "reference-audio init latents and the reference itself must be supplied together, saw \
             init_latents={init_latents_present} reference={reference_present}"
        )));
    }
    Ok(())
}

/// The conditioning a request carries must be the conditioning the pipeline is handed (sc-14548).
///
/// # What this exists to stop, stated exactly
///
/// `StableAudio3Generator::generate` resolves its conditioning through two weight-free functions
/// ([`crate::model::reference_audio_for`], [`crate::model::audio_edit_for`]) and then forwards the
/// results. Both forwards are one token wide, and substituting either for `None` is a **complete,
/// silent feature deletion**: an inpaint request degrades to plain text-to-audio at the same seed,
/// the same length, and the same sample rate, and every shape, length and finiteness assertion still
/// passes. `generate` needs multi-gigabyte weights, so nothing in the PR lane evaluates that
/// forward — this is the exact shape sc-14547 was reviewed three times over.
///
/// So the forwarded values are re-derived from the request and compared. The predicate is a free
/// function taking booleans, which is what lets `tests/audio_edit.rs` gate the **rule** without
/// weights; the call site inside `generate` is still weights-only, and that residue is named in
/// `docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md` rather than claimed closed.
///
/// It also refuses the combination, which `validate` has already rejected at the request boundary —
/// two independent expressions of one rule, because the validation path and the synthesis path are
/// edited by different changes.
pub fn conditioning_is_forwarded(
    request_has_reference: bool,
    forwarded_reference: bool,
    request_has_edit: bool,
    forwarded_edit: bool,
) -> Result<()> {
    if request_has_reference != forwarded_reference {
        return Err(AudioError::Msg(format!(
            "a request carrying reference audio ({request_has_reference}) must forward it to the \
             pipeline, saw {forwarded_reference}"
        )));
    }
    if request_has_edit != forwarded_edit {
        return Err(AudioError::Msg(format!(
            "a request carrying an audio edit ({request_has_edit}) must forward it to the \
             pipeline, saw {forwarded_edit}"
        )));
    }
    if forwarded_reference && forwarded_edit {
        return Err(AudioError::Msg(
            "reference audio and audio editing cannot be combined in one request".into(),
        ));
    }
    Ok(())
}

/// A resolved conditioning pair carried together with the request facts it was resolved *from*
/// (sc-14548).
///
/// # Why this is a struct rather than two arguments
///
/// [`conditioning_is_forwarded`] pins the values where they are **computed**, inside
/// `StableAudio3Generator::generate`. That leaves the argument list of the call one line below it
/// unchecked: writing `None` in place of the resolved `edit` at the call site satisfies the guard,
/// because the guard read the local, not the argument. That is the identical shape to the finding it
/// was written for, one seam further along.
///
/// Bundling the resolved values with the request booleans **moves** that seam rather than removing
/// it. [`Self::check`] runs inside [`StableAudio3Pipeline::synthesize_conditioned`] — the far side
/// of the boundary — so nulling a *field* is refused there, and the single-token edit on the
/// receipt itself becomes two tokens (a field and its matching boolean).
///
/// What is **not** claimed is that the nullable argument is gone. `synthesize_conditioned`
/// destructures this receipt and forwards `edit` to
/// `StableAudio3Pipeline::synthesize_traced` as a plain argument one line later, after
/// [`Self::check`] has already run against the receipt rather than the argument, so that forward
/// remains a one-token site. It is caught, with weights, by `real_inpaint_…`'s bit-exact-outside
/// assertion — the same catcher as row 7 of the per-site table in
/// `docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md`, where the mutation is recorded as row 13 with
/// the output of the run that verified it.
// Not `Copy`: `AudioEdit` owns its region list (sc-14549). The bound was never load-bearing — this
// receipt is constructed once and moved into `synthesize_conditioned`.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardedConditioning<'a> {
    /// Whether the request carried a `Conditioning::ReferenceAudio` item.
    pub request_has_reference: bool,
    /// Whether the request carried **either** audio-edit carrier (`Conditioning::AudioEdit` or
    /// `Conditioning::AudioEditRegions`).
    pub request_has_edit: bool,
    pub reference: Option<ReferenceAudio<'a>>,
    pub edit: Option<AudioEdit<'a>>,
}

impl ForwardedConditioning<'_> {
    /// Nothing requested, nothing forwarded.
    pub fn none() -> Self {
        Self {
            request_has_reference: false,
            request_has_edit: false,
            reference: None,
            edit: None,
        }
    }

    /// Apply [`conditioning_is_forwarded`] to this receipt.
    pub fn check(&self) -> Result<()> {
        conditioning_is_forwarded(
            self.request_has_reference,
            self.reference.is_some(),
            self.request_has_edit,
            self.edit.is_some(),
        )
    }
}

/// A forwarding receipt constructed only after provider request validation has completed.
///
/// The crate-private validated synthesis entry accepts this type while the public defensive entry
/// accepts [`ForwardedConditioning`]. Consequently, changing the production call back to the public
/// entry is a type error rather than a silent second finiteness scan.
pub(crate) struct ValidatedForwardedConditioning<'a>(ForwardedConditioning<'a>);

impl<'a> ValidatedForwardedConditioning<'a> {
    pub(crate) fn new(conditioning: ForwardedConditioning<'a>) -> Self {
        Self(conditioning)
    }

    fn into_inner(self) -> ForwardedConditioning<'a> {
        self.0
    }
}

/// Where the request-local stream's draws sat relative to the source encode (sc-14547).
///
/// Exposed for the draw-order gate. The invariant is `draws_after_initial_noise == 1`: the
/// sampler's initial noise is drawn before the source is encoded, so a reference clip never
/// perturbs the draw a text-only request at the same seed would have made.
///
/// # How much this actually discriminates
///
/// Less than the invariant's phrasing suggests, and the difference matters. **SAME-S consumes zero
/// draws on encode**, so on the four small ids `draws_after_source_encode` equals
/// `draws_after_initial_noise` and reordering the two operations would not move either count. Only
/// medium's SAME-L encode draws, so only `medium` and `medium_base` can falsify the ordering at
/// all. That is why the real-weight case runs all six *and* separately requires at least one of
/// them to report a drawing encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDrawOrder {
    pub draws_after_initial_noise: usize,
    pub draws_after_source_encode: usize,
}

/// Conform one caller clip onto this model's 44.1 kHz stereo timeline (sc-14547).
///
/// The order is fixed and each step depends on the previous one:
///
/// 1. **Resample only the retained global-output window** to [`SAMPLE_RATE`] through the shared
///    [`candle_audio::dsp::resample_range`]. The full source remains visible to the FIR, so phase
///    and clip-boundary semantics are byte-identical to slicing the whole-buffer result. Off-rate
///    source audio is converted, never rejected: the
///    lane's other two audio generators emit 48 kHz, so rejecting would refuse audio produced one
///    step earlier in the same product, and the 160:147 stereo ratio is already gated in
///    `candle-audio`'s own resampler tests. (ACE-Step's "must be 48000 Hz" is a missing resampler,
///    not a policy — that crate contains no DSP resampling of any kind.)
/// 2. **Conform only retained frames**: mono duplicates, stereo passes through, more than two
///    channels keeps the first two.
/// 3. **Right-zero-pad** that stereo prefix to exactly `target_frames`, which is the adapted sample
///    size for the *requested* duration. Source extent never moves the geometry.
///
/// This ordering avoids both the complete resampled clip and the former target-sized buffer at the
/// source channel count. Because the pad value is zero, moving channel conformance before padding
/// preserves the exact prior output for mono, stereo, and multichannel clips.
///
/// Returns interleaved stereo of exactly `target_frames * CHANNELS` values.
pub fn prepare_reference_pcm(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    target_frames: usize,
) -> Result<Vec<f32>> {
    validate_reference_pcm(samples, sample_rate, channels)?;
    prepare_validated_reference_pcm(samples, sample_rate, channels, target_frames)
}

fn validate_reference_pcm(samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    if channels == 0 {
        return Err(AudioError::Msg(
            "reference audio must declare at least one channel".into(),
        ));
    }
    if sample_rate == 0 {
        return Err(AudioError::Msg(
            "reference audio must declare a non-zero sample rate".into(),
        ));
    }
    if samples.is_empty() {
        return Err(AudioError::Msg(
            "reference audio must contain at least one frame".into(),
        ));
    }
    let source_channels = channels as usize;
    if !samples.len().is_multiple_of(source_channels) {
        return Err(AudioError::Msg(format!(
            "reference audio has {} samples, not a whole number of {source_channels}-channel frames",
            samples.len()
        )));
    }
    if let Some(value) = samples.iter().copied().find(|value| !value.is_finite()) {
        return Err(AudioError::Msg(format!(
            "reference audio contains the non-finite sample {value}"
        )));
    }
    Ok(())
}

/// Prepare PCM whose request-facing shape and finiteness checks have already completed.
///
/// This is private because bypassing [`validate_reference_pcm`] is sound only on the generator's
/// post-validation route. The resampler receives the complete source clip but evaluates only the
/// retained prefix on its global output timeline, preserving whole-clip phase and FIR boundaries.
/// Channel conformance writes that bounded prefix directly into the fixed stereo output; the tail
/// is already zero, so there is no source-channel-wide padded allocation.
fn prepare_validated_reference_pcm(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    target_frames: usize,
) -> Result<Vec<f32>> {
    let source_channels = channels as usize;
    let source_frames = samples.len() / source_channels;
    let full_output_frames =
        candle_audio::dsp::resample_output_frames(source_frames, sample_rate, SAMPLE_RATE)?;
    let retained_frames = full_output_frames.min(target_frames);
    let resampled = candle_audio::dsp::resample_range(
        samples,
        sample_rate,
        SAMPLE_RATE,
        channels,
        0..retained_frames,
    )?;
    let output_samples = target_frames.checked_mul(CHANNELS).ok_or_else(|| {
        AudioError::Msg(format!(
            "reference output sample count overflows usize ({target_frames} frames, {CHANNELS} channels)"
        ))
    })?;
    let mut output = vec![0.0f32; output_samples];
    for frame in 0..retained_frames {
        let source = &resampled[frame * source_channels..(frame + 1) * source_channels];
        let (left, right) = match source_channels {
            1 => (source[0], source[0]),
            _ => (source[0], source[1]),
        };
        output[frame * CHANNELS] = left;
        output[frame * CHANNELS + 1] = right;
    }
    Ok(output)
}

/// A caller-supplied bounded edit of an existing clip (sc-14548).
///
/// # There is no `mode` field, and that is the proof obligation being discharged
///
/// gen-core names three edit modes this family serves —
/// [`Inpaint`](candle_audio::gen_core::AudioEditMode::Inpaint),
/// [`Repaint`](candle_audio::gen_core::AudioEditMode::Repaint) and
/// [`Extend`](candle_audio::gen_core::AudioEditMode::Extend) — and Stable Audio 3 has **one**
/// mechanism behind all three: mask the region, SAME-encode the source, hand the DiT
/// `[inpaint_mask, inpaint_masked_input]`. gen-core's mode docs describe `Inpaint` as
/// silence-substituted and `Repaint` as context-conditioned, but that distinction is written against
/// ACE-Step's two native tasks; upstream SA3 exposes no such split, so for this family the two are
/// aliases and must be byte-identical for the same request and seed.
///
/// Rather than assert that with a test alone, the mode is **consumed entirely** in
/// [`crate::model::audio_edit_for`], which turns a mode plus an
/// [`Option<TimeRegion>`](candle_audio::gen_core::TimeRegion) into the resolved
/// `[start_secs, end_secs)` below. Nothing downstream of that can branch on the mode because nothing
/// downstream can see it. `Extend` differs from `Inpaint` here only in *where* the region sits and
/// how long the output is — both of which are already numbers in this struct. The real-weight case
/// `real_repaint_is_byte_identical_to_inpaint` asserts the consequence as well, because a structural
/// argument does not survive somebody adding a field.
///
/// `start_secs` / `end_secs` are on the **post-resample** 44.1 kHz timeline, i.e. the timeline
/// [`prepare_reference_pcm`] produces, not the caller's source rate. Applying the caller's seconds
/// against their original rate and then resampling the audio underneath the mask moves the edit
/// window silently, which is why the conversion to sample indices in [`edit_geometry`] uses
/// [`SAMPLE_RATE`] and never [`Self::sample_rate`].
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEdit<'a> {
    /// Interleaved source PCM, `channels` values per frame.
    pub samples: &'a [f32],
    pub sample_rate: u32,
    pub channels: u16,
    /// The spans to regenerate, seconds on the 44.1 kHz timeline. **Non-empty**; single-region
    /// editing is the one-element case and travels this same path rather than a parallel one
    /// (sc-14549). Any order, and overlapping / touching / duplicate spans are legal — they are
    /// normalized into a canonical union by [`edit_geometry`].
    pub regions: Vec<EditRegionSecs>,
}

/// One requested edit span, seconds on the post-resample 44.1 kHz timeline (sc-14549).
///
/// Unlike gen-core's [`TimeRegion`](candle_audio::gen_core::TimeRegion) the end is **resolved**: an
/// absent `end_secs` has already become the source's end in [`crate::model::resolve_audio_edit`], so
/// nothing downstream has to know what "the end of the clip" meant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditRegionSecs {
    pub start_secs: f32,
    /// Exclusive.
    pub end_secs: f32,
}

/// Where **one** span of the normalized union lands, at both resolutions (sc-14548 / sc-14549).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditSpan {
    /// Span start in 44.1 kHz frames, `int(start_secs * 44_100)`.
    pub start_sample: usize,
    /// Span end in 44.1 kHz frames, `int(end_secs * 44_100)`, clamped to `adapted_size`.
    pub end_sample: usize,
    /// `ceil(start_sample / 4096)` — where the edit begins once the mask is resized.
    pub start_latent: usize,
    /// `ceil(end_sample / 4096)`.
    pub end_latent: usize,
}

/// Where an edit's regions land, at both resolutions (sc-14548, generalized to a list by sc-14549).
///
/// Produced by [`edit_geometry`] and consumed by [`edit_keep_mask`], [`edit_local_conditioning`] and
/// [`stitch_outside_region`], so every one of those reads the *same* resolved indices rather than
/// re-deriving them from seconds and drifting apart.
///
/// # `spans` is the normalized union, and that is load-bearing
///
/// It is **sorted, disjoint and non-touching** in audio-sample space — the output of
/// [`merge_edit_regions`]. Downstream code therefore never has to reason about overlap: the mask
/// zeroes each span independently and the stitch preserves everything outside all of them, and
/// neither can double-apply because the spans cannot intersect. Any crossfade a future change adds
/// lands wholly inside one merged span, so no frame could receive two fades.
///
/// There is deliberately **no** promoted `start_sample` / `end_sample` pair on this struct. Keeping
/// the first span in named fields alongside a list of "the others" is exactly what makes a
/// "honours the first region, drops the rest" regression easy to write and invisible to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditGeometry {
    /// The adapted sample size the mask is built over — **not** the requested duration's extent.
    pub adapted_size: usize,
    /// `adapted_size / 4096`.
    pub latent_length: usize,
    /// `int(seconds_total * 44_100)`: everything at or past this is zeroed for training parity.
    pub effective_samples: usize,
    /// The normalized union: sorted, disjoint, non-touching, non-empty.
    pub spans: Vec<EditSpan>,
}

/// Frames a clip occupies on the model's own 44.1 kHz timeline once resampled (sc-14548).
///
/// The inpaint contract is stated in seconds against the *post-resample* timeline, and the geometry
/// (adapted sample size, latent length, output length) has to be resolved **before** the buffer is
/// resampled — so this reproduces [`candle_audio::dsp::resample`]'s own output-length rule rather
/// than measuring it. That duplication is the hazard, so it is gated: `tests/audio_edit.rs`
/// `resampled_frame_count_agrees_with_the_shared_resampler` runs the real resampler at a spread of
/// rates and requires the two to agree exactly.
pub fn resampled_frame_count(source_frames: usize, sample_rate: u32) -> usize {
    if sample_rate == 0 || source_frames == 0 || sample_rate == SAMPLE_RATE {
        return source_frames;
    }
    let frames = ((source_frames as u128) * (SAMPLE_RATE as u128) + (sample_rate as u128 / 2))
        / sample_rate as u128;
    usize::try_from(frames.max(1)).unwrap_or(usize::MAX)
}

/// Seconds to 44.1 kHz frames, truncating — upstream's own integer cast (sc-14548).
///
/// Always [`SAMPLE_RATE`], never the caller's source rate: the region is stated on the timeline the
/// source is resampled **onto**. Applying the caller's seconds against their original rate and then
/// resampling the audio underneath the mask moves the edit window silently, and every "the outside
/// is preserved" assertion still passes while it happens.
fn seconds_to_samples(seconds: f32) -> usize {
    ((seconds as f64) * SAMPLE_RATE as f64) as usize
}

/// The `[start, end)` region in 44.1 kHz frames (sc-14548).
///
/// Shared by [`edit_geometry`] and by the provider's own validation, so the window the request is
/// *checked* against and the window it is *masked* against cannot drift apart.
pub fn edit_region_samples(start_secs: f32, end_secs: f32) -> (usize, usize) {
    (seconds_to_samples(start_secs), seconds_to_samples(end_secs))
}

/// The latent positions an audio-frame region occupies once the mask is nearest-resized (sc-14548).
///
/// `ceil`, on both ends, and that is a derived fact rather than a choice: the mask is built at audio
/// resolution and resized by [`candle_audio::ops::nearest_downsample1d`], which takes `src[t*4096]`,
/// so latent `t` is zeroed exactly when `t*4096 ∈ [start, end)` — i.e. on
/// `[ceil(start/4096), ceil(end/4096))`. Computing latent indices *directly* by rounding the
/// seconds, rather than deriving them from the audio-resolution mask this way, is the mutation that
/// shifts the window by a frame at some region boundaries and by none at others.
pub fn edit_region_latents(start_sample: usize, end_sample: usize) -> (usize, usize) {
    (
        start_sample.div_ceil(LATENT_DOWNSAMPLING),
        end_sample.div_ceil(LATENT_DOWNSAMPLING),
    )
}

/// Resolve one edit's region onto the sampling geometry it will be masked against (sc-14548).
///
/// `duration_secs` is the **output** duration — the source's own duration for an inpaint, the new
/// total length for an extend — and `geometry` is what [`crate::sampler::adapt_sample_size`] made of
/// it. Both endpoints are converted with truncation (`int(seconds * 44_100)`), matching upstream's
/// integer cast, and the upper one is clamped to the adapted size.
///
/// Two things here are the whole point of the function and each has its own gate:
///
/// * the conversion uses [`SAMPLE_RATE`], never `edit.sample_rate`, so a 48 kHz source's region
///   lands at the same seconds as a 44.1 kHz one's;
/// * the mask is sized by `geometry.sample_size` — the **adapted** size, which includes the
///   configured 6 s headroom and the 8192-alignment — not by `duration_secs * 44_100`. A mask built
///   at the un-adapted size and resized to the latent length would compress the whole timeline and
///   move the edit window while every "the outside is preserved" assertion still passed.
pub fn edit_geometry(
    edit: &AudioEdit<'_>,
    geometry: &SampleGeometry,
    duration_secs: f32,
) -> Result<EditGeometry> {
    if !duration_secs.is_finite() {
        return Err(AudioError::Msg(
            "audio edit region and duration must be finite".into(),
        ));
    }
    if edit.regions.is_empty() {
        return Err(AudioError::Msg(
            "an audio edit must name at least one region".into(),
        ));
    }
    let adapted_size = geometry.sample_size;
    let mut clamped = Vec::with_capacity(edit.regions.len());
    for region in &edit.regions {
        if !(region.start_secs.is_finite() && region.end_secs.is_finite()) {
            return Err(AudioError::Msg(
                "audio edit region and duration must be finite".into(),
            ));
        }
        if region.start_secs < 0.0 || region.end_secs <= region.start_secs {
            return Err(AudioError::Msg(format!(
                "audio edit region [{}, {}) must be non-negative and non-empty",
                region.start_secs, region.end_secs
            )));
        }
        let (start_sample, end_sample) = edit_region_samples(region.start_secs, region.end_secs);
        let (start_sample, end_sample) =
            (start_sample.min(adapted_size), end_sample.min(adapted_size));
        // Checked per **requested** region, not per merged span: a caller who names a span narrower
        // than one latent frame gets told about *that* span, and two such spans that happen to
        // merge into a wide one do not silently become acceptable. This is the same rule the
        // single-region path has always enforced, applied to every entry.
        let (start_latent, end_latent) = edit_region_latents(start_sample, end_sample);
        if end_latent <= start_latent {
            return Err(AudioError::Msg(format!(
                "audio edit region [{}, {}) collapses to zero latent positions ({} .. {}); it must \
                 span at least one {LATENT_DOWNSAMPLING}-sample latent frame",
                region.start_secs, region.end_secs, start_latent, end_latent
            )));
        }
        clamped.push((start_sample, end_sample));
    }
    let effective_samples = seconds_to_samples(duration_secs).min(adapted_size);
    let spans = merge_edit_regions(&clamped)
        .into_iter()
        .map(|(start_sample, end_sample)| {
            let (start_latent, end_latent) = edit_region_latents(start_sample, end_sample);
            EditSpan {
                start_sample,
                end_sample,
                start_latent,
                end_latent,
            }
        })
        .collect();
    Ok(EditGeometry {
        adapted_size,
        latent_length: geometry.latent_length,
        effective_samples,
        spans,
    })
}

/// Normalize requested audio-frame regions into a canonical union: sorted, disjoint, non-touching
/// (sc-14549).
///
/// # Why this exists, and why it merges rather than rejects
///
/// The contract accepts regions in **any order**, and accepts overlapping, touching and duplicate
/// spans (see [`Conditioning::AudioEditRegions`](candle_audio::gen_core::Conditioning)). All of
/// those describe the same set of frames, so rejecting them would refuse requests with an
/// unambiguous meaning — and, worse, would make the result depend on the order the caller happened
/// to write. Merging makes order-independence a *property of the data structure* rather than
/// something every consumer has to remember: `[a, b]` and `[b, a]` produce the identical `Vec`, so
/// the mask, the local conditioning and the stitch cannot disagree about which is authoritative.
///
/// # Why the merge is at integer 44.1 kHz sample resolution
///
/// Because that is the resolution the mask is *built* at. Merging in seconds would leave two spans
/// that round to the same sample boundary looking distinct; merging at latent resolution would
/// coalesce spans that are genuinely separate in the PCM the stitch preserves. Neither is the
/// resolution the decision is actually made at. Latent quantization happens **after** this, per
/// merged span, and two merged spans coalescing into the same latent range is fine — the audio-space
/// union stays authoritative for what the final PCM preserves.
///
/// **Touching counts as overlapping**: `[a, b)` and `[b, c)` are merged into `[a, c)`. They cover a
/// contiguous set of frames with no gap, so leaving them separate would produce two adjacent spans
/// whose union is identical but whose representation is not — order-independence would hold while
/// canonical form did not.
///
/// Input pairs must already be non-empty and clamped; empty input yields empty output.
pub fn merge_edit_regions(regions: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut sorted = regions.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(sorted.len());
    for (start, end) in sorted {
        match merged.last_mut() {
            // `start <= last.1` rather than `<`: touching spans merge too.
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// The audio-resolution **keep** mask: `1.0` keeps the source, `0.0` is regenerated (sc-14548).
///
/// Ones everywhere except two spans, both zeroed:
///
/// * the requested edit region `[start_sample, end_sample)`;
/// * `[effective_samples, adapted_size)` — everything past `seconds_total`. Upstream zeroes it too,
///   and not as a tidiness measure: the model was trained with the local conditioner zero over the
///   padding, so leaving ones there presents an out-of-distribution input the DiT has never seen.
///   The attention/padding mask still carries the configured 6 s headroom; these are different
///   masks and this one is the stricter of the two.
///
/// Built at audio-sample resolution over the **adapted** size and resized afterwards, in that order.
/// The polarity is the other thing worth stating once: this is a *keep* mask, so it multiplies the
/// encoded source to zero out the region the model must invent.
pub fn edit_keep_mask(geometry: &EditGeometry) -> Vec<f32> {
    let mut mask = vec![1.0f32; geometry.adapted_size];
    // **Every** span, in one mask — not one mask per span and not one sampler run per span. That
    // is the whole point of the multi-region contract: the regions share a single denoising
    // trajectory, which N sequential single-region edits cannot reproduce because each of those
    // re-noises and re-decodes. The spans are disjoint by construction (`merge_edit_regions`), so
    // no frame is zeroed twice and the loop is order-independent.
    for span in &geometry.spans {
        for value in &mut mask[span.start_sample.min(geometry.adapted_size)
            ..span.end_sample.min(geometry.adapted_size)]
        {
            *value = 0.0;
        }
    }
    for value in &mut mask[geometry.effective_samples.min(geometry.adapted_size)..] {
        *value = 0.0;
    }
    mask
}

/// Build the DiT's `[1, 257, latent_length]` local conditioning from an encoded source (sc-14548).
///
/// Channel order is `[inpaint_mask (1), inpaint_masked_input (256)]`, which is the order
/// `local_add_cond_ids` declares in every one of the six shipped configs and the order
/// [`crate::dit::StableAudio3Dit`] slices back out. Concatenating the other way round is a silent
/// mis-wiring: the shape is identical and the model still runs.
///
/// The mask is built at audio resolution by [`edit_keep_mask`] and resized to latent resolution with
/// [`candle_audio::ops::nearest_downsample1d`], whose own tests pin that a zeroed audio span
/// `[start, end)` lands on `[ceil(start/4096), ceil(end/4096))`. `masked_input` is the encoded
/// source **times that mask**, so the latents inside the edit window are exactly zero — handing the
/// DiT the unmasked source there would show it the answer it is being asked to invent.
///
/// Takes the encoded source as a plain tensor rather than encoding it here, deliberately: that is
/// what lets `tests/audio_edit.rs` drive the whole mask → resize → multiply → concat chain with a
/// synthetic latents tensor and no weights at all. The same construction inside the SAME-encoding
/// synthesis method would be reachable only from a real-weight lane.
pub fn edit_local_conditioning(
    geometry: &EditGeometry,
    source_latents: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if source_latents.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
        return Err(AudioError::Msg(format!(
            "audio edit source latents have shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
            source_latents.dims(),
            geometry.latent_length
        )));
    }
    let audio_mask = Tensor::from_vec(
        edit_keep_mask(geometry),
        (1, 1, geometry.adapted_size),
        device,
    )?;
    let latent_mask = candle_audio::ops::nearest_downsample1d(&audio_mask, LATENT_DOWNSAMPLING)?
        .to_dtype(dtype)?;
    if latent_mask.dims() != [1, 1, geometry.latent_length] {
        return Err(AudioError::Msg(format!(
            "resized inpaint mask has shape {:?}, expected [1,1,{}]",
            latent_mask.dims(),
            geometry.latent_length
        )));
    }
    let masked_input = source_latents
        .to_dtype(dtype)?
        .broadcast_mul(&latent_mask)?;
    // `[mask, masked_input]`, in that order — see the doc comment.
    Ok(Tensor::cat(&[&latent_mask, &masked_input], 1)?)
}

/// Whether a tensor holds any non-zero element, as an `f32` reduction on the tensor's own device.
///
/// The reduction is over `|x|`, not `x`: a conditioning tensor whose elements happen to cancel to a
/// zero sum is still present, and a bare sum would call it absent.
///
/// Exported (`#[doc(hidden)]`) rather than private so the weight-free lane can drive **this**
/// predicate instead of restating it. Its only production call site is inside
/// `StableAudio3Pipeline::synthesize_traced`, which needs weights, and a test that reimplements
/// the rule it is checking moves with the mutation instead of catching it — the same
/// "asserts a value rather than derives it" shape the rest of this module is written against.
/// `the_local_conditioning_handoff_must_still_be_present_where_it_crosses_into_the_dit` calls it,
/// including on a deliberately sign-cancelling tensor so the `.abs()` is load-bearing there.
#[doc(hidden)]
pub fn tensor_has_nonzero(tensor: &Tensor) -> Result<bool> {
    let total = tensor
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(total > 0.0)
}

/// How many latent positions the edit's keep mask actually retains (sc-14548).
///
/// Derived from the resolved indices alone — no tensor, no mask buffer, no encoder — so it is a
/// **second, independent** expression of what [`edit_local_conditioning`] builds. That is the point:
/// it is the expectation half of [`edit_local_conditioning_is_present`], and an expectation computed
/// by re-running the thing it is checking would move with the mutation instead of catching it.
///
/// The rule it reproduces is [`edit_keep_mask`] seen *after*
/// [`candle_audio::ops::nearest_downsample1d`], which samples `src[t * 4096]`. Latent `t` survives
/// exactly when that audio frame is retained, i.e. when `t * 4096 < effective_samples` and
/// `t * 4096` is outside `[start_sample, end_sample)` — which is `[start_latent, end_latent)` in
/// latent units.
///
/// Zero is a legitimate result, not a bug: an edit whose region covers the whole effective span
/// hands the DiT an all-zero local conditioning on purpose (regenerate everything). Naming that is
/// what keeps the guard below from having to pretend zero is impossible.
/// **The zeroed latent positions are unioned, not summed — and the reason is not the one it looks
/// like.**
///
/// Stated precisely, because an earlier revision of this comment got it wrong and the test written
/// against that claim was vacuous: for any [`EditGeometry`] that [`edit_geometry`] can actually
/// produce, the union and the sum are **equal**. [`merge_edit_regions`] leaves the audio spans
/// disjoint and non-touching, `x -> ceil(x/4096)` is monotonic, and every requested region is
/// already refused unless it spans at least one latent frame — so consecutive spans satisfy
/// `start_latent[i+1] >= end_latent[i]` and the latent ranges can touch but can never overlap.
///
/// What the union buys is therefore **independence from that invariant**, not a fix for a bug
/// reachable today. The invariant lives in a different function: delete the merge, or weaken it to
/// leave overlapping requests separate, and `[2,6)` + `[4,8)` becomes latent `[22,65)` + `[44,87)`,
/// whose shared 21 positions a sum double-counts. On an edit covering most of the clip that
/// under-report can reach zero retained latents, which would take
/// [`edit_local_conditioning_is_present`] into its "an all-zero local conditioning is correct here"
/// branch and **disarm the presence guard** on exactly the geometry that needed it.
///
/// So `the_retained_latent_count_unions_overlapping_latent_ranges` gates this by constructing an
/// overlapping-span [`EditGeometry`] **directly**, bypassing [`edit_geometry`]'s normalization.
/// Driving it only through the normalizer proves nothing, because the normalizer is what makes the
/// two formulas agree.
pub fn edit_retained_latent_count(geometry: &EditGeometry) -> usize {
    let effective_latents = geometry
        .effective_samples
        .div_ceil(LATENT_DOWNSAMPLING)
        .min(geometry.latent_length);
    // `spans` is sorted by audio-sample start, so the latent ranges derived from it are sorted too
    // (`div_ceil` is monotonic) — a single pass merges them.
    let mut zeroed = 0usize;
    let mut cursor = 0usize;
    for span in &geometry.spans {
        let start = span.start_latent.min(effective_latents).max(cursor);
        let end = span.end_latent.min(effective_latents);
        if end > start {
            zeroed += end - start;
            cursor = end;
        }
    }
    effective_latents - zeroed.min(effective_latents)
}

/// The edit's local conditioning must still be there when it crosses into the DiT (sc-14548).
///
/// # What this exists to stop, stated exactly
///
/// `local = edit_local_conditioning(...)` → `let _ = edit_local_conditioning(...)` inside
/// `StableAudio3Pipeline::synthesize_traced` is one token wide, and it leaves `local` as the zero
/// tensor the text-only path allocates. Every inpaint, repaint and extend then runs as **plain
/// text-to-audio inside the region**, and none of the obvious acceptance measurements can see it:
/// the outside is still bit-exact (that span is written by [`stitch_outside_region`] from the
/// prepared source, which does not read `local` at all), the region still changes, two seeds still
/// diverge — in fact an *unconditioned* interior diverges **more** on both counts, so a divergence
/// floor is wrong-signed against this mutation and tightening it makes matters worse.
///
/// So the check is placed at the boundary the value **crosses** rather than where it is computed.
/// [`conditioning_is_forwarded`] is the same shape one seam earlier and pinned at the computation
/// site; this is the one that observes the handoff itself. Like that one, this call site needs
/// weights, so the weight-free lane gates the **rule** and the real-weight lane is where a zeroed
/// handoff actually fails closed — immediately, before any sampling, rather than after a plausible
/// render.
///
/// # What it does and does not discriminate
///
/// * `expected` is [`edit_retained_latent_count`]` > 0`, computed from the resolved geometry;
///   `observed` is whether the tensor handed to `sample` has any non-zero element. A zeroed handoff
///   on any ordinary edit is refused here.
/// * It is an **existence** check, not an equality one. A mutation that scales `local`, or builds it
///   from the wrong source latents, keeps `observed` true and passes; those are covered by
///   `the_local_conditioning_is_mask_first_then_the_masked_source` at the construction site and by
///   the real-weight source-sensitivity assertion.
/// * On the one geometry where the region covers the entire effective span, `expected` is `false`
///   and an all-zero `local` is correct, so the zeroing mutation is invisible **there**. No shipped
///   case uses such a region; the real-weight cases all edit an interior window.
pub fn edit_local_conditioning_is_present(expected: bool, observed: bool) -> Result<()> {
    if expected != observed {
        return Err(AudioError::Msg(format!(
            "an audio edit whose keep mask retains source latents ({expected}) must reach the DiT \
             as non-zero local conditioning, saw {observed}"
        )));
    }
    Ok(())
}

/// The resolved edit geometry must describe the geometry the request is actually sampled at
/// (sc-14548).
///
/// [`edit_geometry`] takes `duration_secs` as a *separate* argument from the [`SampleGeometry`] it
/// is resolved against, because the two are independent inputs — the output duration for an extend
/// is not the source's. That separation is also the hazard: the `duration_secs` argument at the
/// call site inside `StableAudio3Pipeline::synthesize_traced` is one token wide and it sets
/// `effective_samples`, the training-parity padding boundary that [`edit_keep_mask`] zeroes from.
/// Getting it wrong moves where the local conditioner stops describing the source, and every
/// length, rate and preservation assertion still passes.
///
/// So the resolved values are compared against the sampling geometry, which was built from the same
/// duration through [`crate::sampler::adapt_sample_size`] — a genuinely different derivation
/// (`ceil` to a 4096 multiple, in latent units) rather than a copy of this one (`trunc`, in audio
/// samples).
///
/// **Granularity, stated rather than implied**: `effective_lengths` is in latent units, so this
/// compares `ceil(effective_samples / 4096)`. A wrong duration that lands inside the same
/// 4096-sample latent frame — under 93 ms — moves `effective_samples` without moving this
/// comparison. The alignment examples in `tests/audio_edit.rs` are what pin the sample-resolution
/// value itself.
pub fn edit_geometry_matches_request(
    resolved: &EditGeometry,
    geometry: &SampleGeometry,
) -> Result<()> {
    if resolved.adapted_size != geometry.sample_size
        || resolved.latent_length != geometry.latent_length
    {
        return Err(AudioError::Msg(format!(
            "resolved edit geometry spans {}/{} samples/latents, but the request is sampled at \
             {}/{}",
            resolved.adapted_size,
            resolved.latent_length,
            geometry.sample_size,
            geometry.latent_length
        )));
    }
    if let Some(effective_lengths) = geometry.effective_lengths.as_ref() {
        let expected = effective_lengths
            .first()
            .copied()
            .unwrap_or(geometry.latent_length)
            .min(geometry.latent_length);
        let actual = resolved
            .effective_samples
            .div_ceil(LATENT_DOWNSAMPLING)
            .min(geometry.latent_length);
        if actual != expected {
            return Err(AudioError::Msg(format!(
                "the edit's effective span covers {actual} latents, but the request's own duration \
                 covers {expected}"
            )));
        }
    }
    Ok(())
}

/// Put the canonical prepared source back outside the edit window, bit-exactly (sc-14548).
///
/// Frozen upstream regenerates and decodes the *whole* clip; it never pastes anything back. This
/// repository's `AudioEdit` contract is stricter than that — "the rest of the clip is preserved" —
/// so the frozen behaviour is preserved through the raw decode and the preservation is then
/// satisfied here, explicitly, by copying `prepared` over every frame outside
/// `[start_sample, end_sample)`.
///
/// **Exact equality, not a tight bound.** The whole point of doing it here rather than trusting the
/// model is that the result is bit-identical to `prepare_reference_pcm`'s own output; a numeric
/// tolerance would pass a stitch that had drifted by a frame, which is precisely the failure this
/// exists to exclude. Any future crossfade must therefore live **wholly inside** the region.
///
/// `rendered` and `prepared` are interleaved stereo. `prepared` is at least as long as the output
/// (it is sized to the adapted sample size); the result has exactly `rendered`'s length.
pub fn stitch_outside_region(
    rendered: &[f32],
    prepared: &[f32],
    geometry: &EditGeometry,
) -> Result<Vec<f32>> {
    if !rendered.len().is_multiple_of(CHANNELS) {
        return Err(AudioError::Msg(format!(
            "rendered audio has {} samples, not whole {CHANNELS}-channel frames",
            rendered.len()
        )));
    }
    let frames = rendered.len() / CHANNELS;
    if prepared.len() < frames * CHANNELS {
        return Err(AudioError::Msg(format!(
            "prepared source covers {} frames, fewer than the rendered {frames}",
            prepared.len() / CHANNELS
        )));
    }
    let mut output = rendered.to_vec();
    // Outside **every** span. The union is sorted and disjoint, so a single forward walk suffices
    // and no frame is visited twice — which is what makes "each merged span receives at most one
    // treatment" structural rather than a rule a future crossfade has to remember.
    let mut span = 0usize;
    for frame in 0..frames {
        while span < geometry.spans.len() && frame >= geometry.spans[span].end_sample {
            span += 1;
        }
        if span < geometry.spans.len() && frame >= geometry.spans[span].start_sample {
            continue;
        }
        output[frame * CHANNELS] = prepared[frame * CHANNELS];
        output[frame * CHANNELS + 1] = prepared[frame * CHANNELS + 1];
    }
    Ok(output)
}

/// Runtime choices mapped from the backend-neutral generation request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthesisParameters {
    pub duration_secs: f32,
    pub steps: usize,
    pub sampler: SamplerKind,
    pub guidance: Guidance,
    pub seed: u64,
}

/// A loaded, connected text-to-stereo-audio graph.
pub struct StableAudio3Pipeline {
    config: crate::config::StableAudioConfig,
    conditioner: T5GemmaConditioner,
    dit: StableAudio3Dit,
    same: SameAutoencoder,
    device: Device,
    dtypes: ComputeDTypes,
}

impl StableAudio3Pipeline {
    /// Build the connected graph, refusing any snapshot that is not `geometry`'s exact checkpoint.
    pub fn from_layout(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
    ) -> Result<Self> {
        Self::from_layout_with_dtypes(layout, geometry, device, ComputeDTypes::for_device(device))
    }

    /// Build the connected graph with an explicit compute policy.
    ///
    /// Production loads go through [`Self::from_layout`], which derives the policy from the selected
    /// device. This form exists so a backend-dtype oracle can hold the device fixed and vary the
    /// dtype, which is the only way to measure an F16-vs-F32 bound rather than assert one.
    pub fn from_layout_with_dtypes(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
        dtypes: ComputeDTypes,
    ) -> Result<Self> {
        Self::from_layout_with_dtypes_and_adapters(
            layout,
            geometry,
            device,
            dtypes,
            &crate::adapters::AdapterPlan::default(),
        )
    }

    /// Build the connected graph with a resolved adapter stack folded into the root weights.
    ///
    /// An **empty** plan is not merely a no-op fold — the wrapper is never installed, so this is the
    /// same object graph, byte for byte, as [`Self::from_layout`]. That is what a `scale == 0.0`
    /// request resolves to, and it is why bit identity is a structural property here rather than a
    /// numeric bound.
    pub fn from_layout_with_adapters(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
        adapters: &crate::adapters::AdapterPlan,
    ) -> Result<Self> {
        Self::from_layout_with_dtypes_and_adapters(
            layout,
            geometry,
            device,
            ComputeDTypes::for_device(device),
            adapters,
        )
    }

    /// [`Self::from_layout_with_dtypes`] plus an adapter stack.
    pub fn from_layout_with_dtypes_and_adapters(
        layout: &SnapshotLayout,
        geometry: VariantGeometry,
        device: &Device,
        dtypes: ComputeDTypes,
        adapters: &crate::adapters::AdapterPlan,
    ) -> Result<Self> {
        validate_layout(layout, geometry)?;
        let diffusion = match &layout.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let builders = layout.full_pipeline_builders_with_adapters(
            dtypes.root,
            dtypes.text,
            device,
            Some(adapters),
        )?;
        let conditioner =
            T5GemmaConditioner::from_layout_with_builders(layout, device, builders.clone())?;
        let dit = StableAudio3Dit::load(diffusion, builders.clone())
            .map_err(|error| AudioError::Msg(format!("load SA3 DiT: {error}")))?;
        let same = SameAutoencoder::load(&diffusion.pretransform.config, builders)
            .map_err(|error| AudioError::Msg(format!("load SA3 SAME: {error}")))?;
        Ok(Self {
            config: layout.config.clone(),
            conditioner,
            dit,
            same,
            device: device.clone(),
            dtypes,
        })
    }

    /// The compute policy this graph was loaded with.
    pub fn dtypes(&self) -> ComputeDTypes {
        self.dtypes
    }

    pub fn synthesize(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        self.synthesize_with_reference(
            prompt,
            negative_prompt,
            parameters,
            None,
            on_progress,
            on_decoding,
            is_canceled,
        )
    }

    /// Synthesize, optionally starting from a caller-supplied source clip (sc-14547).
    ///
    /// With `reference` present the source is conformed by [`prepare_reference_pcm`], SAME-encoded
    /// on the request's own stream, and mixed into the sampler's initial noise at
    /// [`ReferenceAudio::noise_level`]. The `local` inpaint conditioning stays exactly zero on this
    /// path: a whole-clip restyle supplies its source as `init_data` only, never as masked local
    /// input, and the attention/padding mask still comes from the *requested* duration plus the
    /// configured headroom rather than from the source's extent.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_with_reference(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        reference: Option<ReferenceAudio<'_>>,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        Ok(self
            .synthesize_with_reference_traced(
                prompt,
                negative_prompt,
                parameters,
                reference,
                on_progress,
                on_decoding,
                is_canceled,
            )?
            .0)
    }

    /// [`Self::synthesize_with_reference`] plus the request stream's draw-order provenance.
    ///
    /// Exists so the draw-order requirement is *gated* rather than asserted in prose: the returned
    /// [`ReferenceDrawOrder`] is the only observation that separates "initial noise first, then the
    /// source encode" from the reverse.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_with_reference_traced(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        reference: Option<ReferenceAudio<'_>>,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Vec<f32>, Option<ReferenceDrawOrder>)> {
        self.synthesize_traced(
            prompt,
            negative_prompt,
            parameters,
            reference,
            None,
            ReferenceInputValidation::Required,
            on_progress,
            on_decoding,
            is_canceled,
        )
    }

    /// The one entry point a provider calls: text-only, restyle, or bounded edit (sc-14548).
    ///
    /// With `edit` present this is the whole of Stable Audio 3's inpaint / repaint / extend surface:
    /// the source is conformed by [`prepare_reference_pcm`] onto the model's 44.1 kHz stereo
    /// timeline, SAME-encoded on the request's own stream, masked over the region, and handed to the
    /// DiT as the `[inpaint_mask, inpaint_masked_input]` local conditioning. Sampling starts from
    /// **pure seeded noise** — no `init_data`, no `init_noise_level`; the source enters only through
    /// the local conditioner, which is what distinguishes it from the restyle path. Afterwards the
    /// prepared source PCM is stitched back over everything outside the region, bit-exactly
    /// ([`stitch_outside_region`]).
    ///
    /// # Why this takes both conditionings instead of there being a method per mode
    ///
    /// Deliberate, and it is the sc-14547 carry-forward. With one method per mode, `generate` has to
    /// carry a `match` selecting the path — and selecting the wrong arm, or scrutinising a constant
    /// `None`, is a single-token edit inside a method that needs multi-gigabyte weights, i.e.
    /// invisible to every weight-free lane and a complete silent feature deletion. With one entry
    /// point there is no arm to select. There is deliberately **no** `synthesize_with_edit`
    /// convenience wrapper for the same reason: it would reintroduce the choice.
    ///
    /// Supplying both is refused rather than resolved by precedence.
    ///
    /// The conditioning arrives as a [`ForwardedConditioning`] receipt rather than as two optional
    /// arguments, and is re-checked here, on the far side of the boundary — see that type.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_conditioned(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        conditioning: ForwardedConditioning<'_>,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        self.synthesize_conditioned_with_validation(
            prompt,
            negative_prompt,
            parameters,
            conditioning,
            ReferenceInputValidation::Required,
            on_progress,
            on_decoding,
            is_canceled,
        )
    }

    /// Production request route after [`crate::model::validate_request`] has checked source shape
    /// and finiteness. Kept crate-private so direct pipeline callers retain defensive validation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synthesize_conditioned_validated(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        conditioning: ValidatedForwardedConditioning<'_>,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        self.synthesize_conditioned_with_validation(
            prompt,
            negative_prompt,
            parameters,
            conditioning.into_inner(),
            ReferenceInputValidation::Complete,
            on_progress,
            on_decoding,
            is_canceled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_conditioned_with_validation(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        conditioning: ForwardedConditioning<'_>,
        reference_validation: ReferenceInputValidation,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        conditioning.check()?;
        let ForwardedConditioning {
            reference, edit, ..
        } = conditioning;
        Ok(self
            .synthesize_traced(
                prompt,
                negative_prompt,
                parameters,
                reference,
                edit,
                reference_validation,
                on_progress,
                on_decoding,
                is_canceled,
            )?
            .0)
    }

    /// [`Self::synthesize_conditioned`] with an edit, plus the request stream's draw-order
    /// provenance. Exists so the draw order is *gated* on this path rather than asserted in prose.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_with_edit_traced(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        edit: AudioEdit<'_>,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Vec<f32>, Option<ReferenceDrawOrder>)> {
        self.synthesize_traced(
            prompt,
            negative_prompt,
            parameters,
            None,
            Some(edit),
            ReferenceInputValidation::Required,
            on_progress,
            on_decoding,
            is_canceled,
        )
    }

    /// The one synthesis body: text-only, restyle, or bounded edit.
    ///
    /// Deliberately a single function rather than one per mode. The frozen draw order (initial
    /// sampler noise **first**, then any source encode) and the decode/finish tail are invariants of
    /// all three paths, and a second copy of them is a second place for them to drift.
    #[allow(clippy::too_many_arguments)]
    fn synthesize_traced(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        reference: Option<ReferenceAudio<'_>>,
        edit: Option<AudioEdit<'_>>,
        reference_validation: ReferenceInputValidation,
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Vec<f32>, Option<ReferenceDrawOrder>)> {
        canceled(is_canceled)?;
        if reference.is_some() && edit.is_some() {
            return Err(AudioError::Msg(
                "reference audio and audio editing cannot be combined in one request".into(),
            ));
        }
        let geometry = self.geometry(parameters.duration_secs)?;
        let template = Tensor::zeros(
            (1, LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        let mut noise = SeededNoise::new(parameters.seed);
        // Frozen order, and the reason this is the first thing that happens: the sampler's initial
        // noise is drawn *before* the source is encoded, so attaching a reference clip does not
        // move the draw a text-only request at the same seed would have made.
        let initial = noise.standard_normal_like(&template)?;
        let draws_after_initial_noise = noise.draws();
        // A guard against a *future* edit, not a live check: as written the stream is constructed
        // three lines up and drawn from exactly once, so this can only fire if something is
        // inserted between the two. Say so rather than calling it "fails closed" unqualified —
        // and note it discriminates a reordering only where the encode itself draws, i.e. on
        // SAME-L (`medium`, `medium_base`), never on the four SAME-S ids.
        if draws_after_initial_noise != 1 {
            return Err(AudioError::Msg(format!(
                "initial sampler noise must be the request stream's first draw, saw \
                 {draws_after_initial_noise}"
            )));
        }
        let mut init_latents = None;
        let mut order = None;
        // The edit path's two carry-overs into the tail: the prepared source PCM that gets stitched
        // back outside the region, and the resolved region itself.
        let mut edit_state: Option<(Vec<f32>, EditGeometry)> = None;
        // Exactly zero on the text-only and restyle paths. A whole-clip restyle hands its source to
        // the sampler as `init_data`; supplying it here as masked local input as well would be the
        // inpaint contract, which is the arm below.
        let mut local = Tensor::zeros(
            (1, 1 + LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        if let Some(reference) = reference {
            let (_, latents) = self.prepare_and_encode(
                reference.samples,
                reference.sample_rate,
                reference.channels,
                &geometry,
                &mut noise,
                reference_validation,
                is_canceled,
            )?;
            init_latents = Some(latents);
            order = Some(ReferenceDrawOrder {
                draws_after_initial_noise,
                draws_after_source_encode: noise.draws(),
            });
        }
        if let Some(edit) = edit {
            let (prepared, latents) = self.prepare_and_encode(
                edit.samples,
                edit.sample_rate,
                edit.channels,
                &geometry,
                &mut noise,
                reference_validation,
                is_canceled,
            )?;
            order = Some(ReferenceDrawOrder {
                draws_after_initial_noise,
                draws_after_source_encode: noise.draws(),
            });
            let resolved = edit_geometry(&edit, &geometry, parameters.duration_secs)?;
            // The `duration_secs` argument one line up is weights-only and sets the padding
            // boundary; this compares the result against the sampling geometry's own,
            // independently-derived effective span. See `edit_geometry_matches_request`.
            edit_geometry_matches_request(&resolved, &geometry)?;
            local = edit_local_conditioning(&resolved, &latents, &self.device, self.dtypes.root)?;
            // `init_latents` stays `None`: an edit starts from pure seeded noise. The source
            // reaches the model through the local conditioner only.
            edit_state = Some((prepared, resolved));
        }
        // The handoff itself, checked where it *crosses* rather than where it was computed: with
        // `local` left as the zeros allocated above, every edit degrades to unconditioned
        // text-to-audio inside the region and every divergence measurement gets *better*. See
        // `edit_local_conditioning_is_present`.
        edit_local_conditioning_is_present(
            edit_state
                .as_ref()
                .is_some_and(|(_, resolved)| edit_retained_latent_count(resolved) > 0),
            tensor_has_nonzero(&local)?,
        )?;
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            init_latents.as_ref(),
            &local,
            // The reference itself, not a scalar derived here: `sample` runs `sampler_strength_for`
            // separately at each of its two consumer sites, so no single expression feeds both sides
            // of the sampler's schedule/strength cross-check. See that function. Dropping this
            // argument while keeping `init_latents` above is rejected by `reference_halves_agree`.
            reference.as_ref(),
            on_progress,
            &mut noise,
            is_canceled,
        )?;
        canceled(is_canceled)?;
        on_decoding();

        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        // The full small-variant config owns the default-on outer chunking decision. The same
        // request-local stream has already produced initial latents and every Pingpong draw.
        let decoded = self.same.decode_audio_with_request_rng(
            &sampled,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            &mut noise,
            is_canceled,
        )?;
        let audio = self.finish(decoded, &padding, parameters.duration_secs, is_canceled)?;
        // Frozen upstream stops one line above: it regenerates and decodes the whole clip. This
        // repository's `AudioEdit` contract promises the rest of the clip is preserved, so the
        // canonical prepared source goes back over everything outside the region, bit-exactly.
        let audio = match &edit_state {
            Some((prepared, resolved)) => stitch_outside_region(&audio, prepared, resolved)?,
            None => audio,
        };
        Ok((audio, order))
    }

    /// Conform and SAME-encode one source clip onto this request's own stream.
    ///
    /// Returns the prepared interleaved PCM alongside the latents because the edit path needs the
    /// exact same buffer again for [`stitch_outside_region`] — re-running `prepare_reference_pcm` at
    /// the tail would be a second, independently-drifting copy of the source's timeline.
    #[allow(clippy::too_many_arguments)]
    fn prepare_and_encode(
        &self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
        geometry: &SampleGeometry,
        noise: &mut SeededNoise,
        reference_validation: ReferenceInputValidation,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Vec<f32>, Tensor)> {
        canceled(is_canceled)?;
        let prepared = match reference_validation {
            ReferenceInputValidation::Required => {
                prepare_reference_pcm(samples, sample_rate, channels, geometry.sample_size)?
            }
            ReferenceInputValidation::Complete => prepare_validated_reference_pcm(
                samples,
                sample_rate,
                channels,
                geometry.sample_size,
            )?,
        };
        let planar = interleaved_to_planar(&prepared, geometry.sample_size, &self.device)?
            .to_dtype(self.dtypes.root)?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let latents = self.same.encode_audio_with_request_rng(
            &planar,
            SameChunkingPolicy::full_model_encode(diffusion.pretransform.chunked),
            SameChunkingParameters::default(),
            noise,
            is_canceled,
        )?;
        if latents.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "SAME-encoded source has shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                latents.dims(),
                geometry.latent_length
            )));
        }
        Ok((prepared, latents))
    }

    /// Replay the complete frozen-upstream path with explicit stochastic inputs.
    ///
    /// This exists for numerical-oracle tests. Production requests always use one request-local
    /// [`SeededNoise`] stream instead.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_controlled(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        initial: &Tensor,
        pingpong_noises: Vec<Tensor>,
        decode_noises: &[SameDecodeChunkNoise],
        on_progress: &mut dyn FnMut(usize, usize),
        on_decoding: &mut dyn FnMut(),
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Tensor, Vec<f32>)> {
        canceled(is_canceled)?;
        let geometry = self.geometry(parameters.duration_secs)?;
        if initial.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "controlled initial noise has shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                initial.dims(),
                geometry.latent_length
            )));
        }
        let initial = initial
            .to_device(&self.device)?
            .to_dtype(self.dtypes.root)?;
        let mut noise = InjectedNoise::new(pingpong_noises);
        // The replay path is text-only: no reference, no edit, so the local conditioning is zero.
        let local = Tensor::zeros(
            (1, 1 + LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            None,
            &local,
            // No reference: the replay path is text-only, and `sampler_strength_for` turns that
            // into the `1.0` this call used to spell out as a literal.
            None,
            on_progress,
            &mut noise,
            is_canceled,
        )?;
        if noise.draws() != parameters.steps {
            return Err(AudioError::Msg(format!(
                "controlled Pingpong replay consumed {} draws, expected {}",
                noise.draws(),
                parameters.steps
            )));
        }
        canceled(is_canceled)?;
        on_decoding();
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let decoded = self.same.decode_audio_with_chunk_noises(
            &sampled,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            decode_noises,
        )?;
        let audio = self.finish(decoded, &padding, parameters.duration_secs, is_canceled)?;
        Ok((sampled, audio))
    }

    /// Replay only the decode boundary with explicit latents and SAME noise.
    #[doc(hidden)]
    pub fn decode_controlled(
        &self,
        latents: &Tensor,
        duration_secs: f32,
        decode_noises: &[SameDecodeChunkNoise],
    ) -> Result<Vec<f32>> {
        let geometry = self.geometry(duration_secs)?;
        if latents.dims() != [1, LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "controlled latents have shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                latents.dims(),
                geometry.latent_length
            )));
        }
        let padding = padding_mask(
            &geometry.valid_lengths,
            geometry.latent_length,
            &self.device,
        )?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let decoded = self.same.decode_audio_with_chunk_noises(
            &latents
                .to_device(&self.device)?
                .to_dtype(self.dtypes.root)?,
            SameChunkingPolicy::full_model_decode(diffusion.pretransform.chunked, None),
            SameChunkingParameters::default(),
            decode_noises,
        )?;
        self.finish(decoded, &padding, duration_secs, &|| false)
    }

    /// Replay one unchunked SAME dispatch for oracle boundary diagnosis.
    #[doc(hidden)]
    pub fn decode_chunk_controlled(
        &self,
        latents: &Tensor,
        noise: &SameDecodeChunkNoise,
    ) -> Result<Tensor> {
        Ok(self.same.decode_with_noise(
            &latents
                .to_device(&self.device)?
                .to_dtype(self.dtypes.root)?,
            None,
            noise.regularization_noise.as_ref(),
            Some(&noise.mask_noises),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample<N: NoiseSource>(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        parameters: SynthesisParameters,
        geometry: &SampleGeometry,
        initial: &Tensor,
        init_latents: Option<&Tensor>,
        // The `[1, 257, latent_length]` local conditioning: exactly zero on the text-only and
        // restyle paths, the masked source on the edit path. Built by the caller rather than here
        // so the construction is reachable — and gated — without weights.
        local: &Tensor,
        // The request's reference, if any. `sample` takes the reference rather than a pre-computed
        // `strength` on purpose — see `sampler_strength_for`.
        reference: Option<&ReferenceAudio<'_>>,
        on_progress: &mut dyn FnMut(usize, usize),
        noise: &mut N,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Tensor, Tensor)> {
        // `init_latents` and `reference` are two forwarded halves of one decision: the caller
        // encodes the source *because* there is a reference, and the strength that decides what to
        // do with the encode is derived from that same reference below. Dropping either half at the
        // call site while keeping the other is a one-token edit that the weight-free suite cannot
        // see (`sample` needs weights). Dropping `reference` alone is the worse of the two — the
        // source is still encoded, strength becomes `1.0`, and the restyle silently degrades to
        // plain text-to-audio. This makes both halves fail closed instead.
        reference_halves_agree(init_latents.is_some(), reference.is_some())?;
        let positive = self
            .conditioner
            .encode_with_cancel(&[prompt.to_owned()], is_canceled)?;
        canceled(is_canceled)?;
        let negative = negative_prompt
            .map(|prompt| {
                self.conditioner
                    .encode_with_cancel(&[prompt.to_owned()], is_canceled)
            })
            .transpose()?;
        canceled(is_canceled)?;

        let seconds = Tensor::new(&[parameters.duration_secs], &self.device)?;
        if local.dims() != [1, 1 + LATENT_CHANNELS, geometry.latent_length] {
            return Err(AudioError::Msg(format!(
                "local conditioning has shape {:?}, expected [1,{},{}]",
                local.dims(),
                1 + LATENT_CHANNELS,
                geometry.latent_length
            )));
        }
        let padding = padding_mask(
            &geometry.valid_lengths,
            geometry.latent_length,
            &self.device,
        )?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        // The init noise level: `1.0` on the text-to-audio path, `1.0 - retention` on the reference
        // path. The schedule's first sigma must equal the strength handed to the DiT call below,
        // which the sampler enforces (`initialized_start`). The two sites call
        // `sampler_strength_for` separately rather than sharing a local on purpose — see that
        // function.
        let schedule = build_schedule(
            parameters.steps,
            sampler_strength_for(reference),
            &inference_shift(&diffusion.diffusion),
            geometry.effective_lengths.as_deref(),
            geometry.latent_length,
        )?;
        let total = parameters.steps;
        let mut progress = |step: &crate::sampler::SampleStep| -> Result<()> {
            canceled(is_canceled)?;
            on_progress(step.index + 1, total);
            Ok(())
        };
        let sampled = sample_dit_initialized_with_interval_and_cancel(
            &self.dit,
            parameters.sampler,
            initial,
            init_latents,
            // Recomputed, not read from a binding shared with `build_schedule` above: an edit to
            // this expression alone leaves it disagreeing with `schedule[0]`, which
            // `initialized_start` rejects.
            sampler_strength_for(reference),
            &schedule,
            &positive.embeddings,
            negative.as_ref().map(|value| &value.embeddings),
            negative.as_ref().map(|value| &value.attention_mask),
            &seconds,
            local,
            Some(&padding),
            parameters.guidance,
            GuidanceInterval::FULL,
            Some(&mut progress as &mut ProgressCallback<'_>),
            noise,
            false,
            is_canceled,
        )?;
        Ok((sampled.latents, padding))
    }

    fn geometry(&self, duration_secs: f32) -> Result<SampleGeometry> {
        default_sample_geometry(&self.config, &[Some(duration_secs as f64)])
    }

    fn finish(
        &self,
        decoded: Tensor,
        padding: &Tensor,
        duration_secs: f32,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>> {
        canceled(is_canceled)?;
        let decoded = apply_padding_mask(&decoded, padding, 4_096)?;
        let frames = ((duration_secs as f64) * SAMPLE_RATE as f64) as usize;
        let decoded = SameAutoencoder::crop_valid_prefix(&decoded, frames)?.clamp(-1.0, 1.0)?;
        planar_to_interleaved(&decoded)
    }
}

/// The conditioner `repo_id` declared by a full Stable Audio 3 snapshot's `model_config.json`.
///
/// This is the only field in the shipped configs that separates `small-music` from `small-sfx`:
/// they are otherwise architecturally identical, differing only in training-only ARC fields and
/// demo prompts.
///
/// It separates **nothing** on any post-trained/base pair — every base config declares its
/// post-trained sibling's repository — which is why
/// [`VariantGeometry::expected_conditioner_repo`] is a distinct field from
/// [`VariantGeometry::hub_repo`], and why [`VariantGeometry::expected_objective`] carries the
/// base/post-trained decision instead.
pub fn conditioner_repo_id(layout: &SnapshotLayout) -> Option<&str> {
    let diffusion = match &layout.config.model {
        ModelConfig::Diffusion(model) => model,
        ModelConfig::Autoencoder(_) => return None,
    };
    diffusion
        .conditioning
        .configs
        .iter()
        .find_map(|config| match config {
            ConditionerConfig::T5gemma { config, .. } => config.repo_id.as_deref(),
            ConditionerConfig::Number { .. } => None,
        })
}

/// Validate that `layout` is the exact SA3 checkpoint `geometry` describes.
///
/// Three independent discriminators, each per registered variant:
///
/// * **architecture + `sample_size`** — a small snapshot cannot authenticate as
///   `stable_audio_3_medium` on shape alone and vice versa, and the two small post-trained ids are
///   separated from their base siblings by `sample_size` (`5,292,032` against `5,324,800`);
/// * **`diffusion_objective`** — `rf_denoiser` for the post-trained ids, `rectified_flow` for the
///   `-base` ids. This is the *only* thing separating `stable_audio_3_medium` from
///   `stable-audio-3-medium-base` in the entire config, so it is parameterised per variant rather
///   than loosened: a global relaxation would open all three post-trained ids to their base
///   siblings at once;
/// * **conditioner `repo_id`** — separates the two architecturally identical smalls, and the two
///   architecturally identical small bases, from each other. It separates no post-trained/base pair
///   (see [`conditioner_repo_id`]).
///
/// `ModelRegistration::load` carries no provider id, so without this the registry would happily
/// serve one checkpoint's weights from another's registration.
pub fn validate_layout(layout: &SnapshotLayout, geometry: VariantGeometry) -> Result<()> {
    let expected_repo = geometry.hub_repo;
    let shape = geometry.shape;
    if layout.kind != SnapshotKind::Full
        || layout.config.sample_rate != SAMPLE_RATE
        || layout.config.sample_size != shape.sample_size
        || layout.config.audio_channels != CHANNELS
        || layout.keys.total != shape.total_keys
        || layout.keys.dit != shape.dit_keys
        || layout.keys.encoder + layout.keys.decoder + layout.keys.bottleneck
            != shape.autoencoder_keys
        || layout.keys.conditioner != 3
    {
        return Err(AudioError::Msg(format!(
            "snapshot is not the exact {expected_repo} full checkpoint"
        )));
    }
    let diffusion = match &layout.config.model {
        ModelConfig::Diffusion(model) => model,
        ModelConfig::Autoencoder(_) => {
            return Err(AudioError::Msg(format!(
                "{expected_repo} provider rejects standalone SAME snapshots"
            )));
        }
    };
    let dit = &diffusion.diffusion;
    let cfg = &dit.config;
    if dit.diffusion_objective != geometry.expected_objective
        || diffusion.io_channels != LATENT_CHANNELS
        || cfg.io_channels != LATENT_CHANNELS
        || cfg.embed_dim != shape.embed_dim
        || cfg.depth != shape.depth
        || cfg.num_heads != shape.num_heads
        || cfg.attn_kwargs.differential != shape.differential
        || cfg.cond_token_dim != 768
        || cfg.global_cond_dim != 768
        || cfg.local_add_cond_dim != Some(257)
        || diffusion.pretransform.config.latent_dim != LATENT_CHANNELS
        || diffusion.pretransform.config.downsampling_ratio != 4_096
        || !diffusion.pretransform.chunked
    {
        return Err(AudioError::Msg(format!(
            "snapshot architecture is not {expected_repo}"
        )));
    }
    let expected_conditioner_repo = geometry.expected_conditioner_repo;
    match conditioner_repo_id(layout) {
        Some(repo) if repo == expected_conditioner_repo => {}
        Some(repo) => {
            return Err(AudioError::Msg(format!(
                "snapshot declares conditioner repo_id {repo}, which is not \
                 {expected_conditioner_repo}; refusing to serve one Stable Audio 3 checkpoint \
                 under {expected_repo}'s provider id"
            )));
        }
        None => {
            return Err(AudioError::Msg(format!(
                "snapshot declares no conditioner repo_id and cannot be authenticated as \
                 {expected_repo}"
            )));
        }
    }
    let text = layout.text_keys.as_ref().ok_or_else(|| {
        AudioError::Msg(format!("{expected_repo} snapshot has no bundled T5Gemma"))
    })?;
    if text.total != crate::weights::TextWeightSummary::TOTAL
        || text.encoder != crate::weights::TextWeightSummary::ENCODER
        || text.decoder != crate::weights::TextWeightSummary::DECODER
    {
        return Err(AudioError::Msg(format!(
            "{expected_repo} bundled T5Gemma inventory mismatch"
        )));
    }
    Ok(())
}

fn canceled(is_canceled: &dyn Fn() -> bool) -> Result<()> {
    if is_canceled() {
        Err(AudioError::Canceled)
    } else {
        Ok(())
    }
}

fn apply_padding_mask(audio: &Tensor, padding: &Tensor, ratio: usize) -> Result<Tensor> {
    let (batch, channels, samples) = audio.dims3()?;
    let mask = padding
        .unsqueeze(1)?
        .unsqueeze(3)?
        .repeat((1, channels, 1, ratio))?
        .reshape((batch, channels, padding.dim(1)? * ratio))?;
    let mask = if mask.dim(2)? >= samples {
        mask.narrow(2, 0, samples)?
    } else {
        mask.pad_with_zeros(2, 0, samples - mask.dim(2)?)?
    };
    Ok(audio.broadcast_mul(&mask.to_dtype(audio.dtype())?)?)
}

/// Interleaved stereo PCM to the `[1, CHANNELS, frames]` channel-first tensor SAME encodes.
fn interleaved_to_planar(samples: &[f32], frames: usize, device: &Device) -> Result<Tensor> {
    if samples.len() != frames * CHANNELS {
        return Err(AudioError::Msg(format!(
            "prepared reference has {} samples, expected {}",
            samples.len(),
            frames * CHANNELS
        )));
    }
    let mut planar = Vec::with_capacity(samples.len());
    for channel in 0..CHANNELS {
        planar.extend((0..frames).map(|frame| samples[frame * CHANNELS + channel]));
    }
    Ok(Tensor::from_vec(planar, (1, CHANNELS, frames), device)?)
}

fn planar_to_interleaved(audio: &Tensor) -> Result<Vec<f32>> {
    let (batch, channels, frames) = audio.dims3()?;
    if batch != 1 || channels != CHANNELS {
        return Err(AudioError::Msg(format!(
            "SA3 small decode expected [1,{CHANNELS},frames], got [{batch},{channels},{frames}]"
        )));
    }
    let planar = audio.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output = Vec::with_capacity(frames * channels);
    for (&left, &right) in planar[0][0].iter().zip(&planar[0][1]) {
        output.push(left);
        output.push(right);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_mask_repeats_each_latent_over_its_complete_audio_span() {
        let device = Device::Cpu;
        let audio = Tensor::ones((1, 2, 12), DType::F32, &device).unwrap();
        let padding = Tensor::new(&[[1u8, 1, 0]], &device).unwrap();
        let masked = apply_padding_mask(&audio, &padding, 4)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(
            masked,
            vec![vec![
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]]
        );
    }

    #[test]
    fn validated_reference_preparation_does_not_repeat_the_finiteness_scan() {
        let samples = [0.25, -0.5, f32::NAN, 0.75];
        assert!(prepare_reference_pcm(&samples, SAMPLE_RATE, 2, 2).is_err());

        // The crate-private route is reachable only after model request validation. Feeding it the
        // value that the public route rejects makes the absence of a second full-slice scan
        // observable: equal-rate preparation copies the NaN instead of rediscovering it.
        let prepared = prepare_validated_reference_pcm(&samples, SAMPLE_RATE, 2, 2).unwrap();
        assert!(prepared[2].is_nan());
    }
}
