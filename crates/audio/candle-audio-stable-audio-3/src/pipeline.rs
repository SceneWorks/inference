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
    ProgressCallback, SampleGeometry, SamplerKind, SeededNoise,
};
use crate::t5gemma::T5GemmaConditioner;
use crate::weights::{SnapshotKind, SnapshotLayout};

pub const SAMPLE_RATE: u32 = 44_100;
pub const CHANNELS: usize = 2;
pub const LATENT_CHANNELS: usize = 256;
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
/// The text side was never part of that decision. sc-14537 pinned the canonical cross-runtime text
/// policy — BF16 weights on disk, F32 compute, one BF16 rounding at the raw-embedding boundary —
/// and `tests/text_oracle.rs` gates it against the frozen Transformers 5.8.0 oracle on CPU and
/// Metal. Switching T5Gemma to BF16 *compute* would move a surface with a numeric parity gate behind
/// it for no memory win worth having: the encoder is 281 MB of medium's 10.4 GB resident set.
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
/// inversion on all six ids — left the whole weight-free suite green. It is *also* invisible to the
/// sampler's own `schedule[0] == strength` cross-check, because both downstream consumers
/// ([`build_schedule`] and the DiT sample call) read this same scalar, so an inversion moves them
/// together and they still agree. Mutating either consumer alone is loud; mutating their shared
/// input was silent. Extracted here it is gated weight-free by `tests/reference_audio.rs`
/// `the_pipeline_hands_the_sampler_the_converted_noise_level_as_strength`, which feeds it the output
/// of `reference_audio_for` so the whole request → sampler-strength chain is one assertion.
///
/// # And it is the *only* site that produces the value
///
/// Extracting the decision is not on its own enough: a `let strength = sampler_strength_for(..)`
/// forwarded into a weights-only helper just moves the silent site to the forwarding argument, where
/// the same "both consumers move together" reasoning still applies. So the scalar never crosses a
/// call boundary — `StableAudio3Pipeline::sample` takes `Option<&ReferenceAudio>` and calls this,
/// and the text-only replay path passes `None` instead of spelling `1.0` out a second time. Every
/// remaining use of `strength` in this file is inside `sample`, where mutating one alone trips the
/// sampler's `schedule[0] == strength` check.
///
/// The `None` arm is the text-to-audio path: `1.0` is pure noise, i.e. no source influence, which is
/// exactly what "no reference" means in the sampler's orientation.
pub fn sampler_strength_for(reference: Option<&ReferenceAudio<'_>>) -> f32 {
    reference.map_or(1.0, |reference| reference.noise_level)
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
/// 1. **Resample the whole buffer** to [`SAMPLE_RATE`] through the shared
///    [`candle_audio::dsp::resample`]. Off-rate source audio is converted, never rejected: the
///    lane's other two audio generators emit 48 kHz, so rejecting would refuse audio produced one
///    step earlier in the same product, and the 160:147 stereo ratio is already gated in
///    `candle-audio`'s own resampler tests. (ACE-Step's "must be 48000 Hz" is a missing resampler,
///    not a policy — that crate contains no DSP resampling of any kind.)
/// 2. **Trim or right-zero-pad from offset 0** to exactly `target_frames`, which is the adapted
///    sample size for the *requested* duration. Source extent never moves the geometry.
/// 3. **Conform channels after padding**: mono duplicates, stereo passes through, more than two
///    channels keeps the first two.
///
/// Steps 1 and 2 genuinely depend on their predecessor and their results are asserted. Step 3's
/// *position* is not observable and is not claimed to be gated: because the pad value is zero,
/// conforming before padding and conforming after it produce byte-identical output (duplicating a
/// zero and padding with zeros commute, as does taking the first two of four zeros). The spec's
/// "conform after padding" bullet is therefore satisfied here **by construction, not by a test** —
/// stated plainly so nobody reads the ordering as load-bearing and builds on it. It would only
/// become observable if the pad value ever stopped being zero.
///
/// Returns interleaved stereo of exactly `target_frames * CHANNELS` values.
pub fn prepare_reference_pcm(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    target_frames: usize,
) -> Result<Vec<f32>> {
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
    let resampled = candle_audio::dsp::resample(samples, sample_rate, SAMPLE_RATE, channels)?;
    let available = (resampled.len() / source_channels).min(target_frames);
    let mut conformed = vec![0.0f32; target_frames * source_channels];
    conformed[..available * source_channels]
        .copy_from_slice(&resampled[..available * source_channels]);
    let mut output = vec![0.0f32; target_frames * CHANNELS];
    for frame in 0..target_frames {
        let source = &conformed[frame * source_channels..(frame + 1) * source_channels];
        let (left, right) = match source_channels {
            1 => (source[0], source[0]),
            _ => (source[0], source[1]),
        };
        output[frame * CHANNELS] = left;
        output[frame * CHANNELS + 1] = right;
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
        validate_layout(layout, geometry)?;
        let diffusion = match &layout.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        let builders = layout.full_pipeline_builders(dtypes.root, dtypes.text, device)?;
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
        canceled(is_canceled)?;
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
        let (init_latents, order) = match reference {
            Some(reference) => {
                let latents =
                    self.reference_latents(&reference, &geometry, &mut noise, is_canceled)?;
                (
                    Some(latents),
                    Some(ReferenceDrawOrder {
                        draws_after_initial_noise,
                        draws_after_source_encode: noise.draws(),
                    }),
                )
            }
            None => (None, None),
        };
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            init_latents.as_ref(),
            // The reference itself, not a scalar derived here: `sample` runs the one shipped
            // decision (`sampler_strength_for`) so the sampler-facing value never crosses a
            // weights-only call boundary as a bare local. See that function for why forwarding the
            // scalar was silent to *both* the weight-free suite and the sampler's own
            // schedule/strength cross-check.
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
        Ok((audio, order))
    }

    /// Conform and SAME-encode one source clip onto this request's own stream.
    fn reference_latents(
        &self,
        reference: &ReferenceAudio<'_>,
        geometry: &SampleGeometry,
        noise: &mut SeededNoise,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<Tensor> {
        canceled(is_canceled)?;
        let prepared = prepare_reference_pcm(
            reference.samples,
            reference.sample_rate,
            reference.channels,
            geometry.sample_size,
        )?;
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
                "SAME-encoded reference has shape {:?}, expected [1,{LATENT_CHANNELS},{}]",
                latents.dims(),
                geometry.latent_length
            )));
        }
        Ok(latents)
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
        let (sampled, padding) = self.sample(
            prompt,
            negative_prompt,
            parameters,
            &geometry,
            &initial,
            None,
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
        // The request's reference, if any. `sample` takes the reference rather than a pre-computed
        // `strength` on purpose — see `sampler_strength_for`.
        reference: Option<&ReferenceAudio<'_>>,
        on_progress: &mut dyn FnMut(usize, usize),
        noise: &mut N,
        is_canceled: &dyn Fn() -> bool,
    ) -> Result<(Tensor, Tensor)> {
        let strength = sampler_strength_for(reference);
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
        // Exactly zero on every path, reference or not. A whole-clip restyle hands its source to the
        // sampler as `init_data`; supplying it here as masked local input as well would be a
        // different (inpaint) conditioning contract, which sc-14548 owns.
        let local = Tensor::zeros(
            (1, 1 + LATENT_CHANNELS, geometry.latent_length),
            self.dtypes.root,
            &self.device,
        )?;
        let padding = padding_mask(
            &geometry.valid_lengths,
            geometry.latent_length,
            &self.device,
        )?;
        let diffusion = match &self.config.model {
            ModelConfig::Diffusion(model) => model,
            ModelConfig::Autoencoder(_) => unreachable!("validated full snapshot"),
        };
        // `strength` is the init noise level: `1.0` on the text-to-audio path, `1.0 - retention` on
        // the reference path. The schedule's first sigma must equal it, which the sampler enforces.
        let schedule = build_schedule(
            parameters.steps,
            strength,
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
            strength,
            &schedule,
            &positive.embeddings,
            negative.as_ref().map(|value| &value.embeddings),
            negative.as_ref().map(|value| &value.attention_mask),
            &seconds,
            &local,
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
}
