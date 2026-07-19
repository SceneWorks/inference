//! The `AudioTransform` contract — non-prompt audio→audio / audio→stems, the audio sibling of
//! [`Transform`](crate::transform::Transform). See `docs/MODEL_ARCHITECTURE.md` §3.3 / §9.
//!
//! Exactly the [`Transform`](crate::transform::Transform) rationale, one modality over: a restorer /
//! converter / separator is **not** a [`Generator`](crate::generator::Generator) — there is no
//! prompt, the input *audio* clip is the subject. §9 anticipated this fork as an additive extension
//! ("media-enum vs a parallel typed transform"): rather than generalize the shared image
//! [`Transform`](crate::transform::Transform) input/output into a `Image | AudioTrack |
//! Vec<AudioTrack>` media enum — which would ripple the trait signature through every existing image
//! transform impl and the candle-gen image lane that shares it — this is a **parallel** trait, so the
//! image lane is untouched. The choice mirrors sc-12838's [`VoiceEmbedder`](crate::voice_embed::VoiceEmbedder),
//! which added an audio-identity sibling next to the image face/CLIP embedders instead of widening them.
//!
//! Backend-neutral like every other gen-core contract — host types only ([`AudioTrack`] in and out,
//! no `mlx_rs::Array` / candle `Tensor`). The real transforms (an RVC voice converter, a stem
//! separator, an AudioLDM-2-class super-resolver) land in a `crates/audio` provider (sc-12844); this
//! contract is what they plug into.
//!
//! The family covers three non-prompt audio→audio shapes, distinguished by
//! [`AudioTransformKind`]:
//! - **voice conversion** ([`VoiceConversion`](AudioTransformKind::VoiceConversion)) — audio→audio;
//!   the target voice is either baked into the loaded weights (an RVC-style single-target model,
//!   like SeedVR2's fixed text embedding — the request carries only the source clip) **or** supplied
//!   per request as a tone-color reference clip via
//!   [`AudioTransformRequest::target_reference`] (a reference-based converter, OpenVoice V2 —
//!   sc-13223);
//! - **stem separation** ([`StemSeparation`](AudioTransformKind::StemSeparation)) — audio→`Vec`
//!   audio (vocals / drums / bass / other);
//! - **super-resolution / restoration / bandwidth-extension**
//!   ([`SuperResolution`](AudioTransformKind::SuperResolution)) — audio→audio, the direct SeedVR2
//!   analog (a low-rate/degraded clip restored to a higher target sample rate).
//!
//! Prompted audio editing (inpaint / extend / cover) is deliberately **not** here: that is the
//! [`Generator`](crate::generator::Generator) + conditioning shape (sc-12847), exactly as prompted
//! image editing is a `Generator`, not a `Transform`.

use crate::media::AudioTrack;
use crate::runtime::{CancelFlag, Progress};
use crate::Result;

/// A non-prompt audio→audio / audio→stems transform (voice conversion, stem separation,
/// super-resolution). The audio sibling of [`Transform`](crate::transform::Transform).
///
/// [`apply`](Self::apply) always returns a `Vec<AudioTrack>`: exactly one track for the single-output
/// kinds ([`VoiceConversion`](AudioTransformKind::VoiceConversion) /
/// [`SuperResolution`](AudioTransformKind::SuperResolution)) and one per stem for
/// [`StemSeparation`](AudioTransformKind::StemSeparation) — the single vs multi-output shape the
/// family requires, expressed by one signature rather than two methods.
pub trait AudioTransform {
    fn descriptor(&self) -> &AudioTransformDescriptor;
    fn validate(&self, req: &AudioTransformRequest) -> Result<()>;
    fn apply(
        &self,
        req: &AudioTransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<AudioTrack>>;
}

/// An audio-transform request — `Default`-able like
/// [`TransformRequest`](crate::transform::TransformRequest).
#[derive(Clone, Debug, Default)]
pub struct AudioTransformRequest {
    /// The source clip — the subject of the transform (there is no prompt).
    pub audio: AudioTrack,
    /// An optional **target tone-color reference** clip (additive, sc-13223). Some voice
    /// converters bake the target voice into the loaded weights (an RVC-style single-target
    /// model — `None` here), but a *reference-based* converter (OpenVoice V2) needs the target
    /// speaker supplied at request time: this clip is the "how should it sound" example whose
    /// timbre is transferred onto [`audio`](Self::audio)'s content. It is a plain host
    /// [`AudioTrack`] (tensor-free like every other request field); the source clip stays in
    /// [`audio`](Self::audio). Ignored by the kinds that do not consume a reference
    /// ([`StemSeparation`](AudioTransformKind::StemSeparation) /
    /// [`SuperResolution`](AudioTransformKind::SuperResolution)) and by weight-baked converters;
    /// a reference-based [`VoiceConversion`](AudioTransformKind::VoiceConversion) provider
    /// rejects a request whose `target_reference` is `None`.
    pub target_reference: Option<AudioTrack>,
    /// Output rate target (super-resolution / bandwidth-extension); [`AudioTarget::Preserve`] for
    /// rate-preserving kinds (voice conversion, stem separation).
    pub target: AudioTarget,
    /// Diffusion restorers (an AudioLDM-2-class super-resolver) use this; deterministic ones ignore
    /// it — mirrors [`TransformRequest::seed`](crate::transform::TransformRequest::seed).
    pub seed: Option<u64>,
    /// Model-defined restoration knob (0..1), the audio analog of SeedVR2 "softness".
    pub strength: Option<f32>,
    /// Diffusion restorers may be multi-step; override only if the model allows it.
    pub steps: Option<u32>,
    pub cancel: CancelFlag,
}

/// The output sample-rate target — the audio analog of
/// [`TargetSize`](crate::transform::TargetSize).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AudioTarget {
    /// Keep the input's sample rate (voice conversion, stem separation).
    #[default]
    Preserve,
    /// Restore/extend to this output sample rate (super-resolution / bandwidth extension).
    SampleRate(u32),
}

/// Which audio→audio shape a transform implements — the field that tells a caller how to interpret
/// [`AudioTransform::apply`]'s output (one track, or one per stem). The audio analog of the target-mode
/// flags carried by [`TransformCapabilities`](crate::transform::TransformCapabilities).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AudioTransformKind {
    /// audio→audio: convert the source into the loaded target voice (RVC). Single output.
    #[default]
    VoiceConversion,
    /// audio→`Vec`audio: separate the source into stems (vocals / drums / bass / other). Multi
    /// output — [`AudioTransformCapabilities::stem_count`] advertises how many.
    StemSeparation,
    /// audio→audio: super-resolution / restoration / bandwidth-extension — the direct SeedVR2
    /// analog. Single output.
    SuperResolution,
}

/// An audio transform's stable identity + advertised capabilities. Mirrors
/// [`TransformDescriptor`](crate::transform::TransformDescriptor) field-for-field.
#[derive(Clone, Debug)]
pub struct AudioTransformDescriptor {
    /// Stable id (e.g. `"rvc"`).
    pub id: &'static str,
    /// Provider family (e.g. `"audio"`).
    pub family: &'static str,
    /// Tensor backend that registered this transform (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement.
    pub backend: &'static str,
    pub capabilities: AudioTransformCapabilities,
}

/// What shape / knobs an audio transform supports. Mirrors
/// [`TransformCapabilities`](crate::transform::TransformCapabilities).
#[derive(Clone, Debug, Default)]
pub struct AudioTransformCapabilities {
    /// Which of the three audio→audio shapes this transform is.
    pub kind: AudioTransformKind,
    /// For [`StemSeparation`](AudioTransformKind::StemSeparation): the number of stems produced
    /// (≥ 2). `0` for the single-output kinds. Checked for coherence by the descriptor conformance
    /// sweep.
    pub stem_count: u16,
    /// Uses a seed (diffusion-based, e.g. an AudioLDM-2-class super-resolver).
    pub is_diffusion: bool,
    /// Honors [`AudioTransformRequest::strength`].
    pub supports_strength: bool,
    /// Supports an [`AudioTarget::SampleRate`] target (super-resolution / bandwidth extension).
    pub supports_resample: bool,
    /// Whether this transform only runs on macOS (an MLX implementation); a candle implementation
    /// sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory transform covering one kind. `apply` is exercised without a tensor
    /// backend: it returns `output_tracks` copies of the input clip (one for the single-output kinds,
    /// `stem_count` for separation), and drives one progress tick so the seam is covered.
    struct StubAudioTransform {
        descriptor: AudioTransformDescriptor,
        output_tracks: usize,
    }

    impl AudioTransform for StubAudioTransform {
        fn descriptor(&self) -> &AudioTransformDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &AudioTransformRequest) -> Result<()> {
            Ok(())
        }
        fn apply(
            &self,
            req: &AudioTransformRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> Result<Vec<AudioTrack>> {
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let rate = match req.target {
                AudioTarget::Preserve => req.audio.sample_rate,
                AudioTarget::SampleRate(r) => r,
            };
            Ok(vec![
                AudioTrack {
                    sample_rate: rate,
                    ..req.audio.clone()
                };
                self.output_tracks
            ])
        }
    }

    fn track(samples: usize, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn voice_conversion_is_audio_to_single_audio() {
        let t = StubAudioTransform {
            descriptor: AudioTransformDescriptor {
                id: "stub_vc",
                family: "audio",
                backend: "candle",
                capabilities: AudioTransformCapabilities {
                    kind: AudioTransformKind::VoiceConversion,
                    ..Default::default()
                },
            },
            output_tracks: 1,
        };
        let out = t
            .apply(
                &AudioTransformRequest {
                    audio: track(8, 24_000),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(
            t.descriptor().capabilities.kind,
            AudioTransformKind::VoiceConversion
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sample_rate, 24_000);
        assert_eq!(out[0].samples.len(), 8);
    }

    #[test]
    fn stem_separation_is_audio_to_many_audio() {
        let t = StubAudioTransform {
            descriptor: AudioTransformDescriptor {
                id: "stub_stems",
                family: "audio",
                backend: "candle",
                capabilities: AudioTransformCapabilities {
                    kind: AudioTransformKind::StemSeparation,
                    stem_count: 4,
                    ..Default::default()
                },
            },
            output_tracks: 4,
        };
        let out = t
            .apply(
                &AudioTransformRequest {
                    audio: track(16, 44_100),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(t.descriptor().capabilities.stem_count as usize, out.len());
        assert!(out.iter().all(|s| s.sample_rate == 44_100));
    }

    #[test]
    fn super_resolution_extends_to_the_target_rate() {
        let t = StubAudioTransform {
            descriptor: AudioTransformDescriptor {
                id: "stub_sr",
                family: "audio",
                backend: "candle",
                capabilities: AudioTransformCapabilities {
                    kind: AudioTransformKind::SuperResolution,
                    supports_resample: true,
                    is_diffusion: true,
                    ..Default::default()
                },
            },
            output_tracks: 1,
        };
        let out = t
            .apply(
                &AudioTransformRequest {
                    audio: track(8, 16_000),
                    target: AudioTarget::SampleRate(48_000),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].sample_rate, 48_000);
    }

    #[test]
    fn audio_target_defaults_to_preserve() {
        assert_eq!(AudioTarget::default(), AudioTarget::Preserve);
        assert_eq!(
            AudioTransformKind::default(),
            AudioTransformKind::VoiceConversion
        );
    }
}
