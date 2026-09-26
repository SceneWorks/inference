//! The production YuE2 decode path (sc-22993): cached [`AcousticLatents`] → 48 kHz stereo audio,
//! with the decoder and latent identities carried in the output metadata.
//!
//! This is upstream `YuE2Pipeline.decode` natively: verify the latents, decode `[1, 64, T]` with
//! the selected VAE (halo/crop tiles by default, or the full reference decode), refuse non-finite
//! audio, `clamp(-1, 1)`, and return channel-last samples (`[S, 2]`, i.e. interleaved
//! `L0 R0 L1 R1 …`; channel 0 is left). The output length is the decoder's natural length
//! `1920·T − 64` — no padding, no trimming.
//!
//! [`decode_latents`] takes only a loaded [`Yue2Vae`] and the latents: nothing about planning,
//! semantic generation or synthesis is an input, so switching between the standard and legacy
//! decoder re-decodes the *same* verified latents (upstream: "decode the same cached latents
//! instead of generating a new song") and the two outputs share one latent identity in their
//! metadata while their decoder identities differ.

use candle_audio::candle_core::{DType, Device};
use serde_json::{json, Value};

use crate::latent::{AcousticLatents, LatentIdentity};
use crate::vae::{
    DecoderIdentity, VaeError, Yue2Vae, AUDIO_CHANNELS, DEFAULT_CORE_FRAMES, DEFAULT_HALO_FRAMES,
    SAMPLE_RATE,
};

/// How the VAE runs over the latents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Exact-boundary halo/crop tiles of `core_frames` latent frames (upstream's default,
    /// `vae_decode = "halo_crop"`). Bounds decoder activation memory by the tile, not the song.
    Tiled {
        /// Latent frames per tile core.
        core_frames: usize,
    },
    /// One full-length FP32 decode — the reference path retained for fidelity validation
    /// (upstream `decode(full=True)`).
    Full,
}

/// Decode settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Tiled or full.
    pub mode: DecodeMode,
    /// Latent frames of context per tile side (tiled mode; at least the decoder's required halo).
    pub halo_frames: usize,
}

impl DecodeOptions {
    /// Upstream's defaults: tiles of 1024 frames with a 16-frame halo.
    pub fn production() -> Self {
        Self {
            mode: DecodeMode::Tiled {
                core_frames: DEFAULT_CORE_FRAMES,
            },
            halo_frames: DEFAULT_HALO_FRAMES,
        }
    }

    /// Upstream's memory-budget rule (`YuE2Pipeline.__init__`): 512-frame cores when the budget is
    /// at most 12 GiB, otherwise 1024.
    pub fn for_memory_budget_gib(budget_gib: f64) -> Self {
        Self {
            mode: DecodeMode::Tiled {
                core_frames: if budget_gib <= 12.0 { 512 } else { 1024 },
            },
            halo_frames: DEFAULT_HALO_FRAMES,
        }
    }

    /// The full reference FP32 decode.
    pub fn reference_full() -> Self {
        Self {
            mode: DecodeMode::Full,
            halo_frames: DEFAULT_HALO_FRAMES,
        }
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::production()
    }
}

/// What produced a [`DecodedAudio`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeMetadata {
    /// The decoder that ran.
    pub decoder: DecoderIdentity,
    /// The latents it decoded.
    pub latents: LatentIdentity,
    /// Sample rate (48000).
    pub sample_rate: u32,
    /// Channels (2).
    pub channels: usize,
    /// Samples per channel.
    pub samples: usize,
    /// How it ran.
    pub mode: DecodeMode,
    /// Halo frames (tiled mode).
    pub halo_frames: Option<usize>,
    /// Samples (over both channels) whose raw value lay outside `[-1, 1]` and were clamped.
    pub clamped_samples: usize,
}

impl DecodeMetadata {
    /// Upstream's `vae_decode` label.
    pub fn vae_decode(&self) -> &'static str {
        match self.mode {
            DecodeMode::Tiled { .. } => "halo_crop",
            DecodeMode::Full => "full",
        }
    }

    /// The metadata as JSON, with upstream's effective-config key names where they exist
    /// (`vae_dtype`, `vae_decode`, `vae_core_frames`, `vae_halo_frames`, `decoder_release`).
    pub fn to_json(&self) -> Value {
        let (core, halo) = match self.mode {
            DecodeMode::Tiled { core_frames } => (json!(core_frames), json!(self.halo_frames)),
            DecodeMode::Full => (Value::Null, Value::Null),
        };
        json!({
            "decoder": self.decoder.to_json(),
            "decoder_release": self.decoder.release(),
            "latent": self.latents.to_json(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "vae_dtype": "float32",
            "vae_decode": self.vae_decode(),
            "vae_core_frames": core,
            "vae_halo_frames": halo,
            "clamped_samples": self.clamped_samples,
        })
    }
}

/// Decoded 48 kHz stereo audio.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    samples: Vec<f32>,
    metadata: DecodeMetadata,
}

impl DecodedAudio {
    /// Interleaved samples `L0 R0 L1 R1 …` (upstream's `[S, 2]` array, row-major), in `[-1, 1]`.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// One channel (0 = left, 1 = right).
    pub fn channel(&self, channel: usize) -> Vec<f32> {
        assert!(channel < AUDIO_CHANNELS, "channel {channel} out of range");
        self.samples
            .iter()
            .skip(channel)
            .step_by(AUDIO_CHANNELS)
            .copied()
            .collect()
    }

    /// Samples per channel.
    pub fn frames(&self) -> usize {
        self.metadata.samples
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.metadata.samples as f64 / self.metadata.sample_rate as f64
    }

    /// The provenance.
    pub fn metadata(&self) -> &DecodeMetadata {
        &self.metadata
    }
}

/// Decode cached latents to 48 kHz stereo (see the [module docs](self)). The latents are
/// re-verified against their identity first; `cancel` is polled before each tile and
/// `on_progress(completed, total)` runs after each (a full decode is one tile).
pub fn decode_latents(
    vae: &Yue2Vae,
    latents: &AcousticLatents,
    options: &DecodeOptions,
    cancel: &dyn Fn() -> bool,
    on_progress: &mut dyn FnMut(usize, usize),
) -> Result<DecodedAudio, VaeError> {
    latents.verify()?;
    let z = latents.to_decoder_input(vae.device())?;
    let raw = match options.mode {
        DecodeMode::Tiled { core_frames } => {
            vae.decode_tiled(&z, core_frames, options.halo_frames, cancel, on_progress)?
        }
        DecodeMode::Full => {
            if cancel() {
                return Err(VaeError::Cancelled {
                    completed: 0,
                    total: 1,
                });
            }
            let audio = vae.decode_full(&z)?;
            on_progress(1, 1);
            audio
        }
    };
    let expected = vae.natural_output_length(latents.frames())?;
    if raw.dims() != [1, AUDIO_CHANNELS, expected] {
        return Err(VaeError::Input(format!(
            "decoder returned {:?}, expected [1, {AUDIO_CHANNELS}, {expected}]",
            raw.dims()
        )));
    }
    let planar = raw
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (samples, clamped) = interleave_clamped(&planar, expected)?;
    Ok(DecodedAudio {
        samples,
        metadata: DecodeMetadata {
            decoder: vae.identity().clone(),
            latents: latents.identity().clone(),
            sample_rate: SAMPLE_RATE,
            channels: AUDIO_CHANNELS,
            samples: expected,
            mode: options.mode,
            halo_frames: match options.mode {
                DecodeMode::Tiled { .. } => Some(options.halo_frames),
                DecodeMode::Full => None,
            },
            clamped_samples: clamped,
        },
    })
}

/// Planar `[2, S]` raw decoder output → interleaved, clamped samples plus the clamped count.
/// Any non-finite sample is an error (checked before clamping, which would hide `±inf`).
fn interleave_clamped(planar: &[f32], samples: usize) -> Result<(Vec<f32>, usize), VaeError> {
    let mut out = vec![0f32; planar.len()];
    let mut clamped = 0;
    for channel in 0..AUDIO_CHANNELS {
        for (i, &v) in planar[channel * samples..(channel + 1) * samples]
            .iter()
            .enumerate()
        {
            if !v.is_finite() {
                return Err(VaeError::NonFiniteAudio { channel, sample: i });
            }
            if !(-1.0..=1.0).contains(&v) {
                clamped += 1;
            }
            out[i * AUDIO_CHANNELS + channel] = v.clamp(-1.0, 1.0);
        }
    }
    Ok((out, clamped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::VaeVariant;
    use crate::latent::{LatentSource, LATENT_FILE};
    use crate::vae::tests::{fixture_meta, reference, tiny};
    use crate::vae::{variant_name, VaeParts};

    /// Same bound and reasoning as the VAE module's `TINY_TOL`: measured native-vs-PyTorch max |Δ|
    /// ~1e-6 on the tiny models; structural defects move samples by ≥ 0.1.
    const TOL: f32 = 1e-5;

    fn latents() -> AcousticLatents {
        AcousticLatents::from_tensor(
            &reference()["latent"],
            LatentSource::Synthesis {
                stage_identity: "tiny-fixture".into(),
            },
        )
        .unwrap()
    }

    fn tiled() -> DecodeOptions {
        DecodeOptions {
            mode: DecodeMode::Tiled {
                core_frames: fixture_meta()["core_frames"].as_u64().unwrap() as usize,
            },
            halo_frames: 16,
        }
    }

    fn decode(vae: &Yue2Vae, l: &AcousticLatents, o: &DecodeOptions) -> DecodedAudio {
        decode_latents(vae, l, o, &|| false, &mut |_, _| {}).unwrap()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The production path's waveform — natural length, interleaved L/R, clamped — equals upstream
    /// `YuE2Pipeline.decode`'s `[S, 2]` output for both decoders from the same latents.
    /// Mutations (each run): swap the interleave channel index → red; drop the clamp → red
    /// (upstream is clamped and ~10% of the raw waveform exceeds ±1); emit planar instead of interleaved → red.
    #[test]
    fn production_decode_matches_upstream_pipeline_for_both_decoders() {
        let r = reference();
        let l = latents();
        let meta = fixture_meta();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let out = decode(&vae, &l, &tiled());
            let name = variant_name(v);
            let want = r[&format!("{name}.pipeline_tiled")]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let natural = meta["variants"][name]["natural_output_length"]
                .as_u64()
                .unwrap() as usize;
            assert_eq!(out.frames(), natural);
            assert_eq!(out.frames(), 1920 * l.frames() - 64);
            assert_eq!(out.samples().len(), natural * 2);
            assert_eq!(r[&format!("{name}.pipeline_tiled")].dims(), &[natural, 2]);
            let err = max_abs(out.samples(), &want);
            println!("{v:?}: production decode vs upstream pipeline max|Δ| = {err:e}");
            assert!(err < TOL, "{v:?}: {err}");
            assert!(out.samples().iter().all(|s| s.is_finite()));
        }
    }

    /// Clamping: every output sample is in [-1, 1], the count of clamped samples equals the raw
    /// decoder's out-of-range count, and clamped samples sit exactly at ±1 with the raw sign
    /// (mutation: clamp to [-0.99, 0.99] or skip counting → red).
    #[test]
    fn output_is_clamped_to_unit_range() {
        let l = latents();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let out = decode(&vae, &l, &DecodeOptions::reference_full());
            let raw = vae
                .decode_full(&l.to_decoder_input(&Device::Cpu).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap()
                .t()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let beyond = raw.iter().filter(|x| x.abs() > 1.0).count();
            assert!(beyond > 0, "{v:?}: fixture must overshoot ±1");
            assert_eq!(out.metadata().clamped_samples, beyond);
            for (o, r) in out.samples().iter().zip(&raw) {
                assert!((-1.0..=1.0).contains(o));
                if r.abs() > 1.0 {
                    assert_eq!(*o, r.signum());
                } else {
                    assert_eq!(o, r);
                }
            }
        }
    }

    /// Channel order: `channel(0)` is the decoder's output channel 0 (left) and `channel(1)` its
    /// channel 1, and the two differ (mutation: `skip(1 - channel)` → red).
    #[test]
    fn channel_accessor_preserves_left_right_order() {
        let l = latents();
        let vae = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        let out = decode(&vae, &l, &DecodeOptions::reference_full());
        let raw = vae
            .decode_full(&l.to_decoder_input(&Device::Cpu).unwrap())
            .unwrap()
            .clamp(-1f32, 1f32)
            .unwrap();
        for c in 0..2 {
            let want = raw
                .get(0)
                .unwrap()
                .get(c)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(out.channel(c), want, "channel {c}");
        }
        assert!(max_abs(&out.channel(0), &out.channel(1)) > 0.1);
    }

    /// Tiled production decode and the full reference decode agree (the full FP32 path stays
    /// selectable for fidelity validation); metadata names the mode.
    #[test]
    fn tiled_and_full_reference_paths_agree() {
        let l = latents();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let tiled_out = decode(&vae, &l, &tiled());
            let full_out = decode(&vae, &l, &DecodeOptions::reference_full());
            let err = max_abs(tiled_out.samples(), full_out.samples());
            println!("{v:?}: tiled vs full production max|Δ| = {err:e}");
            assert!(err < TOL, "{v:?}: {err}");
            assert_eq!(tiled_out.metadata().vae_decode(), "halo_crop");
            assert_eq!(full_out.metadata().vae_decode(), "full");
            assert_eq!(
                full_out.metadata().to_json()["vae_core_frames"],
                Value::Null
            );
            assert_eq!(tiled_out.metadata().to_json()["vae_halo_frames"], json!(16));
        }
        assert_eq!(
            DecodeOptions::for_memory_budget_gib(12.0).mode,
            DecodeMode::Tiled { core_frames: 512 }
        );
        assert_eq!(
            DecodeOptions::default(),
            DecodeOptions::for_memory_budget_gib(24.0)
        );
    }

    /// Decoder identity is in the output metadata and differs between the two decoders, while the
    /// latent identity is the one that was decoded (mutation: record the other decoder's identity,
    /// or a fresh latent identity → red).
    #[test]
    fn metadata_preserves_decoder_and_latent_identity() {
        let l = latents();
        let std_vae = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        let leg_vae = tiny(VaeVariant::Legacy, VaeParts::DecoderOnly);
        let a = decode(&std_vae, &l, &tiled());
        let b = decode(&leg_vae, &l, &tiled());
        assert_eq!(a.metadata().decoder, *std_vae.identity());
        assert_eq!(b.metadata().decoder, *leg_vae.identity());
        assert_eq!(a.metadata().decoder.release(), "standard");
        assert_eq!(b.metadata().decoder.release(), "legacy");
        assert_ne!(
            a.metadata().decoder.weights_sha256,
            b.metadata().decoder.weights_sha256
        );
        let meta = fixture_meta();
        assert_eq!(
            a.metadata().decoder.weights_sha256,
            meta["variants"]["standard"]["weights_sha256"]
                .as_str()
                .unwrap()
        );
        assert_eq!(a.metadata().latents, *l.identity());
        assert_eq!(b.metadata().latents, *l.identity());
        let j = b.metadata().to_json();
        assert_eq!(j["decoder_release"], "legacy");
        assert_eq!(j["decoder"]["decoder_release"], "legacy");
        assert_eq!(j["latent"]["sha256"], json!(l.identity().sha256));
        assert_eq!(j["sample_rate"], 48000);
        assert_eq!(j["channels"], 2);
        assert_eq!(j["vae_dtype"], "float32");
        assert_eq!(a.metadata().sample_rate, 48_000);
        assert!((a.duration_seconds() - a.frames() as f64 / 48_000.0).abs() < 1e-12);
    }

    /// Switching decoder reuses the persisted, verified latents: one saved artifact is loaded and
    /// decoded by both decoders; the artifact bytes are unchanged afterwards, both outputs carry
    /// the same latent identity, and each output equals decoding the in-memory latents directly —
    /// no planning, semantic or synthesis input exists on the decode path.
    #[test]
    fn decoder_switch_reuses_verified_cached_latents() {
        let dir = tempfile::tempdir().unwrap();
        let original = latents();
        original.save(dir.path()).unwrap();
        let before = std::fs::read(dir.path().join(LATENT_FILE)).unwrap();

        let cached = AcousticLatents::load(dir.path()).unwrap();
        let mut outputs = Vec::new();
        for v in [
            VaeVariant::Standard,
            VaeVariant::Legacy,
            VaeVariant::Standard,
        ] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let from_cache = decode(&vae, &cached, &tiled());
            let direct = decode(&vae, &original, &tiled());
            assert_eq!(from_cache.samples(), direct.samples());
            assert_eq!(from_cache.metadata().latents, *original.identity());
            outputs.push(from_cache);
        }
        assert_eq!(std::fs::read(dir.path().join(LATENT_FILE)).unwrap(), before);
        assert_eq!(AcousticLatents::load(dir.path()).unwrap(), cached);
        // Switching back to the standard decoder reproduces its first output exactly.
        assert_eq!(outputs[0], outputs[2]);
        assert!(max_abs(outputs[0].samples(), outputs[1].samples()) > 0.1);
    }

    /// Latents whose content no longer matches their identity are refused before decoding.
    #[test]
    fn tampered_latents_are_refused() {
        let mut l = latents();
        l.values_mut_for_test()[3] += 1.0;
        let vae = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        let err = decode_latents(&vae, &l, &tiled(), &|| false, &mut |_, _| {}).unwrap_err();
        assert!(matches!(err, VaeError::Latent(_)), "{err}");
    }

    /// Non-finite decoder output is an error naming the sample, never silently clamped (upstream
    /// raises "VAE produced non-finite audio"); `+inf` would otherwise clamp to 1.0 (mutation:
    /// drop the finite check → red).
    #[test]
    fn non_finite_audio_is_an_error() {
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut planar = vec![0.5f32; 8];
            planar[6] = bad;
            let err = interleave_clamped(&planar, 4).unwrap_err();
            assert!(
                matches!(
                    err,
                    VaeError::NonFiniteAudio {
                        channel: 1,
                        sample: 2
                    }
                ),
                "{err}"
            );
        }
        let (out, clamped) = interleave_clamped(&[0.5, 2.0, -0.25, -3.0], 2).unwrap();
        assert_eq!(out, vec![0.5, -0.25, 1.0, -1.0]);
        assert_eq!(clamped, 2);
    }

    /// Cancellation surfaces from both modes; progress reaches the caller.
    #[test]
    fn cancellation_and_progress_propagate() {
        let l = latents();
        let vae = tiny(VaeVariant::Legacy, VaeParts::DecoderOnly);
        for o in [tiled(), DecodeOptions::reference_full()] {
            let err = decode_latents(&vae, &l, &o, &|| true, &mut |_, _| {}).unwrap_err();
            assert!(
                matches!(err, VaeError::Cancelled { completed: 0, .. }),
                "{err}"
            );
        }
        let mut seen = Vec::new();
        decode_latents(&vae, &l, &tiled(), &|| false, &mut |c, t| seen.push((c, t))).unwrap();
        assert_eq!(seen, vec![(1, 3), (2, 3), (3, 3)]);
    }
}
