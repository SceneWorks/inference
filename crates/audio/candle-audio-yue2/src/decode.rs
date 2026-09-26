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

/// Peak-memory growth per latent frame of the decode tile (bytes): 28 MiB.
///
/// Measured 2026-09-26 on the **Candle CPU backend** (release build, Apple M-series), standard
/// decoder (the legacy decoder has the same architecture and tensor shapes), with the `#[ignore]`d
/// `tile_footprint_probe` in `tests/vae_real_weights.rs`. Each run is one process; peak RSS comes
/// from `/usr/bin/time -l`.
///
/// * Production-shaped tiled decodes (several consecutive tiles, 16-frame halo), by tile frames
///   (core + 32): 64 → 1.75 GB, 128 → 3.36 GB, 192 → 4.99 GB, 256 → 6.43 GB (3 tiles) and
///   6.76 GB (7 tiles), 352 → 8.50 GB. Incremental slopes are 25.1, 25.5, 22.5–27.6 and
///   18.1–21.6 MB per frame.
/// * A single full-length decode is cheaper: N = 1 / 64 / 128 / 256 / 384 frames gave
///   0.80 / 1.50 / 2.48 / 4.74 / 6.86 GB, about 17 MB per frame. Consecutive tiles hold roughly
///   1.5–2 GB more, which is allocator retention of the previous tile's buffers. That is why the
///   multi-tile series sets the constant.
///
/// The constant rounds the steepest multi-tile slope up (27.6 → 29.4 MB). The dominant terms are
/// the 64-channel stages at 1920 samples per frame: im2col buffers of the k7 convolutions and the
/// SnakeBeta temporaries. Treat it as a conservative bound for other backends; cross-platform
/// admission is calibrated elsewhere (sc-23001). This constant only has to make the tiling honour
/// the budget it is given.
pub const TILE_BYTES_PER_FRAME: u64 = 28 << 20;

/// Fixed peak-memory term of a tile decode (bytes): 1 GiB. It covers the process, the
/// decoder-only weights (265 MB FP32, folded at load) and the mapped weight pages. The probe
/// measured 0.80 GB for a one-frame decode, and the multi-tile series extrapolates to about
/// 0.1 GB at zero frames. With [`TILE_BYTES_PER_FRAME`], every measurement above sits 25–70%
/// below its estimate.
pub const TILE_RESERVE_BYTES: u64 = 1 << 30;

/// The default decode memory budget [`DecodeOptions::production`] tiles for, in GiB.
pub const DEFAULT_DECODE_BUDGET_GIB: f64 = 8.0;

const GIB: f64 = (1u64 << 30) as f64;

/// The estimated peak memory (bytes) of decoding one tile of `tile_frames` latent frames:
/// [`TILE_RESERVE_BYTES`] + `tile_frames` × [`TILE_BYTES_PER_FRAME`].
///
/// The decoded waveform itself is not part of this estimate. It is 2 channels × 1920 samples ×
/// 4 bytes ≈ 15 KB per latent frame of the whole song, held about three times over while tiles
/// are concatenated and interleaved, so it grows with the song length rather than the tile.
pub fn estimated_tile_bytes(tile_frames: usize) -> u64 {
    TILE_RESERVE_BYTES.saturating_add(TILE_BYTES_PER_FRAME.saturating_mul(tile_frames as u64))
}

/// How the VAE runs over the latents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Exact-boundary halo/crop tiles of `core_frames` latent frames (upstream's default,
    /// `vae_decode = "halo_crop"`). Decoder activation memory is bounded by the tile
    /// (`core_frames + 2 × halo` frames; see [`estimated_tile_bytes`]), not by the song.
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
    /// The production tiling: [`Self::for_memory_budget_gib`] at [`DEFAULT_DECODE_BUDGET_GIB`]
    /// (8 GiB). That gives 224-frame cores (8.96 s) with the 16-frame halo, a 256-frame tile whose
    /// estimated peak is 8.0 GiB (8.59 GB). Measured on Candle CPU: 6.43–6.76 GB for 3–7
    /// consecutive tiles, and 5.82 GB for the real-weight production test. Upstream's own
    /// 1024/512-frame rule was calibrated for PyTorch on CUDA; natively, a 1024-frame core
    /// (1056-frame tile) is estimated at about 30 GiB.
    pub fn production() -> Self {
        Self::for_memory_budget_gib(DEFAULT_DECODE_BUDGET_GIB)
            .expect("the default decode budget admits a tile (unit-tested)")
    }

    /// The largest tile core (at most upstream's 1024 frames) whose estimated tile footprint,
    /// [`estimated_tile_bytes`]`(core + 2 × halo)`, fits `budget_gib`:
    /// `core = min(⌊(budget − reserve) / per_frame⌋ − 2·halo, 1024)`.
    ///
    /// A non-finite or non-positive budget is refused ("memory_budget_gib must be positive", as
    /// upstream refuses it). So is a budget too small for even a one-frame core: clamping the core
    /// up to 1 would silently exceed the budget.
    pub fn for_memory_budget_gib(budget_gib: f64) -> Result<Self, VaeError> {
        if !budget_gib.is_finite() || budget_gib <= 0.0 {
            return Err(VaeError::MemoryBudget(format!(
                "memory_budget_gib must be positive and finite (got {budget_gib})"
            )));
        }
        let halo = DEFAULT_HALO_FRAMES;
        // `as` saturates, so an enormous budget becomes u64::MAX and is capped below.
        let budget = (budget_gib * GIB) as u64;
        let fit = budget.saturating_sub(TILE_RESERVE_BYTES) / TILE_BYTES_PER_FRAME;
        let core = usize::try_from(fit)
            .unwrap_or(usize::MAX)
            .saturating_sub(2 * halo)
            .min(DEFAULT_CORE_FRAMES);
        if core == 0 {
            return Err(VaeError::MemoryBudget(format!(
                "a {budget_gib} GiB decode budget is below the smallest tile's estimated \
                 footprint of {:.2} GiB (1 core + 2 × {halo} halo frames)",
                estimated_tile_bytes(1 + 2 * halo) as f64 / GIB
            )));
        }
        Ok(Self {
            mode: DecodeMode::Tiled { core_frames: core },
            halo_frames: halo,
        })
    }

    /// The estimated peak decode memory (bytes) for a `frames`-frame latent under these options.
    /// This is the largest tile, or the whole latent for [`DecodeMode::Full`]; see
    /// [`estimated_tile_bytes`] for what it excludes.
    pub fn estimated_peak_bytes(&self, frames: usize) -> u64 {
        let tile = match self.mode {
            DecodeMode::Tiled { core_frames } => frames.min(core_frames + 2 * self.halo_frames),
            DecodeMode::Full => frames,
        };
        estimated_tile_bytes(tile)
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
        assert_eq!(DecodeOptions::default(), DecodeOptions::production());
    }

    /// The budget rule honours the budget it is given. At 8, 12, 16 and 24 GiB the estimated
    /// footprint of the chosen tile (core + 2 × halo) is within the budget, and one more core frame
    /// would not be (unless the core is already capped at 1024). NaN, ±inf, 0, −1 and a budget
    /// below the smallest tile are refused. Mutations: dropping the `− 2·halo` term, or the
    /// reserve, puts the estimate over budget (red); clamping the core up to 1 instead of
    /// refusing makes the 1 GiB case pass (red).
    #[test]
    fn memory_budget_tiling_fits_its_budget() {
        for gib in [2.0, 8.0, 12.0, 16.0, 24.0, DEFAULT_DECODE_BUDGET_GIB] {
            let o = DecodeOptions::for_memory_budget_gib(gib).unwrap();
            let DecodeMode::Tiled { core_frames } = o.mode else {
                panic!("{gib}: not tiled")
            };
            assert_eq!(o.halo_frames, DEFAULT_HALO_FRAMES);
            let budget = (gib * GIB) as u64;
            let est = estimated_tile_bytes(core_frames + 2 * o.halo_frames);
            assert!(
                est <= budget,
                "{gib} GiB: core {core_frames} estimates {est} bytes"
            );
            assert_eq!(o.estimated_peak_bytes(100_000), est);
            if core_frames < DEFAULT_CORE_FRAMES {
                assert!(estimated_tile_bytes(core_frames + 1 + 2 * o.halo_frames) > budget);
            }
            assert!((1..=DEFAULT_CORE_FRAMES).contains(&core_frames));
        }
        assert_eq!(
            DecodeOptions::production().mode,
            DecodeMode::Tiled { core_frames: 224 }
        );
        // Each refusal comes from the intended rule: invalid budgets from the positivity check,
        // a too-small one from the minimum-tile check.
        let refusal = |gib: f64| match DecodeOptions::for_memory_budget_gib(gib) {
            Err(VaeError::MemoryBudget(m)) => m,
            other => panic!("{gib} GiB must be refused, got {other:?}"),
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            assert!(refusal(bad).contains("must be positive"), "{bad}");
        }
        assert!(refusal(1.0).contains("below the smallest tile"));
        // A full decode's estimate grows with the latent; a tiled one stops at the tile.
        let full = DecodeOptions::reference_full();
        assert!(full.estimated_peak_bytes(2000) > full.estimated_peak_bytes(1000));
        let tiled = DecodeOptions::production();
        assert_eq!(
            tiled.estimated_peak_bytes(2000),
            tiled.estimated_peak_bytes(1000)
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
