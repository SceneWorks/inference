//! **Seam: ICL reference encoder.** A reference clip window (single-track mix, or dual-track vocal
//! + instrumental) → the mm-vocabulary codec token block the ICL prompt embeds.
//!
//! Loaded only for ICL renders and released before stage 1 loads. [`load`] is the native candle
//! port (sc-19379) of YuE-v1 `infer.py`'s reference path; [`StubIclEncoder`] stays as the
//! end-to-end seam test's weights-free double.
//!
//! # The reference path, ported
//!
//! ```text
//!   track ─ load_audio_mono: channel mean → torchaudio Resample(sr → 16 kHz)   [downmix, resample_to_16k]
//!         ─ encode_audio(target_bw = 0.5) = SoundStream.encode:                [XcodecEncoder]
//!             semantic: HuBERT(pad(x, 160, 160)) mean of 13 hidden states
//!                       → RepCodec encoder_semantic                              [1, 768, T]
//!             acoustic: DAC encoder(x) (re-run on pad(x, 160, 160) if its T differs) [1, 256, T]
//!             fc_prior(cat[acoustic, semantic]) → RVQ codebook 0 nearest codeword [T] codes
//!         ─ CodecManipulator("xcodec", 0, 1).npy2ids: code + 45334               [codec_token]
//!   window: single  ids[int(start·50) : int(end·50)]                             (50 tok/s)
//!           dual    interleave(vocals, instrumental)[int(start·50·2) : int(end·50·2)] (100 tok/s)
//! ```
//!
//! 0.5 kb/s at 10 bits × 50 Hz per quantizer is exactly one quantizer, so only codebook 0 is ever
//! computed. The whole clip is encoded before the window is cut (HuBERT's attention is global, so a
//! cropped encode would change the tokens); the prompt builder ([`crate::tokenizer`]) adds the
//! `[start_of_reference] <SOA><xcodec> … <EOA> [end_of_reference]` wrapping around
//! [`IclPromptCodes`].
//!
//! Precision: float32 on the audio lane's device, like the codec decoder (the approved epic-R2
//! carve-out — the staged `xcodec-mini-infer` checkpoint is float32 and never tiered).
//!
//! Memory: the sample-rate stages (DAC encoder, HuBERT's conv feature encoder) run
//! [`DEFAULT_CHUNK_FRAMES`] codec frames at a time with exact seams, so the CPU working set stays
//! ~3.5 GB for a 60 s or a 240 s reference (measured; whole-clip evaluation, as the reference does
//! it, grew to 9 GB at 60 s). HuBERT's attention runs over the whole clip, a bounded block of query
//! rows at a time.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::safetensors::MmapedSafetensors;
use candle_audio::candle_core::{DType, Device, Module, Tensor, D};
use candle_audio::gen_core::{self, AudioTrack, CancelFlag};
use candle_audio::neural_codec::{resolve_weight_norm, rvq_encode, DacEncoder};
use candle_audio::AudioError;
use candle_nn::{Conv1d, Conv1dConfig, Linear};

use crate::codec::{CHECKPOINT, DECODER_RATES, SAMPLE_RATE};
use crate::config::{Assets, IclReference, IclTracks};
use crate::hubert::Hubert;
use crate::tokens::{codec_token, CODEBOOK_SIZE, FRAMES_PER_SECOND};

/// The encoded reference block: codebook-0 mm-vocabulary ids (interleaved vocal/instrumental per
/// frame for a dual-track reference).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IclPromptCodes {
    /// The token block.
    pub ids: Vec<u32>,
}

/// The ICL encoder seam.
pub trait IclEncoder: Send {
    /// Encode the reference window. Must honor `cancel` between clips/chunks by returning
    /// [`gen_core::Error::Canceled`].
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> gen_core::Result<IclPromptCodes>;
}

/// Production loader: the xcodec reference encoder from the staged snapshot's [`CHECKPOINT`] on
/// the audio lane's default device.
pub fn load(assets: &Assets) -> gen_core::Result<Box<dyn IclEncoder>> {
    let device = candle_audio::default_device()?;
    let encoder = XcodecEncoder::load(&assets.xcodec_root().join(CHECKPOINT), &device)?;
    Ok(Box::new(encoder))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets) -> gen_core::Result<Box<dyn IclEncoder>> {
    Ok(Box::new(StubIclEncoder))
}

/// The DAC acoustic encoder's downsampling rates (`generator.config.ratios`, the same list the
/// decoder upsamples by) — one 20 ms frame per 320 samples.
pub const ENCODER_RATES: [usize; 4] = DECODER_RATES;
/// Samples of zero padding each side of the waveform before HuBERT (`get_regress_target`).
pub const SEMANTIC_PAD: usize = 160;
/// Quantizers used at the reference's `target_bw = 0.5` kb/s.
pub const ICL_QUANTIZERS: usize = 1;
/// Samples per codec frame at 16 kHz (the product of [`ENCODER_RATES`]).
const HOP: usize = 320;
/// Default codec frames (20 ms each) the sample-rate stages — the DAC acoustic encoder and
/// HuBERT's conv feature encoder — evaluate at once. Whole-clip evaluation (what the reference
/// does) grows ~130 MB of CPU working set per second of reference audio, so a full-length song
/// would need tens of GB; chunking bounds it to one chunk's worth (~10 s) at any clip length.
pub const DEFAULT_CHUNK_FRAMES: usize = 500;
/// Frames of real-audio context each side of an acoustic-encoder chunk, cropped after encoding.
/// The DAC encoder's receptive field is ±~9 100 samples (≈ 28.4 frames: the dilated k7 residual
/// units at each stage's rate plus the strided convs), so 64 frames puts every kept frame's full
/// receptive field inside real audio — the chunk seams never see the chunk's own zero padding.
pub const ACOUSTIC_CONTEXT_FRAMES: usize = 64;

// ------------------------------------------------------------------------------------------------
// load_audio_mono: channel mean + torchaudio Resample
// ------------------------------------------------------------------------------------------------

/// `torch.mean(audio, dim=0)` over an interleaved track's channels.
pub fn downmix(track: &AudioTrack) -> Result<Vec<f32>, AudioError> {
    let c = usize::from(track.channels);
    if c == 0 || !track.samples.len().is_multiple_of(c) {
        return Err(AudioError::Msg(format!(
            "ICL reference: {} samples are not whole {c}-channel frames",
            track.samples.len()
        )));
    }
    if c == 1 {
        return Ok(track.samples.clone());
    }
    Ok(track
        .samples
        .chunks_exact(c)
        .map(|frame| frame.iter().sum::<f32>() / c as f32)
        .collect())
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// torchaudio's resampler as `load_audio_mono` builds it: `Resample(orig_freq = sr, new_freq =
/// 16000)` with the defaults `sinc_interp_hann`, `lowpass_filter_width = 6`, `rolloff = 0.99`.
///
/// Ported from `torchaudio.functional._get_sinc_resample_kernel` / `_apply_sinc_resample_kernel`
/// (torchaudio 2.11) with its dtype path: the phase offsets `arange(0, -new, -1) / new` are float32
/// (integer true-division), promoted to float64 for the kernel, which is cached as float32; the
/// output length is `ceil(float32(new · len / orig))`. The convolution accumulates in float64 (the
/// reference's float32 conv differs only by summation order).
pub fn resample_to_16k(mono: &[f32], sample_rate: u32) -> Result<Vec<f32>, AudioError> {
    if sample_rate == 0 {
        return Err(AudioError::Msg("ICL reference: sample rate 0".into()));
    }
    if sample_rate == SAMPLE_RATE {
        return Ok(mono.to_vec());
    }
    const WIDTH: f64 = 6.0;
    let g = gcd(u64::from(sample_rate), u64::from(SAMPLE_RATE));
    let orig = (u64::from(sample_rate) / g) as usize;
    let new = (u64::from(SAMPLE_RATE) / g) as usize;
    let base = orig.min(new) as f64 * 0.99;
    let width = (WIDTH * orig as f64 / base).ceil() as usize;
    let taps = 2 * width + orig;
    let scale = base / orig as f64;
    let pi = std::f64::consts::PI;

    let mut kernel = vec![0f32; new * taps];
    for j in 0..new {
        let t0 = f64::from(-(j as f32) / new as f32);
        for k in 0..taps {
            let idx = (k as f64 - width as f64) / orig as f64;
            let mut t = ((t0 + idx) * base).clamp(-WIDTH, WIDTH);
            let window = (t * pi / WIDTH / 2.0).cos().powi(2);
            t *= pi;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[j * taps + k] = (sinc * (window * scale)) as f32;
        }
    }

    let len = mono.len();
    let target = ((new as f64 * len as f64 / orig as f64) as f32).ceil() as usize;
    let mut out = Vec::with_capacity(target);
    // `pad(x, (width, width + orig))`, strided by `orig`: output `block·new + j` is phase `j`'s
    // kernel over padded[block·orig ..], where padded[p] = x[p − width] inside the clip, else 0.
    let mut block = 0;
    while out.len() < target {
        for j in 0..new {
            if out.len() == target {
                break;
            }
            let start = block * orig;
            let row = &kernel[j * taps..(j + 1) * taps];
            let mut acc = 0f64;
            for (k, &w) in row.iter().enumerate() {
                let p = start + k;
                if p >= width && p - width < len {
                    acc += f64::from(mono[p - width]) * f64::from(w);
                }
            }
            out.push(acc as f32);
        }
        block += 1;
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------
// The xcodec encoder
// ------------------------------------------------------------------------------------------------

/// One RepCodec residual unit's two convs, then the block's stride-1 conv.
type SemanticBlock = (Vec<(Conv1d, Conv1d)>, Conv1d);

/// RepCodec's `Encoder` as xcodec's `encoder_semantic` builds it: `Conv1d(k3, no bias)` → 2 ×
/// `EncoderBlock(stride 1)`, each `2 × ResidualUnit(ELU → Conv1d(k3, no bias) → ELU → Conv1d(1×1,
/// no bias), + x)` → `Conv1d(k3, bias)`. Every conv is length-preserving (`padding = 1`).
struct SemanticEncoder {
    conv: Conv1d,
    blocks: Vec<SemanticBlock>,
}

impl SemanticEncoder {
    fn load(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self, AudioError> {
        let get = |name: &str| -> Result<Tensor, AudioError> {
            map.get(&format!("{prefix}.{name}"))
                .cloned()
                .ok_or_else(|| AudioError::Msg(format!("xcodec: missing {prefix}.{name}")))
        };
        let k3 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = Conv1d::new(get("conv.conv.weight")?, None, k3);
        let mut blocks = Vec::new();
        while map.contains_key(&format!(
            "{prefix}.conv_blocks.{}.conv.conv.weight",
            blocks.len()
        )) {
            let b = format!("conv_blocks.{}", blocks.len());
            let mut units = Vec::new();
            while map.contains_key(&format!(
                "{prefix}.{b}.res_units.{}.conv1.conv.weight",
                units.len()
            )) {
                let u = format!("{b}.res_units.{}", units.len());
                units.push((
                    Conv1d::new(get(&format!("{u}.conv1.conv.weight"))?, None, k3),
                    Conv1d::new(
                        get(&format!("{u}.conv2.weight"))?,
                        None,
                        Conv1dConfig::default(),
                    ),
                ));
            }
            let down = Conv1d::new(
                get(&format!("{b}.conv.conv.weight"))?,
                Some(get(&format!("{b}.conv.conv.bias"))?),
                k3,
            );
            if down.weight().dim(2)? != 3 {
                return Err(AudioError::Msg(format!(
                    "xcodec: {prefix}.{b} conv must be the stride-1 kernel-3 conv"
                )));
            }
            blocks.push((units, down));
        }
        if blocks.is_empty() {
            return Err(AudioError::Msg(format!(
                "xcodec: {prefix} has no conv blocks"
            )));
        }
        Ok(Self { conv, blocks })
    }

    fn output_dim(&self) -> Result<usize, AudioError> {
        Ok(self
            .blocks
            .last()
            .map_or(self.conv.weight().dim(0), |(_, c)| c.weight().dim(0))?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, AudioError> {
        let mut x = self.conv.forward(x)?;
        for (units, down) in &self.blocks {
            for (c1, c2) in units {
                let y = c2.forward(&c1.forward(&x.elu(1.0)?)?.elu(1.0)?)?;
                x = (x + y)?;
            }
            x = down.forward(&x)?;
        }
        Ok(x)
    }
}

/// The intermediate tensors of one [`XcodecEncoder::encode_stages`] run (batch 1).
#[derive(Clone, Debug)]
pub struct EncodeStages {
    /// `get_regress_target`: the mean of HuBERT's hidden states, `[1, T, hidden]`.
    pub hubert_mean: Tensor,
    /// `e_semantic`: the RepCodec semantic encoding, `[1, hidden, T]`.
    pub semantic: Tensor,
    /// `e_acoustic`: the DAC encoding (padded re-encode when upstream takes it), `[1, latent, T]`.
    pub acoustic: Tensor,
    /// `fc_prior(cat[e_acoustic, e_semantic])`, `[1, dim, T]` — the quantizer's input.
    pub prior: Tensor,
}

/// The native xcodec reference encoder: `SoundStream.encode` at 0.5 kb/s.
pub struct XcodecEncoder {
    hubert: Hubert,
    semantic: SemanticEncoder,
    acoustic: DacEncoder,
    fc_prior: Linear,
    codebook0: Tensor,
    device: Device,
    chunk_frames: usize,
}

impl XcodecEncoder {
    /// Load the encode path from a codec checkpoint (safetensors, the `codec_model` state dict):
    /// `encoder.*` (weight-norm folded), `encoder_semantic.*`, `fc_prior.*`, `semantic_model.*`
    /// (HuBERT — the checkpoint's copy, which upstream's `load_state_dict` puts over the
    /// `semantic_ckpts` one) and quantizer 0's codebook. The decoder half is never paged in.
    pub fn load(checkpoint: &Path, device: &Device) -> Result<Self, AudioError> {
        // SAFETY: the checkpoint is a caller-staged, read-only weight file; nothing in this process
        // writes it while the map is alive (the contract every candle mmap loader relies on).
        let st = unsafe { MmapedSafetensors::new(checkpoint) }
            .map_err(|e| AudioError::Msg(format!("xcodec: open {}: {e}", checkpoint.display())))?;
        let load = |name: &str| -> Result<Tensor, AudioError> {
            Ok(st
                .load(name, device)
                .map_err(|e| AudioError::Msg(format!("xcodec: tensor {name:?}: {e}")))?
                .to_dtype(DType::F32)?)
        };
        let (mut acoustic, mut semantic, mut hubert) =
            (HashMap::new(), HashMap::new(), HashMap::new());
        for (name, _) in st.tensors() {
            let bucket = if name.starts_with("encoder.") {
                &mut acoustic
            } else if name.starts_with("encoder_semantic.") {
                &mut semantic
            } else if name.starts_with("semantic_model.") {
                &mut hubert
            } else {
                continue;
            };
            let t = load(&name)?;
            bucket.insert(name, t);
        }
        // Only the DAC encoder's pairs are dim-0 weight norms; HuBERT's positional conv is
        // `weight_norm(dim = 2)` and is folded by `Hubert::load`.
        let acoustic =
            DacEncoder::load(&resolve_weight_norm(acoustic)?, "encoder", &ENCODER_RATES)?;
        let semantic = SemanticEncoder::load(&semantic, "encoder_semantic")?;
        let hubert = Hubert::load(&hubert, "semantic_model.")?;

        let codebook0 = load("quantizer.vq.layers.0._codebook.embed")?;
        let (size, dim) = codebook0.dims2()?;
        if size != CODEBOOK_SIZE as usize {
            return Err(AudioError::Msg(format!(
                "xcodec: codebook 0 has {size} entries, expected {CODEBOOK_SIZE}"
            )));
        }
        let fc_w = load("fc_prior.weight")?;
        let (fc_out, fc_in) = fc_w.dims2()?;
        let (ac_out, sem_out) = (acoustic.output_dim()?, semantic.output_dim()?);
        if fc_out != dim || fc_in != ac_out + sem_out {
            return Err(AudioError::Msg(format!(
                "xcodec: fc_prior is [{fc_out}, {fc_in}] but the encoders emit {ac_out} + \
                 {sem_out} channels and codebook 0 is {dim} wide"
            )));
        }
        Ok(Self {
            hubert,
            semantic,
            acoustic,
            fc_prior: Linear::new(fc_w, Some(load("fc_prior.bias")?)),
            codebook0,
            device: device.clone(),
            chunk_frames: DEFAULT_CHUNK_FRAMES,
        })
    }

    /// Evaluate the sample-rate stages `frames` codec frames at a time (default
    /// [`DEFAULT_CHUNK_FRAMES`]; clamped to ≥ 1). Only the working set changes: chunk seams are
    /// exact by construction (see [`ACOUSTIC_CONTEXT_FRAMES`] and [`Hubert::features`]), up to
    /// float summation order.
    pub fn with_chunk_frames(mut self, frames: usize) -> Self {
        self.chunk_frames = frames.max(1);
        self
    }

    /// The DAC acoustic encoder over `[1, 1, n]`, [`Self::with_chunk_frames`] frames at a time:
    /// each chunk is encoded with [`ACOUSTIC_CONTEXT_FRAMES`] of real audio each side (clip edges
    /// keep the encoder's own zero padding) and cropped to its frames. Chunk starts are whole
    /// frames, so every layer's output grid lines up with the whole-clip encode.
    fn acoustic_encode(
        &self,
        x: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Tensor>, AudioError> {
        let n = x.dim(D::Minus1)?;
        let total = acoustic_frames(n);
        if total <= self.chunk_frames {
            return Ok(self.acoustic.encode(x, cancel)?);
        }
        let mut parts = Vec::with_capacity(total.div_ceil(self.chunk_frames));
        for f0 in (0..total).step_by(self.chunk_frames) {
            let f1 = (f0 + self.chunk_frames).min(total);
            let first = f0.saturating_sub(ACOUSTIC_CONTEXT_FRAMES);
            let start = first * HOP;
            let end = ((f1 + ACOUSTIC_CONTEXT_FRAMES) * HOP).min(n);
            let Some(y) = self
                .acoustic
                .encode(&x.narrow(D::Minus1, start, end - start)?, cancel)?
            else {
                return Ok(None);
            };
            parts.push(y.narrow(D::Minus1, f0 - first, f1 - f0)?);
        }
        Ok(Some(Tensor::cat(&parts, D::Minus1)?))
    }

    /// Attend HuBERT's queries `rows` at a time ([`Hubert::with_query_chunk`]).
    pub fn with_query_chunk(mut self, rows: usize) -> Self {
        self.hubert = self.hubert.with_query_chunk(rows);
        self
    }

    /// `fc_prior(cat[acoustic, semantic])` — the pre-quantization embedding `[1, dim, T]` for one
    /// mono 16 kHz waveform. `None` when `cancel` trips between stages.
    pub fn prior_embedding(
        &self,
        wave: &[f32],
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Tensor>, AudioError> {
        Ok(self.encode_stages(wave, cancel)?.map(|s| s.prior))
    }

    /// Every stage of `SoundStream.encode` before the quantizer, for one mono 16 kHz waveform —
    /// what the parity tests hold to the reference stage by stage. `None` when `cancel` trips.
    pub fn encode_stages(
        &self,
        wave: &[f32],
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<EncodeStages>, AudioError> {
        let n = wave.len();
        if Hubert::frames_for(n + 2 * SEMANTIC_PAD) == 0 {
            return Err(AudioError::Msg(format!(
                "ICL reference: {n} samples at 16 kHz is too short for the codec"
            )));
        }
        let x = Tensor::from_slice(wave, (1, 1, n), &self.device)?;
        let padded = x.pad_with_zeros(D::Minus1, SEMANTIC_PAD, SEMANTIC_PAD)?;

        // Semantic branch: HuBERT layer mean → RepCodec encoder.
        let Some(sem_in) =
            self.hubert
                .mean_hidden_states(&padded.squeeze(1)?, self.chunk_frames, cancel)?
        else {
            return Ok(None);
        };
        let semantic = self
            .semantic
            .forward(&sem_in.transpose(1, 2)?.contiguous()?)?;
        let frames = semantic.dim(D::Minus1)?;

        // Acoustic branch, re-run on the padded wave when its frame count disagrees (upstream).
        let Some(mut acoustic) = self.acoustic_encode(&x, cancel)? else {
            return Ok(None);
        };
        if acoustic.dim(D::Minus1)? != frames {
            let Some(again) = self.acoustic_encode(&padded, cancel)? else {
                return Ok(None);
            };
            acoustic = again;
        }
        if acoustic.dim(D::Minus1)? != frames {
            return Err(AudioError::Msg(format!(
                "ICL reference: acoustic ({}) and semantic ({frames}) frame counts disagree for \
                 {n} samples (the reference codec cannot concatenate them either)",
                acoustic.dim(D::Minus1)?
            )));
        }
        if cancel() {
            return Ok(None);
        }
        let e = Tensor::cat(&[&acoustic, &semantic], 1)?;
        let prior = self
            .fc_prior
            .forward(&e.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(Some(EncodeStages {
            hubert_mean: sem_in,
            semantic,
            acoustic,
            prior,
        }))
    }

    /// `SoundStream.encode(x, target_bw = 0.5)` for one mono 16 kHz waveform: the codebook-0
    /// codes, one per 20 ms frame. `None` when `cancel` trips between stages.
    pub fn encode_codes(
        &self,
        wave: &[f32],
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<u32>>, AudioError> {
        let Some(e) = self.prior_embedding(wave, cancel)? else {
            return Ok(None);
        };
        let mut codes = rvq_encode(std::slice::from_ref(&self.codebook0), &e, ICL_QUANTIZERS)?;
        Ok(Some(codes.remove(0)))
    }

    /// `load_audio_mono` + `encode_audio` + `npy2ids` for one reference track: its codebook-0 mm
    /// ids over the whole clip.
    pub fn track_ids(
        &self,
        track: &AudioTrack,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<u32>>, AudioError> {
        let wave = resample_to_16k(&downmix(track)?, track.sample_rate)?;
        Ok(self
            .encode_codes(&wave, cancel)?
            .map(|codes| codes.into_iter().map(|c| codec_token(0, c)).collect()))
    }
}

/// Frames the DAC acoustic encoder yields for `n` samples: per strided conv (kernel `2s`, padding
/// `⌈s/2⌉`) `⌊(L + 2⌈s/2⌉ − 2s) / s⌋ + 1`; the k7 input and k3 output convs keep the length.
pub fn acoustic_frames(n: usize) -> usize {
    ENCODER_RATES.iter().fold(n, |l, &s| {
        let padded = l + 2 * s.div_ceil(2);
        if padded < 2 * s {
            0
        } else {
            (padded - 2 * s) / s + 1
        }
    })
}

/// Python `int(secs · per_sec)` for a non-negative window bound, clamped to a sequence of `len`.
///
/// The reference parses its window from the command line as a float64, so the bound is taken at
/// the float64 nearest the decimal the `f32` request field denotes (its shortest round-trip
/// decimal), not at the `f32`'s exact binary value: `0.01 s · 50 · 2` is `int(1.0) = 1` upstream,
/// whereas the `f32` nearest 0.01 is slightly below it and would truncate to 0. The product is
/// evaluated left to right, as the reference writes it.
fn window_index(secs: f32, per_sec: &[f64], len: usize) -> usize {
    let secs: f64 = secs.to_string().parse().unwrap_or(f64::from(secs));
    let v = per_sec.iter().fold(secs, |acc, &m| acc * m);
    (v.trunc().max(0.0) as usize).min(len)
}

/// The prompt window over a single track's ids: `ids[int(start·50) : int(end·50)]`.
pub fn window_single(ids: &[u32], start_secs: f32, end_secs: f32) -> Vec<u32> {
    let fps = [f64::from(FRAMES_PER_SECOND)];
    let a = window_index(start_secs, &fps, ids.len());
    let b = window_index(end_secs, &fps, ids.len());
    ids[a..b.max(a)].to_vec()
}

/// The dual-track window: vocal/instrumental ids interleaved per frame (`rearrange(... 'b n -> (n
/// b)')`), then `[int(start·50·2) : int(end·50·2)]`. The tracks must encode to the same number of
/// frames (the reference's `np.array` stack requires it).
pub fn window_dual(
    vocals: &[u32],
    instrumental: &[u32],
    start_secs: f32,
    end_secs: f32,
) -> Result<Vec<u32>, AudioError> {
    if vocals.len() != instrumental.len() {
        return Err(AudioError::Msg(format!(
            "ICL reference: the vocal ({}) and instrumental ({}) stems encode to different frame \
             counts; dual-track references must be the same length",
            vocals.len(),
            instrumental.len()
        )));
    }
    let interleaved: Vec<u32> = vocals
        .iter()
        .zip(instrumental)
        .flat_map(|(&v, &i)| [v, i])
        .collect();
    let per = [f64::from(FRAMES_PER_SECOND), 2.0];
    let a = window_index(start_secs, &per, interleaved.len());
    let b = window_index(end_secs, &per, interleaved.len());
    Ok(interleaved[a..b.max(a)].to_vec())
}

impl IclEncoder for XcodecEncoder {
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> gen_core::Result<IclPromptCodes> {
        let poll = || cancel.is_cancelled();
        let track = |t: &AudioTrack| -> gen_core::Result<Vec<u32>> {
            if poll() {
                return Err(gen_core::Error::Canceled);
            }
            self.track_ids(t, &poll)?.ok_or(gen_core::Error::Canceled)
        };
        let ids = match &reference.tracks {
            IclTracks::Single(t) => {
                window_single(&track(t)?, reference.start_secs, reference.end_secs)
            }
            IclTracks::Dual {
                vocals,
                instrumental,
            } => {
                let v = track(vocals)?;
                let i = track(instrumental)?;
                window_dual(&v, &i, reference.start_secs, reference.end_secs)?
            }
        };
        if ids.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "ICL reference: the window {}..{} s holds no codec frames of the clip",
                reference.start_secs, reference.end_secs
            )));
        }
        Ok(IclPromptCodes { ids })
    }
}

/// **Stub ICL encoder** — the end-to-end seam test's weights-free double. Emits one
/// deterministic codebook-0 id per 20 ms frame of the window (two, interleaved, for dual-track),
/// keyed by the clip's sample count.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubIclEncoder;

impl IclEncoder for StubIclEncoder {
    fn encode(
        &self,
        reference: &IclReference,
        cancel: &CancelFlag,
    ) -> gen_core::Result<IclPromptCodes> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let frames = ((reference.end_secs - reference.start_secs) * FRAMES_PER_SECOND as f32)
            .ceil()
            .max(1.0) as usize;
        let keys: Vec<u64> = match &reference.tracks {
            IclTracks::Single(t) => vec![t.samples.len() as u64],
            IclTracks::Dual {
                vocals,
                instrumental,
            } => vec![
                vocals.samples.len() as u64,
                instrumental.samples.len() as u64,
            ],
        };
        let mut ids = Vec::with_capacity(frames * keys.len());
        for f in 0..frames {
            for &k in &keys {
                let code = (crate::stub::hash(&[k, f as u64]) % CODEBOOK_SIZE as u64) as u32;
                ids.push(codec_token(0, code));
            }
        }
        Ok(IclPromptCodes { ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::safetensors::save;

    /// Tiny synthetic widths — every loader reads its widths from the tensors, so a small
    /// checkpoint drives the real load + encode path end to end.
    const LAT: usize = 3; // DAC latent
    const H: usize = 48; // HuBERT hidden (÷ 12 heads, ÷ 16 pos-conv groups)
    const C: usize = 4; // HuBERT conv width
    const FF: usize = 8; // HuBERT feed-forward width
    const PK: usize = 4; // positional-conv kernel (even: one frame trimmed)
    const E: usize = 5; // codebook width

    fn det(shape: &[usize], seed: u32, scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503);
                ((h % 1000) as f32 / 1000.0 - 0.5) * scale
            })
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    struct Ckpt {
        t: HashMap<String, Tensor>,
    }

    impl Ckpt {
        fn put(&mut self, name: &str, shape: &[usize], scale: f32) {
            let seed = self.t.len() as u32 + 1;
            self.t.insert(name.to_string(), det(shape, seed, scale));
        }
        fn ones(&mut self, name: &str, shape: &[usize]) {
            let v = (det(shape, self.t.len() as u32 + 1, 0.2) + 1.0).unwrap();
            self.t.insert(name.to_string(), v);
        }
        /// A weight-norm conv (`weight_v`, `weight_g`, `bias`).
        fn wn(&mut self, name: &str, shape: [usize; 3]) {
            self.put(&format!("{name}.weight_v"), &shape, 1.0);
            self.put(&format!("{name}.weight_g"), &[shape[0], 1, 1], 1.0);
            self.put(&format!("{name}.bias"), &[shape[0]], 0.1);
        }
        fn snake(&mut self, name: &str, w: usize) {
            self.ones(&format!("{name}.alpha"), &[1, w, 1]);
        }
        fn linear(&mut self, name: &str, out: usize, inp: usize) {
            let s = 1.0 / (inp as f32).sqrt();
            self.put(&format!("{name}.weight"), &[out, inp], 2.0 * s);
            self.put(&format!("{name}.bias"), &[out], 0.1);
        }
        fn ln(&mut self, name: &str, n: usize) {
            self.ones(&format!("{name}.weight"), &[n]);
            self.put(&format!("{name}.bias"), &[n], 0.1);
        }
    }

    /// Write a tiny xcodec-shaped checkpoint (encode half only) at [`CHECKPOINT`].
    fn tiny_snapshot() -> tempfile::TempDir {
        let mut c = Ckpt { t: HashMap::new() };
        // DAC acoustic encoder.
        c.wn("encoder.block.0", [2, 1, 7]);
        let mut w = 2;
        for (i, &r) in ENCODER_RATES.iter().enumerate() {
            let base = format!("encoder.block.{}.block", i + 1);
            for u in 0..3 {
                c.snake(&format!("{base}.{u}.block.0"), w);
                c.wn(&format!("{base}.{u}.block.1"), [w, w, 7]);
                c.snake(&format!("{base}.{u}.block.2"), w);
                c.wn(&format!("{base}.{u}.block.3"), [w, w, 1]);
            }
            c.snake(&format!("{base}.3"), w);
            c.wn(&format!("{base}.4"), [2 * w, w, 2 * r]);
            w *= 2;
        }
        c.snake("encoder.block.5", w);
        c.wn("encoder.block.6", [LAT, w, 3]);
        // HuBERT.
        let s = "semantic_model.";
        for (i, &k) in crate::hubert::CONV_KERNELS.iter().enumerate() {
            let inp = if i == 0 { 1 } else { C };
            c.put(
                &format!("{s}feature_extractor.conv_layers.{i}.conv.weight"),
                &[C, inp, k],
                1.0,
            );
        }
        c.ln(&format!("{s}feature_extractor.conv_layers.0.layer_norm"), C);
        c.ln(&format!("{s}feature_projection.layer_norm"), C);
        c.linear(&format!("{s}feature_projection.projection"), H, C);
        c.put(
            &format!("{s}encoder.pos_conv_embed.conv.weight_v"),
            &[H, H / crate::hubert::POS_CONV_GROUPS, PK],
            1.0,
        );
        c.put(
            &format!("{s}encoder.pos_conv_embed.conv.weight_g"),
            &[1, 1, PK],
            1.0,
        );
        c.put(&format!("{s}encoder.pos_conv_embed.conv.bias"), &[H], 0.1);
        c.ln(&format!("{s}encoder.layer_norm"), H);
        for l in 0..2 {
            let p = format!("{s}encoder.layers.{l}");
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                c.linear(&format!("{p}.attention.{proj}"), H, H);
            }
            c.ln(&format!("{p}.layer_norm"), H);
            c.linear(&format!("{p}.feed_forward.intermediate_dense"), FF, H);
            c.linear(&format!("{p}.feed_forward.output_dense"), H, FF);
            c.ln(&format!("{p}.final_layer_norm"), H);
        }
        // RepCodec semantic encoder.
        let k = 1.0 / (3.0 * H as f32).sqrt();
        c.put("encoder_semantic.conv.conv.weight", &[H, H, 3], k);
        for b in 0..2 {
            for u in 0..2 {
                let p = format!("encoder_semantic.conv_blocks.{b}.res_units.{u}");
                c.put(&format!("{p}.conv1.conv.weight"), &[H, H, 3], k);
                c.put(&format!("{p}.conv2.weight"), &[H, H, 1], k);
            }
            let p = format!("encoder_semantic.conv_blocks.{b}.conv.conv");
            c.put(&format!("{p}.weight"), &[H, H, 3], k);
            c.put(&format!("{p}.bias"), &[H], 0.1);
        }
        c.linear("fc_prior", E, LAT + H);
        c.put("quantizer.vq.layers.0._codebook.embed", &[1024, E], 0.1);
        // A decode-half tensor the encoder must never need.
        c.put("decoder_2.model.0.bias", &[2], 1.0);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CHECKPOINT);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        save(&c.t, &path).unwrap();
        dir
    }

    fn assets(xcodec: &Path) -> Assets {
        Assets {
            stage1: "/unused/s1".into(),
            stage2: "/unused/s2".into(),
            xcodec: xcodec.to_path_buf(),
        }
    }

    fn clip(frames: usize, rate: u32, channels: u16, seed: u32) -> AudioTrack {
        let n = frames * usize::from(channels);
        AudioTrack {
            samples: det(&[n], seed, 1.2).to_vec1().unwrap(),
            sample_rate: rate,
            channels,
            stems: Vec::new(),
        }
    }

    fn codebook0_range() -> std::ops::Range<u32> {
        codec_token(0, 0)..codec_token(0, CODEBOOK_SIZE)
    }

    #[test]
    fn production_load_encodes_a_staged_checkpoint_instead_of_refusing() {
        let snap = tiny_snapshot();
        let enc = load(&assets(snap.path())).expect("production encoder loads staged weights");
        // 1 s + 37 samples at 16 kHz: not a whole number of frames (the re-encode branch).
        let n = 16_037;
        let reference = IclReference {
            tracks: IclTracks::Single(clip(n, SAMPLE_RATE, 1, 3)),
            start_secs: 0.0,
            end_secs: 30.0,
        };
        let codes = enc.encode(&reference, &CancelFlag::new()).unwrap();
        assert_eq!(codes.ids.len(), Hubert::frames_for(n + 2 * SEMANTIC_PAD));
        assert_eq!(codes.ids.len(), 50);
        assert!(codes.ids.iter().all(|id| codebook0_range().contains(id)));
        assert!(
            codes.ids.iter().any(|&id| id != codes.ids[0]),
            "a real encode varies over the clip"
        );
    }

    #[test]
    fn frame_counts_agree_across_the_padded_re_encode_branch() {
        let snap = tiny_snapshot();
        let enc = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu).unwrap();
        // Whole-frame lengths take the direct branch; the rest re-encode the padded wave.
        for n in [3_200usize, 3_201, 3_359, 3_520, 4_000, 400] {
            let wave: Vec<f32> = det(&[n], n as u32, 1.0).to_vec1().unwrap();
            let e = enc.prior_embedding(&wave, &|| false).unwrap().unwrap();
            assert_eq!(
                e.dims(),
                [1, E, Hubert::frames_for(n + 2 * SEMANTIC_PAD)],
                "{n}"
            );
        }
        assert!(
            enc.prior_embedding(&[0.1; 50], &|| false).is_err(),
            "too short"
        );
        for n in [3_200usize, 3_201, 3_359, 16_037, 64_123] {
            let x = Tensor::zeros((1, 1, n), DType::F32, &Device::Cpu).unwrap();
            let y = enc.acoustic.encode(&x, &|| false).unwrap().unwrap();
            assert_eq!(y.dim(2).unwrap(), acoustic_frames(n), "{n}");
        }
    }

    #[test]
    fn chunked_sample_rate_stages_match_the_whole_clip_encode() {
        let snap = tiny_snapshot();
        let whole = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu)
            .unwrap()
            .with_chunk_frames(usize::MAX);
        // 200 frames + a remainder (the padded re-encode branch), in 37-frame chunks: seams deep
        // inside the clip, each with its full acoustic context, plus a short last chunk.
        let n = 64_123;
        let wave: Vec<f32> = det(&[n], 77, 1.0).to_vec1().unwrap();
        let want = whole.prior_embedding(&wave, &|| false).unwrap().unwrap();
        let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let peak = want_v.iter().fold(0f32, |m, v| m.max(v.abs()));
        for chunk in [37usize, 5, 200] {
            let enc = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu)
                .unwrap()
                .with_chunk_frames(chunk);
            let got = enc.prior_embedding(&wave, &|| false).unwrap().unwrap();
            assert_eq!(got.dims(), want.dims());
            let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            let diff = got_v
                .iter()
                .zip(&want_v)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            // Only float summation order differs (the GroupNorm statistics and conv GEMM tiling).
            assert!(
                diff <= 1e-5 * peak,
                "chunk {chunk}: max|Δ| {diff} vs peak {peak}"
            );
            let codes = |e: &XcodecEncoder| e.encode_codes(&wave, &|| false).unwrap().unwrap();
            assert_eq!(codes(&enc), codes(&whole), "chunk {chunk}: codes");
        }

        // HuBERT attention in 7-row query blocks (the 202 frames split 29 ways, last block short)
        // reproduces the unsplit layer mean.
        let padded = Tensor::from_slice(&wave, (1, n), &Device::Cpu)
            .unwrap()
            .pad_with_zeros(1, SEMANTIC_PAD, SEMANTIC_PAD)
            .unwrap();
        let mean = |e: &XcodecEncoder| -> Vec<f32> {
            e.hubert
                .mean_hidden_states(&padded, usize::MAX, &|| false)
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let unsplit = mean(&whole.with_query_chunk(usize::MAX));
        let blocked = mean(
            &XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu)
                .unwrap()
                .with_query_chunk(7),
        );
        let peak = unsplit.iter().fold(0f32, |m, v| m.max(v.abs()));
        let diff = blocked
            .iter()
            .zip(&unsplit)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            diff <= 1e-5 * peak,
            "query blocks: max|Δ| {diff} vs peak {peak}"
        );
    }

    #[test]
    fn dual_track_interleaves_vocals_first_and_windows_at_100_per_second() {
        let snap = tiny_snapshot();
        let enc = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu).unwrap();
        // 48 kHz stereo stems (downmix + 3:1 resample on the way in).
        let (v, i) = (clip(24_000, 48_000, 2, 5), clip(24_000, 48_000, 2, 9));
        let vi = enc.track_ids(&v, &|| false).unwrap().unwrap();
        let ii = enc.track_ids(&i, &|| false).unwrap().unwrap();
        assert_eq!(vi.len(), 25);
        assert_ne!(vi, ii);
        let reference = IclReference {
            tracks: IclTracks::Dual {
                vocals: v,
                instrumental: i,
            },
            start_secs: 0.1,
            end_secs: 0.4,
        };
        let codes = enc.encode(&reference, &CancelFlag::new()).unwrap();
        // int(0.1·50·2) = 10 .. int(0.4·50·2) = 40 of the interleave: frames 5..20.
        let want: Vec<u32> = (5..20).flat_map(|f| [vi[f], ii[f]]).collect();
        assert_eq!(codes.ids, want);
    }

    #[test]
    fn single_track_windows_at_50_per_second_after_a_stereo_downmix() {
        let snap = tiny_snapshot();
        let enc = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu).unwrap();
        let stereo = clip(22_050, 44_100, 2, 11);
        let mono = AudioTrack {
            samples: downmix(&stereo).unwrap(),
            channels: 1,
            ..stereo.clone()
        };
        let all = enc.track_ids(&mono, &|| false).unwrap().unwrap();
        let reference = IclReference {
            tracks: IclTracks::Single(stereo),
            start_secs: 0.1,
            end_secs: 0.3,
        };
        let codes = enc.encode(&reference, &CancelFlag::new()).unwrap();
        assert_eq!(codes.ids, all[5..15]);
    }

    #[test]
    fn encode_refuses_mismatched_stems_empty_windows_and_honors_cancel() {
        let snap = tiny_snapshot();
        let enc = XcodecEncoder::load(&snap.path().join(CHECKPOINT), &Device::Cpu).unwrap();
        let dual = |a: usize, b: usize| IclReference {
            tracks: IclTracks::Dual {
                vocals: clip(a, SAMPLE_RATE, 1, 1),
                instrumental: clip(b, SAMPLE_RATE, 1, 2),
            },
            start_secs: 0.0,
            end_secs: 1.0,
        };
        assert!(matches!(
            enc.encode(&dual(6_400, 9_600), &CancelFlag::new()),
            Err(gen_core::Error::Msg(m)) if m.contains("different frame counts")
        ));
        let beyond = IclReference {
            tracks: IclTracks::Single(clip(6_400, SAMPLE_RATE, 1, 4)),
            start_secs: 5.0,
            end_secs: 6.0,
        };
        assert!(matches!(
            enc.encode(&beyond, &CancelFlag::new()),
            Err(gen_core::Error::Msg(m)) if m.contains("no codec frames")
        ));
        let cancel = CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            enc.encode(&dual(6_400, 6_400), &cancel),
            Err(gen_core::Error::Canceled)
        ));
    }

    #[test]
    fn windows_follow_python_int_truncation_and_slice_clamping() {
        let ids: Vec<u32> = (0..10).collect();
        assert_eq!(window_single(&ids, 0.03, 0.1), vec![1, 2, 3, 4]); // int(1.5) .. int(5.0)
        assert_eq!(window_single(&ids, 0.0, 30.0), ids); // end clamps to the clip
        assert!(window_single(&ids, 0.5, 30.0).is_empty()); // start past the clip
        let v = [10, 11, 12];
        let i = [20, 21, 22];
        assert_eq!(
            window_dual(&v, &i, 0.0, 30.0).unwrap(),
            [10, 20, 11, 21, 12, 22]
        );
        // int(0.01·50·2) = 1: an odd start lands on an instrumental id, as upstream's slice does.
        assert_eq!(window_dual(&v, &i, 0.01, 0.03).unwrap(), [20, 11]);
        assert!(window_dual(&v, &i[..2], 0.0, 1.0).is_err());
    }

    #[test]
    fn downmix_is_the_channel_mean_and_refuses_ragged_frames() {
        let t = AudioTrack {
            samples: vec![1.0, 0.0, 0.5, -0.5],
            sample_rate: 16_000,
            channels: 2,
            stems: Vec::new(),
        };
        assert_eq!(downmix(&t).unwrap(), [0.5, 0.0]);
        let ragged = AudioTrack {
            samples: vec![1.0, 0.0, 0.5],
            ..t
        };
        assert!(downmix(&ragged).is_err());
    }

    #[test]
    fn resample_is_identity_at_16k_and_sizes_like_torchaudio() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(resample_to_16k(&x, SAMPLE_RATE).unwrap(), x);
        // ceil(160 · 1000 / 441) = 363; ceil(1000 / 3) = 334; 2 · 1000 = 2000.
        assert_eq!(resample_to_16k(&x, 44_100).unwrap().len(), 363);
        assert_eq!(resample_to_16k(&x, 48_000).unwrap().len(), 334);
        assert_eq!(resample_to_16k(&x, 8_000).unwrap().len(), 2000);
        assert!(resample_to_16k(&x, 0).is_err());
        // A 200 Hz tone survives 48 k → 16 k (well inside the passband).
        let tone: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 48_000.0).sin())
            .collect();
        let y = resample_to_16k(&tone, 48_000).unwrap();
        for (j, &v) in y.iter().enumerate().skip(100).take(1000) {
            let want = (2.0 * std::f32::consts::PI * 200.0 * j as f32 / 16_000.0).sin();
            assert!((v - want).abs() < 1e-2, "sample {j}: {v} vs {want}");
        }
    }

    #[test]
    fn a_missing_checkpoint_is_an_error_not_a_refusal() {
        let dir = tempfile::tempdir().unwrap();
        match load(&assets(dir.path())) {
            Err(gen_core::Error::Unsupported(m)) => panic!("still refusing: {m}"),
            Err(_) => {}
            Ok(_) => panic!("loaded without a checkpoint"),
        }
    }
}
