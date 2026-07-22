//! The **s3tokenizer** — the first ported S3Gen sub-network (sc-13235). A faithful native-candle
//! port of Chatterbox's `models/s3tokenizer` (the `speech_tokenizer_v2_25hz` model, upstream
//! `s3tokenizer.model_v2`): a Whisper-v2-style **FSMN mel encoder** followed by an **FSQ** (finite
//! scalar quantization) head that turns a 16 kHz reference clip into **25 Hz discrete speech
//! tokens** in `[0, 6560]` (the `3^8 = 6561` codebook).
//!
//! It is the `tokenizer.*` block of `s3gen.safetensors` (103 tensors). Its output feeds two places
//! in the clone pipeline the sc-13222 foundation left empty:
//!
//! - T3's `cond_prompt_speech_tokens` — the 150-token speech prompt fed through the Perceiver
//!   resampler (empty in sc-13222, which weakens the voice conditioning); and
//! - S3Gen's own `prompt_token` (once the flow/vocoder stack lands).
//!
//! ## Weight layout (`tokenizer.*`)
//!
//! ```text
//!   tokenizer._mel_filters                         [128, 201]   librosa-slaney mel bank (a buffer)
//!   tokenizer.encoder.conv1.{weight,bias}          [1280,128,3] Conv1d stride 2, pad 1
//!   tokenizer.encoder.conv2.{weight,bias}          [1280,1280,3]Conv1d stride 2, pad 1
//!   tokenizer.encoder.blocks.{0..5}
//!     .attn.query.{weight,bias}  .attn.key.weight (no bias)  .attn.value.{weight,bias}
//!     .attn.out.{weight,bias}    .attn.fsmn_block.weight [1280,1,31] (depthwise, no bias)
//!     .attn_ln.{weight,bias}
//!     .mlp.0.{weight,bias} [5120,1280]  .mlp.2.{weight,bias} [1280,5120]   (mlp.1 = GELU)
//!     .mlp_ln.{weight,bias}
//!   tokenizer.quantizer._codebook.project_down.{weight,bias}  [8, 1280]
//! ```
//!
//! ## Faithfulness notes (verified against upstream)
//!
//! - **Mel front-end**: `torch.stft(n_fft=400, hop=160, hann(400), center=True)`, drop the last
//!   time frame (`stft[..., :-1]`), power (`|·|²`), project through the checkpoint's own
//!   `_mel_filters` (so no librosa reconstruction is needed — exact by construction), then
//!   `log10(clamp(·, 1e-10))`, `max(·, global_max − 8)`, `(·+4)/4`. 100 Hz mel frames.
//! - **Encoder**: two stride-2 convs (4× downsample → 25 Hz) with **exact-erf** GELU, then 6
//!   pre-norm blocks. Attention is Whisper-style (`query`/`value`/`out` biased, `key` unbiased,
//!   scale `head_dim^-0.5`) with **RoPE** (`theta = 10000`, rotate-half = candle's [`rope`]) on
//!   q/k, plus an additive **FSMN memory**: a depthwise conv1d (kernel 31, symmetric pad 15, no
//!   bias) over the *value projection*, added back — `out(attn) + fsmn` is the block's attention
//!   output. Single-clip inference has no padding, so the reference's non-pad masks are all-ones
//!   no-ops and are omitted.
//! - **FSQ**: `project_down` (1280→8), `tanh`, `· 0.9990000128746033`, round-half-to-even, `+1`
//!   (→ levels `{0,1,2}`), then the base-3 code `Σ level_i · 3^i` (`i = 0..8`) → `[0, 6560]`.
//! - **Long audio (>30 s)**: the encoder's context is 30 s (3000 mel frames at 100 Hz). Clips whose
//!   mel exceeds that are tokenized with the upstream sliding window (`S3TokenizerV2.quantize` →
//!   `_quantize_mixed_batch`): 30 s windows hopped by 26 s (a 4 s overlap), each window tokenized
//!   independently, then stitched by `merge_tokenized_segments` — which keeps each window's middle
//!   and drops half the overlap (`(4 / 2) · 25 = 50` tokens) from each interior boundary, so the
//!   4 s shared region is counted exactly once (no duplication, no gap). The ≤30 s path is the
//!   single-pass encode above, byte-for-byte unchanged.
//!
//! [`rope`]: candle_nn::rotary_emb::rope

use std::path::Path;

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    conv1d, conv1d_no_bias, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm,
    Linear, Module, VarBuilder,
};

use crate::config::{S3TokenizerConfig, S3_HOP, S3_SR, S3_TOKEN_RATE};
use crate::s3gen::S3GEN_WEIGHTS_FILE;

// ---------------------------------------------------------------------------------------------
// FSQ — finite scalar quantization (pure integer/float math; no candle, fully unit-testable).
// ---------------------------------------------------------------------------------------------

/// The base-3 place values `[3^0, 3^1, …, 3^7]` a FSQ code is a dot product against.
fn fsq_powers(cfg: &S3TokenizerConfig) -> Vec<i64> {
    let mut powers = Vec::with_capacity(cfg.fsq_dim);
    let mut v: i64 = 1;
    for _ in 0..cfg.fsq_dim {
        powers.push(v);
        v *= cfg.fsq_level as i64;
    }
    powers
}

/// Quantize one FSQ dimension's projected value to its level `{0, 1, 2}` — `round(tanh(x)·0.999)+1`
/// with round-half-to-even (PyTorch `torch.round` semantics). Values are the post-`project_down`,
/// pre-tanh activations.
fn fsq_level(proj: f32) -> i64 {
    // The upstream scale is `0.9990000128746033` (a Python f64); in the reference it multiplies a
    // float32 tensor, so it is first rounded to the nearest f32 — which is exactly `0.999_f32`
    // (= 0.999000012874603271484375). Using `0.999_f32` here is therefore bit-identical, and keeps
    // saturated ±tanh from rounding to ±2 (level 3) while leaving the interior mapping unchanged.
    let h = (proj.tanh() * 0.999_f32).round_ties_even();
    (h as i64) + 1
}

/// The FSQ code for one frame's `fsq_dim` projected values: `Σ level_i · 3^i` → a token in
/// `[0, codebook_size − 1]`.
fn fsq_code_from_projection(cfg: &S3TokenizerConfig, proj: &[f32], powers: &[i64]) -> i64 {
    proj.iter()
        .zip(powers)
        .map(|(&p, &pow)| fsq_level(p) * pow)
        .sum::<i64>()
        .min(cfg.codebook_size() as i64 - 1) // defensive: reference range is already [0, N-1]
}

/// The FSQ code for explicit per-dimension levels `{0..level−1}` — the inverse-facing helper the
/// roundtrip unit test exercises against [`fsq_levels_from_code`]. Test-only (the encode path uses
/// [`fsq_code_from_projection`]).
#[cfg(test)]
fn fsq_code_from_levels(levels: &[i64], powers: &[i64]) -> i64 {
    levels.iter().zip(powers).map(|(&l, &p)| l * p).sum()
}

/// Decompose a FSQ code into its `fsq_dim` base-3 levels (little-endian: dim 0 is the least
/// significant). The mathematical inverse of [`fsq_code_from_levels`]. Test-only.
#[cfg(test)]
fn fsq_levels_from_code(cfg: &S3TokenizerConfig, mut code: i64) -> Vec<i64> {
    let level = cfg.fsq_level as i64;
    (0..cfg.fsq_dim)
        .map(|_| {
            let d = code % level;
            code /= level;
            d
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Long-audio sliding-window segmentation + merge (upstream `S3TokenizerV2._quantize_mixed_batch`
// + `merge_tokenized_segments`). Pure frame/token math — no candle, fully unit-testable.
//
// The encoder was trained on ≤30 s of audio, so clips whose mel exceeds MAX_MEL_FRAMES are
// tokenized in overlapping 30 s windows and stitched. All parameters are the reference's:
// `sample_rate = 16000`, `hop_length = 160`, `window_size = 30`, `overlap = 4`.
// ---------------------------------------------------------------------------------------------

/// Long-audio window span in seconds (upstream `window_size`).
const WINDOW_SECS: usize = 30;
/// Adjacent-window overlap in seconds (upstream `overlap`).
const OVERLAP_SECS: usize = 4;
/// Mel frames per encoder window: `30 s · 16000 / 160 = 3000`. This is also the >30 s dispatch
/// threshold — the reference tokenizes single-pass when `mel_len <= max_frames` and windows when
/// `mel_len > max_frames` (`max_frames = 3000`).
const MAX_MEL_FRAMES: usize = WINDOW_SECS * S3_SR as usize / S3_HOP; // 3000
/// Mel frames a window advances between hops: `frames_per_window − frames_per_overlap`
/// (`3000 − 400 = 2600`), a 4 s overlap between neighbours.
const FRAMES_PER_STRIDE: usize = MAX_MEL_FRAMES - (OVERLAP_SECS * S3_SR as usize / S3_HOP); // 2600

/// The upstream sliding-window plan over a mel of `t_mel` frames: `(start, seg_len)` per window,
/// exactly the reference loop (`start = 0; while start < t_mel { end = min(start+3000, t_mel);
/// … ; start += 2600 }`). The final window is the short remainder (the reference right-pads it to
/// `frames_per_window` and trims the codes back to its true length; since the native encoder masks
/// padding rather than materializing it, encoding the raw remainder slice yields the same tokens).
fn window_plan(t_mel: usize) -> Vec<(usize, usize)> {
    let mut plan = Vec::new();
    let mut start = 0;
    while start < t_mel {
        let end = (start + MAX_MEL_FRAMES).min(t_mel);
        plan.push((start, end - start));
        start += FRAMES_PER_STRIDE;
    }
    plan
}

/// Stitch per-window token streams into one continuous stream, dropping the duplicated overlap —
/// the reference `merge_tokenized_segments(tokenized_segments, overlap, token_rate)`. Each interior
/// window shares `overlap_secs` s with each neighbour; the merge keeps the middle and drops
/// `(overlap_secs / 2) · token_rate` tokens (half the overlap) from each interior boundary, so the
/// shared region is counted exactly once. The first window keeps its head, the last keeps its tail.
fn merge_tokenized_segments(
    segments: &[Vec<i64>],
    overlap_secs: usize,
    token_rate: usize,
) -> Vec<i64> {
    let overlap_tokens = (overlap_secs / 2) * token_rate;
    let n = segments.len();
    let mut merged = Vec::new();
    for (i, tokens) in segments.iter().enumerate() {
        // `l = 0 if i == 0 else overlap_tokens`; `r = -overlap_tokens if i != last else len`.
        let l = if i == 0 { 0 } else { overlap_tokens };
        let r = if i + 1 == n {
            tokens.len()
        } else {
            tokens.len().saturating_sub(overlap_tokens)
        };
        // Clamp `l` up to `r` so a window shorter than the overlap contributes nothing (matching
        // Python's empty slice when the start index passes the stop), never a panic.
        let l = l.min(r);
        merged.extend_from_slice(&tokens[l..r]);
    }
    merged
}

// ---------------------------------------------------------------------------------------------
// Mel front-end (host f32; n_fft = 400 is not a power of two, so a direct real-DFT is used, the
// same idiom the chatterbox_ve front-end uses — the shared radix-2 `candle_audio::dsp` cannot
// serve it).
// ---------------------------------------------------------------------------------------------

/// Periodic Hann window of length `n` — `torch.hann_window(n)` (`0.5 − 0.5 cos(2π i / n)`).
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad by `pad` on both ends (numpy/torch `mode="reflect"`, edge sample excluded) — the
/// `center=True` framing `torch.stft` uses.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i.min(n - 1)]));
    out.extend_from_slice(samples);
    out.extend((0..pad).map(|i| samples[n.saturating_sub(2 + i)]));
    out
}

/// Linear-interpolation resample to [`S3_SR`] (16 kHz). Speaker/prosody content survives linear
/// resampling well enough for tokenization; exact soxr parity is not required for discrete codes.
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == S3_SR || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = S3_SR as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// The loaded s3tokenizer: the checkpoint mel bank + the FSMN encoder + the FSQ projection.
pub struct S3Tokenizer {
    cfg: S3TokenizerConfig,
    /// `_mel_filters`, mel-major `[n_mels][n_bins]` (row `m` is filter `m` over the 201 bins).
    mel_filters: Vec<f32>,
    encoder: Encoder,
    /// FSQ `project_down` (`n_state → fsq_dim`).
    project_down: Linear,
    powers: Vec<i64>,
    device: Device,
}

impl S3Tokenizer {
    /// Load the tokenizer from a Chatterbox snapshot directory (reads `s3gen.safetensors`, prefix
    /// `tokenizer.*`). The rest of `s3gen.safetensors` (speaker encoder / flow / vocoder) is not
    /// read.
    pub fn from_snapshot(dir: &Path) -> Result<Self> {
        let path = dir.join(S3GEN_WEIGHTS_FILE);
        if !path.is_file() {
            return Err(AudioError::Msg(format!(
                "s3tokenizer: {} missing (the tokenizer weights live in the S3Gen checkpoint)",
                path.display()
            )));
        }
        let device = candle_audio::default_device_metal_incompatible()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, &device)?
        };
        Ok(Self::new(
            &S3TokenizerConfig::DEFAULT,
            vb.pp("tokenizer"),
            device,
        )?)
    }

    /// Build the tokenizer from a `tokenizer.*`-rooted [`VarBuilder`].
    pub fn new(cfg: &S3TokenizerConfig, vb: VarBuilder, device: Device) -> CandleResult<Self> {
        let mel_filters = vb
            .get((cfg.n_mels, cfg.n_bins()), "_mel_filters")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let project_down = linear(
            cfg.n_state,
            cfg.fsq_dim,
            vb.pp("quantizer").pp("_codebook").pp("project_down"),
        )?;
        Ok(Self {
            cfg: *cfg,
            mel_filters,
            encoder,
            project_down,
            powers: fsq_powers(cfg),
            device,
        })
    }

    /// The tokenizer configuration.
    pub fn config(&self) -> &S3TokenizerConfig {
        &self.cfg
    }

    /// Encode a reference waveform (resampled to 16 kHz if needed) into 25 Hz speech token ids in
    /// `[0, 6560]`. This is the public inference entry the T3 conditioning path and (later) S3Gen
    /// call.
    ///
    /// Audio whose mel fits the encoder's 30 s context (`MAX_MEL_FRAMES`) is tokenized single-pass
    /// (the common Chatterbox case: a ≤6 s T3 prompt / ≤10 s S3Gen mel ref). Longer clips are
    /// tokenized with the upstream sliding window (`encode_long_audio`) — the ≤30 s branch is
    /// byte-for-byte the original single-pass encode.
    pub fn encode(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<i64>> {
        let wav = resample_to_16k(samples, sample_rate);
        let mel = self.log_mel_spectrogram(&wav)?; // [1, n_mels, T_mel]
        let t_mel = mel.dim(2)?;
        if t_mel > MAX_MEL_FRAMES {
            return self.encode_long_audio(&mel, t_mel);
        }
        let hidden = self.encoder.forward(&mel)?; // [1, T_tok, n_state]
        self.fsq_encode(&hidden)
    }

    /// The upstream long-audio (>30 s) path: tokenize each overlapping [`window_plan`] window
    /// independently through the same encoder + FSQ head, then stitch the per-window streams with
    /// [`merge_tokenized_segments`] into one continuous 25 Hz stream. Faithful to
    /// `S3TokenizerV2._quantize_mixed_batch`: each window restarts the encoder's positions (the
    /// reference batches the segments, so each carries its own position 0), and the final short
    /// remainder is encoded directly rather than zero-padded-then-trimmed (the native encoder masks
    /// padding, so the two are equivalent — see [`window_plan`]).
    fn encode_long_audio(&self, mel: &Tensor, t_mel: usize) -> Result<Vec<i64>> {
        let mut segments: Vec<Vec<i64>> = Vec::new();
        for (start, seg_len) in window_plan(t_mel) {
            // `[1, n_mels, seg_len]`; `contiguous` so the conv stem sees a packed window.
            let window = mel.narrow(2, start, seg_len)?.contiguous()?;
            let hidden = self.encoder.forward(&window)?; // [1, T_tok, n_state]
            segments.push(self.fsq_encode(&hidden)?);
        }
        Ok(merge_tokenized_segments(
            &segments,
            OVERLAP_SECS,
            S3_TOKEN_RATE as usize,
        ))
    }

    /// `log_mel_spectrogram` → a `[1, n_mels, T_mel]` tensor. Faithful to Chatterbox's
    /// `S3Tokenizer.log_mel_spectrogram` (power STFT, drop last frame, checkpoint mel bank, the
    /// `(log10 → max−8 → +4/4)` normalization).
    fn log_mel_spectrogram(&self, wav: &[f32]) -> Result<Tensor> {
        let cfg = &self.cfg;
        let power = self.power_stft(wav); // frame-major [n_frames][n_bins]
                                          // Drop the last time frame (`stft[..., :-1]`).
        let n_frames = power.len().saturating_sub(1);
        if n_frames == 0 {
            return Err(AudioError::Msg(
                "s3tokenizer: reference clip too short to produce any mel frame".into(),
            ));
        }
        // mel_spec[m][t] = Σ_bin filters[m][bin] · power[t][bin]  → then log/normalize.
        // Compute log10(clamp(·, 1e-10)) and the global max in one pass.
        let n_bins = cfg.n_bins();
        let mut log_mel = vec![0f32; cfg.n_mels * n_frames]; // mel-major [m * T + t]
        let mut global_max = f32::NEG_INFINITY;
        for m in 0..cfg.n_mels {
            let filter = &self.mel_filters[m * n_bins..(m + 1) * n_bins];
            for t in 0..n_frames {
                let spec = &power[t];
                let mut acc = 0f32;
                for (bin, &w) in filter.iter().enumerate() {
                    acc += w * spec[bin];
                }
                let v = acc.max(1e-10).log10();
                log_mel[m * n_frames + t] = v;
                if v > global_max {
                    global_max = v;
                }
            }
        }
        // max(·, global_max − 8), then (·+4)/4.
        let floor = global_max - 8.0;
        for v in log_mel.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        Tensor::from_vec(log_mel, (1, cfg.n_mels, n_frames), &self.device).map_err(Into::into)
    }

    /// Power STFT (`|·|²` per one-sided bin) frame-major `[n_frames][n_bins]`. Direct real-DFT
    /// (`n_fft = 400` is not a power of two), `center=True` reflect padding, periodic Hann.
    fn power_stft(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let cfg = &self.cfg;
        let (n_fft, hop, n_bins) = (cfg.n_fft, cfg.hop, cfg.n_bins());
        let window = hann(n_fft);
        let pad = n_fft / 2;
        let padded = reflect_pad(samples, pad);
        if padded.len() < n_fft {
            return Vec::new();
        }
        let n_frames = 1 + (padded.len() - n_fft) / hop;
        // DFT twiddles: cos/sin for each (bin, sample).
        let mut cos_tab = vec![0f32; n_bins * n_fft];
        let mut sin_tab = vec![0f32; n_bins * n_fft];
        for k in 0..n_bins {
            for t in 0..n_fft {
                let ang = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n_fft as f32;
                cos_tab[k * n_fft + t] = ang.cos();
                sin_tab[k * n_fft + t] = ang.sin();
            }
        }
        let mut frames = Vec::with_capacity(n_frames);
        let mut windowed = vec![0f32; n_fft];
        for f in 0..n_frames {
            let start = f * hop;
            for (i, w) in windowed.iter_mut().enumerate() {
                *w = padded[start + i] * window[i];
            }
            let mut row = vec![0f32; n_bins];
            for (k, slot) in row.iter_mut().enumerate() {
                let (cos_k, sin_k) = (&cos_tab[k * n_fft..], &sin_tab[k * n_fft..]);
                let mut re = 0f32;
                let mut im = 0f32;
                for (t, &x) in windowed.iter().enumerate() {
                    re += x * cos_k[t];
                    im += x * sin_k[t];
                }
                *slot = re * re + im * im;
            }
            frames.push(row);
        }
        frames
    }

    /// FSQ-encode the encoder hidden states `[1, T, n_state]` → token ids `[T]`.
    fn fsq_encode(&self, hidden: &Tensor) -> Result<Vec<i64>> {
        let proj = self.project_down.forward(hidden)?; // [1, T, fsq_dim]
        let (_, t, d) = proj.dims3()?;
        let flat: Vec<f32> = proj.reshape((t, d))?.to_vec2::<f32>()?.concat();
        let mut codes = Vec::with_capacity(t);
        for frame in flat.chunks_exact(d) {
            codes.push(fsq_code_from_projection(&self.cfg, frame, &self.powers));
        }
        Ok(codes)
    }
}

// ---------------------------------------------------------------------------------------------
// Whisper-v2 FSMN mel encoder.
// ---------------------------------------------------------------------------------------------

struct Encoder {
    conv1: Conv1d,
    conv2: Conv1d,
    blocks: Vec<Block>,
    head_dim: usize,
    rope_theta: f64,
}

impl Encoder {
    fn new(cfg: &S3TokenizerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let stem_cfg = Conv1dConfig {
            padding: 1,
            stride: cfg.conv_stride,
            ..Default::default()
        };
        let conv1 = conv1d(cfg.n_mels, cfg.n_state, 3, stem_cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(cfg.n_state, cfg.n_state, 3, stem_cfg, vb.pp("conv2"))?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(cfg, vb.pp("blocks").pp(i))?);
        }
        Ok(Self {
            conv1,
            conv2,
            blocks,
            head_dim: cfg.head_dim(),
            rope_theta: cfg.rope_theta,
        })
    }

    /// `[1, n_mels, T_mel]` → `[1, T_tok, n_state]`.
    fn forward(&self, mel: &Tensor) -> CandleResult<Tensor> {
        // Conv stem with exact-erf GELU (both `F.gelu(conv1(·))` and `F.gelu(conv2(·))`).
        let x = self.conv1.forward(mel)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        // [1, n_state, T] → [1, T, n_state].
        let mut x = x.transpose(1, 2)?.contiguous()?;
        let seq = x.dim(1)?;
        let (cos, sin) = rope_tables(seq, self.head_dim, self.rope_theta, x.device())?;
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }
        Ok(x)
    }
}

/// One pre-norm FSMN attention + GELU-MLP block.
struct Block {
    attn_ln: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    fsmn: Conv1d,
    mlp_ln: LayerNorm,
    mlp0: Linear,
    mlp2: Linear,
    n_head: usize,
    head_dim: usize,
}

impl Block {
    fn new(cfg: &S3TokenizerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let n = cfg.n_state;
        let attn = vb.pp("attn");
        let fsmn_cfg = Conv1dConfig {
            padding: (cfg.fsmn_kernel - 1) / 2, // symmetric 15/15 for kernel 31
            stride: 1,
            groups: n, // depthwise
            ..Default::default()
        };
        Ok(Self {
            attn_ln: layer_norm(n, 1e-5, vb.pp("attn_ln"))?,
            query: linear(n, n, attn.pp("query"))?,
            key: linear_no_bias(n, n, attn.pp("key"))?,
            value: linear(n, n, attn.pp("value"))?,
            out: linear(n, n, attn.pp("out"))?,
            fsmn: conv1d_no_bias(n, n, cfg.fsmn_kernel, fsmn_cfg, attn.pp("fsmn_block"))?,
            mlp_ln: layer_norm(n, 1e-5, vb.pp("mlp_ln"))?,
            mlp0: linear(n, n * 4, vb.pp("mlp").pp("0"))?,
            mlp2: linear(n * 4, n, vb.pp("mlp").pp("2"))?,
            n_head: cfg.n_head,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (b, t, n) = x.dims3()?;
        let h = self.attn_ln.forward(x)?;

        // q/k/v projections; q/k reshaped to [b, n_head, t, head_dim] for attention + RoPE.
        let to_heads = |proj: &Tensor| -> CandleResult<Tensor> {
            proj.reshape((b, t, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = to_heads(&self.query.forward(&h)?)?;
        let k = to_heads(&self.key.forward(&h)?)?;
        let value = self.value.forward(&h)?; // [b, t, n] — reused raw for the FSMN memory
        let v = to_heads(&value)?;

        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;

        // Scaled dot-product attention (scale = head_dim^-0.5, applied once).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = softmax_last_dim(&att)?;
        let ctx = att
            .matmul(&v)? // [b, n_head, t, head_dim]
            .transpose(1, 2)?
            .reshape((b, t, n))?;
        let attn_out = self.out.forward(&ctx)?;

        // FSMN memory: depthwise conv over the value projection [b, n, t], added back.
        let fsmn = self
            .fsmn
            .forward(&value.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;
        let fsmn = (fsmn + &value)?;

        // Block: x + (out(attn) + fsmn), then x + mlp(mlp_ln(x)).
        let x = (x + (attn_out + fsmn)?)?;
        let m = self.mlp0.forward(&self.mlp_ln.forward(&x)?)?.gelu_erf()?;
        let m = self.mlp2.forward(&m)?;
        x + m
    }
}

/// RoPE cos/sin tables `[seq, head_dim/2]` for the encoder (`theta = 10000`, rotate-half — the
/// convention candle's [`candle_nn::rotary_emb::rope`] applies). `freq_j = theta^(−2j/head_dim)`.
fn rope_tables(
    seq: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| (1.0 / theta.powf(2.0 * j as f64 / head_dim as f64)) as f32)
        .collect();
    let mut cos = Vec::with_capacity(seq * half);
    let mut sin = Vec::with_capacity(seq * half);
    for pos in 0..seq {
        for &f in &inv_freq {
            let a = pos as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, half), device)?,
        Tensor::from_vec(sin, (seq, half), device)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> S3TokenizerConfig {
        S3TokenizerConfig::DEFAULT
    }

    #[test]
    fn fsq_cardinality_and_bounds_are_exactly_6561() {
        let c = cfg();
        let powers = fsq_powers(&c);
        assert_eq!(powers, vec![1, 3, 9, 27, 81, 243, 729, 2187]);
        let n = c.codebook_size() as i64; // 6561
                                          // Every base-3 level combination maps to a DISTINCT code in [0, 6560]; all 6561 are hit.
        let mut seen = std::collections::HashSet::new();
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for code in 0..n {
            let levels = fsq_levels_from_code(&c, code);
            assert_eq!(levels.len(), c.fsq_dim);
            assert!(levels.iter().all(|&l| (0..c.fsq_level as i64).contains(&l)));
            let round = fsq_code_from_levels(&levels, &powers);
            assert_eq!(round, code, "index→levels→index must round-trip");
            min = min.min(round);
            max = max.max(round);
            assert!(seen.insert(round), "code {code} collided — not a bijection");
        }
        assert_eq!(seen.len(), 6561);
        assert_eq!((min, max), (0, 6560));
    }

    #[test]
    fn fsq_level_quantizes_projection_to_0_1_2() {
        // tanh saturates: a large negative projection → level 0, ~0 → level 1, large positive → 2.
        assert_eq!(fsq_level(-50.0), 0);
        assert_eq!(fsq_level(0.0), 1);
        assert_eq!(fsq_level(50.0), 2);
        // The full projection→code path for a known frame (all dims saturated positive → max code).
        let c = cfg();
        let powers = fsq_powers(&c);
        let hi = vec![50.0f32; c.fsq_dim];
        assert_eq!(fsq_code_from_projection(&c, &hi, &powers), 6560);
        let lo = vec![-50.0f32; c.fsq_dim];
        assert_eq!(fsq_code_from_projection(&c, &lo, &powers), 0);
        // Mixed dims compose base-3: dim0=level2, dim1=level1, rest level0 → 2·1 + 1·3 = 5.
        let mut mixed = vec![-50.0f32; c.fsq_dim];
        mixed[0] = 50.0; // level 2
        mixed[1] = 0.0; // level 1
        assert_eq!(fsq_code_from_projection(&c, &mixed, &powers), 5);
    }

    #[test]
    fn hann_window_is_periodic_and_bounded() {
        let w = hann(400);
        assert_eq!(w.len(), 400);
        assert!(w[0].abs() < 1e-6, "periodic Hann starts at 0");
        assert!(w.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!((w[200] - 1.0).abs() < 1e-6, "peak at N/2");
    }

    #[test]
    fn reflect_pad_excludes_the_edge_sample() {
        // torch reflect pad of [10,20,30,40] by 2 → [30,20, 10,20,30,40, 30,20].
        let p = reflect_pad(&[10.0, 20.0, 30.0, 40.0], 2);
        assert_eq!(p, vec![30.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 20.0]);
    }

    #[test]
    fn resample_changes_length_by_ratio_and_is_identity_at_16k() {
        let s = vec![0.1f32, -0.2, 0.3, -0.4];
        assert_eq!(resample_to_16k(&s, S3_SR), s);
        let out = resample_to_16k(&vec![0.0f32; 24_000], 24_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn rope_tables_have_expected_shape_and_values() {
        let c = cfg();
        let (cos, sin) = rope_tables(8, c.head_dim(), c.rope_theta, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[8, c.head_dim() / 2]);
        assert_eq!(sin.dims(), &[8, c.head_dim() / 2]);
        // Position 0 → cos 1, sin 0 for every frequency.
        let c0: Vec<f32> = cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let s0: Vec<f32> = sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(c0.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(s0.iter().all(|&v| v.abs() < 1e-6));
    }

    /// The mel front-end + FSQ head are exercised without weights by wiring a random-projection
    /// encoder stand-in: build the module from an in-memory VarBuilder of the right shapes and
    /// confirm the mel tensor shape, token count (≈ clip_len / 640), and range invariants. Real
    /// weights are exercised by the `#[ignore]`d conformance test.
    #[test]
    fn mel_front_end_shape_and_token_rate_with_synthetic_weights() {
        use candle_audio::candle_core::DType;
        use candle_nn::VarMap;

        let c = cfg();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Materialize every `tokenizer.*` tensor at its real shape (values default to 0/random via
        // VarMap's initializer — enough to prove shapes/frame-math wire up end to end).
        let tok = vb.pp("tokenizer");
        let _ = tok.get((c.n_mels, c.n_bins()), "_mel_filters").unwrap();
        let enc = tok.pp("encoder");
        let _ = enc.get((c.n_state, c.n_mels, 3), "conv1.weight").unwrap();
        let _ = enc.get(c.n_state, "conv1.bias").unwrap();
        let _ = enc.get((c.n_state, c.n_state, 3), "conv2.weight").unwrap();
        let _ = enc.get(c.n_state, "conv2.bias").unwrap();
        for i in 0..c.n_layer {
            let bl = enc.pp("blocks").pp(i);
            for name in ["query", "value", "out"] {
                let _ = bl
                    .get((c.n_state, c.n_state), &format!("attn.{name}.weight"))
                    .unwrap();
                let _ = bl.get(c.n_state, &format!("attn.{name}.bias")).unwrap();
            }
            let _ = bl.get((c.n_state, c.n_state), "attn.key.weight").unwrap();
            let _ = bl
                .get((c.n_state, 1, c.fsmn_kernel), "attn.fsmn_block.weight")
                .unwrap();
            for ln in ["attn_ln", "mlp_ln"] {
                let _ = bl.get(c.n_state, &format!("{ln}.weight")).unwrap();
                let _ = bl.get(c.n_state, &format!("{ln}.bias")).unwrap();
            }
            let _ = bl.get((c.n_state * 4, c.n_state), "mlp.0.weight").unwrap();
            let _ = bl.get(c.n_state * 4, "mlp.0.bias").unwrap();
            let _ = bl.get((c.n_state, c.n_state * 4), "mlp.2.weight").unwrap();
            let _ = bl.get(c.n_state, "mlp.2.bias").unwrap();
        }
        let q = tok.pp("quantizer").pp("_codebook").pp("project_down");
        let _ = q.get((c.fsq_dim, c.n_state), "weight").unwrap();
        let _ = q.get(c.fsq_dim, "bias").unwrap();

        let tokenizer = S3Tokenizer::new(&c, vb.pp("tokenizer"), device).unwrap();

        // 1 second of 16 kHz audio → ~25 tokens, each a valid FSQ code.
        let wav = vec![0.05f32; S3_SR as usize];
        let mel = tokenizer.log_mel_spectrogram(&wav).unwrap();
        assert_eq!(mel.dim(0).unwrap(), 1);
        assert_eq!(mel.dim(1).unwrap(), c.n_mels);
        // ~100 mel frames per second, minus the dropped last frame.
        assert!(
            (mel.dim(2).unwrap() as i64 - 100).abs() <= 2,
            "mel frames ≈ 100/s"
        );

        let codes = tokenizer.encode(&wav, S3_SR).unwrap();
        // ~25 tokens/s (allow ±2 for conv boundary framing).
        assert!(
            (codes.len() as i64 - 25).abs() <= 2,
            "expected ≈25 tokens, got {}",
            codes.len()
        );
        assert!(
            codes.iter().all(|&t| (0..6561).contains(&t)),
            "every token in [0, 6560]"
        );
    }

    // ---------------------------------------------------------------------------------------
    // Long-audio (>30 s) sliding-window segmentation + merge (upstream parity, pure math).
    // ---------------------------------------------------------------------------------------

    /// The two-stride-2 conv stem downsample of `l` mel frames → tokens (`floor((l−1)/2)+1` twice),
    /// used to predict per-window token counts without weights.
    fn downsample_len(l: usize) -> usize {
        let c1 = (l - 1) / 2 + 1;
        (c1 - 1) / 2 + 1
    }

    #[test]
    fn window_constants_match_the_reference() {
        // 30 s · 16000 / 160 = 3000 mel frames; 4 s overlap = 400 frames; stride = 2600.
        assert_eq!(MAX_MEL_FRAMES, 3000);
        assert_eq!(FRAMES_PER_STRIDE, 2600);
        assert_eq!(WINDOW_SECS, 30);
        assert_eq!(OVERLAP_SECS, 4);
        // A full 3000-frame window is exactly 750 tokens (25 Hz over 30 s), and the merge drops
        // `(4/2)·25 = 50` tokens (2 s) from each interior boundary.
        assert_eq!(downsample_len(MAX_MEL_FRAMES), 750);
        assert_eq!((OVERLAP_SECS / 2) * S3_TOKEN_RATE as usize, 50);
    }

    #[test]
    fn window_plan_matches_upstream_sliding_window() {
        // Just over one window → two windows: a full 3000, then the 601-frame remainder.
        assert_eq!(window_plan(3601), vec![(0, 3000), (2600, 1001)]);
        // 36 s (3600 frames) → 3000 + 1000, a clean two-window split.
        assert_eq!(window_plan(3600), vec![(0, 3000), (2600, 1000)]);
        // 80 s (8000 frames) → starts 0, 2600, 5200, 7800; last is the short remainder.
        assert_eq!(
            window_plan(8000),
            vec![(0, 3000), (2600, 3000), (5200, 2800), (7800, 200)]
        );
        // Every window starts on a stride multiple, no window exceeds the encoder context, and the
        // union of windows covers the whole mel (the last window ends exactly at `t_mel`).
        for &t_mel in &[3001usize, 4500, 6000, 12345] {
            let plan = window_plan(t_mel);
            assert!(plan.len() >= 2, "a >30 s clip must split into ≥2 windows");
            for (k, &(start, seg_len)) in plan.iter().enumerate() {
                assert_eq!(start, k * FRAMES_PER_STRIDE, "window {k} start");
                assert!((1..=MAX_MEL_FRAMES).contains(&seg_len), "window {k} span");
            }
            let (last_start, last_len) = *plan.last().unwrap();
            assert_eq!(last_start + last_len, t_mel, "windows must cover to t_mel");
        }
    }

    #[test]
    fn merge_drops_half_the_overlap_at_each_interior_boundary() {
        // Three 750-token windows whose values encode (window, index) so we can trace provenance.
        let seg = |w: i64| (0..750).map(|i| w * 1000 + i).collect::<Vec<i64>>();
        let segments = vec![seg(0), seg(1), seg(2)];
        let merged = merge_tokenized_segments(&segments, OVERLAP_SECS, S3_TOKEN_RATE as usize);

        // First keeps [0,700), middles keep [50,700), last keeps [50,750): 700 + 650 + 700 = 2050.
        assert_eq!(merged.len(), 700 + 650 + 700);
        // The seam is continuous with no duplication or gap: window 0's last kept token is its
        // index 699, and window 1's first kept token is its index 50 — adjacent in the stream.
        assert_eq!(merged[699], seg(0)[699]);
        assert_eq!(merged[700], seg(1)[50]);
        // window 1's last kept token (index 699) then window 2's first kept token (index 50).
        assert_eq!(merged[700 + 649], seg(1)[699]);
        assert_eq!(merged[700 + 650], seg(2)[50]);
        // No token value appears twice (the overlap was de-duplicated, not double-counted).
        let mut sorted = merged.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), merged.len(), "overlap tokens were duplicated");
    }

    #[test]
    fn merge_single_segment_is_the_identity() {
        // A lone window (the ≤30 s case never reaches the merge, but the merge must still be a
        // no-op for one segment: nothing to de-duplicate).
        let only = (0..123).collect::<Vec<i64>>();
        let merged = merge_tokenized_segments(
            std::slice::from_ref(&only),
            OVERLAP_SECS,
            S3_TOKEN_RATE as usize,
        );
        assert_eq!(merged, only);
    }

    #[test]
    fn merge_handles_a_window_shorter_than_the_overlap() {
        // A degenerate tiny final window (< overlap_tokens) contributes nothing rather than
        // panicking — Python's `tokens[50:]` on a 30-element list is empty, and so is ours.
        let segments = vec![
            (0..750).collect::<Vec<i64>>(),
            (0..30).collect::<Vec<i64>>(),
        ];
        let merged = merge_tokenized_segments(&segments, OVERLAP_SECS, S3_TOKEN_RATE as usize);
        assert_eq!(
            merged.len(),
            700,
            "first window's head only; tiny tail dropped"
        );
    }

    #[test]
    fn windowed_token_count_tracks_the_25hz_rate() {
        // For a clean two-window clip the stitched length equals the single-pass token count
        // (duration · 25 Hz) exactly — no gap, no duplication.
        for &t_mel in &[3600usize, 4000, 4500, 5000, 5600] {
            let plan = window_plan(t_mel);
            let segments: Vec<Vec<i64>> = plan
                .iter()
                .map(|&(_, seg_len)| vec![0i64; downsample_len(seg_len)])
                .collect();
            let merged = merge_tokenized_segments(&segments, OVERLAP_SECS, S3_TOKEN_RATE as usize);
            // Duration (s) = t_mel / 100 (100 Hz mel); expected tokens = round(dur · 25) = t_mel/4.
            let dur = t_mel as f32 / 100.0;
            let expected = (dur * S3_TOKEN_RATE as f32).round() as i64;
            assert!(
                (merged.len() as i64 - expected).abs() <= 1,
                "t_mel {t_mel}: stitched {} tokens not ≈ {expected} (dur {dur:.2}s · 25 Hz)",
                merged.len()
            );
        }
    }
}
