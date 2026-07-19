//! Frozen architecture + preprocessing constants for `laion/clap-htsat-unfused` (sc-12851).
//!
//! These are pinned in code (not read from `config.json`) because the provider ports a *specific*
//! checkpoint's architecture — the HTSAT (Swin) audio tower and the RoBERTa text tower — and a
//! mismatch between the code and a config field would be a silent numeric bug, not a graceful
//! reconfiguration. `config.json` is still fetched (and its `model_type` checked by the preparer),
//! but the shapes below are the contract the ported modules assume.

/// The joint-space projection dimension (both towers project to this; the returned embedding dim).
pub const PROJECTION_DIM: usize = 512;

// ---- audio tower (HTSAT / Swin) --------------------------------------------------------------

/// Swin embed dim `C` (`patch_embeds_hidden_size`).
pub const AUDIO_EMBED_DIM: usize = 96;
/// Per-stage transformer depths.
pub const AUDIO_DEPTHS: [usize; 4] = [2, 2, 6, 2];
/// Per-stage attention head counts.
pub const AUDIO_HEADS: [usize; 4] = [4, 8, 16, 32];
/// Attention window edge (tokens).
pub const AUDIO_WINDOW_SIZE: usize = 8;
/// The square Swin input image edge (`spec_size`).
pub const AUDIO_SPEC_SIZE: usize = 256;
/// Number of mel bins.
pub const AUDIO_NUM_MEL_BINS: usize = 64;
/// Patch conv kernel/stride edge.
pub const AUDIO_PATCH_SIZE: usize = 4;
/// MLP hidden expansion ratio.
pub const AUDIO_MLP_RATIO: f64 = 4.0;
/// LayerNorm epsilon for the audio tower.
pub const AUDIO_LN_EPS: f64 = 1e-5;
/// BatchNorm2d epsilon (torch default) applied over the mel-bin channels before `reshape_mel2img`.
pub const AUDIO_BN_EPS: f64 = 1e-5;
/// `spec_size / num_mel_bins` — the mel→image freq replication ratio (256/64 = 4).
pub const AUDIO_FREQ_RATIO: usize = AUDIO_SPEC_SIZE / AUDIO_NUM_MEL_BINS;

// ---- text tower (RoBERTa) --------------------------------------------------------------------

/// Text hidden size.
pub const TEXT_HIDDEN: usize = 768;
/// Text transformer layers.
pub const TEXT_LAYERS: usize = 12;
/// Text attention heads.
pub const TEXT_HEADS: usize = 12;
/// Text FFN intermediate size.
pub const TEXT_INTERMEDIATE: usize = 3072;
/// Vocab size (RoBERTa BPE).
pub const TEXT_VOCAB: usize = 50265;
/// Max position embeddings (RoBERTa uses `max_len + pad + 1` = 514).
pub const TEXT_MAX_POS: usize = 514;
/// Token-type vocab (RoBERTa has one type).
pub const TEXT_TYPE_VOCAB: usize = 1;
/// LayerNorm epsilon for the text tower.
pub const TEXT_LN_EPS: f64 = 1e-12;
/// Padding token id — RoBERTa's position ids are offset past this (positions start at `pad+1`).
pub const TEXT_PAD_TOKEN_ID: u32 = 1;
/// Cap on text query length in tokens (defensive; CLAP text is trained on short captions).
pub const TEXT_MAX_TOKENS: usize = 256;

// ---- mel front-end (ClapFeatureExtractor, `truncation="rand_trunc"`) --------------------------

/// Native CLAP sample rate (Hz).
pub const SAMPLE_RATE: u32 = 48_000;
/// STFT window / FFT size.
pub const N_FFT: usize = 1024;
/// STFT hop.
pub const HOP: usize = 480;
/// One-sided FFT bins (`N_FFT/2 + 1`).
pub const N_FREQ_BINS: usize = N_FFT / 2 + 1;
/// Slaney mel filterbank lower edge (Hz).
pub const MEL_FMIN: f32 = 50.0;
/// Slaney mel filterbank upper edge (Hz).
pub const MEL_FMAX: f32 = 14_000.0;

/// The exact number of STFT frames we feed the Swin tower: `spec_size * freq_ratio` (256 * 4 =
/// 1024). We pad/truncate the waveform to produce exactly this many frames so `reshape_mel2img`
/// hits its native `spec_width` with **no** bicubic interpolation — a faithful, interpolation-free
/// CLAP inference over a ~10.2 s window.
pub const TARGET_FRAMES: usize = AUDIO_SPEC_SIZE * AUDIO_FREQ_RATIO;

/// The waveform sample count (at [`SAMPLE_RATE`]) that yields exactly [`TARGET_FRAMES`] frames under
/// a `center=True` STFT: `n_frames = 1 + len/hop` ⇒ `len = (TARGET_FRAMES - 1) * HOP`.
pub const TARGET_SAMPLES: usize = (TARGET_FRAMES - 1) * HOP;
