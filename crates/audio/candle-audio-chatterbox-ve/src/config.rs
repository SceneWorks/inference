//! Front-end + encoder constants for the Chatterbox voice encoder (sc-12844).
//!
//! These reproduce the Resemblyzer/Chatterbox `VoiceEncoder` preprocessing exactly (Chatterbox's
//! `models/voice_encoder`): a 16 kHz mono mel front-end (`n_fft = 400`, `hop = 160`, `40` mels,
//! librosa Slaney mel scale, **raw power** — no log) feeding a 3-layer LSTM (256 hidden) + a
//! 256→256 projection, embeddings averaged over ~1.6 s partial utterances. Getting these values
//! wrong would silently mis-condition the encoder against its trained front-end, so they live in
//! one audited place next to the port.

/// Encoder operating sample rate (Hz). Reference clips are resampled to this before analysis.
pub const SAMPLE_RATE: u32 = 16_000;

/// STFT window / FFT size (samples) — `25 ms` at 16 kHz, the Resemblyzer `mel_window_length`.
pub const N_FFT: usize = 400;

/// STFT hop (samples) — `10 ms` at 16 kHz, the Resemblyzer `mel_window_step`.
pub const HOP: usize = 160;

/// Mel channels — the encoder's input feature dimension.
pub const N_MELS: usize = 40;

/// LSTM hidden size (and the projected embedding dimension).
pub const HIDDEN: usize = 256;

/// Number of stacked LSTM layers.
pub const NUM_LAYERS: usize = 3;

/// Advertised (and produced) speaker-embedding dimensionality.
pub const EMBED_DIM: usize = 256;

/// Frames per partial utterance (Resemblyzer `partials_n_frames`) — `1.6 s` at a 10 ms hop.
pub const PARTIALS_N_FRAMES: usize = 160;

/// Target loudness (dBFS, power) the reference waveform is normalized to before analysis
/// (Resemblyzer `audio_norm_target_dBFS`, increase-only).
pub const AUDIO_NORM_TARGET_DBFS: f32 = -30.0;
