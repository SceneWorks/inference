//! MiniMax-H3 audio-VAE configuration (`MiniMaxH3AudioVAE` / `DacAudioVAE`).
//!
//! The config parsing, the `sample_rate`-selected BigVGAN block and every validation are
//! backend-neutral and live in [`candle_gen::gen_core::h3_audio_config`] (sc-27256), shared with the
//! MLX twin; this module re-exports it so `audio_config::*` paths stay stable.

pub use candle_gen::gen_core::h3_audio_config::*;
