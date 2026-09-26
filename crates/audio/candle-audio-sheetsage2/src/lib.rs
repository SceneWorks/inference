//! Native SheetSage2 + MERT-v2-FullSong recording-to-score transcription, and the YuE2 zero-shot
//! cover path built on it (epic sc-22988, story sc-22996).
//!
//! # Licence — noncommercial, never relicensed
//!
//! This crate is a port of the remote Python published inside `m-a-p/SheetSage2` and
//! `m-a-p/MERT-v2-FullSong`. That code carries no code licence of its own and both repositories
//! declare `cc-by-nc-4.0`, so — per the disposition recorded by sc-22989 in
//! [`candle_audio_yue2::license::CODE_TERMS`] — code derived from it is treated as **CC BY-NC 4.0**:
//! noncommercial use only, with attribution to Multimodal Art Projection (SheetSage2 /
//! MERT-v2-FullSong, <https://huggingface.co/m-a-p/SheetSage2>,
//! <https://huggingface.co/m-a-p/MERT-v2-FullSong>). It is a separate crate precisely so that none of
//! it enters the Apache-2.0 `candle-audio-yue2` crate; the Cargo `license` field says
//! `CC-BY-NC-4.0`, and `deny.toml` admits it by a scoped, reasoned exception. Every entry point that
//! runs the models first passes [`candle_audio_yue2::license::authorize`] for
//! [`IntendedUse::NoncommercialExperimentation`](candle_audio_yue2::license::IntendedUse) over the
//! cover closure.
//!
//! # What is here
//!
//! * [`tokenizer`], [`grammar`], [`events`], [`pipeline`] — the computed vocabulary (fingerprint
//!   checked against the checkpoint), the decoding grammar, token ↔ event conversion, the whole-song
//!   window plan and overlap stitching.
//! * [`model`] — the Candle model: the MERT2 mel front end (from the checkpoint's own window,
//!   filterbank and normalization buffers), ConvNeXt-v2 + GRN subsampler, 24 Conformer blocks with
//!   query-chunked attention, the rank-64 LoRA merged in fp32 at load, the 25-state layer mix, and
//!   the BART decoder with cached self- and cross-attention, decoded greedily under the grammar.
//! * [`exports`], [`notation`], [`midi`], [`chord_spelling`] — the upstream export set (events,
//!   LAB, MIDI, playback map) and the validated two-voice ABC, byte-identical to upstream's for the
//!   same tokens; head-revision key-aware chord spelling.
//! * [`review`] — the persisted, replayable review artifact: source identity, settings, closure
//!   identity, exact tokens, every export, full **and** melody-only ABC, warnings (octave evidence,
//!   harmony collapse, voice assignment, sparse melody) and the explicit empty-melody refusal
//!   upstream lacks.
//! * [`provider`] — [`provider::Transcriber`]: offline load from the verified cover closure,
//!   transcription, and an observable [`provider::Transcriber::unload`].
//! * [`cover`] — reviewed ABC → a YuE2 `cot=melody` / `cot=full` [`candle_audio_yue2::SongRequest`]
//!   with target style and aligned source or translated lyrics, and the transcribe → unload →
//!   generate sequencing.
//!
//! # Code revision
//!
//! The weights are pinned at `m-a-p/SheetSage2@eab522a8168e8b8b8c4856bf8609cd86198f01fe` (the
//! inventory's pin). The post-processing is ported from the **head** code revision
//! `4f89269db831bdc1880124164a00d4f9385cd129`, whose weights are the same LFS object and whose only
//! code differences are key-aware chord spelling and canonical key labels (see [`chord_spelling`]).

#![deny(rustdoc::private_intra_doc_links)]

pub mod chord_spelling;
// pub mod cover;
pub mod events;
pub mod exports;
pub mod grammar;
pub mod midi;
pub mod model;
pub mod notation;
pub mod octave;
pub mod pipeline;
pub mod provider;
pub mod pyfmt;
pub mod review;
pub mod tokenizer;

pub use candle_audio::candle_core;

/// The upstream code revision the post-processing is ported from.
pub const PORTED_CODE_REVISION: &str = "4f89269db831bdc1880124164a00d4f9385cd129";

/// Every failure of the transcription and cover path.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A checkpoint or configuration does not match what the port implements.
    #[error("configuration: {0}")]
    Config(String),
    /// A caller request is invalid.
    #[error("request: {0}")]
    Request(String),
    /// A token sequence could not be decoded.
    #[error("decode: {0}")]
    Decode(String),
    /// Symbolic post-processing failed.
    #[error("symbolic: {0}")]
    Symbolic(String),
    /// The cover closure failed verification or the licence gate.
    #[error("closure: {0}")]
    Closure(String),
    /// A persisted artifact failed its integrity or replay check.
    #[error("replay: {0}")]
    Replay(String),
    /// The transcription is not usable for the requested purpose (e.g. no melody for a cover).
    #[error("refused: {0}")]
    Refused(String),
    /// Filesystem failure.
    #[error("io {path}: {source}")]
    Io {
        /// The path.
        path: std::path::PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// A tensor operation failed.
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
}

impl Error {
    pub(crate) fn io(path: &std::path::Path, source: std::io::Error) -> Self {
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Serializes tests that observe process-wide state (the live-model count).
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
