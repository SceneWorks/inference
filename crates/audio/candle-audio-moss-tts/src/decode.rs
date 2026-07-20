//! The MOSS-TTSD **delay-pattern** autoregressive decode loop (sc-13360): dialogue text → 8-channel
//! RVQ speech-codebook frames.
//!
//! MOSS-TTSD is a *delay-pattern* multi-codebook TTS model (MusicGen/Parler-style), **not** a
//! CSM-style local/depth transformer (that is the sibling `candle-audio-moss-tts-realtime`). One
//! backbone step produces all `channels` (8) codebook logits **simultaneously** through the tied
//! per-channel heads; inter-codebook structure is carried by **time-shifting** each channel `j` by
//! `j` positions (the delay pattern). This module reimplements the reference
//! `MossTTSDGenerationMixin._sample` for batch = 1: build the multi-channel prompt grid, apply the
//! delay shift, prefill, then per step sample every channel (with the reference per-channel
//! constraints), apply the delay-ramp teacher-forcing at the start and the delay-tail **drain** at
//! the end (`needs_additional_steps`), and finally **un-shift** the generated positions back into
//! clean 8-codebook frames (channel-0 mapped out of `speech_token_range`).
//!
//! **Codec boundary.** These frames are the input to OpenMOSS's **XY_Tokenizer** codec (RVQ codes →
//! 24 kHz PCM), a separate from-scratch port not yet landed (see [`crate::model`]); this crate emits
//! and verifies the real AR tokens and errors at the codec boundary rather than fabricate audio.

use candle_audio::candle_core::Result as CandleResult;
use tokenizers::Tokenizer;

use crate::backbone::{Backbone, Frame};
use crate::config::MossTtsdConfig;
use crate::sampling::{sample, Rng, SamplingParams};

/// The default MOSS-TTSD dialogue system prompt (the tokenizer `chat_template` default).
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a speech synthesizer that generates natural, \
realistic, and human-like conversational audio from dialogue text.";

/// One decoded audio frame: `channels` codebook token ids — codebook 0 in `[0, 1024)` (mapped out of
/// the text-channel speech range) and codebooks 1..channels-1 in `[0, speech_vocab_size)`.
pub type RvqFrame = Vec<u32>;

/// Format a single-turn (single-voice) or multi-turn (dialogue) request into the MOSS-TTSD
/// chat-template text, substituting `[S1]`/`[S2]` speaker tags for the `<speaker1>`/`<speaker2>`
/// tokens the model was trained on.
pub fn format_dialogue_text(text: &str) -> String {
    text.replace("[S1]", "<speaker1>")
        .replace("[S2]", "<speaker2>")
}

/// Build the multi-channel prompt grid for a request. Column 0 (text) carries the chat-templated
/// dialogue (ending in `<|begin_of_speech|>`, which kicks off audio generation); every audio channel
/// is the speech pad. Mirrors `MossTTSDSampleProcessor._build_inputs` for the no-reference-audio
/// (pure TTS) path.
pub fn build_prompt_grid(
    tokenizer: &Tokenizer,
    cfg: &MossTtsdConfig,
    dialogue_text: &str,
    system_prompt: &str,
) -> Result<Vec<Frame>, String> {
    let final_text = format_dialogue_text(dialogue_text);
    // The exact tokenizer `chat_template` (no Python/jinja at runtime).
    let prompt = format!(
        "<|begin_of_style|>{system}<|end_of_style|>\n<|begin_of_text|>{text}<|end_of_text|>\n<|begin_of_speech|>",
        system = system_prompt,
        text = final_text,
    );
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenize prompt: {e}"))?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return Err("empty prompt after tokenization".into());
    }
    let pad = cfg.speech_pad_token;
    Ok(ids
        .iter()
        .map(|&id| {
            let mut f = vec![pad; cfg.channels];
            f[0] = id;
            f
        })
        .collect())
}

/// Apply the delay-pattern shift to a `(T, channels)` grid → `(T + channels - 1, channels)`: channel
/// `j` is delayed by `j` positions. The text channel is filled with `text_pad_id`, the audio
/// channels with the speech pad. Mirrors `MossTTSDSampleProcessor._shift_inputs`.
pub fn shift_grid(grid: &[Frame], cfg: &MossTtsdConfig) -> Vec<Frame> {
    let t = grid.len();
    let ch = cfg.channels;
    let new_len = t + ch - 1;
    let mut shifted: Vec<Frame> = (0..new_len)
        .map(|_| {
            let mut f = vec![cfg.speech_pad_token; ch];
            f[0] = cfg.text_pad_id;
            f
        })
        .collect();
    for j in 0..ch {
        for (r, src) in grid.iter().enumerate() {
            shifted[j + r][j] = src[j];
        }
    }
    shifted
}

/// The assembled AR decoder (backbone + config).
pub struct Decoder {
    pub backbone: Backbone,
    pub cfg: MossTtsdConfig,
}

/// Why an AR run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The channel-0 stream emitted the end-of-speech token and the delay tail drained.
    Eos,
    /// The position budget was reached first.
    Budget,
}

/// The result of an AR run: the clean (un-shifted, trimmed) `channels`-codebook frames and why it
/// stopped.
pub struct DecodeResult {
    pub frames: Vec<RvqFrame>,
    pub stop: StopReason,
}

impl Decoder {
    fn is_speech(&self, tok: u32) -> bool {
        tok >= self.cfg.speech_token_range[0] && tok < self.cfg.speech_token_range[1]
    }

    /// Run the delay-pattern AR loop over the prompt `grid` for at most `max_positions` generated
    /// positions, consulting `cancel` before each step and invoking `on_position(step)` for progress.
    /// Returns the clean `channels`-codebook frames (un-shifted, ramp/drain trimmed), or `Ok(None)`
    /// if cancelled. `seed` seeds the per-channel sampler deterministically.
    pub fn run(
        &self,
        grid: Vec<Frame>,
        max_positions: usize,
        seed: u64,
        cancel: &dyn Fn() -> bool,
        on_position: &mut dyn FnMut(usize),
    ) -> CandleResult<Option<DecodeResult>> {
        let cfg = &self.cfg;
        let ch = cfg.channels;
        let strip = ch - 1;
        let params = SamplingParams::default();
        let mut rng = Rng::seed(seed);

        // Delay-shifted grid; strip the last `channels - 1` positions to get the prefill prompt.
        let tf = shift_grid(&grid, cfg);
        let tf_len = tf.len();
        let mut input: Vec<Frame> = tf[..tf_len - strip].to_vec();
        let base = input.len();

        let mut cache = self.backbone.new_cache();
        let mut prefilled = false;
        let mut needs_additional: i64 = -1;
        let mut emitted: Vec<Frame> = Vec::new();
        let mut stop = StopReason::Budget;

        for step in 0..max_positions {
            if cancel() {
                return Ok(None);
            }
            let hidden = if !prefilled {
                prefilled = true;
                self.backbone.prefill(&input, &mut cache)?
            } else {
                self.backbone
                    .step(input.last().expect("a prior position"), &mut cache)?
            };
            let logits = self.backbone.channel_logits(&hidden)?;
            let pos = input.len();

            // Sample every channel with the reference per-channel constraints.
            let mut next: Frame = vec![0u32; ch];
            for (i, row) in logits.into_iter().enumerate() {
                let mut lg = row;
                if i != 0 {
                    // Channel i > 0 may not emit the speech pad until its delay window has opened.
                    if pos >= tf_len - strip + i {
                        let p = cfg.speech_pad_token as usize;
                        if p < lg.len() {
                            lg[p] = f32::NEG_INFINITY;
                        }
                    }
                } else if pos < tf_len {
                    // Channel 0 may not emit end-of-speech while still inside the delay ramp.
                    let e = cfg.speech_eos_token as usize;
                    if e < lg.len() {
                        lg[e] = f32::NEG_INFINITY;
                    }
                }
                let history: Vec<u32> = emitted.iter().map(|f| f[i]).collect();
                next[i] = sample(&mut lg, &history, &params, &mut rng);
            }

            // Drain trigger: channel 0 leaving the speech range starts the delay-tail flush.
            if !self.is_speech(next[0]) && needs_additional < 0 {
                needs_additional = strip as i64;
            }

            // Delay-ramp teacher-forcing: channels whose delay window has not yet opened take their
            // ground-truth shifted value (the pad, for pure TTS) rather than a sampled token.
            if pos < tf_len {
                let i0 = pos + 1 - base;
                next[i0..ch].copy_from_slice(&tf[pos][i0..ch]);
            }

            // Delay-tail drain: once channel 0 has emitted EOS, force EOS on channel 0 and pad each
            // audio channel as its delay window closes (channel 1 first, channel `strip` last).
            if needs_additional > 0 && needs_additional < strip as i64 {
                next[0] = cfg.speech_eos_token;
                for (c, slot) in next.iter_mut().enumerate().skip(1) {
                    if needs_additional < (ch - c) as i64 {
                        *slot = cfg.speech_pad_token;
                    }
                }
            }

            input.push(next.clone());
            emitted.push(next);

            if needs_additional > 0 {
                needs_additional -= 1;
            }
            if needs_additional == 0 {
                stop = StopReason::Eos;
                break;
            }
            on_position(step);
        }

        Ok(Some(DecodeResult {
            frames: self.unshift(&emitted),
            stop,
        }))
    }

    /// Un-shift the generated positions into clean `channels`-codebook frames and trim the delay
    /// ramp/drain: for output index `t`, codebook `j` = `emitted[t + j][j]`; channel-0 codes are
    /// mapped out of `speech_token_range`; only frames whose audio codebooks are all valid (not the
    /// pad) are kept. Mirrors `MossTTSDProcessor.shifting_outputs` + `_find_max_valid_positions`.
    fn unshift(&self, emitted: &[Frame]) -> Vec<RvqFrame> {
        let cfg = &self.cfg;
        let ch = cfg.channels;
        let strip = ch - 1;
        let g = emitted.len();
        if g <= strip {
            return Vec::new();
        }
        let range0 = cfg.speech_token_range[0];
        let mut out = Vec::new();
        for t in 0..(g - strip) {
            let mut cb: RvqFrame = (0..ch).map(|j| emitted[t + j][j]).collect();
            // Keep only fully-active frames (every audio codebook a real, non-pad code).
            if cb[1..].contains(&cfg.speech_pad_token) {
                continue;
            }
            // Channel 0 carries codebook 0 inside the text-vocab speech range; map it back to [0,1024).
            cb[0] = cb[0].saturating_sub(range0);
            out.push(cb);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backbone::tests::{tiny_backbone, tiny_cfg};

    fn tiny_decoder() -> Decoder {
        let cfg = tiny_cfg();
        let backbone = tiny_backbone(&cfg);
        Decoder { backbone, cfg }
    }

    /// A small text-only grid (ids below the tiny text vocab), no real tokenizer.
    fn manual_grid(cfg: &MossTtsdConfig, t: usize) -> Vec<Frame> {
        (0..t)
            .map(|k| {
                let mut f = vec![cfg.speech_pad_token; cfg.channels];
                f[0] = (k as u32 + 1) % cfg.vocab_size as u32;
                f
            })
            .collect()
    }

    #[test]
    fn shift_applies_the_delay_pattern() {
        let cfg = tiny_cfg();
        let grid = manual_grid(&cfg, 3);
        let shifted = shift_grid(&grid, &cfg);
        assert_eq!(shifted.len(), 3 + cfg.channels - 1);
        // Channel j carries grid[.][j] starting at row j.
        for j in 0..cfg.channels {
            for (r, src) in grid.iter().enumerate() {
                assert_eq!(shifted[j + r][j], src[j], "channel {j} row {r}");
            }
        }
        // Text channel outside the content is the text pad.
        assert_eq!(shifted[grid.len()][0], cfg.text_pad_id);
    }

    #[test]
    fn ar_loop_emits_inrange_frames_and_is_deterministic() {
        let dec = tiny_decoder();
        let cfg = &dec.cfg;
        let no_cancel = || false;
        let grid = manual_grid(cfg, 3);
        let a = dec
            .run(grid.clone(), 24, 7, &no_cancel, &mut |_| {})
            .unwrap()
            .unwrap();
        // Some clean frames were produced, each `channels` wide and in range.
        assert!(!a.frames.is_empty(), "the AR loop emitted clean frames");
        for f in &a.frames {
            assert_eq!(f.len(), cfg.channels, "every frame carries all codebooks");
            // codebook 0 in [0,1024)-analog (< speech-range width); audio codebooks < speech vocab.
            let width0 = cfg.speech_token_range[1] - cfg.speech_token_range[0];
            assert!(f[0] < width0, "codebook 0 in range");
            for (c, &code) in f.iter().enumerate().skip(1) {
                assert!(code < cfg.speech_vocab_size as u32, "codebook {c} in range");
            }
        }
        // Deterministic: same seed ⇒ byte-identical frames.
        let b = dec
            .run(grid, 24, 7, &no_cancel, &mut |_| {})
            .unwrap()
            .unwrap();
        assert_eq!(a.frames, b.frames, "seeded AR decode is reproducible");
    }

    #[test]
    fn cancel_stops_the_ar_loop_promptly() {
        let dec = tiny_decoder();
        let cfg = &dec.cfg;
        let seen = std::cell::Cell::new(0usize);
        let cancel = || seen.get() >= 2;
        let out = dec
            .run(manual_grid(cfg, 3), 500, 7, &cancel, &mut |_| {
                seen.set(seen.get() + 1);
            })
            .unwrap();
        assert!(
            out.is_none(),
            "a mid-loop cancel returns None (→ typed Canceled)"
        );
        assert!(
            seen.get() <= 3,
            "cancel honored within a step, not after the budget"
        );
    }
}
