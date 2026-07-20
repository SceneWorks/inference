//! The autoregressive decode loop (sc-13334): text → RVQ speech-token frames.
//!
//! This assembles the two ported stacks into the streaming AR loop of the reference
//! `MossTTSRealtimeInference` (`prefill` + `step`): build the multi-channel prompt, run the Qwen3
//! backbone to a last hidden state, decode one 16-codebook RVQ frame with the local/depth
//! transformer, feed that frame back as the next position's audio channels (text channel = text
//! pad), and repeat until the codebook-0 token is the audio-EOS or a frame budget is hit. Each
//! iteration emits one RVQ frame; a consumer decodes a block of frames into a block of PCM (the
//! streaming chunk). Cancellation is consulted at every frame so the loop stops promptly.
//!
//! **Codec input.** These frames are the input to the MOSS-Audio-Tokenizer codec ([`crate::codec`],
//! RVQ tokens → 24 kHz waveform); [`crate::model`] wires the two together for real streaming TTS.

use candle_audio::candle_core::Result as CandleResult;
use tokenizers::Tokenizer;

use crate::backbone::{Backbone, Frame};
use crate::config::MossTtsRealtimeConfig;
use crate::local::LocalTransformer;
use crate::sampling::{Rng, SamplingParams};

/// In-codebook padding id (also the multi-channel "no audio here" fill).
pub const AUDIO_CHANNEL_PAD: u32 = 1024;
/// Audio begin-of-stream marker placed on codebook 0 at the last prompt position.
pub const AUDIO_BOS: u32 = 1025;
/// Audio end-of-stream marker: a decoded frame whose codebook-0 token equals this ends the stream.
pub const AUDIO_EOS: u32 = 1026;

/// One decoded audio frame: `rvq` RVQ codebook token ids (length `config.rvq`).
pub type RvqFrame = Vec<u32>;

/// Build the multi-channel prompt frames for a single-turn TTS request. Column 0 (text) carries the
/// system prompt + the user turn + the assistant open; every audio channel is `AUDIO_CHANNEL_PAD`,
/// except codebook 0 of the final position which is `AUDIO_BOS` to kick off audio generation —
/// mirroring `MossTTSRealtimeInference.prefill`.
pub fn build_prompt_frames(
    tokenizer: &Tokenizer,
    cfg: &MossTtsRealtimeConfig,
    text: &str,
) -> Result<Vec<Frame>, String> {
    let prompt = format!(
        "{system}<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n",
        system = TTS_SYSTEM_PROMPT,
        text = text,
    );
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenize prompt: {e}"))?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return Err("empty prompt after tokenization".into());
    }
    let mut frames: Vec<Frame> = ids
        .iter()
        .map(|&id| Frame {
            text: id,
            audio: vec![AUDIO_CHANNEL_PAD; cfg.rvq],
        })
        .collect();
    // Kick off audio: codebook 0 of the final prompt position is the audio BOS.
    if let Some(last) = frames.last_mut() {
        last.audio[0] = AUDIO_BOS;
    }
    Ok(frames)
}

/// The reference TTS system prompt (`MossTTSRealtimeProcessor` default).
pub const TTS_SYSTEM_PROMPT: &str = "<|im_start|>system\n\
You are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \n\
You possess natural language understanding, emotional modeling, and multi-style speech generation \
capabilities, allowing you to generate the corresponding speech based on the text given in the \
assistant.<|im_end|>\n";

/// The assembled AR decoder (backbone + local/depth transformer + config).
pub struct Decoder {
    pub backbone: Backbone,
    pub local: LocalTransformer,
    pub cfg: MossTtsRealtimeConfig,
}

/// Why an AR run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A frame's codebook-0 token was the audio-EOS.
    Eos,
    /// The frame budget was reached first.
    Budget,
}

/// The result of an AR run: the emitted RVQ frames and why it stopped.
pub struct DecodeResult {
    pub frames: Vec<RvqFrame>,
    pub stop: StopReason,
}

impl Decoder {
    /// Run the AR loop over `prompt_frames` for at most `max_frames`, consulting `cancel` before each
    /// frame and invoking `on_frame(step, &frame)` for each **emitted** (non-EOS) RVQ frame — the
    /// hook the streaming path uses to decode the codec and emit chunks block-wise *while* the loop
    /// keeps running. Returns `Ok(None)` if cancelled. The emitted frames exclude the terminal EOS
    /// frame. `seed` seeds the token sampler deterministically (same seed ⇒ same frames), so a caller
    /// reproduces the run exactly. `on_frame` is fallible so a consumer (e.g. the codec decode) can
    /// abort the AR loop; it stays the single AR driver.
    pub fn run(
        &self,
        prompt_frames: Vec<Frame>,
        max_frames: usize,
        seed: u64,
        cancel: &dyn Fn() -> bool,
        on_frame: &mut dyn FnMut(usize, &[u32]) -> CandleResult<()>,
    ) -> CandleResult<Option<DecodeResult>> {
        let mut out: Vec<RvqFrame> = Vec::new();
        let mut stop = StopReason::Budget;
        let params = SamplingParams::default();
        let mut rng = Rng::seed(seed);
        // KV-cache AR (sc-13417): prefill the prompt once, then feed each emitted frame back as a
        // single-token step, so per-frame backbone cost is O(1) amortized instead of O(seq). The
        // cache produces byte-identical hidden states to the old full-recompute path (proven in
        // `backbone::tests::kv_cache_is_byte_identical_to_full_recompute`), so the sampled frames —
        // and the reproducibility/streaming determinism gates — are unchanged.
        let mut cache = self.backbone.new_cache();
        let mut prefilled = false;
        for step in 0..max_frames {
            if cancel() {
                return Ok(None);
            }
            // Advance the backbone by exactly the positions the recompute path would have: the whole
            // prompt on the first iteration, then one fed-back frame per iteration thereafter.
            let hidden = if !prefilled {
                prefilled = true;
                self.backbone.prefill(&prompt_frames, &mut cache)?
            } else {
                // The previous iteration's emitted frame, fed back with the text channel = text pad.
                let prev = out.last().expect("a prior frame was emitted");
                let fed = Frame {
                    text: self.cfg.text_pad,
                    audio: prev.clone(),
                };
                self.backbone.step(&fed, &mut cache)?
            };
            let frame = self.local.decode_frame(&hidden, &out, &params, &mut rng)?;
            if frame.first().copied() == Some(AUDIO_EOS) {
                stop = StopReason::Eos;
                break;
            }
            // Hand the just-emitted frame to the consumer (streaming codec decode) before the next
            // iteration feeds it back — so a block of frames can be decoded and streamed as we advance.
            on_frame(step, &frame)?;
            out.push(frame);
        }
        Ok(Some(DecodeResult { frames: out, stop }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// A tiny but structurally faithful config (real block shapes, small dims) — no real weights.
    fn tiny_cfg() -> MossTtsRealtimeConfig {
        MossTtsRealtimeConfig::from_json(
            r#"{
              "architectures": ["MossTTSRealtime"],
              "audio_pad_token": 0, "audio_vocab_size": 8, "rvq": 4,
              "text_pad": 6, "reference_audio_pad": 7,
              "language_config": {
                "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "attention_bias": false, "bos_token_id": 1, "eos_token_id": 2
              },
              "local_config": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "attention_bias": false,
                "rvq": 4, "audio_vocab_size": 8, "audio_pad_token": 0
              }
            }"#,
        )
        .unwrap()
    }

    /// Build a decoder over a deterministically-seeded VarMap (exercises the exact weight paths the
    /// real loader uses: `embed_tokens.N`, `language_model.*`, `local_transformer.*`).
    fn tiny_decoder() -> Decoder {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let backbone = crate::backbone::Backbone::new(
            &cfg.language_config,
            cfg.rvq,
            cfg.audio_vocab_size,
            vb.clone(),
        )
        .unwrap();
        let local = crate::local::LocalTransformer::new(&cfg.local_config, vb.clone()).unwrap();
        // Give the zero-init VarMap deterministic non-trivial values.
        for (i, (_, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n = t.shape().elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 - 0.18) as f32)
                .collect();
            var.set(&Tensor::from_vec(vals, t.shape(), &Device::Cpu).unwrap())
                .unwrap();
        }
        Decoder {
            backbone,
            local,
            cfg,
        }
    }

    fn manual_prompt(rvq: usize) -> Vec<Frame> {
        // Small in-range ids (avoid the real BOS/EOS constants, which exceed the tiny audio vocab).
        (0..3)
            .map(|k| Frame {
                text: k as u32 + 1,
                audio: vec![0u32; rvq],
            })
            .collect()
    }

    #[test]
    fn ar_loop_emits_inrange_rvq_frames_and_is_deterministic() {
        let dec = tiny_decoder();
        let rvq = dec.cfg.rvq;
        let vocab = dec.cfg.audio_vocab_size as u32;
        let no_cancel = || false;
        let mut steps_a = 0usize;
        let a = dec
            .run(manual_prompt(rvq), 5, 42, &no_cancel, &mut |_, _| {
                steps_a += 1;
                Ok(())
            })
            .unwrap()
            .unwrap();
        // Ran to the frame budget (tiny tokens never hit the real EOS), emitting one frame per step.
        assert_eq!(a.stop, StopReason::Budget);
        assert_eq!(a.frames.len(), 5);
        assert_eq!(steps_a, 5, "one progress tick per frame");
        for frame in &a.frames {
            assert_eq!(frame.len(), rvq, "every frame carries rvq codebook tokens");
            assert!(
                frame.iter().all(|&t| t < vocab),
                "tokens are in the codebook vocabulary"
            );
        }
        // Determinism: greedy decode ⇒ byte-identical frames on a re-run.
        let b = dec
            .run(manual_prompt(rvq), 5, 42, &no_cancel, &mut |_, _| Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(a.frames, b.frames, "greedy AR decode is reproducible");
    }

    #[test]
    fn cancel_stops_the_ar_loop_promptly() {
        let dec = tiny_decoder();
        let rvq = dec.cfg.rvq;
        // Cancel trips after the 2nd frame; the loop must stop before the budget.
        let seen = std::cell::Cell::new(0usize);
        let cancel = || seen.get() >= 2;
        let out = dec
            .run(manual_prompt(rvq), 100, 42, &cancel, &mut |_, _| {
                seen.set(seen.get() + 1);
                Ok(())
            })
            .unwrap();
        assert!(
            out.is_none(),
            "a mid-loop cancel returns None (→ typed Canceled)"
        );
        assert!(
            seen.get() <= 3,
            "cancel is honored within a frame, not after the 100-frame budget"
        );
    }
}
