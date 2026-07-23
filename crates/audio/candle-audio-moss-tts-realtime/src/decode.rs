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

/// The reference text→audio delay (`MossTTSRealtimeProcessor.delay_tokens_len`, `prefill_max_text`):
/// audio generation begins after the first `DELAY_TOKENS_LEN` text tokens are prefilled, and the
/// remaining text tokens are streamed one per audio frame during the AR loop. So audio frame *t* is
/// conditioned on the text up to ~*t+12* — the hierarchical streaming-text design the model was
/// trained on, not a chat completion over the whole text at once (sc-13570).
pub const DELAY_TOKENS_LEN: usize = 12;

/// The reference assistant-turn opener placed immediately after the system prompt — with **no user
/// turn** between them. `MossTTSRealtimeInference`'s `make_ensemble` builds `system prompt +
/// "<|im_start|>assistant\n"`; the text tokens are then appended (prefill) / streamed (AR loop) into
/// the assistant response, not wrapped in a `<|im_start|>user\n…<|im_end|>` turn (sc-13570).
const ASSISTANT_OPEN: &str = "<|im_start|>assistant\n";

/// The reference voice-clone `context` block (`MossTTSRealtimeProcessor.make_voice_clone_prompt`),
/// split around the `<|audio_pad|>` run that carries the reference-speaker timbre. `T_audio`
/// [`AUDIO_PAD_STR`] tokens are emitted between the prefix and suffix, one per reference-audio frame;
/// the encoded reference codes ride the audio channels of those positions (sc-14149).
const VOICE_CLONE_PREFIX: &str =
    "<|im_start|>context\nThe assistant section should be synthesized using the following voice timbre:";
const VOICE_CLONE_SUFFIX: &str = "<|im_end|>\n";
/// The `<|audio_pad|>` marker (`reference_audio_pad`, 151654) whose positions carry the reference
/// timbre codes on the audio channels.
const AUDIO_PAD_STR: &str = "<|audio_pad|>";

/// The prefill prompt frames plus the text tokens to stream during generation — the two halves of the
/// reference delay-pattern conditioning (sc-13570).
pub struct PromptPlan {
    /// The multi-channel prefill frames: `system prompt + "<|im_start|>assistant\n"` followed by the
    /// first [`DELAY_TOKENS_LEN`] text tokens, with `AUDIO_BOS` on codebook 0 of the **last** text
    /// position (the reference `seg[cur_len - 1, 1] = bos`).
    pub prefill: Vec<Frame>,
    /// The remaining text tokens (positions `DELAY_TOKENS_LEN..`), fed one per audio frame on the text
    /// channel during the AR loop; once exhausted the loop feeds `text_pad` (the reference
    /// `_next_text_tokens`). Empty for a prompt of `≤ DELAY_TOKENS_LEN` tokens.
    pub streamed_text: Vec<u32>,
}

/// Build the reference multi-channel conditioning for a single-turn TTS request (sc-13570), faithful
/// to `MossTTSRealtimeInference._generate_from_ids` and its `_build_prefill_batch` /
/// `_next_text_tokens` streaming helpers:
/// - Channel 0 (text): the system prompt, then `<|im_start|>assistant\n` (**no** user turn), then the
///   text tokens — the first [`DELAY_TOKENS_LEN`] in the prefill and the rest streamed one per frame.
/// - Every audio channel is `AUDIO_CHANNEL_PAD`, except codebook 0 of the **last prefilled text**
///   position, which is `AUDIO_BOS` to kick off audio generation aligned with the text stream.
///
/// **Voice cloning (sc-14149):** when `reference_audio` (the encoded reference clip, `[T_audio][rvq]`
/// codebook rows) is supplied, a `context` timbre block is inserted into the system prompt —
/// `<|im_start|>context\n…voice timbre:{<|audio_pad|>×T_audio}<|im_end|>\n` — and the reference codes
/// are threaded onto the audio channels of exactly those `<|audio_pad|>` (`reference_audio_pad`)
/// positions, in order (the reference `MossTTSRealtimeProcessor.make_ensemble`). The rest is
/// unchanged, so the cloned voice conditions the same delay-pattern text generation.
///
/// The system prompt, assistant opener, and text are tokenized **separately** and concatenated (the
/// reference tokenizes `make_ensemble` and the text independently); this tokenizer adds no special
/// tokens, so a separate encode equals a slice of the joint encode.
pub fn build_prompt_frames(
    tokenizer: &Tokenizer,
    cfg: &MossTtsRealtimeConfig,
    text: &str,
    reference_audio: Option<&[Vec<u32>]>,
) -> Result<PromptPlan, String> {
    let encode = |s: &str, what: &str| {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| format!("tokenize {what}: {e}"))
    };
    // The system prompt, plus the voice-clone timbre `context` block when a reference clip is given.
    let system_text = match reference_audio {
        Some(codes) if !codes.is_empty() => format!(
            "{TTS_SYSTEM_PROMPT}{VOICE_CLONE_PREFIX}{pads}{VOICE_CLONE_SUFFIX}",
            pads = AUDIO_PAD_STR.repeat(codes.len()),
        ),
        _ => TTS_SYSTEM_PROMPT.to_string(),
    };
    let system = encode(&system_text, "system prompt")?;
    let assistant = encode(ASSISTANT_OPEN, "assistant open")?;
    let text_ids = encode(text, "prompt text")?;
    if text_ids.is_empty() {
        return Err("empty prompt after tokenization".into());
    }
    let pad_frame = |id: u32| Frame {
        text: id,
        audio: vec![AUDIO_CHANNEL_PAD; cfg.rvq],
    };
    let prefill_text = text_ids.len().min(DELAY_TOKENS_LEN);
    // System frames, with the reference-audio codes threaded onto the `<|audio_pad|>` positions'
    // audio channels (make_ensemble fills `[audio_pad_start..=audio_pad_end, 1:] = prompt_audio_tokens`).
    let mut prefill: Vec<Frame> = Vec::with_capacity(system.len() + assistant.len() + prefill_text);
    let mut audio_idx = 0usize;
    for &id in &system {
        let mut frame = pad_frame(id);
        if id == cfg.reference_audio_pad {
            let codes = reference_audio.expect("audio_pad emitted only for a reference clip");
            let row = codes.get(audio_idx).ok_or_else(|| {
                format!("voice-clone: audio_pad position {audio_idx} exceeds the reference frames")
            })?;
            if row.len() != cfg.rvq {
                return Err(format!(
                    "voice-clone: reference frame {audio_idx} has {} codes, expected {}",
                    row.len(),
                    cfg.rvq
                ));
            }
            frame.audio = row.clone();
            audio_idx += 1;
        }
        prefill.push(frame);
    }
    if let Some(codes) = reference_audio {
        if audio_idx != codes.len() {
            return Err(format!(
                "voice-clone: {} reference frames but {audio_idx} <|audio_pad|> positions",
                codes.len()
            ));
        }
    }
    // assistant-open, then the first DELAY_TOKENS_LEN text tokens (all in the prefill).
    for &id in assistant.iter().chain(&text_ids[..prefill_text]) {
        prefill.push(pad_frame(id));
    }
    // AUDIO_BOS on codebook 0 of the last prefilled text position (prefill_text ≥ 1 here).
    prefill
        .last_mut()
        .expect("prefill has ≥ 1 text token")
        .audio[0] = AUDIO_BOS;
    Ok(PromptPlan {
        prefill,
        streamed_text: text_ids[prefill_text..].to_vec(),
    })
}

/// Default minimum audio frames before an audio-EOS may terminate the AR loop. **Off (0) by default**
/// — the reference (`MossTTSRealtimeInference`) applies *no* minimum-length floor; it stops at the
/// first codebook-0 EOS. sc-13433 added a 16-frame floor as a band-aid for spurious early EOS, but
/// that was a symptom of the wrong prompt conditioning: the port fed the whole text as a chat turn
/// and then generated audio against `text_pad`, driving certain phrasings to an immediate EOS
/// (silence/babble, sc-13570). With the reference delay-pattern conditioning restored
/// ([`build_prompt_frames`]) the model reaches its natural EOS on its own, so the floor is no longer
/// the fidelity lever and forcing frames past a real EOS only manufactures silence. The
/// `MOSS_TTS_MIN_FRAMES` env override is retained for experiments/sweeps (the loop caps any floor at
/// half the frame budget so a genuinely short clip still terminates).
pub const DEFAULT_MIN_EOS_FRAMES: usize = 0;

/// Resolve the minimum-length EOS-suppression floor: `MOSS_TTS_MIN_FRAMES` if set, else
/// [`DEFAULT_MIN_EOS_FRAMES`] (0 ⇒ disabled, reference parity).
fn min_eos_frames() -> usize {
    std::env::var("MOSS_TTS_MIN_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIN_EOS_FRAMES)
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
    ///
    /// `streamed_text` is the reference delay-pattern text stream ([`PromptPlan::streamed_text`], the
    /// tokens beyond the prefill): the *k*-th generation step (the *k*-th fed-back frame, 1-indexed)
    /// carries `streamed_text[k - 1]` on the text channel, and once the stream is exhausted the loop
    /// feeds `text_pad` (mirroring `MossTTSRealtimeInference._next_text_tokens`). Pass `&[]` for a
    /// prompt that fits entirely in the prefill (`≤ DELAY_TOKENS_LEN` tokens).
    pub fn run(
        &self,
        prompt_frames: Vec<Frame>,
        streamed_text: &[u32],
        max_frames: usize,
        seed: u64,
        cancel: &dyn Fn() -> bool,
        on_frame: &mut dyn FnMut(usize, &[u32]) -> CandleResult<()>,
    ) -> CandleResult<Option<DecodeResult>> {
        let mut out: Vec<RvqFrame> = Vec::new();
        let mut stop = StopReason::Budget;
        // The reference sampling distribution (temp 0.8 / top-k 30 / top-p 0.6 / rep-penalty 1.1),
        // with optional operator overrides from the environment (unset ⇒ exactly the defaults, so the
        // determinism/streaming gates are unchanged). The override drives the sc-13433 CER-vs-temp
        // sweep without a rebuild; see `SamplingParams::from_env_or_default`.
        let params = SamplingParams::from_env_or_default();
        // Optional minimum-length EOS suppression (off by default — reference parity, sc-13570): when
        // `MOSS_TTS_MIN_FRAMES` is set, audio-EOS on codebook 0 is masked for the first `min_frames`
        // steps (capped at half the budget). The reference applies no floor; correct delay-pattern
        // conditioning ([`build_prompt_frames`]) — not this floor — is what keeps prompts faithful.
        let min_frames = min_eos_frames().min(max_frames / 2);
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
                // The previous iteration's emitted frame, fed back with the text channel carrying the
                // next streamed text token (delay-pattern), or text_pad once the text is exhausted.
                let prev = out.last().expect("a prior frame was emitted");
                let text = streamed_text
                    .get(step - 1)
                    .copied()
                    .unwrap_or(self.cfg.text_pad);
                let fed = Frame {
                    text,
                    audio: prev.clone(),
                };
                self.backbone.step(&fed, &mut cache)?
            };
            let suppress_eos = out.len() < min_frames;
            let frame = self
                .local
                .decode_frame(&hidden, &out, &params, &mut rng, suppress_eos)?;
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

    /// Like [`tiny_decoder`] but with a **dominant, per-token-distinct text embedding**
    /// (`embed_tokens.0`): each text id maps to a large, well-separated direction (a ±3 bit-code), so
    /// the text channel drives the backbone sum and the sampled frames are demonstrably sensitive to
    /// *which* text token is fed. The default `tiny_decoder`'s tiny uniform weights are numerically
    /// too flat for the discrete 8-token sampler to resolve the text channel, which would mask the
    /// streaming feed under test.
    fn text_sensitive_decoder() -> Decoder {
        let cfg = tiny_cfg();
        let hidden = cfg.language_config.hidden_size;
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
        for (i, (name, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n = t.shape().elem_count();
            let vals: Vec<f32> = if name == "embed_tokens.0.weight" {
                // Row `id`, col `j`: a ±3 bit-code of `id` — distinct direction per text token.
                (0..n)
                    .map(|k| {
                        let (id, j) = (k / hidden, k % hidden);
                        if (id >> (j % 5)) & 1 == 1 {
                            3.0
                        } else {
                            -3.0
                        }
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 - 0.18) as f32)
                    .collect()
            };
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
            .run(manual_prompt(rvq), &[], 5, 42, &no_cancel, &mut |_, _| {
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
            .run(manual_prompt(rvq), &[], 5, 42, &no_cancel, &mut |_, _| {
                Ok(())
            })
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
            .run(manual_prompt(rvq), &[], 100, 42, &cancel, &mut |_, _| {
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

    /// The streamed text tokens (the delay-pattern beyond the prefill, sc-13570) must actually be
    /// **consumed** on the text channel during the AR loop — not silently ignored. This is the cheap
    /// mutation-discriminator for the streaming path: reverting `run` to feed `text_pad` every step
    /// (the pre-sc-13570 bug) makes the two runs identical and turns this test RED. Real weights are
    /// not needed — different text conditioning changes the summed multi-channel embedding, hence the
    /// backbone hidden state, hence the sampled frames.
    #[test]
    fn streamed_text_is_consumed_by_the_ar_loop() {
        let dec = text_sensitive_decoder();
        let rvq = dec.cfg.rvq;
        let no_cancel = || false;
        // Non-pad text ids (text vocab is 32; text_pad is 6, avoided) fed one per generation step.
        let streamed: [u32; 4] = [4, 5, 7, 8];
        let with_stream = dec
            .run(
                manual_prompt(rvq),
                &streamed,
                5,
                42,
                &no_cancel,
                &mut |_, _| Ok(()),
            )
            .unwrap()
            .unwrap();
        // Empty stream ⇒ every step feeds `text_pad` (the old, buggy behavior).
        let all_pad = dec
            .run(manual_prompt(rvq), &[], 5, 42, &no_cancel, &mut |_, _| {
                Ok(())
            })
            .unwrap()
            .unwrap();
        assert_ne!(
            with_stream.frames, all_pad.frames,
            "streamed text must change the frames — a text_pad-only loop ignores the prompt tail"
        );
        // Same stream + seed ⇒ byte-identical frames (the streaming path is reproducible too).
        let again = dec
            .run(
                manual_prompt(rvq),
                &streamed,
                5,
                42,
                &no_cancel,
                &mut |_, _| Ok(()),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            with_stream.frames, again.frames,
            "seeded AR decode over a streamed prompt is reproducible"
        );
    }
}
