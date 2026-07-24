//! Multi-turn conversational continuation (sc-14151) — MOSS-TTS-Realtime's headline capability, the
//! context-aware voice-agent path where turn *N*'s speech is conditioned on turns `1..N-1`.
//!
//! This assembles the sc-13570 delay-pattern per-turn conditioning ([`crate::decode`]) over the
//! sc-13417 KV-cache backbone ([`crate::backbone`]) into a **single shared per-turn core**
//! ([`advance_turn`]) that both multi-turn shapes drive:
//!
//! - **(A) stateless history-in-request** — [`render_conversation`] walks the whole conversation in
//!   one call, keeping the backbone KV cache warm *internally* across turns (never round-tripping a
//!   generated turn back through PCM), and concatenates the assistant replies.
//! - **(B) stateful session** — the provider's `ConversationSession` (in [`crate::model`]) holds a
//!   [`ConvState`] and drives [`advance_turn`] one turn per `step`, keeping the same warm cache live
//!   across turns (the reference `MossTTSRealtimeStreamingSession`), so a turn does not recompute the
//!   prefix.
//!
//! Because both paths run the identical [`advance_turn`] against the cache — A with the loop unrolled
//! inside one `generate`, B across `step` calls — the two emit **byte-identical** audio for the same
//! conversation + seed (the A≡B equivalence law). Each turn's generation is seeded from
//! `base_seed + turn_ordinal` so a turn is independent of the sampling in prior turns, making the two
//! paths agree regardless of call granularity. The reference layout is faithful: `make_ensemble`
//! (system + optional voice-clone timbre, held constant across the conversation) is prefilled once;
//! each user turn is the reference `make_user_prompt` block (its own speech delay-aligned on the audio
//! channels); each assistant turn is the delay-pattern text prefill + AR generation.

use candle_audio::candle_core::Result as CandleResult;
use tokenizers::Tokenizer;

use crate::backbone::{BackboneCache, Frame};
use crate::config::MossTtsRealtimeConfig;
use crate::decode::{
    assistant_prefill_frames, pad_frame, system_frames, Decoder, RvqFrame, AUDIO_BOS, AUDIO_EOS,
    DELAY_TOKENS_LEN,
};

/// The reference `make_user_prompt` prefix (`MossTTSRealtimeProcessor.make_user_prompt`) — a user
/// turn opens by closing the prior section and starting a `user` block. The trailing
/// `<|im_end|>\n<|im_start|>assistant\n` the reference appends is emitted by the *following* assistant
/// turn's opener ([`ASSISTANT_OPENER_AFTER_TURN`]) instead, so the token stream is identical.
const USER_PREFILL_TEMPLATE: &str = "<|im_end|>\n<|im_start|>user\n";

/// The assistant opener after the system block (turn 0 with no user turn) — the system prompt already
/// ends with `<|im_end|>\n`, so only the assistant tag is added (the single-turn layout).
pub(crate) const ASSISTANT_OPENER_AFTER_SYSTEM: &str = "<|im_start|>assistant\n";
/// The assistant opener after a prior turn (a user turn or a previous assistant turn) — closes that
/// section and opens the assistant reply. After a user turn this reproduces the reference
/// `make_user_prompt` `begin_of_response`.
pub(crate) const ASSISTANT_OPENER_AFTER_TURN: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// Who speaks a prepared turn (the crate-internal mirror of `gen_core::ConversationRole`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One conversation turn, prepared for the engine: its role, text, and — for a **context** turn (a
/// user turn's speech, or a previously-generated assistant turn resumed from elsewhere) — the audio
/// already encoded to RVQ frames (`[T][rvq]`) by the provider. An assistant turn with `audio_codes:
/// None` is the reply to **synthesize**.
#[derive(Debug, Clone)]
pub struct PreparedTurn {
    pub role: Role,
    pub text: String,
    pub audio_codes: Option<Vec<RvqFrame>>,
}

/// The section immediately preceding the turn being processed — selects the assistant opener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preceding {
    System,
    Turn,
}

/// The evolving state threaded through [`advance_turn`]: the warm cache, whether the system block is
/// still to be emitted, the running synthesis-turn ordinal (per-turn seed), and what preceded. Path A
/// ([`render_conversation`]) creates a fresh one per call; path B (the provider's session) holds one
/// across turns — the same state, driven at different call granularities.
pub struct ConvState {
    cache: BackboneCache,
    first: bool,
    synth_ordinal: u64,
    preceding: Preceding,
}

impl ConvState {
    /// A fresh conversation state over a new (cold) cache for `decoder`.
    pub fn new(decoder: &Decoder) -> Self {
        Self {
            cache: decoder.backbone.new_cache(),
            first: true,
            synth_ordinal: 0,
            preceding: Preceding::System,
        }
    }
}

/// Derive a turn's sampling seed from the conversation base seed and the synthesis-turn ordinal, so a
/// turn's generation is independent of the sampling in prior turns — the property that makes the
/// stateless (A) and stateful (B) paths byte-identical regardless of call granularity. A large odd
/// stride keeps successive turns' streams well separated.
fn turn_seed(base_seed: u64, synth_ordinal: u64) -> u64 {
    base_seed.wrapping_add(synth_ordinal.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// The reference `make_user_prompt` user block (sc-14151), WITHOUT the trailing assistant opener (the
/// following assistant turn emits it). Channel 0 = `"<|im_end|>\n<|im_start|>user\n{text}" +
/// "<|text_pad|>"*pad`; the user's own audio `codes` ride the audio channels **delay-aligned** to the
/// text (audio frame *t* conditioned on text up to ~*t*+[`DELAY_TOKENS_LEN`]), with `AUDIO_BOS` just
/// before the audio span and `AUDIO_EOS` just after — the exact indexing of the reference's two
/// branches (`text_len >= delay` vs shorter).
fn user_body_frames(
    tokenizer: &Tokenizer,
    cfg: &MossTtsRealtimeConfig,
    text: &str,
    codes: &[RvqFrame],
) -> Result<Vec<Frame>, String> {
    if codes.is_empty() {
        return Err("multi-turn: a user turn carries no audio".into());
    }
    for (i, row) in codes.iter().enumerate() {
        if row.len() != cfg.rvq {
            return Err(format!(
                "multi-turn: user audio frame {i} has {} codes, expected {}",
                row.len(),
                cfg.rvq
            ));
        }
    }
    let encode = |s: &str, what: &str| {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| format!("tokenize {what}: {e}"))
    };
    let delay = DELAY_TOKENS_LEN;
    let audio_len = codes.len();
    let text_start = encode(USER_PREFILL_TEMPLATE, "user prefill template")?.len();
    let text_len = encode(text, "user text")?.len();
    if text_len == 0 {
        return Err("multi-turn: empty user text".into());
    }

    // Build the channel-0 string (prefix + text + text-pad run) exactly as the reference, then locate
    // the audio span / BOS / EOS positions per its two branches.
    let (ch1_str, audio_at) = if text_len >= delay {
        let pad = audio_len + delay - text_len + 1;
        (
            format!(
                "{USER_PREFILL_TEMPLATE}{text}{pads}",
                pads = "<|text_pad|>".repeat(pad)
            ),
            text_start + delay,
        )
    } else {
        let pad = audio_len + 1;
        (
            format!(
                "{USER_PREFILL_TEMPLATE}{text}{pads}",
                pads = "<|text_pad|>".repeat(pad)
            ),
            // The short-text branch places the audio at the tail; resolved from the true length below.
            usize::MAX,
        )
    };
    let ch1 = encode(&ch1_str, "user body")?;
    let mut frames: Vec<Frame> = ch1.iter().map(|&id| pad_frame(cfg, id)).collect();
    let len = frames.len();
    let audio_at = if audio_at == usize::MAX {
        // Short text: audio occupies the last `audio_len` positions before the final pad.
        len.checked_sub(audio_len + 1)
            .ok_or_else(|| "multi-turn: user body too short for its audio".to_string())?
    } else {
        audio_at
    };
    let eos_at = audio_at + audio_len;
    if eos_at >= len {
        return Err(format!(
            "multi-turn: user audio span [{audio_at}..{eos_at}] does not fit the {len}-token user \
             body (tokenizer merged across a boundary?)"
        ));
    }
    // Thread the user's audio onto the audio channels, then the BOS/EOS markers on codebook 0 just
    // outside the span (the reference `cur[audio_span, 1:] = codes; cur[bos, 1] = BOS; cur[eos, 1] = EOS`).
    for (k, row) in codes.iter().enumerate() {
        frames[audio_at + k].audio = row.clone();
    }
    frames[audio_at - 1].audio[0] = AUDIO_BOS;
    frames[eos_at].audio[0] = AUDIO_EOS;
    Ok(frames)
}

/// The **shared per-turn core** both paths run against `state.cache` (sc-14151). Emits the system
/// block once (turn 0), builds this turn's conditioning frames, and:
/// - a **user** turn (or an assistant turn carrying `audio_codes`) is folded in as context — its
///   frames are prefilled (a user turn) or replayed frame-by-frame (a provided assistant turn) so
///   later turns attend over it; returns its provided audio unchanged;
/// - an **assistant** turn with no audio is **synthesized** (seeded per turn), its generated frames
///   staying in the warm cache as context; returns the generated frames.
///
/// `on_frame` streams each emitted synthesized frame (for the low-latency session path). Returns
/// `Ok(None)` on cancel.
///
/// This is the single entry point the provider's stateful session drives one turn at a time (path B),
/// and [`render_conversation`] drives for a whole conversation (path A) — the same computation.
#[allow(clippy::too_many_arguments)]
pub fn advance_turn(
    decoder: &Decoder,
    tokenizer: &Tokenizer,
    cfg: &MossTtsRealtimeConfig,
    voice_clone: Option<&[RvqFrame]>,
    state: &mut ConvState,
    turn: &PreparedTurn,
    base_seed: u64,
    budget: usize,
    cancel: &dyn Fn() -> bool,
    on_frame: &mut dyn FnMut(usize, &[u32]) -> CandleResult<()>,
) -> Result<Option<Vec<RvqFrame>>, String> {
    if cancel() {
        return Ok(None);
    }
    // The system (+ voice-clone timbre) block is prefilled once, warming the cache for turn 0.
    let mut block: Vec<Frame> = Vec::new();
    if state.first {
        block.extend(system_frames(tokenizer, cfg, voice_clone)?);
        state.first = false;
    }

    match turn.role {
        Role::User => {
            let codes = turn.audio_codes.as_ref().ok_or_else(|| {
                "multi-turn: a user turn must carry its audio (provided context)".to_string()
            })?;
            block.extend(user_body_frames(tokenizer, cfg, &turn.text, codes)?);
            // A user turn is context only: prefill it to warm the cache (no generation).
            decoder
                .backbone
                .prefill(&block, &mut state.cache)
                .map_err(|e| format!("multi-turn: prefill user turn: {e}"))?;
            state.preceding = Preceding::Turn;
            Ok(Some(codes.clone()))
        }
        Role::Assistant => {
            let opener = match state.preceding {
                Preceding::System => ASSISTANT_OPENER_AFTER_SYSTEM,
                Preceding::Turn => ASSISTANT_OPENER_AFTER_TURN,
            };
            let streamed =
                assistant_prefill_frames(tokenizer, cfg, opener, &turn.text, &mut block)?;
            state.preceding = Preceding::Turn;
            match &turn.audio_codes {
                // A provided assistant turn (resuming a conversation): replay its known frames so the
                // cache matches a live session that had generated them.
                Some(codes) => {
                    for (i, row) in codes.iter().enumerate() {
                        if row.len() != cfg.rvq {
                            return Err(format!(
                                "multi-turn: provided assistant frame {i} has {} codes, expected {}",
                                row.len(),
                                cfg.rvq
                            ));
                        }
                    }
                    match decoder
                        .replay_turn(&mut state.cache, &block, &streamed, codes, cancel)
                        .map_err(|e| format!("multi-turn: replay assistant turn: {e}"))?
                    {
                        Some(()) => Ok(Some(codes.clone())),
                        None => Ok(None),
                    }
                }
                // The reply to synthesize — seeded per turn so A and B agree.
                None => {
                    let seed = turn_seed(base_seed, state.synth_ordinal);
                    state.synth_ordinal += 1;
                    let result = decoder
                        .generate_turn(
                            &mut state.cache,
                            &block,
                            &streamed,
                            budget,
                            seed,
                            cancel,
                            on_frame,
                        )
                        .map_err(|e| format!("multi-turn: generate assistant turn: {e}"))?;
                    Ok(result.map(|r| r.frames))
                }
            }
        }
    }
}

/// The generated frames of each **synthesized** assistant turn, in conversation order (context turns
/// contribute nothing). The assistant's side of the conversation.
pub struct ConversationFrames {
    pub turns: Vec<Vec<RvqFrame>>,
}

/// **Path A** (stateless history-in-request): render a whole conversation in one call. A fresh cache
/// is walked through every turn via the shared [`advance_turn`] core, kept warm internally across
/// turns; the synthesized assistant replies are returned per turn. `voice_clone` (the encoded
/// reference clip) is held constant across the conversation. Returns `Ok(None)` on cancel.
#[allow(clippy::too_many_arguments)]
pub fn render_conversation(
    decoder: &Decoder,
    tokenizer: &Tokenizer,
    cfg: &MossTtsRealtimeConfig,
    voice_clone: Option<&[RvqFrame]>,
    turns: &[PreparedTurn],
    base_seed: u64,
    budget: usize,
    cancel: &dyn Fn() -> bool,
    on_frame: &mut dyn FnMut(usize, &[u32]) -> CandleResult<()>,
) -> Result<Option<ConversationFrames>, String> {
    let mut state = ConvState::new(decoder);
    let mut out: Vec<Vec<RvqFrame>> = Vec::new();
    for turn in turns {
        let frames = advance_turn(
            decoder,
            tokenizer,
            cfg,
            voice_clone,
            &mut state,
            turn,
            base_seed,
            budget,
            cancel,
            on_frame,
        )?;
        match frames {
            None => return Ok(None),
            Some(f) => {
                if turn.role == Role::Assistant && turn.audio_codes.is_none() {
                    out.push(f);
                }
            }
        }
    }
    Ok(Some(ConversationFrames { turns: out }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MossTtsRealtimeConfig;
    use crate::decode::AUDIO_CHANNEL_PAD;
    use candle_audio::candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// The tiny config used across the crate's weightless tests (`text_pad = 6`,
    /// `reference_audio_pad = 7`, `rvq = 4`). The backbone's audio embedding vocab is the real
    /// `1027` so the multi-channel markers (`AUDIO_CHANNEL_PAD` 1024 / `AUDIO_BOS` 1025 / `AUDIO_EOS`
    /// 1026) that ride the prompt's audio channels are in range when fed through the backbone; the
    /// local head keeps a small `8`-way vocab, so sampled frames are `0..8` (well below `AUDIO_EOS`,
    /// giving fixed budget-length turns — no early EOS to complicate the equivalence test).
    fn tiny_cfg() -> MossTtsRealtimeConfig {
        MossTtsRealtimeConfig::from_json(
            r#"{
              "architectures": ["MossTTSRealtime"],
              "audio_pad_token": 0, "audio_vocab_size": 1027, "rvq": 4,
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

    /// A synthetic tokenizer for the conversation layout: a Whitespace pre-tokenizer (so text does not
    /// merge across the prefix/pad boundaries, matching the real Qwen tokenizer's behavior the
    /// reference relies on) + a WordLevel model, with the layout special tokens as added tokens at the
    /// tiny config's ids (`<|text_pad|>` = 6, `<|audio_pad|>` = 7) plus `<|im_start|>` / `<|im_end|>`.
    fn conv_tokenizer() -> Tokenizer {
        let json = r#"{
          "version": "1.0", "truncation": null, "padding": null,
          "added_tokens": [
            {"id": 6, "content": "<|text_pad|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 7, "content": "<|audio_pad|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 8, "content": "<|im_start|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true},
            {"id": 9, "content": "<|im_end|>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": null,
          "pre_tokenizer": {"type": "Whitespace"},
          "post_processor": null, "decoder": null,
          "model": {"type": "WordLevel", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "w0": 1, "w1": 2, "w2": 3, "w3": 4, "w4": 5, "user": 10,
              "assistant": 11, "w5": 12, "w6": 13, "w7": 14, "w8": 15, "w9": 16, "w10": 17,
              "w11": 18, "w12": 19, "w13": 20, "w14": 21}}
        }"#;
        json.parse()
            .expect("build the synthetic conversation tokenizer")
    }

    fn distinct_codes(cfg: &MossTtsRealtimeConfig, n: usize) -> Vec<RvqFrame> {
        (0..n)
            .map(|i| (0..cfg.rvq).map(|c| ((i * 3 + c + 1) % 6) as u32).collect())
            .collect()
    }

    /// A [`Decoder`] over deterministically-seeded tiny weights — enough to drive the AR loop through
    /// the conversation engine without real weights (the exact `embed_tokens.*` / `language_model.*` /
    /// `local_transformer.*` paths the real loader uses).
    fn tiny_decoder(cfg: &MossTtsRealtimeConfig) -> Decoder {
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
            cfg: cfg.clone(),
        }
    }

    /// The stateless batch render (path A, [`render_conversation`]) and the stateful stepwise drive
    /// (path B, [`advance_turn`] one turn at a time on a persisted [`ConvState`]) produce
    /// **byte-identical** synthesized frames for the same conversation + seed — the provider-level A≡B
    /// equivalence law, on tiny weights, exercising ALL turn kinds: a **user** context turn (its own
    /// audio delay-aligned by `user_body_frames`), a **generated** assistant turn, and a **provided**
    /// assistant turn folded in by `replay_turn`. A divergence between the two drive granularities, or
    /// a per-turn seed that leaked prior-turn sampling, turns this RED.
    #[test]
    fn render_conversation_equals_stepwise_advance_turn() {
        let cfg = tiny_cfg();
        let dec = tiny_decoder(&cfg);
        let tk = conv_tokenizer();
        let no_cancel = || false;
        let turns = vec![
            PreparedTurn {
                role: Role::User,
                text: "w0 w1".into(),
                audio_codes: Some(distinct_codes(&cfg, 2)),
            },
            PreparedTurn {
                role: Role::Assistant,
                text: "w2 w3".into(),
                audio_codes: None, // generate
            },
            PreparedTurn {
                role: Role::Assistant,
                text: "w4 w5".into(),
                audio_codes: Some(distinct_codes(&cfg, 3)), // provided → replay
            },
            PreparedTurn {
                role: Role::Assistant,
                text: "w6 w7".into(),
                audio_codes: None, // generate, conditioned on all the above
            },
        ];
        let budget = 4;
        let seed = 7;

        // Path A: one batch render.
        let a = render_conversation(
            &dec,
            &tk,
            &cfg,
            None,
            &turns,
            seed,
            budget,
            &no_cancel,
            &mut |_, _| Ok(()),
        )
        .unwrap()
        .unwrap();

        // Path B: drive the same turns one at a time on a persisted state (the session's inner loop).
        let mut state = ConvState::new(&dec);
        let mut b_synth: Vec<Vec<RvqFrame>> = Vec::new();
        for turn in &turns {
            let frames = advance_turn(
                &dec,
                &tk,
                &cfg,
                None,
                &mut state,
                turn,
                seed,
                budget,
                &no_cancel,
                &mut |_, _| Ok(()),
            )
            .unwrap()
            .unwrap();
            if turn.role == Role::Assistant && turn.audio_codes.is_none() {
                b_synth.push(frames);
            }
        }

        assert_eq!(a.turns.len(), 2, "two synthesized assistant turns");
        assert_eq!(
            a.turns, b_synth,
            "A≡B: the batch render must equal the stepwise drive, frame-for-frame"
        );

        // Determinism: the same conversation + seed re-renders identically.
        let a2 = render_conversation(
            &dec,
            &tk,
            &cfg,
            None,
            &turns,
            seed,
            budget,
            &no_cancel,
            &mut |_, _| Ok(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(a.turns, a2.turns, "per-conversation determinism");
    }

    #[test]
    fn user_body_short_text_threads_audio_at_the_tail() {
        // text_len (2) < DELAY_TOKENS_LEN (12): the reference's short-text branch places the audio at
        // the tail, `AUDIO_BOS` just before it and `AUDIO_EOS` at the last position.
        let cfg = tiny_cfg();
        let tk = conv_tokenizer();
        let codes = distinct_codes(&cfg, 3); // audio_len = 3
        let frames = user_body_frames(&tk, &cfg, "w0 w1", &codes).unwrap();
        let len = frames.len();
        let audio_at = len - (codes.len() + 1);
        // The audio codes ride the audio channels of `audio_len` positions, in order.
        for (k, row) in codes.iter().enumerate() {
            assert_eq!(&frames[audio_at + k].audio, row, "audio frame {k} at tail");
        }
        // BOS just before the span, EOS at the very end; nothing else carries a BOS/EOS marker.
        assert_eq!(frames[audio_at - 1].audio[0], AUDIO_BOS);
        assert_eq!(frames[len - 1].audio[0], AUDIO_EOS);
        for (i, f) in frames.iter().enumerate() {
            let in_span = (audio_at..audio_at + codes.len()).contains(&i);
            if !in_span && i != audio_at - 1 && i != len - 1 {
                assert!(
                    f.audio.iter().all(|&c| c == AUDIO_CHANNEL_PAD),
                    "position {i} outside the audio span stays padded"
                );
            }
        }
    }

    #[test]
    fn user_body_long_text_threads_audio_delay_aligned() {
        // text_len (14) >= DELAY_TOKENS_LEN (12): the reference's long-text branch offsets the audio
        // by `delay` from the text start (`text_start + delay`), with BOS at `text_start + delay - 1`.
        let cfg = tiny_cfg();
        let tk = conv_tokenizer();
        let codes = distinct_codes(&cfg, 3);
        let text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13"; // 14 tokens
        let text_start = tk
            .encode(USER_PREFILL_TEMPLATE, false)
            .unwrap()
            .get_ids()
            .len();
        let frames = user_body_frames(&tk, &cfg, text, &codes).unwrap();
        let audio_at = text_start + DELAY_TOKENS_LEN;
        for (k, row) in codes.iter().enumerate() {
            assert_eq!(
                &frames[audio_at + k].audio,
                row,
                "audio frame {k} delay-aligned at text_start+delay"
            );
        }
        assert_eq!(frames[audio_at - 1].audio[0], AUDIO_BOS);
        assert_eq!(frames[audio_at + codes.len()].audio[0], AUDIO_EOS);
    }

    #[test]
    fn user_body_rejects_wrong_code_count_and_empty() {
        let cfg = tiny_cfg();
        let tk = conv_tokenizer();
        // A frame with the wrong codebook count is a typed error, not silent corruption.
        let bad = vec![vec![1u32, 2]]; // 2 codes, cfg.rvq == 4
        assert!(user_body_frames(&tk, &cfg, "w0", &bad).is_err());
        // No audio at all is rejected.
        assert!(user_body_frames(&tk, &cfg, "w0", &[]).is_err());
    }

    #[test]
    fn turn_seed_is_per_turn_and_independent() {
        // Distinct ordinals give distinct seeds (so successive turns sample independently), and the
        // derivation is a pure function of (base, ordinal) — the property that makes A and B agree.
        assert_ne!(turn_seed(42, 0), turn_seed(42, 1));
        assert_ne!(turn_seed(42, 1), turn_seed(42, 2));
        assert_eq!(turn_seed(42, 3), turn_seed(42, 3));
    }
}
