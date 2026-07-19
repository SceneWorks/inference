//! The Whisper autoregressive decode loop (sc-12850): a log-mel spectrogram → encoder audio
//! features → token-by-token text decode, chunked over Whisper's fixed 30 s windows.
//!
//! The transformer stacks themselves are candle's ([`candle_transformers::models::whisper`]); this
//! module owns only the decode *policy* — the special-token prompt (`<|sot|>` + language + task +
//! optional `<|notimestamps|>`), greedy-or-temperature next-token selection honoring the request's
//! sampling knobs, the suppressed-token mask, cooperative cancellation between steps, and the
//! timestamp-token → [`TranscriptSegment`] parse. It is deliberately a faithful, trimmed port of
//! the upstream candle whisper example's `Decoder` (no multi-temperature fallback sweep: this
//! provider honors the caller's single requested temperature).

use candle_audio::candle_core::{self as candle, IndexOp, Tensor};
use candle_audio::gen_core::{TranscribeFinishReason, TranscriptSegment, TranscriptWord};
use candle_audio::{AudioError, Result};
use candle_transformers::models::whisper::{self as whisper, model::Whisper, Config};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::SeedableRng;
use tokenizers::Tokenizer;

/// A resolved decode outcome for the whole clip.
pub struct DecodeOutput {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub tokens: u32,
    pub finish_reason: TranscribeFinishReason,
}

/// The Whisper decoder: borrowed handles to the cached model + tokenizer, the resolved special
/// tokens, and the per-call decode policy. Borrows (rather than owns) the model so the expensive
/// weights stay cached in the transcriber across requests while the cheap per-request suppressed-
/// token mask (which depends on the request's timestamp mode) is rebuilt each call.
pub struct WhisperDecoder<'a> {
    model: &'a mut Whisper,
    tokenizer: &'a Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_timestamps_token: u32,
    device: candle::Device,
}

/// Look up a token id by its literal, erroring (not panicking) when absent.
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| AudioError::Msg(format!("whisper: tokenizer has no id for {token:?}")))
}

impl<'a> WhisperDecoder<'a> {
    /// Build the decoder over borrowed model + tokenizer handles. `timestamps` fixes the
    /// suppressed-token mask (the `<|notimestamps|>` token is suppressed while decoding timestamps).
    pub fn new(
        model: &'a mut Whisper,
        tokenizer: &'a Tokenizer,
        timestamps: bool,
        device: candle::Device,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
        let suppress: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                    || (timestamps && i == no_timestamps_token)
                {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let suppress_tokens =
            Tensor::new(suppress.as_slice(), &device).map_err(AudioError::from)?;
        Ok(Self {
            sot_token: token_id(tokenizer, whisper::SOT_TOKEN)?,
            transcribe_token: token_id(tokenizer, whisper::TRANSCRIBE_TOKEN)?,
            translate_token: token_id(tokenizer, whisper::TRANSLATE_TOKEN)?,
            eot_token: token_id(tokenizer, whisper::EOT_TOKEN)?,
            no_timestamps_token,
            suppress_tokens,
            model,
            tokenizer,
            device,
        })
    }

    pub fn config(&self) -> &Config {
        &self.model.config
    }

    /// Auto-detect the spoken language, returning its `<|lang|>` token id (multilingual models
    /// only). Mirrors the upstream `multilingual::detect_language`.
    pub fn detect_language(&mut self, mel: &Tensor) -> Result<u32> {
        let (_b, _m, seq_len) = mel.dims3().map_err(AudioError::from)?;
        let n = usize::min(seq_len, self.model.config.max_source_positions);
        let mel = mel.narrow(2, 0, n).map_err(AudioError::from)?;
        let language_token_ids = LANGUAGES
            .iter()
            .map(|(t, _)| token_id(self.tokenizer, &format!("<|{t}|>")))
            .collect::<Result<Vec<_>>>()?;
        let audio_features = self
            .model
            .encoder
            .forward(&mel, true)
            .map_err(AudioError::from)?;
        let tokens = Tensor::new(&[[self.sot_token]], &self.device).map_err(AudioError::from)?;
        let ys = self
            .model
            .decoder
            .forward(&tokens, &audio_features, true)
            .map_err(AudioError::from)?;
        let logits = self
            .model
            .decoder
            .final_linear(&ys.i(..1).map_err(AudioError::from)?)
            .map_err(AudioError::from)?
            .i(0)
            .map_err(AudioError::from)?
            .i(0)
            .map_err(AudioError::from)?;
        let ids =
            Tensor::new(language_token_ids.as_slice(), &self.device).map_err(AudioError::from)?;
        let logits = logits.index_select(&ids, 0).map_err(AudioError::from)?;
        let probs = candle_nn::ops::softmax(&logits, candle::D::Minus1)
            .map_err(AudioError::from)?
            .to_vec1::<f32>()
            .map_err(AudioError::from)?;
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(language_token_ids[best])
    }

    /// Language-code → `<|lang|>` token id, erroring for an unsupported code.
    pub fn language_token(&self, code: &str) -> Result<u32> {
        if !LANGUAGES.iter().any(|(t, _)| *t == code) {
            return Err(AudioError::Msg(format!(
                "whisper: language {code:?} is not a Whisper language code"
            )));
        }
        token_id(self.tokenizer, &format!("<|{code}|>"))
    }

    /// Whether `token` is a timestamp token, and its absolute time in seconds (0.02 s resolution).
    fn timestamp_seconds(&self, token: u32) -> Option<f32> {
        let timestamp_begin = self.no_timestamps_token + 1;
        (token >= timestamp_begin).then(|| (token - timestamp_begin) as f32 * 0.02)
    }

    /// Decode one ≤30 s mel window into its raw token sequence, honoring `temperature`/`seed`/
    /// `max_new_tokens`, the language/task prompt, and cooperative cancellation.
    #[allow(clippy::too_many_arguments)]
    fn decode_window(
        &mut self,
        mel: &Tensor,
        language_token: Option<u32>,
        translate: bool,
        timestamps: bool,
        temperature: f64,
        seed: u64,
        max_new_tokens: u32,
        cancel: &dyn Fn() -> bool,
    ) -> Result<(Vec<u32>, bool)> {
        self.model.reset_kv_cache();
        let audio_features = self
            .model
            .encoder
            .forward(mel, true)
            .map_err(AudioError::from)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = language_token {
            tokens.push(language_token);
        }
        tokens.push(if translate {
            self.translate_token
        } else {
            self.transcribe_token
        });
        if !timestamps {
            tokens.push(self.no_timestamps_token);
        }
        let prompt_len = tokens.len();
        let ceiling = (max_new_tokens as usize).min(self.model.config.max_target_positions);
        let mut hit_eot = false;
        for i in 0..ceiling {
            if cancel() {
                return Err(AudioError::Canceled);
            }
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(AudioError::from)?
                .unsqueeze(0)
                .map_err(AudioError::from)?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)
                .map_err(AudioError::from)?;
            let (_, seq_len, _) = ys.dims3().map_err(AudioError::from)?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..)).map_err(AudioError::from)?)
                .map_err(AudioError::from)?
                .i(0)
                .map_err(AudioError::from)?
                .i(0)
                .map_err(AudioError::from)?;
            let logits = logits
                .broadcast_add(&self.suppress_tokens)
                .map_err(AudioError::from)?;
            let next = if temperature > 0.0 {
                let prs = candle_nn::ops::softmax(
                    &(&logits / temperature).map_err(AudioError::from)?,
                    candle::D::Minus1,
                )
                .map_err(AudioError::from)?
                .to_vec1::<f32>()
                .map_err(AudioError::from)?;
                let distr = WeightedIndex::new(&prs)
                    .map_err(|e| AudioError::Msg(format!("whisper: {e}")))?;
                distr.sample(&mut rng) as u32
            } else {
                let v = logits.to_vec1::<f32>().map_err(AudioError::from)?;
                v.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(self.eot_token)
            };
            tokens.push(next);
            if next == self.eot_token || tokens.len() > self.model.config.max_target_positions {
                hit_eot = next == self.eot_token;
                break;
            }
        }
        // Drop the special prompt prefix; keep the emitted content (including timestamp tokens).
        Ok((tokens[prompt_len..].to_vec(), hit_eot))
    }

    /// Decode the whole clip's mel, chunked over 30 s windows, into text + optional timestamped
    /// segments. `mel` is `[1, n_mels, n_frames]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        mel: &Tensor,
        language_token: Option<u32>,
        translate: bool,
        timestamps: bool,
        temperature: f64,
        seed: u64,
        max_new_tokens: u32,
        cancel: &dyn Fn() -> bool,
    ) -> Result<DecodeOutput> {
        let (_, _, content_frames) = mel.dims3().map_err(AudioError::from)?;
        let mut seek = 0usize;
        let mut all_text = String::new();
        let mut segments = Vec::new();
        let mut total_tokens = 0u32;
        let mut finish = TranscribeFinishReason::StopToken;
        while seek < content_frames {
            let time_offset = (seek * whisper::HOP_LENGTH) as f32 / whisper::SAMPLE_RATE as f32;
            let segment_size = usize::min(content_frames - seek, whisper::N_FRAMES);
            let mel_segment = mel
                .narrow(2, seek, segment_size)
                .map_err(AudioError::from)?;
            let (window_tokens, hit_eot) = self.decode_window(
                &mel_segment,
                language_token,
                translate,
                timestamps,
                temperature,
                seed,
                max_new_tokens,
                cancel,
            )?;
            total_tokens += window_tokens.len() as u32;
            if !hit_eot {
                finish = TranscribeFinishReason::MaxTokens;
            }
            self.append_window(
                &window_tokens,
                time_offset,
                timestamps,
                &mut all_text,
                &mut segments,
            )?;
            seek += segment_size;
        }
        Ok(DecodeOutput {
            text: all_text.trim().to_string(),
            segments,
            tokens: total_tokens,
            finish_reason: finish,
        })
    }

    /// Fold one decoded window's tokens into the running transcript: decode the text tokens, and —
    /// in timestamp mode — split into `[prev_ts, ts]` segments at each timestamp token.
    fn append_window(
        &self,
        window_tokens: &[u32],
        time_offset: f32,
        timestamps: bool,
        all_text: &mut String,
        segments: &mut Vec<TranscriptSegment>,
    ) -> Result<()> {
        if !timestamps {
            let text = self.decode_text(window_tokens)?;
            if !text.trim().is_empty() {
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(text.trim());
            }
            return Ok(());
        }
        let mut pending: Vec<u32> = Vec::new();
        let mut seg_start = time_offset;
        for &token in window_tokens {
            if token == self.sot_token || token == self.eot_token {
                continue;
            }
            match self.timestamp_seconds(token) {
                Some(ts) => {
                    let ts = time_offset + ts;
                    if !pending.is_empty() {
                        let text = self.decode_text(&pending)?;
                        if !text.trim().is_empty() {
                            segments.push(TranscriptSegment {
                                text: text.trim().to_string(),
                                start: seg_start,
                                end: ts,
                                words: Vec::<TranscriptWord>::new(),
                            });
                            if !all_text.is_empty() {
                                all_text.push(' ');
                            }
                            all_text.push_str(text.trim());
                        }
                        pending.clear();
                    }
                    seg_start = ts;
                }
                None => pending.push(token),
            }
        }
        // A trailing run with no closing timestamp (e.g. decode hit max_new_tokens): close it at
        // the window end so no text is silently dropped.
        if !pending.is_empty() {
            let text = self.decode_text(&pending)?;
            if !text.trim().is_empty() {
                segments.push(TranscriptSegment {
                    text: text.trim().to_string(),
                    start: seg_start,
                    end: seg_start,
                    words: Vec::new(),
                });
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(text.trim());
            }
        }
        Ok(())
    }

    /// Decode a run of content tokens to text (special tokens skipped).
    fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| AudioError::Msg(format!("whisper: tokenizer decode failed: {e}")))
    }
}

/// The 99 Whisper language codes (name kept for documentation), from the reference implementation —
/// used for `<|lang|>` token resolution and auto-detection.
pub const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];
