//! Prompt / lyric front-end for ACE-Step 1.5 (sc-12842).
//!
//! The reference `AceStepPipeline.encode_prompt` splits conditioning into two token streams:
//!
//! - **Prompt** — the style/genre/instrument/mood/tempo caption, optionally prefixed with an
//!   auto-generated task instruction and suffixed with musical metadata (`bpm`, `keyscale`,
//!   `timesignature`, `vocal_language`). It is encoded through the *full* Qwen3-Embedding-0.6B
//!   text encoder (contextual hidden states), truncated to [`MAX_TEXT_LEN`] tokens.
//! - **Lyrics** — structured with `[verse]` / `[chorus]` / … tags; encoded through the text
//!   encoder's **embedding layer only** (token lookup), truncated to [`MAX_LYRIC_LEN`]. The
//!   contextual encoding of those embeddings is done downstream by the condition encoder's lyric
//!   encoder, so this module only supplies the token ids.
//!
//! The metadata weave below is the reference's textual conditioning convention (metadata that is
//! `None` is estimated by the model, so an absent field is simply omitted from the prompt).

use candle_audio::{AudioError, Result};
use tokenizers::Tokenizer;

/// Max prompt tokens (reference `max_text_length`).
pub const MAX_TEXT_LEN: usize = 256;

/// Max lyric tokens (reference `max_lyric_length`).
pub const MAX_LYRIC_LEN: usize = 2048;

/// Optional musical metadata woven into the prompt (each `None` field is left to the model).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub bpm: Option<f32>,
    pub key: Option<String>,
    pub time_signature: Option<String>,
    pub vocal_language: Option<String>,
}

/// Assemble the text prompt actually fed to the encoder: the style caption plus any supplied
/// musical metadata, in a stable order. Whitespace is collapsed and the result trimmed.
pub fn build_prompt(prompt: &str, meta: &Metadata) -> String {
    let mut parts: Vec<String> = Vec::new();
    let base = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if !base.is_empty() {
        parts.push(base);
    }
    if let Some(bpm) = meta.bpm {
        // Whole-number BPMs render without a trailing ".0" (the reference passes an int).
        if (bpm.fract()).abs() < f32::EPSILON {
            parts.push(format!("bpm: {}", bpm as i64));
        } else {
            parts.push(format!("bpm: {bpm}"));
        }
    }
    if let Some(key) = &meta.key {
        let key = key.trim();
        if !key.is_empty() {
            parts.push(format!("key: {key}"));
        }
    }
    if let Some(ts) = &meta.time_signature {
        let ts = ts.trim();
        if !ts.is_empty() {
            parts.push(format!("time signature: {ts}"));
        }
    }
    if let Some(lang) = &meta.vocal_language {
        let lang = lang.trim();
        if !lang.is_empty() {
            parts.push(format!("language: {lang}"));
        }
    }
    parts.join(", ")
}

/// Tokenize the assembled prompt to at most [`MAX_TEXT_LEN`] ids (reference truncation). An empty
/// prompt yields no ids, which the pipeline maps to the all-zero prompt context.
pub fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    tokenize(tokenizer, prompt, MAX_TEXT_LEN)
}

/// Tokenize lyrics to at most [`MAX_LYRIC_LEN`] ids (the token lookup the condition encoder's
/// lyric encoder contextualizes). Empty lyrics yield no ids (instrumental generation).
pub fn tokenize_lyrics(tokenizer: &Tokenizer, lyrics: &str) -> Result<Vec<u32>> {
    tokenize(tokenizer, lyrics, MAX_LYRIC_LEN)
}

fn tokenize(tokenizer: &Tokenizer, text: &str, max_len: usize) -> Result<Vec<u32>> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| AudioError::Msg(format!("acestep tokenize: {e}")))?;
    let mut ids = encoding.get_ids().to_vec();
    ids.truncate(max_len);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_weaves_metadata_in_stable_order() {
        let meta = Metadata {
            bpm: Some(128.0),
            key: Some("C minor".into()),
            time_signature: Some("4".into()),
            vocal_language: Some("en".into()),
        };
        assert_eq!(
            build_prompt("  upbeat   electronic\ndance track ", &meta),
            "upbeat electronic dance track, bpm: 128, key: C minor, time signature: 4, language: en"
        );
    }

    #[test]
    fn absent_metadata_is_omitted() {
        assert_eq!(
            build_prompt("ambient pad", &Metadata::default()),
            "ambient pad"
        );
        let meta = Metadata {
            bpm: Some(90.0),
            ..Default::default()
        };
        assert_eq!(build_prompt("lofi", &meta), "lofi, bpm: 90");
    }
}
