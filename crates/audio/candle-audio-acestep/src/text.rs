//! Prompt / lyric front-end for ACE-Step 1.5 (sc-12842).
//!
//! The reference `AceStepPipeline.encode_prompt` splits conditioning into two token streams:
//!
//! - **Prompt** — the reference `SFT_GEN_PROMPT` document: a structured markdown block carrying an
//!   instruction, the style/genre/mood caption, and a `# Metas` list (`bpm`, `timesignature`,
//!   `keyscale`, `duration`), terminated by `<|endoftext|>`. Encoded through the *full* Qwen3
//!   text encoder (contextual hidden states), truncated to [`MAX_TEXT_LEN`] tokens. See
//!   [`build_prompt`].
//! - **Lyrics** — the reference lyric document (`# Languages` + `# Lyric`), encoded through the
//!   text encoder's **embedding layer only** (token lookup), truncated to [`MAX_LYRIC_LEN`]. The
//!   contextual encoding of those embeddings is done downstream by the condition encoder's lyric
//!   encoder, so this module only supplies the token ids. See [`build_lyrics`].
//!
//! Both templates are load-bearing: the model was trained on them, so absent metadata renders as an
//! explicit `N/A` rather than being omitted, and an instrumental request still emits a non-empty
//! lyric document. Emitting a bare comma-joined caption instead — as this module originally did —
//! is out-of-distribution conditioning and yields incoherent audio.

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

/// The reference `DEFAULT_DIT_INSTRUCTION` — the instruction slot of the SFT prompt template when
/// the caller supplies none. The reference appends a `:` if the instruction lacks one; this
/// constant already ends in `:`.
pub const DEFAULT_DIT_INSTRUCTION: &str =
    "Fill the audio semantic mask based on the given conditions:";

/// Assemble the text prompt actually fed to the encoder.
///
/// This is the reference `SFT_GEN_PROMPT` template verbatim, **not** a comma-joined caption:
///
/// ```text
/// # Instruction
/// {instruction}
///
/// # Caption
/// {prompt}
///
/// # Metas
/// - bpm: {bpm|N/A}
/// - timesignature: {ts|N/A}
/// - keyscale: {keyscale|N/A}
/// - duration: {int(seconds)} seconds
/// <|endoftext|>
/// ```
///
/// The model is trained on this structured markdown document, including the `N/A` placeholders for
/// absent metadata (they are *not* omitted) and the trailing `<|endoftext|>`. Feeding it a bare
/// caption instead is badly out-of-distribution conditioning: the port previously emitted
/// `"<caption>, bpm: 133, language: en"` (23 tokens against the reference's 67), which the DiT
/// faithfully rendered as incoherent audio. Note `vocal_language` does **not** belong here — the
/// reference carries it on the *lyric* stream (see [`build_lyrics`]).
pub fn build_prompt(prompt: &str, meta: &Metadata, seconds: f32) -> String {
    // The reference collapses nothing; only the caption slot carries user text.
    let caption = prompt.trim();
    let bpm_str = match meta.bpm {
        // `str(bpm)` on an int in the reference (`bpm > 0` guards the N/A branch).
        Some(bpm) if bpm > 0.0 => format!("{}", bpm as i64),
        _ => "N/A".to_string(),
    };
    let ts_str = meta
        .time_signature
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("N/A");
    let ks_str = meta
        .key
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("N/A");
    // `f"{int(audio_duration)} seconds"`, falling back to 30 for a non-positive duration.
    let dur_str = if seconds > 0.0 {
        format!("{} seconds", seconds as i64)
    } else {
        "30 seconds".to_string()
    };
    let metas =
        format!("- bpm: {bpm_str}\n- timesignature: {ts_str}\n- keyscale: {ks_str}\n- duration: {dur_str}\n");
    format!("# Instruction\n{DEFAULT_DIT_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n")
}

/// Assemble the lyric stream, which the reference encodes on **every** call — including a purely
/// instrumental request, where the template still contributes ~11 tokens of conditioning:
///
/// ```text
/// # Languages
/// {vocal_language}
///
/// # Lyric
/// {lyrics}<|endoftext|>
/// ```
///
/// The port previously skipped the lyric stream entirely when `lyrics` was empty, dropping those
/// rows (and the language signal) from the condition encoder's packed context.
pub fn build_lyrics(lyrics: &str, vocal_language: Option<&str>) -> String {
    let lang = vocal_language
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("en");
    format!(
        "# Languages\n{lang}\n\n# Lyric\n{}<|endoftext|>",
        lyrics.trim()
    )
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
    fn prompt_matches_the_reference_sft_template() {
        let meta = Metadata {
            bpm: Some(128.0),
            key: Some("C minor".into()),
            time_signature: Some("4/4".into()),
            vocal_language: Some("en".into()),
        };
        // The reference `SFT_GEN_PROMPT` document, verbatim. vocal_language is NOT here — it rides
        // the lyric stream.
        let expected = concat!(
            "# Instruction\n",
            "Fill the audio semantic mask based on the given conditions:\n",
            "\n",
            "# Caption\n",
            "upbeat electronic dance track\n",
            "\n",
            "# Metas\n",
            "- bpm: 128\n",
            "- timesignature: 4/4\n",
            "- keyscale: C minor\n",
            "- duration: 12 seconds\n",
            "<|endoftext|>\n",
        );
        assert_eq!(
            build_prompt("upbeat electronic dance track", &meta, 12.0),
            expected
        );
    }

    #[test]
    fn absent_metadata_renders_na_not_omitted() {
        // The reference emits explicit `N/A` placeholders; omitting the lines would change the
        // token stream the model was trained on.
        let p = build_prompt("ambient pad", &Metadata::default(), 30.0);
        assert!(p.contains("- bpm: N/A"), "{p}");
        assert!(p.contains("- timesignature: N/A"), "{p}");
        assert!(p.contains("- keyscale: N/A"), "{p}");
        assert!(p.contains("- duration: 30 seconds"), "{p}");
    }

    #[test]
    fn non_positive_duration_falls_back_to_thirty_seconds() {
        assert!(build_prompt("x", &Metadata::default(), 0.0).contains("- duration: 30 seconds"));
    }

    #[test]
    fn lyrics_are_templated_even_when_empty() {
        // An instrumental request still contributes real conditioning rows.
        assert_eq!(
            build_lyrics("", Some("en")),
            "# Languages\nen\n\n# Lyric\n<|endoftext|>"
        );
        assert_eq!(
            build_lyrics("[verse]\nhello", Some("zh")),
            "# Languages\nzh\n\n# Lyric\n[verse]\nhello<|endoftext|>"
        );
        // A missing language defaults to the reference's "en".
        assert!(build_lyrics("", None).starts_with("# Languages\nen"));
    }
}
