//! Host-side tokenizer: text ↔ token ids.
//!
//! A thin wrapper over the Hugging Face `tokenizers` crate (the same Rust core `transformers`
//! wraps), so a model's `tokenizer.json` reproduces its exact token ids. This is host policy and
//! lives in `core-llm` so it is shared by every backend; backends consume ids only.

use std::collections::HashSet;
use std::path::Path;

use serde_json::{json, Value};

use crate::constraint::ConstraintDecodeTable;
use crate::error::{Error, Result};

/// A loaded tokenizer.
#[derive(Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl Tokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| Error::Load(format!("tokenizer {}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Load from an in-memory `tokenizer.json` string.
    pub fn from_json(json: &str) -> Result<Self> {
        let inner: tokenizers::Tokenizer =
            serde_json::from_str(json).map_err(|e| Error::Load(format!("tokenizer json: {e}")))?;
        Ok(Self { inner })
    }

    /// Build a native Hugging Face byte-level BPE tokenizer from a stock snapshot that ships
    /// `vocab.json`, `merges.txt`, and `tokenizer_config.json`, but no serialized
    /// `tokenizer.json`.
    ///
    /// This is deliberately a host-side reconstruction of the documented tokenizer assets, not a
    /// call into Python/Transformers. Added-token flags come from `tokenizer_config.json`, so EOS
    /// and ordinary added vocabulary retain their distinct decode behavior.
    pub fn from_hf_byte_level_bpe(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
        tokenizer_config_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let vocab_path = vocab_path.as_ref();
        let merges_path = merges_path.as_ref();
        let tokenizer_config_path = tokenizer_config_path.as_ref();
        let vocab: Value = serde_json::from_str(&std::fs::read_to_string(vocab_path)?)
            .map_err(|error| Error::Load(format!("BPE vocab {}: {error}", vocab_path.display())))?;
        if !vocab.is_object() {
            return Err(Error::Load(format!(
                "BPE vocab {} is not a JSON object",
                vocab_path.display()
            )));
        }
        let merges = std::fs::read_to_string(merges_path)?;
        let merges: Vec<Value> = merges
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| Value::String(line.to_owned()))
            .collect();
        let tokenizer_config: Value = serde_json::from_str(&std::fs::read_to_string(
            tokenizer_config_path,
        )?)
        .map_err(|error| {
            Error::Load(format!(
                "BPE tokenizer config {}: {error}",
                tokenizer_config_path.display()
            ))
        })?;
        let mut added_tokens = Vec::new();
        if let Some(tokens) = tokenizer_config.get("added_tokens_decoder") {
            let tokens = tokens.as_object().ok_or_else(|| {
                Error::Load(format!(
                    "BPE tokenizer config {}: added_tokens_decoder is not an object",
                    tokenizer_config_path.display()
                ))
            })?;
            for (id, token) in tokens {
                let id = id.parse::<u64>().map_err(|_| {
                    Error::Load(format!(
                        "BPE tokenizer config {}: non-numeric added-token id {id:?}",
                        tokenizer_config_path.display()
                    ))
                })?;
                let mut token = token.as_object().cloned().ok_or_else(|| {
                    Error::Load(format!(
                        "BPE tokenizer config {}: added token {id} is not an object",
                        tokenizer_config_path.display()
                    ))
                })?;
                token.insert("id".into(), Value::from(id));
                added_tokens.push(Value::Object(token));
            }
        }
        let document = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true,
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true,
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges,
            },
        });
        Self::from_json(&document.to_string())
    }

    /// Encode `text` to token ids. `add_special_tokens` controls auto BOS/EOS per the tokenizer's
    /// post-processor; special-token *strings* already present in `text` always map regardless.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Msg(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Msg(format!("decode: {e}")))
    }

    /// Total vocabulary size (including added tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Build the per-vocab decode table for constrained decoding: the literal text of each token id
    /// (empty for special ids), plus the special-id set. Run once and cache — this decodes
    /// every id in the vocabulary. Delegates to [`build_constraint_decode_table`], the single
    /// decode-table policy in the workspace.
    pub fn constraint_decode_table(&self) -> ConstraintDecodeTable {
        build_constraint_decode_table(&self.inner)
    }
}

/// Build a [`ConstraintDecodeTable`] from a raw HF [`tokenizers::Tokenizer`].
///
/// This is **the single decode-table special-token policy for the workspace** (sc-12467): only
/// added tokens flagged `special == true` (BOS/EOS/turn markers) enter `special` and get an empty
/// `pieces[id]`. Added tokens that are *not* special (ordinary vocabulary words added post-hoc)
/// are regular content — they decode to their literal text and may appear in constrained JSON
/// output like any other token. `sceneworks-gen-core`'s `TextTokenizer::constraint_decode_table`
/// delegates here; its former private copy marked *all* added tokens special (and blanked their
/// pieces), which over-masked valid content — external consumers of that crate now see this
/// (more correct) policy.
///
/// Public-dependency note: the parameter type ties this signature to the workspace's pinned
/// `tokenizers` 0.21 line (the deliberate 0.21/0.22 split is enforced by
/// `scripts/check-workspace.py`); bumping that pin is a semver-visible change here.
pub fn build_constraint_decode_table(inner: &tokenizers::Tokenizer) -> ConstraintDecodeTable {
    let vocab = inner.get_vocab_size(true) as u32;
    let special: HashSet<u32> = inner
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, tok)| tok.special)
        .map(|(id, _)| id)
        .collect();

    let pieces = (0..vocab)
        .map(|id| {
            if special.contains(&id) {
                String::new()
            } else {
                // Partial-UTF-8 byte tokens decode to U+FFFD; acceptable inside JSON strings.
                inner.decode(&[id], false).unwrap_or_default()
            }
        })
        .collect();

    ConstraintDecodeTable { pieces, special }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // A minimal whitespace-split WordLevel tokenizer.json — no model file needed.
    const TINY_JSON: &str = r#"{
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": { "<unk>": 0, "hello": 1, "world": 2, "foo": 3 },
            "unk_token": "<unk>"
        }
    }"#;

    fn tiny() -> Tokenizer {
        Tokenizer::from_json(TINY_JSON).unwrap()
    }

    #[test]
    fn encode_decode_round_trip() {
        let t = tiny();
        let ids = t.encode("hello world", false).unwrap();
        assert_eq!(ids, vec![1, 2]);
        let text = t.decode(&ids, false).unwrap();
        assert!(text.contains("hello") && text.contains("world"));
    }

    #[test]
    fn constraint_table_covers_vocab() {
        let t = tiny();
        let table = t.constraint_decode_table();
        assert_eq!(table.pieces.len(), t.vocab_size());
        assert_eq!(table.pieces[1], "hello");
    }

    #[test]
    fn reconstructs_stock_byte_level_bpe_without_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocab.json");
        let merges = dir.path().join("merges.txt");
        let config = dir.path().join("tokenizer_config.json");
        fs::write(&vocab, r#"{"<|endoftext|>":0,"a":1}"#).unwrap();
        fs::write(&merges, "#version: 0.2\n").unwrap();
        fs::write(
            &config,
            r#"{"added_tokens_decoder":{"0":{"content":"<|endoftext|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true}}}"#,
        )
        .unwrap();
        let tokenizer = Tokenizer::from_hf_byte_level_bpe(vocab, merges, config).unwrap();
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.encode("<|endoftext|>", false).unwrap(), vec![0]);
        assert_eq!(tokenizer.decode(&[0], true).unwrap(), "");
    }

    // Like TINY_JSON but with added tokens: id 4 is a special added token (an EOS marker), id 5 is
    // a NON-special added token (an ordinary word added to the vocab post-hoc).
    const ADDED_TOKENS_JSON: &str = r#"{
        "version": "1.0",
        "added_tokens": [
            { "id": 4, "content": "<eos>", "single_word": false, "lstrip": false,
              "rstrip": false, "normalized": false, "special": true },
            { "id": 5, "content": "wombat", "single_word": false, "lstrip": false,
              "rstrip": false, "normalized": false, "special": false }
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": { "<unk>": 0, "hello": 1, "world": 2, "foo": 3 },
            "unk_token": "<unk>"
        }
    }"#;

    /// Pins the workspace decode-table special-token policy (sc-12467): only `special == true`
    /// added tokens are masked; non-special added tokens keep their literal text and are usable
    /// as JSON content. (The pre-sc-12467 gen-core copy marked ALL added tokens special, which
    /// this fixture would have caught.)
    #[test]
    fn constraint_table_masks_only_special_added_tokens() {
        let t = Tokenizer::from_json(ADDED_TOKENS_JSON).unwrap();
        let table = t.constraint_decode_table();
        assert_eq!(table.pieces.len(), 6, "vocab includes both added tokens");

        // Special added token: in the special set, no decode text.
        assert!(table.special.contains(&4), "<eos> (special=true) is masked");
        assert_eq!(table.pieces[4], "", "special tokens carry no JSON content");

        // Non-special added token: NOT masked, decodes to its literal text.
        assert!(
            !table.special.contains(&5),
            "non-special added token must not be masked as special"
        );
        assert_eq!(
            table.pieces[5], "wombat",
            "non-special added token keeps its decoded text"
        );

        // Ordinary vocab tokens are untouched.
        assert!(!table.special.contains(&1));
        assert_eq!(table.pieces[1], "hello");
    }
}
