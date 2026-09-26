//! The frozen YuE2 text/ABC BPE (`qwen.tiktoken`) — a native port of upstream
//! `yue2.tokenization_yue2.YuE2TextTokenizer` (sc-22990).
//!
//! This is the **text** tokenizer: it turns the request prompt and ABC scores into ordinary token
//! IDs `0..151643`. It is not the audio codec tokenizer.
//!
//! Behaviour ported exactly:
//!
//! * `qwen.tiktoken` is `base64(token) rank` lines; it must hold exactly
//!   [`ORDINARY_TOKENS`] ordinary tokens (upstream refuses anything else).
//! * The 208 special tokens are `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<R>`, `<S>`, `<X>`,
//!   `<mask>`, `<sep>`, then `<extra_0>`…`<extra_199>` with slots 204/205 replaced by `<abc>` /
//!   `</abc>`, at IDs `151643 + i` ([`special_token`]).
//! * [`Yue2TextTokenizer::encode`] is tiktoken's `encode_ordinary` after Unicode NFC: special-token
//!   *text* in the input is encoded as ordinary text, never as a special ID. Pieces come from
//!   upstream's pre-tokenizer pattern; each piece is looked up whole, then byte-pair merged by
//!   ascending rank (leftmost first among equal ranks), exactly as tiktoken's `CoreBPE` does.
//! * [`Yue2TextTokenizer::decode`] drops IDs outside `0..151851` (as upstream does before calling
//!   tiktoken) and decodes the bytes with U+FFFD replacement of malformed UTF-8
//!   (`errors="replace"`).
//!
//! # The reference stack's Unicode versions
//!
//! Upstream runs NFC through Python 3.12's `unicodedata` (Unicode **15.0**) and its pattern through
//! tiktoken 0.12's regex tables (Unicode **16.0**). This port pins both instead of inheriting
//! whatever the Rust crates ship: NFC is applied only to runs of code points assigned by Unicode
//! 15.0, with every later code point passed through untouched as the non-composing starter Python
//! treats it as; and `\p{L}` / `\p{N}` in the pattern are intersected with `\p{Age=16.0}`. Without
//! this, text using characters added in Unicode 16/17 (for example Todhri or Tulu-Tigalari
//! composites) would tokenize differently from the reference.
//!
//! # Loading
//!
//! [`Yue2TextTokenizer::load`] resolves and verifies the pinned `qwen.tiktoken` through
//! [`crate::snapshot::resolve_component`] immediately before reading it, then re-hashes the exact
//! bytes it parses against the pinned SHA-256 (the crate's "verify at the load boundary" rule).
//! There is no public constructor from arbitrary bytes.

use std::collections::HashMap;
use std::path::PathBuf;

use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::inventory::ComponentId;
use crate::snapshot::{self, AssetError, SnapshotDirs, VerifiedComponent};

/// Ordinary (BPE) tokens in `qwen.tiktoken`; also the first special-token ID (`<|endoftext|>`).
pub const ORDINARY_TOKENS: u32 = 151_643;
/// Special tokens appended after the ordinary vocabulary.
pub const SPECIAL_TOKENS: u32 = 208;
/// Size of the text vocabulary (ordinary + special). IDs at or above it are not text.
pub const TEXT_VOCAB: u32 = ORDINARY_TOKENS + SPECIAL_TOKENS;

/// The snapshot-relative file name of the rank table.
pub const TIKTOKEN_FILE: &str = "qwen.tiktoken";

/// Upstream's pre-tokenizer pattern with `\p{L}` / `\p{N}` pinned to Unicode 16.0 (the tables of
/// the reference tiktoken build). Upstream's literal is
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
const PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n[\p{L}&&\p{Age:16.0}][\p{N}&&\p{Age:16.0}]]?[\p{L}&&\p{Age:16.0}]+",
    r"|[\p{N}&&\p{Age:16.0}]",
    r"| ?[^\s[\p{L}&&\p{Age:16.0}][\p{N}&&\p{Age:16.0}]]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Code points the reference NFC (Python 3.12, Unicode 15.0) does not know.
const AFTER_UNICODE_15: &str = r"[^\p{Age:15.0}]";

/// Every way loading or running the text tokenizer fails.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// The pinned `qwen.tiktoken` failed resolution or verification.
    #[error(transparent)]
    Asset(#[from] AssetError),
    /// A verified component that is not `qwen.tiktoken` was handed to the tokenizer loader.
    #[error("the text tokenizer loads from the qwen.tiktoken component, not `{0}`")]
    WrongComponent(&'static str),
    /// Reading the verified file failed.
    #[error("reading {}: {source}", path.display())]
    Io {
        /// The verified path.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// The bytes read for parsing are not the bytes that were pinned (changed after verification).
    #[error("{}: read SHA-256 {actual}, pinned {expected}; the file changed after verification", path.display())]
    ChangedAfterVerification {
        /// The verified path.
        path: PathBuf,
        /// The pinned SHA-256.
        expected: &'static str,
        /// The SHA-256 of the bytes read.
        actual: String,
    },
    /// A rank-table line is malformed.
    #[error("qwen.tiktoken line {line}: {detail}")]
    Malformed {
        /// 1-based line number.
        line: usize,
        /// What is wrong.
        detail: String,
    },
    /// The rank table does not hold the checkpoint-native vocabulary.
    #[error(
        "expected checkpoint-native qwen.tiktoken ({expected} ordinary tokens with ranks \
         0..{expected}), found {found}"
    )]
    Vocabulary {
        /// Required ordinary-token count.
        expected: u32,
        /// What the table holds.
        found: String,
    },
    /// The pre-tokenizer pattern failed at run time (fancy-regex backtracking limit).
    #[error("pre-tokenizer failed: {0}")]
    Pretokenize(String),
}

/// The text of special token `id`, or `None` when `id` is not a special-token ID.
pub fn special_token(id: u32) -> Option<String> {
    const NAMED: [&str; 8] = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<R>",
        "<S>",
        "<X>",
        "<mask>",
        "<sep>",
    ];
    let index = id.checked_sub(ORDINARY_TOKENS)?;
    Some(match index {
        0..=7 => NAMED[index as usize].to_string(),
        204 => "<abc>".to_string(),
        205 => "</abc>".to_string(),
        8..=207 => format!("<extra_{}>", index - 8),
        _ => return None,
    })
}

/// The native YuE2 text tokenizer. See the [module docs](self).
#[derive(Debug)]
pub struct Yue2TextTokenizer {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: Vec<Vec<u8>>,
    specials: Vec<Vec<u8>>,
    pattern: fancy_regex::Regex,
    after_unicode_15: regex::Regex,
}

impl Yue2TextTokenizer {
    /// Resolve and verify the pinned `qwen.tiktoken` from `dirs`, then load it. Call this
    /// immediately before tokenizing a request; never cache the verification.
    pub fn load(dirs: &SnapshotDirs) -> Result<Self, TokenizerError> {
        let verified = snapshot::resolve_component(ComponentId::QwenTiktoken, dirs)?;
        Self::from_verified(&verified)
    }

    /// Load from a just-verified `qwen.tiktoken` component (for example
    /// `resolve_closure(..)?.get(ComponentId::QwenTiktoken)`). The bytes parsed are re-hashed
    /// against the pinned SHA-256, so a file swapped after verification is refused.
    pub fn from_verified(component: &VerifiedComponent) -> Result<Self, TokenizerError> {
        let entry = component.component();
        if entry.id != ComponentId::QwenTiktoken {
            return Err(TokenizerError::WrongComponent(entry.key));
        }
        let pinned = entry
            .file(TIKTOKEN_FILE)
            .ok_or(TokenizerError::WrongComponent(entry.key))?;
        let path = component
            .path(TIKTOKEN_FILE)
            .ok_or(TokenizerError::WrongComponent(entry.key))?;
        let bytes = std::fs::read(path).map_err(|source| TokenizerError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let actual = hex(&Sha256::digest(&bytes));
        if actual != pinned.sha256 {
            return Err(TokenizerError::ChangedAfterVerification {
                path: path.to_path_buf(),
                expected: pinned.sha256,
                actual,
            });
        }
        Self::from_tiktoken_bytes(&bytes)
    }

    /// Parse checkpoint-native `qwen.tiktoken` bytes (exactly [`ORDINARY_TOKENS`] ranks).
    pub(crate) fn from_tiktoken_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let ranks = parse_ranks(bytes)?;
        if ranks.len() != ORDINARY_TOKENS as usize {
            return Err(TokenizerError::Vocabulary {
                expected: ORDINARY_TOKENS,
                found: format!("{} tokens", ranks.len()),
            });
        }
        Self::from_ranks(ranks)
    }

    /// A tokenizer over a rank table smaller than the checkpoint's (special IDs stay at their
    /// checkpoint positions). Test fixtures only: the committed synthetic table stands in for
    /// `qwen.tiktoken`, whose redistribution the licence policy gates.
    #[cfg(test)]
    pub(crate) fn from_partial_ranks_for_tests(bytes: &[u8]) -> Result<Self, TokenizerError> {
        Self::from_ranks(parse_ranks(bytes)?)
    }

    fn from_ranks(decoder: Vec<Vec<u8>>) -> Result<Self, TokenizerError> {
        let mut encoder = HashMap::with_capacity(decoder.len());
        for (rank, token) in decoder.iter().enumerate() {
            if encoder.insert(token.clone(), rank as u32).is_some() {
                return Err(TokenizerError::Vocabulary {
                    expected: ORDINARY_TOKENS,
                    found: format!("duplicate token at rank {rank}"),
                });
            }
        }
        // Every piece falls back to single bytes, so all 256 must be ranked (tiktoken panics
        // mid-encode otherwise).
        if let Some(byte) = (0..=255u8).find(|b| !encoder.contains_key(&[*b][..])) {
            return Err(TokenizerError::Vocabulary {
                expected: ORDINARY_TOKENS,
                found: format!("no rank for byte 0x{byte:02x}"),
            });
        }
        let specials = (ORDINARY_TOKENS..TEXT_VOCAB)
            .map(|id| {
                special_token(id)
                    .expect("every ID in ORDINARY_TOKENS..TEXT_VOCAB is special")
                    .into_bytes()
            })
            .collect();
        let pattern = fancy_regex::Regex::new(PATTERN)
            .map_err(|e| TokenizerError::Pretokenize(e.to_string()))?;
        let after_unicode_15 = regex::Regex::new(AFTER_UNICODE_15)
            .map_err(|e| TokenizerError::Pretokenize(e.to_string()))?;
        Ok(Self {
            encoder,
            decoder,
            specials,
            pattern,
            after_unicode_15,
        })
    }

    /// NFC as the reference computes it (Unicode 15.0 data): code points assigned later are
    /// starters that never decompose or compose, so they split the text into independently
    /// normalized runs.
    pub fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut start = 0;
        for later in self.after_unicode_15.find_iter(text) {
            out.extend(text[start..later.start()].nfc());
            out.push_str(later.as_str());
            start = later.end();
        }
        out.extend(text[start..].nfc());
        out
    }

    /// Ordinary-token IDs of `text` (NFC first). Special-token text is encoded as ordinary text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let text = self.normalize(text);
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        for piece in self.pattern.find_iter(&text) {
            let piece = piece.map_err(|e| TokenizerError::Pretokenize(e.to_string()))?;
            self.encode_piece(piece.as_str().as_bytes(), &mut ids);
        }
        Ok(ids)
    }

    fn encode_piece(&self, piece: &[u8], ids: &mut Vec<u32>) {
        if let Some(&rank) = self.encoder.get(piece) {
            ids.push(rank);
            return;
        }
        // Byte-pair merge: repeatedly join the adjacent pair whose concatenation has the lowest
        // rank (the leftmost one when the same bytes recur), until no pair is ranked.
        let mut bounds: Vec<usize> = (0..=piece.len()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..bounds.len().saturating_sub(2) {
                if let Some(&rank) = self.encoder.get(&piece[bounds[i]..bounds[i + 2]]) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    bounds.remove(i + 1);
                }
                None => break,
            }
        }
        ids.extend(bounds.windows(2).map(|w| {
            // Every part is a single byte (all 256 are ranked, checked at load) or a merge result
            // that was looked up above.
            *self
                .encoder
                .get(&piece[w[0]..w[1]])
                .expect("every byte-pair part is ranked")
        }));
    }

    /// Text of `ids`. IDs outside the text vocabulary (`>= 151851`, e.g. music or codec tokens)
    /// are dropped, as upstream does; malformed UTF-8 decodes to U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token) = self.decoder.get(id as usize) {
                bytes.extend_from_slice(token);
            } else if (ORDINARY_TOKENS..TEXT_VOCAB).contains(&id) {
                bytes.extend_from_slice(&self.specials[(id - ORDINARY_TOKENS) as usize]);
            } else {
                // Only a test table smaller than the checkpoint's has a gap below 151643.
                assert!(
                    id >= TEXT_VOCAB,
                    "rank {id} is not in this (partial, test-only) table"
                );
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// `base64(token) rank` lines, ranks exactly `0..n` in any order.
fn parse_ranks(bytes: &[u8]) -> Result<Vec<Vec<u8>>, TokenizerError> {
    let mut by_rank: Vec<Option<Vec<u8>>> = Vec::new();
    let mut count = 0usize;
    for (index, line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let malformed = |detail: String| TokenizerError::Malformed {
            line: index + 1,
            detail,
        };
        let mut fields = line
            .split(|b| b.is_ascii_whitespace())
            .filter(|f| !f.is_empty());
        let (Some(token), Some(rank), None) = (fields.next(), fields.next(), fields.next()) else {
            return Err(malformed("expected `<base64 token> <rank>`".into()));
        };
        let token = base64_decode(token).ok_or_else(|| malformed("invalid base64 token".into()))?;
        let rank = std::str::from_utf8(rank)
            .ok()
            .filter(|r| r.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|r| r.parse::<usize>().ok())
            .ok_or_else(|| malformed("rank is not a decimal integer".into()))?;
        if rank >= ORDINARY_TOKENS as usize {
            return Err(malformed(format!(
                "rank {rank} is past the ordinary vocabulary"
            )));
        }
        if by_rank.len() <= rank {
            by_rank.resize(rank + 1, None);
        }
        if by_rank[rank].replace(token).is_some() {
            return Err(malformed(format!("rank {rank} appears twice")));
        }
        count += 1;
    }
    if count != by_rank.len() {
        return Err(TokenizerError::Vocabulary {
            expected: ORDINARY_TOKENS,
            found: format!("{count} tokens with gaps below rank {}", by_rank.len()),
        });
    }
    Ok(by_rank.into_iter().flatten().collect())
}

/// Strict standard-alphabet base64 with `=` padding.
fn base64_decode(text: &[u8]) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }
    if text.is_empty() || !text.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let quads = text.chunks_exact(4);
    let last = quads.len() - 1;
    for (i, quad) in quads.enumerate() {
        let pad = quad.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && i != last) {
            return None;
        }
        let mut n = 0u32;
        for &c in &quad[..4 - pad] {
            n = (n << 6) | value(c)?;
        }
        n <<= 6 * pad as u32;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        // Non-canonical encodings (set bits in the padding) are refused.
        if (pad == 1 && bytes[2] != 0) || (pad == 2 && (bytes[1] != 0 || bytes[2] != 0)) {
            return None;
        }
        out.extend_from_slice(&bytes[..3 - pad]);
    }
    Some(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
