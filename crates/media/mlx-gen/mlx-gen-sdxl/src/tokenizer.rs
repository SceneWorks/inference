//! CLIP byte-pair tokenizer for SDXL and its shared CLIP consumers.
//! `vocab.json` + `merges.txt` encode UTF-8 bytes through CLIP's reversible byte alphabet.
//! Keep the vendored pipeline's dynamic batch-max padding and long-prompt windowing.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use regex::Regex;

use mlx_gen::{Error, Result};

mod host_preflight;

/// The vendored CLIP word-splitting pattern (`Tokenizer.pat`), case-insensitive. Order matters:
/// the two special tokens, then apostrophe contractions, then letter runs, a single digit, and a
/// run of "other" (non-space, non-letter, non-digit) characters.
const CLIP_PATTERN: &str =
    r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+";

/// Merges-file slice end: `49152 - 256 - 2 + 1` (the vendored `bpe_merges` slice `[1:48895]`),
/// dropping the `#version` header line and the trailing unused entries.
const MERGES_END: usize = 49152 - 256 - 2 + 1;

/// The padding token id the vendored `StableDiffusion._tokenize` uses (`0`, the CLIP `!` token) —
/// NOT EOS, so the encoder's `argmax(-1)` EOS-pooling still finds the real end-of-text token.
pub const PAD_ID: i32 = 0;

/// CLIP context length (`ClipTextConfig.max_length`). The position-embedding table is `[77, D]`, so a
/// single encoder forward can never see more than this many ids — one past it would gather
/// out-of-bounds position rows (silent garbage in MLX), which is why
/// [`ClipTextEncoder::forward`](crate::text_encoder::ClipTextEncoder::forward) guards on it (F-062).
///
/// It is **not** a prompt-length cap. Up to sc-20528 [`ClipBpeTokenizer::tokenize`] truncated here,
/// so a long prompt silently lost its tail on this lane while the candle lane conditioned on all of
/// it. A prompt past the window is now split into windows by
/// [`ClipBpeTokenizer::tokenize_windows`] and the per-window hidden states are concatenated on the
/// sequence axis (the `long_prompt` module).
pub const MAX_LENGTH: usize = 77;

/// CLIP's reversible byte alphabet (OpenAI CLIP `bytes_to_unicode`).
fn clip_byte_encoder() -> [char; 256] {
    let mut next = 256;
    std::array::from_fn(|byte| {
        let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
            byte as u32
        } else {
            let code = next;
            next += 1;
            code
        };
        char::from_u32(code).expect("CLIP byte alphabet is valid Unicode")
    })
}

/// A loaded CLIP BPE tokenizer.
pub struct ClipBpeTokenizer {
    /// Adjacent-symbol bigram → merge rank (lower = merged first).
    bpe_ranks: HashMap<(String, String), usize>,
    /// Token string → id.
    vocab: HashMap<String, i32>,
    pat: Regex,
    byte_encoder: [char; 256],
    bos_id: i32,
    eos_id: i32,
}

impl ClipBpeTokenizer {
    /// Load from a CLIP tokenizer directory (`<dir>/vocab.json` + `<dir>/merges.txt`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        Self::from_files(dir.join("vocab.json"), dir.join("merges.txt"))
    }

    /// Load from explicit `vocab.json` + `merges.txt` paths.
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        Self::from_readers(
            || {
                std::fs::read_to_string(vocab_path.as_ref())
                    .map_err(|e| Error::Msg(format!("sdxl tokenizer: read vocab.json: {e}")))
            },
            || {
                std::fs::read_to_string(merges_path.as_ref())
                    .map_err(|e| Error::Msg(format!("sdxl tokenizer: read merges.txt: {e}")))
            },
            || Ok(()),
        )
    }

    // Readers are FnOnce and invoked lazily in the original constructor order. In particular,
    // malformed vocab cannot trigger the future merges read. BOS/EOS still precede Regex::new.
    fn from_readers(
        read_vocab: impl FnOnce() -> Result<String>,
        read_merges: impl FnOnce() -> Result<String>,
        before_regex: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let vocab_raw = read_vocab()?;
        let vocab: HashMap<String, i32> = serde_json::from_str(&vocab_raw)
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: parse vocab.json: {e}")))?;

        let merges_raw = read_merges()?;
        // `f.read().strip().split("\n")[1 : MERGES_END]` — drop the header line, cap the count.
        let lines: Vec<&str> = merges_raw.trim().split('\n').collect();
        let end = MERGES_END.min(lines.len());
        let mut bpe_ranks = HashMap::new();
        for (rank, line) in lines[1..end].iter().enumerate() {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                bpe_ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let bos_id = *vocab
            .get("<|startoftext|>")
            .ok_or_else(|| Error::Msg("sdxl tokenizer: vocab lacks <|startoftext|>".into()))?;
        let eos_id = *vocab
            .get("<|endoftext|>")
            .ok_or_else(|| Error::Msg("sdxl tokenizer: vocab lacks <|endoftext|>".into()))?;

        before_regex()?;
        let pat = Regex::new(CLIP_PATTERN)
            .map_err(|e| Error::Msg(format!("sdxl tokenizer: bad pattern: {e}")))?;

        Ok(Self {
            bpe_ranks,
            vocab,
            pat,
            byte_encoder: clip_byte_encoder(),
            bos_id,
            eos_id,
        })
    }

    /// The `<|endoftext|>` id read from this checkpoint's `vocab.json` — the token
    /// [`Self::tokenize`] appends and the one
    /// [`ClipTextEncoder::forward`](crate::text_encoder::ClipTextEncoder::forward) pools at
    /// (`argmax` over the row finds it because EOS is the highest CLIP id). Exposed for consumers
    /// that build their own fixed-width CLIP row and must terminate it with the tokenizer's OWN eos
    /// rather than a hard-coded 49407 (sc-20528: `mlx-gen-sd3`'s `clip_token_ids`).
    pub fn eos_id(&self) -> i32 {
        self.eos_id
    }

    /// BPE-merge one whitespace-split word into its sub-token strings (the vendored `Tokenizer.bpe`).
    /// The last symbol carries the `</w>` end-of-word marker.
    fn bpe(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        // `unigrams = list(text[:-1]) + [text[-1] + "</w>"]`
        let mut unigrams: Vec<String> = Vec::with_capacity(chars.len());
        for (i, c) in chars.iter().enumerate() {
            if i + 1 == chars.len() {
                unigrams.push(format!("{c}</w>"));
            } else {
                unigrams.push(c.to_string());
            }
        }
        if unigrams.len() < 2 {
            return unigrams;
        }

        loop {
            // Find the adjacent bigram with the lowest merge rank.
            let mut best: Option<(usize, (&str, &str))> = None;
            for w in unigrams.windows(2) {
                if let Some(&rank) = self.bpe_ranks.get(&(w[0].clone(), w[1].clone())) {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, (&w[0], &w[1])));
                    }
                }
            }
            let Some((_, (a, b))) = best else { break };
            let (a, b) = (a.to_string(), b.to_string());

            // Merge every non-overlapping occurrence of (a, b), left to right.
            let mut merged: Vec<String> = Vec::with_capacity(unigrams.len());
            let mut i = 0;
            while i < unigrams.len() {
                if i + 1 < unigrams.len() && unigrams[i] == a && unigrams[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(unigrams[i].clone());
                    i += 1;
                }
            }
            unigrams = merged;
            if unigrams.len() < 2 {
                break;
            }
        }
        unigrams
    }

    /// Tokenize one prompt to CLIP token ids: lowercase + collapse whitespace, regex-split, BPE each
    /// word, map to ids, then prepend BOS and append EOS (the vendored `Tokenizer.tokenize`
    /// defaults). Encode UTF-8 bytes with the CLIP alphabet before BPE, so punctuation, accents,
    /// emoji, and non-Latin scripts use the same vocabulary as ASCII.
    ///
    /// **The full encoding, uncapped** (sc-20528). It used to truncate at [`MAX_LENGTH`], which
    /// silently dropped a long prompt's tail — the divergence from the candle lane this story
    /// closes. Splitting an over-long encoding across CLIP windows is
    /// [`Self::tokenize_windows`]'s job; the position-embedding bound is enforced (typed, not
    /// silent) inside the encoder.
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // `clean_text = regex.sub(r"\s+", " ", text.lower())`
        let lowered = text.to_lowercase();
        let clean = collapse_whitespace(&lowered);

        let mut ids = Vec::new();
        ids.push(self.bos_id);
        for m in self.pat.find_iter(&clean) {
            let word = m.as_str();
            if word == "<|startoftext|>" || word == "<|endoftext|>" {
                ids.push(self.vocab[word]);
                continue;
            }
            let encoded: String = word
                .bytes()
                .map(|byte| self.byte_encoder[byte as usize])
                .collect();
            for sub in self.bpe(&encoded) {
                let id = *self.vocab.get(&sub).ok_or_else(|| {
                    Error::Msg(format!("sdxl tokenizer: token {sub:?} not in vocab"))
                })?;
                ids.push(id);
            }
        }
        ids.push(self.eos_id);
        // sc-20528: NO cap here. The pre-sc-20528 code truncated to `MAX_LENGTH` (F-062) so the ids
        // could never index past the `[77, D]` position table — correct about the table, wrong about
        // the prompt: it dropped the tail without telling anyone, and the candle twin conditioned on
        // the whole thing. `tokenize_windows` splits an over-long encoding across CLIP windows
        // instead; the table bound is enforced by `ClipTextEncoder::forward`'s typed guard.
        Ok(ids)
    }

    /// Tokenize a CFG batch the way the vendored `StableDiffusion._tokenize` does: one row for the
    /// prompt, plus (when `negative` is `Some`) a second row, both padded with [`PAD_ID`] to the
    /// batch-max length. Returns an int32 `[batch, N]` array. When `negative` is `None` the batch is
    /// `[1, N]` (CFG off).
    ///
    /// The **single-window** API: it is the goldens'/parity harnesses' entry point and it holds the
    /// pre-sc-20528 behaviour for every prompt inside the CLIP context. A row past [`MAX_LENGTH`]
    /// cannot be expressed as one window, so it is a typed error naming
    /// [`Self::tokenize_windows`] — the production path, which chunks instead. Nothing truncates.
    pub fn tokenize_batch(&self, prompt: &str, negative: Option<&str>) -> Result<Array> {
        let mut rows = vec![self.tokenize(prompt)?];
        if let Some(neg) = negative {
            rows.push(self.tokenize(neg)?);
        }
        if let Some(len) = rows.iter().map(Vec::len).find(|len| *len > MAX_LENGTH) {
            return Err(Error::Msg(format!(
                "sdxl tokenizer: {len} tokens exceed CLIP's {MAX_LENGTH}-token context; use \
                 `tokenize_windows` (sc-20528), which splits the prompt across windows"
            )));
        }
        Ok(crate::long_prompt::legacy_batch(&rows))
    }
}

/// Replace every run of (unicode) whitespace with a single ASCII space — the vendored
/// `regex.sub(r"\s+", " ", ...)`. Leading/trailing whitespace becomes a single space (not stripped),
/// matching the reference; the splitting pattern then ignores those spaces.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(c);
            in_ws = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_whitespace_matches_reference() {
        assert_eq!(collapse_whitespace("a  b\t c\n"), "a b c ");
        assert_eq!(collapse_whitespace("  x"), " x");
    }

    #[test]
    fn pattern_splits_words_digits_and_punct() {
        let pat = Regex::new(CLIP_PATTERN).unwrap();
        let toks: Vec<&str> = pat
            .find_iter("a red fox, 1024!")
            .map(|m| m.as_str())
            .collect();
        // letters run together; each digit is separate; punctuation runs grouped.
        assert_eq!(toks, vec!["a", "red", "fox", ",", "1", "0", "2", "4", "!"]);
    }

    /// Build a tiny tokenizer with no merges: each single-letter word `x` → one `x</w>` token. Enough
    /// to exercise the length cap without loading the real vocab/merges.
    fn tiny_tokenizer() -> ClipBpeTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("<|startoftext|>".to_string(), 49406);
        vocab.insert("<|endoftext|>".to_string(), 49407);
        vocab.insert("a</w>".to_string(), 320);
        ClipBpeTokenizer {
            bpe_ranks: HashMap::new(),
            vocab,
            pat: Regex::new(CLIP_PATTERN).unwrap(),
            byte_encoder: clip_byte_encoder(),
            bos_id: 49406,
            eos_id: 49407,
        }
    }

    #[test]
    #[ignore = "requires installed CLIP vocabulary; set SDXL_TOKENIZER_DIR"]
    fn installed_clip_vocabulary_encodes_unicode() {
        let tok = ClipBpeTokenizer::from_dir(std::env::var("SDXL_TOKENIZER_DIR").unwrap()).unwrap();
        // CLIP's published byte vocabulary/merges: independent known IDs for em dash and café.
        assert_eq!(tok.tokenize("—").unwrap(), vec![49406, 2005, 49407]);
        for prompt in ["portrait — textured bark", "café", "🙂 世界", "“quoted”"] {
            assert!(tok.tokenize(prompt).unwrap().len() > 2, "{prompt}");
        }
    }

    #[test]
    fn unicode_uses_clip_utf8_bytes_before_merging() {
        let mut tok = tiny_tokenizer();
        // Independent byte-alphabet fixture: UTF-8 em dash E2 80 94 -> â Ģ Ķ.
        for (symbol, id) in [
            ("â", 1),
            ("Ģ", 2),
            ("Ķ</w>", 3),
            ("âĢĶ</w>", 4),
            ("Ã", 5),
            ("©</w>", 6),
        ] {
            tok.vocab.insert(symbol.to_owned(), id);
        }
        assert_eq!(tok.tokenize("—").unwrap(), vec![49406, 1, 2, 3, 49407]);
        assert_eq!(tok.tokenize("É").unwrap(), vec![49406, 5, 6, 49407]);
        tok.bpe_ranks.insert(("â".into(), "Ģ".into()), 0);
        tok.bpe_ranks.insert(("âĢ".into(), "Ķ</w>".into()), 1);
        assert_eq!(
            tok.tokenize("a — a").unwrap(),
            vec![49406, 320, 4, 320, 49407]
        );
    }

    #[test]
    fn byte_alphabet_covers_unicode_and_preserves_special_tokens() {
        let mut tok = tiny_tokenizer();
        for (byte, symbol) in clip_byte_encoder().iter().enumerate() {
            tok.vocab.insert(symbol.to_string(), byte as i32);
            tok.vocab.insert(format!("{symbol}</w>"), byte as i32 + 256);
        }
        for prompt in ["“hello”—世界", "café", "🙂", "日本語", "مرحبا"] {
            let ids = tok.tokenize(prompt).unwrap();
            assert!(ids.len() > 2, "{prompt}");
        }
        assert_eq!(
            tok.tokenize("<|endoftext|>").unwrap(),
            vec![49406, 49407, 49407]
        );
    }

    #[test]
    fn tokenize_keeps_every_token_of_a_long_prompt() {
        // sc-20528: the pre-fix code truncated to MAX_LENGTH here, silently dropping the tail while
        // the candle twin conditioned on all of it. `tokenize` now returns the FULL encoding;
        // windowing it is `tokenize_windows`' job.
        let tok = tiny_tokenizer();
        // 100 single-letter words → 100 content tokens + BOS + EOS = 102 ids.
        let prompt = "a ".repeat(100);
        let ids = tok.tokenize(&prompt).unwrap();
        assert_eq!(ids.len(), 102, "no token may be dropped");
        assert!(
            ids.len() > MAX_LENGTH,
            "the encoding is genuinely over-long"
        );
        assert_eq!(ids[0], 49406, "BOS first");
        assert_eq!(ids[101], 49407, "EOS last");
        assert!(
            ids[1..101].iter().all(|&t| t == 320),
            "all 100 content tokens survive"
        );
    }

    #[test]
    fn tokenize_short_prompt_is_unchanged() {
        // Removing the cap must not touch prompts within the window — parity for the golden domain.
        let tok = tiny_tokenizer();
        let ids = tok.tokenize("a a a").unwrap();
        assert_eq!(ids, vec![49406, 320, 320, 320, 49407]);
    }

    /// The single-window API refuses what it cannot express instead of truncating, and names the
    /// chunking path. (Production never hits this: it calls `tokenize_windows`.)
    #[test]
    fn tokenize_batch_rejects_an_over_long_row() {
        let tok = tiny_tokenizer();
        let long = "a ".repeat(100);
        let err = tok.tokenize_batch(&long, None).unwrap_err().to_string();
        assert!(err.contains("tokenize_windows"), "unexpected error: {err}");
        // The negative row is checked on the same footing as the positive one.
        let err = tok
            .tokenize_batch("a", Some(long.as_str()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("tokenize_windows"), "unexpected error: {err}");
        // …and a batch inside the window still builds.
        assert!(tok.tokenize_batch("a a", Some("a")).is_ok());
    }

    /// The ≤77 request is one window, and that window IS `tokenize_batch`'s array — the structural
    /// half of the sc-20528 byte-identity guarantee (same builder, same dynamic batch-max padding).
    #[test]
    fn short_request_windows_to_the_legacy_batch() {
        let tok = tiny_tokenizer();
        let windows = tok.tokenize_windows("a a a", Some("a")).unwrap();
        assert_eq!(windows.len(), 1, "a short request is a single window");
        let legacy = tok.tokenize_batch("a a a", Some("a")).unwrap();
        assert_eq!(windows.windows()[0].shape(), legacy.shape());
        assert_eq!(
            windows.windows()[0].as_slice::<i32>(),
            legacy.as_slice::<i32>(),
            "the single window must be the pre-sc-20528 token batch, id for id"
        );
    }

    /// A long prompt becomes `ceil(content / 75)` windows of exactly `MAX_LENGTH` ids, with the
    /// short negative row aligned to the same count (so `[cond, uncond]` stays stackable) — the
    /// array-level half of the port, exercised through the real BPE path.
    #[test]
    fn long_request_windows_are_rectangular_and_aligned() {
        let tok = tiny_tokenizer();
        // 100 content tokens ⇒ ceil(100 / 75) = 2 windows.
        let windows = tok.tokenize_windows(&"a ".repeat(100), Some("a")).unwrap();
        assert_eq!(windows.len(), 2);
        for w in windows.windows() {
            assert_eq!(
                w.shape(),
                &[2, MAX_LENGTH as i32],
                "[batch, window] per chunk"
            );
        }
        // Window 0 row 0: BOS + 75 content + EOS, no padding.
        let w0 = windows.windows()[0].as_slice::<i32>().to_vec();
        assert_eq!(w0[0], 49406);
        assert_eq!(w0[76], 49407);
        assert!(w0[1..76].iter().all(|&t| t == 320));
        // Window 1 row 0: BOS + the remaining 25 content tokens + EOS, then padding.
        let w1 = windows.windows()[1].as_slice::<i32>().to_vec();
        assert_eq!(w1[0], 49406);
        assert_eq!(w1[26], 49407);
        assert!(w1[27..MAX_LENGTH].iter().all(|&t| t == PAD_ID));
        // Row 1 (the short negative) is the same width in both windows, empty in the second.
        let neg_w1 = &w1[MAX_LENGTH..];
        assert_eq!(neg_w1[..2], [49406, 49407], "empty filler window");
        assert!(neg_w1[2..].iter().all(|&t| t == PAD_ID));
    }
}

/// Pure-host pre-key count with existing retained asset tokens. The source certificate is checked
/// before numerical guards or reads: an unclosed term is an explicit error, never zero/NoEvidence.
/// Both facts must pass the source-bound checked generator before any guarded read.
pub(crate) fn count_pinned_with_source_capacity(
    vocab: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    merges: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    prompt: &str,
    negative: Option<&str>,
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<u32> {
    let regex = host_preflight::reviewed_regex_source_facts()?;
    count_pinned_using_facts(vocab, merges, prompt, negative, next, &regex)
}

fn count_pinned_using_facts(
    vocab: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    merges: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    prompt: &str,
    negative: Option<&str>,
    next: &mut dyn FnMut(u64) -> Result<()>,
    regex: &host_preflight::RegexSourceFacts,
) -> Result<u32> {
    count_pinned_using_reader(
        vocab,
        merges,
        prompt,
        negative,
        next,
        regex,
        host_preflight::read_pinned_utf8,
    )
}

type PinnedReader = fn(
    &mlx_gen::gen_core::PinnedWeightsFile,
    &str,
    &mut dyn FnMut(u64) -> Result<()>,
) -> Result<String>;

fn count_pinned_using_reader(
    vocab: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    merges: Option<&mlx_gen::gen_core::PinnedWeightsFile>,
    prompt: &str,
    negative: Option<&str>,
    next: &mut dyn FnMut(u64) -> Result<()>,
    regex: &host_preflight::RegexSourceFacts,
    read_pin: PinnedReader,
) -> Result<u32> {
    // Stack-only shared access to one sequential callback; no cache/global/framework is created.
    let next = std::cell::RefCell::new(next);
    let tokenizer = ClipBpeTokenizer::from_readers(
        || {
            let pin = vocab.ok_or_else(|| host_preflight::missing_member("vocab.json"))?;
            (**next.borrow_mut())(host_preflight::vocab_new_upper(
                pin.target_fingerprint().size,
            )?)?;
            read_pin(pin, "vocab.json", &mut **next.borrow_mut())
        },
        || {
            let pin = merges.ok_or_else(|| host_preflight::missing_member("merges.txt"))?;
            (**next.borrow_mut())(host_preflight::merges_new_upper(
                pin.target_fingerprint().size,
            )?)?;
            read_pin(pin, "merges.txt", &mut **next.borrow_mut())
        },
        || (**next.borrow_mut())(regex.constructor()?),
    )?;
    // Current snapshot contains retained tokenizer/vocab/regex. Only NEW row/BPE/cache requests
    // are checked here, not those retained objects again. Empty ChunkPlan probe is included.
    (**next.borrow_mut())(host_preflight::prompt_new_upper(prompt, negative, regex)?)?;
    let mut rows = vec![tokenizer.tokenize(prompt)?];
    if let Some(negative) = negative {
        rows.push(tokenizer.tokenize(negative)?);
    }
    let plan = crate::long_prompt::ChunkPlan::from_tokenizer(&tokenizer)?;
    u32::try_from(crate::long_prompt::common_chunks(&plan, &rows))
        .map_err(|_| Error::Msg("sdxl: CLIP window count overflows u32".into()))
}

#[cfg(test)]
mod pre_key_reader_order_tests {
    use super::*;
    use std::cell::Cell;
    #[test]
    fn lazy_readers_preserve_vocab_parse_merges_bos_regex_error_order_host() {
        let merges = Cell::new(false);
        let regex = Cell::new(false);
        let failed = ClipBpeTokenizer::from_readers(
            || Ok("{".into()),
            || {
                merges.set(true);
                Err(Error::Msg("future merges read".into()))
            },
            || {
                regex.set(true);
                Ok(())
            },
        );
        assert!(failed
            .err()
            .expect("malformed vocab")
            .to_string()
            .contains("parse vocab.json"));
        assert!(!merges.get());
        assert!(!regex.get());
        let failed = ClipBpeTokenizer::from_readers(
            || Ok("{}".into()),
            || {
                merges.set(true);
                Ok("#version\n".into())
            },
            || {
                regex.set(true);
                Ok(())
            },
        );
        assert!(failed
            .err()
            .expect("missing BOS")
            .to_string()
            .contains("vocab lacks <|startoftext|>"));
        assert!(merges.get());
        assert!(!regex.get());
    }
}

#[cfg(test)]
mod pre_key_guarded_count_tests {
    use super::*;
    use mlx_gen::gen_core::{LoadSpec, SourceInventory, WeightsSource};

    fn source_fixture(vocab: &str) -> (tempfile::TempDir, SourceInventory) {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("vocab.json"), vocab).unwrap();
        std::fs::write(root.path().join("merges.txt"), "#version\n").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let source = SourceInventory::capture(&spec).unwrap();
        (root, source)
    }

    const VOCAB: &str = r#"{"<|startoftext|>":49406,"<|endoftext|>":49407,"a</w>":1}"#;

    #[test]
    fn ordinary_readers_keep_all_constructor_error_boundaries() {
        use std::cell::Cell;
        let merges = Cell::new(false);
        let regex = Cell::new(false);
        let failed = ClipBpeTokenizer::from_readers(
            || Err(Error::Msg("vocab read refusal".into())),
            || {
                merges.set(true);
                Ok("#version\n".into())
            },
            || {
                regex.set(true);
                Ok(())
            },
        );
        assert!(failed
            .err()
            .unwrap()
            .to_string()
            .contains("vocab read refusal"));
        assert!(!merges.get() && !regex.get());
        let failed = ClipBpeTokenizer::from_readers(
            || Ok(VOCAB.into()),
            || {
                merges.set(true);
                Err(Error::Msg("merges read refusal".into()))
            },
            || {
                regex.set(true);
                Ok(())
            },
        );
        assert!(failed
            .err()
            .unwrap()
            .to_string()
            .contains("merges read refusal"));
        assert!(merges.get() && !regex.get());
        let failed = ClipBpeTokenizer::from_readers(
            || Ok(r#"{"<|startoftext|>":49406}"#.into()),
            || Ok("#version\n".into()),
            || {
                regex.set(true);
                Ok(())
            },
        );
        assert!(failed
            .err()
            .unwrap()
            .to_string()
            .contains("vocab lacks <|endoftext|>"));
        assert!(!regex.get());
        let failed = ClipBpeTokenizer::from_readers(
            || Ok(VOCAB.into()),
            || Ok("#version\n".into()),
            || Err(Error::Msg("constructor capacity refusal".into())),
        );
        assert!(failed
            .err()
            .unwrap()
            .to_string()
            .contains("constructor capacity refusal"));
    }

    // The hashbrown allocation law is explicitly target-bound. These tests use
    // finite arithmetic/native-reader fixtures, NOT guarded filesystem/regex activation.
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    #[test]
    fn every_requested_phase_can_refuse_before_the_next_phase() {
        let (root, source) = source_fixture(VOCAB);
        let vocab = source.member_pin(&root.path().join("vocab.json")).unwrap();
        let merges = source.member_pin(&root.path().join("merges.txt")).unwrap();
        for stop_at in 0..4 {
            let mut requests = Vec::new();
            let mut next = |bytes| {
                requests.push(bytes);
                if requests.len() == stop_at + 1 {
                    Err(Error::Msg(format!("phase {stop_at} refused")))
                } else {
                    Ok(())
                }
            };
            let result = count_pinned_using_reader(
                vocab,
                merges,
                "a",
                Some("a"),
                &mut next,
                &host_preflight::finite_test_fixture(),
                host_preflight::native_pin_reader_test_fixture,
            );
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains(&format!("phase {stop_at} refused")));
            assert_eq!(requests.len(), stop_at + 1);
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    #[test]
    fn malformed_vocab_does_not_request_merges_and_bos_eos_precede_regex() {
        for (vocab_json, expected, expected_phases) in [
            ("{", "parse vocab.json", 1),
            ("{}", "vocab lacks <|startoftext|>", 2),
            (
                r#"{"<|startoftext|>":49406}"#,
                "vocab lacks <|endoftext|>",
                2,
            ),
        ] {
            let (root, source) = source_fixture(vocab_json);
            let vocab = source.member_pin(&root.path().join("vocab.json")).unwrap();
            let merges = source.member_pin(&root.path().join("merges.txt")).unwrap();
            let mut requests = Vec::new();
            let mut next = |bytes| {
                requests.push(bytes);
                Ok(())
            };
            let result = count_pinned_using_reader(
                vocab,
                merges,
                "a",
                None,
                &mut next,
                &host_preflight::finite_test_fixture(),
                host_preflight::native_pin_reader_test_fixture,
            );
            assert!(result.err().unwrap().to_string().contains(expected));
            assert_eq!(requests.len(), expected_phases);
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    #[test]
    fn guarded_windows_match_both_shared_clip_rows_and_real_empty_probe() {
        let (root, source) = source_fixture(VOCAB);
        let vocab = source.member_pin(&root.path().join("vocab.json")).unwrap();
        let merges = source.member_pin(&root.path().join("merges.txt")).unwrap();
        let long = std::iter::repeat_n("a", 151).collect::<Vec<_>>().join(" ");
        for (positive, negative) in [
            ("", None),
            ("a", Some("")),
            ("a", Some(long.as_str())),
            (long.as_str(), None),
            (long.as_str(), Some("a")),
        ] {
            let tokenizer = ClipBpeTokenizer::from_dir(root.path()).unwrap();
            let mut rows = vec![tokenizer.tokenize(positive).unwrap()];
            if let Some(negative) = negative {
                rows.push(tokenizer.tokenize(negative).unwrap());
            }
            let plan = crate::long_prompt::ChunkPlan::from_tokenizer(&tokenizer).unwrap();
            let expected = crate::long_prompt::common_chunks(&plan, &rows) as u32;
            let mut requests = Vec::new();
            let mut next = |bytes| {
                requests.push(bytes);
                Ok(())
            };
            let actual = count_pinned_using_reader(
                vocab,
                merges,
                positive,
                negative,
                &mut next,
                &host_preflight::finite_test_fixture(),
                host_preflight::native_pin_reader_test_fixture,
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                requests.len(),
                4,
                "vocab, merges, constructor, all serial rows/probe"
            );
            assert_eq!(requests[2], 64, "labeled constructor fixture");
            assert!(
                requests[3] >= 128,
                "includes labeled search high-water fixture"
            );
            if positive == long.as_str() || negative == Some(long.as_str()) {
                assert_eq!(
                    actual, 3,
                    "no prompt truncation; max over both actual CFG rows"
                );
            }
        }
    }
}
