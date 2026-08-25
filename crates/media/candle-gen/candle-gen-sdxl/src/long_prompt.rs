//! Long-prompt chunk-and-concatenate for the SDXL CLIP text encoders (sc-20528).
//!
//! CLIP's 77-token context is architectural — the position-embedding table has exactly
//! `max_position_embeddings` rows, so no single forward can see more. Before sc-20528 every
//! CLIP-conditioned SDXL family (`sdxl`, `illustrious_xl_v1/v2`, `realvisxl`, `realvisxl_lightning`
//! — they all share [`crate::pipeline`]) answered an over-long prompt with a hard error
//! (`prompt too long: 146 tokens > 77`), while the T5/Qwen3-VL families rendered the same prompt
//! fine.
//!
//! This module implements the mainstream fix (A1111 / compel "long prompt weighting"): split the
//! prompt into `≤ window - 2` content-token pieces, re-wrap **each** piece in the tokenizer's own
//! `BOS … EOS` and pad it to the full window, run every window through the encoder separately, and
//! concatenate the resulting hidden states along the **sequence** axis. Cross-attention consumes an
//! arbitrary key/value length, so a `[B, n·77, 2048]` conditioning is a drop-in for `[B, 77, 2048]`.
//! Nothing is truncated and nothing is dropped silently.
//!
//! Two invariants the callers depend on:
//!
//! 1. **`ids.len() <= window` is the pre-sc-20528 encoding, bit for bit** — [`ChunkPlan::rows`]
//!    returns exactly one row that is the tokenizer output right-padded with the pad id, which is
//!    literally what the old `encode` closure built. Every ≤77-token prompt therefore produces
//!    byte-identical conditioning (pinned by `short_prompt_row_is_the_legacy_padded_encoding`).
//! 2. **All four encodings of a request share one sequence length.** SDXL concatenates CLIP-L and
//!    CLIP-bigG on the feature axis and stacks `[uncond, cond]` on the batch axis, so the positive
//!    prompt, the negative prompt and both encoders must agree on the chunk count. Callers take the
//!    max over all of them ([`ChunkPlan::rows_aligned`] tops the short ones up with empty
//!    `BOS EOS pad…` windows — exactly what an empty prompt encodes to).

use candle_gen::{CandleError, Result};
use tokenizers::Tokenizer;

/// The CLIP default pad token when an encoder config carries no `pad_with` (the candle txt2img rule;
/// SDXL itself sets `"!"`).
const DEFAULT_PAD_TOKEN: &str = "<|endoftext|>";

/// One CLIP encoder's tokenization contract: the `(bos, eos)` pair its tokenizer wraps every
/// encoding in, its pad-token id, and its position-embedding window (`max_position_embeddings`).
///
/// Built once per encoder per encode call — it holds no tensors and no weights.
#[derive(Debug, Clone)]
pub(crate) struct ChunkPlan {
    /// The special-token wrapper probed off the tokenizer, or `None` for a tokenizer that adds no
    /// specials (then a chunk is `window` raw content tokens).
    specials: Option<(u32, u32)>,
    pad_id: u32,
    window: usize,
}

impl ChunkPlan {
    /// Resolve the pad id from the encoder config's `pad_with` (default [`DEFAULT_PAD_TOKEN`]) and
    /// probe the tokenizer's special-token wrapper by encoding the empty string: a CLIP tokenizer
    /// answers `[BOS, EOS]`, anything else is treated as "no specials".
    pub(crate) fn new(tok: &Tokenizer, pad_with: Option<&str>, window: usize) -> Result<Self> {
        if window < 2 {
            return Err(CandleError::Msg(format!(
                "sdxl: CLIP context window {window} is too small to chunk a prompt"
            )));
        }
        let pad_token = pad_with.unwrap_or(DEFAULT_PAD_TOKEN);
        let pad_id = *tok
            .get_vocab(true)
            .get(pad_token)
            .ok_or_else(|| CandleError::Msg(format!("pad token {pad_token:?} not in vocab")))?;
        let specials = match encode_ids(tok, "")?.as_slice() {
            [bos, eos] => Some((*bos, *eos)),
            _ => None,
        };
        Ok(Self {
            specials,
            pad_id,
            window,
        })
    }

    /// This encoder's context window (`max_position_embeddings`) — the width of every row
    /// [`Self::rows`] emits.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// Tokenize `text` and split it into `window`-wide id rows. One row for any prompt that already
    /// fits (byte-identical to the pre-sc-20528 padded encoding); `ceil(content / (window - 2))`
    /// rows otherwise, never fewer than one.
    pub(crate) fn rows(&self, tok: &Tokenizer, text: &str) -> Result<Vec<Vec<u32>>> {
        self.chunk(&encode_ids(tok, text)?)
    }

    /// [`Self::rows`], topped up with empty `BOS EOS pad…` windows until there are exactly `target`
    /// of them — how the negative prompt and the shorter of the two encoders are brought to the
    /// request's common sequence length. `target` below the natural count is ignored (never
    /// truncates).
    pub(crate) fn rows_aligned(
        &self,
        tok: &Tokenizer,
        text: &str,
        target: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let mut rows = self.rows(tok, text)?;
        while rows.len() < target {
            rows.push(self.build_row(&[]));
        }
        Ok(rows)
    }

    /// Split a full tokenizer encoding (specials included) into `window`-wide rows.
    fn chunk(&self, ids: &[u32]) -> Result<Vec<Vec<u32>>> {
        // The short path is the legacy path verbatim: the tokenizer output, right-padded. Keeping it
        // a distinct branch (rather than a strip-and-rewrap that "should" round-trip) is what makes
        // the ≤77 byte-identity a structural property instead of a hope.
        if ids.len() <= self.window {
            let mut row = ids.to_vec();
            row.resize(self.window, self.pad_id);
            return Ok(vec![row]);
        }

        let content = self.strip_specials(ids);
        let per_chunk = self.window - self.specials.map_or(0, |_| 2);
        if per_chunk == 0 {
            return Err(CandleError::Msg(format!(
                "sdxl: CLIP context window {} leaves no room for prompt tokens",
                self.window
            )));
        }
        let rows: Vec<Vec<u32>> = content
            .chunks(per_chunk)
            .map(|piece| self.build_row(piece))
            .collect();
        Ok(if rows.is_empty() {
            vec![self.build_row(&[])]
        } else {
            rows
        })
    }

    /// Drop the tokenizer's leading BOS / trailing EOS so the remaining content tokens can be
    /// re-wrapped per chunk. Defensive: only strips what is actually there.
    fn strip_specials<'a>(&self, ids: &'a [u32]) -> &'a [u32] {
        let Some((bos, eos)) = self.specials else {
            return ids;
        };
        let mut content = ids;
        if content.first() == Some(&bos) {
            content = &content[1..];
        }
        if content.last() == Some(&eos) {
            content = &content[..content.len() - 1];
        }
        content
    }

    /// `[BOS] piece [EOS] pad…` at exactly `window` ids. `piece` must be ≤ `window - 2` long
    /// (guaranteed by [`Self::chunk`]'s `chunks(per_chunk)`); an empty `piece` yields the empty-row
    /// filler [`Self::rows_aligned`] pads with.
    fn build_row(&self, piece: &[u32]) -> Vec<u32> {
        let mut row = Vec::with_capacity(self.window);
        if let Some((bos, _)) = self.specials {
            row.push(bos);
        }
        row.extend_from_slice(piece);
        if let Some((_, eos)) = self.specials {
            row.push(eos);
        }
        row.resize(self.window, self.pad_id);
        row
    }
}

/// Tokenize `text` with special tokens, mapping the tokenizer's error into the crate error (the
/// tokenizer error type is not `std::error::Error + Send + Sync` everywhere we need it).
pub(crate) fn encode_ids(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok
        .encode(text, true)
        .map_err(|e| CandleError::Msg(format!("tokenize: {e}")))?
        .get_ids()
        .to_vec())
}

/// The common chunk count for a whole request: the max over every `(plan, tokenizer)` encoder and
/// every text (positive + negative). Both encoders and both CFG rows are then encoded at this count
/// so the feature-axis concat and the batch-axis stack line up.
pub(crate) fn common_chunks(
    encoders: &[(&ChunkPlan, &Tokenizer)],
    texts: &[&str],
) -> Result<usize> {
    let mut chunks = 1usize;
    for (plan, tok) in encoders {
        for text in texts {
            chunks = chunks.max(plan.rows(tok, text)?.len());
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOS: u32 = 49406;
    const EOS: u32 = 49407;
    const PAD: u32 = 256; // SDXL pads with "!"
    const WINDOW: usize = 77;

    /// A plan with the SDXL contract, built without a tokenizer (the probe is the only thing a real
    /// tokenizer supplies, and it is exercised separately by the callers).
    fn sdxl_plan() -> ChunkPlan {
        ChunkPlan {
            specials: Some((BOS, EOS)),
            pad_id: PAD,
            window: WINDOW,
        }
    }

    /// A full tokenizer encoding of `n` content tokens: `[BOS] t… [EOS]`, so `len == n + 2`.
    fn encoding(n: usize) -> Vec<u32> {
        let mut ids = vec![BOS];
        ids.extend((0..n as u32).map(|i| i + 1000));
        ids.push(EOS);
        ids
    }

    /// AC5's structural half: a prompt that already fits produces **exactly** the pre-sc-20528 row —
    /// the tokenizer ids, right-padded with the pad id to the window. Byte equality, not a metric.
    #[test]
    fn short_prompt_row_is_the_legacy_padded_encoding() {
        let plan = sdxl_plan();
        for content in [0usize, 1, 10, 74, 75] {
            let ids = encoding(content);
            // The legacy `encode` closure, verbatim.
            let mut legacy = ids.clone();
            while legacy.len() < WINDOW {
                legacy.push(PAD);
            }
            let rows = plan.chunk(&ids).unwrap();
            assert_eq!(
                rows.len(),
                1,
                "{content} content tokens must stay one chunk"
            );
            assert_eq!(
                rows[0], legacy,
                "{content} content tokens must encode byte-identically to the legacy padding"
            );
        }
    }

    /// The boundary the bug report sits on: 77 total ids (75 content + BOS/EOS) is still one chunk;
    /// 78 is two. Both rows are exactly `window` wide.
    #[test]
    fn chunk_boundary_at_the_context_window() {
        let plan = sdxl_plan();

        let exactly_77 = encoding(75);
        assert_eq!(exactly_77.len(), 77);
        let rows = plan.chunk(&exactly_77).unwrap();
        assert_eq!(rows.len(), 1, "exactly 77 ids must not chunk");
        assert_eq!(rows[0].len(), WINDOW);
        // No padding at all at the boundary — the row IS the encoding.
        assert_eq!(rows[0], exactly_77);

        let seventy_eight = encoding(76);
        assert_eq!(seventy_eight.len(), 78);
        let rows = plan.chunk(&seventy_eight).unwrap();
        assert_eq!(rows.len(), 2, "78 ids must split into two windows");
        assert!(rows.iter().all(|r| r.len() == WINDOW));
        // Chunk 0 carries the first 75 content tokens; chunk 1 carries the 76th and then pads.
        assert_eq!(rows[0][0], BOS);
        assert_eq!(rows[0][76], EOS);
        assert_eq!(rows[1][..3], [BOS, 1000 + 75, EOS]);
        assert!(rows[1][3..].iter().all(|&t| t == PAD));
    }

    /// The story's own repro (146 tokens) and a 150+ prompt: content is split into 75-token pieces,
    /// every piece is re-wrapped, and **no token is dropped** — the concatenation of the chunks'
    /// content is the original content, in order.
    #[test]
    fn long_prompt_chunks_losslessly() {
        let plan = sdxl_plan();
        for content in [76usize, 144, 150, 300, 301] {
            let ids = encoding(content);
            let rows = plan.chunk(&ids).unwrap();
            assert_eq!(
                rows.len(),
                content.div_ceil(WINDOW - 2),
                "{content} content tokens ⇒ ceil({content}/75) chunks"
            );
            let mut seen = Vec::new();
            for row in &rows {
                assert_eq!(row.len(), WINDOW, "every chunk is a full window");
                assert_eq!(row[0], BOS, "every chunk re-opens with BOS");
                let eos_at = row
                    .iter()
                    .position(|&t| t == EOS)
                    .expect("chunk has an EOS");
                seen.extend_from_slice(&row[1..eos_at]);
                assert!(
                    row[eos_at + 1..].iter().all(|&t| t == PAD),
                    "everything after a chunk's EOS is padding"
                );
            }
            assert_eq!(
                seen,
                ids[1..ids.len() - 1].to_vec(),
                "chunking must not drop or reorder a single content token"
            );
        }
    }

    /// `rows_aligned` tops a short text up with empty `BOS EOS pad…` windows so the negative prompt
    /// (AC3) and the shorter encoder reach the request's common length — and never truncates a text
    /// that is already longer than `target`.
    #[test]
    fn alignment_pads_with_empty_windows_and_never_truncates() {
        let plan = sdxl_plan();
        let empty_row = plan.build_row(&[]);
        assert_eq!(empty_row[..2], [BOS, EOS]);
        assert!(empty_row[2..].iter().all(|&t| t == PAD));

        let mut rows = plan.chunk(&encoding(5)).unwrap();
        assert_eq!(rows.len(), 1);
        while rows.len() < 3 {
            rows.push(plan.build_row(&[]));
        }
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1], empty_row);
        assert_eq!(rows[2], empty_row);

        // A text past `target` keeps all of its chunks.
        let long = plan.chunk(&encoding(200)).unwrap();
        assert_eq!(long.len(), 3);
    }

    /// A tokenizer that adds no special tokens degrades to plain `window`-wide slicing rather than
    /// erroring or mis-stripping a content token.
    #[test]
    fn no_specials_slices_on_the_bare_window() {
        let plan = ChunkPlan {
            specials: None,
            pad_id: PAD,
            window: 4,
        };
        let rows = plan.chunk(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(rows, vec![vec![1, 2, 3, 4], vec![5, 6, PAD, PAD]]);
    }

    /// A degenerate window is rejected at plan-construction rather than producing a chunk that
    /// cannot hold a single content token.
    #[test]
    fn degenerate_window_is_rejected() {
        // `new` needs a tokenizer, so assert the same guard on the branch `chunk` owns: a 2-wide
        // window with specials has no room for content.
        let plan = ChunkPlan {
            specials: Some((BOS, EOS)),
            pad_id: PAD,
            window: 2,
        };
        assert!(plan.chunk(&[BOS, 1, 2, EOS]).is_err());
    }
}

/// [`ChunkPlan::new`] against a real [`Tokenizer`] — the two things only a live tokenizer supplies:
/// the pad-token lookup and the BOS/EOS probe. Built from a hand-written WordLevel `tokenizer.json`
/// shaped like CLIP's (a `TemplateProcessing` post-processor that wraps every encoding), so the test
/// needs no downloaded artifact.
#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use std::io::Write;

    const BOS_ID: u32 = 0;
    const EOS_ID: u32 = 1;
    const PAD_ID: u32 = 2; // "!" — SDXL's `pad_with`

    /// A CLIP-shaped WordLevel tokenizer: `word₀ … wordₙ` → ids 3.., wrapped in
    /// `<|startoftext|> … <|endoftext|>` by the post-processor exactly as CLIP's does.
    fn clip_shaped_tokenizer(dir: &std::path::Path) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<|startoftext|>".into(), BOS_ID.into());
        vocab.insert("<|endoftext|>".into(), EOS_ID.into());
        vocab.insert("!".into(), PAD_ID.into());
        for i in 0..64u32 {
            vocab.insert(format!("w{i}"), (i + 3).into());
        }
        let special = |content: &str, id: u32| {
            serde_json::json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        };
        let spec = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                special("<|startoftext|>", BOS_ID),
                special("<|endoftext|>", EOS_ID),
            ],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    { "SpecialToken": { "id": "<|startoftext|>", "type_id": 0 } },
                    { "Sequence": { "id": "A", "type_id": 0 } },
                    { "SpecialToken": { "id": "<|endoftext|>", "type_id": 0 } }
                ],
                "pair": [{ "Sequence": { "id": "A", "type_id": 0 } }],
                "special_tokens": {
                    "<|startoftext|>": {
                        "id": "<|startoftext|>", "ids": [BOS_ID], "tokens": ["<|startoftext|>"]
                    },
                    "<|endoftext|>": {
                        "id": "<|endoftext|>", "ids": [EOS_ID], "tokens": ["<|endoftext|>"]
                    }
                }
            },
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
        });
        let path = dir.join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(spec.to_string().as_bytes()).unwrap();
        drop(f);
        Tokenizer::from_file(&path).unwrap()
    }

    /// `ChunkPlan::new` resolves the config pad token and probes `[BOS, EOS]` off the empty encoding,
    /// then windows a real tokenization: a short prompt stays one legacy-shaped row, a long one splits
    /// into `window - 2`-token pieces that each re-open with BOS and close with EOS.
    #[test]
    fn plan_from_a_real_tokenizer_windows_a_long_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let tok = clip_shaped_tokenizer(tmp.path());
        let plan = ChunkPlan::new(&tok, Some("!"), 8).unwrap();
        assert_eq!(
            plan.specials,
            Some((BOS_ID, EOS_ID)),
            "BOS/EOS probed off \"\""
        );
        assert_eq!(plan.pad_id, PAD_ID, "pad id resolved from `pad_with`");

        // 4 words ⇒ 6 ids ⇒ one window, tokenizer output right-padded (the legacy encoding).
        let short = plan.rows(&tok, "w0 w1 w2 w3").unwrap();
        assert_eq!(short.len(), 1);
        assert_eq!(short[0], vec![BOS_ID, 3, 4, 5, 6, EOS_ID, PAD_ID, PAD_ID]);

        // 13 words ⇒ 15 ids > 8 ⇒ ceil(13 / 6) = 3 windows, nothing dropped.
        let words: Vec<String> = (0..13u32).map(|i| format!("w{i}")).collect();
        let long = plan.rows(&tok, &words.join(" ")).unwrap();
        assert_eq!(long.len(), 3);
        assert!(long.iter().all(|r| r.len() == 8 && r[0] == BOS_ID));
        let content: Vec<u32> = long
            .iter()
            .flat_map(|r| {
                let eos = r.iter().position(|&t| t == EOS_ID).unwrap();
                r[1..eos].to_vec()
            })
            .collect();
        assert_eq!(content, (3..16u32).collect::<Vec<_>>(), "no token dropped");

        // `common_chunks` over the positive + an empty negative takes the max (AC3's alignment input).
        let chunks = common_chunks(&[(&plan, &tok)], &["", &words.join(" ")]).unwrap();
        assert_eq!(chunks, 3);
        assert_eq!(plan.rows_aligned(&tok, "", chunks).unwrap().len(), 3);
    }
}

/// Encoder-level coverage: the chunk rows driven through a real (tiny, randomly-initialized) CLIP
/// tower on CPU. Sub-second — no weights, no GPU, no network.
#[cfg(test)]
mod encoder_tests {
    use super::*;
    use crate::clip::{Activation, ClipTextTransformer, Config};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    const BOS: u32 = 62;
    const EOS: u32 = 63;
    const PAD: u32 = 61;
    const WINDOW: usize = 8;

    /// A CLIP tower small enough to run in a unit test, with an 8-wide context (so "over-long" is 9
    /// ids, not 78) — the chunking contract is a function of `max_position_embeddings`, not of 77.
    fn tiny_tower() -> ClipTextTransformer {
        let cfg = Config {
            vocab_size: 64,
            embed_dim: 32,
            intermediate_size: 64,
            max_position_embeddings: WINDOW,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 2,
            num_attention_heads: 4,
            projection_dim: 32,
            activation: Activation::QuickGelu,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        ClipTextTransformer::new(vb, &cfg).unwrap()
    }

    fn plan() -> ChunkPlan {
        ChunkPlan {
            specials: Some((BOS, EOS)),
            pad_id: PAD,
            window: WINDOW,
        }
    }

    /// Run one id row through the tower, batch 1 — the production per-window forward.
    fn encode_row(tower: &ClipTextTransformer, row: &[u32]) -> Tensor {
        let ids = Tensor::new(row, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        tower.forward_with_mask(&ids, usize::MAX).unwrap()
    }

    fn bytes(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// AC5 at the tensor level: a prompt that fits the context produces conditioning **byte-identical**
    /// to the pre-sc-20528 encode (tokenizer ids right-padded, one forward). Exact `f32` equality —
    /// not a cosine, which is scale-invariant and would pass on a genuinely different embedding.
    #[test]
    fn short_prompt_conditioning_is_byte_identical_to_the_legacy_encode() {
        let tower = tiny_tower();
        let plan = plan();
        let ids = vec![BOS, 1, 2, 3, EOS];

        // Legacy: tokenizer ids, right-padded to the window, one forward.
        let mut legacy_row = ids.clone();
        legacy_row.resize(WINDOW, PAD);
        let legacy = encode_row(&tower, &legacy_row);

        // New: chunk (one window) then forward.
        let rows = plan.chunk(&ids).unwrap();
        assert_eq!(rows.len(), 1);
        let chunked = encode_row(&tower, &rows[0]);

        assert_eq!(chunked.dims(), legacy.dims());
        assert_eq!(
            bytes(&chunked),
            bytes(&legacy),
            "a prompt inside the CLIP context must encode byte-identically to the legacy path"
        );
    }

    /// AC1/AC2: an over-long prompt encodes to `[1, n·window, embed]` — the windows concatenated on
    /// the **sequence** axis — and window 0 of that concatenation is exactly the standalone encode of
    /// window 0, i.e. the leading tokens keep the conditioning the old code would have produced.
    #[test]
    fn long_prompt_concatenates_windows_on_the_sequence_axis() {
        let tower = tiny_tower();
        let plan = plan();
        // 20 content tokens at 6 per window ⇒ 4 windows (the tiny-context analogue of 146 > 77).
        let mut ids = vec![BOS];
        ids.extend(1..=20u32);
        ids.push(EOS);

        let rows = plan.chunk(&ids).unwrap();
        assert_eq!(rows.len(), 4, "20 content tokens at 6/window ⇒ 4 windows");

        let per_window: Vec<Tensor> = rows.iter().map(|r| encode_row(&tower, r)).collect();
        let joined = Tensor::cat(&per_window, 1).unwrap();
        assert_eq!(
            joined.dims(),
            &[1, rows.len() * WINDOW, 32],
            "conditioning grows along the sequence axis, one window per chunk"
        );

        let first = joined.narrow(1, 0, WINDOW).unwrap();
        assert_eq!(
            bytes(&first),
            bytes(&per_window[0]),
            "window 0 of the concatenation is the standalone encode of window 0"
        );
    }

    /// AC3: the negative prompt takes the same path. A short negative aligned to a long positive's
    /// window count yields the SAME sequence length, so the `[uncond, cond]` batch stack the pipeline
    /// builds still concatenates — this is the shape failure a naive per-text chunking would hit.
    #[test]
    fn negative_prompt_aligns_to_the_positive_window_count() {
        let tower = tiny_tower();
        let plan = plan();

        let mut positive = vec![BOS];
        positive.extend(1..=20u32);
        positive.push(EOS);
        let pos_rows = plan.chunk(&positive).unwrap();
        let chunks = pos_rows.len();
        assert_eq!(chunks, 4);

        // The empty negative: one natural window, topped up to the request's four.
        let mut neg_rows = plan.chunk(&[BOS, EOS]).unwrap();
        while neg_rows.len() < chunks {
            neg_rows.push(plan.build_row(&[]));
        }
        assert_eq!(neg_rows.len(), chunks);

        let enc = |rows: &[Vec<u32>]| {
            let per: Vec<Tensor> = rows.iter().map(|r| encode_row(&tower, r)).collect();
            Tensor::cat(&per, 1).unwrap()
        };
        let cond = enc(&pos_rows);
        let uncond = enc(&neg_rows);
        assert_eq!(cond.dims(), uncond.dims());

        // The batch stack the pipeline hands the UNet: `[uncond, cond]`.
        let batched = Tensor::cat(&[uncond, cond], 0).unwrap();
        assert_eq!(batched.dims(), &[2, chunks * WINDOW, 32]);
    }
}
