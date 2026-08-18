//! Long-prompt chunk-and-concatenate for the SDXL CLIP text encoders (sc-20528) — the MLX twin of
//! `candle-gen-sdxl/src/long_prompt.rs`.
//!
//! CLIP's 77-token context is architectural: the position-embedding table has exactly
//! `max_position_embeddings` rows, so no single forward can see more. Before sc-20528 the candle
//! lane answered an over-long prompt with a hard error and **this** lane silently truncated at 77
//! (`ClipBpeTokenizer::tokenize`, citing F-062/diffusers) — so the same model id conditioned on the
//! full prompt on CUDA and on a clipped prompt on macOS.
//!
//! Both lanes now implement the mainstream fix (A1111 / compel "long prompt weighting"): split the
//! prompt into `≤ window - 2` content-token pieces, re-wrap **each** piece in the tokenizer's own
//! `BOS … EOS` and pad it to the full window, run every window through the encoders separately, and
//! concatenate the resulting hidden states along the **sequence** axis. Cross-attention consumes an
//! arbitrary key/value length, so a `[B, n·77, 2048]` conditioning is a drop-in for `[B, 77, 2048]`.
//! Nothing is truncated and nothing is dropped silently.
//!
//! Two invariants the callers depend on — the same two the candle module documents:
//!
//! 1. **A request whose every row fits the window is the pre-sc-20528 encoding, bit for bit.**
//!    `tokenize_windows` short-circuits to `legacy_batch` — literally the body `tokenize_batch` has
//!    always used (dynamic batch-max padding with `PAD_ID`, *not* a 77-wide pad) — and
//!    `pipeline::encode_conditioning_windows` hands that single window straight to the untouched
//!    `pipeline::encode_conditioning`. The ≤77 path is therefore a *structural* property, not a
//!    numeric hope.
//! 2. **All four encodings of a request share one sequence length.** SDXL concatenates CLIP-L and
//!    CLIP-bigG on the feature axis and stacks `[uncond, cond]` on the batch axis, so the positive
//!    prompt, the negative prompt and both encoders must agree on the window count. On MLX both
//!    encoders read **one** shared token batch (SDXL's `tokenizer/` and `tokenizer_2/` ship
//!    byte-identical `vocab.json` + `merges.txt`), so the max over "both encoders and both CFG rows"
//!    collapses to the max over the rows — computed once, in `common_chunks`, before any row is
//!    windowed. Shorter rows are topped up with empty `BOS EOS pad…` windows, which is exactly what
//!    an empty prompt encodes to.

use mlx_rs::Array;

use mlx_gen::{Error, Result};

use crate::tokenizer::{ClipBpeTokenizer, MAX_LENGTH, PAD_ID};

/// One CLIP tokenization contract: the `(bos, eos)` pair the tokenizer wraps every encoding in, its
/// pad-token id, and its position-embedding window (`ClipTextConfig.max_length`).
///
/// Built once per encode call — it holds no tensors and no weights, and every method on it is pure
/// `Vec<i32>` arithmetic (which is what lets the unit tests below run for free on any host).
#[derive(Debug, Clone)]
pub(crate) struct ChunkPlan {
    /// The special-token wrapper probed off the tokenizer, or `None` for a tokenizer that adds no
    /// specials (then a chunk is `window` raw content tokens).
    specials: Option<(i32, i32)>,
    pad_id: i32,
    window: usize,
}

impl ChunkPlan {
    /// The SDXL plan for a loaded tokenizer: [`PAD_ID`] (the vendored `!` pad, *not* EOS — the
    /// encoder's `argmax` EOS-pooling depends on it), the [`MAX_LENGTH`] window, and the BOS/EOS
    /// wrapper probed by tokenizing the empty string (a CLIP tokenizer answers `[BOS, EOS]`;
    /// anything else is treated as "no specials").
    pub(crate) fn from_tokenizer(tok: &ClipBpeTokenizer) -> Result<Self> {
        let specials = match tok.tokenize("")?.as_slice() {
            [bos, eos] => Some((*bos, *eos)),
            _ => None,
        };
        Self::new(specials, PAD_ID, MAX_LENGTH)
    }

    /// Build a plan, rejecting a window too narrow to hold a single content token. Enforcing it here
    /// (the only constructor) is what makes [`Self::per_chunk`] infallible.
    pub(crate) fn new(specials: Option<(i32, i32)>, pad_id: i32, window: usize) -> Result<Self> {
        let reserved = if specials.is_some() { 2 } else { 0 };
        if window <= reserved {
            return Err(Error::Msg(format!(
                "sdxl: CLIP context window {window} leaves no room for prompt tokens"
            )));
        }
        Ok(Self {
            specials,
            pad_id,
            window,
        })
    }

    /// This encoder's context window — the width of every row [`Self::rows_aligned`] emits when the
    /// request needs more than one window.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// Content tokens per chunk: the window minus the BOS/EOS this plan re-wraps every piece in.
    /// Non-zero by construction ([`Self::new`]).
    fn per_chunk(&self) -> usize {
        self.window - if self.specials.is_some() { 2 } else { 0 }
    }

    /// How many windows a full tokenizer encoding (specials included) needs: one for anything that
    /// already fits, `ceil(content / (window - 2))` otherwise, never fewer than one.
    pub(crate) fn chunk_count(&self, ids: &[i32]) -> usize {
        if ids.len() <= self.window {
            return 1;
        }
        self.strip_specials(ids)
            .len()
            .div_ceil(self.per_chunk())
            .max(1)
    }

    /// Split a full tokenizer encoding into exactly `max(target, self.chunk_count(ids))` rows.
    ///
    /// At `target == 1` (the whole request fits) the row is the encoding **verbatim** — unpadded, so
    /// [`legacy_batch`] can apply the pre-sc-20528 dynamic batch-max padding to it. Past that the
    /// rows are full `window`-wide `[BOS] piece [EOS] pad…` windows, and short texts are topped up
    /// with empty windows so the negative prompt and the positive prompt stack.
    pub(crate) fn rows_aligned(&self, ids: &[i32], target: usize) -> Vec<Vec<i32>> {
        // The short path is the legacy path verbatim. Keeping it a distinct branch (rather than a
        // strip-and-rewrap that "should" round-trip) is what makes the ≤77 byte-identity structural.
        if target <= 1 && self.chunk_count(ids) <= 1 {
            return vec![ids.to_vec()];
        }
        let content = self.strip_specials(ids);
        let mut rows: Vec<Vec<i32>> = content
            .chunks(self.per_chunk())
            .map(|piece| self.build_row(piece))
            .collect();
        if rows.is_empty() {
            rows.push(self.build_row(&[]));
        }
        while rows.len() < target {
            rows.push(self.build_row(&[]));
        }
        rows
    }

    /// Drop the tokenizer's leading BOS / trailing EOS so the remaining content tokens can be
    /// re-wrapped per chunk. Defensive: only strips what is actually there.
    fn strip_specials<'a>(&self, ids: &'a [i32]) -> &'a [i32] {
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

    /// `[BOS] piece [EOS] pad…` at exactly `window` ids. `piece` must be ≤ `per_chunk()` long
    /// (guaranteed by [`Self::rows_aligned`]'s `chunks(..)`); an empty `piece` yields the empty-row
    /// filler alignment pads with.
    fn build_row(&self, piece: &[i32]) -> Vec<i32> {
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

/// The common window count for a whole request: the max over every text (positive + negative).
/// Both CFG rows are then windowed at this count so the batch stack lines up, and — because SDXL's
/// two CLIP encoders share one token batch on this lane — so does the feature-axis concat.
pub(crate) fn common_chunks(plan: &ChunkPlan, rows: &[Vec<i32>]) -> usize {
    rows.iter()
        .map(|ids| plan.chunk_count(ids))
        .max()
        .unwrap_or(1)
}

/// The pre-sc-20528 token batch: the rows right-padded with [`PAD_ID`] to the **batch-max** length
/// (the vendored `StableDiffusion._tokenize` shape — dynamic, not 77-wide) as an int32 `[B, N]`.
///
/// The single definition both `ClipBpeTokenizer::tokenize_batch` and the one-window branch of
/// `ClipBpeTokenizer::tokenize_windows` call, so the two cannot drift and the ≤77 byte-identity
/// survives any later edit to either.
pub(crate) fn legacy_batch(rows: &[Vec<i32>]) -> Array {
    let n = rows.iter().map(Vec::len).max().unwrap_or(0);
    let batch = rows.len() as i32;
    let mut flat = Vec::with_capacity(rows.len() * n);
    for row in rows {
        flat.extend_from_slice(row);
        flat.extend(std::iter::repeat_n(PAD_ID, n - row.len()));
    }
    Array::from_slice(&flat, &[batch, n as i32])
}

/// A tokenized request as the CLIP windows the encoders actually forward: `n` int32 `[B, window]`
/// arrays, one per window, all sharing the batch order `[prompt]` or `[prompt, negative]`.
///
/// `n == 1` is the pre-sc-20528 shape *and* the pre-sc-20528 contents — that window is the array
/// [`ClipBpeTokenizer::tokenize_batch`] would have returned, batch-max padded rather than 77-wide.
pub struct ChunkedTokens {
    windows: Vec<Array>,
}

impl ChunkedTokens {
    /// The per-window token arrays, in prompt order. Never empty.
    pub fn windows(&self) -> &[Array] {
        &self.windows
    }

    /// Window count — `1` for every request whose rows all fit the CLIP context.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Never true (a request always has at least one window); present for the `len`/`is_empty` pair.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

impl ClipBpeTokenizer {
    /// Tokenize a CFG batch into CLIP windows (sc-20528): one row for the prompt, plus (when
    /// `negative` is `Some`) a second row, each split into `window - 2`-token pieces re-wrapped in
    /// `BOS … EOS` and padded to the full window, all rows aligned to one request-wide window count.
    ///
    /// A request whose rows all fit the context collapses to a single window that is **exactly**
    /// [`Self::tokenize_batch`]'s array (both go through the one `legacy_batch` builder), which is
    /// what keeps every ≤77-token render byte-identical to the pre-sc-20528 behaviour. Longer
    /// prompts are chunked, never truncated — this is the method that replaced the silent clip.
    pub fn tokenize_windows(&self, prompt: &str, negative: Option<&str>) -> Result<ChunkedTokens> {
        let mut rows = vec![self.tokenize(prompt)?];
        if let Some(neg) = negative {
            rows.push(self.tokenize(neg)?);
        }

        let plan = ChunkPlan::from_tokenizer(self)?;
        let chunks = common_chunks(&plan, &rows);

        // ── The ≤77 request: the legacy batch, verbatim. No re-wrap, no 77-wide pad, no `cat`.
        if chunks == 1 {
            return Ok(ChunkedTokens {
                windows: vec![legacy_batch(&rows)],
            });
        }

        // ── The chunked request: `chunks` windows of exactly `window` ids per row. `rows_aligned`
        // makes every row's window list the same length, so transposing rows×windows into
        // windows×rows yields `chunks` rectangular `[B, window]` batches in prompt order.
        let batch = rows.len() as i32;
        let width = plan.window();
        let mut flats: Vec<Vec<i32>> = (0..chunks)
            .map(|_| Vec::with_capacity(rows.len() * width))
            .collect();
        for ids in &rows {
            for (flat, window) in flats.iter_mut().zip(plan.rows_aligned(ids, chunks)) {
                flat.extend_from_slice(&window);
            }
        }
        let windows = flats
            .iter()
            .map(|flat| Array::from_slice(flat, &[batch, width as i32]))
            .collect();
        Ok(ChunkedTokens { windows })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOS: i32 = 49406;
    const EOS: i32 = 49407;
    const WINDOW: usize = MAX_LENGTH;

    /// The SDXL contract, built without a tokenizer (the BOS/EOS probe is the only thing a real
    /// tokenizer supplies, and `tokenizer.rs` covers it).
    fn sdxl_plan() -> ChunkPlan {
        ChunkPlan::new(Some((BOS, EOS)), PAD_ID, WINDOW).unwrap()
    }

    /// A full tokenizer encoding of `n` content tokens: `[BOS] t… [EOS]`, so `len == n + 2`.
    fn encoding(n: usize) -> Vec<i32> {
        let mut ids = vec![BOS];
        ids.extend((0..n as i32).map(|i| i + 1000));
        ids.push(EOS);
        ids
    }

    /// The structural half of the ≤77 guarantee: a request that fits stays one row, and that row is
    /// the tokenizer output **verbatim** — the input [`legacy_batch`] has always padded. Byte
    /// equality, not a metric.
    #[test]
    fn short_prompt_row_is_the_tokenizer_encoding_verbatim() {
        let plan = sdxl_plan();
        for content in [0usize, 1, 10, 74, 75] {
            let ids = encoding(content);
            assert_eq!(
                plan.chunk_count(&ids),
                1,
                "{content} tokens must stay one window"
            );
            let rows = plan.rows_aligned(&ids, 1);
            assert_eq!(rows.len(), 1);
            assert_eq!(
                rows[0], ids,
                "{content} content tokens must reach the batch builder unmodified"
            );
        }
    }

    /// The boundary the defect sits on: 76 and 77 total ids are still one window; 78 is two, and
    /// both of those rows are exactly `window` wide.
    #[test]
    fn chunk_boundary_at_the_context_window() {
        let plan = sdxl_plan();

        let seventy_six = encoding(74);
        assert_eq!(seventy_six.len(), 76);
        assert_eq!(plan.chunk_count(&seventy_six), 1, "76 ids must not chunk");

        let exactly_77 = encoding(75);
        assert_eq!(exactly_77.len(), 77);
        assert_eq!(
            plan.chunk_count(&exactly_77),
            1,
            "exactly 77 ids must not chunk"
        );
        assert_eq!(plan.rows_aligned(&exactly_77, 1), vec![exactly_77.clone()]);

        let seventy_eight = encoding(76);
        assert_eq!(seventy_eight.len(), 78);
        assert_eq!(
            plan.chunk_count(&seventy_eight),
            2,
            "78 ids must split into two windows"
        );
        let rows = plan.rows_aligned(&seventy_eight, 2);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.len() == WINDOW));
        // Window 0 carries the first 75 content tokens; window 1 carries the 76th and then pads.
        assert_eq!(rows[0][0], BOS);
        assert_eq!(rows[0][76], EOS);
        assert_eq!(rows[1][..3], [BOS, 1000 + 75, EOS]);
        assert!(rows[1][3..].iter().all(|&t| t == PAD_ID));
    }

    /// The 150/151-token boundary (a second full window plus one token) and the story's own
    /// 146-token repro: content is split into 75-token pieces, every piece is re-wrapped, and **no
    /// token is dropped** — the concatenation of the windows' content is the original, in order.
    #[test]
    fn long_prompt_chunks_losslessly() {
        let plan = sdxl_plan();
        for content in [76usize, 144, 150, 151, 300, 301] {
            let ids = encoding(content);
            let chunks = plan.chunk_count(&ids);
            assert_eq!(
                chunks,
                content.div_ceil(WINDOW - 2),
                "{content} content tokens ⇒ ceil({content}/75) windows"
            );
            let rows = plan.rows_aligned(&ids, chunks);
            assert_eq!(rows.len(), chunks);
            let mut seen = Vec::new();
            for row in &rows {
                assert_eq!(row.len(), WINDOW, "every window is a full context");
                assert_eq!(row[0], BOS, "every window re-opens with BOS");
                let eos_at = row
                    .iter()
                    .position(|&t| t == EOS)
                    .expect("window has an EOS");
                seen.extend_from_slice(&row[1..eos_at]);
                assert!(
                    row[eos_at + 1..].iter().all(|&t| t == PAD_ID),
                    "everything after a window's EOS is padding"
                );
            }
            assert_eq!(
                seen,
                ids[1..ids.len() - 1].to_vec(),
                "chunking must not drop or reorder a single content token"
            );
        }
    }

    /// Alignment tops a short text up with empty `BOS EOS pad…` windows so the negative prompt
    /// reaches the request's common length — and never truncates a text already past `target`.
    #[test]
    fn alignment_pads_with_empty_windows_and_never_truncates() {
        let plan = sdxl_plan();
        let empty_row = plan.build_row(&[]);
        assert_eq!(empty_row[..2], [BOS, EOS]);
        assert!(empty_row[2..].iter().all(|&t| t == PAD_ID));
        assert_eq!(empty_row.len(), WINDOW);

        // An empty negative in a 3-window request: one empty window, topped up to three.
        let rows = plan.rows_aligned(&[BOS, EOS], 3);
        assert_eq!(rows, vec![empty_row.clone(); 3]);

        // A short-but-real negative: its content in window 0, empty windows after.
        let rows = plan.rows_aligned(&encoding(5), 3);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0][..7], [BOS, 1000, 1001, 1002, 1003, 1004, EOS]);
        assert_eq!(rows[1], empty_row);
        assert_eq!(rows[2], empty_row);

        // A text past `target` keeps all of its windows.
        assert_eq!(plan.rows_aligned(&encoding(200), 2).len(), 3);
    }

    /// The empty prompt: one window at `target == 1` (the legacy `[BOS, EOS]` row), the empty
    /// `BOS EOS pad…` window once the request needs more.
    #[test]
    fn empty_prompt_is_one_window() {
        let plan = sdxl_plan();
        let ids = encoding(0);
        assert_eq!(ids, vec![BOS, EOS]);
        assert_eq!(plan.chunk_count(&ids), 1);
        assert_eq!(plan.rows_aligned(&ids, 1), vec![vec![BOS, EOS]]);
        assert_eq!(
            plan.rows_aligned(&ids, 1)[0].len(),
            2,
            "unpadded at target 1"
        );
    }

    /// A tokenizer that adds no special tokens degrades to plain `window`-wide slicing rather than
    /// erroring or mis-stripping a content token.
    #[test]
    fn no_specials_slices_on_the_bare_window() {
        let plan = ChunkPlan::new(None, PAD_ID, 4).unwrap();
        let ids = [1, 2, 3, 4, 5, 6];
        assert_eq!(plan.chunk_count(&ids), 2);
        assert_eq!(
            plan.rows_aligned(&ids, 2),
            vec![vec![1, 2, 3, 4], vec![5, 6, PAD_ID, PAD_ID]]
        );
    }

    /// A degenerate window is rejected at plan construction rather than producing a window that
    /// cannot hold a single content token.
    #[test]
    fn degenerate_window_is_rejected() {
        assert!(ChunkPlan::new(Some((BOS, EOS)), PAD_ID, 2).is_err());
        assert!(ChunkPlan::new(Some((BOS, EOS)), PAD_ID, 1).is_err());
        assert!(ChunkPlan::new(None, PAD_ID, 0).is_err());
        // One content token is enough to be usable.
        assert!(ChunkPlan::new(Some((BOS, EOS)), PAD_ID, 3).is_ok());
    }

    /// The request-wide window count is the max over both CFG rows — the invariant that keeps
    /// `[uncond, cond]` stackable. A long negative against a short positive counts too.
    #[test]
    fn common_chunks_is_the_max_over_both_cfg_rows() {
        let plan = sdxl_plan();
        assert_eq!(common_chunks(&plan, &[encoding(10), encoding(5)]), 1);
        assert_eq!(common_chunks(&plan, &[encoding(150), encoding(0)]), 2);
        assert_eq!(common_chunks(&plan, &[encoding(0), encoding(150)]), 2);
        assert_eq!(common_chunks(&plan, &[encoding(300)]), 4);
        assert_eq!(
            common_chunks(&plan, &[]),
            1,
            "no rows still means one window"
        );
    }

    /// Aligned rows are rectangular across the CFG batch — the precondition
    /// [`ClipBpeTokenizer::tokenize_windows`] relies on when it flattens each window into a
    /// `[B, window]` array.
    #[test]
    fn aligned_rows_are_rectangular_across_the_batch() {
        let plan = sdxl_plan();
        let rows = [encoding(200), encoding(3)];
        let chunks = common_chunks(&plan, &rows);
        assert_eq!(chunks, 3);
        for ids in &rows {
            let windows = plan.rows_aligned(ids, chunks);
            assert_eq!(windows.len(), chunks);
            assert!(windows.iter().all(|w| w.len() == WINDOW));
        }
    }
}
