//! **Seam: stage 2** — the 1B Llama (`m-a-p/YuE-s2-1B-general`) that upsamples one track's
//! codebook-0 codes to all [`NUM_CODEBOOKS`] xcodec codebooks, **teacher-forced**: per frame it is
//! fed the codebook-0 token and greedily decodes the seven residuals with the logits sliced to
//! `[STAGE2_SLICE_MIN, STAGE2_SLICE_MAX]`.
//!
//! A port of upstream YuE-v1 `infer.py`'s `stage2_generate` + `stage2_inference`, with the same
//! output on every input upstream accepts (sc-19381; epic sc-19373 R9):
//!
//! - **Chunking** ([`chunk_plan`]): the track is cut into [`CHUNK_FRAMES`]-frame (6 s) chunks plus
//!   one ragged tail. Each chunk is an independent sequence
//!   `<SOA> <stage_1> cb0₀ … cb0ₙ₋₁ <stage_2>` followed, per frame, by `cb0ₜ r₁ … r₇`. Full chunks
//!   decode **length-grouped** in batches of up to `batch_size` rows (upstream
//!   `--stage2_batch_size`, default [`DEFAULT_BATCH_SIZE`]); the tail decodes alone. Chunks are
//!   reassembled in track order, so batching never changes the output.
//! - **Slice**: upstream blocks `[0, 46358)` and `[53526, mm_vocab)` and so leaves the checkpoint's
//!   untrained `[mm_vocab 83738, model_vocab 83840)` rows open — but its `ids2npy` asserts every id
//!   is below `57622`, so a pick there aborts upstream. The port argmaxes over the slice alone:
//!   identical wherever upstream returns output, and never an abort.
//! - **Greedy ties** break to the lowest id (`torch.argmax`).
//! - **Repair** ([`fix_output`]): a residual the model placed in the wrong codebook decodes outside
//!   `0..1024`; upstream replaces it with the most frequent value of its (unrepaired) codebook row,
//!   the first-seen value winning a count tie.
//! - **Sub-chunk tracks**: upstream crashes on a track shorter than one chunk (`num_batch == 0`
//!   still calls `stage2_generate` on an empty prompt, whose `max()` raises). The port decodes such
//!   a track as the tail chunk it evidently intended (see the fixture producer,
//!   `scripts/reference/yue_stage2_reference.py`).
//!
//! The decode drives candle-llm's Llama ([`candle_llm::CausalLm`]) through a preallocated static
//! KV cache ("gen consumes llm", epic R4): each chunk's prefix is prefilled once and every step
//! appends only the new tokens, where upstream re-prefills the whole context every frame. The
//! cancel flag is checked before every LM step (epic R6).
//!
//! **Numerics.** On the CPU the dense tier computes in f32 (bf16 weights upcast) and matches the
//! reference token-for-token (`tests/stage2_real_weights.rs`). A GPU computes in bf16, whose logits
//! tie or near-tie where f32's do not, so it can pick a different residual at a near-tie — measured
//! and bounded by [`characterise`] (the `metal` tests), not assumed away. The q8/q4 tiers quantize
//! the projections and are characterised the same way.
//!
//! [`StubStage2`] stays as the end-to-end seam test's weights-free double.

use candle_audio::candle_core::{DType, Tensor};
use candle_audio::gen_core::{self, CancelFlag};
use candle_llm::primitives::{KvCache, StaticKvCache};
use candle_llm::{CausalLm, LlamaProvider};

use crate::config::{Assets, Tier};
use crate::tokens::{
    codec_token, CodecFrames, CODEBOOK_SIZE, CODEC_OFFSET, NUM_CODEBOOKS, SOA, STAGE2_SLICE_MAX,
    STAGE2_SLICE_MIN, STAGE_1, STAGE_2,
};

/// The stage-2 seam.
pub trait Stage2Model: Send {
    /// Upsample one track. The returned grid's row 0 must equal `cb0` (checked by the engine).
    /// Must honor `cancel` at chunk granularity by returning [`gen_core::Error::Canceled`].
    fn upsample(&mut self, cb0: &[u32], cancel: &CancelFlag) -> gen_core::Result<CodecFrames>;
}

/// Frames per stage-2 chunk: 6 s at 50 frames/s (upstream's fixed `300`).
pub const CHUNK_FRAMES: usize = 300;
/// Full chunks decoded together (upstream `--stage2_batch_size` default).
pub const DEFAULT_BATCH_SIZE: usize = 4;
/// Residual codebooks decoded per frame.
pub const RESIDUALS: usize = NUM_CODEBOOKS - 1;
/// Width of the residual logit slice `[STAGE2_SLICE_MIN, STAGE2_SLICE_MAX]`.
pub const SLICE_LEN: usize = (STAGE2_SLICE_MAX - STAGE2_SLICE_MIN + 1) as usize;

/// Production loader: resolve the staged tier (`tier` asserts one; `None` loads the staged tier —
/// see [`crate::snapshot::resolve_tier_dir`]) and load it through [`LlamaProvider::load`] (the
/// shared device selection, memory admission and persisted-`quantization` handling).
pub fn load(assets: &Assets, tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage2Model>> {
    let (dir, _tier) = crate::snapshot::resolve_tier_dir(&assets.stage2, tier, "stage-2")?;
    let lm = CandleStage2Lm::load(&dir)?;
    Ok(Box::new(TeacherForcedStage2::new(lm, DEFAULT_BATCH_SIZE)))
}

/// The weights-free stub loader (the seam test's double).
pub fn load_stub(_assets: &Assets, _tier: Option<Tier>) -> gen_core::Result<Box<dyn Stage2Model>> {
    Ok(Box::new(StubStage2))
}

/// **Stub stage 2** — the end-to-end seam test's weights-free double. Codebook `k` of frame `t` is
/// `(cb0[t] + 97·k) mod 1024`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubStage2;

impl Stage2Model for StubStage2 {
    fn upsample(&mut self, cb0: &[u32], cancel: &CancelFlag) -> gen_core::Result<CodecFrames> {
        if cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let codebooks = (0..NUM_CODEBOOKS as u32)
            .map(|k| cb0.iter().map(|&c| (c + 97 * k) % CODEBOOK_SIZE).collect())
            .collect();
        Ok(CodecFrames { codebooks })
    }
}

// ---------------------------------------------------------------------------------------------
// The LM surface.
// ---------------------------------------------------------------------------------------------

/// What the teacher-forced loop needs from a language model: batched incremental decode returning
/// the residual-slice logits.
pub trait Stage2Lm: Send {
    /// Start `batch` fresh sequences, each of which will grow to at most `capacity` tokens.
    fn begin(&mut self, batch: usize, capacity: usize) -> gen_core::Result<()>;
    /// Append `tokens[i]` (every row the same length) to sequence `i` and return each sequence's
    /// next-token logits over `[STAGE2_SLICE_MIN, STAGE2_SLICE_MAX]` — `batch` rows of
    /// [`SLICE_LEN`].
    fn step(&mut self, tokens: &[Vec<u32>]) -> gen_core::Result<Vec<Vec<f32>>>;
}

fn backend(e: impl std::error::Error + Send + Sync + 'static) -> gen_core::Error {
    gen_core::Error::backend(e)
}

/// The candle-llm stage-2 LM: a Llama [`CausalLm`] (bf16 dense, or a prepared q8/q4 tier) decoded
/// through a preallocated per-group [`StaticKvCache`].
pub struct CandleStage2Lm {
    model: CausalLm,
    cache: Option<StaticKvCache>,
    offset: i32,
}

impl std::fmt::Debug for CandleStage2Lm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleStage2Lm")
            .field("device", &self.model().device().location())
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl CandleStage2Lm {
    /// Load one tier directory through [`LlamaProvider::load`] (device from
    /// [`candle_llm::select_device`]).
    pub fn load(dir: &std::path::Path) -> gen_core::Result<Self> {
        let spec = candle_llm::core_llm::LoadSpec::dense(dir.display().to_string());
        let provider = LlamaProvider::load(&spec).map_err(|e| match e {
            candle_llm::core_llm::Error::Unsupported(m) => gen_core::Error::Unsupported(m),
            candle_llm::core_llm::Error::Canceled => gen_core::Error::Canceled,
            // Boxed, not stringified: a memory-admission refusal stays a downcastable
            // `RequestResourceExhausted`.
            other => backend(other),
        })?;
        let model = provider.into_causal_lm().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "candle-audio-yue: {} is not a Llama-family checkpoint",
                dir.display()
            ))
        })?;
        Ok(Self::from_model(model))
    }

    /// Wrap an already-built model (explicit device / dtype — the parity and divergence tests).
    pub fn from_model(model: CausalLm) -> Self {
        Self {
            model,
            cache: None,
            offset: 0,
        }
    }

    /// The decoder.
    pub fn model(&self) -> &CausalLm {
        &self.model
    }
}

impl Stage2Lm for CandleStage2Lm {
    fn begin(&mut self, batch: usize, capacity: usize) -> gen_core::Result<()> {
        self.cache = None; // release the previous group's buffers before allocating
        let model = &self.model;
        let max = usize::try_from(model.config().max_position_embeddings).unwrap_or(0);
        if max > 0 && capacity > max {
            return Err(gen_core::Error::Msg(format!(
                "candle-audio-yue: a stage-2 chunk needs {capacity} positions, the model has {max}"
            )));
        }
        let layout = model.kv_layout();
        let shape = layout
            .layers
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| gen_core::Error::Msg("stage-2 model has no caching layer".into()))?;
        let cache = StaticKvCache::with_value_dim(
            layout.layers.len(),
            batch,
            shape.kv_heads,
            shape.key_dim,
            shape.value_dim,
            capacity,
            layout.dtype,
            &shape.device,
        )
        .map_err(backend)?;
        self.cache = Some(cache);
        self.offset = 0;
        Ok(())
    }

    fn step(&mut self, tokens: &[Vec<u32>]) -> gen_core::Result<Vec<Vec<f32>>> {
        let batch = tokens.len();
        let len = tokens.first().map_or(0, Vec::len);
        if batch == 0 || len == 0 || tokens.iter().any(|t| t.len() != len) {
            return Err(gen_core::Error::Msg(
                "stage-2 step needs a non-empty, uniform token batch".into(),
            ));
        }
        let flat: Vec<u32> = tokens.iter().flatten().copied().collect();
        let model = &self.model;
        let cache = self
            .cache
            .as_mut()
            .ok_or_else(|| gen_core::Error::Msg("stage-2 step before begin".into()))?;
        let ids = Tensor::from_vec(flat, (batch, len), model.device()).map_err(backend)?;
        let logits = model
            .decode_logits(&ids, cache as &mut dyn KvCache, self.offset)
            .map_err(backend)?;
        self.offset += len as i32;
        logits
            .narrow(1, STAGE2_SLICE_MIN as usize, SLICE_LEN)
            .and_then(|l| l.to_dtype(DType::F32))
            .and_then(|l| l.to_vec2::<f32>())
            .map_err(backend)
    }
}

// ---------------------------------------------------------------------------------------------
// The teacher-forced upsampler.
// ---------------------------------------------------------------------------------------------

/// One length group: `rows` consecutive chunks of `frames` frames each, the first starting at
/// frame `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkGroup {
    /// First frame of the group's first chunk.
    pub start: usize,
    /// Chunks decoded together.
    pub rows: usize,
    /// Frames per chunk.
    pub frames: usize,
}

/// Upstream's chunk schedule for a `total`-frame track: the whole [`CHUNK_FRAMES`] chunks in
/// consecutive groups of at most `batch_size` (the last one partial), then the ragged tail as its
/// own single-row group. (Upstream's two branches — everything in one batch when it fits, else
/// `batch_size` groups — are both this schedule.)
pub fn chunk_plan(total: usize, batch_size: usize) -> Vec<ChunkGroup> {
    let batch_size = batch_size.max(1);
    let full = total / CHUNK_FRAMES;
    let mut plan = Vec::new();
    let mut chunk = 0;
    while chunk < full {
        let rows = batch_size.min(full - chunk);
        plan.push(ChunkGroup {
            start: chunk * CHUNK_FRAMES,
            rows,
            frames: CHUNK_FRAMES,
        });
        chunk += rows;
    }
    let tail = total % CHUNK_FRAMES;
    if tail > 0 {
        plan.push(ChunkGroup {
            start: full * CHUNK_FRAMES,
            rows: 1,
            frames: tail,
        });
    }
    plan
}

/// Index of the maximum logit, the lowest index winning a tie (`torch.argmax`). `None` when the
/// row is empty or holds a NaN.
fn argmax(row: &[f32]) -> Option<usize> {
    if row.is_empty() || row.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut best = 0;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > row[best] {
            best = i;
        }
    }
    Some(best)
}

/// The greedy residual id for one row of slice logits.
fn pick(row: &[f32]) -> gen_core::Result<u32> {
    if row.len() != SLICE_LEN {
        return Err(gen_core::Error::Msg(format!(
            "stage-2 LM returned {} logits, expected the {SLICE_LEN}-wide residual slice",
            row.len()
        )));
    }
    argmax(row)
        .map(|i| STAGE2_SLICE_MIN + i as u32)
        .ok_or_else(|| gen_core::Error::Msg("stage-2 LM produced NaN logits".into()))
}

fn check_cancel(cancel: &CancelFlag) -> gen_core::Result<()> {
    if cancel.is_cancelled() {
        Err(gen_core::Error::Canceled)
    } else {
        Ok(())
    }
}

/// The per-chunk prompt: `<SOA> <stage_1> cb0… <stage_2>`.
fn chunk_prefix(cb0: &[u32]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(cb0.len() + 3);
    ids.extend([SOA, STAGE_1]);
    ids.extend(cb0.iter().map(|&c| codec_token(0, c)));
    ids.push(STAGE_2);
    ids
}

/// Positions a chunk of `frames` frames occupies: the prefix, then 8 tokens per frame, less the
/// last frame's final residual (never fed back).
fn chunk_capacity(frames: usize) -> usize {
    frames + 3 + NUM_CODEBOOKS * frames - 1
}

/// Teacher-force one length group. Returns, per row, the frame-major token ids
/// `[cb0ₜ, r₁ … r₇]` for every frame (upstream's `prompt_ids[:, len_prompt:]`).
fn decode_group(
    lm: &mut dyn Stage2Lm,
    rows: &[&[u32]],
    cancel: &CancelFlag,
) -> gen_core::Result<Vec<Vec<u32>>> {
    let frames = rows[0].len();
    lm.begin(rows.len(), chunk_capacity(frames))?;
    let mut feed: Vec<Vec<u32>> = rows.iter().map(|r| chunk_prefix(r)).collect();
    let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(frames * NUM_CODEBOOKS); rows.len()];
    for t in 0..frames {
        for (i, row) in rows.iter().enumerate() {
            let tok = codec_token(0, row[t]);
            feed[i].push(tok);
            out[i].push(tok);
        }
        for _ in 0..RESIDUALS {
            check_cancel(cancel)?;
            let logits = lm.step(&feed)?;
            if logits.len() != rows.len() {
                return Err(gen_core::Error::Msg(format!(
                    "stage-2 LM returned {} rows for a batch of {}",
                    logits.len(),
                    rows.len()
                )));
            }
            for (i, row) in logits.iter().enumerate() {
                let id = pick(row)?;
                out[i].push(id);
                feed[i] = vec![id];
            }
        }
    }
    Ok(out)
}

/// Upstream `ids2npy`: frame-major ids → `[8][T]` codes, each codebook's offset removed. A residual
/// in the wrong codebook decodes outside `0..1024` (negative, or past the end).
fn unflatten(ids: &[u32], frames: usize) -> Vec<Vec<i64>> {
    (0..NUM_CODEBOOKS)
        .map(|k| {
            let base = i64::from(CODEC_OFFSET) + k as i64 * i64::from(CODEBOOK_SIZE);
            (0..frames)
                .map(|t| i64::from(ids[t * NUM_CODEBOOKS + k]) - base)
                .collect()
        })
        .collect()
}

fn in_range(c: i64) -> bool {
    (0..i64::from(CODEBOOK_SIZE)).contains(&c)
}

/// Upstream's `fix_output` repair: every code outside `0..1024` becomes its row's most frequent
/// **unrepaired** value (`Counter(line)` sorted by count, stably — the first-seen value wins a
/// tie). Returns the repaired grid and the number of codes rewritten.
pub fn fix_output(grid: &[Vec<i64>]) -> (Vec<Vec<i64>>, usize) {
    let mut repaired = 0;
    let fixed = grid
        .iter()
        .map(|line| {
            if line.iter().all(|&c| in_range(c)) {
                return line.clone();
            }
            // First-seen order + counts (Python's Counter keeps insertion order).
            let mut seen: Vec<(i64, usize)> = Vec::new();
            for &c in line {
                match seen.iter_mut().find(|(v, _)| *v == c) {
                    Some((_, n)) => *n += 1,
                    None => seen.push((c, 1)),
                }
            }
            let top = seen.iter().map(|&(_, n)| n).max().unwrap_or(0);
            let most = seen.iter().find(|&&(_, n)| n == top).map_or(0, |&(v, _)| v);
            line.iter()
                .map(|&c| {
                    if in_range(c) {
                        c
                    } else {
                        repaired += 1;
                        most
                    }
                })
                .collect()
        })
        .collect();
    (fixed, repaired)
}

/// Run upstream's full stage-2 schedule over `cb0` with `lm`: chunk, decode each length group,
/// reassemble, `ids2npy`, `fix_output`. Returns the grid and the number of repaired codes.
pub fn upsample_with(
    lm: &mut dyn Stage2Lm,
    cb0: &[u32],
    batch_size: usize,
    cancel: &CancelFlag,
) -> gen_core::Result<(CodecFrames, usize)> {
    if cb0.is_empty() {
        return Err(gen_core::Error::Msg(
            "stage 2 needs at least one codebook-0 frame".into(),
        ));
    }
    if let Some(bad) = cb0.iter().find(|&&c| c >= CODEBOOK_SIZE) {
        return Err(gen_core::Error::Msg(format!(
            "stage-2 input code {bad} is outside 0..{CODEBOOK_SIZE}"
        )));
    }
    let mut ids = Vec::with_capacity(cb0.len() * NUM_CODEBOOKS);
    for group in chunk_plan(cb0.len(), batch_size) {
        check_cancel(cancel)?;
        let rows: Vec<&[u32]> = (0..group.rows)
            .map(|r| &cb0[group.start + r * group.frames..][..group.frames])
            .collect();
        for row in decode_group(lm, &rows, cancel)? {
            ids.extend(row);
        }
    }
    let (grid, repaired) = fix_output(&unflatten(&ids, cb0.len()));
    let codebooks = grid
        .into_iter()
        .enumerate()
        .map(|(k, line)| {
            line.into_iter()
                .map(|c| {
                    u32::try_from(c)
                        .ok()
                        .filter(|&c| c < CODEBOOK_SIZE)
                        .ok_or_else(|| {
                            // Upstream would hand this code to the codec; the port refuses it.
                            gen_core::Error::Msg(format!(
                                "stage-2 codebook {k}: the repair left code {c} (a row dominated \
                                 by wrong-codebook residuals)"
                            ))
                        })
                })
                .collect::<gen_core::Result<Vec<u32>>>()
        })
        .collect::<gen_core::Result<Vec<_>>>()?;
    Ok((CodecFrames { codebooks }, repaired))
}

/// The production stage 2: [`upsample_with`] over an owned [`Stage2Lm`].
#[derive(Debug)]
pub struct TeacherForcedStage2<L> {
    lm: L,
    batch_size: usize,
}

impl<L: Stage2Lm> TeacherForcedStage2<L> {
    /// Wrap `lm`, decoding up to `batch_size` full chunks together.
    pub fn new(lm: L, batch_size: usize) -> Self {
        Self {
            lm,
            batch_size: batch_size.max(1),
        }
    }
}

impl<L: Stage2Lm> Stage2Model for TeacherForcedStage2<L> {
    fn upsample(&mut self, cb0: &[u32], cancel: &CancelFlag) -> gen_core::Result<CodecFrames> {
        upsample_with(&mut self.lm, cb0, self.batch_size, cancel).map(|(grid, _)| grid)
    }
}

// ---------------------------------------------------------------------------------------------
// Divergence characterisation.
// ---------------------------------------------------------------------------------------------

/// One residual position where two LMs' greedy picks differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mismatch {
    /// Frame index.
    pub frame: usize,
    /// Residual position within the frame (1..=7).
    pub codebook: usize,
    /// The reference LM's pick.
    pub reference: u32,
    /// The other LM's pick.
    pub other: u32,
    /// The reference LM's logit gap between its pick and the other's (`>= 0`).
    pub reference_margin: f32,
}

/// How far a second LM's greedy residual picks stray from a reference LM's, both teacher-forced
/// along the **reference's** token stream (so one flip does not cascade into the rest).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Divergence {
    /// The reference's frame-major ids `[cb0ₜ, r₁ … r₇]` per frame (its own greedy decode).
    pub reference_ids: Vec<u32>,
    /// Residual positions compared.
    pub positions: usize,
    /// Positions whose picks differ.
    pub mismatches: Vec<Mismatch>,
    /// Largest `|reference − other|` logit over the slice at any position.
    pub max_abs_logit_delta: f32,
    /// Largest `|reference|` logit over the slice at any position (the delta's scale).
    pub max_abs_logit: f32,
}

impl Divergence {
    /// `max |Δlogit| / max |logit|` — the logit noise relative to the logit scale (0 when the
    /// reference logits are all zero).
    pub fn relative_logit_delta(&self) -> f32 {
        if self.max_abs_logit > 0.0 {
            self.max_abs_logit_delta / self.max_abs_logit
        } else {
            0.0
        }
    }

    /// Fraction of compared residual picks that differ (0 when nothing was compared).
    pub fn flip_fraction(&self) -> f64 {
        if self.positions == 0 {
            0.0
        } else {
            self.mismatches.len() as f64 / self.positions as f64
        }
    }
}

/// Characterise `other` against `reference` on one chunk (`1..=CHUNK_FRAMES` frames).
pub fn characterise(
    reference: &mut dyn Stage2Lm,
    other: &mut dyn Stage2Lm,
    cb0: &[u32],
) -> gen_core::Result<Divergence> {
    if cb0.is_empty() || cb0.len() > CHUNK_FRAMES {
        return Err(gen_core::Error::Msg(format!(
            "characterise takes one chunk of 1..={CHUNK_FRAMES} frames, got {}",
            cb0.len()
        )));
    }
    let capacity = chunk_capacity(cb0.len());
    reference.begin(1, capacity)?;
    other.begin(1, capacity)?;
    let mut feed = chunk_prefix(cb0);
    let mut d = Divergence::default();
    for (t, &c) in cb0.iter().enumerate() {
        let tok = codec_token(0, c);
        feed.push(tok);
        d.reference_ids.push(tok);
        for k in 1..=RESIDUALS {
            let batch = [feed];
            let r = reference.step(&batch)?.remove(0);
            let o = other.step(&batch)?.remove(0);
            let (rp, op) = (pick(&r)?, pick(&o)?);
            for (a, b) in r.iter().zip(&o) {
                d.max_abs_logit_delta = d.max_abs_logit_delta.max((a - b).abs());
                d.max_abs_logit = d.max_abs_logit.max(a.abs());
            }
            d.positions += 1;
            if rp != op {
                let at = |id: u32| r[(id - STAGE2_SLICE_MIN) as usize];
                d.mismatches.push(Mismatch {
                    frame: t,
                    codebook: k,
                    reference: rp,
                    other: op,
                    reference_margin: at(rp) - at(op),
                });
            }
            d.reference_ids.push(rp);
            feed = vec![rp];
        }
    }
    Ok(d)
}

#[cfg(test)]
pub(crate) mod tests;
