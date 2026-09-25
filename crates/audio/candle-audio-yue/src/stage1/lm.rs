//! The production stage 1: the YuE `LlamaForCausalLM` driven through candle-llm's primitives — the
//! Llama decoder ([`CausalLm`], GQA, bf16/q8/q4 projections), its contiguous KV cache, the batched
//! masked forward ([`CausalLm::decode_logits_masked`]) and the host reference sampler
//! ([`sample_row_host`] with its `allowed` mask and the seeded [`SplitMix64`] stream).
//!
//! ## Reference semantics
//!
//! One render is one sequence: every segment's prompt block, its generated tokens and the `<EOA>`
//! that closed it, in order (the reference `raw_output`). Per segment, with the reference
//! `generate` arguments as defaults ([`DecodeConfig`]):
//!
//! 1. **Context.** If the whole sequence outgrows `context_limit − max_new_tokens − 1`, the "smart
//!    context" ([`shorten_context`]) drops the oldest `[start_of_segment]` blocks, falling back to
//!    the last tokens when fewer than three segments are left; the KV cache is then rebuilt from
//!    the shortened window. Otherwise only the tokens the cache has not seen are prefilled.
//! 2. **CFG** (Hugging Face `UnbatchedClassifierFreeGuidanceLogitsProcessor`, run here as one
//!    batch-of-2 forward): row 0 is the conditional stream; row 1 is the unconditional stream,
//!    which sees only the segment prompt's last token (`<xcodec>`) and what is generated after it —
//!    every earlier cache column is masked out of row 1 and its RoPE positions restart at 0. The
//!    scores are `scale · (log_softmax(cond) − log_softmax(uncond)) + log_softmax(uncond)`. A scale
//!    of exactly 1 is no guidance (the raw conditional logits, as `generate` skips the processor)
//!    and, like guidance off, runs batch-of-1.
//! 3. **Shaping and draw**: the repetition penalty (once per distinct id in the window + generated
//!    tokens, Hugging Face's CTRL form), the allow-list `[EOA] + [CODEC_OFFSET, STAGE1_ALLOW_MAX]`
//!    (with `<EOA>` also barred until `min_new_tokens` tokens are out), temperature, top-k, top-p and
//!    the inverse-CDF draw from the render's single seeded stream.
//! 4. **End**: a sampled `<EOA>` ends the segment; when the engine's budget ends it instead,
//!    [`Stage1Model::end_segment`] appends the forced `<EOA>`.

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::gen_core;
use candle_llm::core_llm::{LoadSpec, Quantize};
use candle_llm::primitives::kv_cache::{ContiguousKvCache, KvCache};
use candle_llm::primitives::sampler::{
    logits_rows_host, sample_row_host, SamplingParams, SplitMix64,
};
use candle_llm::{CausalLm, LlamaProvider};

use super::{SegmentStart, Stage1Model, Stage1Step};
use crate::config::Tier;
use crate::tokens::{CODEC_OFFSET, EOA, STAGE1_ALLOW_MAX, START_OF_SEGMENT};

/// Additive-mask fill for a blocked attention column (candle-llm's batched-decode convention: a
/// large finite negative, representable in bf16, so a fully blocked row stays NaN-free).
const MASK_NEG: f32 = -1e30;
/// Query columns per prefill forward. Bounds the `[batch, 1, chunk, keys]` mask a long prefill
/// (a rebuilt 16k-token window) would otherwise materialise at once.
const PREFILL_CHUNK: usize = 512;

fn llm_err(what: &str) -> impl Fn(candle_llm::Error) -> gen_core::Error + '_ {
    move |e| gen_core::Error::Msg(format!("candle-audio-yue stage 1: {what}: {e}"))
}

fn tensor_err(what: &str) -> impl Fn(candle_audio::candle_core::Error) -> gen_core::Error + '_ {
    move |e| gen_core::Error::Msg(format!("candle-audio-yue stage 1: {what}: {e}"))
}

/// The per-segment decode state.
struct Segment {
    /// CFG scale; `None` = no guidance.
    scale: Option<f32>,
    params: SamplingParams,
    min_new_tokens: usize,
    generated: usize,
    /// `<EOA>` was sampled (the segment is over).
    ended: bool,
    /// Last-position logits of the most recent forward, `[batch, vocab]`, not yet sampled from.
    logits: Option<Tensor>,
}

/// The candle-llm driven stage 1 (see the module docs).
pub struct Stage1Lm {
    model: Box<CausalLm>,
    /// Positions the model's context holds (`max_position_embeddings`).
    context_limit: usize,
    vocab: usize,
    rng: SplitMix64,
    /// The render's whole sequence (the reference `raw_output`).
    history: Vec<u32>,
    /// The sequence the KV cache represents: `history`, or its smart-context shortening. The cache
    /// holds `window[..cache.offset()]`; the rest is fed on the next forward.
    window: Vec<u32>,
    cache: Option<ContiguousKvCache>,
    /// Rows in `cache` (1 without guidance, 2 with).
    batch: usize,
    /// First cache column the unconditional row attends (the segment prompt's last token).
    uncond_start: usize,
    allowed: Vec<bool>,
    allowed_no_eoa: Vec<bool>,
    segment: Option<Segment>,
    /// Test hook: the scores each segment's first draw shapes (CFG-mixed log-probabilities, or the
    /// raw conditional logits), before the penalty and the masks.
    #[cfg(test)]
    pub(crate) first_scores: Vec<Vec<f32>>,
}

impl std::fmt::Debug for Stage1Lm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage1Lm")
            .field("context_limit", &self.context_limit)
            .field("vocab", &self.vocab)
            .field("history", &self.history.len())
            .finish_non_exhaustive()
    }
}

/// The directory a stage-1 (or stage-2) `LoadSpec` loads from: `root` itself when it is a model
/// snapshot (`config.json`), else — for a tiered repo root (`sceneworks-tiers.json` with `bf16/`,
/// `q8/`, `q4/`) — the asserted tier's directory, or `bf16/` when no tier was asserted. A tier
/// directory that is not staged falls back to `bf16/`, which the loader then quantizes to the
/// asserted tier on load.
pub(crate) fn lm_snapshot_dir(root: &std::path::Path, tier: Option<Tier>) -> std::path::PathBuf {
    if root.join("config.json").is_file() {
        return root.to_path_buf();
    }
    let sub = match tier {
        Some(Tier::Q8) => "q8",
        Some(Tier::Q4) => "q4",
        Some(Tier::Bf16) | None => "bf16",
    };
    let dir = root.join(sub);
    if dir.join("config.json").is_file() {
        dir
    } else {
        root.join("bf16")
    }
}

impl Stage1Lm {
    /// Load the stage-1 LM from `root` — a snapshot directory, or a tiered repo root whose
    /// `bf16/` / `q8/` / `q4/` directory the tier picks — through [`LlamaProvider::load`]
    /// — the shared `LoadSpec` path: memory admission, device selection, and a prepared q8/q4
    /// snapshot's stored GGML blocks. `tier` asserts Q8/Q4 (quantizing a dense snapshot on load,
    /// refusing a snapshot stored at a different tier); `None` loads whatever tier is staged.
    pub fn load(root: &std::path::Path, tier: Option<Tier>) -> gen_core::Result<Self> {
        let dir = lm_snapshot_dir(root, tier);
        let mut spec = LoadSpec::dense(dir.to_string_lossy().into_owned());
        spec.quantize = match tier {
            Some(Tier::Q8) => Some(Quantize::Q8),
            Some(Tier::Q4) => Some(Quantize::Q4),
            Some(Tier::Bf16) | None => None,
        };
        let provider = LlamaProvider::load(&spec).map_err(|e| {
            gen_core::Error::Msg(format!(
                "candle-audio-yue stage 1: load {}: {e}",
                dir.display()
            ))
        })?;
        let model = provider.into_causal_lm().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "candle-audio-yue stage 1: {} is not a Llama-family checkpoint",
                dir.display()
            ))
        })?;
        Self::from_model(model)
    }

    /// Drive an already-built decoder (e.g. on an explicitly chosen device).
    pub fn from_model(model: CausalLm) -> gen_core::Result<Self> {
        let model = Box::new(model);
        let cfg = model.config();
        let vocab = usize::try_from(cfg.vocab_size).unwrap_or(0);
        let context_limit = usize::try_from(cfg.max_position_embeddings).unwrap_or(0);
        if vocab <= STAGE1_ALLOW_MAX as usize {
            return Err(gen_core::Error::Msg(format!(
                "candle-audio-yue stage 1: vocabulary of {vocab} does not reach the stage-1 \
                 allow-list (ids up to {STAGE1_ALLOW_MAX}); not a YuE stage-1 checkpoint"
            )));
        }
        let mut allowed = vec![false; vocab];
        allowed[EOA as usize] = true;
        for slot in &mut allowed[CODEC_OFFSET as usize..=STAGE1_ALLOW_MAX as usize] {
            *slot = true;
        }
        let mut allowed_no_eoa = allowed.clone();
        allowed_no_eoa[EOA as usize] = false;
        Ok(Self {
            model,
            context_limit,
            vocab,
            rng: SplitMix64::new(0),
            history: Vec::new(),
            window: Vec::new(),
            cache: None,
            batch: 1,
            uncond_start: 0,
            allowed,
            allowed_no_eoa,
            segment: None,
            #[cfg(test)]
            first_scores: Vec::new(),
        })
    }

    /// Override the context the smart-context rule budgets against (default: the checkpoint's
    /// `max_position_embeddings`, 16 384 for YuE).
    pub fn with_context_limit(mut self, positions: usize) -> Self {
        self.context_limit = positions;
        self
    }

    /// The render's whole stage-1 sequence so far (every prompt block, generated token and closing
    /// `<EOA>`) — the reference `raw_output`.
    pub fn sequence(&self) -> &[u32] {
        &self.history
    }

    /// Rows the KV cache runs (1 without effective guidance, 2 with).
    #[cfg(test)]
    pub(crate) fn cache_rows(&self) -> usize {
        self.batch
    }

    /// Feed every window token the cache has not seen, in chunks, returning the last position's
    /// logits `[batch, vocab]`.
    fn feed(&mut self) -> gen_core::Result<Tensor> {
        let lm: &CausalLm = &self.model;
        let device = lm.device().clone();
        let cache = self
            .cache
            .as_mut()
            .expect("the cache is created before any feed");
        let mut start = cache.offset() as usize;
        let end = self.window.len();
        if start >= end {
            return Err(gen_core::Error::Msg(
                "candle-audio-yue stage 1: nothing to feed (internal sequencing defect)".into(),
            ));
        }
        let mut logits = None;
        while start < end {
            let stop = (start + PREFILL_CHUNK).min(end);
            let n = stop - start;
            let tokens = &self.window[start..stop];
            logits = Some(if self.batch == 1 {
                let ids = Tensor::from_vec(tokens.to_vec(), (1, n), &device)
                    .map_err(tensor_err("input ids"))?;
                lm.decode_logits(&ids, cache, start as i32)
                    .map_err(llm_err("forward"))?
            } else {
                forward_cfg(lm, cache, &device, tokens, start, self.uncond_start)?
            });
            start = stop;
        }
        Ok(logits.expect("at least one chunk"))
    }

    /// The scores the draw shapes: CFG-mixed log-probabilities, or the raw conditional logits.
    fn scores(&self, logits: &Tensor, scale: Option<f32>) -> gen_core::Result<Vec<f32>> {
        let mut rows = logits_rows_host(logits).map_err(llm_err("read logits"))?;
        match scale {
            Some(s) if self.batch == 2 && s != 1.0 => {
                let uncond = log_softmax(&rows[1]);
                let mut cond = log_softmax(&rows[0]);
                for (c, u) in cond.iter_mut().zip(&uncond) {
                    *c = s * (*c - u) + u;
                }
                Ok(cond)
            }
            _ => Ok(rows.swap_remove(0)),
        }
    }
}

/// Numerically stable `log_softmax` over one row. The normalizer is accumulated in f64: an f32
/// running sum over an 84k-token row drifts by ~1e-5 relative, which shifts every score by that
/// much (measured against torch's reduction on the parity fixture).
fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = row.iter().map(|&x| f64::from((x - max).exp())).sum();
    let log_z = max + sum.ln() as f32;
    row.iter().map(|&x| x - log_z).collect()
}

/// One batch-of-2 CFG forward over `tokens` (the window columns `start..start + tokens.len()`):
/// both rows are fed the same ids; row 0 attends causally over every column at its absolute
/// position, row 1 only columns `>= uncond_start` with positions shifted to start at 0.
fn forward_cfg(
    lm: &CausalLm,
    cache: &mut dyn KvCache,
    device: &Device,
    tokens: &[u32],
    start: usize,
    uncond_start: usize,
) -> gen_core::Result<Tensor> {
    let n = tokens.len();
    let keys = start + n;
    let ids: Vec<u32> = tokens.iter().chain(tokens).copied().collect();
    let ids = Tensor::from_vec(ids, (2, n), device).map_err(tensor_err("input ids"))?;
    let mut positions = Vec::with_capacity(2 * n);
    positions.extend((start..keys).map(|c| c as i32));
    positions.extend((start..keys).map(|c| c.saturating_sub(uncond_start) as i32));
    let tables = lm
        .rope_tables(&positions, 2, n as i32)
        .map_err(llm_err("rope tables"))?;
    let mut mask = vec![MASK_NEG; 2 * n * keys];
    for q in 0..n {
        let col = start + q;
        for k in 0..=col {
            mask[q * keys + k] = 0.0;
            if k >= uncond_start {
                mask[(n + q) * keys + k] = 0.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (2, 1, n, keys), device)
        .and_then(|m| m.to_dtype(lm.compute_dtype()))
        .map_err(tensor_err("attention mask"))?;
    lm.decode_logits_masked(&ids, cache, &tables, &mask)
        .map_err(llm_err("CFG forward"))
}

/// The smart context (YuE-exllamav2 `shorten_input`): while `seq` is longer than `max_context`,
/// drop everything from the first `[start_of_segment]` up to the second — the oldest segment
/// block. When fewer than three `[start_of_segment]` markers remain (so no segment could be
/// dropped while keeping one before the current), fall back to the last `max_context` tokens.
pub fn shorten_context(seq: &[u32], max_context: usize) -> Vec<u32> {
    let mut seq = seq.to_vec();
    while seq.len() > max_context {
        let starts: Vec<usize> = seq
            .windows(START_OF_SEGMENT.len())
            .enumerate()
            .filter(|(_, w)| *w == START_OF_SEGMENT)
            .map(|(i, _)| i)
            .collect();
        if starts.len() < 3 {
            return seq[seq.len() - max_context..].to_vec();
        }
        seq.drain(starts[0]..starts[1]);
    }
    seq
}

impl Stage1Model for Stage1Lm {
    fn begin_render(&mut self, seed: u64) -> gen_core::Result<()> {
        self.rng = SplitMix64::new(seed);
        self.history.clear();
        self.window.clear();
        self.cache = None;
        self.segment = None;
        #[cfg(test)]
        self.first_scores.clear();
        Ok(())
    }

    fn begin_segment(&mut self, segment: &SegmentStart<'_>) -> gen_core::Result<()> {
        if segment.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "candle-audio-yue stage 1: segment {} has an empty prompt block",
                segment.index
            )));
        }
        if let Some(&bad) = segment.prompt.iter().find(|&&t| t as usize >= self.vocab) {
            return Err(gen_core::Error::Msg(format!(
                "candle-audio-yue stage 1: prompt token {bad} is outside the {}-token vocabulary",
                self.vocab
            )));
        }
        let d = segment.decode;
        let max_new = d.max_new_tokens as usize;
        let max_context = self
            .context_limit
            .checked_sub(max_new + 1)
            .filter(|&m| m > 0)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "candle-audio-yue stage 1: max_new_tokens {max_new} leaves no prompt room in \
                     the {}-position context",
                    self.context_limit
                ))
            })?;
        self.segment = None;
        self.history.extend_from_slice(segment.prompt);
        // A scale of exactly 1 is no guidance (the raw conditional logits), so the unconditional
        // row would be computed only to be discarded: run batch-of-1.
        let batch = match segment.guidance_scale {
            Some(s) if s != 1.0 => 2,
            _ => 1,
        };
        if self.history.len() > max_context {
            self.window = shorten_context(&self.history, max_context);
            self.cache = None;
        } else {
            self.window.extend_from_slice(segment.prompt);
        }
        if batch != self.batch {
            self.cache = None;
            self.batch = batch;
        }
        if self.cache.is_none() {
            self.cache = Some(self.model.new_cache());
        }
        self.uncond_start = self.window.len() - 1;
        let logits = self.feed()?;
        self.segment = Some(Segment {
            scale: segment.guidance_scale,
            params: SamplingParams {
                temperature: d.temperature,
                top_p: d.top_p,
                top_k: d.top_k as usize,
                repetition_penalty: d.repetition_penalty,
                repetition_context: usize::MAX,
                presence_penalty: 0.0,
            },
            min_new_tokens: d.min_new_tokens as usize,
            generated: 0,
            ended: false,
            logits: Some(logits),
        });
        Ok(())
    }

    fn step(&mut self) -> gen_core::Result<Stage1Step> {
        let (scale, logits) = match self.segment.as_mut() {
            Some(s) if !s.ended => (s.scale, s.logits.take()),
            _ => {
                return Err(gen_core::Error::Msg(
                    "candle-audio-yue stage 1: step outside an open segment".into(),
                ))
            }
        };
        let logits = match logits {
            Some(l) => l,
            None => self.feed()?,
        };
        let scores = self.scores(&logits, scale)?;
        #[cfg(test)]
        if self.segment.as_ref().is_some_and(|s| s.generated == 0) {
            self.first_scores.push(scores.clone());
        }
        let seg = self.segment.as_mut().expect("checked above");
        let allowed = if seg.generated < seg.min_new_tokens {
            &self.allowed_no_eoa
        } else {
            &self.allowed
        };
        // The repetition-penalty window is the whole sequence the model sees (`window`, which
        // already holds every generated token); the sampler penalises each distinct id once.
        let penalty_window: Vec<i32> = self.window.iter().map(|&t| t as i32).collect();
        let token = sample_row_host(
            scores,
            &penalty_window,
            &seg.params,
            &mut self.rng,
            Some(allowed),
        ) as u32;
        seg.generated += 1;
        self.history.push(token);
        self.window.push(token);
        if token == EOA {
            self.segment.as_mut().expect("open").ended = true;
            Ok(Stage1Step::EndOfAudio)
        } else {
            Ok(Stage1Step::Token(token))
        }
    }

    fn end_segment(&mut self) -> gen_core::Result<()> {
        let seg = self.segment.take().ok_or_else(|| {
            gen_core::Error::Msg("candle-audio-yue stage 1: end_segment without a segment".into())
        })?;
        if !seg.ended {
            // The budget, not the model, ended the segment: close it with `<EOA>` as the reference
            // does, so the next segment's context carries the terminator.
            self.history.push(EOA);
            self.window.push(EOA);
        }
        Ok(())
    }
}
