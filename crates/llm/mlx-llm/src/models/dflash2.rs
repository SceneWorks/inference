//! The DFlash2 block draft on MLX (epic SW-F360, story SW-7194) — a port of the MIT z-lab DFlash
//! reference's `DFlash2DraftModel` (`dflash/model_mlx.py` @ `07ebd93d`: `GroupedDynamicCausalConv`,
//! `DFlash2DecoderLayer`, `CandidateSelector`), the architecture Inco's Apache-2.0 Splash runs on
//! Metal (`runtime/model/DFlashDraft.cpp`, `ops/DraftSelector.cpp`).
//!
//! The draft reads the target's tapped rows as every block draft does
//! ([`crate::models::block_draft`]): `fc` + `hidden_norm` fuse the five tapped layers into the
//! context, whose roped keys and values each draft layer caches over the 2,048-token window
//! (E5). A block `[anchor, mask × (block_size - 1)]` — the trained block whatever the depth, cut
//! short only at the token budget's end — embedded by the target, runs through the draft layers
//! in **one parallel pass** — the published drafts are non-causal (`is_causal: false`): every
//! block row attends to the whole block, and to the context rows its window still covers. Each
//! layer wraps its attention and its MLP in a **two-tap dynamic convolution**: a grouped causal
//! convolution over the block whose per-position, per-group taps are a learned base kernel plus a
//! projection of the row, applied once before the sublayer (`prepare`) and once to its output
//! (`finish`).
//!
//! The **candidate-path selector** then keeps each draft row's top-`K` candidates under the
//! target's LM head and scores the edge from a predecessor token `p` to a candidate `c` at row
//! `i` as `unary(c) + Σ_r pred[p]_r · h_i,r · succ[c]_r` (rank-256 codebooks, `h_i` the row's
//! selector projection). [`DFlash2Draft::table`] scores every candidate of a row against every
//! candidate of the row before (row 0 against the anchor), the same `[rows, K, K]` table a DSpine
//! draft yields: the proposer walks it (greedy: the argmax path the reference takes; sampled: a
//! draw from each scored row), and the candidates are the tree verifier's (SW-7196).
//!
//! Scores are computed in f32 (the reference keeps BF16 logits and edges); the verifier commits
//! only the target's tokens, so the draft's rounding moves acceptance, never the output.

use mlx_rs::ops::indexing::{take_along_axis, TryIndexOp};
use mlx_rs::ops::{add, argpartition_axis, concatenate_axis, matmul, multiply, sort_axis, tanh};
use mlx_rs::{Array, Dtype};

use core_llm::{BlockDraftConfig, BlockDraftHead, BlockDraftPrecision, BlockDraftRefusal};

use crate::error::{Error, Result};
use crate::models::block_draft::{
    bf16, f32, rows_from, BlockDraftContext, ContextWriter, DraftLayer, GroupReader,
};
use crate::primitives::attention::AttnMask;
use crate::primitives::nn::rms_norm;
use crate::primitives::projection::{Projection, QuantGrid, QuantSpec};
use crate::primitives::rope::Rope;
use crate::primitives::Weights;

/// The additive mask's blocked value (the attention primitives' own).
const MASK_NEG: f32 = -1e30;

/// The block forward's activation dtype: its queries, keys and values are BF16, so its additive
/// mask is too (MLX's fused attention needs a mask that promotes to the output dtype).
const ATTN_DTYPE: Dtype = Dtype::Bfloat16;

/// One grouped dynamic causal convolution (`GroupedDynamicCausalConv`): a `[2, taps, hidden]`
/// base kernel (stage × tap) and the projection that gives every row its own per-group taps.
#[derive(Debug)]
struct DynamicConv {
    base_kernel: Array,
    kernel_projection: Projection,
}

/// One DFlash2 layer: the Qwen3 body and its two convolutions.
#[derive(Debug)]
struct DFlash2Layer {
    body: DraftLayer,
    attention_conv: DynamicConv,
    mlp_conv: DynamicConv,
}

/// A loaded DFlash2 draft.
#[derive(Debug)]
pub struct DFlash2Draft {
    cfg: BlockDraftConfig,
    precision: Option<BlockDraftPrecision>,
    fc: Projection,
    hidden_norm: Array,
    layers: Vec<DFlash2Layer>,
    norm: Array,
    /// `candidate_selector.hidden_projection` `[R, hidden]`.
    selector: Projection,
    /// `candidate_selector.predecessor_codebook` / `successor_codebook` `[vocab, R]`, BF16.
    predecessor: Array,
    successor: Array,
    rope: Rope,
    taps: usize,
    group_size: usize,
    causal: bool,
    embedding_scale: f32,
    output_multiplier: f32,
    softcap: Option<f32>,
}

/// Rows `0..n` of `[1, T, ..]` along axis 1.
fn first_rows(a: &Array, n: i32) -> Result<Array> {
    Ok(a.take_axis(&Array::arange::<i32, i32>(0, n, None)?, 1)?)
}

/// `index` along `axis`, the axis kept at size 1.
fn at(a: &Array, index: i32, axis: i32) -> Result<Array> {
    Ok(a.take_axis(Array::from_slice(&[index], &[1]), axis)?)
}

/// The reference's `_grouped_dynamic_convolve`: `hidden` `[1, T, H]` viewed as `G` groups of
/// `group_size` channels; output row `t` sums, over taps `o`, `(base[o] + dynamic[t, o, g]) ·
/// hidden[t - o]` (zero before the block), in the reference's order and dtype.
fn convolve(hidden: &Array, dynamic: &Array, base: &Array, group_size: usize) -> Result<Array> {
    let s = hidden.shape();
    let (t, h) = (s[1], s[2]);
    let groups = h / group_size as i32;
    let taps = base.shape()[0];
    let blocks = hidden.reshape(&[1, t, groups, group_size as i32])?;
    let dynamic = dynamic.reshape(&[1, t, taps, groups, 1])?;
    let mut out =
        Array::zeros::<f32>(&[1, t, groups, group_size as i32])?.as_dtype(hidden.dtype())?;
    for offset in 0..taps {
        let values = if offset == 0 {
            blocks.clone()
        } else {
            let zeros = Array::zeros::<f32>(&[1, offset.min(t), groups, group_size as i32])?
                .as_dtype(hidden.dtype())?;
            if offset >= t {
                zeros
            } else {
                concatenate_axis(&[&zeros, &first_rows(&blocks, t - offset)?], 1)?
            }
        };
        let kernel = base
            .try_index(offset)?
            .reshape(&[1, 1, groups, group_size as i32])?
            .as_dtype(hidden.dtype())?;
        out = add(&out, &multiply(&kernel, &values)?)?;
        let tap = at(&dynamic, offset, 2)?.reshape(&[1, t, groups, 1])?;
        out = add(&out, &multiply(&tap, &values)?)?;
    }
    Ok(out.reshape(&[1, t, h])?)
}

impl DynamicConv {
    fn load(r: &GroupReader<'_>, prefix: &str, quant: Option<QuantSpec>) -> Result<Self> {
        Ok(Self {
            base_kernel: bf16(r.require(&format!("{prefix}.base_kernel"))?)?,
            kernel_projection: Projection::load(
                bf16(r.require(&format!("{prefix}.kernel_projection.weight"))?)?,
                quant,
            )?,
        })
    }

    /// `prepare`: the first-stage convolution of `hidden` and the second stage's taps.
    fn prepare(&self, hidden: &Array, taps: usize, group_size: usize) -> Result<(Array, Array)> {
        let s = hidden.shape();
        let groups = s[2] / group_size as i32;
        let dynamic =
            self.kernel_projection
                .forward(hidden)?
                .reshape(&[1, s[1], 2, taps as i32, groups])?;
        let first = at(&dynamic, 0, 2)?;
        let second = at(&dynamic, 1, 2)?;
        let base = self.base_kernel.try_index(0)?;
        Ok((convolve(hidden, &first, &base, group_size)?, second))
    }

    /// `finish`: the second-stage convolution of the sublayer's output with `prepare`'s taps.
    fn finish(&self, hidden: &Array, second: &Array, group_size: usize) -> Result<Array> {
        convolve(hidden, second, &self.base_kernel.try_index(1)?, group_size)
    }

    fn push_arrays(&self, out: &mut Vec<Array>) {
        out.push(self.base_kernel.clone());
        self.kernel_projection.push_arrays(out);
    }
}

impl DFlash2Draft {
    /// Load the DFlash2 draft in `dir` at `precision` after checking its config's tensors
    /// ([`BlockDraftConfig::check_tensors`]). The caller checks the config against its target
    /// first.
    pub fn load(
        dir: &std::path::Path,
        cfg: BlockDraftConfig,
        precision: BlockDraftPrecision,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let mut w = Weights::from_dir(dir)
            .map_err(|e| BlockDraftRefusal::Tensors(format!("read the weights: {e}")))?;
        Self::from_weights(&mut w, cfg, Some(precision))
    }

    /// Build from `w` — quantized at `precision`, or dense BF16 for `None` (the reference the
    /// quantized tiers are compared with).
    ///
    /// The load runs group by group ([`BlockDraftConfig::load_groups`], the structure
    /// [`BlockDraftConfig::load_bytes`] prices: one group per layer, then the head) through
    /// [`Weights::materialize_groups`]: each group reads its sources on the CPU stream, verifies
    /// them, converts and releases them before the next group reads its own, so the BF16 payload
    /// is never resident beside the converted draft (sc-24245).
    pub fn from_weights(
        w: &mut Weights,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let tensors = |e: Error| BlockDraftRefusal::Tensors(e.to_string());
        let stored: Vec<String> = w.keys().map(str::to_string).collect();
        cfg.check_tensors(
            |name| {
                w.shape(name)
                    .map(|s| s.iter().map(|&d| d.max(0) as usize).collect())
            },
            stored.iter().map(String::as_str),
        )?;
        let groups = cfg.load_groups();
        let draft = Self::build(w, cfg, precision, &groups).map_err(tensors)?;
        w.materialize_groups(&draft.param_groups())
            .map_err(tensors)?;
        Ok(draft)
    }

    /// The draft over `w`, built lazily from `groups` — one per layer, then the head — each
    /// group's arrays read only from its own tensors. Nothing is evaluated.
    fn build(
        w: &Weights,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
        groups: &[core_llm::BlockDraftLoadGroup],
    ) -> Result<Self> {
        let BlockDraftHead::DFlash2 {
            conv_kernel_size,
            conv_group_size,
            causal,
            input_embedding_scale,
            output_multiplier,
            final_logit_softcapping,
            ..
        } = cfg.head
        else {
            return Err(Error::Msg(format!(
                "a {} config is not a DFlash2 draft",
                cfg.format.label()
            )));
        };
        let layer_count = cfg.num_hidden_layers;
        if groups.len() != layer_count + 1 {
            return Err(Error::Msg(format!(
                "{} load groups for {layer_count} layers and the head",
                groups.len()
            )));
        }
        let quant = precision.map(|p| QuantSpec {
            group_size: core_llm::BLOCK_DRAFT_GROUP_SIZE as i32,
            bits: p.bits() as i32,
            grid: QuantGrid::Computed,
        });
        let layers = groups[..layer_count]
            .iter()
            .enumerate()
            .map(|(i, group)| {
                let r = GroupReader {
                    w,
                    tensors: &group.tensors,
                };
                Ok(DFlash2Layer {
                    body: DraftLayer::load(&r, i, quant)?,
                    attention_conv: DynamicConv::load(
                        &r,
                        &format!("layers.{i}.attention_conv"),
                        quant,
                    )?,
                    mlp_conv: DynamicConv::load(&r, &format!("layers.{i}.mlp_conv"), quant)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let r = GroupReader {
            w,
            tensors: &groups[layer_count].tensors,
        };
        let vec = |name: &str| bf16(r.require(name)?);
        Ok(Self {
            fc: Projection::load(vec("fc.weight")?, quant)?,
            hidden_norm: vec("hidden_norm.weight")?,
            norm: vec("norm.weight")?,
            selector: Projection::load(vec("candidate_selector.hidden_projection.weight")?, quant)?,
            predecessor: vec("candidate_selector.predecessor_codebook")?,
            successor: vec("candidate_selector.successor_codebook")?,
            rope: Rope::standard(cfg.head_dim as i32, cfg.rope_theta),
            layers,
            precision,
            taps: conv_kernel_size,
            group_size: conv_group_size,
            causal,
            embedding_scale: input_embedding_scale,
            output_multiplier,
            softcap: final_logit_softcapping,
            cfg,
        })
    }

    /// The draft's arrays in load order ([`BlockDraftConfig::load_groups`]): one group per layer,
    /// then the head (`fc`, the outer norms and the selector).
    fn param_groups(&self) -> Vec<Vec<Array>> {
        let mut groups: Vec<Vec<Array>> = self
            .layers
            .iter()
            .map(|l| {
                let mut out = Vec::new();
                l.body.push_arrays(&mut out);
                l.attention_conv.push_arrays(&mut out);
                l.mlp_conv.push_arrays(&mut out);
                out
            })
            .collect();
        let mut head = Vec::new();
        self.fc.push_arrays(&mut head);
        self.selector.push_arrays(&mut head);
        head.extend([
            self.hidden_norm.clone(),
            self.norm.clone(),
            self.predecessor.clone(),
            self.successor.clone(),
        ]);
        groups.push(head);
        groups
    }

    /// The parsed config.
    pub fn config(&self) -> &BlockDraftConfig {
        &self.cfg
    }

    /// The load precision (`None`: dense BF16).
    pub fn precision(&self) -> Option<BlockDraftPrecision> {
        self.precision
    }

    /// Append target rows `taps` `[1, n, tap_width]` — positions `ctx.next..` — to the context,
    /// then drop rows no later block can see.
    pub fn append_context(&self, ctx: &mut BlockDraftContext, taps: &Array) -> Result<()> {
        ContextWriter {
            cfg: &self.cfg,
            rope: &self.rope,
            fc: &self.fc,
            hidden_norm: &self.hidden_norm,
            layers: self.layers.iter().map(|l| &l.body).collect(),
        }
        .append(ctx, taps)
    }

    /// The attention mask of a `t`-row block at `ctx.next..` over the context's rows then the
    /// block's: a block row at position `p` sees a context row at `c` while `p - c` is inside the
    /// window, and every block row (non-causal) or the block rows up to itself (causal) — the
    /// reference's `context | block` mask. Built in [`ATTN_DTYPE`], the queries' dtype.
    fn mask(&self, ctx: &BlockDraftContext, t: i32) -> Result<Option<Array>> {
        let rows = ctx.rows();
        let window = self.cfg.sliding_window.map(|w| w as i32);
        let sees_context = |i: i32, j: i32| window.is_none_or(|w| i + rows - j < w);
        let all_context = (0..t).all(|i| (0..rows).all(|j| sees_context(i, j)));
        if all_context && !self.causal {
            return Ok(None);
        }
        let k = rows + t;
        let mut data = vec![0f32; (t * k) as usize];
        for i in 0..t {
            for j in 0..k {
                let keep = if j < rows {
                    sees_context(i, j)
                } else {
                    !self.causal || j - rows <= i
                };
                if !keep {
                    data[(i * k + j) as usize] = MASK_NEG;
                }
            }
        }
        Ok(Some(
            Array::from_slice(&data, &[1, 1, t, k]).as_dtype(ATTN_DTYPE)?,
        ))
    }

    /// One parallel block forward over `embeds` `[1, T, hidden]` (the target's embedding of
    /// `[anchor, mask × (T - 1)]`) at positions `ctx.next..`: the final-normalized draft rows
    /// (block rows `1..`) `[T - 1, hidden]`. The context is read, not written.
    pub fn block(&self, ctx: &BlockDraftContext, embeds: &Array) -> Result<Array> {
        let t = embeds.shape()[1];
        let eps = self.cfg.rms_norm_eps;
        let (cos, sin) = self.rope.cos_sin(t, ctx.next, ATTN_DTYPE)?;
        let additive = self.mask(ctx, t)?;
        let mask = match &additive {
            Some(m) => AttnMask::Additive(m),
            None => AttnMask::None,
        };
        let (taps, groups) = (self.taps, self.group_size);
        let mut x = bf16(embeds)?;
        if self.embedding_scale != 1.0 {
            x = bf16(&multiply(&x, Array::from_f32(self.embedding_scale))?)?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&x, &layer.body.input_layernorm, eps)?;
            let (h, second) = layer.attention_conv.prepare(&normed, taps, groups)?;
            let attn = layer
                .body
                .attend(&self.cfg, &h, ctx.layer(i), &cos, &sin, mask)?;
            x = add(&x, &layer.attention_conv.finish(&attn, &second, groups)?)?;
            let normed = rms_norm(&x, &layer.body.post_attention_layernorm, eps)?;
            let (h, second) = layer.mlp_conv.prepare(&normed, taps, groups)?;
            let mlp = layer.body.mlp(&h)?;
            x = add(&x, &layer.mlp_conv.finish(&mlp, &second, groups)?)?;
        }
        let rows = rows_from(&x, 1, 1)?;
        Ok(rms_norm(&rows, &self.norm, eps)?.reshape(&[t - 1, -1])?)
    }

    /// The candidate-path selector's table for the draft `rows` `[n, hidden]`: each row's top-`K`
    /// candidate ids under the target's LM head (`head`: rows → logits), ascending, `[n, K]` i32,
    /// and `[n, K, K]` f32 scores — row `i`'s candidates' `unary + edge` for each candidate sender
    /// of row `i - 1` (row 0's sender is the anchor).
    pub fn table(
        &self,
        rows: &Array,
        anchor: i32,
        head: &dyn Fn(&Array) -> Result<Array>,
    ) -> Result<(Array, Array)> {
        let n = rows.shape()[0];
        let mut logits = f32(&head(rows)?)?;
        if self.output_multiplier != 1.0 {
            logits = multiply(&logits, Array::from_f32(self.output_multiplier))?;
        }
        if let Some(cap) = self.softcap {
            let c = Array::from_f32(cap);
            logits = multiply(&tanh(&logits.divide(&c)?)?, &c)?;
        }
        let width = logits.shape()[1];
        let k = (self.cfg.head_topk as i32).min(width);
        let top = argpartition_axis(&logits.negative()?, k - 1, -1)?;
        let cand = sort_axis(
            &top.take_axis(&Array::arange::<i32, i32>(0, k, None)?, 1)?,
            -1,
        )?
        .as_dtype(Dtype::Int32)?;
        let unary = take_along_axis(&logits, &cand, -1)?;
        let anchor_row = mlx_rs::ops::full::<i32>(&[1, k], Array::from_int(anchor))?;
        let senders = if n > 1 {
            let prev = cand.take_axis(&Array::arange::<i32, i32>(0, n - 1, None)?, 0)?;
            concatenate_axis(&[&anchor_row, &prev], 0)?
        } else {
            anchor_row
        };
        let gather = |book: &Array, ids: &Array| -> Result<Array> {
            Ok(f32(&book.take_axis(&ids.reshape(&[-1])?, 0)?)?.reshape(&[n, k, -1])?)
        };
        let hidden = f32(&self.selector.forward(rows)?)?.expand_dims(1)?;
        let weighted = multiply(&gather(&self.predecessor, &senders)?, &hidden)?;
        let edges = matmul(&weighted, &gather(&self.successor, &cand)?.swap_axes(1, 2)?)?;
        let table = add(&edges, &unary.expand_dims(1)?)?;
        Ok((cand, table))
    }
}

#[cfg(test)]
pub(crate) mod tests;
