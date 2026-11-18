//! The DSpine block draft on MLX (epic SW-F360, story SW-7193) — a port of Tencent's Apache-2.0
//! DSpine MLX reference (`dspine_mlx/model.py`, `bridge.py`, `propose.py`).
//!
//! A DFlash-style block draft: the target's tapped hidden rows (its layers'
//! `target_layer_ids` outputs, concatenated) are projected by `fc` + `hidden_norm` into the
//! draft's **context**, whose per-layer keys and values the draft caches; a block `[anchor,
//! mask × depth]`, embedded with the target's embedding, runs through the draft layers attending
//! to the context and causally within itself. After every layer the **code bridge** writes a
//! message from each row's predecessor into the row. The last write is redone per candidate
//! sender (the exact-sender refinement): each block row's top-`K` candidates under the target's
//! LM head are rescored against every candidate of the row before, giving a `[rows, K, K]` table
//! the proposer walks serially ([`DSpineDraft::table`]).
//!
//! The context is bounded by the draft's sliding window (E5): a query sees the `window - 1` rows
//! before it, so the cache keeps only those, and a fresh draft is fed only the last ones.

use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{
    add, argpartition_axis, concatenate_axis, matmul, maximum, mean_axis, multiply, sigmoid,
    sort_axis, sqrt, sum_axis,
};
use mlx_rs::{Array, Dtype};

use core_llm::{BlockDraftConfig, BlockDraftPrecision, BlockDraftRefusal};

use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::nn::{rms_norm, silu};
use crate::primitives::projection::{Projection, QuantGrid, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

/// `torch.finfo(torch.float32).tiny`, the clamp floor of the bridge's read RMS.
const F32_TINY: f32 = 1.175_494_4e-38;

/// One draft decoder layer (Qwen3 attention with q/k norms, SwiGLU MLP).
#[derive(Debug)]
struct DraftLayer {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Array,
    k_norm: Array,
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
    input_layernorm: Array,
    post_attention_layernorm: Array,
}

/// The code bridge's tensors (BF16, never quantized: the refinement depends on their rounding),
/// plus the f32 copies the refinement computes with.
#[derive(Debug)]
struct Bridge {
    read_weight: Array,
    early_read_weights: Array,
    feature_write_weights: Array,
    codebook: Array,
    gate_bias: Array,
    /// `[3R, C]` = `[gate; predecessor_gate; prev]`: one matmul yields a row's query, sender and
    /// value parts.
    packed: Array,
    predecessor_gate_f32: Array,
    prev_f32: Array,
    last_write_f32: Array,
}

/// The draft's context: per layer, the roped keys and the values of the target rows it has read,
/// at most [`BlockDraftConfig::context_rows`] of them (E5).
#[derive(Clone, Debug, Default)]
pub struct DSpineContext {
    keys: Vec<Array>,
    values: Vec<Array>,
    /// The absolute position the next context row (and so the next block's anchor) occupies.
    next: i32,
}

impl DSpineContext {
    /// The absolute position of the next row — the next block's anchor.
    pub fn next_position(&self) -> i32 {
        self.next
    }

    /// Context rows held (bounded by the window).
    pub fn rows(&self) -> i32 {
        self.keys.first().map_or(0, |k| k.shape()[2])
    }

    /// Every layer's keys then values (equivalence tests compare these).
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.keys.iter().chain(&self.values)
    }

    /// Bytes the context holds.
    pub fn bytes(&self) -> u64 {
        self.arrays().map(|a| a.nbytes() as u64).sum()
    }
}

/// One block forward's outputs: the final-normalized rows after the anchor, and the last
/// write's pre-write state for those rows, which the refinement redoes per sender.
pub struct DSpineBlock {
    /// `[n, hidden]` — the draft rows (block rows `1..`), final-normalized, BF16.
    pub rows: Array,
    raw: Array,
    rms: Array,
    query: Array,
}

/// A loaded DSpine draft.
#[derive(Debug)]
pub struct DSpineDraft {
    cfg: BlockDraftConfig,
    precision: Option<BlockDraftPrecision>,
    fc: Projection,
    hidden_norm: Array,
    layers: Vec<DraftLayer>,
    norm: Array,
    bridge: Bridge,
    rope: Rope,
}

fn bf16(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Bfloat16)?)
}

fn f32(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Float32)?)
}

/// Rows `start..` of a `[.., S, ..]` array along `axis`.
fn rows_from(a: &Array, start: i32, axis: i32) -> Result<Array> {
    let n = a.shape()[axis as usize];
    let idx = Array::arange::<i32, i32>(start, n, None)?;
    Ok(a.take_axis(&idx, axis)?)
}

impl DSpineDraft {
    /// Load the DSpine draft in `dir` at `precision` after checking its config and every stored
    /// tensor ([`BlockDraftConfig::check_tensors`]), its version markers and its disabled terminal
    /// projection. The caller checks the config against its target first.
    pub fn load(
        dir: &std::path::Path,
        cfg: BlockDraftConfig,
        precision: BlockDraftPrecision,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let w = Weights::from_dir(dir)
            .map_err(|e| BlockDraftRefusal::Tensors(format!("read the weights: {e}")))?;
        Self::from_weights(&w, cfg, Some(precision))
    }

    /// Build from `w` — quantized at `precision`, or dense BF16 for `None` (the reference the
    /// quantized tiers are compared with).
    pub fn from_weights(
        w: &Weights,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let tensors = |e: Error| BlockDraftRefusal::Tensors(e.to_string());
        let stored: Vec<String> = w.keys().map(str::to_string).collect();
        cfg.check_tensors(
            |name| {
                w.get(name)
                    .map(|a| a.shape().iter().map(|&d| d.max(0) as usize).collect())
            },
            stored.iter().map(String::as_str),
        )?;
        for marker in [
            "draft_block_causal_version",
            "last_write_refine_version",
            "selected_prefix_bridge.independent_write_version",
            "selected_prefix_bridge.terminal_disabled_version",
        ] {
            let version = w
                .require(marker)
                .and_then(|a| Ok(a.as_dtype(Dtype::Int64)?.reshape(&[-1])?))
                .map_err(tensors)?;
            if version.as_slice::<i64>() != [1] {
                return Err(BlockDraftRefusal::Tensors(format!(
                    "`{marker}` is not version 1"
                )));
            }
        }
        let out = w
            .require("selected_prefix_bridge.out.weight")
            .map_err(tensors)?;
        let nonzero = mlx_rs::ops::any(
            &out.ne(Array::from_f32(0.0))
                .map_err(Error::from)
                .map_err(tensors)?,
            None,
        )
        .map_err(Error::from)
        .map_err(tensors)?;
        if nonzero.item::<bool>() {
            return Err(BlockDraftRefusal::Tensors(
                "the terminal-disabled checkpoint has a nonzero terminal projection".into(),
            ));
        }
        Self::build(w, cfg, precision).map_err(tensors)
    }

    fn build(
        w: &Weights,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
    ) -> Result<Self> {
        let quant = precision.map(|p| QuantSpec {
            group_size: core_llm::BLOCK_DRAFT_GROUP_SIZE as i32,
            bits: p.bits() as i32,
            grid: QuantGrid::Computed,
        });
        let proj = |name: &str| Projection::load(bf16(w.require(name)?)?, quant);
        let vec = |name: &str| bf16(w.require(name)?);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = |s: &str| format!("layers.{i}.{s}");
                Ok(DraftLayer {
                    q_proj: proj(&p("self_attn.q_proj.weight"))?,
                    k_proj: proj(&p("self_attn.k_proj.weight"))?,
                    v_proj: proj(&p("self_attn.v_proj.weight"))?,
                    o_proj: proj(&p("self_attn.o_proj.weight"))?,
                    q_norm: vec(&p("self_attn.q_norm.weight"))?,
                    k_norm: vec(&p("self_attn.k_norm.weight"))?,
                    gate_proj: proj(&p("mlp.gate_proj.weight"))?,
                    up_proj: proj(&p("mlp.up_proj.weight"))?,
                    down_proj: proj(&p("mlp.down_proj.weight"))?,
                    input_layernorm: vec(&p("input_layernorm.weight"))?,
                    post_attention_layernorm: vec(&p("post_attention_layernorm.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let b = |s: &str| vec(&format!("selected_prefix_bridge.{s}"));
        let (gate, predecessor, prev) = (
            b("gate.weight")?,
            b("predecessor_gate.weight")?,
            b("prev.weight")?,
        );
        let feature_write_weights = b("feature_write_weights")?;
        let last = cfg.num_hidden_layers as i32 - 1;
        let bridge = Bridge {
            read_weight: b("read_weight")?,
            early_read_weights: b("early_read_weights")?,
            codebook: b("codebook")?,
            gate_bias: f32(&b("gate.bias")?)?,
            packed: concatenate_axis(&[&gate, &predecessor, &prev], 0)?,
            predecessor_gate_f32: f32(&predecessor)?,
            prev_f32: f32(&prev)?,
            last_write_f32: f32(&feature_write_weights.try_index(last)?)?,
            feature_write_weights,
        };
        let draft = Self {
            fc: proj("fc.weight")?,
            hidden_norm: vec("hidden_norm.weight")?,
            norm: vec("norm.weight")?,
            rope: Rope::standard(cfg.head_dim as i32, cfg.rope_theta),
            layers,
            bridge,
            precision,
            cfg,
        };
        crate::primitives::weights::eval_groups(&[draft.arrays()])?;
        Ok(draft)
    }

    fn arrays(&self) -> Vec<Array> {
        let mut out = Vec::new();
        self.fc.push_arrays(&mut out);
        out.extend([self.hidden_norm.clone(), self.norm.clone()]);
        for l in &self.layers {
            for p in [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ] {
                p.push_arrays(&mut out);
            }
            out.extend([
                l.q_norm.clone(),
                l.k_norm.clone(),
                l.input_layernorm.clone(),
                l.post_attention_layernorm.clone(),
            ]);
        }
        let b = &self.bridge;
        out.extend(
            [
                &b.read_weight,
                &b.early_read_weights,
                &b.feature_write_weights,
                &b.codebook,
                &b.gate_bias,
                &b.packed,
                &b.predecessor_gate_f32,
                &b.prev_f32,
                &b.last_write_f32,
            ]
            .map(Array::clone),
        );
        out
    }

    /// The parsed config.
    pub fn config(&self) -> &BlockDraftConfig {
        &self.cfg
    }

    /// The load precision (`None`: dense BF16).
    pub fn precision(&self) -> Option<BlockDraftPrecision> {
        self.precision
    }

    /// An empty context at absolute position `start` (the position of the first row it will be
    /// fed).
    pub fn new_context(&self, start: i32) -> DSpineContext {
        DSpineContext {
            keys: Vec::new(),
            values: Vec::new(),
            next: start,
        }
    }

    fn heads(&self, x: &Array, heads: usize) -> Result<Array> {
        let s = x.shape();
        Ok(x.reshape(&[s[0], s[1], heads as i32, self.cfg.head_dim as i32])?)
    }

    /// Append target rows `taps` `[1, n, tap_width]` — positions `ctx.next..` — to the context,
    /// then drop rows no later block can see.
    pub fn append_context(&self, ctx: &mut DSpineContext, taps: &Array) -> Result<()> {
        let n = taps.shape()[1];
        if n == 0 {
            return Ok(());
        }
        // A row older than the window can never be seen again: feed only the last ones.
        let taps = match self.cfg.context_rows() {
            Some(keep) if n > keep as i32 => rows_from(taps, n - keep as i32, 1)?,
            _ => taps.clone(),
        };
        let fed = taps.shape()[1];
        let skipped = n - fed;
        let x = rms_norm(
            &self.fc.forward(&bf16(&taps)?)?,
            &self.hidden_norm,
            self.cfg.rms_norm_eps,
        )?;
        let (cos, sin) = self
            .rope
            .cos_sin(fed, ctx.next + skipped, Dtype::Bfloat16)?;
        let kv = self.cfg.num_key_value_heads;
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let k = rms_norm(
                &self.heads(&layer.k_proj.forward(&x)?, kv)?,
                &layer.k_norm,
                self.cfg.rms_norm_eps,
            )?;
            let k = apply_rope(&k, &cos, &sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = self
                .heads(&layer.v_proj.forward(&x)?, kv)?
                .transpose_axes(&[0, 2, 1, 3])?;
            let (k, v) = match (ctx.keys.get(i), ctx.values.get(i)) {
                (Some(old_k), Some(old_v)) => (
                    concatenate_axis(&[old_k, &k], 2)?,
                    concatenate_axis(&[old_v, &v], 2)?,
                ),
                _ => (k, v),
            };
            let (k, v) = match self.cfg.context_rows() {
                Some(keep) if k.shape()[2] > keep as i32 => {
                    let start = k.shape()[2] - keep as i32;
                    (rows_from(&k, start, 2)?, rows_from(&v, start, 2)?)
                }
                _ => (k, v),
            };
            keys.push(k);
            values.push(v);
        }
        ctx.keys = keys;
        ctx.values = values;
        ctx.next += n;
        Ok(())
    }

    /// One block forward over `embeds` `[1, T, hidden]` (the target's embedding of `[anchor,
    /// mask × (T - 1)]`) at positions `ctx.next..`, attending to the context and causally within
    /// the block. The context is read, not written: the block's own keys never enter it.
    pub fn block(&self, ctx: &DSpineContext, embeds: &Array, anchor: i32) -> Result<DSpineBlock> {
        let t = embeds.shape()[1];
        let (heads, kv) = (self.cfg.num_attention_heads, self.cfg.num_key_value_heads);
        let (cos, sin) = self.rope.cos_sin(t, ctx.next, Dtype::Bfloat16)?;
        let scale = (self.cfg.head_dim as f32).powf(-0.5);
        let keys_total = ctx.rows() + t;
        let mask = match self.cfg.sliding_window {
            Some(w) if keys_total > w as i32 => AttnMask::SlidingCausal { window: w as i32 },
            _ => AttnMask::Causal,
        };
        let sqrt_c = (self.cfg.code_size as f32).sqrt();
        let anchor_code =
            bf16(&(f32(&self.bridge.codebook.try_index(anchor)?)? * Array::from_f32(sqrt_c)))?
                .reshape(&[1, 1, -1])?;

        let mut residual = bf16(embeds)?;
        let mut x_in = rms_norm(
            &residual,
            &self.layers[0].input_layernorm,
            self.cfg.rms_norm_eps,
        )?;
        let last = self.layers.len() - 1;
        let mut stash = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let q = rms_norm(
                &self.heads(&layer.q_proj.forward(&x_in)?, heads)?,
                &layer.q_norm,
                self.cfg.rms_norm_eps,
            )?;
            let k = rms_norm(
                &self.heads(&layer.k_proj.forward(&x_in)?, kv)?,
                &layer.k_norm,
                self.cfg.rms_norm_eps,
            )?;
            let v = self.heads(&layer.v_proj.forward(&x_in)?, kv)?;
            let q = apply_rope(&q, &cos, &sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
            let k = apply_rope(&k, &cos, &sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = v.transpose_axes(&[0, 2, 1, 3])?;
            let (k, v) = match (ctx.keys.get(i), ctx.values.get(i)) {
                (Some(ck), Some(cv)) => (
                    concatenate_axis(&[ck, &k], 2)?,
                    concatenate_axis(&[cv, &v], 2)?,
                ),
                _ => (k, v),
            };
            let attn = sdpa(&q, &k, &v, scale, mask)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, t, -1])?;
            let h = add(&residual, &layer.o_proj.forward(&attn)?)?;
            let normed = rms_norm(&h, &layer.post_attention_layernorm, self.cfg.rms_norm_eps)?;
            let mlp = layer.down_proj.forward(&multiply(
                &silu(&layer.gate_proj.forward(&normed)?)?,
                &layer.up_proj.forward(&normed)?,
            )?)?;
            let out = add(&h, &mlp)?;
            let next_norm = if i < last {
                &self.layers[i + 1].input_layernorm
            } else {
                &self.norm
            };
            let (normed, written, pre) = self.write(&out, i, next_norm, &anchor_code)?;
            if i == last {
                stash = Some(pre);
            }
            x_in = normed;
            residual = written;
        }
        let (raw, rms, query) = stash.ok_or_else(|| Error::Msg("DSpine: no draft layer".into()))?;
        Ok(DSpineBlock {
            rows: rows_from(&x_in, 1, 1)?.reshape(&[t - 1, -1])?,
            raw: rows_from(&raw, 1, 1)?.reshape(&[t - 1, -1])?,
            rms: rows_from(&rms, 1, 1)?.reshape(&[t - 1])?,
            query: rows_from(&query, 1, 1)?.reshape(&[t - 1, -1])?,
        })
    }

    /// One code-bridge write after `layer` (the reference's `DSpineCodeBridge.__call__`, rounding
    /// point by rounding point): returns `(next_norm(out), out, (raw f32, rms, query f32))`.
    fn write(
        &self,
        hidden: &Array,
        layer: usize,
        next_norm: &Array,
        anchor_code: &Array,
    ) -> Result<(Array, Array, (Array, Array, Array))> {
        let b = &self.bridge;
        let t = hidden.shape()[1];
        let r = self.cfg.bridge_rank as i32;
        let eps = Array::from_f32(self.cfg.bridge_eps);
        let sqrt_c = Array::from_f32((self.cfg.code_size as f32).sqrt());
        let f = f32(&bf16(hidden)?)?;
        let rms = sqrt(&maximum(
            &mean_axis(&f.square()?, -1, None)?,
            Array::from_f32(F32_TINY),
        )?)?;
        let rms_col = rms.expand_dims(-1)?;
        let normalized = bf16(&f.divide(&add(&rms_col, &eps)?)?)?;
        let read = if layer + 1 == self.layers.len() {
            b.read_weight.clone()
        } else {
            b.early_read_weights.try_index(layer as i32)?
        };
        let projected = f32(&matmul(&normalized, read.t())?)?;
        let norm = sqrt(&sum_axis(&projected.square()?, -1, true)?)?;
        let codes = bf16(&multiply(&sqrt_c, &projected)?.divide(&add(&norm, &eps)?)?)?;
        let codes = concatenate_axis(&[anchor_code, &rows_from(&codes, 1, 1)?], 1)?;
        let packed = matmul(&codes, b.packed.t())?;
        let packed_f = f32(&packed)?;
        let part = |i: i32| -> Result<Array> {
            let idx = Array::arange::<i32, i32>(i * r, (i + 1) * r, None)?;
            Ok(packed_f.take_axis(&idx, 2)?)
        };
        let (q, s, v) = (part(0)?, part(1)?, part(2)?);
        let prefix = |a: &Array| -> Result<Array> {
            let idx = Array::arange::<i32, i32>(0, t - 1, None)?;
            Ok(a.take_axis(&idx, 1)?)
        };
        let gate = sigmoid(&add(
            &add(&rows_from(&q, 1, 1)?, &prefix(&s)?)?,
            &b.gate_bias,
        )?)?;
        let message = bf16(&concatenate_axis(
            &[
                &mlx_rs::ops::zeros::<f32>(&[1, 1, r])?,
                &multiply(&gate, &prefix(&v)?)?,
            ],
            1,
        )?)?;
        let write = b.feature_write_weights.try_index(layer as i32)?;
        let delta = f32(&matmul(&message, write.t())?)?;
        let mut not_anchor = vec![1.0f32; t as usize];
        not_anchor[0] = 0.0;
        let not_anchor = Array::from_slice(&not_anchor, &[1, t, 1]);
        let out = bf16(&add(
            &f,
            &multiply(&multiply(&rms_col, &delta)?, &not_anchor)?,
        )?)?;
        let x = f32(&out)?;
        let y = bf16(&multiply(
            &x,
            &mlx_rs::ops::rsqrt(&add(
                &mean_axis(&x.square()?, -1, true)?,
                Array::from_f32(self.cfg.rms_norm_eps),
            )?)?,
        )?)?;
        let normed = bf16(&multiply(&f32(&y)?, &f32(next_norm)?)?)?;
        Ok((normed, out, (f, rms, q)))
    }

    /// The exact-sender refinement table for `block` (the reference's `sampled_table`): each
    /// draft row's top-`K` candidate ids under the target's LM head (`head`: rows → logits),
    /// ascending, `[n, K]` i32, and `[n, K, K]` f32 scores — row `i`'s candidates rescored for
    /// each candidate sender of row `i - 1` (row 0's sender is the anchor). `head_rows` gathers
    /// the LM head's dense rows for ids.
    pub fn table(
        &self,
        block: &DSpineBlock,
        anchor: i32,
        head: &dyn Fn(&Array) -> Result<Array>,
        head_rows: &dyn Fn(&Array) -> Result<Array>,
    ) -> Result<(Array, Array)> {
        let b = &self.bridge;
        let k = self.cfg.head_topk as i32;
        let n = block.rows.shape()[0];
        let logits = f32(&head(&block.rows)?)?;
        let width = logits.shape()[1];
        let k = k.min(width);
        let top = argpartition_axis(&logits.negative()?, k - 1, -1)?;
        let cand = sort_axis(
            &top.take_axis(&Array::arange::<i32, i32>(0, k, None)?, 1)?,
            -1,
        )?
        .as_dtype(Dtype::Int32)?;
        let anchor_row = mlx_rs::ops::full::<i32>(&[1, k], Array::from_int(anchor))?;
        let senders = if n > 1 {
            let prev = cand.take_axis(&Array::arange::<i32, i32>(0, n - 1, None)?, 0)?;
            concatenate_axis(&[&anchor_row, &prev], 0)?
        } else {
            anchor_row
        };
        let sqrt_c = Array::from_f32((self.cfg.code_size as f32).sqrt());
        let e = multiply(
            &f32(&b.codebook.take_axis(&senders.reshape(&[-1])?, 0)?)?,
            &sqrt_c,
        )?
        .reshape(&[n, k, -1])?;
        let gate = sigmoid(&add(
            &add(
                &block.query.expand_dims(1)?,
                &matmul(&e, b.predecessor_gate_f32.t())?,
            )?,
            &b.gate_bias,
        )?)?;
        let value = matmul(&e, b.prev_f32.t())?;
        let delta = matmul(&multiply(&gate, &value)?, b.last_write_f32.t())?;
        let refined = add(
            &block.raw.expand_dims(1)?,
            &multiply(&block.rms.reshape(&[n, 1, 1])?, &delta)?,
        )?;
        let norm = sqrt(&add(
            &mean_axis(&refined.square()?, -1, true)?,
            Array::from_f32(self.cfg.rms_norm_eps),
        )?)?;
        let normalized = multiply(&refined.divide(&norm)?, &f32(&self.norm)?)?;
        let rows = f32(&head_rows(&cand)?)?;
        let table = matmul(&normalized, &rows.swap_axes(1, 2)?)?;
        Ok((cand, table))
    }
}

#[cfg(test)]
pub(crate) mod tests;
