//! What every block-draft format shares on MLX (epic SW-F360: DSpine SW-7193, DFlash2 SW-7194):
//! the draft's context over the target's tapped rows, a Qwen3 draft layer's attention and MLP,
//! the group-scoped reader a priced load builds through, and [`BlockDraft`] — the format a load
//! resolved from the checkpoint's `architectures`.
//!
//! Both formats project the target's tapped rows with `fc` + `hidden_norm` into the draft's
//! **context**, whose per-layer roped keys and values the draft caches over its sliding window
//! (E5), and run a block `[anchor, mask × depth]` through Qwen3 layers whose queries, keys and
//! values come from the block rows. They differ in how a block's rows become candidates and
//! predecessor-conditioned scores: DSpine's code bridge ([`DSpineDraft`]) and DFlash2's dynamic
//! convolutions and candidate-path selector ([`DFlash2Draft`]). Either way a proposal is the same
//! `[rows, K]` candidate ids and `[rows, K, K]` score table the proposer walks.

use mlx_rs::ops::{concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use core_llm::{BlockDraftConfig, BlockDraftFormat, BlockDraftPrecision, BlockDraftRefusal};

use crate::error::{Error, Result};
use crate::models::{DFlash2Draft, DSpineDraft};
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::nn::{rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

pub(crate) fn bf16(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Bfloat16)?)
}

pub(crate) fn f32(a: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Float32)?)
}

/// Rows `start..` of a `[.., S, ..]` array along `axis`.
pub(crate) fn rows_from(a: &Array, start: i32, axis: i32) -> Result<Array> {
    let n = a.shape()[axis as usize];
    let idx = Array::arange::<i32, i32>(start, n, None)?;
    Ok(a.take_axis(&idx, axis)?)
}

/// One load group's view of the weights ([`BlockDraftConfig::load_groups`]): a read outside the
/// group's tensors is an error, so the draft converts exactly what the group was priced for.
pub(crate) struct GroupReader<'a> {
    pub(crate) w: &'a Weights,
    pub(crate) tensors: &'a [String],
}

impl GroupReader<'_> {
    pub(crate) fn require(&self, name: &str) -> Result<&Array> {
        if !self.tensors.iter().any(|t| t == name) {
            return Err(Error::Msg(format!(
                "`{name}` is read outside its priced load group"
            )));
        }
        self.w.require(name)
    }
}

/// One draft decoder layer's Qwen3 attention (with q/k norms) and SwiGLU MLP.
#[derive(Debug)]
pub(crate) struct DraftLayer {
    pub(crate) q_proj: Projection,
    pub(crate) k_proj: Projection,
    pub(crate) v_proj: Projection,
    pub(crate) o_proj: Projection,
    pub(crate) q_norm: Array,
    pub(crate) k_norm: Array,
    pub(crate) gate_proj: Projection,
    pub(crate) up_proj: Projection,
    pub(crate) down_proj: Projection,
    pub(crate) input_layernorm: Array,
    pub(crate) post_attention_layernorm: Array,
}

impl DraftLayer {
    /// Layer `i` from its group (`quant`: the projections' load precision, `None`: dense BF16).
    pub(crate) fn load(r: &GroupReader<'_>, i: usize, quant: Option<QuantSpec>) -> Result<Self> {
        let proj = |s: &str| Projection::load(bf16(r.require(&format!("layers.{i}.{s}"))?)?, quant);
        let vec = |s: &str| bf16(r.require(&format!("layers.{i}.{s}"))?);
        Ok(Self {
            q_proj: proj("self_attn.q_proj.weight")?,
            k_proj: proj("self_attn.k_proj.weight")?,
            v_proj: proj("self_attn.v_proj.weight")?,
            o_proj: proj("self_attn.o_proj.weight")?,
            q_norm: vec("self_attn.q_norm.weight")?,
            k_norm: vec("self_attn.k_norm.weight")?,
            gate_proj: proj("mlp.gate_proj.weight")?,
            up_proj: proj("mlp.up_proj.weight")?,
            down_proj: proj("mlp.down_proj.weight")?,
            input_layernorm: vec("input_layernorm.weight")?,
            post_attention_layernorm: vec("post_attention_layernorm.weight")?,
        })
    }

    /// The layer's arrays, projections first, in a fixed order.
    pub(crate) fn push_arrays(&self, out: &mut Vec<Array>) {
        for p in [
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        ] {
            p.push_arrays(out);
        }
        out.extend([
            self.q_norm.clone(),
            self.k_norm.clone(),
            self.input_layernorm.clone(),
            self.post_attention_layernorm.clone(),
        ]);
    }

    /// The block's attention output `o_proj(attn)` `[1, T, hidden]` for block rows `x`
    /// `[1, T, hidden]` roped by (`cos`, `sin`): queries, keys and values from the block rows, the
    /// block's keys and values appended to the context's `(keys, values)` of this layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend(
        &self,
        cfg: &BlockDraftConfig,
        x: &Array,
        context: Option<(&Array, &Array)>,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let t = x.shape()[1];
        let eps = cfg.rms_norm_eps;
        let q = rms_norm(
            &heads(cfg, &self.q_proj.forward(x)?, cfg.num_attention_heads)?,
            &self.q_norm,
            eps,
        )?;
        let k = rms_norm(
            &heads(cfg, &self.k_proj.forward(x)?, cfg.num_key_value_heads)?,
            &self.k_norm,
            eps,
        )?;
        let v = heads(cfg, &self.v_proj.forward(x)?, cfg.num_key_value_heads)?;
        let q = apply_rope(&q, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let (k, v) = match context {
            Some((ck, cv)) => (
                concatenate_axis(&[ck, &k], 2)?,
                concatenate_axis(&[cv, &v], 2)?,
            ),
            None => (k, v),
        };
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let attn = sdpa(&q, &k, &v, scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[1, t, -1])?;
        self.o_proj.forward(&attn)
    }

    /// The SwiGLU MLP over already-normalized rows.
    pub(crate) fn mlp(&self, normed: &Array) -> Result<Array> {
        self.down_proj.forward(&multiply(
            &silu(&self.gate_proj.forward(normed)?)?,
            &self.up_proj.forward(normed)?,
        )?)
    }
}

/// `[1, S, heads × head_dim]` → `[1, S, heads, head_dim]`.
pub(crate) fn heads(cfg: &BlockDraftConfig, x: &Array, heads: usize) -> Result<Array> {
    let s = x.shape();
    Ok(x.reshape(&[s[0], s[1], heads as i32, cfg.head_dim as i32])?)
}

/// The draft's context: per layer, the roped keys and the values of the target rows it has read,
/// at most [`BlockDraftConfig::context_rows`] of them (E5).
#[derive(Clone, Debug, Default)]
pub struct BlockDraftContext {
    pub(crate) keys: Vec<Array>,
    pub(crate) values: Vec<Array>,
    /// The absolute position the next context row (and so the next block's anchor) occupies.
    pub(crate) next: i32,
}

impl BlockDraftContext {
    /// An empty context at absolute position `start` (the position of the first row it will be
    /// fed).
    pub fn new(start: i32) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            next: start,
        }
    }

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

    /// Layer `i`'s `(keys, values)`, `None` before the first row.
    pub(crate) fn layer(&self, i: usize) -> Option<(&Array, &Array)> {
        self.keys.get(i).zip(self.values.get(i))
    }
}

/// The context-writing half of a draft: `fc`, `hidden_norm` and each layer's key/value side.
pub(crate) struct ContextWriter<'a> {
    pub(crate) cfg: &'a BlockDraftConfig,
    pub(crate) rope: &'a Rope,
    pub(crate) fc: &'a Projection,
    pub(crate) hidden_norm: &'a Array,
    pub(crate) layers: Vec<&'a DraftLayer>,
}

impl ContextWriter<'_> {
    /// Append target rows `taps` `[1, n, tap_width]` — positions `ctx.next..` — to the context,
    /// then drop rows no later block can see.
    pub(crate) fn append(&self, ctx: &mut BlockDraftContext, taps: &Array) -> Result<()> {
        let cfg = self.cfg;
        let n = taps.shape()[1];
        if n == 0 {
            return Ok(());
        }
        // A row older than the window can never be seen again: feed only the last ones.
        let taps = match cfg.context_rows() {
            Some(keep) if n > keep as i32 => rows_from(taps, n - keep as i32, 1)?,
            _ => taps.clone(),
        };
        let fed = taps.shape()[1];
        let skipped = n - fed;
        let x = rms_norm(
            &self.fc.forward(&bf16(&taps)?)?,
            self.hidden_norm,
            cfg.rms_norm_eps,
        )?;
        let (cos, sin) = self
            .rope
            .cos_sin(fed, ctx.next + skipped, Dtype::Bfloat16)?;
        let kv = cfg.num_key_value_heads;
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let k = rms_norm(
                &heads(cfg, &layer.k_proj.forward(&x)?, kv)?,
                &layer.k_norm,
                cfg.rms_norm_eps,
            )?;
            let k = apply_rope(&k, &cos, &sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = heads(cfg, &layer.v_proj.forward(&x)?, kv)?.transpose_axes(&[0, 2, 1, 3])?;
            let (k, v) = match ctx.layer(i) {
                Some((old_k, old_v)) => (
                    concatenate_axis(&[old_k, &k], 2)?,
                    concatenate_axis(&[old_v, &v], 2)?,
                ),
                _ => (k, v),
            };
            let (k, v) = match cfg.context_rows() {
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
}

/// A loaded block draft in the format its checkpoint's `architectures` names (E2, AC2).
#[derive(Debug)]
pub enum BlockDraft {
    /// Tencent DSpine (`DSpineDraftModel`).
    DSpine(DSpineDraft),
    /// Inco DFlash2 (`DFlash2DraftModel`).
    DFlash2(DFlash2Draft),
}

impl BlockDraft {
    /// Load the draft in `dir` at `precision` in the format `cfg` was parsed as. The caller checks
    /// the config against its target first ([`BlockDraftConfig::mismatches`]).
    pub fn load(
        dir: &std::path::Path,
        cfg: BlockDraftConfig,
        precision: BlockDraftPrecision,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        match cfg.format {
            BlockDraftFormat::DSpine => DSpineDraft::load(dir, cfg, precision).map(Self::DSpine),
            BlockDraftFormat::DFlash2 => DFlash2Draft::load(dir, cfg, precision).map(Self::DFlash2),
        }
    }

    /// The parsed config.
    pub fn config(&self) -> &BlockDraftConfig {
        match self {
            Self::DSpine(d) => d.config(),
            Self::DFlash2(d) => d.config(),
        }
    }

    /// The load precision (`None`: dense BF16).
    pub fn precision(&self) -> Option<BlockDraftPrecision> {
        match self {
            Self::DSpine(d) => d.precision(),
            Self::DFlash2(d) => d.precision(),
        }
    }

    /// An empty context at absolute position `start`.
    pub fn new_context(&self, start: i32) -> BlockDraftContext {
        BlockDraftContext::new(start)
    }

    /// Append target rows `taps` `[1, n, tap_width]` to the context ([`ContextWriter::append`]).
    pub fn append_context(&self, ctx: &mut BlockDraftContext, taps: &Array) -> Result<()> {
        match self {
            Self::DSpine(d) => d.append_context(ctx, taps),
            Self::DFlash2(d) => d.append_context(ctx, taps),
        }
    }

    /// One proposal for the block `embeds` `[1, T, hidden]` (the target's embedding of
    /// `[anchor, mask × (T - 1)]`): each draft row's candidate ids, ascending, `[T - 1, K]` i32,
    /// and their `[T - 1, K, K]` f32 scores — row `i`'s candidates scored for each candidate
    /// sender of row `i - 1` (row 0's sender is the anchor, the same for every sender column).
    /// `head` is the target's LM head over normalized rows; `head_rows` gathers its dense rows.
    pub fn propose(
        &self,
        ctx: &BlockDraftContext,
        embeds: &Array,
        anchor: i32,
        head: &dyn Fn(&Array) -> Result<Array>,
        head_rows: &dyn Fn(&Array) -> Result<Array>,
    ) -> Result<(Array, Array)> {
        match self {
            Self::DSpine(d) => d.table(&d.block(ctx, embeds, anchor)?, anchor, head, head_rows),
            Self::DFlash2(d) => d.table(&d.block(ctx, embeds)?, anchor, head),
        }
    }
}
