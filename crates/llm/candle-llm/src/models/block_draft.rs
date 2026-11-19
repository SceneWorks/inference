//! Block drafts on Candle (epic SW-F360, story SW-7197) — the port of mlx-llm's
//! `models::block_draft`, `models::dspine` and `models::dflash2`: what every format shares (the
//! draft's context over the target's tapped rows, a Qwen3 draft layer, the group-scoped load) and
//! [`BlockDraft`], the format a load resolved from the checkpoint's `architectures`.
//!
//! Both formats project the target's tapped rows with `fc` + `hidden_norm` into the draft's
//! **context**, whose per-layer roped keys and values the draft keeps over its sliding window
//! (E5: DSpine-b16 8,192 rows, DFlash2 2,048), and run a block `[anchor, mask × …]`, embedded by
//! the target, through Qwen3 layers attending to the context and to the block. They differ in how
//! a block's rows become per-row candidates and predecessor-conditioned scores: DSpine's code
//! bridge ([`DSpineDraft`]) and DFlash2's dynamic convolutions and candidate-path selector
//! ([`DFlash2Draft`]). Either way a proposal is each scored row's top-`K` candidate ids
//! (ascending) and the `[rows, K, K]` score table — row `i`'s candidates scored for each candidate
//! sender of row `i - 1` — the proposer walks (and a tree proposal, SW-7198, branches over).
//!
//! **Load** (E8): the config is checked against the target and the stored tensors before anything
//! is built (the shared [`BlockDraftConfig`] rules). The load then runs group by group
//! ([`BlockDraftConfig::load_groups`], the structure [`BlockDraftConfig::load_bytes`] prices):
//! each group's tensors are read from the memory-mapped checkpoint one at a time, converted —
//! a quantized projection is quantized on the host and placed on the device as GGML blocks (8-bit
//! `Q8_0` by default, 4-bit `Q4_K` on request: the Candle tiers' formats), everything else cast to
//! the compute dtype — and their sources released before the next group reads its own, so the
//! device never holds the BF16 payload beside the converted draft.
//!
//! **Scores** are f32. Where MLX gathers the LM head's dense rows for DSpine's refinement, Candle
//! runs the target's head itself over the refined rows (its head may be GGML-packed, without dense
//! rows) and gathers the candidates' logits: the verifier commits only the target's tokens, so the
//! draft's rounding moves acceptance, never the output.

use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::sigmoid;

use core_llm::{
    BlockDraftConfig, BlockDraftFormat, BlockDraftHead, BlockDraftLoadGroup, BlockDraftPrecision,
    BlockDraftRefusal, DFLASH2_CONVS,
};

use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa_gqa, AttnMask, MASK_NEG};
use crate::primitives::nn::{rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};

/// `torch.finfo(torch.float32).tiny`, the clamp floor of DSpine's bridge read RMS.
const F32_TINY: f64 = 1.175_494_4e-38;

/// `a @ wᵀ` for `a` `[.., k]` and `w` `[n, k]`, as one 2-D matmul.
fn mm(a: &Tensor, w: &Tensor) -> Result<Tensor> {
    let mut dims = a.dims().to_vec();
    let k = dims.pop().unwrap_or(1);
    let out = a.reshape(((), k))?.matmul(&w.t()?)?;
    dims.push(w.dim(0)?);
    Ok(out.reshape(dims)?)
}

fn f32(a: &Tensor) -> Result<Tensor> {
    Ok(a.to_dtype(DType::F32)?)
}

/// Rows `start..` of `a` along `dim`.
fn rows_from(a: &Tensor, start: usize, dim: usize) -> Result<Tensor> {
    let n = a.dim(dim)?;
    Ok(a.narrow(dim, start, n - start)?)
}

/// The block's GGML format at `precision` (`None`: dense).
fn quant_spec(precision: Option<BlockDraftPrecision>) -> Option<QuantSpec> {
    precision.map(|p| match p {
        BlockDraftPrecision::Q8 => QuantSpec::q8(),
        BlockDraftPrecision::Q4 => QuantSpec::q4(),
    })
}

/// Where a draft's stored tensors come from: the memory-mapped checkpoint (each read on its own,
/// on the host) or an in-memory map (the tiny fixtures).
pub enum DraftSource {
    /// The checkpoint's `*.safetensors` shards, memory-mapped.
    Disk(candle_core::safetensors::MmapedSafetensors),
    /// Stored tensors already in memory.
    Map(HashMap<String, Tensor>),
}

impl DraftSource {
    /// Map every `*.safetensors` shard in `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort();
        // SAFETY: the shards are read-only checkpoint files the load does not modify while mapped.
        let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&shards)? };
        #[cfg(feature = "model-inference-proof")]
        crate::model_input_proof::record_reader_paths(
            "candle-llm::models::block_draft::BlockDraft::load",
            crate::model_input_proof::InputRole::ModelWeights,
            &shards,
        );
        Ok(Self::Disk(mapped))
    }

    fn names(&self) -> Vec<String> {
        match self {
            Self::Disk(m) => m.tensors().into_iter().map(|(n, _)| n).collect(),
            Self::Map(m) => m.keys().cloned().collect(),
        }
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match self {
            Self::Disk(m) => m.get(name).ok().map(|v| v.shape().to_vec()),
            Self::Map(m) => m.get(name).map(|t| t.dims().to_vec()),
        }
    }

    /// `name` as stored, on the host.
    fn host(&self, name: &str) -> Result<Tensor> {
        match self {
            Self::Disk(m) => Ok(m.load(name, &Device::Cpu)?),
            Self::Map(m) => Ok(m
                .get(name)
                .ok_or_else(|| Error::Msg(format!("missing tensor `{name}`")))?
                .to_device(&Device::Cpu)?),
        }
    }
}

/// One load group's view of the source ([`BlockDraftConfig::load_groups`]): a read outside the
/// group's tensors is an error, so the draft converts exactly what the group was priced for. Each
/// read is converted at once; its host source is dropped with the read.
pub(crate) struct GroupReader<'a> {
    pub(crate) source: &'a DraftSource,
    pub(crate) tensors: &'a [String],
    pub(crate) device: &'a Device,
    pub(crate) dtype: DType,
}

impl<'a> GroupReader<'a> {
    fn of(
        source: &'a DraftSource,
        group: &'a BlockDraftLoadGroup,
        device: &'a Device,
        dtype: DType,
    ) -> Self {
        Self {
            source,
            tensors: &group.tensors,
            device,
            dtype,
        }
    }

    fn host(&self, name: &str) -> Result<Tensor> {
        if !self.tensors.iter().any(|t| t == name) {
            return Err(Error::Msg(format!(
                "`{name}` is read outside its priced load group"
            )));
        }
        self.source.host(name)
    }

    /// `name` in the compute dtype on the device (a norm, a codebook, a base kernel).
    pub(crate) fn dense(&self, name: &str) -> Result<Tensor> {
        Ok(self
            .host(name)?
            .to_dtype(self.dtype)?
            .to_device(self.device)?)
    }

    /// The `[out, in]` matrix `name` as a projection: quantized on the host and placed on the
    /// device as GGML blocks under `quant`, else dense in the compute dtype.
    pub(crate) fn proj(&self, name: &str, quant: Option<QuantSpec>) -> Result<Projection> {
        match quant {
            None => Projection::load(self.dense(name)?, None),
            Some(q) => {
                let source = f32(&self.host(name)?)?;
                let in_dim = source.dim(1)?;
                let qt = QTensor::quantize_onto(&source, q.dtype_for_in_dim(in_dim), self.device)?;
                Projection::load_qtensor(qt, None)
            }
        }
    }
}

/// One draft decoder layer's Qwen3 attention (with q/k norms) and SwiGLU MLP.
pub(crate) struct DraftLayer {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
    pub(crate) input_layernorm: Tensor,
    pub(crate) post_attention_layernorm: Tensor,
}

impl DraftLayer {
    /// Layer `i` from its group.
    pub(crate) fn load(r: &GroupReader<'_>, i: usize, quant: Option<QuantSpec>) -> Result<Self> {
        let proj = |s: &str| r.proj(&format!("layers.{i}.{s}"), quant);
        let vec = |s: &str| r.dense(&format!("layers.{i}.{s}"));
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

    pub(crate) fn is_quantized(&self) -> bool {
        self.q_proj.is_quantized() && self.down_proj.is_quantized()
    }

    /// Roped, normalized keys and the values of rows `x` `[1, S, hidden]`, `[1, kv, S, head_dim]`.
    fn key_values(
        &self,
        cfg: &BlockDraftConfig,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (kv, hd, s) = (cfg.num_key_value_heads, cfg.head_dim, x.dim(1)?);
        let eps = f64::from(cfg.rms_norm_eps);
        let k = rms_norm(
            &self.k_proj.forward(x)?.reshape((1, s, kv, hd))?,
            &self.k_norm,
            eps,
        )?;
        let k = apply_rope(&k, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((1, s, kv, hd))?;
        Ok((k, v.transpose(1, 2)?.contiguous()?))
    }

    /// The block's attention output `o_proj(attn)` `[1, T, hidden]` for block rows `x` roped by
    /// (`cos`, `sin`): the block's keys and values appended to this layer's context.
    pub(crate) fn attend(
        &self,
        cfg: &BlockDraftConfig,
        x: &Tensor,
        context: Option<(&Tensor, &Tensor)>,
        rope: (&Tensor, &Tensor),
        mask: AttnMask<'_>,
    ) -> Result<Tensor> {
        let (cos, sin) = rope;
        let (nq, hd, t) = (cfg.num_attention_heads, cfg.head_dim, x.dim(1)?);
        let q = self.q_proj.forward(x)?.reshape((1, t, nq, hd))?;
        let q = rms_norm(&q, &self.q_norm, f64::from(cfg.rms_norm_eps))?;
        let q = apply_rope(&q, cos, sin, false)?
            .transpose(1, 2)?
            .contiguous()?;
        let (k, v) = self.key_values(cfg, x, cos, sin)?;
        let (k, v) = match context {
            Some((ck, cv)) => (Tensor::cat(&[ck, &k], 2)?, Tensor::cat(&[cv, &v], 2)?),
            None => (k, v),
        };
        let scale = (hd as f32).powf(-0.5);
        let attn = sdpa_gqa(&q, &k, &v, scale, mask)?
            .transpose(1, 2)?
            .reshape((1, t, nq * hd))?;
        self.o_proj.forward(&attn)
    }

    /// The SwiGLU MLP over already-normalized rows.
    pub(crate) fn mlp(&self, normed: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.gate_proj.forward(normed)?)?;
        self.down_proj
            .forward(&(gate * self.up_proj.forward(normed)?)?)
    }
}

/// The draft's context: per layer, the roped keys and the values of the target rows it has read,
/// at most [`BlockDraftConfig::context_rows`] of them (E5).
#[derive(Clone, Debug, Default)]
pub struct BlockDraftContext {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    /// The absolute position the next context row (and so the next block's anchor) occupies.
    next: i32,
}

impl BlockDraftContext {
    /// An empty context at absolute position `start` (the position of the first row it is fed).
    pub fn new(start: i32) -> Self {
        Self {
            next: start,
            ..Self::default()
        }
    }

    /// The absolute position of the next row — the next block's anchor.
    pub fn next_position(&self) -> i32 {
        self.next
    }

    /// Context rows held (bounded by the window).
    pub fn rows(&self) -> usize {
        self.keys.first().map_or(0, |k| k.dim(2).unwrap_or(0))
    }

    /// Every layer's keys then values (equivalence tests compare these).
    pub fn tensors(&self) -> impl Iterator<Item = &Tensor> {
        self.keys.iter().chain(&self.values)
    }

    fn layer(&self, i: usize) -> Option<(&Tensor, &Tensor)> {
        self.keys.get(i).zip(self.values.get(i))
    }
}

/// What both formats load identically: `fc`, the two outer norms and the RoPE table.
pub(crate) struct DraftBody {
    pub(crate) cfg: BlockDraftConfig,
    pub(crate) precision: Option<BlockDraftPrecision>,
    pub(crate) fc: Projection,
    pub(crate) hidden_norm: Tensor,
    pub(crate) norm: Tensor,
    pub(crate) rope: Rope,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
}

impl DraftBody {
    /// `fc` and the outer norms from the head group `r`.
    pub(crate) fn load(
        r: &GroupReader<'_>,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
    ) -> Result<Self> {
        Ok(Self {
            fc: r.proj("fc.weight", quant_spec(precision))?,
            hidden_norm: r.dense("hidden_norm.weight")?,
            norm: r.dense("norm.weight")?,
            rope: Rope::standard(cfg.head_dim as i32, cfg.rope_theta),
            device: r.device.clone(),
            dtype: r.dtype,
            precision,
            cfg,
        })
    }

    fn eps(&self) -> f64 {
        f64::from(self.cfg.rms_norm_eps)
    }

    /// `(cos, sin)` for `n` positions from `start`.
    pub(crate) fn cos_sin(&self, n: usize, start: i32) -> Result<(Tensor, Tensor)> {
        self.rope.cos_sin(n as i32, start, self.dtype, &self.device)
    }

    /// Append target rows `taps` `[1, n, tap_width]` — positions `ctx.next..` — to the context
    /// through `layers`' key/value side, then drop rows no later block can see.
    pub(crate) fn append(
        &self,
        layers: &[&DraftLayer],
        ctx: &mut BlockDraftContext,
        taps: &Tensor,
    ) -> Result<()> {
        let n = taps.dim(1)?;
        if n == 0 {
            return Ok(());
        }
        // A row older than the window can never be seen again: feed only the last ones.
        let keep = self.cfg.context_rows();
        let skipped = keep.map_or(0, |k| n.saturating_sub(k));
        let taps = rows_from(taps, skipped, 1)?.to_dtype(self.dtype)?;
        let x = rms_norm(&self.fc.forward(&taps)?, &self.hidden_norm, self.eps())?;
        let (cos, sin) = self.cos_sin(n - skipped, ctx.next + skipped as i32)?;
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for (i, layer) in layers.iter().enumerate() {
            let (k, v) = layer.key_values(&self.cfg, &x, &cos, &sin)?;
            let (k, v) = match ctx.layer(i) {
                Some((ok, ov)) => (Tensor::cat(&[ok, &k], 2)?, Tensor::cat(&[ov, &v], 2)?),
                None => (k, v),
            };
            let held = k.dim(2)?;
            let (k, v) = match keep {
                Some(keep) if held > keep => (
                    k.narrow(2, held - keep, keep)?.contiguous()?,
                    v.narrow(2, held - keep, keep)?.contiguous()?,
                ),
                _ => (k, v),
            };
            keys.push(k);
            values.push(v);
        }
        ctx.keys = keys;
        ctx.values = values;
        ctx.next += n as i32;
        Ok(())
    }

    /// The attention mask of a `t`-row block at `ctx.next..` over the context's rows then the
    /// block's: a block row at position `p` sees a context row at `c` while `p - c` is inside the
    /// window, and every block row (non-causal) or the block rows up to itself (causal). Built in
    /// the compute dtype — the queries' — on the device; `None` when it masks nothing.
    pub(crate) fn mask(
        &self,
        ctx: &BlockDraftContext,
        t: usize,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        let rows = ctx.rows();
        let window = self.cfg.sliding_window;
        let sees = |i: usize, j: usize| window.is_none_or(|w| i + rows - j < w);
        if !causal && (rows == 0 || sees(t - 1, 0)) {
            return Ok(None);
        }
        let k = rows + t;
        let mut data = vec![0f32; t * k];
        for i in 0..t {
            for j in 0..k {
                let keep = if j < rows {
                    sees(i, j)
                } else {
                    !causal || j - rows <= i
                };
                if !keep {
                    data[i * k + j] = MASK_NEG;
                }
            }
        }
        Ok(Some(
            Tensor::from_vec(data, (1, 1, t, k), &self.device)?.to_dtype(self.dtype)?,
        ))
    }
}

/// The top-`k` ids of each row of `logits` `[n, V]` (f32, on the device), ascending per row:
/// `k` device argmaxes, each masking its winner, then one transfer of the `[n, k]` ids. Returns
/// the ids on the host and as a `[n, k]` `u32` tensor on the device.
fn top_k_ascending(logits: &Tensor, k: usize) -> Result<(Vec<Vec<i32>>, Tensor)> {
    let (n, width) = logits.dims2()?;
    let k = k.min(width);
    let device = logits.device();
    let columns = Tensor::arange(0u32, width as u32, device)?.unsqueeze(0)?;
    let blocked = Tensor::full(f32::NEG_INFINITY, (n, width), device)?;
    let mut rest = logits.clone();
    let mut picks = Vec::with_capacity(k);
    for _ in 0..k {
        let best = rest.argmax_keepdim(1)?; // [n, 1] u32
        let hit = columns.broadcast_eq(&best)?;
        rest = hit.where_cond(&blocked, &rest)?;
        picks.push(best);
    }
    let ids = Tensor::cat(&picks, 1)?.to_vec2::<u32>()?;
    let rows: Vec<Vec<i32>> = ids
        .into_iter()
        .map(|mut row| {
            row.sort_unstable();
            row.into_iter().map(|t| t as i32).collect()
        })
        .collect();
    let flat: Vec<u32> = rows.iter().flatten().map(|&t| t as u32).collect();
    let on_device = Tensor::from_vec(flat, (n, k), device)?;
    Ok((rows, on_device))
}

/// The ids every scored row's candidates are scored against: the anchor for row 0, row `i - 1`'s
/// candidates for row `i` — `[n · k]` `u32` on the device.
fn senders(anchor: i32, cand: &Tensor) -> Result<Tensor> {
    let (n, k) = cand.dims2()?;
    let anchor = Tensor::full(anchor as u32, (1, k), cand.device())?;
    let senders = if n > 1 {
        Tensor::cat(&[&anchor, &cand.narrow(0, 0, n - 1)?], 0)?
    } else {
        anchor
    };
    Ok(senders.flatten_all()?)
}

/// A proposal's per-row candidates (ascending) and its `[rows, K, K]` row-major score table.
pub type BlockProposal = (Vec<Vec<i32>>, Vec<f32>);

/// The code bridge's tensors (never quantized: the refinement depends on their rounding), plus
/// the f32 copies the refinement computes with.
struct Bridge {
    read_weight: Tensor,
    early_read_weights: Tensor,
    feature_write_weights: Tensor,
    codebook: Tensor,
    gate_bias: Tensor,
    /// `[3R, C]` = `[gate; predecessor_gate; prev]`: one matmul yields a row's query, sender and
    /// value parts.
    packed: Tensor,
    predecessor_gate_f32: Tensor,
    prev_f32: Tensor,
    last_write_f32: Tensor,
}

/// A loaded DSpine draft (Tencent's `DSpineDraftModel`): a causal block whose every layer carries
/// a code-bridge write from each row's predecessor, the last write redone per candidate sender.
pub struct DSpineDraft {
    body: DraftBody,
    layers: Vec<DraftLayer>,
    bridge: Bridge,
    code_size: usize,
    bridge_rank: usize,
    bridge_eps: f64,
}

/// One DSpine block forward's outputs: the final-normalized draft rows, and the last write's
/// pre-write state for those rows, which the refinement redoes per sender.
struct DSpineBlock {
    rows: Tensor,
    raw: Tensor,
    rms: Tensor,
    query: Tensor,
}

impl DSpineDraft {
    /// Check the version markers and the disabled terminal projection (the checks group, read and
    /// dropped before anything is built), then build from the remaining groups.
    fn load(
        source: &DraftSource,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
        groups: &[BlockDraftLoadGroup],
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let tensors = |e: Error| BlockDraftRefusal::Tensors(e.to_string());
        let (checks, built) = groups
            .split_first()
            .ok_or_else(|| BlockDraftRefusal::Tensors("a DSpine load has a checks group".into()))?;
        let reader = GroupReader {
            source,
            tensors: &checks.tensors,
            device: &Device::Cpu,
            dtype: DType::F32,
        };
        let (markers, out) = checks.tensors.split_at(checks.tensors.len() - 1);
        for marker in markers {
            let version = reader
                .host(marker)
                .and_then(|t| Ok(t.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?))
                .map_err(tensors)?;
            if version != [1] {
                return Err(BlockDraftRefusal::Tensors(format!(
                    "`{marker}` is not version 1"
                )));
            }
        }
        let peak = reader
            .host(&out[0])
            .and_then(|t| Ok(f32(&t)?.abs()?.max_all()?.to_scalar::<f32>()?))
            .map_err(tensors)?;
        if peak != 0.0 {
            return Err(BlockDraftRefusal::Tensors(
                "the terminal-disabled checkpoint has a nonzero terminal projection".into(),
            ));
        }
        Self::build(source, cfg, precision, built, device, dtype).map_err(tensors)
    }

    fn build(
        source: &DraftSource,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
        groups: &[BlockDraftLoadGroup],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let BlockDraftHead::DSpine {
            code_size,
            bridge_rank,
            bridge_eps,
        } = cfg.head
        else {
            return Err(Error::Msg("not a DSpine config".into()));
        };
        let reader = |g| GroupReader::of(source, g, device, dtype);
        let (head, layer_groups) = groups
            .split_last()
            .ok_or_else(|| Error::Msg("a DSpine load has a head group".into()))?;
        let layers = layer_groups
            .iter()
            .enumerate()
            .map(|(i, g)| DraftLayer::load(&reader(g), i, quant_spec(precision)))
            .collect::<Result<Vec<_>>>()?;
        if layers.len() != cfg.num_hidden_layers {
            return Err(Error::Msg(format!(
                "{} load groups for {} layers",
                layers.len(),
                cfg.num_hidden_layers
            )));
        }
        let r = reader(head);
        let b = |s: &str| r.dense(&format!("selected_prefix_bridge.{s}"));
        let (gate, predecessor, prev) = (
            b("gate.weight")?,
            b("predecessor_gate.weight")?,
            b("prev.weight")?,
        );
        let feature_write_weights = b("feature_write_weights")?;
        let bridge = Bridge {
            read_weight: b("read_weight")?,
            early_read_weights: b("early_read_weights")?,
            codebook: b("codebook")?,
            gate_bias: f32(&b("gate.bias")?)?,
            packed: Tensor::cat(&[&gate, &predecessor, &prev], 0)?,
            predecessor_gate_f32: f32(&predecessor)?,
            prev_f32: f32(&prev)?,
            last_write_f32: f32(&feature_write_weights.get(layers.len() - 1)?)?,
            feature_write_weights,
        };
        Ok(Self {
            body: DraftBody::load(&r, cfg, precision)?,
            layers,
            bridge,
            code_size,
            bridge_rank,
            bridge_eps: f64::from(bridge_eps),
        })
    }

    /// One block forward over `embeds` `[1, T, hidden]` at positions `ctx.next..`, attending to
    /// the context and causally within the block. The context is read, not written.
    fn block(&self, ctx: &BlockDraftContext, embeds: &Tensor, anchor: i32) -> Result<DSpineBlock> {
        let body = &self.body;
        let t = embeds.dim(1)?;
        let (cos, sin) = body.cos_sin(t, ctx.next)?;
        let additive = body.mask(ctx, t, true)?;
        let mask = additive
            .as_ref()
            .map_or(AttnMask::Causal, AttnMask::Additive);
        let sqrt_c = (self.code_size as f64).sqrt();
        let anchor_code = (f32(&self.bridge.codebook.get(anchor as usize)?)? * sqrt_c)?
            .to_dtype(body.dtype)?
            .reshape((1, 1, ()))?;
        let mut residual = embeds.to_dtype(body.dtype)?;
        let mut x_in = rms_norm(&residual, &self.layers[0].input_layernorm, body.eps())?;
        let last = self.layers.len() - 1;
        let mut stash = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let attn = layer.attend(&body.cfg, &x_in, ctx.layer(i), (&cos, &sin), mask)?;
            let h = (&residual + attn)?;
            let normed = rms_norm(&h, &layer.post_attention_layernorm, body.eps())?;
            let out = (&h + layer.mlp(&normed)?)?;
            let next_norm = match self.layers.get(i + 1) {
                Some(next) => &next.input_layernorm,
                None => &body.norm,
            };
            let (normed, written, pre) = self.write(&out, i, next_norm, &anchor_code)?;
            if i == last {
                stash = Some(pre);
            }
            x_in = normed;
            residual = written;
        }
        let (raw, rms, query) = stash.ok_or_else(|| Error::Msg("DSpine: no draft layer".into()))?;
        let tail = |a: &Tensor| -> Result<Tensor> { Ok(rows_from(a, 1, 1)?.squeeze(0)?) };
        Ok(DSpineBlock {
            rows: tail(&x_in)?,
            raw: tail(&raw)?,
            rms: tail(&rms)?,
            query: tail(&query)?,
        })
    }

    /// One code-bridge write after `layer` (the reference's `DSpineCodeBridge.__call__`): returns
    /// `(next_norm(out), out, (raw f32, rms, query f32))`.
    #[allow(clippy::type_complexity)]
    fn write(
        &self,
        hidden: &Tensor,
        layer: usize,
        next_norm: &Tensor,
        anchor_code: &Tensor,
    ) -> Result<(Tensor, Tensor, (Tensor, Tensor, Tensor))> {
        let (b, body) = (&self.bridge, &self.body);
        let (t, r) = (hidden.dim(1)?, self.bridge_rank);
        let sqrt_c = (self.code_size as f64).sqrt();
        let f = f32(hidden)?;
        let rms = f
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .maximum(F32_TINY)?
            .sqrt()?; // [1, T, 1]
        let normalized = f
            .broadcast_div(&(&rms + self.bridge_eps)?)?
            .to_dtype(body.dtype)?;
        let read = if layer + 1 == self.layers.len() {
            b.read_weight.clone()
        } else {
            b.early_read_weights.get(layer)?
        };
        let projected = f32(&mm(&normalized, &read)?)?;
        let norm = projected.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let codes = (projected * sqrt_c)?
            .broadcast_div(&(norm + self.bridge_eps)?)?
            .to_dtype(body.dtype)?;
        let codes = Tensor::cat(&[anchor_code, &rows_from(&codes, 1, 1)?], 1)?;
        let packed = f32(&mm(&codes, &b.packed)?)?;
        let (q, s, v) = (
            packed.narrow(2, 0, r)?,
            packed.narrow(2, r, r)?,
            packed.narrow(2, 2 * r, r)?,
        );
        let gate = sigmoid(
            &(rows_from(&q, 1, 1)? + s.narrow(1, 0, t - 1)?)?.broadcast_add(&b.gate_bias)?,
        )?;
        let zeros = Tensor::zeros((1, 1, r), DType::F32, &body.device)?;
        let message =
            Tensor::cat(&[&zeros, &(gate * v.narrow(1, 0, t - 1)?)?], 1)?.to_dtype(body.dtype)?;
        let write = b.feature_write_weights.get(layer)?;
        let delta = f32(&mm(&message, &write)?)?;
        let mut not_anchor = vec![1f32; t];
        not_anchor[0] = 0.0;
        let not_anchor = Tensor::from_vec(not_anchor, (1, t, 1), &body.device)?;
        let out =
            (&f + delta.broadcast_mul(&rms)?.broadcast_mul(&not_anchor)?)?.to_dtype(body.dtype)?;
        let x = f32(&out)?;
        let y = x
            .broadcast_mul(
                &(x.sqr()?.mean_keepdim(D::Minus1)? + body.eps())?
                    .recip()?
                    .sqrt()?,
            )?
            .to_dtype(body.dtype)?;
        let normed = f32(&y)?
            .broadcast_mul(&f32(next_norm)?)?
            .to_dtype(body.dtype)?;
        Ok((normed, out, (f, rms.squeeze(D::Minus1)?, q)))
    }

    /// The exact-sender refinement table (the reference's `sampled_table`): each draft row's
    /// top-`K` candidates under the target's LM head `head`, and their scores for every candidate
    /// sender of the row before — the refined rows scored by the same head.
    fn table(
        &self,
        block: &DSpineBlock,
        anchor: i32,
        head: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<BlockProposal> {
        let (b, body) = (&self.bridge, &self.body);
        let n = block.rows.dim(0)?;
        let logits = f32(&head(&block.rows)?)?;
        let (candidates, cand) = top_k_ascending(&logits, body.cfg.head_topk)?;
        let k = cand.dim(1)?;
        let sqrt_c = (self.code_size as f64).sqrt();
        let e = (f32(&b.codebook.index_select(&senders(anchor, &cand)?, 0)?)? * sqrt_c)?
            .reshape((n, k, ()))?;
        let gate = sigmoid(
            &mm(&e, &b.predecessor_gate_f32)?
                .broadcast_add(&block.query.unsqueeze(1)?)?
                .broadcast_add(&b.gate_bias)?,
        )?;
        let delta = mm(&(gate * mm(&e, &b.prev_f32)?)?, &b.last_write_f32)?;
        let refined = delta
            .broadcast_mul(&block.rms.reshape((n, 1, 1))?)?
            .broadcast_add(&block.raw.unsqueeze(1)?)?;
        let norm = (refined.sqr()?.mean_keepdim(D::Minus1)? + body.eps())?.sqrt()?;
        let normalized = refined
            .broadcast_div(&norm)?
            .broadcast_mul(&f32(&body.norm)?)?
            .to_dtype(body.dtype)?
            .reshape((n * k, ()))?;
        let refined_logits = f32(&head(&normalized)?)?;
        let index = cand
            .unsqueeze(1)?
            .broadcast_as((n, k, k))?
            .reshape((n * k, k))?
            .contiguous()?;
        let table = refined_logits
            .gather(&index, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((candidates, table))
    }
}

/// One grouped dynamic causal convolution (`GroupedDynamicCausalConv`): a `[2, taps, hidden]`
/// base kernel (stage × tap) and the projection that gives every row its own per-group taps.
struct DynamicConv {
    base_kernel: Tensor,
    kernel_projection: Projection,
}

/// The reference's `_grouped_dynamic_convolve`: `hidden` `[1, T, H]` viewed as `G` groups of
/// `group_size` channels; output row `t` sums, over taps `o`, `(base[o] + dynamic[t, o, g]) ·
/// hidden[t - o]` (zero before the block).
fn convolve(hidden: &Tensor, dynamic: &Tensor, base: &Tensor, group_size: usize) -> Result<Tensor> {
    let (_, t, h) = hidden.dims3()?;
    let groups = h / group_size;
    let taps = base.dim(0)?;
    let blocks = hidden.reshape((1, t, groups, group_size))?;
    let mut out = blocks.zeros_like()?;
    for offset in 0..taps {
        let values = if offset == 0 {
            blocks.clone()
        } else {
            let zeros = Tensor::zeros(
                (1, offset.min(t), groups, group_size),
                hidden.dtype(),
                hidden.device(),
            )?;
            if offset >= t {
                zeros
            } else {
                Tensor::cat(&[&zeros, &blocks.narrow(1, 0, t - offset)?], 1)?
            }
        };
        let kernel = base.get(offset)?.reshape((1, 1, groups, group_size))?;
        out = (out + values.broadcast_mul(&kernel)?)?;
        let tap = dynamic.narrow(2, offset, 1)?.reshape((1, t, groups, 1))?;
        out = (out + values.broadcast_mul(&tap)?)?;
    }
    Ok(out.reshape((1, t, h))?)
}

impl DynamicConv {
    fn load(r: &GroupReader<'_>, prefix: &str, quant: Option<QuantSpec>) -> Result<Self> {
        Ok(Self {
            base_kernel: r.dense(&format!("{prefix}.base_kernel"))?,
            kernel_projection: r.proj(&format!("{prefix}.kernel_projection.weight"), quant)?,
        })
    }

    /// `prepare`: the first-stage convolution of `hidden` and the second stage's taps.
    fn prepare(&self, hidden: &Tensor, taps: usize, group_size: usize) -> Result<(Tensor, Tensor)> {
        let (_, t, h) = hidden.dims3()?;
        let dynamic =
            self.kernel_projection
                .forward(hidden)?
                .reshape((1, t, 2, taps, h / group_size))?;
        let first = dynamic.narrow(2, 0, 1)?.squeeze(2)?;
        let second = dynamic.narrow(2, 1, 1)?.squeeze(2)?;
        let base = self.base_kernel.get(0)?;
        Ok((convolve(hidden, &first, &base, group_size)?, second))
    }

    /// `finish`: the second-stage convolution of the sublayer's output with `prepare`'s taps.
    fn finish(&self, hidden: &Tensor, second: &Tensor, group_size: usize) -> Result<Tensor> {
        convolve(hidden, second, &self.base_kernel.get(1)?, group_size)
    }
}

/// One DFlash2 layer: the Qwen3 body and its two convolutions.
struct DFlash2Layer {
    body: DraftLayer,
    attention_conv: DynamicConv,
    mlp_conv: DynamicConv,
}

/// A loaded DFlash2 draft (Inco's `DFlash2DraftModel`): a parallel block whose layers carry
/// two-tap dynamic convolutions, with a candidate-path selector that scores each row's top-`K`
/// candidates against its predecessor: `unary(c) + Σ_r pred[p]_r · h_i,r · succ[c]_r`.
pub struct DFlash2Draft {
    body: DraftBody,
    layers: Vec<DFlash2Layer>,
    selector: Projection,
    predecessor: Tensor,
    successor: Tensor,
    taps: usize,
    group_size: usize,
    causal: bool,
    embedding_scale: f32,
    output_multiplier: f32,
    softcap: Option<f32>,
}

impl DFlash2Draft {
    fn build(
        source: &DraftSource,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
        groups: &[BlockDraftLoadGroup],
        device: &Device,
        dtype: DType,
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
            return Err(Error::Msg("not a DFlash2 config".into()));
        };
        let quant = quant_spec(precision);
        let reader = |g| GroupReader::of(source, g, device, dtype);
        let (head, layer_groups) = groups
            .split_last()
            .ok_or_else(|| Error::Msg("a DFlash2 load has a head group".into()))?;
        let layers = layer_groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let r = reader(g);
                let [attention, mlp] = DFLASH2_CONVS;
                Ok(DFlash2Layer {
                    body: DraftLayer::load(&r, i, quant)?,
                    attention_conv: DynamicConv::load(
                        &r,
                        &format!("layers.{i}.{attention}"),
                        quant,
                    )?,
                    mlp_conv: DynamicConv::load(&r, &format!("layers.{i}.{mlp}"), quant)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if layers.len() != cfg.num_hidden_layers {
            return Err(Error::Msg(format!(
                "{} load groups for {} layers",
                layers.len(),
                cfg.num_hidden_layers
            )));
        }
        let r = reader(head);
        let s = |n: &str| format!("candidate_selector.{n}");
        Ok(Self {
            selector: r.proj(&s("hidden_projection.weight"), quant)?,
            predecessor: r.dense(&s("predecessor_codebook"))?,
            successor: r.dense(&s("successor_codebook"))?,
            body: DraftBody::load(&r, cfg, precision)?,
            layers,
            taps: conv_kernel_size,
            group_size: conv_group_size,
            causal,
            embedding_scale: input_embedding_scale,
            output_multiplier,
            softcap: final_logit_softcapping,
        })
    }

    /// One parallel block forward over `embeds` `[1, T, hidden]` at positions `ctx.next..`: the
    /// final-normalized draft rows (block rows `1..`) `[T - 1, hidden]`.
    fn block(&self, ctx: &BlockDraftContext, embeds: &Tensor) -> Result<Tensor> {
        let body = &self.body;
        let t = embeds.dim(1)?;
        let (cos, sin) = body.cos_sin(t, ctx.next)?;
        let additive = body.mask(ctx, t, self.causal)?;
        let mask = additive.as_ref().map_or(AttnMask::None, AttnMask::Additive);
        let (taps, groups, eps) = (self.taps, self.group_size, body.eps());
        let mut x = embeds.to_dtype(body.dtype)?;
        if self.embedding_scale != 1.0 {
            x = (x * f64::from(self.embedding_scale))?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm(&x, &layer.body.input_layernorm, eps)?;
            let (h, second) = layer.attention_conv.prepare(&normed, taps, groups)?;
            let attn = layer
                .body
                .attend(&body.cfg, &h, ctx.layer(i), (&cos, &sin), mask)?;
            x = (&x + layer.attention_conv.finish(&attn, &second, groups)?)?;
            let normed = rms_norm(&x, &layer.body.post_attention_layernorm, eps)?;
            let (h, second) = layer.mlp_conv.prepare(&normed, taps, groups)?;
            let mlp = layer.body.mlp(&h)?;
            x = (&x + layer.mlp_conv.finish(&mlp, &second, groups)?)?;
        }
        Ok(rms_norm(&rows_from(&x, 1, 1)?, &body.norm, eps)?.squeeze(0)?)
    }

    /// The candidate-path selector's table for the draft `rows` `[n, hidden]`.
    fn table(
        &self,
        rows: &Tensor,
        anchor: i32,
        head: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<BlockProposal> {
        let n = rows.dim(0)?;
        let mut logits = f32(&head(rows)?)?;
        if self.output_multiplier != 1.0 {
            logits = (logits * f64::from(self.output_multiplier))?;
        }
        if let Some(cap) = self.softcap {
            logits = ((logits / f64::from(cap))?.tanh()? * f64::from(cap))?;
        }
        let (candidates, cand) = top_k_ascending(&logits, self.body.cfg.head_topk)?;
        let k = cand.dim(1)?;
        let unary = logits.gather(&cand, 1)?; // [n, k]
        let gather = |book: &Tensor, ids: &Tensor| -> Result<Tensor> {
            Ok(f32(&book.index_select(&ids.flatten_all()?, 0)?)?.reshape((n, k, ()))?)
        };
        let hidden = f32(&self.selector.forward(rows)?)?.unsqueeze(1)?;
        let weighted =
            gather(&self.predecessor, &senders(anchor, &cand)?)?.broadcast_mul(&hidden)?;
        let edges = weighted.matmul(&gather(&self.successor, &cand)?.transpose(1, 2)?)?;
        let table = edges.broadcast_add(&unary.unsqueeze(1)?)?;
        Ok((candidates, table.flatten_all()?.to_vec1::<f32>()?))
    }
}

/// A loaded block draft in the format its checkpoint's `architectures` names (E2, AC2).
pub enum BlockDraft {
    /// Tencent DSpine (`DSpineDraftModel`).
    DSpine(DSpineDraft),
    /// Inco DFlash2 (`DFlash2DraftModel`).
    DFlash2(DFlash2Draft),
}

impl BlockDraft {
    /// Load the draft in `dir` at `precision` on `device` in `dtype` (the target's), in the
    /// format `cfg` was parsed as. The caller checks the config against its target first
    /// ([`BlockDraftConfig::mismatches`]).
    pub fn load(
        dir: &Path,
        cfg: BlockDraftConfig,
        precision: BlockDraftPrecision,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let source = DraftSource::open(dir)
            .map_err(|e| BlockDraftRefusal::Tensors(format!("read the weights: {e}")))?;
        Self::from_source(&source, cfg, Some(precision), device, dtype)
    }

    /// Build from `source` — quantized at `precision`, or dense for `None` — after checking every
    /// stored tensor ([`BlockDraftConfig::check_tensors`]), group by group.
    pub fn from_source(
        source: &DraftSource,
        cfg: BlockDraftConfig,
        precision: Option<BlockDraftPrecision>,
        device: &Device,
        dtype: DType,
    ) -> std::result::Result<Self, BlockDraftRefusal> {
        let stored = source.names();
        cfg.check_tensors(|name| source.shape(name), stored.iter().map(String::as_str))?;
        let groups = cfg.load_groups();
        match cfg.format {
            BlockDraftFormat::DSpine => {
                DSpineDraft::load(source, cfg, precision, &groups, device, dtype).map(Self::DSpine)
            }
            BlockDraftFormat::DFlash2 => {
                DFlash2Draft::build(source, cfg, precision, &groups, device, dtype)
                    .map(Self::DFlash2)
                    .map_err(|e| BlockDraftRefusal::Tensors(e.to_string()))
            }
        }
    }

    fn body(&self) -> &DraftBody {
        match self {
            Self::DSpine(d) => &d.body,
            Self::DFlash2(d) => &d.body,
        }
    }

    fn layers(&self) -> Vec<&DraftLayer> {
        match self {
            Self::DSpine(d) => d.layers.iter().collect(),
            Self::DFlash2(d) => d.layers.iter().map(|l| &l.body).collect(),
        }
    }

    /// The parsed config.
    pub fn config(&self) -> &BlockDraftConfig {
        &self.body().cfg
    }

    /// The load precision (`None`: dense).
    pub fn precision(&self) -> Option<BlockDraftPrecision> {
        self.body().precision
    }

    /// Whether every projection (`fc` and each layer's) holds GGML blocks.
    pub fn is_quantized(&self) -> bool {
        self.body().fc.is_quantized() && self.layers().iter().all(|l| l.is_quantized())
    }

    /// Whether a block row sees only the block rows before it: DSpine always, DFlash2 when its
    /// checkpoint says `is_causal` (the published ones are non-causal — every row sees the whole
    /// block, so a shorter block changes every row's output).
    pub fn causal(&self) -> bool {
        !matches!(
            self.config().head,
            BlockDraftHead::DFlash2 { causal: false, .. }
        )
    }

    /// An empty context at absolute position `start`.
    pub fn new_context(&self, start: i32) -> BlockDraftContext {
        BlockDraftContext::new(start)
    }

    /// Append target rows `taps` `[1, n, tap_width]` — positions `ctx.next..` — to the context,
    /// then drop rows no later block can see.
    pub fn append_context(&self, ctx: &mut BlockDraftContext, taps: &Tensor) -> Result<()> {
        self.body().append(&self.layers(), ctx, taps)
    }

    /// One proposal for the block `embeds` `[1, T, hidden]` (the target's embedding of
    /// `[anchor, mask × (T - 1)]`), scoring its first `n` draft rows (`1 <= n <= T - 1`; the rest
    /// only condition a non-causal block, and a causal DSpine block scores all `T - 1`): each
    /// scored row's candidates (ascending) and the `[n, K, K]` table. `head` is the target's LM
    /// head over normalized rows.
    pub fn propose(
        &self,
        ctx: &BlockDraftContext,
        embeds: &Tensor,
        n: usize,
        anchor: i32,
        head: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<BlockProposal> {
        let t = embeds.dim(1)?;
        if n == 0 || n >= t {
            return Err(Error::Msg(format!(
                "block draft: {n} rows scored of a {t}-row block"
            )));
        }
        match self {
            Self::DSpine(_) if n != t - 1 => Err(Error::Msg(format!(
                "block draft: a causal draft scores its whole block ({} rows), not {n}",
                t - 1
            ))),
            Self::DSpine(d) => d.table(&d.block(ctx, embeds, anchor)?, anchor, head),
            Self::DFlash2(d) => d.table(&d.block(ctx, embeds)?.narrow(0, 0, n)?, anchor, head),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
