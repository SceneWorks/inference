//! One `MiniMaxH3TransformerBlock` — pre-norm self-attention and feed-forward, each modulated by
//! AdaLN parameters selected **per row** of the packed sequence.
//!
//! # Single-stream
//!
//! There is no cross-attention and no per-modality block weight anywhere in MiniMax-H3. Video,
//! audio and text rows share one stack and one attention document. Modality enters only through
//! the AdaLN row index, so a Wan/LTX-style dual-stream block is the wrong shape entirely.
//!
//! # The AdaLN row layout is the trap
//!
//! `adaln_proj` emits `6 · 3 · hidden_size` columns from `time_embed_dim`, and the reference then
//! does
//!
//! ```text
//! temb = linear(silu(temb))          # [T, 18·hidden]
//! temb = temb.view(-1, 6·hidden)     # [T·3, 6·hidden]      <- modality becomes a ROW axis
//! shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb.chunk(6, dim=-1)
//! ```
//!
//! so each of the six is `[T·3, hidden]` with rows ordered `[t0_mod0, t0_mod1, t0_mod2, t1_mod0,
//! …]` — which is what `timestep_index · 3 + token_tag` addresses. Chunking the `[T, 18·hidden]`
//! tensor into six `[T, 3·hidden]` pieces instead produces six tensors of the right total size and
//! the wrong contents; [`AdaLnProjection::forward`] does the reshape first, and
//! `tests/dit_parity.rs` mutates it to prove the difference is observable.
//!
//! # Precompute and evict (sc-17155, the candle half of sc-17145)
//!
//! The projection lives here because the reference block owns it and one-block parity cannot be
//! shown without it. [`AdaLnModulation`] is a standalone value produced by
//! [`DitBlock::modulation`] rather than an internal of [`DitBlock::forward`] precisely so that
//! [`crate::dit::adaln`] can build one per block for a whole schedule and then take the projection
//! away with [`DitBlock::evict_adaln`] — ~13 B of the 33 B, **26.02 GB at bf16**.
//!
//! After eviction a block is a **body-only** block: [`DitBlock::forward`] still runs unchanged (it
//! consumes a modulation table and never touches the projection), while [`DitBlock::modulation`]
//! and [`DitBlock::forward_with_temb`] become typed errors rather than panics or silent zeros.
//!
//! ## The candle trap is not MLX's
//!
//! MLX is lazily evaluated, so its eviction has to force evaluation before dropping or the
//! modulation's pending graph node keeps the weight alive. **candle is eager**, so that particular
//! trap does not exist here and no forced-evaluation step is needed or meaningful.
//!
//! The candle trap is a different one: a `Tensor` is an `Arc` over its storage, and
//! `Weights::require` hands out a **clone that shares that storage**. `Tensor::to_dtype` is
//! additionally a no-op clone when the dtype already matches. So a loader map still held anywhere
//! keeps all 26.02 GB resident no matter how many blocks drop their projections, and every counter
//! reads exactly as it would if the eviction had never run. [`crate::dit::adaln`] documents the
//! measurement that distinguishes the two, and `tests/adaln_evict_memory.rs`'s control arm holds a
//! surviving clone to prove the gate is not vacuous.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::quant::AdaptLinear;
use candle_gen::{CandleError, Result, Weights};

use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::layers::{DitAttention, DitFeedForward, LinearNoBias, RmsNorm};
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::nn::silu;

/// The six modulation vectors of one block, each `[num_timesteps · MODALITY_NUM, hidden_size]`.
///
/// A function of the timestep embedding only — never of the tokens — which is what makes the
/// precompute-and-evict possible.
///
/// Every table is **contiguous**, deliberately. A `narrow` view would alias the whole
/// `[T·3, 6·hidden]` projection result, so the cache's byte accounting would describe six tensors
/// that are really one allocation — right total, wrong model of what is resident — and the
/// `index_select` in [`DitBlock::forward`] would run against a strided source on every layer of
/// every step.
#[derive(Debug, Clone)]
pub struct AdaLnModulation {
    /// Pre-attention shift.
    pub shift_msa: Tensor,
    /// Pre-attention scale, applied as `1 + scale`.
    pub scale_msa: Tensor,
    /// Attention residual gate.
    pub gate_msa: Tensor,
    /// Pre-feed-forward shift.
    pub shift_mlp: Tensor,
    /// Pre-feed-forward scale, applied as `1 + scale`.
    pub scale_mlp: Tensor,
    /// Feed-forward residual gate.
    pub gate_mlp: Tensor,
}

impl AdaLnModulation {
    /// The six tables in the reference's chunk order, for bulk byte accounting.
    pub fn tables(&self) -> impl Iterator<Item = &Tensor> {
        [
            &self.shift_msa,
            &self.scale_msa,
            &self.gate_msa,
            &self.shift_mlp,
            &self.scale_mlp,
            &self.gate_mlp,
        ]
        .into_iter()
    }

    /// Device bytes these six tables occupy.
    pub fn nbytes(&self) -> usize {
        self.tables()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// `adaln_proj`: `time_embed_dim → 6 · MODALITY_NUM · hidden_size`.
///
/// # The second of the two loaders the tiers pack (sc-20267)
///
/// `adaln_proj.linear` is 39.2 % of the DiT (26.01 GB at bf16), so it is packed in every quantized
/// tier — at its own width, because it is precomputed and evicted before step 0 (sc-17145) and
/// therefore buys disk rather than denoise-resident memory. [`crate::quant::lin`] detects it on the
/// `{prefix}.scales` sibling; a dense `bf16` tier takes the identical call unchanged.
#[derive(Debug, Clone)]
pub struct AdaLnProjection {
    inner: AdaptLinear,
    /// Device bytes the frozen base holds, recorded at load — a packed base has no dense weight to
    /// measure afterwards. See [`crate::quant::TieredLinear`].
    base_bytes: usize,
    /// The dtype the projection computes in — the loader's compute dtype. Held explicitly because a
    /// packed base has no dense weight to read it off, and the activation cast below is
    /// load-bearing (see [`Self::forward`]).
    compute_dtype: DType,
    hidden_size: usize,
}

impl AdaLnProjection {
    /// The width of the timestep embedding this projection consumes — `time_embed_dim`, 2688
    /// shipped.
    ///
    /// Read from the loaded projection rather than from a config, so that
    /// [`crate::dit::adaln::AdaLnCache`] validates a caller-supplied `temb` against the weights it
    /// will actually be multiplied by. On a packed base the shape is recovered from the `scales`,
    /// so it is the loaded artifact's answer on every tier.
    pub fn time_embed_dim(&self) -> usize {
        self.inner.in_features()
    }

    /// Columns emitted per timestep — `6 · MODALITY_NUM · hidden_size`, 96768 shipped.
    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    /// Device bytes this projection holds: the frozen base (packed blocks or dense weight) plus the
    /// bias, which stays full-precision on either tier.
    ///
    /// The arithmetic the eviction lever is sized on — 50 × (96768·2688 + 96768) × 2 B =
    /// **26_020_915_200 B (26.02 GB)** at bf16, and correspondingly less on a packed tier.
    pub fn nbytes(&self) -> usize {
        self.base_bytes
    }

    /// Whether the base loaded from a **packed** (pre-quantized) tier.
    pub fn is_packed(&self) -> bool {
        self.inner.is_packed()
    }

    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: DType,
    ) -> Result<Self> {
        let loaded = crate::quant::lin(w, prefix, true, dtype)?;
        let want = (cfg.adaln_out_features(), cfg.time_embed_dim);
        let got = loaded.linear.base_shape();
        if got != want {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit {prefix}.weight: expected [{}, {}], got [{}, {}]",
                want.0, want.1, got.0, got.1
            )));
        }
        Ok(Self {
            inner: loaded.linear,
            base_bytes: loaded.base_bytes,
            compute_dtype: dtype,
            hidden_size: cfg.hidden_size,
        })
    }

    fn names(prefix: &str) -> [String; 2] {
        [format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// Project one `[num_timesteps, time_embed_dim]` embedding into the six modulation tables.
    ///
    /// The activation runs at `temb`'s own precision and only its result is cast to the
    /// projection's dtype — the reference is explicit about this because `time_embedder` is float32
    /// in the mixed-precision checkpoint while `adaln_proj` is bfloat16, and rounding before the
    /// activation would bias every block identically at every step.
    pub fn forward(&self, temb: &Tensor) -> Result<AdaLnModulation> {
        let s = temb.dims();
        if s.len() != 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit adaln: expected temb as [num_timesteps, time_embed_dim], got {s:?}"
            )));
        }
        let steps = s[0];
        if s[1] != self.time_embed_dim() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit adaln: temb is {s:?} but this projection consumes width {}",
                self.time_embed_dim()
            )));
        }
        let activated = silu(temb)?.to_dtype(self.compute_dtype)?;
        let projected = self.inner.forward_upcast(&activated)?;

        // `view(-1, 6·hidden)` BEFORE the chunk: modality becomes a row axis, not a column one.
        let rows = steps * MODALITY_NUM;
        let table = projected.reshape((rows, MODULATION_PARAMS * self.hidden_size))?;
        let part = |i: usize| -> Result<Tensor> {
            Ok(table
                .narrow(1, i * self.hidden_size, self.hidden_size)?
                .contiguous()?)
        };
        Ok(AdaLnModulation {
            shift_msa: part(0)?,
            scale_msa: part(1)?,
            gate_msa: part(2)?,
            shift_mlp: part(3)?,
            scale_mlp: part(4)?,
            gate_mlp: part(5)?,
        })
    }
}

/// `x · (1 + scale[idx]) + shift[idx]`, gathering one modulation row per sequence row.
fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let rows = x.dims()[1];
    let scale = scale
        .index_select(indices, 0)?
        .to_dtype(x.dtype())?
        .reshape((1, rows, scale.dims()[1]))?;
    let shift = shift
        .index_select(indices, 0)?
        .to_dtype(x.dtype())?
        .reshape((1, rows, shift.dims()[1]))?;
    Ok(x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?)
}

/// Gather one gate per sequence row, shaped to broadcast over `[B, S, hidden]`.
fn gate_rows(gate: &Tensor, indices: &Tensor, rows: usize, dtype: DType) -> Result<Tensor> {
    Ok(gate
        .index_select(indices, 0)?
        .to_dtype(dtype)?
        .reshape((1, rows, gate.dims()[1]))?)
}

/// Reject an `adaln_indices` entry that does not address a modulation-table row.
///
/// candle's `index_select` **does** bounds-check on the CPU backend, but the failure surfaces as an
/// opaque narrow/index error from deep inside a kernel rather than as a statement about the model,
/// and the CUDA kernel's behaviour on an out-of-range index is not a contract candle documents. The
/// reference's `index_select` raises `IndexError`
/// (`transformer_minimax_h3.py::MiniMaxH3TransformerBlock.forward`), so this raises a typed,
/// legible error at the same point and does not depend on a backend detail.
///
/// The failure is reachable from ordinary caller arithmetic: `adaln_indices` is
/// `timestep_indices · MODALITY_NUM + token_tags`, so a single stale timestep index or an
/// out-of-range modality tag produces it.
///
/// **Called once per step by [`crate::dit::MiniMaxH3Dit::forward_packed`], not per block.** The
/// check reads the indices back to the host — a blocking D2H transfer — and the same index tensor
/// is gathered against the same-shaped table by all 50 blocks, so running it inside
/// [`DitBlock::forward`] repeated that stall 50× per step for one answer. A direct
/// [`DitBlock::forward`] caller (the parity tests) therefore does not get this typed error; on the
/// CPU backend candle's own `index_select` bounds-check still fails, just opaquely.
pub(crate) fn check_adaln_indices(indices: &Tensor, rows: usize) -> Result<()> {
    let idx = indices
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    if let Some(&bad) = idx.iter().find(|&&i| i as usize >= rows) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit block: adaln index {bad} is outside the modulation table's {rows} rows \
             (num_timesteps · {MODALITY_NUM})"
        )));
    }
    Ok(())
}

/// One block of the 50-layer stack.
#[derive(Debug, Clone)]
pub struct DitBlock {
    norm1: RmsNorm,
    attn: DitAttention,
    norm2: RmsNorm,
    ff: DitFeedForward,
    /// `None` once [`Self::evict_adaln`] has taken it. An `Option` rather than a `pub` field so
    /// that the post-eviction state is a **typed** one: every path that needs the projection
    /// reports its absence instead of reading a stale or fabricated table.
    adaln_proj: Option<AdaLnProjection>,
}

impl DitBlock {
    /// Load block `prefix` (e.g. `transformer_blocks.0`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: DType,
    ) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            norm1: RmsNorm::from_weights(w, &format!("{prefix}.norm1"), cfg.norm_eps, dtype)?,
            attn: DitAttention::from_weights(w, &format!("{prefix}.attn"), cfg, dtype)?,
            norm2: RmsNorm::from_weights(w, &format!("{prefix}.norm2"), cfg.norm_eps, dtype)?,
            ff: DitFeedForward::from_weights(w, &format!("{prefix}.ff"), dtype)?,
            adaln_proj: Some(AdaLnProjection::from_weights(
                w,
                &format!("{prefix}.adaln_proj.linear"),
                cfg,
                dtype,
            )?),
        })
    }

    /// This block's AdaLN projection, or `None` once it has been evicted.
    pub fn adaln_proj(&self) -> Option<&AdaLnProjection> {
        self.adaln_proj.as_ref()
    }

    /// Whether this block still holds its AdaLN projection.
    pub fn holds_adaln(&self) -> bool {
        self.adaln_proj.is_some()
    }

    /// Device bytes the block body holds — everything except `adaln_proj`.
    pub fn body_nbytes(&self) -> usize {
        self.attn.nbytes() + self.ff.nbytes()
    }

    /// Resolve one adapter path segment run **relative to this block** to its projection (sc-18728).
    ///
    /// **`adaln_proj` is deliberately unreachable.** Nothing in the published turbo set targets it,
    /// and making it addressable would let a LoRA change the modulation table
    /// [`crate::dit::adaln`] precomputes and then evicts — turning one checkpoint into two models
    /// depending on whether the eviction had already run. The MLX sibling makes the same exclusion.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            _ => None,
        }
    }

    /// Drop every residual attached anywhere in this block.
    pub fn clear_adapters(&mut self) {
        self.attn.clear_adapters();
        self.ff.clear_adapters();
    }

    /// Take the AdaLN projection out of the block, **returning** it so the caller controls the drop
    /// point.
    ///
    /// Returning rather than dropping in place is deliberate. On candle a `Tensor` is an `Arc` over
    /// its storage, so dropping this handle releases the buffer only if it is the last one — a
    /// surviving `Weights` map, or any other clone, keeps all of it resident.
    /// [`crate::dit::adaln::AdaLnCache::precompute_and_evict`] is the path that gets the whole
    /// sequence right (take → drop → synchronize → return pool pages to the driver); this method on
    /// its own only breaks the block's reference.
    ///
    /// Idempotent: a second call returns `None`.
    pub fn evict_adaln(&mut self) -> Option<AdaLnProjection> {
        self.adaln_proj.take()
    }

    /// Every tensor name a block consumes — 12 in the published checkpoint.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = RmsNorm::names(&format!("{prefix}.norm1")).to_vec();
        v.extend(RmsNorm::names(&format!("{prefix}.norm2")));
        v.extend(DitAttention::names(&format!("{prefix}.attn")));
        v.extend(DitFeedForward::names(&format!("{prefix}.ff")));
        v.extend(AdaLnProjection::names(&format!(
            "{prefix}.adaln_proj.linear"
        )));
        v
    }

    /// Project the timestep embedding into this block's modulation tables.
    ///
    /// Separated from [`Self::forward`] so [`crate::dit::adaln`] can build these once per schedule,
    /// hold them, and release `adaln_proj` — the single biggest memory lever in the port (~13 B of
    /// the 33 B).
    ///
    /// Errors once the projection has been evicted. That is the point: a denoise loop that reaches
    /// for the projection after the eviction has a mis-wired residency plan, and must say so rather
    /// than silently fall back.
    pub fn modulation(&self, temb: &Tensor) -> Result<AdaLnModulation> {
        self.adaln_proj
            .as_ref()
            .ok_or_else(|| {
                CandleError::Msg(
                    "minimax-h3 dit block: adaln_proj has been evicted; use the precomputed \
                     `AdaLnCache` modulation for this layer, or load with \
                     `AdaLnResidency::Resident` if the schedule cannot be enumerated up front"
                        .into(),
                )
            })?
            .forward(temb)
    }

    /// `x + gate_msa·attn(mod(norm1(x)))`, then `x + gate_mlp·ff(mod(norm2(x)))`.
    ///
    /// `adaln_indices` is `timestep_indices · MODALITY_NUM + token_tags`, one entry per row of the
    /// packed sequence.
    pub fn forward(
        &self,
        x: &Tensor,
        modulation: &AdaLnModulation,
        adaln_indices: &Tensor,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit block: expected [B, S, hidden], got {s:?}"
            )));
        }
        if adaln_indices.dims() != [s[1]] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit block: adaln_indices must be [seq_len={}], got {:?}",
                s[1],
                adaln_indices.dims()
            )));
        }
        // NOTE: no `check_adaln_indices` here — `forward_packed` runs it once per step, because
        // the check is a blocking D2H readback and every block gathers with the same tensor
        // against a same-shaped table. See its doc comment.
        let idx = adaln_indices.to_dtype(DType::U32)?;
        let rows = s[1];

        let normed = modulate(
            &self.norm1.forward(x)?,
            &modulation.scale_msa,
            &modulation.shift_msa,
            &idx,
        )?;
        let attn = self.attn.forward(&normed, Some((rope, tables)))?;
        let gate = gate_rows(&modulation.gate_msa, &idx, rows, x.dtype())?;
        let x = x.add(&gate.broadcast_mul(&attn)?)?;

        let normed = modulate(
            &self.norm2.forward(&x)?,
            &modulation.scale_mlp,
            &modulation.shift_mlp,
            &idx,
        )?;
        let ff = self.ff.forward(&normed)?;
        let gate = gate_rows(&modulation.gate_mlp, &idx, rows, x.dtype())?;
        Ok(x.add(&gate.broadcast_mul(&ff)?)?)
    }

    /// [`Self::modulation`] followed by [`Self::forward`] — the un-cached path the reference block
    /// takes, kept so parity can be shown against it directly.
    pub fn forward_with_temb(
        &self,
        x: &Tensor,
        temb: &Tensor,
        adaln_indices: &Tensor,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Tensor> {
        let modulation = self.modulation(temb)?;
        self.forward(x, &modulation, adaln_indices, rope, tables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The published block ships exactly 12 tensors, and none of the attention or feed-forward
    /// ones is a bias.
    #[test]
    fn block_names_cover_the_published_twelve_tensors() {
        let names = DitBlock::names("transformer_blocks.0");
        assert_eq!(names.len(), 12, "got {names:?}");
        for expect in [
            "transformer_blocks.0.norm1.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.adaln_proj.linear.bias",
        ] {
            assert!(names.contains(&expect.to_string()), "missing {expect}");
        }
        assert!(
            !names
                .iter()
                .any(|n| n.starts_with("transformer_blocks.0.attn.to_q.bias")),
            "the attention projections are bias-free"
        );
        assert!(
            !names
                .iter()
                .any(|n| n.contains(".ff.") && n.ends_with(".bias")),
            "the feed-forward is bias-free"
        );
    }

    /// An out-of-range AdaLN index must be a typed error naming the modulation table, not an opaque
    /// kernel failure or a backend-dependent gather.
    #[test]
    fn an_out_of_range_adaln_index_is_rejected() {
        let dev = Device::Cpu;
        let idx = |v: Vec<i64>| Tensor::from_vec(v, (2,), &dev).unwrap();
        // A 6-row table: 2 timesteps × 3 modalities.
        check_adaln_indices(&idx(vec![0, 5]), 6).unwrap();

        for (label, bad) in [
            ("past the end", idx(vec![0, 6])),
            (
                "a tag beyond MODALITY_NUM",
                // timestep 1, tag 3 -> 1·3 + 3 = 6, one past a 2-timestep table.
                idx(vec![0, MODALITY_NUM as i64 + 3]),
            ),
        ] {
            let e = check_adaln_indices(&bad, 6)
                .expect_err(&format!("{label} must be rejected"))
                .to_string();
            assert!(e.contains("outside the modulation table"), "{label}: {e}");
        }
    }

    /// `1 + scale`, not `scale`. An implementation that multiplied by the raw scale would be an
    /// identity at the reference's zero-initialization and near-identity thereafter.
    #[test]
    fn modulate_applies_one_plus_scale() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![2.0f32, 4.0], (1, 1, 2), &dev).unwrap();
        // Two table rows; index 1 selects the second.
        let scale = Tensor::from_vec(vec![9.0f32, 9.0, 0.5, 0.5], (2, 2), &dev).unwrap();
        let shift = Tensor::from_vec(vec![9.0f32, 9.0, 1.0, -1.0], (2, 2), &dev).unwrap();
        let idx = Tensor::from_vec(vec![1u32], (1,), &dev).unwrap();
        let out = modulate(&x, &scale, &shift, &idx).unwrap();
        assert_eq!(
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0 * 1.5 + 1.0, 4.0 * 1.5 - 1.0]
        );
    }

    /// The six modulation tables must each own their storage, so the cache's byte accounting
    /// describes six allocations rather than six views of one.
    #[test]
    fn the_modulation_tables_are_contiguous() {
        let dev = Device::Cpu;
        let cfg = MiniMaxH3DitConfig {
            hidden_size: 2,
            time_embed_dim: 3,
            ..MiniMaxH3DitConfig::default()
        };
        let out = cfg.adaln_out_features();
        let mut map = std::collections::HashMap::new();
        map.insert(
            "adaln_proj.linear.weight".to_string(),
            Tensor::ones((out, 3), DType::F32, &dev).unwrap(),
        );
        map.insert(
            "adaln_proj.linear.bias".to_string(),
            Tensor::zeros(out, DType::F32, &dev).unwrap(),
        );
        let proj = AdaLnProjection::from_weights(
            &Weights::from_map(map),
            "adaln_proj.linear",
            &cfg,
            DType::F32,
        )
        .unwrap();
        let temb = Tensor::ones((2, 3), DType::F32, &dev).unwrap();
        let m = proj.forward(&temb).unwrap();
        for (i, t) in m.tables().enumerate() {
            assert_eq!(t.dims(), &[2 * MODALITY_NUM, cfg.hidden_size], "table {i}");
            assert!(t.is_contiguous(), "table {i} must own its storage");
        }
        assert_eq!(
            m.nbytes(),
            MODULATION_PARAMS * 2 * MODALITY_NUM * cfg.hidden_size * 4
        );
        // A temb of the wrong width is a typed error, not a silent matmul failure.
        assert!(proj
            .forward(&Tensor::ones((2, 4), DType::F32, &dev).unwrap())
            .is_err());
        assert!(proj
            .forward(&Tensor::ones(3, DType::F32, &dev).unwrap())
            .is_err());
    }

    // ─── the packed tier (sc-20267) ─────────────────────────────────────────────────────────────

    /// A tiny config whose AdaLN projection has a legal packed geometry — `time_embed_dim` must be
    /// a multiple of the group size (64), which every shipped width is (2688 = 42 groups).
    fn packable_cfg() -> MiniMaxH3DitConfig {
        MiniMaxH3DitConfig {
            hidden_size: 2,
            time_embed_dim: 64,
            ..MiniMaxH3DitConfig::default()
        }
    }

    /// **The AdaLN projection detects a packed tier** through its real loader, keeps the base
    /// quantized, and reports the shrunken residency the eviction lever is sized on.
    #[test]
    fn a_packed_adaln_projection_loads_quantized() {
        let cfg = packable_cfg();
        let (out, in_) = (cfg.adaln_out_features(), cfg.time_embed_dim);
        let mut map = std::collections::HashMap::new();
        crate::quant::testkit::insert_packed(&mut map, "adaln_proj.linear", out, in_, 31);
        map.insert(
            "adaln_proj.linear.bias".to_string(),
            Tensor::zeros(out, DType::F32, &Device::Cpu).unwrap(),
        );
        let proj = AdaLnProjection::from_weights(
            &Weights::from_map(map),
            "adaln_proj.linear",
            &cfg,
            DType::F32,
        )
        .expect("packed load");
        assert!(proj.is_packed(), "the `.scales` sibling must pack");
        assert_eq!(proj.time_embed_dim(), in_, "recovered from the scales");
        assert_eq!(proj.out_features(), out);
        // The weight is Q4_1 blocks; only the bias stays dense f32.
        assert_eq!(proj.nbytes(), (out * in_ / 32) * 20 + out * 4);
        assert!(proj.nbytes() < out * in_ * 4);
    }

    /// **The packed AdaLN forward matches a dense reference of the same dequantized weights**, on
    /// relative max-abs. Both sides run the identical `AdaLnProjection::forward`, so this measures
    /// the packed path rather than the modulation split.
    #[test]
    fn the_packed_adaln_forward_matches_a_dense_reference() {
        let cfg = packable_cfg();
        let (out, in_) = (cfg.adaln_out_features(), cfg.time_embed_dim);
        let dev = Device::Cpu;
        let bias = Tensor::from_vec(
            (0..out)
                .map(|i| (i as f32 * 0.11).cos() * 0.3)
                .collect::<Vec<_>>(),
            out,
            &dev,
        )
        .unwrap();

        let mut packed_map = std::collections::HashMap::new();
        let grid = crate::quant::testkit::insert_packed(
            &mut packed_map,
            "adaln_proj.linear",
            out,
            in_,
            37,
        );
        packed_map.insert("adaln_proj.linear.bias".to_string(), bias.clone());
        let packed = AdaLnProjection::from_weights(
            &Weights::from_map(packed_map),
            "adaln_proj.linear",
            &cfg,
            DType::F32,
        )
        .expect("packed load");

        let mut dense_map = std::collections::HashMap::new();
        dense_map.insert("adaln_proj.linear.weight".to_string(), grid);
        dense_map.insert("adaln_proj.linear.bias".to_string(), bias);
        let dense = AdaLnProjection::from_weights(
            &Weights::from_map(dense_map),
            "adaln_proj.linear",
            &cfg,
            DType::F32,
        )
        .expect("dense load");
        assert!(
            !dense.is_packed(),
            "no `.scales` ⇒ the dense path, unchanged"
        );

        let temb = Tensor::from_vec(
            (0..2 * in_)
                .map(|i| (i as f32 * 0.23).sin())
                .collect::<Vec<_>>(),
            (2, in_),
            &dev,
        )
        .unwrap();
        let got = packed.forward(&temb).expect("packed modulation");
        let want = dense.forward(&temb).expect("dense modulation");
        for (i, (g, w)) in got.tables().zip(want.tables()).enumerate() {
            let drift = crate::quant::testkit::rel_max_abs(g, w);
            println!("[adaln] table {i} packed-vs-dense rel-max-abs = {drift:.3e}");
            assert!(
                drift < 5e-3,
                "table {i}: the Q4_1 repack is lossless up to the f16 scale/bias cast; got \
                 {drift:.3e}"
            );
        }
    }

    /// A packed projection whose recovered geometry disagrees with the config is refused by the
    /// same shape check the dense path runs — the check reads the `AdaptLinear`'s shape, so it is
    /// not silently skipped when there is no dense weight to measure.
    #[test]
    fn a_packed_adaln_of_the_wrong_shape_is_refused() {
        let cfg = packable_cfg();
        let mut map = std::collections::HashMap::new();
        // Half the expected output width.
        let out = cfg.adaln_out_features() / 2;
        crate::quant::testkit::insert_packed(&mut map, "adaln_proj.linear", out, 64, 41);
        map.insert(
            "adaln_proj.linear.bias".to_string(),
            Tensor::zeros(out, DType::F32, &Device::Cpu).unwrap(),
        );
        let err = AdaLnProjection::from_weights(
            &Weights::from_map(map),
            "adaln_proj.linear",
            &cfg,
            DType::F32,
        )
        .expect_err("a mis-shaped packed projection must be refused");
        assert!(err.to_string().contains("expected"), "{err}");
    }
}
