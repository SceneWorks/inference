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
//! # Precompute and evict (sc-17145)
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

use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::BoundedAttention;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::dit::layers::{DitAttention, DitFeedForward, RmsNorm};
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::tensor::slice_axis;

/// The six modulation vectors of one block, each `[num_timesteps · MODALITY_NUM, hidden_size]`.
///
/// A function of the timestep embedding only — never of the tokens — which is what makes
/// sc-17145's precompute-and-evict possible.
#[derive(Debug, Clone)]
pub struct AdaLnModulation {
    /// Pre-attention shift.
    pub shift_msa: Array,
    /// Pre-attention scale, applied as `1 + scale`.
    pub scale_msa: Array,
    /// Attention residual gate.
    pub gate_msa: Array,
    /// Pre-feed-forward shift.
    pub shift_mlp: Array,
    /// Pre-feed-forward scale, applied as `1 + scale`.
    pub scale_mlp: Array,
    /// Feed-forward residual gate.
    pub gate_mlp: Array,
}

impl AdaLnModulation {
    /// The six tables in the reference's chunk order, for bulk evaluation and byte accounting.
    ///
    /// [`crate::dit::adaln::AdaLnCache`] evaluates through this before releasing the projection —
    /// see that module's "lazy-eval trap".
    pub fn tables(&self) -> impl Iterator<Item = &Array> {
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
}

/// `adaln_proj`: `time_embed_dim → 6 · MODALITY_NUM · hidden_size`.
///
/// **Tier-aware** (sc-17150). The second of the crate's two packed loaders — see [`crate::quant`]
/// for why the set is exactly two. At 26_020_915_200 bf16 bytes this is 39.2% of the DiT, so it is
/// also the single tensor group whose width most moves a tier's hosted size: packing it at the
/// tier's own width is what makes `q4` 18_779_814_400 B rather than 25_282_624_000 B.
#[derive(Clone)]
pub struct AdaLnProjection {
    linear: AdaptableLinear,
    hidden_size: i32,
}

/// Logical shape and packed-ness, not the opaque u32 code buffer.
impl std::fmt::Debug for AdaLnProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaLnProjection")
            .field("shape", &self.linear.base_shape())
            .field("quantized", &self.linear.is_quantized())
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}

impl AdaLnProjection {
    /// The width of the timestep embedding this projection consumes — `time_embed_dim`, 2688
    /// shipped.
    ///
    /// Read from the loaded tensor rather than from a config, so that
    /// [`crate::dit::adaln::AdaLnCache`] validates a caller-supplied `temb` against the weights it
    /// will actually be multiplied by. On a packed tier the logical width is recovered from the
    /// scales grid (`scales_cols · group_size`), which is exact because every packable DiT width is
    /// a multiple of [`crate::convert::GROUP_SIZE`].
    pub fn time_embed_dim(&self) -> i32 {
        self.linear.base_shape()[1]
    }

    /// Columns emitted per timestep — `6 · MODALITY_NUM · hidden_size`, 96768 shipped.
    pub fn out_features(&self) -> i32 {
        self.linear.base_shape()[0]
    }

    /// Device bytes this projection **actually** holds — the packed triple on a `q4`/`q8` tier, the
    /// dense weight + bias on `bf16`.
    ///
    /// The arithmetic the eviction lever is sized on. At bf16 that is 50 × (96768·2688 + 96768) ×
    /// 2 B = **26_020_915_200 B (26.02 GB)**; the same 50 projections are ~13.02 GB at q8 and
    /// ~6.52 GB at q4, so the lever shrinks with the tier rather than staying at its headline value.
    pub fn nbytes(&self) -> usize {
        crate::quant::nbytes(&self.linear)
    }

    /// `true` on a `q4` / `q8` tier.
    pub fn is_quantized(&self) -> bool {
        self.linear.is_quantized()
    }

    fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let mut linear = crate::quant::lin(w, prefix, true)?;
        // No-op on a packed base, whose compute dtype is fixed by its scales.
        linear.cast_weights(dtype)?;
        let want = vec![cfg.adaln_out_features(), cfg.time_embed_dim];
        // Checked against the LOGICAL shape, so the guard survives the packed tier — a packed
        // `weight` is `[out, in·bits/32]` u32 and would fail this for a correct artifact.
        if linear.base_shape() != want {
            return Err(Error::Msg(format!(
                "minimax-h3 dit {prefix}.weight: expected {want:?}, got {:?}",
                linear.base_shape()
            )));
        }
        Ok(Self {
            linear,
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
    pub fn forward(&self, temb: &Array) -> Result<AdaLnModulation> {
        let s = temb.shape();
        if s.len() != 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit adaln: expected temb as [num_timesteps, time_embed_dim], got {s:?}"
            )));
        }
        let steps = s[0];
        // On a packed tier the compute dtype comes from the scales (bf16), not from a dense weight
        // that no longer exists — see `crate::quant::compute_dtype`.
        let activated = silu(temb)?.as_dtype(crate::quant::compute_dtype(&self.linear))?;
        let projected = self.linear.forward(&activated)?;

        // `view(-1, 6·hidden)` BEFORE the chunk: modality becomes a row axis, not a column one.
        let rows = steps * MODALITY_NUM;
        let table = projected.reshape(&[rows, MODULATION_PARAMS * self.hidden_size])?;
        let part = |i: i32| slice_axis(&table, 1, i * self.hidden_size, (i + 1) * self.hidden_size);
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
fn modulate(x: &Array, scale: &Array, shift: &Array, indices: &Array) -> Result<Array> {
    let scale = scale.take_axis(indices, 0)?;
    let shift = shift.take_axis(indices, 0)?;
    let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
    Ok(add(&multiply(x, &add(&one, &scale)?)?, &shift)?)
}

/// Reject an `adaln_indices` entry that does not address a modulation-table row.
///
/// **MLX does not bounds-check a gather** — `mlx-rs`' own indexing docs state that "mlx allows out
/// of bounds indexing", and the Metal kernel only fixes up *negative* indices — so an index past
/// the end reads whatever is adjacent in the buffer and the block computes silent garbage. The
/// reference's `index_select` raises `IndexError` instead
/// (`transformer_minimax_h3.py::MiniMaxH3TransformerBlock.forward`).
///
/// The failure is reachable from ordinary caller arithmetic: `adaln_indices` is
/// `timestep_indices · MODALITY_NUM + token_tags`, so a single stale timestep index or an
/// out-of-range modality tag produces it — and sc-17146 is the story that will build those tensors.
fn check_adaln_indices(indices: &Array, rows: i32) -> Result<()> {
    let idx = indices.as_dtype(Dtype::Int32)?;
    let min: i32 = idx.min(None)?.item();
    let max: i32 = idx.max(None)?.item();
    if min < 0 || max >= rows {
        return Err(Error::Msg(format!(
            "minimax-h3 dit block: adaln_indices range [{min}, {max}] is outside the modulation \
             table's {rows} rows (num_timesteps · {MODALITY_NUM}); MLX gathers out of bounds \
             silently rather than failing"
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
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: Dtype,
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

    /// Take the AdaLN projection out of the block, **returning** it so the caller controls the drop
    /// point.
    ///
    /// Returning rather than dropping in place is deliberate. Dropping a Rust handle is not by
    /// itself a device-memory release: MLX arrays are reference-counted behind a lazy graph, so the
    /// buffer survives for as long as *anything* still references it — including an un-evaluated
    /// modulation table computed from it. [`crate::dit::adaln::AdaLnCache::precompute_and_evict`]
    /// is the path that gets the ordering right (force evaluation → take → drop → drain the
    /// allocator cache); this method on its own only breaks the block's reference.
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
    pub fn modulation(&self, temb: &Array) -> Result<AdaLnModulation> {
        self.adaln_proj
            .as_ref()
            .ok_or_else(|| {
                Error::Msg(
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
        x: &Array,
        modulation: &AdaLnModulation,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Array> {
        self.forward_bounded(
            x,
            modulation,
            adaln_indices,
            rope,
            tables,
            BoundedAttention::UNBOUNDED,
        )
    }

    /// [`Self::forward`] under an explicit [`BoundedAttention`] — the rung-3 seam (sc-18661).
    ///
    /// The plan is threaded rather than read from a block-level field so that the *same* loaded stack
    /// can run a bounded and an un-bounded arm back to back in one process, which is what
    /// `tests/bounded_attention_real.rs` needs to compare peaks against a 17 GB DiT it can only
    /// afford to load once.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_bounded(
        &self,
        x: &Array,
        modulation: &AdaLnModulation,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
        let s = x.shape();
        if s.len() != 3 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit block: expected [B, S, hidden], got {s:?}"
            )));
        }
        if adaln_indices.shape() != [s[1]] {
            return Err(Error::Msg(format!(
                "minimax-h3 dit block: adaln_indices must be [seq_len={}], got {:?}",
                s[1],
                adaln_indices.shape()
            )));
        }
        check_adaln_indices(adaln_indices, modulation.scale_msa.shape()[0])?;

        let normed = modulate(
            &self.norm1.forward(x)?,
            &modulation.scale_msa,
            &modulation.shift_msa,
            adaln_indices,
        )?;
        let attn = self
            .attn
            .forward_bounded(&normed, Some((rope, tables)), bounded)?;
        let gate = modulation.gate_msa.take_axis(adaln_indices, 0)?;
        let x = add(x, &multiply(&gate, &attn)?)?;

        let normed = modulate(
            &self.norm2.forward(&x)?,
            &modulation.scale_mlp,
            &modulation.shift_mlp,
            adaln_indices,
        )?;
        let ff = self.ff.forward(&normed)?;
        let gate = modulation.gate_mlp.take_axis(adaln_indices, 0)?;
        Ok(add(&x, &multiply(&gate, &ff)?)?)
    }

    /// [`Self::modulation`] followed by [`Self::forward`] — the un-cached path the reference block
    /// takes, kept so parity can be shown against it directly.
    pub fn forward_with_temb(
        &self,
        x: &Array,
        temb: &Array,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
    ) -> Result<Array> {
        self.forward_with_temb_bounded(
            x,
            temb,
            adaln_indices,
            rope,
            tables,
            BoundedAttention::UNBOUNDED,
        )
    }

    /// [`Self::forward_with_temb`] under an explicit [`BoundedAttention`].
    ///
    /// The resident-AdaLN arm carries the plan too: a rung that reached only the precompute-and-evict
    /// path would be selectable on a request the solver routes to the other one, and would then bound
    /// nothing while the contract said it did.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_temb_bounded(
        &self,
        x: &Array,
        temb: &Array,
        adaln_indices: &Array,
        rope: &MmRope,
        tables: &MmRopeTables,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
        let modulation = self.modulation(temb)?;
        self.forward_bounded(x, &modulation, adaln_indices, rope, tables, bounded)
    }
}

/// `attn.*` / `ff.*` — the six adaptable leaves per block (sc-18724).
///
/// **`adaln_proj` is deliberately unreachable.** No published MiniMax-H3 LoRA targets it, and it is
/// the one projection [`DitBlock::evict_adaln`] *removes* mid-render (sc-17145): an adapter installed on
/// it would be silently discarded with the eviction on the precompute path but survive on the
/// resident one, making the same file produce two different models. `norm1`/`norm2` are RMSNorm
/// gains, not Linears. Either key surfaces as unmatched (loud) — see [`crate::adapters`].
impl AdaptableHost for DitBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// An out-of-range AdaLN index must be an error, not a silent out-of-bounds gather. MLX does
    /// not bounds-check, so without this the block would compute plausible garbage.
    #[test]
    fn an_out_of_range_adaln_index_is_rejected() {
        // A 6-row table: 2 timesteps × 3 modalities.
        let ok = Array::from_slice(&[0i32, 5, 2], &[3]);
        check_adaln_indices(&ok, 6).unwrap();

        for (label, bad) in [
            ("past the end", Array::from_slice(&[0i32, 6], &[2])),
            ("negative", Array::from_slice(&[-1i32, 0], &[2])),
            (
                "a tag beyond MODALITY_NUM",
                // timestep 1, tag 3 -> 1·3 + 3 = 6, one past a 2-timestep table.
                Array::from_slice(&[MODALITY_NUM + 3], &[1]),
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
        let x = Array::from_slice(&[2.0f32, 4.0], &[1, 2]);
        // Two table rows; index 1 selects the second.
        let scale = Array::from_slice(&[9.0f32, 9.0, 0.5, 0.5], &[2, 2]);
        let shift = Array::from_slice(&[9.0f32, 9.0, 1.0, -1.0], &[2, 2]);
        let idx = Array::from_slice(&[1i32], &[1]);
        let out = modulate(&x, &scale, &shift, &idx).unwrap();
        assert_eq!(out.as_slice::<f32>(), &[2.0 * 1.5 + 1.0, 4.0 * 1.5 - 1.0]);
    }
}
