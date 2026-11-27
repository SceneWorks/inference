//! Z-Image attention: QKV projections → optional QK-RMSNorm → 3D RoPE → SDPA → out
//! projection. Port of the Python fork's `models/z_image/.../attention.py`, made
//! dimension-parametric. `to_q/to_k/to_v/to_out` are adapter hosts (LoRA/LoKr targets).
//!
//! Numeric parity proven stage-by-stage in the sc-2338 spike (tolerance 1e-2 — MLX runs
//! fp32 matmul in reduced precision on Metal). The DiT is **intentionally maskless** (SDPA
//! `mask=None`, final by design): the fork builds all-ones masks, and padded positions are
//! handled by the learned pad-token embeddings rather than attention masking — so there is no
//! mask to wire (see `transformer.rs`).

use mlx_rs::{
    fast::{rms_norm, scaled_dot_product_attention},
    ops::{concatenate_axis, split},
    transforms::checkpoint,
    Array, Dtype,
};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{sdpa_budgeted_bhsd, slice_axis, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Split lengths for a Z-Image unified `[x_emb ; cap_emb]` sequence under PAG perturbation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagAttentionSplit {
    pub(crate) x_len: i32,
    pub(crate) cap_len: i32,
}

impl PagAttentionSplit {
    /// Validate `x_len` against `total_len` and derive the trailing `cap_len`.
    pub(crate) fn from_total(x_len: i32, total_len: i32) -> Result<Self> {
        if x_len <= 0 || x_len > total_len {
            return Err(Error::Msg(format!(
                "z-image PAG split invalid: x_len={x_len}, total_len={total_len}"
            )));
        }
        Ok(Self {
            x_len,
            cap_len: total_len - x_len,
        })
    }
}

/// Step-level PAG routing context threaded into an MLX transformer forward pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PagStepContext<'a> {
    pub(crate) resolved: &'a mlx_gen::gen_core::PagResolved,
    pub(crate) step_idx: usize,
    pub(crate) total_steps: usize,
}

impl<'a> PagStepContext<'a> {
    /// Return `Some(x_len)` when main layer `layer_idx` should replace `x_emb` self-attention with
    /// `V_x` at this step while preserving `cap_emb` attention.
    pub(crate) fn layer_x_len(self, layer_idx: usize, x_len: i32) -> Option<i32> {
        self.resolved
            .perturbs_block(layer_idx as u32, self.step_idx, self.total_steps)
            .then_some(x_len)
    }
}

/// Apply Z-Image PAG attention perturbation on `[B, H, seq_len, D]` BHSD arrays when
/// `pag_x_len` is `Some(x_len)`: image prefix `0..x_len` receives `V_x` directly, while caption
/// tail `x_len..seq_len` runs `sdpa(&q_cap, k, v)` against the full `[K_x, K_cap]` and
/// `[V_x, V_cap]`.
pub(crate) fn apply_attention_with_pag<F>(
    q: &Array,
    k: &Array,
    v: &Array,
    pag_x_len: Option<i32>,
    sdpa: F,
) -> Result<Array>
where
    F: FnOnce(&Array, &Array, &Array) -> Result<Array>,
{
    let Some(x_len) = pag_x_len else {
        return sdpa(q, k, v);
    };
    let total_len = q.shape()[2];
    let split = PagAttentionSplit::from_total(x_len, total_len)?;
    let v_x = slice_axis(v, 2, 0, split.x_len)?;
    if split.cap_len == 0 {
        return Ok(v_x);
    }
    let q_cap = slice_axis(q, 2, split.x_len, split.cap_len)?;
    let o_cap = sdpa(&q_cap, k, v)?;
    Ok(concatenate_axis(&[&v_x, &o_cap], 2)?)
}

#[derive(Clone)]
pub struct ZImageAttention {
    pub to_q: AdaptableLinear,
    pub to_k: AdaptableLinear,
    pub to_v: AdaptableLinear,
    pub to_out: AdaptableLinear,
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    /// sc-4886 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// attention instead of retaining the `[heads, s, s]` probability matrix (the grad through
    /// `fast::scaled_dot_product_attention` decomposes to naive attention — MLX has no fused SDPA
    /// backward — and that one retained seq² array per block is ~half the dense training working
    /// set). Numerically identical (same math, recomputed); inference never sets it (default off,
    /// zero cost), the trainer enables it unconditionally.
    ckpt_sdpa: bool,
}

impl ZImageAttention {
    /// Load from `{prefix}.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`. QK-norm weights
    /// are optional (present iff `qk_norm=True`, which Z-Image-turbo uses).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dim: i32,
        n_heads: i32,
        eps: f32,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        Ok(Self {
            // Packed-detect (sc-8670): each QKV/out projection loads packed from a pre-quantized
            // snapshot (`{base}.scales` present) or dense otherwise. No bias on the Z-Image DiT
            // attention projections.
            to_q: crate::quant::lin(
                w,
                &format!("{prefix}.to_q"),
                crate::quant::DIT_ATTENTION_HAS_BIAS,
            )?,
            to_k: crate::quant::lin(
                w,
                &format!("{prefix}.to_k"),
                crate::quant::DIT_ATTENTION_HAS_BIAS,
            )?,
            to_v: crate::quant::lin(
                w,
                &format!("{prefix}.to_v"),
                crate::quant::DIT_ATTENTION_HAS_BIAS,
            )?,
            to_out: crate::quant::lin(
                w,
                &format!("{prefix}.to_out.0"),
                crate::quant::DIT_ATTENTION_HAS_BIAS,
            )?,
            norm_q: w.get(&format!("{prefix}.norm_q.weight")).cloned(),
            norm_k: w.get(&format!("{prefix}.norm_k.weight")).cloned(),
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
            ckpt_sdpa: false,
        })
    }

    /// `(heads, head_dim)`, as the iOS quantum accounting reads them.
    pub(crate) fn head_geometry(&self) -> (i32, i32) {
        (self.n_heads, self.head_dim)
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-4886). Training-only knob — see `ckpt_sdpa`.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Cast the projection weights + QK-norm scales to `dtype` (sc-4887 bf16 training). Quantized
    /// projections are left untouched (see [`AdaptableLinear::cast_weights`]).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for lin in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
        ] {
            lin.cast_weights(dtype)?;
        }
        for norm in [&mut self.norm_q, &mut self.norm_k].into_iter().flatten() {
            if norm.dtype() != dtype {
                *norm = norm.as_dtype(dtype)?;
            }
        }
        Ok(())
    }

    /// The unbounded forward — one fused SDPA over the whole query axis. Byte-identical to the
    /// pre-SC-15615 path and the default for every caller that has not selected the bounded-attention
    /// rung (including the trainer).
    pub fn forward(&self, x: &Array, freqs_cis: &Array) -> Result<Array> {
        self.forward_budgeted(x, freqs_cis, AttentionPlan::UNBOUNDED)
    }

    /// [`Self::forward`] with an explicit attention-score budget (SC-15615, ladder rung 3 — the MLX
    /// twin of `candle_gen_z_image`'s `forward_with_attention_budget`). Query-row chunking preserves
    /// every query's complete key/value domain and the exact projections, QK-norm, RoPE, and scale;
    /// only the number of query rows attended per call changes.
    ///
    /// The plan is threaded, not stored, so it is request-scoped by construction: nothing on a warm
    /// generator carries a prior request's rung — or cancel flag — into the next one.
    pub fn forward_budgeted(
        &self,
        x: &Array,
        freqs_cis: &Array,
        plan: AttentionPlan<'_>,
    ) -> Result<Array> {
        self.forward_budgeted_with_pag(x, freqs_cis, plan, None)
    }

    /// [`Self::forward_budgeted`] with optional Perturbed-Attention Guidance (PAG) applied to the
    /// leading `pag_x_len` image tokens while preserving caption attention.
    pub fn forward_budgeted_with_pag(
        &self,
        x: &Array,
        freqs_cis: &Array,
        plan: AttentionPlan<'_>,
        pag_x_len: Option<i32>,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let dim = self.n_heads * self.head_dim;

        let mut q = self
            .to_q
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let mut k = self
            .to_k
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let v = self
            .to_v
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;

        if let Some(nq) = &self.norm_q {
            q = rms_norm(&q, nq, self.eps)?;
        }
        if let Some(nk) = &self.norm_k {
            k = rms_norm(&k, nk, self.eps)?;
        }

        q = self.apply_rope(&q, freqs_cis)?;
        k = self.apply_rope(&k, freqs_cis)?;

        // (b, s, h, hd) -> (b, h, s, hd)
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // 6th arg is `sinks` (MLX ≥0.30 attention-sinks); `None` = standard attention.
        // Gradient checkpointing (training) and query-row budgeting (inference rung 3) are mutually
        // exclusive by construction: the trainer never selects a bounded rung, and `eval` inside a
        // checkpoint segment is not valid under an autograd trace. The `ckpt_sdpa` arm therefore keeps
        // the single unchunked call and is byte-identical to the pre-SC-15615 training path.
        let o = if self.ckpt_sdpa && pag_x_len.is_none() {
            // sc-4886: checkpoint just the SDPA. q/k/v are the threaded inputs (grads to the
            // QKV projections — and their LoRA — flow through them); only the f32 scale is
            // captured. The backward recomputes the decomposed attention for THIS layer alone,
            // so the seq² probability matrix is a per-layer transient, never 30× retained.
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("z-image: checkpoint SDPA produced no output".into()))?
        } else {
            apply_attention_with_pag(&q, &k, &v, pag_x_len, |q_in, k_in, v_in| {
                sdpa_budgeted_bhsd(q_in, k_in, v_in, self.scale, None, plan)
            })?
        };
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, dim])?;
        self.to_out.forward(&o)
    }

    /// Quantize the QKV + output projections to Q4/Q8 (group_size 64). QK-norm weights are
    /// RMSNorm scales (not Linears), so they stay dense — matching the fork's `nn.quantize`.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
        ] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Port of `ZImageAttention._apply_rotary_emb`. `x`:(b,s,h,hd), `fc`:(s,hd/2,2).
    ///
    /// The `(cos, sin)` pairs are de-interleaved from `fc` here; the rotation is the shared
    /// [`mlx_gen::qkv::apply_rope`] (SW-6140) — the same core `rope_rotate` glue kernel (sc-27453)
    /// over the same `[1, s, 1, hd/2]` tables, no casts, as this crate's private copy.
    fn apply_rope(&self, x: &Array, fc: &Array) -> Result<Array> {
        let sh = x.shape();
        let (s, hd) = (sh[1], sh[3]);
        let half = hd / 2;
        let fc5 = fc.reshape(&[1, s, 1, half, 2])?;
        let fp = split(&fc5, 2, 4)?;
        let cos = fp[0].reshape(&[1, s, 1, half])?;
        let sin = fp[1].reshape(&[1, s, 1, half])?;
        mlx_gen::qkv::apply_rope(
            x,
            mlx_gen::qkv::RopeTables::new(&cos, &sin),
            mlx_gen::qkv::RopeStyle::AdjacentPair,
            mlx_gen::qkv::RotationAxes::TokenMajor,
            None,
            mlx_gen::qkv::RopeDtype::Promoted,
        )
    }
}

pub(crate) const ADAPTER_TARGET_PATHS: &[&str] = &["to_q", "to_k", "to_v", "to_out.0"];

impl AdaptableHost for ZImageAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ADAPTER_TARGET_PATHS
            .iter()
            .map(|path| (*path).to_owned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pag_split_and_attention_prefix_perturbation_preserves_caption_tail() {
        assert_eq!(
            PagAttentionSplit::from_total(3, 5).unwrap(),
            PagAttentionSplit {
                x_len: 3,
                cap_len: 2,
            }
        );
        assert!(PagAttentionSplit::from_total(0, 5).is_err());
        assert!(PagAttentionSplit::from_total(6, 5).is_err());

        let total = 2 * 5 * 4;
        let q_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k_data: Vec<f32> = (0..total).map(|i| ((i % 7) as f32) * 0.2).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.25 + 1.0).collect();
        let q = Array::from_slice(&q_data, &[1, 2, 5, 4]);
        let k = Array::from_slice(&k_data, &[1, 2, 5, 4]);
        let v = Array::from_slice(&v_data, &[1, 2, 5, 4]);

        let scale = (4.0f32).powf(-0.5);
        let unperturbed = apply_attention_with_pag(&q, &k, &v, None, |q_in, k_in, v_in| {
            sdpa_budgeted_bhsd(q_in, k_in, v_in, scale, None, AttentionPlan::UNBOUNDED)
        })
        .unwrap();
        let perturbed = apply_attention_with_pag(&q, &k, &v, Some(3), |q_in, k_in, v_in| {
            sdpa_budgeted_bhsd(q_in, k_in, v_in, scale, None, AttentionPlan::UNBOUNDED)
        })
        .unwrap();

        mlx_rs::transforms::eval([&unperturbed, &perturbed]).unwrap();
        assert_eq!(perturbed.shape(), &[1, 2, 5, 4]);

        let p_x = mlx_gen::array::contiguous(&slice_axis(&perturbed, 2, 0, 3).unwrap()).unwrap();
        let v_x = mlx_gen::array::contiguous(&slice_axis(&v, 2, 0, 3).unwrap()).unwrap();
        mlx_rs::transforms::eval([&p_x, &v_x]).unwrap();
        assert_eq!(p_x.as_slice::<f32>(), v_x.as_slice::<f32>());

        let p_cap = mlx_gen::array::contiguous(&slice_axis(&perturbed, 2, 3, 2).unwrap()).unwrap();
        let u_cap =
            mlx_gen::array::contiguous(&slice_axis(&unperturbed, 2, 3, 2).unwrap()).unwrap();
        mlx_rs::transforms::eval([&p_cap, &u_cap]).unwrap();
        for (a, b) in p_cap.as_slice::<f32>().iter().zip(u_cap.as_slice::<f32>()) {
            assert!((a - b).abs() < 1e-5, "caption tail drifted: {a} vs {b}");
        }
    }
}
