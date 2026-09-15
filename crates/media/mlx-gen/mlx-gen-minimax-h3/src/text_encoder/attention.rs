//! GQA self-attention for the H3 Qwen3-VL-32B condition encoder: 64 query / 8 kv heads, bias-less
//! q/k/v/o, per-head q/k RMSNorm (on the head dim, **before** RoPE), HF half-split RoPE, masked
//! SDPA. Structurally the same block `mlx-gen-krea` / `mlx-gen-ideogram` / `mlx-gen-flux2` ship;
//! the RoPE apply and GQA `repeat_kv` come from `mlx_gen::nn` rather than being open-coded (F-078).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{apply_text_rope as apply_rope, repeat_kv};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join_key, lin};

/// One Qwen3 attention block.
pub struct Qwen3Attention {
    q_w: AdaptableLinear,
    k_w: AdaptableLinear,
    v_w: AdaptableLinear,
    o_w: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen3Attention {
    /// Load `{prefix}.{q,k,v,o}_proj.weight` + `{prefix}.{q,k}_norm.weight`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            q_w: lin(w, &join_key(prefix, "q_proj.weight"))?,
            k_w: lin(w, &join_key(prefix, "k_proj.weight"))?,
            v_w: lin(w, &join_key(prefix, "v_proj.weight"))?,
            o_w: lin(w, &join_key(prefix, "o_proj.weight"))?,
            q_norm: w.require(&join_key(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join_key(prefix, "k_norm.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

    /// Quantize the four projections in place (group-wise affine). The q/k norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q_w.quantize(bits, Some(super::GROUP_SIZE))?;
        self.k_w.quantize(bits, Some(super::GROUP_SIZE))?;
        self.v_w.quantize(bits, Some(super::GROUP_SIZE))?;
        self.o_w.quantize(bits, Some(super::GROUP_SIZE))?;
        Ok(())
    }

    /// The packed width shared by all four projections, or `None` if any loaded dense or they
    /// disagree.
    pub fn packed_bits(&self) -> Option<i32> {
        super::uniform_packed_bits([&self.q_w, &self.k_w, &self.v_w, &self.o_w])
    }

    /// Device bytes the four projections plus the two dense q/k norms hold.
    pub fn nbytes(&self) -> usize {
        [&self.q_w, &self.k_w, &self.v_w, &self.o_w]
            .into_iter()
            .map(crate::quant::nbytes)
            .sum::<usize>()
            + self.q_norm.nbytes()
            + self.k_norm.nbytes()
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b, 1, s, s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = self
            .q_w
            .forward_upcast(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = self
            .k_w
            .forward_upcast(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_w
            .forward_upcast(x)?
            .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Per-head q/k RMSNorm over the head dim (Qwen3), before RoPE.
        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat each kv head `groups` times to match the query heads (8× here).
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b,s,h,hd] → [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_w.forward_upcast(&o)
    }
}
