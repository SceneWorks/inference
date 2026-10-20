//! VAE mid-block spatial self-attention: GroupNorm → QKV → single-head SDPA over the H·W
//! tokens → out projection, with a residual. NCHW I/O.

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionBudget, AttentionPlan};
use mlx_gen::nn::group_norm;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

pub(crate) const QUANTIZATION_LINEAR_PATHS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

pub struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    /// The quantum label of the side (encoder / decoder) this attention runs on.
    label: &'static str,
}

impl VaeAttention {
    /// A decoder-side attention (the decoder's quantum label).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights_labeled(w, prefix, &crate::quantum::DECODE_LABELS)
    }

    /// An attention whose iOS quantum carries `labels.attention`.
    pub(crate) fn from_weights_labeled(
        w: &Weights,
        prefix: &str,
        labels: &'static crate::quantum::VaeLabels,
    ) -> Result<Self> {
        let g = |s: &str| w.require(&format!("{prefix}.{s}")).cloned();
        // Packed-detect (sc-8670): the mid-block attention QKV/out projections load packed from a
        // pre-quantized snapshot or dense otherwise; all carry a bias. They are the only quantizable
        // leaves in the otherwise-conv VAE. The pre-quantized `{base}.{weight,scales,biases,bias}`
        // pass through the loader's diffusers→internal VAE remap untouched (not conv weights).
        let lin = |name: &str| crate::quant::lin(w, &format!("{prefix}.{name}"), true);
        Ok(Self {
            gn_w: g("group_norm.weight")?,
            gn_b: g("group_norm.bias")?,
            q: lin(QUANTIZATION_LINEAR_PATHS[0])?,
            k: lin(QUANTIZATION_LINEAR_PATHS[1])?,
            v: lin(QUANTIZATION_LINEAR_PATHS[2])?,
            out: lin(QUANTIZATION_LINEAR_PATHS[3])?,
            label: labels.attention,
        })
    }

    /// Quantize the QKV + output projections to Q4/Q8 (group_size 64) — the fork's `nn.quantize`
    /// over the VAE hits these Linears (the only quantizable leaves in the otherwise-conv VAE).
    /// GroupNorm scales stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.q, &mut self.k, &mut self.v, &mut self.out] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// The unchunked attention — byte-identical to the historical forward (one SDPA call).
    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        self.forward_planned(x_nchw, AttentionPlan::UNBOUNDED)
    }

    /// The query-row plan this attention runs under: on iOS (the device, or the host CPU fixture
    /// that exercises its barriers) query-chunked so one chunk's `[B, 1, rows, N]` scores stay within
    /// [`mlx_gen::ios_quantum::VAE_ATTENTION_SCORE_ELEMENTS`] and each chunk is materialized before
    /// the next — the VAE's head dim (512) is never a fused Metal SDPA shape, so the unchunked call
    /// materializes `N²` scores (1 GiB per f32 score tensor at a 1024² decode's 16384 latent tokens).
    /// Everywhere else the plan is unbounded: the historical host path, byte for byte.
    pub(crate) fn plan() -> AttentionPlan<'static> {
        if mlx_gen::ios_quantum::enabled() {
            Self::bounded_plan()
        } else {
            AttentionPlan::UNBOUNDED
        }
    }

    /// The iOS plan: one chunk's scores within
    /// [`mlx_gen::ios_quantum::VAE_ATTENTION_SCORE_ELEMENTS`], each chunk materialized.
    pub(crate) fn bounded_plan() -> AttentionPlan<'static> {
        AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            mlx_gen::ios_quantum::VAE_ATTENTION_SCORE_ELEMENTS,
            true,
        ))
    }

    /// [`Self::forward`] as one guarded iOS quantum under [`Self::plan`], priced per query chunk
    /// ([`mlx_gen::ios_quantum::vae_attention_bytes`]).
    pub(crate) fn forward_quantum(&self, x_nchw: &Array) -> Result<Array> {
        let plan = Self::plan();
        crate::quantum::conv_quantum(
            self.label,
            |backend| {
                let [b, h, w, c] = crate::quantum::nhwc(x_nchw);
                let n = h * w;
                mlx_gen::ios_quantum::vae_attention_bytes(
                    b,
                    n,
                    c,
                    plan.budget.query_block(b, 1, n, n),
                    x_nchw.item_size() as u64,
                    backend,
                )
            },
            || self.forward_planned(x_nchw, plan),
        )
    }

    /// The attention under `plan`: [`sdpa_budgeted_bhsd`] is exactly the single
    /// `scaled_dot_product_attention` call when `plan` is unbounded (or the whole call fits), and
    /// otherwise the same single-head attention one query chunk at a time — each chunk's rows see
    /// every key, so the result is the same math row for row.
    pub(crate) fn forward_planned(&self, x_nchw: &Array, plan: AttentionPlan<'_>) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);

        let normed = group_norm(&x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        // (B,H,W,C) -> (B, H*W, 1, C) -> (B, 1, H*W, C) [single head, head_dim = C].
        let proj = |lin: &AdaptableLinear| -> Result<Array> {
            Ok(lin
                .forward(&normed)?
                .reshape(&[b, h * w, 1, c])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = proj(&self.q)?;
        let k = proj(&self.k)?;
        let v = proj(&self.v)?;
        // iOS: the projections are materialized before the attention is built, so the GroupNorm's
        // buffers are released before any score tensor exists (MLX keeps an encoded op's inputs
        // alive until its command buffer completes). Values are unchanged; off device the graph is
        // exactly the historical one.
        if mlx_gen::ios_quantum::enabled() {
            mlx_rs::transforms::eval([&q, &k, &v])?;
        }

        let scale = (c as f32).powf(-0.5);
        // `sdpa_budgeted_bhsd` with an unbounded plan is the single `scaled_dot_product_attention`
        // call (no mask, no sinks) this forward always made.
        let o = sdpa_budgeted_bhsd(&q, &k, &v, scale, None, plan)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, h, w, c])?;
        let o = self.out.forward(&o)?;

        Ok(add(&x, &o)?.transpose_axes(&[0, 3, 1, 2])?) // residual, back to NCHW
    }
}
