//! The two leaf modules every DiT block and every token-refiner block is built from.
//!
//! Both are **bias-free** — `MiniMaxH3Attention` constructs its four projections with
//! `bias=False` and the block's `FeedForward(..., bias=False)` does the same, which the published
//! tensor index corroborates: no `.bias` exists under `attn.` or `ff.` anywhere in
//! `transformer/`. Only `adaln_proj.linear`, the input/output projections and the timestep MLP
//! carry biases.
//!
//! # The attention transient is a genuine cross-backend difference (sc-17152)
//!
//! MLX's fused `scaled_dot_product_attention` **streams the scores**: sc-17152 measured its peak
//! tracking `4·B·H·S·D` exactly, i.e. there is no `[B, H, Sq, Sk]` tensor in memory at all. candle
//! has no fused kernel here, so [`DitAttention::forward`] routes through
//! `candle_gen::sdpa_budgeted_bhsd`, which **does** materialize scores — in bounded row blocks
//! rather than all at once (`ATTN_SCORES_BUDGET`, the shared i32-overflow-safe helper from
//! sc-9116).
//!
//! The consequence is stated rather than glossed: at the shipped geometry a 15 s render packs
//! ~94k rows over 56 heads, where the full score tensor would be ~2.0e12 elements — past `i32::MAX`
//! and far past any card — so the budgeted split is **not optional on this lane** the way MLX's
//! streaming kernel makes it optional there. sc-17152 additionally measured chunked SDPA on MLX at
//! **+50.3 % peak and ~3× wall**, so its conclusion ("do not chunk") is an MLX result and must not
//! be carried across: the two backends have different attention memory shapes, and this lane's
//! numbers are sc-17156's to measure on real hardware.
//!
//! sc-17152 also probed `matmul` correctness above `i32::MAX` **on MLX** and found it exact. That
//! is an MLX result. Nothing here assumes it transfers to candle/CUDA, and the widest single tensor
//! this module writes is the SwiGLU projection `[1, S, 2·ffn_dim]` — `[1, S, 28672]` at the shipped
//! geometry — not anything in the attention.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::dit::config::MiniMaxH3DitConfig;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::layout::split_gate_value;
use crate::nn::{linear_nb, rms_weighted, sdpa, silu};

/// A **forward-time additive LoRA residual** attached to a [`LinearNoBias`] (sc-18728).
///
/// `scale · ((x·a)·b)` with `a` `[in, rank]` (= `downᵀ`) and `b` `[rank, out]` (= `upᵀ` with
/// `alpha/rank` already folded in at `b`'s own dtype). Deliberately the two-small-matmul form and
/// never the `[out, in]` product, so the residual costs `rank·(in+out)` elements rather than a
/// second copy of the projection — the property that lets the same install ride a packed tier
/// unchanged when one lands.
///
/// **The base weight is never mutated.** That is the candle twin of the MLX lane's decision
/// (`mlx_gen_minimax_h3::adapters` installs over `AdaptableLinear` and never merges), and it is what
/// makes the fold strength independent of the quant tier: a merge would have to dequantize.
#[derive(Debug, Clone)]
pub struct LoraResidual {
    /// `[in, rank]` — `downᵀ`, at the factor's loaded dtype.
    a: Tensor,
    /// `[rank, out]` — `upᵀ` scaled by `alpha/rank`, at the factor's loaded dtype.
    b: Tensor,
    /// The caller's per-adapter strength (`AdapterSpec::scale`), applied at the residual.
    scale: f64,
}

impl LoraResidual {
    /// `a`'s dtype — the loaded factor dtype. Read by
    /// `adapters::tests::the_installed_fold_keeps_the_factor_dtype`, which is the **only** thing
    /// that can see a stray `to_dtype` in the install: every published turbo fold is an exact power
    /// of two, so bf16 and f32 products are bit-identical and no numeric assertion can distinguish
    /// them.
    pub fn dtype(&self) -> DType {
        self.b.dtype()
    }

    /// The `(in, rank)` of the down factor and the `(rank, out)` of the up factor, for a guard that
    /// wants to check the installed shapes without reaching into the tensors.
    pub fn shapes(&self) -> ((usize, usize), (usize, usize)) {
        let a = self.a.dims();
        let b = self.b.dims();
        ((a[0], a[1]), (b[0], b[1]))
    }

    /// The per-adapter strength this residual folds at.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

/// `y = x · w` for a `[k, n]` factor and an `[.., k]` activation, folding every leading dim into the
/// GEMM's `M`.
///
/// **Never `broadcast_matmul`.** candle materializes a broadcast rhs through `.contiguous()`, so a
/// 2-D factor against a `[N, S, k]` activation is physically copied `N` times; the shared
/// `candle_gen::quant::adapt` seam carries the measurement that made that pathological (a 5.4 GB
/// copy per leg). Flattening is mathematically identical and issues one large-`M` GEMM.
fn apply_factor(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let k = *dims.last().expect("apply_factor: x has no axes");
    let rows = x.elem_count() / k;
    let n = w.dim(1)?;
    let mut out_dims = dims;
    *out_dims.last_mut().expect("apply_factor: x has no axes") = n;
    Ok(x.reshape((rows, k))?
        .contiguous()?
        .matmul(w)?
        .reshape(out_dims)?)
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight — an `nn.Linear(..., bias=False)` — plus any
/// [`LoraResidual`] stacked onto it.
#[derive(Debug, Clone)]
pub struct LinearNoBias {
    weight: Tensor,
    /// Forward-time additive residuals, applied in push order. Empty on every un-adapted render, in
    /// which case [`Self::forward`] is byte-identical to the bare `linear_nb`.
    adapters: Vec<LoraResidual>,
}

impl LinearNoBias {
    /// Load `{prefix}.weight` at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
            adapters: Vec::new(),
        })
    }

    /// `y = x · Wᵀ` over the last axis of `x`, plus every attached residual.
    ///
    /// The residual runs at `x`'s dtype — matching [`linear_nb`], which upcasts the base weight the
    /// same way — and is narrowed to the host output dtype before the add, exactly as MLX's
    /// `AdaptableLinear::apply_adapters` does. A zero-scale residual is skipped entirely, so a
    /// disabled adapter is byte-identical to the bare base.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = linear_nb(x, &self.weight)?;
        let yd = y.dtype();
        let xd = x.dtype();
        for ad in &self.adapters {
            if ad.scale == 0.0 {
                continue;
            }
            let r = apply_factor(&apply_factor(x, &ad.a.to_dtype(xd)?)?, &ad.b.to_dtype(xd)?)?;
            y = (y + (r * ad.scale)?.to_dtype(yd)?)?;
        }
        Ok(y)
    }

    /// Attach a forward-time LoRA residual. `a` is `[in, rank]`, `b` is `[rank, out]` with
    /// `alpha/rank` already folded in. Multiple pushes stack in order.
    ///
    /// Shape-checked against the projection's own `[out, in]`: a factor pair that would broadcast
    /// into the wrong contraction is refused here rather than producing a runnable, wrong model.
    pub fn push_lora(&mut self, a: Tensor, b: Tensor, scale: f64) -> Result<()> {
        let (out, inn) = (self.weight.dim(0)?, self.weight.dim(1)?);
        if a.rank() != 2 || b.rank() != 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adapter: LoRA factors must be 2-D, got a={:?} b={:?}",
                a.dims(),
                b.dims()
            )));
        }
        let (a0, a1) = (a.dim(0)?, a.dim(1)?);
        let (b0, b1) = (b.dim(0)?, b.dim(1)?);
        if a0 != inn || b1 != out || a1 != b0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adapter: factors a={:?} b={:?} do not compose onto a [{out}, {inn}] \
                 projection — expected a [{inn}, rank] and b [rank, {out}]",
                a.dims(),
                b.dims()
            )));
        }
        self.adapters.push(LoraResidual { a, b, scale });
        Ok(())
    }

    /// Drop every attached residual, reverting to the bare base.
    pub fn clear_adapters(&mut self) {
        self.adapters.clear();
    }

    /// The residuals attached to this projection, in push order.
    pub fn adapters(&self) -> &[LoraResidual] {
        &self.adapters
    }

    /// Whether any residual is attached.
    pub fn is_adapted(&self) -> bool {
        !self.adapters.is_empty()
    }

    /// The projection's logical `(out_features, in_features)`.
    pub fn base_shape(&self) -> Result<(usize, usize)> {
        Ok((self.weight.dim(0)?, self.weight.dim(1)?))
    }

    /// Device bytes this projection holds — the frozen base plus every attached residual factor.
    pub fn nbytes(&self) -> usize {
        let base = self.weight.elem_count() * self.weight.dtype().size_in_bytes();
        base + self
            .adapters
            .iter()
            .map(|ad| {
                ad.a.elem_count() * ad.a.dtype().size_in_bytes()
                    + ad.b.elem_count() * ad.b.dtype().size_in_bytes()
            })
            .sum::<usize>()
    }

    /// The one tensor name this projection consumes.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}

/// Full self-attention with **affine per-head q/k RMSNorm** and MM-RoPE.
///
/// # qk-norm placement and shape
///
/// `norm_q` / `norm_k` are `nn.RMSNorm(dim_head, eps=qk_norm_eps)` — normalizing over a single
/// **head's** 128 channels, not over the 7168-wide projection — and they run **after** the
/// `[B, S, H·D] → [B, S, H, D]` unflatten and **before** the rotary. All three of those choices are
/// invisible in the tensor index, which only shows a `[128]` vector:
///
/// * normalizing the flat projection instead would mix all 56 heads into one RMS;
/// * running the rotary first would rotate un-normalized vectors, changing every attention score.
///
/// The video VAE's equivalent is non-affine and leaves **no tensors at all**, which is why
/// `tests/dit_parity.rs` gates this on *behaviour* rather than on the presence of `norm_q.weight`.
#[derive(Debug, Clone)]
pub struct DitAttention {
    to_q: LinearNoBias,
    to_k: LinearNoBias,
    to_v: LinearNoBias,
    to_out: LinearNoBias,
    norm_q: Tensor,
    norm_k: Tensor,
    heads: usize,
    head_dim: usize,
    qk_norm_eps: f64,
}

impl DitAttention {
    /// Load `{prefix}` (e.g. `transformer_blocks.0.attn`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3DitConfig,
        dtype: DType,
    ) -> Result<Self> {
        let expect = |t: &Tensor, want: &[usize], what: &str| -> Result<()> {
            if t.dims() != want {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit {prefix}.{what}: expected {want:?}, got {:?}",
                    t.dims()
                )));
            }
            Ok(())
        };
        let norm_q = w
            .require(&format!("{prefix}.norm_q.weight"))?
            .to_dtype(dtype)?;
        let norm_k = w
            .require(&format!("{prefix}.norm_k.weight"))?
            .to_dtype(dtype)?;
        // A qk-norm sized to the projection rather than to a head is the plausible wrong shape,
        // and it broadcasts silently against `[B, S, H, D]` only when it is `[D]`.
        expect(&norm_q, &[cfg.attention_head_dim], "norm_q")?;
        expect(&norm_k, &[cfg.attention_head_dim], "norm_k")?;

        Ok(Self {
            to_q: LinearNoBias::from_weights(w, &format!("{prefix}.to_q"), dtype)?,
            to_k: LinearNoBias::from_weights(w, &format!("{prefix}.to_k"), dtype)?,
            to_v: LinearNoBias::from_weights(w, &format!("{prefix}.to_v"), dtype)?,
            to_out: LinearNoBias::from_weights(w, &format!("{prefix}.to_out.0"), dtype)?,
            norm_q,
            norm_k,
            heads: cfg.num_attention_heads,
            head_dim: cfg.attention_head_dim,
            qk_norm_eps: cfg.qk_norm_eps,
        })
    }

    /// Every tensor name this attention consumes — six.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v: Vec<String> = ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .flat_map(|p| LinearNoBias::names(&format!("{prefix}.{p}")))
            .collect();
        v.push(format!("{prefix}.norm_q.weight"));
        v.push(format!("{prefix}.norm_k.weight"));
        v
    }

    /// Resolve one adapter path segment run **relative to this attention** to its projection.
    ///
    /// The four spellings are the published diffusers ones, `to_out.0` included — that trailing `0`
    /// is an `nn.Sequential` index and therefore a real path segment, which is why the lookup takes
    /// a segment slice rather than a dotted string.
    ///
    /// `norm_q` / `norm_k` are deliberately **not** addressable: they are `[head_dim]` vectors, not
    /// projections, and nothing in the published turbo set targets them.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    /// Drop every residual on all four projections.
    pub fn clear_adapters(&mut self) {
        self.to_q.clear_adapters();
        self.to_k.clear_adapters();
        self.to_v.clear_adapters();
        self.to_out.clear_adapters();
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.to_q.nbytes()
            + self.to_k.nbytes()
            + self.to_v.nbytes()
            + self.to_out.nbytes()
            + self.norm_q.elem_count() * self.norm_q.dtype().size_in_bytes()
            + self.norm_k.elem_count() * self.norm_k.dtype().size_in_bytes()
    }

    /// `rope` is `None` for the token refiner, which runs **without** any positional embedding.
    pub fn forward(&self, x: &Tensor, rope: Option<(&MmRope, &MmRopeTables)>) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit attention: expected [B, S, hidden], got {s:?}"
            )));
        }
        let (b, seq) = (s[0], s[1]);
        let (h, d) = (self.heads, self.head_dim);

        // The published checkpoint ships `to_q`/`to_k`/`to_v` already split, so this applies NO
        // fused-QKV transform. See `crate::dit::qkv` for which transform produced them and why
        // that must be asserted rather than assumed.
        let q = self.to_q.forward(x)?.reshape((b, seq, h, d))?;
        let k = self.to_k.forward(x)?.reshape((b, seq, h, d))?;
        let v = self.to_v.forward(x)?.reshape((b, seq, h, d))?;

        // qk-norm first, rotary second.
        let q = rms_weighted(&q, &self.norm_q, self.qk_norm_eps)?;
        let k = rms_weighted(&k, &self.norm_k, self.qk_norm_eps)?;
        let (q, k) = match rope {
            Some((rope, tables)) => (rope.apply(&q, tables)?, rope.apply(&k, tables)?),
            None => (q, k),
        };

        // MiniMax-H3 packs one request into a single attention document: no mask, not causal.
        let qh = q.transpose(1, 2)?.contiguous()?;
        let kh = k.transpose(1, 2)?.contiguous()?;
        let vh = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = sdpa(&qh, &kh, &vh, scale)?;

        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, h * d))?;
        self.to_out.forward(&out)
    }
}

/// Bias-free SwiGLU feed-forward: `w2( silu(gate) · value )`.
///
/// **The published `ff.net.0.proj` emits `[value | gate]`**, so the gate is the SECOND half — the
/// split is delegated to [`crate::layout::split_gate_value`], the crate's single implementation.
/// `convert_minimax_h3_to_diffusers.py` swaps the halves of the DiT's `mlp.fc1` on the way in
/// exactly as it does the video VAE's `ff.w1`, and reading the first half as the gate computes
/// `w2( silu(value) · gate )` — shape-identical, and the sc-18740 defect.
#[derive(Debug, Clone)]
pub struct DitFeedForward {
    proj: LinearNoBias,
    out: LinearNoBias,
}

impl DitFeedForward {
    /// Load `{prefix}` (e.g. `transformer_blocks.0.ff`).
    pub fn from_weights(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: LinearNoBias::from_weights(w, &format!("{prefix}.net.0.proj"), dtype)?,
            out: LinearNoBias::from_weights(w, &format!("{prefix}.net.2"), dtype)?,
        })
    }

    /// The two tensor names the feed-forward consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        let mut v = LinearNoBias::names(&format!("{prefix}.net.0.proj")).to_vec();
        v.extend(LinearNoBias::names(&format!("{prefix}.net.2")));
        v
    }

    /// Resolve one adapter path segment run **relative to this feed-forward** to its projection.
    ///
    /// `net.0.proj` and `net.2` keep their `nn.Sequential` indices, so both are multi-segment.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["net", "0", "proj"] => Some(&mut self.proj),
            ["net", "2"] => Some(&mut self.out),
            _ => None,
        }
    }

    /// Drop every residual on both projections.
    pub fn clear_adapters(&mut self) {
        self.proj.clear_adapters();
        self.out.clear_adapters();
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        self.proj.nbytes() + self.out.nbytes()
    }

    /// `w2( silu(gate) · value )`.
    ///
    /// The `[1, S, 2·ffn_dim]` intermediate this writes is the **widest single tensor in the whole
    /// model** — `[1, 94000, 28672]` at a 15 s render — which is the element-count ceiling worth
    /// watching on this lane rather than anything in the attention (sc-17152).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.proj.forward(x)?;
        let (gate, value) = split_gate_value(&h)?;
        self.out.forward(&(silu(&gate)?.mul(&value)?))
    }
}

/// Affine RMSNorm as a loadable block-level norm (`norm1` / `norm2` / `final_norm` /
/// `norm_out.norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Load `{prefix}.weight` at `dtype`.
    pub fn from_weights(w: &Weights, prefix: &str, eps: f64, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
            eps,
        })
    }

    /// `weight · x / sqrt(mean(x²) + eps)` over the last axis.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_weighted(x, &self.weight, self.eps)
    }

    /// The one tensor name this norm consumes.
    pub fn names(prefix: &str) -> [String; 1] {
        [format!("{prefix}.weight")]
    }
}
