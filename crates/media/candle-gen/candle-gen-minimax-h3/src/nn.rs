//! Small candle neural-net helpers — the candle twins of the `mlx_gen::nn` ops the MLX MiniMax-H3
//! port uses.
//!
//! Kept composable (no fused-kernel dispatch) so the numerics match the MLX lane on CPU and CUDA
//! alike. Every norm computes in **f32** and casts back, mirroring the reference's f32 norm islands.
//! The one exception is attention ([`sdpa_planned`]): a `flash-attn` CUDA build takes the fused
//! FlashAttention-2 kernel for bf16/f16, the way MLX's fused SDPA already streams the scores
//! (sc-26691).
//!
//! The one place this crate deliberately does *not* reach for a candle built-in is
//! `alias_free::transposed_conv1d`: see that function for why every transposed convolution
//! in the audio decoder runs as zero-insert + forward `conv1d` on both backends.

use candle_gen::attention::{AttentionBudget, AttentionPlan};
use candle_gen::candle_core::{DType, Tensor, D};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::gen_core::CancelFlag;
use candle_gen::Result;

/// SiLU / swish: `x · sigmoid(x)`, in the input dtype.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok((x * sigmoid(x)?)?)
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight, no bias, over the **last** axis of `x` (any
/// leading dims).
///
/// Flattens the leading dims to one matmul then restores them — candle's `Linear::forward` only
/// special-cases ranks ≤ 4, and this crate projects rank-5 NCTHW tensors through `post_quant_conv`.
pub fn linear_nb(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_dim = *dims.last().expect("linear_nb: x has no axes");
    let out_dim = w.dim(0)?;
    let rows = x.elem_count() / in_dim;
    let flat = x.reshape((rows, in_dim))?.contiguous()?;
    let w = w.to_dtype(x.dtype())?;
    let y = flat.matmul(&w.t()?.contiguous()?)?;
    let mut out_dims = dims;
    *out_dims.last_mut().expect("linear_nb: x has no axes") = out_dim;
    Ok(y.reshape(out_dims)?)
}

/// `y = x · Wᵀ + b` over the last axis (see [`linear_nb`]). `b` is `[out]`, upcast to `x`'s dtype.
pub fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(linear_nb(x, w)?.broadcast_add(&b.to_dtype(x.dtype())?)?)
}

/// **Weightless** RMS norm over the last axis — `x / sqrt(mean(x²) + eps)`.
///
/// This is the `qk_norm_affine = false` q/k normalization the video VAE's attention applies, and it
/// leaves **no tensor behind**: a port that skipped it would still load every published tensor and
/// pass an exhaustive key-mapping proof (`crate::config`).
///
/// Note the epsilon is INSIDE the square root, matching both the reference and MLX's
/// `mlx_rs::fast::rms_norm`. Moving it outside is a sub-1% divergence that no cross-backend
/// tolerance in this crate can resolve; only this shared formulation keeps the two lanes honest.
pub fn rms_weightless(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xf.broadcast_div(&(ms + eps)?.sqrt()?)?.to_dtype(dtype)?)
}

/// **Weighted** RMS norm over the last axis — `weight · x / sqrt(mean(x²) + eps)`.
///
/// `weight` is `[dim]`. Computed in f32, returned in `x`'s dtype.
pub fn rms_weighted(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(ms + eps)?.sqrt()?)?;
    Ok(normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?)
}

/// LayerNorm **with** affine parameters over the last axis (population variance) — the video
/// decoder's `norm_out`, which unlike its RMSNorms really does ship a `bias`.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .broadcast_add(&bias.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?)
}

/// Scaled-dot-product attention over `[B, H, S, D]` q/k/v, with no request cancel flag.
///
/// The video VAE decoder's attention is dense and unmasked (`causal_decoder: false`), so no mask is
/// taken. Routed through [`sdpa_planned`] with an un-flagged plan — see that function for the kernel
/// choice and the i32 budget.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    sdpa_planned(q, k, v, scale, None)
}

/// Unmasked scaled-dot-product attention over `[B, H, S, D]` q/k/v (sc-26691).
///
/// * **CUDA + `flash-attn` build + bf16/f16** — the fused FlashAttention-2 kernel
///   ([`candle_gen::sdpa_dispatch_bhsd`]). No `[B, H, Sq, Sk]` scores tensor exists, so the i32
///   hazard below does not arise; the cancel flag is checked once per call (one call = one layer).
/// * **everything else** (CPU, Metal, a non-flash build, f32) — the materialized, query-chunked
///   fallback (`sdpa_materialized`) under the shared i32-overflow-safe budget (sc-9116): with
///   `cancel` attached a tripped flag stops the next query-chunk launch and surfaces as
///   [`CandleError::Canceled`](candle_gen::CandleError::Canceled). On CUDA that check runs at
///   enqueue time; the DiT's per-layer synchronize (`dit::model`) is what keeps the host from
///   queueing more than one layer ahead of the device. For bf16/f16 operands it computes the
///   scores and the softmax in **f32**, as the fused kernel and MLX's fused SDPA do (SW-7091).
///
/// The budget matters on the fallback even at modest geometries: a real decode carries `T·H·W + 5`
/// tokens for a 16×-downsampled canvas, and that path materializes `[B, H, block, Sk]`.
pub fn sdpa_planned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    cancel: Option<&CancelFlag>,
) -> Result<Tensor> {
    let plan = attention_plan(cancel);
    let (b, h, sq, _) = q.dims4()?;
    let sk = k.dim(2)?;
    let (query_rows, score_elements) = materializing_chunk_shape(b, h, sq, sk, plan.budget)?;
    let force_materialized = force_materialized_attention();
    let kernel = candle_gen::fused_attention::select_attention_kernel(q, k, v, None);
    let materialized =
        force_materialized || matches!(kernel, candle_gen::AttentionKernel::Budgeted(_));
    tracing::debug!(
        route = if materialized {
            "materialized-query-chunked"
        } else {
            "flash"
        },
        forced = force_materialized,
        ?kernel,
        query_rows,
        score_elements = plan.budget.max_score_elements(),
        "minimax-h3: attention route"
    );
    let out = if materialized {
        sdpa_materialized(q, k, v, scale, plan)
    } else {
        candle_gen::sdpa_dispatch_bhsd(
            q,
            k,
            v,
            scale,
            None,
            candle_gen::candle_nn::ops::softmax_last_dim,
            plan,
        )
    }?;
    record_attention_execution(
        materialized,
        if materialized { query_rows } else { sq as u64 },
        if materialized { score_elements } else { 0 },
    )?;
    Ok(out)
}

/// The materialized, query-chunked attention fallback over `[B, H, S, D]` (any strides).
///
/// **f32** operands take the shared `sdpa_planned_bhsd` verbatim. **bf16/f16** operands keep the
/// scores, the `scale` and the softmax in f32 and round only the probabilities to the value dtype
/// for `probs · v` — the FlashAttention-2 kernel's numerics (f32 `S`, f32 softmax, half `P`, f32
/// GEMM accumulation), so both CUDA routes share them. MLX's fused SDPA likewise never rounds the
/// scores (SW-7091).
///
/// The half-dtype score path this replaces rounded `q·kᵀ` to bf16, rounded again on `· scale`, and
/// softmaxed those: bf16 keeps 8 mantissa bits, so a logit of 16 carries an absolute error up to
/// 1/16, a ~6% error in its probability, on every key of every layer. The CUDA DiT gate (built
/// without `flash-attn`, so on this route) measured a step-3 block-49 residual error of rel-L2
/// 2.0e-1 against the float32 diffusers reference of the published MiniMax-H3-Image DiT, twice
/// MLX's 9.9e-2 on the same inputs, with the two backends level through block 24.
///
/// Memory: `kᵀ` is held once at f32 (twice its half-dtype copy) and each query chunk is upcast on
/// its own; per score element at most 8 bytes are live (f32 scores + f32 probabilities, then f32 +
/// half probabilities), inside the 16 bytes per score `h3_memory::window` prices for this route.
fn sdpa_materialized(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    plan: AttentionPlan<'_>,
) -> Result<Tensor> {
    use candle_gen::candle_nn::ops::softmax_last_dim;
    if q.dtype() == DType::F32 {
        return candle_gen::attention::sdpa_planned_bhsd(
            q,
            k,
            v,
            scale,
            None,
            softmax_last_dim,
            plan,
        );
    }
    let (b, h, sq, _) = q.dims4()?;
    let sk = k.dim(2)?;
    // A cast writes a contiguous tensor, so the f32 `kᵀ` and each f32 query chunk are their own
    // GEMM operands with no separate `contiguous` copy.
    let k_t = k.transpose(2, 3)?.to_dtype(DType::F32)?;
    let v = v.contiguous()?;
    let attend = |start: usize, len: usize| -> Result<Tensor> {
        let probs = {
            let q = q.narrow(2, start, len)?.to_dtype(DType::F32)?;
            let scores = (q.matmul(&k_t)? * scale)?;
            softmax_last_dim(&scores)?
        };
        Ok(probs.to_dtype(v.dtype())?.matmul(&v)?)
    };
    let block = plan.budget.query_block_rows((b * h * sk) as u64, sq as u64) as usize;
    if block >= sq {
        // One pass, like the shared helper's unbounded fast path: no per-chunk cancel check.
        return attend(0, sq);
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < sq {
        if plan.is_cancelled() {
            return Err(candle_gen::CandleError::Canceled);
        }
        let len = block.min(sq - start);
        blocks.push(attend(start, len)?);
        start += len;
    }
    Ok(Tensor::cat(&blocks, 2)?)
}

/// Masked TE attention always uses the shared materializing plan. Record its actual largest
/// completed query chunk only after the entire operation succeeds, just like unmasked attention.
pub(crate) fn sdpa_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: &Tensor,
) -> Result<Tensor> {
    let plan = attention_plan(None);
    let (b, h, sq, _) = q.dims4()?;
    let (query_rows, score_elements) = materializing_chunk_shape(b, h, sq, k.dim(2)?, plan.budget)?;
    let out = candle_gen::attention::sdpa_planned_bhsd(
        q,
        k,
        v,
        scale,
        Some(mask),
        candle_gen::candle_nn::ops::softmax_last_dim,
        plan,
    )?;
    record_attention_execution(true, query_rows, score_elements)?;
    tracing::debug!(
        route = "materialized-query-chunked",
        masked = true,
        query_rows,
        score_elements,
        budget_score_elements = plan.budget.max_score_elements(),
        "minimax-h3: completed attention route"
    );
    Ok(out)
}

/// Derive the executed chunk maximum using the same planner as shared SDPA. A ragged last
/// chunk cannot exceed the first chunk; an all-rows call reports its actual size, not the budget.
fn materializing_chunk_shape(
    b: usize,
    h: usize,
    sq: usize,
    sk: usize,
    budget: AttentionBudget,
) -> Result<(u64, u64)> {
    let row = b
        .checked_mul(h)
        .and_then(|v| v.checked_mul(sk))
        .ok_or_else(|| candle_gen::CandleError::Msg("H3 attention row size overflow".into()))?;
    if row == 0 || sq == 0 || row as u64 > budget.max_score_elements() {
        return Err(candle_gen::CandleError::Msg(
            "H3 minimum materializing attention row exceeds selected workspace budget".into(),
        ));
    }
    let query_rows = budget.query_block_rows(row as u64, sq as u64);
    let score_elements = query_rows
        .checked_mul(row as u64)
        .ok_or_else(|| candle_gen::CandleError::Msg("H3 attention chunk size overflow".into()))?;
    Ok((query_rows, score_elements))
}

fn record_attention_execution(
    materialized: bool,
    query_rows: u64,
    score_elements: u64,
) -> Result<()> {
    ATTENTION_EXECUTION.with(|r| {
        let mut current = r.borrow_mut();
        let mut next = *current;
        let counter = if materialized {
            &mut next.materialized_calls
        } else {
            &mut next.flash_calls
        };
        *counter = counter.checked_add(1).ok_or_else(|| {
            candle_gen::CandleError::Msg("H3 attention execution counter overflow".into())
        })?;
        next.maximum_query_rows = next.maximum_query_rows.max(query_rows);
        next.maximum_materialized_score_elements =
            next.maximum_materialized_score_elements.max(score_elements);
        *current = next;
        Ok(())
    })
}

/// The exact score-element allowance shared by CUDA admission, masked TE attention and FFN
/// chunking. A selected smaller allowance is honored; a larger value cannot weaken the
/// production fallback's bound. This is a byte bound, independent of measurement provenance.
pub fn attention_budget() -> AttentionBudget {
    attention_budget_for_context(crate::memory_strategy::active_context().as_ref())
}

pub fn attention_budget_for_context(
    context: Option<&candle_gen::gen_core::MemoryRunContext>,
) -> AttentionBudget {
    let default = candle_gen::gen_core::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;
    let elements = context
        .and_then(|c| c.selection.parameters.attention_chunk_size)
        .map(u64::from)
        .unwrap_or(default)
        .min(default)
        .max(1);
    AttentionBudget::from_score_elements(elements, false)
}

/// Request-scoped observations; only successfully completed attention calls are counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionExecutionReport {
    pub flash_calls: u64,
    pub materialized_calls: u64,
    pub maximum_query_rows: u64,
    pub maximum_materialized_score_elements: u64,
}
thread_local! {
    static ATTENTION_EXECUTION: std::cell::RefCell<AttentionExecutionReport> = std::cell::RefCell::default();
}
pub fn reset_attention_execution_report() {
    ATTENTION_EXECUTION.with(|r| *r.borrow_mut() = Default::default());
}
pub fn attention_execution_report() -> AttentionExecutionReport {
    ATTENTION_EXECUTION.with(|r| *r.borrow())
}

fn force_materialized_attention() -> bool {
    #[cfg(any(test, feature = "model-inference-proof"))]
    {
        std::env::var("SCENEWORKS_H3_ATTENTION_ROUTE").is_ok_and(|route| route == "materialized")
    }
    #[cfg(not(any(test, feature = "model-inference-proof")))]
    {
        false
    }
}

/// The bounded fallback plan carries cancellation between query chunks.
fn attention_plan(cancel: Option<&CancelFlag>) -> AttentionPlan<'_> {
    let plan = AttentionPlan::budgeted(attention_budget());
    match cancel {
        Some(c) => plan.with_cancel(c),
        None => plan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn materializing_receipt_uses_actual_chunk_extent_and_refuses_unbounded_rows() {
        let budget = AttentionBudget::from_score_elements(50, false);
        // [B=1,H=2,Q=10,K=7]: 3-row chunks plus a one-row tail, not 50 score elements.
        assert_eq!(
            materializing_chunk_shape(1, 2, 10, 7, budget).unwrap(),
            (3, 42)
        );
        assert_eq!(
            materializing_chunk_shape(1, 2, 2, 7, budget).unwrap(),
            (2, 28)
        );
        assert!(materializing_chunk_shape(1, 8, 2, 7, budget).is_err());
        assert!(materializing_chunk_shape(usize::MAX, 2, 2, 7, budget).is_err());
        assert!(materializing_chunk_shape(1, 2, 0, 7, budget).is_err());
        assert!(materializing_chunk_shape(1, 2, 2, 0, budget).is_err());
    }

    #[test]
    fn attention_receipt_counter_overflow_does_not_partially_update_the_report() {
        reset_attention_execution_report();
        record_attention_execution(true, 3, 42).unwrap();
        record_attention_execution(false, 10, 0).unwrap();
        assert_eq!(
            attention_execution_report(),
            AttentionExecutionReport {
                flash_calls: 1,
                materialized_calls: 1,
                maximum_query_rows: 10,
                maximum_materialized_score_elements: 42,
            }
        );
        ATTENTION_EXECUTION.with(|r| r.borrow_mut().materialized_calls = u64::MAX);
        let before = attention_execution_report();
        assert!(record_attention_execution(true, 100, 1000).is_err());
        assert_eq!(attention_execution_report(), before);
        reset_attention_execution_report();
    }

    /// The request flag reaches the plan the budgeted fallback checks between query chunks.
    #[test]
    fn the_attention_plan_carries_the_request_cancel_flag() {
        let cancel = CancelFlag::new();
        assert!(!attention_plan(Some(&cancel)).is_cancelled());
        cancel.cancel();
        assert!(attention_plan(Some(&cancel)).is_cancelled());
        assert!(!attention_plan(None).is_cancelled());
    }

    /// On the CPU the dispatcher refuses the fused kernel, so `sdpa_planned` is the budgeted path
    /// verbatim — strided `[B, H, S, D]` views included (the DiT now passes transposes, not copies).
    #[test]
    fn sdpa_planned_on_cpu_is_the_budgeted_path_for_strided_views() {
        let dev = Device::Cpu;
        let bshd = |seed: f32| {
            Tensor::arange(0f32, 2.0 * 6.0 * 8.0, &dev)
                .unwrap()
                .affine(0.01, f64::from(seed))
                .unwrap()
                .sin()
                .unwrap()
                .reshape((1, 6, 2, 8))
                .unwrap()
        };
        let (q, k, v) = (bshd(0.1), bshd(0.7), bshd(1.3));
        let (qh, kh, vh) = (
            q.transpose(1, 2).unwrap(),
            k.transpose(1, 2).unwrap(),
            v.transpose(1, 2).unwrap(),
        );
        assert!(!qh.is_contiguous());
        assert!(matches!(
            candle_gen::fused_attention::select_attention_kernel(&qh, &kh, &vh, None),
            candle_gen::AttentionKernel::Budgeted(_)
        ));
        let got = sdpa_planned(&qh, &kh, &vh, 0.35, None).unwrap();
        let want = candle_gen::sdpa_budgeted_bhsd(
            &qh.contiguous().unwrap(),
            &kh.contiguous().unwrap(),
            &vh.contiguous().unwrap(),
            0.35,
            None,
            candle_gen::candle_nn::ops::softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )
        .unwrap();
        assert_eq!(flat(&got), flat(&want));
    }

    /// SW-7091: half-dtype attention keeps its scores and softmax in f32. The CPU backend has no
    /// bf16 GEMM, so this runs f16 with logits large enough that an f16 score's rounding step
    /// moves the softmax: the old half-score route lands far from the f32
    /// computation over the same rounded operands, while this route stays at the half
    /// probabilities' own rounding — single-pass and query-chunked alike.
    #[test]
    fn half_attention_computes_its_scores_in_f32() {
        let dev = Device::Cpu;
        let bhsd = |seed: f64, gain: f64| {
            Tensor::arange(0f32, 2.0 * 9.0 * 16.0, &dev)
                .unwrap()
                .affine(0.37, seed)
                .unwrap()
                .sin()
                .unwrap()
                .affine(gain, 0.0)
                .unwrap()
                .reshape((1, 2, 9, 16))
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        let (q, k, v) = (bhsd(0.1, 6.0), bhsd(0.7, 6.0), bhsd(1.3, 1.0));
        let scale = 0.25;
        let f32_of = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
        let want = flat(
            &candle_gen::candle_nn::ops::softmax_last_dim(
                &(f32_of(&q).matmul(&f32_of(&k).t().unwrap()).unwrap() * scale).unwrap(),
            )
            .unwrap()
            .matmul(&f32_of(&v))
            .unwrap(),
        );
        let err = |got: &Tensor| {
            flat(&f32_of(got))
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        };
        let half_scores = candle_gen::attention::sdpa_planned_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            candle_gen::candle_nn::ops::softmax_last_dim,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(1 << 20, false)),
        )
        .unwrap();
        let whole = sdpa_materialized(
            &q,
            &k,
            &v,
            scale,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(1 << 20, false)),
        )
        .unwrap();
        // 2 heads x 9 keys = 18 elements per query row: 2-row chunks plus a one-row tail.
        let chunked = sdpa_materialized(
            &q,
            &k,
            &v,
            scale,
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(36, false)),
        )
        .unwrap();
        assert_eq!(whole.dtype(), DType::F16);
        let (old, new, chunk) = (err(&half_scores), err(&whole), err(&chunked));
        println!("max |error|: half scores {old:.3e}, f32 scores {new:.3e}, chunked {chunk:.3e}");
        assert!(
            new < 2e-3 && chunk < 2e-3,
            "f32 scores: {new:.3e} / chunked {chunk:.3e}"
        );
        assert!(
            old > 4.0 * new,
            "half scores {old:.3e} vs f32 scores {new:.3e}"
        );

        let cancel = CancelFlag::new();
        cancel.cancel();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(36, false))
            .with_cancel(&cancel);
        assert!(matches!(
            sdpa_materialized(&q, &k, &v, scale, plan),
            Err(candle_gen::CandleError::Canceled)
        ));
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn weightless_rms_normalizes_to_unit_mean_square() {
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], (1, 4), &Device::Cpu).unwrap();
        let y = rms_weightless(&x, 1e-5).unwrap();
        // rms = sqrt(mean(9,16,0,0)) = 2.5 -> [1.2, 1.6, 0, 0].
        let v = flat(&y);
        assert!(
            (v[0] - 1.2).abs() < 1e-4 && (v[1] - 1.6).abs() < 1e-4,
            "{v:?}"
        );
        let ms: f32 = v.iter().map(|a| a * a).sum::<f32>() / 4.0;
        assert!(
            (ms - 1.0).abs() < 1e-3,
            "mean square should be ~1, got {ms}"
        );
    }

    /// **The epsilon lives INSIDE the square root.** `x / sqrt(mean(x²) + eps)`, never
    /// `x / (sqrt(mean(x²)) + eps)`.
    ///
    /// This is the one sub-percent divergence class no tolerance in this crate can adjudicate: at
    /// the shipped `norm_eps = 1e-5` the two forms differ by ~5e-6 relative, which
    /// `tests/video_vae_parity.rs` measures at 5.9e-6 through the whole decoder — visible, but
    /// under that suite's 1e-5 bound and four orders under the cross-backend floor. Cosine, norm
    /// and checksum assertions are blind to it by construction.
    ///
    /// So it is pinned structurally instead, against a hand-computed closed form, with an epsilon
    /// large enough that the two placements are unambiguously different numbers. The second half of
    /// the test is the load-bearing one: it asserts the implementation is NOT the outside-the-root
    /// variant, which is what makes this a gate rather than a restatement.
    #[test]
    fn rms_norm_puts_epsilon_inside_the_square_root() {
        let dev = Device::Cpu;
        // mean(x²) = (9 + 16) / 4 = 6.25, so sqrt(ms) = 2.5 exactly.
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let eps = 0.25f64;
        let inside = 3.0 / (6.25f32 + 0.25).sqrt();
        let outside = 3.0 / (2.5f32 + 0.25);
        assert!(
            (inside - outside).abs() > 1e-2,
            "the probe must separate the two placements: {inside} vs {outside}"
        );

        for got in [
            flat(&rms_weightless(&x, eps).unwrap())[0],
            flat(&rms_weighted(&x, &Tensor::ones(4, DType::F32, &dev).unwrap(), eps).unwrap())[0],
        ] {
            assert!(
                (got - inside).abs() < 1e-6,
                "epsilon must be added to the MEAN SQUARE before the root: expected {inside}, \
                 got {got}"
            );
            assert!(
                (got - outside).abs() > 1e-2,
                "this is the outside-the-root form ({outside}); no numeric parity gate in this \
                 crate can catch that, so it has to fail here"
            );
        }
    }

    /// The same rule for the decoder's output LayerNorm: `sqrt(var + eps)`, not `sqrt(var) + eps`.
    #[test]
    fn layer_norm_puts_epsilon_inside_the_square_root() {
        let dev = Device::Cpu;
        // [1,2,3,4]: mean 2.5, population variance 1.25.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let b = Tensor::zeros(4, DType::F32, &dev).unwrap();
        let eps = 0.75f64;
        let inside = -1.5 / (1.25f32 + 0.75).sqrt();
        let outside = -1.5 / (1.25f32.sqrt() + 0.75);
        assert!((inside - outside).abs() > 1e-2, "{inside} vs {outside}");

        let got = flat(&layer_norm(&x, &w, &b, eps).unwrap())[0];
        assert!(
            (got - inside).abs() < 1e-6,
            "expected {inside}, got {got} (outside-the-root would be {outside})"
        );
        assert!((got - outside).abs() > 1e-2);
    }

    /// The weighted form must actually apply the weight, per channel.
    #[test]
    fn weighted_rms_applies_the_weight() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &dev).unwrap();
        let plain = flat(&rms_weightless(&x, 1e-5).unwrap());
        let weighted = flat(&rms_weighted(&x, &w, 1e-5).unwrap());
        assert!((weighted[0] - plain[0]).abs() < 1e-6);
        assert!((weighted[1] - plain[1] * 2.0).abs() < 1e-5);
    }

    /// LayerNorm centres AND scales — an RMSNorm would leave the mean alone.
    #[test]
    fn layer_norm_centres_and_scales() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let b = Tensor::zeros(4, DType::F32, &dev).unwrap();
        let y = flat(&layer_norm(&x, &w, &b, 1e-5).unwrap());
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "layer norm must centre: {mean}");
        // Population variance of [1,2,3,4] is 1.25, so the first entry is (1-2.5)/sqrt(1.25).
        assert!((y[0] - (-1.5 / 1.25f32.sqrt())).abs() < 1e-4, "{y:?}");
    }

    /// `linear` handles a rank-5 input, which candle's own `Linear` does not.
    #[test]
    fn linear_projects_a_rank_five_tensor() {
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 2, 3, 4, 5), DType::F32, &dev).unwrap();
        let w = Tensor::ones((7, 5), DType::F32, &dev).unwrap();
        let b = Tensor::zeros(7, DType::F32, &dev).unwrap();
        let y = linear(&x, &w, &b).unwrap();
        assert_eq!(y.dims(), &[1, 2, 3, 4, 7]);
        assert!((flat(&y)[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn silu_is_x_times_sigmoid() {
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], 3, &Device::Cpu).unwrap();
        let y = flat(&silu(&x).unwrap());
        assert!(y[0].abs() < 1e-7);
        assert!((y[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        assert!((y[2] - -1.0 / (1.0 + 1.0f32.exp())).abs() < 1e-6);
    }
}
