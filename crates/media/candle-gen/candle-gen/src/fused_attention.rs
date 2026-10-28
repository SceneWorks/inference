//! Fused-attention dispatch (sc-26691): route an eligible `[B, H, S, D]` attention through the
//! **FlashAttention-2** CUDA kernel, and everything else through the materialized, query-chunked
//! [`crate::attention::sdpa_planned_bhsd`].
//!
//! # Why this exists
//!
//! [`crate::attention`]'s budgeted path materializes the `[B, H, block, Sk]` scores tensor. That is
//! the right guard for the i32 indexing hazard, but it is the wrong *kernel* for a very long joint
//! sequence: at MiniMax-H3's ~141k-row packed sequence (56 heads, `head_dim` 128) the 1e9 budget
//! splits every layer into ~1,100 query chunks, each re-reading all of K and V, which measured at
//! ≈6+ minutes per DiT forward. FlashAttention streams the scores through on-chip memory, so no
//! `[.., Sq, Sk]` tensor exists at all and the i32 hazard the budget guards never arises.
//!
//! # Selection is pure, and the refusals are typed
//!
//! [`flash_eligibility`] is a pure function of the build, the device kind, the dtype, the shapes and
//! whether a mask is present, so every refusal is unit-testable on the CPU lane without a CUDA
//! toolkit. The kernel is taken only when **all** of these hold:
//!
//! - the build enables `candle-gen/flash-attn` (it compiles CUTLASS kernels and needs nvcc);
//! - the tensors live on a CUDA device;
//! - the dtype is bf16 or f16 (the kernel has no f32 path — f32 callers keep their numerics);
//! - there is **no additive mask** — the wrapper exposes only dense / causal, and an arbitrary
//!   additive bias cannot be expressed, so a masked call keeps the budgeted path;
//! - q/k/v are rank 4, share the batch, K and V share their shape, the query heads are a multiple of
//!   the KV heads, and the common `head_dim` is a multiple of 8 and at most
//!   [`FLASH_MAX_HEAD_DIM`].
//!
//! # Stream ordering
//!
//! candle-flash-attn launches on CUDA stream 0 (the legacy default stream). candle-gen's devices
//! are built by `Device::new_cuda`, which uses the context's default stream, so the kernel is
//! ordered against every other op the media lane issues. A device created on its *own*
//! non-blocking stream (`Device::new_cuda_with_stream`, which only candle-llm's graph runner uses)
//! would NOT be ordered against it; no media provider builds one.
//!
//! # Cancellation
//!
//! The fused kernel is one launch per call, so a cancel cannot land *inside* it. The dispatcher
//! checks the plan's flag once before launching (returning [`crate::CandleError::Canceled`]); the
//! budgeted path keeps its between-chunk checks. Long sequences should additionally check between
//! layers, which is where a fused forward's cancel latency is bounded.

use candle_core::{DType, Tensor};

use crate::attention::{sdpa_planned_bhsd, AttentionPlan};

/// Largest `head_dim` the dispatcher hands to the fused kernel. candle-flash-attn compiles
/// kernels up to 512 at this rev, but the 256 ceiling is the long-standing FA2 envelope and covers
/// every media head dim in the tree (MiniMax-H3's DiT is 128, the Qwen3-VL tower 72).
pub const FLASH_MAX_HEAD_DIM: usize = 256;

/// `true` when this build compiled the fused kernel in (`candle-gen/flash-attn`).
pub const FLASH_ATTN_COMPILED: bool = cfg!(feature = "flash-attn");

/// Why one attention call did not take the fused kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashRefusal {
    /// The build does not enable `candle-gen/flash-attn`.
    NotCompiled,
    /// The tensors are not on a CUDA device.
    NotCuda,
    /// The dtype has no fused kernel (only bf16 / f16 do).
    Dtype(DType),
    /// The call carries an additive mask, which the fused kernel cannot express.
    Masked,
    /// The `head_dim` is not a multiple of 8, exceeds [`FLASH_MAX_HEAD_DIM`], or differs across
    /// q / k / v.
    HeadDim(usize),
    /// Rank, batch, K/V shape or head-group mismatch.
    Shape,
}

/// The kernel one call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKernel {
    /// candle-flash-attn's fused FA2 forward — scores never materialize.
    Flash,
    /// The materialized, query-chunked [`sdpa_planned_bhsd`], with the reason the fused kernel was
    /// refused.
    Budgeted(FlashRefusal),
}

impl AttentionKernel {
    /// A stable short name for logs.
    pub fn name(self) -> &'static str {
        match self {
            AttentionKernel::Flash => "flash_attn_v2",
            AttentionKernel::Budgeted(_) => "budgeted_sdpa",
        }
    }
}

/// The pure selection rule (see the module docs). `q`/`k`/`v` are the `[B, H, S, D]` dims.
pub fn flash_eligibility(
    compiled: bool,
    is_cuda: bool,
    dtype: DType,
    q: &[usize],
    k: &[usize],
    v: &[usize],
    masked: bool,
) -> AttentionKernel {
    use AttentionKernel::Budgeted;
    if !compiled {
        return Budgeted(FlashRefusal::NotCompiled);
    }
    if !is_cuda {
        return Budgeted(FlashRefusal::NotCuda);
    }
    if !matches!(dtype, DType::BF16 | DType::F16) {
        return Budgeted(FlashRefusal::Dtype(dtype));
    }
    if masked {
        return Budgeted(FlashRefusal::Masked);
    }
    let ([qb, qh, qs, qd], [kb, kh, ks, kd]) = match (q, k) {
        (&[a, b, c, d], &[e, f, g, h]) => ([a, b, c, d], [e, f, g, h]),
        _ => return Budgeted(FlashRefusal::Shape),
    };
    if v.len() != 4 || qb != kb || k != v || qs == 0 || ks == 0 || kh == 0 || qh % kh != 0 {
        return Budgeted(FlashRefusal::Shape);
    }
    if qd != kd || qd % 8 != 0 || qd == 0 || qd > FLASH_MAX_HEAD_DIM {
        return Budgeted(FlashRefusal::HeadDim(qd));
    }
    AttentionKernel::Flash
}

/// [`flash_eligibility`] for real tensors in this build.
pub fn select_attention_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> AttentionKernel {
    flash_eligibility(
        FLASH_ATTN_COMPILED,
        q.device().is_cuda(),
        q.dtype(),
        q.dims(),
        k.dims(),
        v.dims(),
        mask.is_some(),
    )
}

/// Scaled-dot-product attention over `q,k,v: [B, H, S, D]` (any strides whose last dim is
/// unit-stride), returning a contiguous `[B, H, Sq, D]`.
///
/// Takes the fused FlashAttention-2 kernel when [`select_attention_kernel`] says so, otherwise
/// exactly [`sdpa_planned_bhsd`] with the caller's `mask`, `softmax` and `plan` — so CPU, Metal,
/// non-flash builds, f32 and masked calls are byte-identical to calling that directly.
///
/// The fused path is **not** bitwise equal to the materialized one (different reduction order,
/// bf16 probabilities); the `#[ignore]`d `flash_matches_budgeted_on_cuda` measures the gap.
pub fn sdpa_dispatch_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    plan: AttentionPlan<'_>,
) -> crate::Result<Tensor> {
    match select_attention_kernel(q, k, v, mask) {
        AttentionKernel::Flash => {
            if plan.is_cancelled() {
                return Err(crate::CandleError::Canceled);
            }
            Ok(flash_bhsd(q, k, v, scale)?)
        }
        AttentionKernel::Budgeted(_) => sdpa_planned_bhsd(q, k, v, scale, mask, softmax, plan),
    }
}

/// The fused forward. The kernel wants `[B, S, H, D]`, which is a free transpose of a `[B, H, S, D]`
/// view (only the last dim must be unit-stride; the batch/row/head strides are passed through).
#[cfg(feature = "flash-attn")]
fn flash_bhsd(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
    let unit_last = |t: &Tensor| -> candle_core::Result<Tensor> {
        let t = t.transpose(1, 2)?;
        if t.stride().last() == Some(&1) {
            Ok(t)
        } else {
            t.contiguous()
        }
    };
    let out = candle_flash_attn::flash_attn(
        &unit_last(q)?,
        &unit_last(k)?,
        &unit_last(v)?,
        scale as f32,
        false,
    )?; // [B, Sq, H, D]
    out.transpose(1, 2)?.contiguous()
}

#[cfg(not(feature = "flash-attn"))]
fn flash_bhsd(_q: &Tensor, _k: &Tensor, _v: &Tensor, _scale: f64) -> candle_core::Result<Tensor> {
    // Unreachable: `flash_eligibility` refuses with `NotCompiled` first.
    candle_core::bail!("candle-gen: the fused attention kernel is not compiled into this build")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{AttentionBudget, ATTN_SCORES_BUDGET};
    use candle_core::Device;
    use candle_nn::ops::softmax_last_dim;

    const H3: [usize; 4] = [1, 56, 4096, 128];

    fn pick(dtype: DType, q: &[usize], k: &[usize], masked: bool) -> AttentionKernel {
        flash_eligibility(true, true, dtype, q, k, k, masked)
    }

    #[test]
    fn the_h3_dit_shape_takes_the_fused_kernel_in_bf16_and_f16() {
        assert_eq!(pick(DType::BF16, &H3, &H3, false), AttentionKernel::Flash);
        assert_eq!(pick(DType::F16, &H3, &H3, false), AttentionKernel::Flash);
        // Cross-attention: a different key length is fine.
        assert_eq!(
            pick(DType::BF16, &H3, &[1, 56, 300, 128], false),
            AttentionKernel::Flash
        );
        // Grouped-query: KV heads dividing the query heads is fine.
        assert_eq!(
            pick(DType::BF16, &H3, &[1, 8, 4096, 128], false),
            AttentionKernel::Flash
        );
        // The Qwen3-VL tower's 72 is a multiple of 8.
        let vit = [1, 16, 4032, 72];
        assert_eq!(pick(DType::BF16, &vit, &vit, false), AttentionKernel::Flash);
    }

    #[test]
    fn every_refusal_is_typed_and_ordered() {
        use AttentionKernel::Budgeted;
        let f =
            |compiled, cuda, dtype| flash_eligibility(compiled, cuda, dtype, &H3, &H3, &H3, false);
        assert_eq!(
            f(false, true, DType::BF16),
            Budgeted(FlashRefusal::NotCompiled)
        );
        assert_eq!(f(true, false, DType::BF16), Budgeted(FlashRefusal::NotCuda));
        assert_eq!(
            f(true, true, DType::F32),
            Budgeted(FlashRefusal::Dtype(DType::F32))
        );
        assert_eq!(
            pick(DType::BF16, &H3, &H3, true),
            Budgeted(FlashRefusal::Masked)
        );
    }

    #[test]
    fn shape_and_head_dim_guards_refuse_what_the_kernel_cannot_serve() {
        use AttentionKernel::Budgeted;
        let shape = Budgeted(FlashRefusal::Shape);
        // Rank 3 (the flat `[N, S, D]` shape) is not this dispatcher's.
        assert_eq!(pick(DType::BF16, &[56, 4096, 128], &H3, false), shape);
        // Batch mismatch.
        assert_eq!(pick(DType::BF16, &H3, &[2, 56, 4096, 128], false), shape);
        // KV heads that do not divide the query heads.
        assert_eq!(pick(DType::BF16, &H3, &[1, 5, 4096, 128], false), shape);
        // K and V disagreeing.
        assert_eq!(
            flash_eligibility(
                true,
                true,
                DType::BF16,
                &H3,
                &H3,
                &[1, 56, 4095, 128],
                false
            ),
            shape
        );
        // Empty sequences.
        assert_eq!(pick(DType::BF16, &[1, 56, 0, 128], &H3, false), shape);
        // Head dims: not a multiple of 8, too wide, or mismatched q vs k.
        let hd = |d| Budgeted(FlashRefusal::HeadDim(d));
        assert_eq!(
            pick(DType::BF16, &[1, 56, 64, 100], &[1, 56, 64, 100], false),
            hd(100)
        );
        assert_eq!(
            pick(DType::BF16, &[1, 4, 64, 512], &[1, 4, 64, 512], false),
            hd(512)
        );
        assert_eq!(
            pick(DType::BF16, &[1, 56, 64, 128], &[1, 56, 64, 64], false),
            hd(128)
        );
    }

    /// On the CPU every call refuses (`NotCuda` or `NotCompiled`), and the dispatcher is then the
    /// planned budgeted path verbatim — the same values, and the same typed cancel.
    #[test]
    fn the_cpu_path_is_the_planned_budgeted_path() -> crate::Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        assert!(matches!(
            select_attention_kernel(&q, &k, &v, None),
            AttentionKernel::Budgeted(FlashRefusal::NotCompiled | FlashRefusal::NotCuda)
        ));
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let got = sdpa_dispatch_bhsd(&q, &k, &v, 0.35, None, softmax_last_dim, plan)?;
        let want = sdpa_planned_bhsd(&q, &k, &v, 0.35, None, softmax_last_dim, plan)?;
        let d = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(d, 0.0);

        // A tripped flag on a *bounded* plan that actually chunks surfaces as `Canceled`.
        let cancel = crate::gen_core::CancelFlag::default();
        cancel.cancel();
        let tiny = AttentionPlan::budgeted(AttentionBudget::from_score_elements(64, false))
            .with_cancel(&cancel);
        let err = sdpa_dispatch_bhsd(&q, &k, &v, 0.35, None, softmax_last_dim, tiny).unwrap_err();
        assert!(matches!(err, crate::CandleError::Canceled), "{err:?}");
        Ok(())
    }

    /// **Inference compute — `#[ignore]`d, never part of normal CI.** Flash vs the materialized
    /// path on small random bf16 tensors on CUDA ordinal 0 of the visible set (run with
    /// `CUDA_VISIBLE_DEVICES` pinned to the validation GPU):
    ///
    /// `cargo test -p candle-gen --features flash-attn --lib fused_attention::tests::flash_matches -- --ignored --nocapture`
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn flash_matches_budgeted_on_cuda() -> crate::Result<()> {
        let dev = Device::new_cuda(0)?;
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let f32_softmax = |s: &Tensor| -> candle_core::Result<Tensor> {
            softmax_last_dim(&s.to_dtype(DType::F32)?)?.to_dtype(s.dtype())
        };
        let mut worst = 0f32;
        for (b, h, sq, sk, d) in [
            (1, 4, 256, 256, 128),
            (2, 8, 333, 333, 128),
            (1, 56, 1024, 1024, 128),
            (1, 16, 300, 300, 72),
            (1, 4, 128, 517, 64),
        ] {
            let q = Tensor::randn(0f32, 1.0, (b, h, sq, d), &dev)?.to_dtype(DType::BF16)?;
            let k = Tensor::randn(0f32, 1.0, (b, h, sk, d), &dev)?.to_dtype(DType::BF16)?;
            let v = Tensor::randn(0f32, 1.0, (b, h, sk, d), &dev)?.to_dtype(DType::BF16)?;
            assert_eq!(
                select_attention_kernel(&q, &k, &v, None),
                AttentionKernel::Flash
            );
            let scale = 1.0 / (d as f64).sqrt();
            let fused = sdpa_dispatch_bhsd(&q, &k, &v, scale, None, f32_softmax, plan)?;
            // The reference: the materialized path entirely in f32.
            let want = sdpa_planned_bhsd(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                scale,
                None,
                softmax_last_dim,
                plan,
            )?;
            let old = sdpa_planned_bhsd(&q, &k, &v, scale, None, f32_softmax, plan)?;
            let diff = |a: &Tensor| -> crate::Result<f32> {
                Ok((a.to_dtype(DType::F32)? - &want)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?)
            };
            let (df, db) = (diff(&fused)?, diff(&old)?);
            println!(
                "flash parity b={b} h={h} sq={sq} sk={sk} d={d}: max|flash - f32 ref| = {df:.3e}, \
                 max|bf16 budgeted - f32 ref| = {db:.3e}"
            );
            worst = worst.max(df);
        }
        assert!(worst < 2e-2, "flash vs f32 reference max|Δ| = {worst}");
        Ok(())
    }

    /// **Inference compute — `#[ignore]`d.** Wall time of one MiniMax-H3-shaped attention
    /// (B=1, 56 heads, d=128, bf16) through the fused kernel and the 1e9-budgeted path.
    /// `H3_ATTN_BENCH_SEQ` overrides the sequence length (default 40,000).
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute benchmark: run by hand, not CI"]
    fn h3_shaped_attention_flash_vs_budgeted_timing() -> crate::Result<()> {
        let dev = Device::new_cuda(0)?;
        let s: usize = std::env::var("H3_ATTN_BENCH_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40_000);
        let (h, d) = (56, 128);
        let q = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let scale = 1.0 / (d as f64).sqrt();
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let time = |f: &dyn Fn() -> crate::Result<Tensor>| -> crate::Result<(f64, Tensor)> {
            let _warm = f()?;
            dev.synchronize()?;
            let t = std::time::Instant::now();
            let out = f()?;
            dev.synchronize()?;
            Ok((t.elapsed().as_secs_f64(), out))
        };
        let (tf, of) =
            time(&|| sdpa_dispatch_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan))?;
        let (tb, ob) =
            time(&|| sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan))?;
        let diff = (of.to_dtype(DType::F32)? - ob.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        println!(
            "h3-shaped attention S={s} H={h} d={d} bf16: flash {tf:.3}s, budgeted {tb:.3}s \
             ({:.1}x), max|Δ| = {diff:.3e}",
            tb / tf
        );
        Ok(())
    }
}
