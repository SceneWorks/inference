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
//! - the tensors live on a CUDA device **whose compute capability the compiled fatbin serves**:
//!   at least sm_8.0 (FlashAttention-2's floor) and covered by the vendored
//!   `vendor/candle-flash-attn` `-gencode` ladder (SASS within one major, PTX forward), read back
//!   through `build.rs` ([`crate::cuda_arch::FatbinLadder::flash_from_build`]). A cubin-only
//!   kernel with no image for the device would not error — it would leave the output unwritten —
//!   so an uncovered device always takes the budgeted path. The capability is read once per GPU
//!   ordinal and cached;
//! - the dtype is bf16 or f16 (the kernel has no f32 path — f32 callers keep their numerics), and
//!   q, k and v share it and share one device (the kernel reads all three as one element type);
//! - there is **no additive mask** — the wrapper exposes only dense / causal, and an arbitrary
//!   additive bias cannot be expressed, so a masked call keeps the budgeted path;
//! - q/k/v are rank 4, share the batch, K and V share their shape, the query heads are a multiple of
//!   the KV heads, and the common `head_dim` is a multiple of 8 and at most
//!   [`FLASH_MAX_HEAD_DIM`];
//! - every stride and extent the kernel receives fits its `u32` parameter ABI: the batch / row /
//!   head strides of the `[B, S, H, D]` views `flash_bhsd` actually hands it (the operand's own
//!   strides when its last dim is unit-stride, else those of the contiguous copy), and the
//!   contiguous output's `Sq·H·D` batch stride. candle-flash-attn's dense path converts these
//!   checked too, but refusing here keeps an oversized call on the budgeted path instead of
//!   failing it.
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

/// First-order training SDPA. The experimental streaming candidate is explicitly
/// scoped by [`crate::training_attention::with_training_attention_mode`]. Callers
/// must supply ordinary last-axis softmax (possibly upcast to f32), with no
/// dropout, softcap or other score transform. Masks always retain the reference.
/// Cancellation is checked before launch; a native launch cannot be interrupted.
pub fn training_sdpa_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    plan: AttentionPlan<'_>,
) -> crate::Result<Tensor> {
    if plan.is_cancelled() {
        return Err(crate::CandleError::Canceled);
    }
    if let Some(out) = crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())? {
        return Ok(out);
    }
    sdpa_planned_bhsd(q, k, v, scale, mask, softmax, plan)
}

/// Legacy-budget overload of [`training_sdpa_bhsd`], retaining the exact
/// composable fallback. GQA inputs must be expanded by the caller when the
/// reference requires it; the candidate itself can sum grouped KV gradients.
pub fn training_sdpa_budgeted_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    budget: usize,
) -> candle_core::Result<Tensor> {
    if let Some(out) = crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())? {
        return Ok(out);
    }
    crate::attention::sdpa_budgeted_bhsd(q, k, v, scale, mask, softmax, budget)
}

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
    /// The CUDA device's compute capability (`None`: unreadable) is below sm_8.0 or not served by
    /// the compiled flash fatbin.
    UnsupportedArch(Option<u32>),
    /// The dtype has no fused kernel (only bf16 / f16 do).
    Dtype(DType),
    /// The call carries an additive mask, which the fused kernel cannot express.
    Masked,
    /// The `head_dim` is not a multiple of 8, exceeds [`FLASH_MAX_HEAD_DIM`], or differs across
    /// q / k / v.
    HeadDim(usize),
    /// Rank, batch, K/V shape or head-group mismatch.
    Shape,
    /// q, k and v do not share one dtype.
    MixedDtype,
    /// q, k and v do not live on one device.
    MixedDevice,
    /// A stride or extent the kernel would receive exceeds `u32::MAX` (its parameter ABI).
    IndexOverflow,
}

/// One attention operand as [`flash_eligibility`] sees it: dtype, `[B, H, S, D]` dims and element
/// strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashOperand<'a> {
    /// The element type.
    pub dtype: DType,
    /// The `[B, H, S, D]` dims.
    pub dims: &'a [usize],
    /// The element strides, one per dim.
    pub strides: &'a [usize],
}

impl<'a> FlashOperand<'a> {
    /// The operand view of a real tensor.
    pub fn of(t: &'a Tensor) -> Self {
        Self {
            dtype: t.dtype(),
            dims: t.dims(),
            strides: t.stride(),
        }
    }
}

/// The `(batch, row, head)` element strides `flash_bhsd` hands the kernel for one `[B, H, S, D]`
/// operand: the `[B, S, H, D]` transpose's own strides when the last dim is unit-stride, else
/// those of the contiguous copy it makes. `None` when the copy's strides overflow `usize`.
fn kernel_strides(dims: [usize; 4], strides: &[usize]) -> Option<[usize; 3]> {
    let [_, h, s, d] = dims;
    match *strides {
        [sb, sh, ss, 1] => Some([sb, ss, sh]),
        _ => {
            let hd = h.checked_mul(d)?;
            Some([s.checked_mul(hd)?, hd, d])
        }
    }
}

/// Whether every value the kernel's `u32` parameter ABI would carry for this call fits.
fn fits_kernel_abi(
    q: [usize; 4],
    k: [usize; 4],
    q_strides: &[usize],
    k_strides: &[usize],
    v_strides: &[usize],
) -> bool {
    let fits = |x: usize| u32::try_from(x).is_ok();
    let strides_fit =
        |dims, strides| kernel_strides(dims, strides).is_some_and(|s| s.into_iter().all(fits));
    let [b, h, sq, d] = q;
    let [_, kh, sk, _] = k;
    // The output is a fresh contiguous `[B, Sq, H, D]`: batch stride `Sq·H·D`, row `H·D`.
    let out_batch = h.checked_mul(d).and_then(|hd| hd.checked_mul(sq));
    // The kernel also takes the 128-rounded sequence lengths.
    let rounded = |x: usize| x.checked_next_multiple_of(128);
    strides_fit(q, q_strides)
        && strides_fit(k, k_strides)
        && strides_fit(k, v_strides)
        && out_batch.is_some_and(fits)
        && rounded(sq).is_some_and(fits)
        && rounded(sk).is_some_and(fits)
        && [b, h, kh, d].into_iter().all(fits)
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

/// The device side of [`flash_eligibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashDevice {
    /// CPU / Metal.
    NotCuda,
    /// A CUDA device with its packed compute capability (`120` = sm_12.0), `None` if unreadable,
    /// and whether [`flash_arch_supported`] accepted it.
    Cuda {
        compute_cap: Option<u32>,
        supported: bool,
    },
}

/// FlashAttention-2's architecture floor: sm_8.0 (Ampere).
pub const FLASH_MIN_COMPUTE_CAP: u32 = 80;

/// Whether the fused kernel may launch on a device of `compute_cap` given the compiled `ladder`:
/// the capability is known, at least [`FLASH_MIN_COMPUTE_CAP`], and covered by a non-empty ladder.
/// Pure — the unit tests drive it with synthetic ladders.
pub fn flash_arch_supported(
    compute_cap: Option<u32>,
    ladder: &crate::cuda_arch::FatbinLadder,
) -> bool {
    match compute_cap {
        Some(cap) => cap >= FLASH_MIN_COMPUTE_CAP && !ladder.is_empty() && ladder.covers(cap),
        None => false,
    }
}

/// The pure selection rule (see the module docs). `q`/`k`/`v` carry the `[B, H, S, D]` dims and
/// strides; `device` classifies q's device and `same_device` says whether k and v share it.
pub fn flash_eligibility(
    compiled: bool,
    device: FlashDevice,
    same_device: bool,
    q: FlashOperand<'_>,
    k: FlashOperand<'_>,
    v: FlashOperand<'_>,
    masked: bool,
) -> AttentionKernel {
    use AttentionKernel::Budgeted;
    if !compiled {
        return Budgeted(FlashRefusal::NotCompiled);
    }
    match device {
        FlashDevice::NotCuda => return Budgeted(FlashRefusal::NotCuda),
        FlashDevice::Cuda {
            compute_cap,
            supported: false,
        } => return Budgeted(FlashRefusal::UnsupportedArch(compute_cap)),
        FlashDevice::Cuda { .. } => {}
    }
    if !matches!(q.dtype, DType::BF16 | DType::F16) {
        return Budgeted(FlashRefusal::Dtype(q.dtype));
    }
    if k.dtype != q.dtype || v.dtype != q.dtype {
        return Budgeted(FlashRefusal::MixedDtype);
    }
    if !same_device {
        return Budgeted(FlashRefusal::MixedDevice);
    }
    if masked {
        return Budgeted(FlashRefusal::Masked);
    }
    let (qdims @ [qb, qh, qs, qd], kdims @ [kb, kh, ks, kd]) = match (q.dims, k.dims) {
        (&[a, b, c, d], &[e, f, g, h]) => ([a, b, c, d], [e, f, g, h]),
        _ => return Budgeted(FlashRefusal::Shape),
    };
    if v.dims.len() != 4
        || qb != kb
        || k.dims != v.dims
        || qs == 0
        || ks == 0
        || kh == 0
        || qh % kh != 0
        || [q.strides.len(), k.strides.len(), v.strides.len()] != [4; 3]
    {
        return Budgeted(FlashRefusal::Shape);
    }
    if qd != kd || qd % 8 != 0 || qd == 0 || qd > FLASH_MAX_HEAD_DIM {
        return Budgeted(FlashRefusal::HeadDim(qd));
    }
    if !fits_kernel_abi(qdims, kdims, q.strides, k.strides, v.strides) {
        return Budgeted(FlashRefusal::IndexOverflow);
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
        flash_device(q.device()),
        q.device().same_device(k.device()) && q.device().same_device(v.device()),
        FlashOperand::of(q),
        FlashOperand::of(k),
        FlashOperand::of(v),
        mask.is_some(),
    )
}

/// Classify a real device for [`flash_eligibility`]. The compute capability is cached per GPU
/// ordinal ([`crate::cuda_arch::device_compute_cap`]) and the baked ladder parsed once, so this is
/// a couple of lookups per attention call. Skipped entirely when the kernel is not compiled.
pub fn flash_device(device: &candle_core::Device) -> FlashDevice {
    if !device.is_cuda() {
        return FlashDevice::NotCuda;
    }
    if !FLASH_ATTN_COMPILED {
        return FlashDevice::Cuda {
            compute_cap: None,
            supported: false,
        };
    }
    static LADDER: std::sync::OnceLock<crate::cuda_arch::FatbinLadder> = std::sync::OnceLock::new();
    let ladder = LADDER.get_or_init(crate::cuda_arch::FatbinLadder::flash_from_build);
    let compute_cap = crate::cuda_arch::device_compute_cap(device);
    FlashDevice::Cuda {
        compute_cap,
        supported: flash_arch_supported(compute_cap, ladder),
    }
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

    const SM120: FlashDevice = FlashDevice::Cuda {
        compute_cap: Some(120),
        supported: true,
    };

    /// Row-major strides of `dims`.
    fn contiguous(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// The eligibility of contiguous operands of one dtype on one device.
    fn elig(
        compiled: bool,
        device: FlashDevice,
        dtype: DType,
        q: &[usize],
        k: &[usize],
        v: &[usize],
        masked: bool,
    ) -> AttentionKernel {
        let (qs, ks, vs) = (contiguous(q), contiguous(k), contiguous(v));
        let op = |dims, strides| FlashOperand {
            dtype,
            dims,
            strides,
        };
        flash_eligibility(
            compiled,
            device,
            true,
            op(q, &qs),
            op(k, &ks),
            op(v, &vs),
            masked,
        )
    }

    fn pick(dtype: DType, q: &[usize], k: &[usize], masked: bool) -> AttentionKernel {
        elig(true, SM120, dtype, q, k, k, masked)
    }

    fn shipped_ladder() -> crate::cuda_arch::FatbinLadder {
        // The vendored fatbin at the SceneWorks cap=80 baseline: sm_80/90/120 SASS + compute_120.
        crate::cuda_arch::FatbinLadder::explicit_plus_baseline("90,120", "120", "80")
    }

    #[test]
    fn the_arch_guard_admits_only_devices_the_fatbin_serves() {
        let ladder = shipped_ladder();
        for cap in [80, 86, 89, 90, 120, 121, 130] {
            assert!(flash_arch_supported(Some(cap), &ladder), "sm_{cap}");
        }
        // Below FA2's floor, the datacenter-Blackwell gap (major 10), and an unreadable device.
        for cap in [Some(61), Some(75), Some(100), Some(103), None] {
            assert!(!flash_arch_supported(cap, &ladder), "{cap:?}");
        }
        // A cap=120-only build (no sm_80 rung) must not launch on Ampere.
        let native = crate::cuda_arch::FatbinLadder::explicit_plus_baseline("120", "", "120");
        assert!(!flash_arch_supported(Some(86), &native));
        assert!(flash_arch_supported(Some(120), &native));
        // An empty ladder is "unknown", which for a launch means "no".
        assert!(!flash_arch_supported(
            Some(120),
            &crate::cuda_arch::FatbinLadder::default()
        ));
    }

    #[test]
    fn an_unsupported_arch_refuses_before_any_shape_check() {
        let old = FlashDevice::Cuda {
            compute_cap: Some(75),
            supported: false,
        };
        assert_eq!(
            elig(true, old, DType::BF16, &H3, &H3, &H3, false),
            AttentionKernel::Budgeted(FlashRefusal::UnsupportedArch(Some(75)))
        );
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
        let f = |compiled, cuda: bool, dtype| {
            let device = if cuda { SM120 } else { FlashDevice::NotCuda };
            elig(compiled, device, dtype, &H3, &H3, &H3, false)
        };
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
            elig(
                true,
                SM120,
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

    #[test]
    fn q_k_v_must_share_one_dtype_and_one_device() {
        use AttentionKernel::Budgeted;
        let st = contiguous(&H3);
        let op = |dtype| FlashOperand {
            dtype,
            dims: &H3,
            strides: &st,
        };
        let (bf, f16) = (op(DType::BF16), op(DType::F16));
        assert_eq!(
            flash_eligibility(true, SM120, true, bf, bf, bf, false),
            AttentionKernel::Flash
        );
        for (k, v) in [(f16, bf), (bf, f16), (op(DType::F32), bf)] {
            assert_eq!(
                flash_eligibility(true, SM120, true, bf, k, v, false),
                Budgeted(FlashRefusal::MixedDtype)
            );
        }
        assert_eq!(
            flash_eligibility(true, SM120, false, bf, bf, bf, false),
            Budgeted(FlashRefusal::MixedDevice)
        );
    }

    #[test]
    fn strides_and_extents_beyond_the_kernels_u32_abi_refuse() {
        use AttentionKernel::Budgeted;
        let overflow = Budgeted(FlashRefusal::IndexOverflow);
        let max = u32::MAX as usize;
        let call = |q: &[usize], qs: &[usize], k: &[usize], ks: &[usize], vs: &[usize]| {
            let op = |dims, strides| FlashOperand {
                dtype: DType::BF16,
                dims,
                strides,
            };
            flash_eligibility(true, SM120, true, op(q, qs), op(k, ks), op(k, vs), false)
        };
        // The longest contiguous `[1, H, S, D]` whose `S·H·D` (the batch stride of q and of the
        // output) fits u32 takes the kernel; one more row does not.
        let (h, d) = (8usize, 128usize);
        let rows = max / (h * d);
        let fit = [1, h, rows, d];
        let st = contiguous(&fit);
        assert_eq!(call(&fit, &st, &fit, &st, &st), AttentionKernel::Flash);
        let over = [1, h, rows + 1, d];
        let so = contiguous(&over);
        assert_eq!(
            call(&over, &so, &fit, &st, &st),
            overflow,
            "q batch / output"
        );
        assert_eq!(
            call(&fit, &st, &over, &so, &so),
            overflow,
            "k/v batch stride"
        );
        // A small strided view: each oversized stride alone refuses — batch, head and sequence
        // (the `[B, S, H, D]` kernel view's row stride).
        let small = [2, 4, 16, 64];
        let ok = contiguous(&small);
        assert_eq!(call(&small, &ok, &small, &ok, &ok), AttentionKernel::Flash);
        for big in [
            [max + 1, 1024, 64, 1],
            [4096, max + 1, 64, 1],
            [4096, 1024, max + 1, 1],
        ] {
            assert_eq!(call(&small, &big, &small, &ok, &ok), overflow, "q {big:?}");
            assert_eq!(call(&small, &ok, &small, &big, &ok), overflow, "k {big:?}");
            assert_eq!(call(&small, &ok, &small, &ok, &big), overflow, "v {big:?}");
        }
        // A non-unit last stride is judged by the contiguous copy `flash_bhsd` makes, so huge
        // strides that the copy discards are fine...
        let gather = [1 << 40, 1 << 36, 1 << 33, 2];
        assert_eq!(
            call(&small, &gather, &small, &ok, &ok),
            AttentionKernel::Flash
        );
        // ...while the copy's own `S·H·D` still has to fit.
        assert_eq!(call(&over, &gather, &fit, &st, &st), overflow);
        // Strides whose rank disagrees with the dims are a shape error, not a launch.
        assert_eq!(
            call(&small, &ok[..3], &small, &ok, &ok),
            Budgeted(FlashRefusal::Shape)
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
