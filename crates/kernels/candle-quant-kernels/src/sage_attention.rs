//! SageAttention-style INT8 QK attention (SW-6131, feature SW-F287 S6): dense, unmasked
//! scaled-dot-product attention whose `Q·Kᵀ` runs on the INT8 tensor cores.
//!
//! A port of the comfy-kitchen `sage_attention` design (Apache-2.0) and the thu-ml SageAttention
//! algorithm (Apache-2.0), flattened into one nvrtc source (`sage_attention.cu`); attribution is in
//! that file's header, `sage_attention-NOTICE.md` and `docs/licensing/third-party-kernel-sources.md`.
//!
//! # Arithmetic
//!
//! 1. Q and K rows are rotated by the same fixed **signed Walsh–Hadamard** transform (comfy-kitchen's
//!    sign diagonal, then a normalized `H_D`). The rotation is orthogonal, so every `q·k` is
//!    unchanged, while an outlier channel is spread over the whole row before quantization.
//! 2. K may be **centered** first: subtracting the per-channel mean of K over the sequence shifts
//!    every score of one query row by the same constant, which the softmax cancels exactly. It is
//!    engaged per `(batch, kv head)` by [`KCentering`] (`Adaptive` measures, on up to 256 sampled
//!    rows, whether centering shrinks the rotated per-row range).
//! 3. Each rotated row is quantized to INT8 with one scale per token (`amax / 127`, round to nearest
//!    even, clamp to ±127). The Q scale also carries `softmax_scale · log2(e)`.
//! 4. The attention kernel computes `Q·Kᵀ` exactly in `int32` (`mma.m16n8k32.s8`), dequantizes with
//!    the two row scales, runs an online softmax in **f32** with `exp2`, rounds `P` to the V dtype
//!    and accumulates `P·V` in f32 on the 16-bit tensor cores (`mma.m16n8k16`). **V stays
//!    bf16/f16**; the only lossy step is the INT8 `Q·Kᵀ`.
//!
//! # Envelope
//!
//! Checked by the pure [`check_sage_attention`] before anything launches (E5): CUDA sm_80+ (the
//! source's `cc_floor`: `mma.m16n8k32.s8`, the bf16 `mma.m16n8k16` and `cp.async` all need it),
//! bf16 or f16 operands of one dtype, `[B, H, S, D]` rank-4 operands with grouped-query KV heads
//! dividing the query heads, head dim **64 or 128**, and every padded INT8 buffer at most
//! `i32::MAX` elements (E6). There is no mask, no causal mode and no dropout.
//!
//! # Prequantize / early release
//!
//! `prequantize_qk` (cuda builds) returns `SageQk` (INT8 codes and scales only — it keeps no reference to the
//! floating-point Q/K), so a caller may drop Q and K before `attend_prequantized` runs.
//! `sage_attention` is the one-call form.

use crate::nvrtc::KernelSource;
use candle_core::DType;

/// The INT8 QK attention kernels (quantizer, centering statistics and the attention kernel).
pub const SOURCE: KernelSource = KernelSource {
    name: "sage-int8-qk-attention-v1",
    src: include_str!("sage_attention.cu"),
    cc_floor: (8, 0),
};

/// The compute-capability floor (packed, `80` = sm_8.0) a launch needs: [`SOURCE`]'s `cc_floor`.
pub const SAGE_MIN_COMPUTE_CAP: u32 = 80;

/// The head dims the kernels are compiled for.
pub const SAGE_HEAD_DIMS: [usize; 2] = [64, 128];

/// Query rows per attention CTA (the Q buffers are padded to it).
pub const SAGE_Q_TILE: usize = 128;

/// Keys per streamed K/V block (the K buffers are padded to it).
pub const SAGE_K_TILE: usize = 64;

/// comfy-kitchen's `apply_convrot_sign128` words (`sage_sign_word` in the kernel): channel `c` is
/// negated when bit `c & 31` of word `c >> 5` is **clear**.
pub const SAGE_SIGN_WORDS: [u32; 4] = [0x1035_997b, 0x8087_f5ee, 0xee2e_4e1a, 0x7113_2418];

/// When the K quantizer subtracts the per-channel mean of K (softmax-invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KCentering {
    /// Never.
    Off,
    /// Always, for every `(batch, kv head)`.
    Always,
    /// Per `(batch, kv head)`, when it shrinks the sampled rotated per-row range (the default).
    #[default]
    Adaptive,
}

/// Why a call cannot take the INT8 attention kernel. Every variant is decided before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SageRefusal {
    /// Not bf16 / f16.
    Dtype(DType),
    /// q, k and v do not share one dtype.
    MixedDtype,
    /// The head dim is not one the kernels are compiled for ([`SAGE_HEAD_DIMS`]), or q/k/v differ.
    HeadDim(usize),
    /// Rank, batch, K/V shape, empty sequence or head-group mismatch.
    Shape,
    /// A padded INT8 buffer would exceed `i32::MAX` elements, or a grid extent its CUDA limit.
    IndexOverflow,
}

/// The launch geometry [`check_sage_attention`] accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SagePlan {
    /// Batch.
    pub batch: usize,
    /// Query heads.
    pub heads: usize,
    /// KV heads (divides `heads`).
    pub kv_heads: usize,
    /// Query rows.
    pub q_len: usize,
    /// Key rows.
    pub k_len: usize,
    /// Head dim (64 or 128).
    pub head_dim: usize,
    /// `q_len` rounded up to [`SAGE_Q_TILE`].
    pub q_pad: usize,
    /// `k_len` rounded up to [`SAGE_K_TILE`].
    pub k_pad: usize,
}

impl SagePlan {
    /// Bytes of device workspace one call allocates besides its output: the INT8 Q/K codes, their
    /// scales and the centering statistics (allocated even when centering is off, except the
    /// partial sums).
    pub fn workspace_bytes(&self) -> u64 {
        let (b, h, hk, d) = (
            self.batch as u64,
            self.heads as u64,
            self.kv_heads as u64,
            self.head_dim as u64,
        );
        let (qp, kp) = (self.q_pad as u64, self.k_pad as u64);
        // The K-centering mean + flag, and its per-256-row partial sums.
        let chunks = (self.k_len as u64).div_ceil(256);
        b * h * qp * (d + 4)
            + b * hk * kp * (d + 4)
            + b * hk * (4 * d + 4)
            + b * hk * chunks * 4 * d
    }
}

/// The pure envelope check: `q` is `[B, H, Sq, D]`, `k`/`v` are `[B, HK, Sk, D]`, one dtype.
pub fn check_sage_attention(
    q_dims: &[usize],
    k_dims: &[usize],
    v_dims: &[usize],
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
) -> Result<SagePlan, SageRefusal> {
    if !matches!(q_dtype, DType::BF16 | DType::F16) {
        return Err(SageRefusal::Dtype(q_dtype));
    }
    if k_dtype != q_dtype || v_dtype != q_dtype {
        return Err(SageRefusal::MixedDtype);
    }
    let (&[b, h, sq, d], &[kb, hk, sk, kd]) = (q_dims, k_dims) else {
        return Err(SageRefusal::Shape);
    };
    if v_dims != k_dims || kb != b || b == 0 || h == 0 || hk == 0 || sq == 0 || sk == 0 {
        return Err(SageRefusal::Shape);
    }
    if !h.is_multiple_of(hk) {
        return Err(SageRefusal::Shape);
    }
    if kd != d || !SAGE_HEAD_DIMS.contains(&d) {
        return Err(SageRefusal::HeadDim(d));
    }
    let q_pad = sq.checked_next_multiple_of(SAGE_Q_TILE);
    let k_pad = sk.checked_next_multiple_of(SAGE_K_TILE);
    let (Some(q_pad), Some(k_pad)) = (q_pad, k_pad) else {
        return Err(SageRefusal::IndexOverflow);
    };
    let fits = |parts: [usize; 4]| {
        parts
            .into_iter()
            .try_fold(1usize, |acc, x| acc.checked_mul(x))
            .is_some_and(|n| n <= i32::MAX as usize)
    };
    // The INT8 buffers, the output, and every strided operand's own extent (its last element's
    // offset is checked by the caller's layout, which candle already bounds).
    let grid_ok = b <= 65_535 && h <= 65_535 && q_pad / SAGE_Q_TILE <= i32::MAX as usize;
    if !fits([b, h, q_pad, d]) || !fits([b, hk, k_pad, d]) || !grid_ok {
        return Err(SageRefusal::IndexOverflow);
    }
    Ok(SagePlan {
        batch: b,
        heads: h,
        kv_heads: hk,
        q_len: sq,
        k_len: sk,
        head_dim: d,
        q_pad,
        k_pad,
    })
}

// ---------------------------------------------------------------------------------------------
// Host oracle of the quantizer (the exact f32 operation order of `sage_rotate` / `sage_quant`).
// ---------------------------------------------------------------------------------------------

/// The kernel's rotation of one `d`-channel row (`d` = 64 or 128), in its exact f32 order: the
/// sign diagonal, the in-lane `H_{d/32}`, the five lane butterflies, then the normalization.
pub fn rotate_reference(row: &mut [f32]) {
    let d = row.len();
    assert!(SAGE_HEAD_DIMS.contains(&d), "rotate_reference: d = {d}");
    let ept = d / 32;
    for (ch, x) in row.iter_mut().enumerate() {
        let flip = ((SAGE_SIGN_WORDS[ch >> 5] >> (ch & 31)) & 1) ^ 1;
        *x = f32::from_bits(x.to_bits() ^ (flip << 31));
    }
    for lane in 0..32 {
        let v = &mut row[lane * ept..lane * ept + ept];
        if ept == 4 {
            let (a0, a1, a2, a3) = (v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3]);
            v[0] = a0 + a2;
            v[1] = a1 + a3;
            v[2] = a0 - a2;
            v[3] = a1 - a3;
        } else {
            let (a0, a1) = (v[0] + v[1], v[0] - v[1]);
            v[0] = a0;
            v[1] = a1;
        }
    }
    let mut bit = 1;
    while bit < 32 {
        let before = row.to_vec();
        for lane in 0..32 {
            for c in 0..ept {
                let mine = before[lane * ept + c];
                let other = before[(lane ^ bit) * ept + c];
                row[lane * ept + c] = if lane & bit != 0 {
                    other - mine
                } else {
                    mine + other
                };
            }
        }
        bit <<= 1;
    }
    // 1/sqrt(128) correctly rounded to f32, the same bits the kernel uses.
    let norm = if d == 128 {
        f32::from_bits(0x3db5_04f3)
    } else {
        0.125
    };
    for x in row.iter_mut() {
        *x *= norm;
    }
}

/// The kernel's per-row quantization of an already-rotated row: `(codes, scale)` with
/// `scale = amax / 127 · pre`.
pub fn quantize_row_reference(rotated: &[f32], pre: f32) -> (Vec<i8>, f32) {
    let amax = rotated.iter().fold(0f32, |m, x| m.max(x.abs()));
    let step = amax / 127.0;
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let codes = rotated
        .iter()
        .map(|x| (x * inv).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect();
    (codes, step * pre)
}

#[cfg(feature = "cuda")]
pub use cuda::{attend_prequantized, prequantize_qk, sage_attention, SageQk};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::CudaDType;
    use candle_core::{op::BackpropOp, CudaStorage, Device, Result, Shape, Storage, Tensor};

    /// Prequantized Q/K for `attend_prequantized`: INT8 codes and per-row scales only — no
    /// reference to the floating-point Q or K survives in it.
    #[derive(Debug, Clone)]
    pub struct SageQk {
        /// `[B, H, q_pad, D]` INT8 codes, stored as U8.
        pub q_codes: Tensor,
        /// `[B, H, q_pad]` f32 scales (softmax scale · log2(e) folded in).
        pub q_scales: Tensor,
        /// `[B, HK, k_pad, D]` INT8 codes, stored as U8.
        pub k_codes: Tensor,
        /// `[B, HK, k_pad]` f32 scales.
        pub k_scales: Tensor,
        /// The geometry.
        pub plan: SagePlan,
        /// The dtype of the floating-point operands (and of the output).
        pub dtype: DType,
    }

    fn msg(text: impl Into<String>) -> candle_core::Error {
        candle_core::Error::Msg(text.into())
    }

    fn refusal(r: SageRefusal) -> candle_core::Error {
        msg(format!("sage int8 attention refused: {r:?}"))
    }

    fn drv<E: std::fmt::Debug>(e: E) -> candle_core::Error {
        msg(format!("sage int8 attention: {e:?}"))
    }

    fn cuda_of(t: &Tensor) -> Result<&candle_core::CudaDevice> {
        match t.device() {
            Device::Cuda(d) => Ok(d),
            _ => Err(msg("sage int8 attention: expected a CUDA tensor")),
        }
    }

    /// `t` itself when its last dim is unit-stride (and, for `aligned`, every row starts on a
    /// 16-byte boundary, which `cp.async` needs), else its contiguous copy.
    fn operand(t: &Tensor, aligned: bool) -> Result<Tensor> {
        let st = t.stride();
        let off = t.layout().start_offset();
        let ok = st.len() == 4
            && st[3] == 1
            && (!aligned || (st[..3].iter().all(|s| s.is_multiple_of(8)) && off.is_multiple_of(8)));
        if ok {
            Ok(t.clone())
        } else {
            t.contiguous()
        }
    }

    fn strides3(t: &Tensor) -> [i64; 3] {
        let s = t.stride();
        [s[0] as i64, s[1] as i64, s[2] as i64]
    }

    fn view<'a, T: CudaDType>(st: &'a Storage, t: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cs) = st else {
            return Err(msg("sage int8 attention: expected CUDA storage"));
        };
        Ok(cs.as_cuda_slice::<T>()?.slice(t.layout().start_offset()..))
    }

    fn wrap<T: CudaDType>(
        slice: CudaSlice<T>,
        dev: &candle_core::CudaDevice,
        shape: impl Into<Shape>,
    ) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            shape.into(),
            BackpropOp::none(),
            false,
        )
    }

    fn i32_of(x: usize) -> Result<i32> {
        i32::try_from(x).map_err(|_| refusal(SageRefusal::IndexOverflow))
    }

    fn launch_cfg(grid: (usize, usize, usize), block: u32) -> Result<LaunchConfig> {
        let dim = |x: usize| u32::try_from(x).map_err(|_| refusal(SageRefusal::IndexOverflow));
        Ok(LaunchConfig {
            grid_dim: (dim(grid.0)?, dim(grid.1)?, dim(grid.2)?),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    /// Rows-per-CTA of the centering-sum pass.
    const SUM_ROWS_PER_CTA: usize = 256;

    /// Rotate, optionally center, and INT8-quantize one `[batch, heads, len, d]` operand into
    /// `[batch, heads, pad, d]` codes and `[batch, heads, pad]` scales.
    #[allow(clippy::too_many_arguments)]
    fn quantize<T: CudaDType + DeviceRepr>(
        dev: &candle_core::CudaDevice,
        module: &crate::nvrtc::CompiledKernel,
        x: &Tensor,
        (batch, heads, len, pad, d): (usize, usize, usize, usize, usize),
        pre: f32,
        centering: KCentering,
    ) -> Result<(Tensor, Tensor)> {
        let x = operand(x, false)?;
        let bf = i32::from(x.dtype() == DType::BF16);
        let [sb, sh, ss] = strides3(&x);
        let (st, _) = x.storage_and_layout();
        let xv = view::<T>(&st, &x)?;
        let stream = dev.cuda_stream();
        let (h, s, sp) = (i32_of(heads)?, i32_of(len)?, i32_of(pad)?);
        let rows = batch * heads;
        // Centering statistics: the per-channel mean and the per-head flag (zero, i.e. unused, when
        // centering is off).
        let mut mean = stream.alloc_zeros::<f32>(rows * d).map_err(drv)?;
        let mut flag = stream.alloc_zeros::<i32>(rows).map_err(drv)?;
        let center = i32::from(centering != KCentering::Off);
        if center != 0 {
            let sum = module
                .function(&format!("sage_k_sum_d{d}"))
                .map_err(candle_core::Error::wrap)?;
            let per = i32_of(SUM_ROWS_PER_CTA)?;
            let chunks = len.div_ceil(SUM_ROWS_PER_CTA);
            // SAFETY: `sage_k_sum` writes every (row, chunk, channel) partial.
            let mut partial = unsafe { stream.alloc::<f32>(rows * chunks * d) }.map_err(drv)?;
            let cfg = launch_cfg((rows, chunks, 1), d as u32)?;
            let mut l = stream.launch_builder(&sum);
            l.arg(&xv)
                .arg(&sb)
                .arg(&sh)
                .arg(&ss)
                .arg(&h)
                .arg(&s)
                .arg(&per)
                .arg(&bf)
                .arg(&mut partial);
            // SAFETY: matches `sage_k_sum_d*`; `partial` holds rows·chunks·d f32.
            unsafe { l.launch(cfg) }.map_err(drv)?;
            let chunks = i32_of(chunks)?;
            let decide = module
                .function(&format!("sage_k_center_d{d}"))
                .map_err(candle_core::Error::wrap)?;
            let adaptive = i32::from(centering == KCentering::Adaptive);
            let mut l = stream.launch_builder(&decide);
            l.arg(&xv)
                .arg(&sb)
                .arg(&sh)
                .arg(&ss)
                .arg(&h)
                .arg(&s)
                .arg(&bf)
                .arg(&adaptive)
                .arg(&partial)
                .arg(&chunks)
                .arg(&mut mean)
                .arg(&mut flag);
            // SAFETY: matches `sage_k_center_d*`; `mean` holds rows·d f32 and `flag` rows i32.
            unsafe { l.launch(launch_cfg((rows, 1, 1), 256)?) }.map_err(drv)?;
        }
        // SAFETY: the quantizer writes every code and scale of the padded buffers (rows at or past
        // `len` as zero).
        let mut codes = unsafe { dev.alloc::<u8>(rows * pad * d) }?;
        let mut scales = unsafe { dev.alloc::<f32>(rows * pad) }?;
        let quant = module
            .function(&format!("sage_quant_d{d}"))
            .map_err(candle_core::Error::wrap)?;
        let mut l = stream.launch_builder(&quant);
        l.arg(&xv)
            .arg(&sb)
            .arg(&sh)
            .arg(&ss)
            .arg(&h)
            .arg(&s)
            .arg(&sp)
            .arg(&bf)
            .arg(&pre)
            .arg(&mean)
            .arg(&flag)
            .arg(&center)
            .arg(&mut codes)
            .arg(&mut scales);
        // SAFETY: matches `sage_quant_d*`.
        unsafe { l.launch(launch_cfg((pad / 8, heads, batch), 256)?) }.map_err(drv)?;
        drop(st);
        Ok((
            wrap(codes, dev, (batch, heads, pad, d)),
            wrap(scales, dev, (batch, heads, pad)),
        ))
    }

    /// Quantize `q` (`[B, H, Sq, D]`) and `k` (`[B, HK, Sk, D]`) for `attend_prequantized`.
    /// `v_dims` / `v_dtype` describe the V that will be attended (checked now, so a refusal comes
    /// before any work). `scale` is the softmax scale. The result holds no reference to `q` / `k`.
    pub fn prequantize_qk(
        q: &Tensor,
        k: &Tensor,
        v_dims: &[usize],
        v_dtype: DType,
        scale: f64,
        centering: KCentering,
    ) -> Result<SageQk> {
        let plan = check_sage_attention(q.dims(), k.dims(), v_dims, q.dtype(), k.dtype(), v_dtype)
            .map_err(refusal)?;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(msg(format!(
                "sage int8 attention: scale {scale} must be finite and positive"
            )));
        }
        let dev = cuda_of(q)?;
        if !q.device().same_device(k.device()) {
            return Err(msg("sage int8 attention: q and k on different devices"));
        }
        let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
        let pre = (scale as f32) * std::f32::consts::LOG2_E;
        let d = plan.head_dim;
        let qg = (plan.batch, plan.heads, plan.q_len, plan.q_pad, d);
        let kg = (plan.batch, plan.kv_heads, plan.k_len, plan.k_pad, d);
        let ((q_codes, q_scales), (k_codes, k_scales)) = match q.dtype() {
            DType::BF16 => (
                quantize::<half::bf16>(dev, &module, q, qg, pre, KCentering::Off)?,
                quantize::<half::bf16>(dev, &module, k, kg, 1.0, centering)?,
            ),
            _ => (
                quantize::<half::f16>(dev, &module, q, qg, pre, KCentering::Off)?,
                quantize::<half::f16>(dev, &module, k, kg, 1.0, centering)?,
            ),
        };
        Ok(SageQk {
            q_codes,
            q_scales,
            k_codes,
            k_scales,
            plan,
            dtype: q.dtype(),
        })
    }

    fn attend_typed<T: CudaDType + DeviceRepr>(
        qk: &SageQk,
        v: &Tensor,
        dev: &candle_core::CudaDevice,
        module: &crate::nvrtc::CompiledKernel,
        suffix: &str,
    ) -> Result<Tensor> {
        let p = qk.plan;
        let attn = module
            .function(&format!("sage_attn_d{}_{suffix}", p.head_dim))
            .map_err(candle_core::Error::wrap)?;
        let v = operand(v, true)?;
        let [vsb, vsh, vss] = strides3(&v);
        let (qst, _) = qk.q_codes.storage_and_layout();
        let (qsst, _) = qk.q_scales.storage_and_layout();
        let (kst, _) = qk.k_codes.storage_and_layout();
        let (ksst, _) = qk.k_scales.storage_and_layout();
        let (vst, _) = v.storage_and_layout();
        let qv = view::<u8>(&qst, &qk.q_codes)?;
        let qsv = view::<f32>(&qsst, &qk.q_scales)?;
        let kv = view::<u8>(&kst, &qk.k_codes)?;
        let ksv = view::<f32>(&ksst, &qk.k_scales)?;
        let vv = view::<T>(&vst, &v)?;
        // SAFETY: the attention kernel writes every element of every live query row.
        let out = unsafe { dev.alloc::<T>(p.batch * p.heads * p.q_len * p.head_dim) }?;
        let (h, hk) = (i32_of(p.heads)?, i32_of(p.kv_heads)?);
        let (sq, sqp, sk, skp) = (
            i32_of(p.q_len)?,
            i32_of(p.q_pad)?,
            i32_of(p.k_len)?,
            i32_of(p.k_pad)?,
        );
        let stream = dev.cuda_stream();
        let mut l = stream.launch_builder(&attn);
        l.arg(&qv)
            .arg(&qsv)
            .arg(&kv)
            .arg(&ksv)
            .arg(&vv)
            .arg(&vsb)
            .arg(&vsh)
            .arg(&vss)
            .arg(&out)
            .arg(&h)
            .arg(&hk)
            .arg(&sq)
            .arg(&sqp)
            .arg(&sk)
            .arg(&skp);
        let cfg = launch_cfg((p.q_pad / SAGE_Q_TILE, p.heads, p.batch), 256)?;
        // SAFETY: matches `sage_attn_d*_*`; every buffer is sized by the checked plan.
        unsafe { l.launch(cfg) }.map_err(drv)?;
        drop((qst, qsst, kst, ksst, vst));
        Ok(wrap(out, dev, (p.batch, p.heads, p.q_len, p.head_dim)))
    }

    /// The attention over prequantized Q/K and the 16-bit `v` (`[B, HK, Sk, D]`): `[B, H, Sq, D]`
    /// in `v`'s dtype, contiguous.
    pub fn attend_prequantized(qk: &SageQk, v: &Tensor) -> Result<Tensor> {
        let p = qk.plan;
        if v.dims() != [p.batch, p.kv_heads, p.k_len, p.head_dim] || v.dtype() != qk.dtype {
            return Err(refusal(SageRefusal::Shape));
        }
        let dev = cuda_of(v)?;
        if !v.device().same_device(qk.q_codes.device()) {
            return Err(msg("sage int8 attention: v on a different device"));
        }
        let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
        match qk.dtype {
            DType::BF16 => attend_typed::<half::bf16>(qk, v, dev, &module, "bf16"),
            _ => attend_typed::<half::f16>(qk, v, dev, &module, "f16"),
        }
    }

    /// One-call INT8 QK attention: `prequantize_qk` then `attend_prequantized`.
    pub fn sage_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        centering: KCentering,
    ) -> Result<Tensor> {
        let qk = prequantize_qk(q, k, v.dims(), v.dtype(), scale, centering)?;
        attend_prequantized(&qk, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_envelope_admits_krea_and_refuses_the_rest() {
        let krea_q = [1, 48, 4608, 128];
        let plan = check_sage_attention(
            &krea_q,
            &krea_q,
            &krea_q,
            DType::BF16,
            DType::BF16,
            DType::BF16,
        )
        .unwrap();
        assert_eq!((plan.q_pad, plan.k_pad), (4608, 4608));
        // GQA and cross attention.
        let kv = [1, 12, 300, 128];
        let plan =
            check_sage_attention(&krea_q, &kv, &kv, DType::F16, DType::F16, DType::F16).unwrap();
        assert_eq!((plan.kv_heads, plan.k_len, plan.k_pad), (12, 300, 320));
        let chk = |q: &[usize], k: &[usize], dt: DType| check_sage_attention(q, k, k, dt, dt, dt);
        assert_eq!(
            chk(&krea_q, &krea_q, DType::F32),
            Err(SageRefusal::Dtype(DType::F32))
        );
        assert_eq!(
            check_sage_attention(
                &krea_q,
                &krea_q,
                &krea_q,
                DType::BF16,
                DType::F16,
                DType::BF16
            ),
            Err(SageRefusal::MixedDtype)
        );
        for d in [8, 72, 96, 256] {
            let x = [1, 4, 64, d];
            assert_eq!(chk(&x, &x, DType::BF16), Err(SageRefusal::HeadDim(d)));
        }
        assert_eq!(
            chk(&[1, 4, 64, 128], &[1, 4, 64, 64], DType::BF16),
            Err(SageRefusal::HeadDim(128))
        );
        assert_eq!(
            chk(&[1, 48, 64, 128], &[1, 5, 64, 128], DType::BF16),
            Err(SageRefusal::Shape)
        );
        assert_eq!(
            chk(&[1, 4, 0, 128], &[1, 4, 64, 128], DType::BF16),
            Err(SageRefusal::Shape)
        );
        assert_eq!(
            chk(&[4, 64, 128], &[4, 64, 128], DType::BF16),
            Err(SageRefusal::Shape)
        );
        // E6: a padded INT8 buffer past i32::MAX elements refuses.
        let rows = (i32::MAX as usize) / (8 * 128);
        let fit = [1, 8, rows / 128 * 128, 128];
        assert!(chk(&fit, &fit, DType::BF16).is_ok());
        let over = [1, 8, rows + 1, 128];
        assert_eq!(
            chk(&over, &fit, DType::BF16),
            Err(SageRefusal::IndexOverflow)
        );
    }

    #[test]
    fn the_rotation_is_orthogonal_and_keeps_dot_products() {
        for d in SAGE_HEAD_DIMS {
            let a: Vec<f32> = (0..d).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
            let b: Vec<f32> = (0..d).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.7).collect();
            let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
            let (mut ra, mut rb) = (a.clone(), b.clone());
            rotate_reference(&mut ra);
            rotate_reference(&mut rb);
            assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-3, "d={d}");
            assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-3, "d={d}");
            // An outlier channel is spread over every channel.
            let mut spike = vec![0f32; d];
            spike[3] = 100.0;
            rotate_reference(&mut spike);
            let amax = spike.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(
                (amax - 100.0 / (d as f32).sqrt()).abs() < 1e-3,
                "d={d}: {amax}"
            );
        }
    }

    #[test]
    fn the_quantizer_rounds_to_nearest_even_and_scales_by_amax() {
        let (codes, scale) = quantize_row_reference(&[1.0, -0.5, 0.25, 0.0], 2.0);
        assert_eq!(codes, vec![127, -64, 32, 0]);
        assert_eq!(scale, 2.0 / 127.0);
        let (codes, scale) = quantize_row_reference(&[0.0; 4], 1.0);
        assert_eq!((codes, scale), (vec![0; 4], 0.0));
    }

    #[test]
    fn workspace_is_the_int8_buffers_and_scales() {
        let q = [1, 48, 4096, 128];
        let plan = check_sage_attention(&q, &q, &q, DType::BF16, DType::BF16, DType::BF16).unwrap();
        // 2 x (48 x 4096 x 132 B) + the centering mean/flag + 16 chunks of partial sums.
        assert_eq!(
            plan.workspace_bytes(),
            2 * 48 * 4096 * 132 + 48 * (4 * 128 + 4) + 48 * 16 * 4 * 128
        );
    }
}

// -------------------------------------------------------------------------------------------------
// CUDA compute - `#[ignore]`d, never part of normal CI. Run on a pinned validation GPU:
//
// CUDA_VISIBLE_DEVICES=<gpu> cargo test -p candle-quant-kernels --features cuda --lib \
//   sage_attention::cuda_tests:: -- --ignored --nocapture --test-threads=1
// -------------------------------------------------------------------------------------------------
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn device() -> Device {
        Device::new_cuda(0).expect("CUDA device 0")
    }

    /// The f32 materialized reference `softmax(q·kᵀ·scale)·v` (GQA expanded), `[B, H, Sq, D]`.
    fn reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
        let (_, h, _, _) = q.dims4()?;
        let hk = k.dim(1)?;
        let rep = |t: &Tensor| -> candle_core::Result<Tensor> {
            let t = t.to_dtype(DType::F32)?;
            if h == hk {
                return Ok(t);
            }
            let (b, _, s, d) = t.dims4()?;
            t.unsqueeze(2)?
                .expand((b, hk, h / hk, s, d))?
                .reshape((b, h, s, d))
        };
        let q = q.to_dtype(DType::F32)?;
        let (k, v) = (rep(k)?, rep(v)?);
        // Query-row chunks keep the f32 scores under ~1 GiB per chunk.
        let sq = q.dim(2)?;
        let sk = k.dim(2)?;
        let rows = (1usize << 28) / (h * sk).max(1);
        let rows = rows.clamp(1, sq);
        let kt = k.transpose(2, 3)?.contiguous()?;
        let mut out = Vec::new();
        let mut start = 0;
        while start < sq {
            let len = rows.min(sq - start);
            let s = (q.narrow(2, start, len)?.contiguous()?.matmul(&kt)? * scale)?;
            let p = candle_nn::ops::softmax_last_dim(&s)?;
            out.push(p.matmul(&v)?);
            start += len;
        }
        Tensor::cat(&out, 2)
    }

    /// `(cosine, rel-L2)` of `a` against `b`, in f32.
    fn similarity(a: &Tensor, b: &Tensor) -> candle_core::Result<(f32, f32)> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
        let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let nb = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let diff = (&a - &b)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok((
            dot / (na * nb).max(f32::MIN_POSITIVE),
            diff / nb.max(f32::MIN_POSITIVE),
        ))
    }

    /// Every entry point compiles through the seam with no local memory (the all-kernels check in
    /// candle-llm covers it too; this names the registers for the record).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sage_kernels_compile_without_local_memory() {
        let dev = device();
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let module = SOURCE.compiled(cuda).expect("compiles");
        for entry in module.entry_points() {
            let f = module.function(entry).unwrap();
            let local = f.local_size_bytes().unwrap();
            println!(
                "{entry}: {local} B local, {} registers",
                f.num_regs().unwrap()
            );
            assert_eq!(local, 0, "{entry} spills to local memory");
        }
        assert_eq!(module.entry_points().len(), 10);
    }

    /// E2: the device quantizer's codes are byte-identical to the host oracle (centering off), and
    /// its scales agree to one f32 ULP, for both head dims and both 16-bit dtypes, on a strided
    /// (head-transposed) operand with a partial tail block.
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sage_quantizer_codes_match_the_host_oracle() -> candle_core::Result<()> {
        let dev = device();
        for (d, dtype) in [
            (128, DType::BF16),
            (64, DType::F16),
            (128, DType::F16),
            (64, DType::BF16),
        ] {
            let (b, h, s) = (2usize, 3usize, 200usize);
            // [B, S, H, D] storage viewed as [B, H, S, D] (Krea's layout before the transpose copy).
            let x = (Tensor::randn(0f32, 1.0, (b, s, h, d), &dev)? * 2.0)?
                .to_dtype(dtype)?
                .transpose(1, 2)?;
            let scale = 1.0 / (d as f64).sqrt();
            let qk = prequantize_qk(&x, &x, x.dims(), dtype, scale, KCentering::Off)?;
            let pre = (scale as f32) * std::f32::consts::LOG2_E;
            let host = x.to_dtype(DType::F32)?.contiguous()?.to_vec3::<f32>().ok();
            let _ = host;
            let xs: Vec<f32> = x
                .to_dtype(DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1()?;
            let codes: Vec<u8> = qk.q_codes.flatten_all()?.to_vec1()?;
            let scales: Vec<f32> = qk.q_scales.flatten_all()?.to_vec1()?;
            let k_codes: Vec<u8> = qk.k_codes.flatten_all()?.to_vec1()?;
            let k_scales: Vec<f32> = qk.k_scales.flatten_all()?.to_vec1()?;
            let (qp, kp) = (qk.plan.q_pad, qk.plan.k_pad);
            let mut mismatches = 0usize;
            for bh in 0..b * h {
                for r in 0..qp.max(kp) {
                    let mut row = if r < s {
                        xs[(bh * s + r) * d..(bh * s + r + 1) * d].to_vec()
                    } else {
                        vec![0f32; d]
                    };
                    rotate_reference(&mut row);
                    for (pre_r, pad, cs, ss) in
                        [(pre, qp, &codes, &scales), (1.0, kp, &k_codes, &k_scales)]
                    {
                        if r >= pad {
                            continue;
                        }
                        let (want, want_scale) = quantize_row_reference(&row, pre_r);
                        let at = (bh * pad + r) * d;
                        let got: Vec<i8> = cs[at..at + d].iter().map(|&c| c as i8).collect();
                        mismatches += got.iter().zip(&want).filter(|(a, b)| a != b).count();
                        let got_scale = ss[bh * pad + r];
                        // The codes are byte-identical; the stored scale (`amax / 127 · pre`) may
                        // round its two f32 steps differently on the device: <= 1 ULP.
                        let ulps = (got_scale.to_bits() as i64 - want_scale.to_bits() as i64).abs();
                        if ulps > 1 {
                            let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                            panic!("scale d={d} {dtype:?} bh={bh} r={r}: got {got_scale:e} want {want_scale:e}, {bad} code mismatches; got {:?} want {:?}", &got[..8], &want[..8]);
                        }
                    }
                }
            }
            println!("quantizer d={d} {dtype:?}: {mismatches} code mismatches");
            assert_eq!(mismatches, 0, "d={d} {dtype:?}");
        }
        Ok(())
    }

    /// One random case through the one-call API vs the f32 reference (a quick smoke and timing).
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_sage_attention_smoke_and_timing() -> candle_core::Result<()> {
        let dev = device();
        let s: usize = std::env::var("SAGE_SMOKE_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let (b, h, d) = (1usize, 48usize, 128usize);
        let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let scale = 1.0 / (d as f64).sqrt();
        let out = sage_attention(&q, &k, &v, scale, KCentering::Adaptive)?;
        let want = reference(&q, &k, &v, scale)?;
        let (cos, rel) = similarity(&out, &want)?;
        println!("sage S={s} H={h} d={d}: cosine {cos:.6} rel-L2 {rel:.3e}");
        // Warm the clocks first (the board idles at a few hundred MHz).
        for _ in 0..50 {
            let _ = sage_attention(&q, &k, &v, scale, KCentering::Adaptive)?;
        }
        dev.synchronize()?;
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = sage_attention(&q, &k, &v, scale, KCentering::Adaptive)?;
        }
        dev.synchronize()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let flops = 4.0 * (b * h * s * s * d) as f64;
        println!(
            "sage S={s}: {ms:.3} ms/iter ({:.1} TFLOP/s)",
            flops / ms / 1e9
        );
        assert!(cos >= 0.999, "cosine {cos}");
        Ok(())
    }
}
