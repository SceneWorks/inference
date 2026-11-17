//! Fused Gated DeltaNet input preparation (Qwen3.5 / Qwen3.8 linear-attention layers),
//! nvrtc-compiled from [`GDN_PREP_SRC`].
//!
//! Three kernels replace the op chains a linear layer runs before its recurrence (about 45 small
//! kernels per layer per decode step on candle's CUDA backend):
//!
//! - [`gdn_conv_silu`]: the short causal depthwise conv over `[conv state ; x]` and its SiLU,
//!   also writing the concatenation the checkpoint ring's conv tails are read from;
//! - [`gdn_qkv`]: the q / k L2 norms (q scaled), widened to `F32` with v;
//! - [`gdn_gates`]: the decay gate `g` and the delta strength `beta`, widened to `F32`.
//!
//! All three are **bit-identical** to the candle op chains they replace (bf16 operands), see the
//! numerics contract at the top of `gdn_prep.cu`; the CUDA parity tests in `candle-llm` compare
//! them bit for bit.

use candle_core::{DType, Device, Tensor};

use crate::nvrtc::KernelSource;

/// The fused GDN-prep source; compiled once per device on first use.
pub const GDN_PREP_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_gdn_prep_v1",
    src: include_str!("gdn_prep.cu"),
    // Warp shuffles, f32 libdevice math and software bf16 conversions only.
    cc_floor: (7, 0),
};

fn msg(text: String) -> candle_core::Error {
    candle_core::Error::Msg(text)
}

/// Whether the fused GDN-prep kernels serve `device` (CUDA where the source compiles).
pub fn gdn_prep_available(device: &Device) -> Result<(), String> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => GDN_PREP_SRC
            .compiled(dev)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        other => Err(format!("gdn_prep: no kernel for {other:?}")),
    }
}

fn require_bf16(name: &str, x: &Tensor) -> candle_core::Result<()> {
    if x.dtype() != DType::BF16 {
        return Err(msg(format!(
            "gdn_prep: {name} must be BF16, got {:?}",
            x.dtype()
        )));
    }
    Ok(())
}

/// The causal depthwise conv + SiLU of one step: `state` `[B, K-1, C]`, `x` `[B, S, C]`,
/// `weight` `[C, K]`, all `BF16`. Returns `(cat, out)`: `cat = [state ; x]` `[B, S+K-1, C]` and
/// `out = silu(conv)` `[B, S, C]` — candle-llm's `causal_depthwise_conv_traced`, bit for bit.
pub fn gdn_conv_silu(
    state: &Tensor,
    x: &Tensor,
    weight: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let (b, _, c) = x.dims3()?;
    let (wc, k) = weight.dims2()?;
    if state.dims() != [b, k.saturating_sub(1), c] || wc != c || k == 0 {
        return Err(msg(format!(
            "gdn_conv_silu: state {:?} / x {:?} / weight {:?} disagree",
            state.dims(),
            x.dims(),
            weight.dims()
        )));
    }
    for (n, t) in [("state", state), ("x", x), ("weight", weight)] {
        require_bf16(n, t)?;
    }
    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::conv_silu(state, x, weight, k),
        other => Err(msg(format!("gdn_conv_silu: no kernel for {other:?}"))),
    }
}

/// Whether [`gdn_qkv`] serves a key head of `dk` (a power of two from 32 to 256).
pub fn gdn_qkv_serves_key_dim(dk: usize) -> bool {
    matches!(dk, 32 | 64 | 128 | 256)
}

/// q / k / v of one step from the conv output `conv` `[B, S, 2·Hk·Dk + Hv·Dv]` (`BF16`, the
/// q ‖ k ‖ v channels): `q = l2norm(q)·scale`, `k = l2norm(k)`, `v` — returned widened to `F32`
/// as `[B, S, Hk, Dk]`, `[B, S, Hk, Dk]`, `[B, S, Hv, Dv]`. `eps` and `scale` are rounded to
/// bf16 exactly as candle's bf16 `affine` receives them. Bit-identical to candle-llm's
/// `l2norm(..)` (+ `affine(scale, 0)`) and `to_dtype(F32)` chain.
#[allow(clippy::too_many_arguments)]
pub fn gdn_qkv(
    conv: &Tensor,
    hk: usize,
    dk: usize,
    hv: usize,
    dv: usize,
    eps: f64,
    scale: f64,
) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let (b, s, c) = conv.dims3()?;
    if c != 2 * hk * dk + hv * dv || !gdn_qkv_serves_key_dim(dk) || hk == 0 || hv == 0 {
        return Err(msg(format!(
            "gdn_qkv: {c} channels for Hk {hk} Dk {dk} Hv {hv} Dv {dv} not served"
        )));
    }
    require_bf16("conv", conv)?;
    let eps = half::bf16::from_f64(eps).to_f32();
    let scale = half::bf16::from_f64(scale).to_f32();
    match conv.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::qkv(conv, (b, s, hk, dk, hv, dv), eps, scale),
        other => {
            let _ = (b, s, eps, scale);
            Err(msg(format!("gdn_qkv: no kernel for {other:?}")))
        }
    }
}

/// The gates of one step: `a`, `b` `[B, S, Hv]`, `a_log`, `dt_bias` `[Hv]`, all `BF16`. Returns
/// `(g, beta)` `[B, S, Hv]` `F32`: `g = bf16(exp(-exp(A_log)·softplus(a + dt_bias)))` and
/// `beta = sigmoid(b)` — candle-llm's `compute_g` and `sigmoid`, widened, bit for bit.
pub fn gdn_gates(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> candle_core::Result<(Tensor, Tensor)> {
    let hv = a.dim(candle_core::D::Minus1)?;
    if b.dims() != a.dims() || a_log.dims() != [hv] || dt_bias.dims() != [hv] {
        return Err(msg(format!(
            "gdn_gates: a {:?} / b {:?} / a_log {:?} / dt_bias {:?} disagree",
            a.dims(),
            b.dims(),
            a_log.dims(),
            dt_bias.dims()
        )));
    }
    for (n, t) in [("a", a), ("b", b), ("a_log", a_log), ("dt_bias", dt_bias)] {
        require_bf16(n, t)?;
    }
    match a.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::gates(a, b, a_log, dt_bias, hv),
        other => Err(msg(format!("gdn_gates: no kernel for {other:?}"))),
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::CudaDType;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Shape, Storage};
    use std::sync::Arc;

    fn drv(e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Cuda(format!("gdn_prep kernel: {e:?}").into())
    }

    fn function(dev: &CudaDevice, name: &str) -> candle_core::Result<CudaFunction> {
        let module = GDN_PREP_SRC.compiled(dev).map_err(|e| msg(e.to_string()))?;
        module.function(name).map_err(|e| msg(e.to_string()))
    }

    fn device(x: &Tensor) -> candle_core::Result<CudaDevice> {
        match x.device() {
            Device::Cuda(d) => Ok(d.clone()),
            other => Err(msg(format!("gdn_prep: expected CUDA, got {other:?}"))),
        }
    }

    /// The device address of a contiguous `BF16` tensor's first element.
    fn bf16_address(x: &Tensor, stream: &Arc<CudaStream>) -> candle_core::Result<u64> {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(c) = &*storage else {
            return Err(msg("gdn_prep: expected a CUDA tensor".into()));
        };
        let s = c.as_cuda_slice::<half::bf16>()?;
        let base = s.device_ptr(stream).0;
        Ok(base + (layout.start_offset() * 2) as u64)
    }

    fn wrap<T: CudaDType>(dev: &CudaDevice, s: CudaSlice<T>, shape: Shape) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev.clone())),
            shape,
            BackpropOp::none(),
            false,
        )
    }

    fn grid(n: usize, block: u32) -> LaunchConfig {
        let blocks = n.div_ceil(block as usize).clamp(1, 1 << 20) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(super) fn conv_silu(
        state: &Tensor,
        x: &Tensor,
        weight: &Tensor,
        k: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (b, s, c) = x.dims3()?;
        let dev = device(x)?;
        let (state, x, weight) = (state.contiguous()?, x.contiguous()?, weight.contiguous()?);
        let stream = dev.cuda_stream();
        let l = s + k - 1;
        // SAFETY: the kernel writes every element of both outputs.
        let cat = unsafe { dev.alloc::<half::bf16>(b * l * c) }?;
        let out = unsafe { dev.alloc::<half::bf16>(b * s * c) }?;
        let (sp, xp, wp) = (
            bf16_address(&state, &stream)?,
            bf16_address(&x, &stream)?,
            bf16_address(&weight, &stream)?,
        );
        let (cp, op) = (cat.device_ptr(&stream).0, out.device_ptr(&stream).0);
        let f = function(&dev, "gdn_conv_silu_bf16")?;
        let dims = [b as i32, s as i32, c as i32, k as i32];
        let mut builder = stream.launch_builder(&f);
        builder.arg(&sp).arg(&xp).arg(&wp).arg(&cp).arg(&op);
        for d in &dims {
            builder.arg(d);
        }
        // SAFETY: the argument list matches `gdn_conv_silu_bf16`; inputs are held by the bindings
        // above for the launch, outputs are fresh allocations of the written size.
        unsafe { builder.launch(grid(b * l * c, 256)) }.map_err(drv)?;
        Ok((
            wrap(&dev, cat, Shape::from((b, l, c))),
            wrap(&dev, out, Shape::from((b, s, c))),
        ))
    }

    pub(super) fn qkv(
        conv: &Tensor,
        (b, s, hk, dk, hv, dv): (usize, usize, usize, usize, usize, usize),
        eps: f32,
        scale: f32,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let dev = device(conv)?;
        let conv = conv.contiguous()?;
        let stream = dev.cuda_stream();
        // SAFETY: the kernel writes every element of q, k and v.
        let q = unsafe { dev.alloc::<f32>(b * s * hk * dk) }?;
        let k = unsafe { dev.alloc::<f32>(b * s * hk * dk) }?;
        let v = unsafe { dev.alloc::<f32>(b * s * hv * dv) }?;
        let cp = bf16_address(&conv, &stream)?;
        let (qp, kp, vp) = (
            q.device_ptr(&stream).0,
            k.device_ptr(&stream).0,
            v.device_ptr(&stream).0,
        );
        let f = function(&dev, &format!("gdn_qkv_bf16_dk{dk}"))?;
        let warps = b * s * (2 * hk + hv);
        let dims = [b as i32, s as i32, hk as i32, hv as i32, dv as i32];
        let mut builder = stream.launch_builder(&f);
        builder.arg(&cp).arg(&qp).arg(&kp).arg(&vp);
        for d in &dims {
            builder.arg(d);
        }
        builder.arg(&eps).arg(&scale);
        // 4 warps per block.
        // SAFETY: the argument list matches `gdn_qkv_bf16_dk*`; the conv input is held above and
        // the outputs are fresh allocations of the written size.
        unsafe { builder.launch(grid(warps * 32, 128)) }.map_err(drv)?;
        Ok((
            wrap(&dev, q, Shape::from((b, s, hk, dk))),
            wrap(&dev, k, Shape::from((b, s, hk, dk))),
            wrap(&dev, v, Shape::from((b, s, hv, dv))),
        ))
    }

    pub(super) fn gates(
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        hv: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let dev = device(a)?;
        let (a, b, a_log, dt_bias) = (
            a.contiguous()?,
            b.contiguous()?,
            a_log.contiguous()?,
            dt_bias.contiguous()?,
        );
        let n = a.elem_count();
        let stream = dev.cuda_stream();
        // SAFETY: the kernel writes every element of g and beta.
        let g = unsafe { dev.alloc::<f32>(n) }?;
        let beta = unsafe { dev.alloc::<f32>(n) }?;
        let (ap, bp, lp, dp) = (
            bf16_address(&a, &stream)?,
            bf16_address(&b, &stream)?,
            bf16_address(&a_log, &stream)?,
            bf16_address(&dt_bias, &stream)?,
        );
        let (gp, betap) = (g.device_ptr(&stream).0, beta.device_ptr(&stream).0);
        let f = function(&dev, "gdn_gates_bf16")?;
        let (n64, hv32) = (n as i64, hv as i32);
        let mut builder = stream.launch_builder(&f);
        builder
            .arg(&ap)
            .arg(&bp)
            .arg(&lp)
            .arg(&dp)
            .arg(&gp)
            .arg(&betap)
            .arg(&n64)
            .arg(&hv32);
        // SAFETY: the argument list matches `gdn_gates_bf16`; inputs are held above and the
        // outputs are fresh allocations of `n` elements.
        unsafe { builder.launch(grid(n, 128)) }.map_err(drv)?;
        let shape = Shape::from(a.dims());
        Ok((wrap(&dev, g, shape.clone()), wrap(&dev, beta, shape)))
    }
}
