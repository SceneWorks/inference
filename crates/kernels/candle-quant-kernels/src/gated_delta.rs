//! The fused gated delta recurrence step (Gated DeltaNet decode and speculative verify), nvrtc-
//! compiled from [`GATED_DELTA_SRC`] through the [`nvrtc`](crate::nvrtc) seam.
//!
//! [`gated_delta_step`] runs the per-token recurrence of a `T`-token step in one kernel: one warp
//! per state row keeps the row in registers across the tokens and writes each token's post-step
//! row straight into its checkpoint-ring slot. The op-chain reference reads and writes the whole
//! `[B, Hv, Dv, Dk]` state about a dozen times per token (decay, two broadcast products and their
//! reductions, the rank-1 update, the ring copy); this reads it once and writes it once per token.
//!
//! Arithmetic: bit-identical to the op-chain reference (each elementwise step is the same IEEE
//! operation, and both per-row dot products sum in exactly candle `fast_sum`'s tree order for a
//! power-of-two `Dk`), so a decode on the fused step is token-identical to one on the op chain.

use candle_core::{DType, Device, Tensor};

use crate::nvrtc::KernelSource;

/// The fused gated-delta source; compiled once per device on first use.
pub const GATED_DELTA_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_gated_delta_v2",
    src: include_str!("gated_delta.cu"),
    // Warp shuffles and f32 arithmetic only.
    cc_floor: (7, 0),
};

/// Widest key head the kernel serves.
pub const GATED_DELTA_MAX_KEY_DIM: usize = 256;

/// Whether the kernel serves a key head of `dk`: a power of two from 32 to 256 (one entry point
/// per width, so `fast_sum`'s tree is fixed at compile time).
pub fn gated_delta_serves_key_dim(dk: usize) -> bool {
    matches!(dk, 32 | 64 | 128 | 256)
}

/// Where a [`gated_delta_step`] reads the live recurrent state from.
#[derive(Clone, Copy, Debug)]
pub enum GatedDeltaState<'a> {
    /// No state yet: the step starts from zeros.
    Zero,
    /// A contiguous `[B, Hv, Dv, Dk]` `F32` state.
    Tensor(&'a Tensor),
    /// Slot `index[0]` (a device `u32`) of a contiguous `[slots, B, Hv, Dv, Dk]` `F32` ring; an
    /// out-of-range index reads zeros.
    Slot {
        /// The ring.
        ring: &'a Tensor,
        /// `[1]` `u32` slot index.
        index: &'a Tensor,
    },
}

/// Where a [`gated_delta_step`] writes each token's post-step state.
#[derive(Clone, Copy, Debug)]
pub struct GatedDeltaRing<'a> {
    /// The contiguous `[slots, B, Hv, Dv, Dk]` `F32` ring, written in place.
    pub ssm: &'a Tensor,
    /// `[T]` `u32` slot per token (contiguous); an out-of-range slot is skipped.
    pub write: &'a Tensor,
    /// Tokens before this one are not written (their slots would be overwritten by later ones).
    pub first_write: usize,
}

fn msg(text: String) -> candle_core::Error {
    candle_core::Error::Msg(text)
}

/// Whether the fused step serves `device` (CUDA where the source compiles); the reason otherwise.
pub fn gated_delta_available(device: &Device) -> Result<(), String> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => GATED_DELTA_SRC
            .compiled(dev)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        other => Err(format!("gated_delta_step: no kernel for {other:?}")),
    }
}

/// The telemetry label of why the fused step cannot serve `device` (`not_cuda`, or the cached
/// compile error's [`label`](crate::nvrtc::KernelCompileError::label)); `None` when it can.
pub fn gated_delta_unavailable_label(device: &Device) -> Option<&'static str> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => GATED_DELTA_SRC.compiled(dev).err().map(|e| e.label()),
        _ => Some("not_cuda"),
    }
}

/// One fused gated-delta step over `T` tokens.
///
/// `q`, `k`: `[B, T, Hk, Dk]`; `v`: `[B, T, Hv, Dv]`; `g` (the multiplicative decay) and `beta`:
/// `[B, T, Hv]` — all `F32`. `Hv` must be a multiple of `Hk` (value head `hv` reads key head
/// `hv / (Hv / Hk)`), `Dk` one [`gated_delta_serves_key_dim`] accepts. Reads the
/// live state from `state`, writes each token `t >= ring.first_write`'s post-step state into slot
/// `ring.write[t]` of `ring.ssm`, and returns `y`: `[B, T, Hv, Dv]` `F32`.
pub fn gated_delta_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: GatedDeltaState<'_>,
    ring: GatedDeltaRing<'_>,
) -> candle_core::Result<Tensor> {
    let (b, t, hk, dk) = q.dims4()?;
    let (vb, vt, hv, dv) = v.dims4()?;
    if k.dims() != q.dims() || (vb, vt) != (b, t) {
        return Err(msg(format!(
            "gated_delta_step: q {:?} / k {:?} / v {:?} disagree",
            q.dims(),
            k.dims(),
            v.dims()
        )));
    }
    if hk == 0 || hv % hk != 0 || !gated_delta_serves_key_dim(dk) {
        return Err(msg(format!(
            "gated_delta_step: Hk {hk} / Hv {hv} / Dk {dk} not served"
        )));
    }
    if g.dims() != [b, t, hv] || beta.dims() != [b, t, hv] {
        return Err(msg(format!(
            "gated_delta_step: g {:?} / beta {:?} must be [{b}, {t}, {hv}]",
            g.dims(),
            beta.dims()
        )));
    }
    let slot = [b, hv, dv, dk];
    let ring_dims = ring.ssm.dims();
    if ring_dims.len() != 5 || ring_dims[1..] != slot || !ring.ssm.is_contiguous() {
        return Err(msg(format!(
            "gated_delta_step: ring {ring_dims:?} is not a contiguous [slots, {b}, {hv}, {dv}, {dk}]"
        )));
    }
    if ring.write.dtype() != DType::U32 || ring.write.elem_count() < t {
        return Err(msg(format!(
            "gated_delta_step: {} write slots for {t} tokens",
            ring.write.elem_count()
        )));
    }
    match state {
        GatedDeltaState::Zero => {}
        GatedDeltaState::Tensor(s) if s.dims() == slot && s.is_contiguous() => {}
        GatedDeltaState::Slot { ring: r, index }
            if r.dims().len() == 5
                && r.dims()[1..] == slot
                && r.is_contiguous()
                && index.dtype() == DType::U32
                && index.elem_count() >= 1 => {}
        other => {
            return Err(msg(format!(
                "gated_delta_step: state {other:?} is not a contiguous {slot:?} state or slot"
            )))
        }
    }
    for (name, x) in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)] {
        if x.dtype() != DType::F32 {
            return Err(msg(format!("gated_delta_step: {name} must be F32")));
        }
    }
    for x in [ring.ssm] {
        if x.dtype() != DType::F32 {
            return Err(msg("gated_delta_step: the ring must be F32".into()));
        }
    }
    match q.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::launch(q, k, v, g, beta, state, ring, (b, t, hk, hv, dk, dv)),
        other => Err(msg(format!("gated_delta_step: no kernel for {other:?}"))),
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, Shape, Storage};

    const ROWS_PER_BLOCK: usize = 4;

    fn drv(e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Cuda(format!("gated_delta kernel: {e:?}").into())
    }

    /// The device address of `x`'s first element (`F32` or `U32`).
    fn address(
        x: &Tensor,
        stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
    ) -> candle_core::Result<u64> {
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(c) = &*storage else {
            return Err(msg("gated_delta_step: expected a CUDA tensor".into()));
        };
        let at = layout.start_offset();
        let ptr = match x.dtype() {
            DType::F32 => {
                let s = c.as_cuda_slice::<f32>()?;
                s.device_ptr(stream).0 + (at * 4) as u64
            }
            DType::U32 => {
                let s = c.as_cuda_slice::<u32>()?;
                s.device_ptr(stream).0 + (at * 4) as u64
            }
            other => return Err(msg(format!("gated_delta_step: dtype {other:?}"))),
        };
        Ok(ptr)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        state: GatedDeltaState<'_>,
        ring: GatedDeltaRing<'_>,
        (b, t, hk, hv, dk, dv): (usize, usize, usize, usize, usize, usize),
    ) -> candle_core::Result<Tensor> {
        let Device::Cuda(dev) = q.device() else {
            unreachable!("dispatched on a CUDA device")
        };
        let module = GATED_DELTA_SRC
            .compiled(dev)
            .map_err(|e| msg(e.to_string()))?;
        let f = module
            .function(&format!("gated_delta_step_f32_dk{dk}"))
            .map_err(|e| msg(e.to_string()))?;
        let (q, k, v, g, beta) = (
            q.contiguous()?,
            k.contiguous()?,
            v.contiguous()?,
            g.contiguous()?,
            beta.contiguous()?,
        );
        let write = ring.write.contiguous()?;
        let stream = dev.cuda_stream();
        let (src, src_index, src_slots, has_state) = match state {
            GatedDeltaState::Zero => (0u64, 0u64, 0u64, 0i32),
            GatedDeltaState::Tensor(s) => (address(s, &stream)?, 0, 1, 1),
            GatedDeltaState::Slot { ring: r, index } => (
                address(r, &stream)?,
                address(index, &stream)?,
                r.dim(0)? as u64,
                1,
            ),
        };
        let ring_ptr = address(ring.ssm, &stream)?;
        let write_ptr = address(&write, &stream)?;
        let ring_slots = ring.ssm.dim(0)? as u64;
        let first_write = ring.first_write.min(t) as i32;
        let (qp, kp, vp, gp, bp) = (
            address(&q, &stream)?,
            address(&k, &stream)?,
            address(&v, &stream)?,
            address(&g, &stream)?,
            address(&beta, &stream)?,
        );
        let n = b * t * hv * dv;
        // SAFETY: the kernel writes every element of y (one per (b, t, hv, dv)).
        let y = unsafe { dev.alloc::<f32>(n) }?;
        let yp = y.device_ptr(&stream).0;
        let rows = b * hv * dv;
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
            block_dim: ((ROWS_PER_BLOCK * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let dims = [b as i32, t as i32, hk as i32, hv as i32, dv as i32];
        let mut builder = stream.launch_builder(&f);
        builder
            .arg(&src)
            .arg(&src_index)
            .arg(&src_slots)
            .arg(&has_state)
            .arg(&ring_ptr)
            .arg(&write_ptr)
            .arg(&ring_slots)
            .arg(&first_write)
            .arg(&qp)
            .arg(&kp)
            .arg(&vp)
            .arg(&gp)
            .arg(&bp)
            .arg(&yp);
        for d in &dims {
            builder.arg(d);
        }
        // SAFETY: the argument list matches `gated_delta_step_f32`; every pointer is a live
        // device buffer of the checked shape (the inputs are held for the launch's duration by
        // the bindings above, and the stream orders the launch before any later free).
        unsafe { builder.launch(cfg) }.map_err(drv)?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(y, dev.clone())),
            Shape::from((b, t, hv, dv)),
            BackpropOp::none(),
            false,
        ))
    }
}
