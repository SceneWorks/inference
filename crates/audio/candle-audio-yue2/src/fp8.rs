//! The experimental FP8 AR mode (sc-22995, epic E8) — a native port of upstream's opt-in
//! `src/yue2/quantization.py` (`prepare_fp8_ar` / `FP8Linear` / `restore_ar` /
//! `quantization_status`), Apache-2.0, commit
//! [`YUE2_SOURCE_COMMIT`](crate::inventory::YUE2_SOURCE_COMMIT).
//!
//! Upstream offers it as `quantization="fp8"` with "no quantized quality or speed claim". What it
//! does, and what this module does the same way:
//!
//! * **Which weights.** Only the AR path's seven projections of every layer
//!   (`model.layers.N.self_attn.{q,k,v,o}_proj`, `model.layers.N.mlp.{gate,up,down}_proj`) —
//!   never `lm_head`, the embedding, the NAR twins or the NAR heads.
//! * **Weights.** Per-tensor E4M3: `scale = max(amax(|w|), 1e-12) / 448`, `q = clamp(w / scale,
//!   ±448)` cast to `float8_e4m3fn`, computed in F32 from the BF16 weight.
//! * **Activations.** Dynamic per-tensor E4M3 of the BF16 input (rows zero-padded to a multiple of
//!   16, as upstream pads for cuBLAS), same formula.
//! * **Matmul.** A real FP8 GEMM — upstream's `torch._scaled_mm(…, use_fast_accum=False)` is the
//!   cuBLASLt E4M3 × E4M3 GEMM with per-tensor scale pointers, FP32 accumulate and BF16 output,
//!   which is exactly [`candle_quant_kernels::cublaslt`]'s `matmul_fp8_staged` (fast-accumulate
//!   off). Nothing dequantizes the weight to BF16 for the matmul.
//! * **Hardware.** CUDA compute capability ≥ 8.9 ([`candle_quant_kernels::FP8_COMPUTE_CAP_FLOOR`],
//!   the workspace's one definition of the 8-bit floor). Any other device — CPU, Metal, an older
//!   CUDA GPU, or a build without the `cuda` feature — is refused with
//!   [`gen_core::Error::Unsupported`]; the mode never falls back to another precision.
//! * **Preconditions.** The released BF16 weights in BF16 compute (the `bf16` tier on an
//!   accelerator): a `q8` / `q4` tier or an F32 model is refused, as upstream refuses non-BF16
//!   AR weights. Every AR projection dimension must be a multiple of 16.
//! * **Originals.** Each replaced BF16 weight is moved to host memory, outside the model, and kept
//!   there; [`Fp8Status::host_original_bytes`] and
//!   [`crate::model::Yue2Lm::weight_residency`]'s `host_bytes` count them (upstream's
//!   `original_weights: "cpu_bfloat16"`).
//! * **Restore.** [`restore_ar_bf16`] puts the exact originals back on the device (a byte copy of
//!   the same tensors — bit-identical), and the engine calls it **before the acoustic stage** (whose
//!   AR-path prefill of the song prefix must run the BF16 weights, as upstream's `synthesize` calls
//!   `restore_ar` first) and whenever the model is used by anything but the AR stages. The next AR
//!   stage prepares FP8 again, as upstream's `_load_model` does.
//!
//! Nothing in the native runtime serializes in-memory model weights: [`crate::tier::convert`] and
//! [`crate::closure::save_closure`] copy verified files from disk, so an FP8-prepared model can never
//! be saved.
//!
//! Stage identities and the effective configuration record the mode, so an FP8 plan or semantic
//! result is never reused for a native run (or the reverse). The acoustic stage — always BF16 — is
//! identical in both modes.

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core;
use serde_json::{json, Value};

use crate::model::Yue2Lm;
use crate::precision::Tier;
#[cfg(feature = "cuda")]
use crate::weights::Proj;

/// FP8 E4M3's largest finite magnitude (upstream `torch.finfo(torch.float8_e4m3fn).max`).
pub const E4M3_MAX: f32 = 448.0;
/// The lower clamp upstream applies to a tensor's absolute maximum before dividing by
/// [`E4M3_MAX`].
pub const AMAX_FLOOR: f32 = 1e-12;
/// Rows of an FP8 GEMM operand are zero-padded to a multiple of this (upstream's cuBLAS padding).
pub const ROW_ALIGN: usize = 16;
/// AR projections per layer (q, k, v, o, gate, up, down).
pub const AR_LINEARS_PER_LAYER: usize = 7;

/// How the AR stages multiply the AR projections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ArPrecision {
    /// The loaded tier's own weights (upstream `quantization="none"`).
    #[default]
    Native,
    /// The experimental FP8 E4M3 AR mode (upstream `quantization="fp8"`): CUDA sm_89+ over the
    /// `bf16` tier only.
    Fp8,
}

impl ArPrecision {
    /// Upstream's `quantization` value: `none` / `fp8`.
    pub fn name(self) -> &'static str {
        match self {
            ArPrecision::Native => "none",
            ArPrecision::Fp8 => "fp8",
        }
    }
}

/// Upstream's `quantization_status`, plus the bytes the mode occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp8Status {
    /// `true` while the AR projections are FP8.
    pub active: bool,
    /// AR linears currently FP8.
    pub active_ar_linears: usize,
    /// Device bytes of the FP8 weights and their scales.
    pub device_fp8_bytes: u64,
    /// Host bytes of the retained BF16 originals.
    pub host_original_bytes: u64,
}

impl Fp8Status {
    /// The status record (upstream's keys, plus the residency).
    pub fn to_json(&self) -> Value {
        json!({
            "mode": if self.active { "fp8" } else { "none" },
            "active_ar_linears": self.active_ar_linears,
            "weight_format": if self.active { Value::from("float8_e4m3fn") } else { Value::Null },
            "original_weights": if self.active { Value::from("cpu_bfloat16") } else { Value::Null },
            "device_fp8_bytes": self.device_fp8_bytes,
            "host_original_bytes": self.host_original_bytes,
            "quality_validation": "experimental (sc-22995 measurements; not a benchmark)",
            "performance_validation": "unvalidated",
        })
    }
}

/// The FP8 mode's retained state on a [`Yue2Lm`]: the host-resident BF16 originals of every
/// replaced AR projection (layer-major, upstream AR-linear order). The cuBLASLt handle is shared by
/// the FP8 weights themselves (`Fp8Weight`, CUDA builds only).
pub struct Fp8State {
    #[cfg(feature = "cuda")]
    originals: Vec<Vec<Tensor>>,
    #[cfg(not(feature = "cuda"))]
    never: std::convert::Infallible,
}

impl std::fmt::Debug for Fp8State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fp8State")
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}

impl Fp8State {
    /// Host bytes of the retained originals.
    pub fn original_bytes(&self) -> u64 {
        #[cfg(feature = "cuda")]
        {
            self.originals
                .iter()
                .flatten()
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum()
        }
        #[cfg(not(feature = "cuda"))]
        match self.never {}
    }
}

fn unsupported(why: impl std::fmt::Display) -> gen_core::Error {
    gen_core::Error::Unsupported(format!(
        "yue2: the experimental FP8 AR mode is unavailable: {why}"
    ))
}

/// Refuse the FP8 mode for anything but the `bf16` tier computing in BF16 on CUDA (the device's
/// compute capability is checked when the mode is prepared, [`prepare_fp8_ar`]).
pub fn check_fp8_request(tier: Tier, dtype: DType, device: &Device) -> gen_core::Result<()> {
    if tier != Tier::Bf16 {
        return Err(unsupported(format!(
            "it quantizes the original BF16 AR weights; the {tier} tier holds none"
        )));
    }
    if dtype != DType::BF16 {
        return Err(unsupported(format!(
            "it needs the BF16 compute dtype, this model computes in {dtype:?}"
        )));
    }
    if !device.is_cuda() {
        return Err(unsupported(format!(
            "it needs a CUDA device of compute capability >= 8.9, this model is on {}",
            match device {
                Device::Cpu => "the CPU",
                Device::Metal(_) => "Metal",
                Device::Cuda(_) => "CUDA",
            }
        )));
    }
    if !cfg!(feature = "cuda") {
        return Err(unsupported("this build has no CUDA support"));
    }
    Ok(())
}

/// Per-tensor E4M3 quantization (upstream `quantize_tensor`): `scale = max(amax, 1e-12) / 448`,
/// `q = clamp(x / scale, ±448)` in F32, cast to `F8E4M3`. The cast runs on the host (Candle's CPU
/// E4M3 conversion; the device cast kernel is not in every fatbin — see
/// `candle_quant_kernels::cublaslt::quantize_activation_fp8`), and `q` is returned on `x`'s
/// device.
pub fn quantize_e4m3(x: &Tensor) -> candle_audio::candle_core::Result<(Tensor, f32)> {
    let f = x.to_dtype(DType::F32)?;
    let amax = f.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let scale = amax.max(AMAX_FLOOR) / E4M3_MAX;
    let q = f
        .broadcast_div(&Tensor::new(scale, f.device())?)?
        .clamp(-E4M3_MAX, E4M3_MAX)?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F8E4M3)?
        .to_device(x.device())?;
    Ok((q, scale))
}

/// Swap every AR projection of `lm` for an FP8 copy (upstream `prepare_fp8_ar`), moving the BF16
/// originals to host memory. Idempotent: an already-prepared model is left as is. Refused (see
/// [`check_fp8_request`]) off CUDA sm_89+, for a quantized tier or a non-BF16 model, while the AR
/// path is offloaded, and when any AR projection is not a dense BF16 weight whose dimensions are
/// multiples of 16. A failure part-way restores the projections already swapped.
pub fn prepare_fp8_ar(lm: &mut Yue2Lm) -> gen_core::Result<Fp8Status> {
    if lm.fp8.is_some() {
        return Ok(status(lm));
    }
    check_fp8_request(lm.tier(), lm.dtype(), lm.device())?;
    if lm.ar_offloaded() {
        return Err(unsupported(
            "the AR path is offloaded; restore it before preparing FP8",
        ));
    }
    for layer in lm.layers() {
        for p in layer.ar.projections() {
            match p.dense() {
                Some(t) if t.dtype() == DType::BF16 => {
                    let dims = t.dims();
                    if dims.iter().any(|d| d % ROW_ALIGN != 0) {
                        return Err(unsupported(format!(
                            "FP8 matrix dimensions must be multiples of 16, got {dims:?}"
                        )));
                    }
                }
                _ => {
                    return Err(unsupported(format!(
                        "it needs the original BF16 AR weights, found a {} weight",
                        p.label()
                    )))
                }
            }
        }
    }
    #[cfg(feature = "cuda")]
    {
        cuda::prepare(lm)?;
        Ok(status(lm))
    }
    #[cfg(not(feature = "cuda"))]
    Err(unsupported("this build has no CUDA support"))
}

/// Put the exact BF16 originals back (upstream `restore_ar`): every FP8 AR projection becomes its
/// retained original again, copied to the model device byte for byte. A no-op when the mode is not
/// active. Retrying after a failure resumes where it stopped (projections already restored are
/// skipped, the originals are released only once all are back).
pub fn restore_ar_bf16(lm: &mut Yue2Lm) -> gen_core::Result<()> {
    #[cfg(feature = "cuda")]
    {
        cuda::restore(lm)
    }
    #[cfg(not(feature = "cuda"))]
    match lm.fp8.as_ref().map(|s| s.never) {
        Some(never) => match never {},
        None => Ok(()),
    }
}

/// The mode's status on `lm` (upstream `quantization_status`).
pub fn status(lm: &Yue2Lm) -> Fp8Status {
    #[allow(unused_mut)]
    let mut s = Fp8Status {
        active: lm.fp8.is_some(),
        active_ar_linears: 0,
        device_fp8_bytes: 0,
        host_original_bytes: lm.fp8.as_ref().map_or(0, Fp8State::original_bytes),
    };
    #[cfg(feature = "cuda")]
    for layer in lm.layers() {
        for p in layer.ar.projections() {
            if let Proj::Fp8(w) = p {
                s.active_ar_linears += 1;
                s.device_fp8_bytes += w.resident_bytes();
            }
        }
    }
    s
}

/// An FP8 E4M3 AR projection resident on a CUDA device: the staged weight bytes, its per-tensor
/// scale, and the shared cuBLASLt handle.
#[cfg(feature = "cuda")]
pub struct Fp8Weight {
    w: candle_quant_kernels::cublaslt::DevFp8,
    scale: f32,
    shape: (usize, usize),
    device: Device,
    lt: std::sync::Arc<candle_quant_kernels::cublaslt::CublasLt>,
}

#[cfg(feature = "cuda")]
impl Fp8Weight {
    /// `(out, in)`.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The weight's per-tensor scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The CUDA device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The FP8 bytes plus the F32 scale.
    pub fn resident_bytes(&self) -> u64 {
        (self.shape.0 * self.shape.1) as u64 + 4
    }

    /// Upstream `FP8Linear.forward`: BF16 `x` `[…, in]` → pad rows to 16 → dynamic per-tensor
    /// E4M3 → cuBLASLt FP8 GEMM (FP32 accumulate) → BF16 `[…, out]`.
    pub fn forward(
        &self,
        x: &Tensor,
        bias: Option<&Tensor>,
    ) -> candle_audio::candle_core::Result<Tensor> {
        use candle_audio::candle_core::bail;
        if !x.device().is_cuda() {
            bail!("FP8 execution requires CUDA; restore the BF16 AR path before CPU/Metal use");
        }
        if x.dtype() != DType::BF16 {
            bail!("FP8 AR expects BF16 input, got {:?}", x.dtype());
        }
        let (n, k) = self.shape;
        let dims = x.dims().to_vec();
        if dims.last() != Some(&k) {
            bail!("FP8 AR: input {dims:?} does not end in {k}");
        }
        let m = x.elem_count() / k;
        let rows = x.reshape((m, k))?;
        let pad = (ROW_ALIGN - m % ROW_ALIGN) % ROW_ALIGN;
        let padded = if pad == 0 {
            rows
        } else {
            Tensor::cat(
                &[&rows, &Tensor::zeros((pad, k), DType::BF16, x.device())?],
                0,
            )?
        };
        let (q, x_scale) = quantize_e4m3(&padded)?;
        let xq = self.lt.stage_fp8(&q)?;
        let y = self
            .lt
            .matmul_fp8_staged(&self.w, self.scale, &xq, x_scale)?;
        let mut out_dims = dims;
        *out_dims.last_mut().expect("non-empty") = n;
        let y = y.narrow(0, 0, m)?.reshape(out_dims)?;
        match bias {
            Some(b) => y.broadcast_add(&b.to_dtype(DType::BF16)?),
            None => Ok(y),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::Arc;

    use candle_quant_kernels::cublaslt::CublasLt;

    use super::*;

    fn err(what: &'static str) -> impl Fn(candle_audio::candle_core::Error) -> gen_core::Error {
        move |e| gen_core::Error::Msg(format!("yue2 FP8 AR {what}: {e}"))
    }

    pub(super) fn prepare(lm: &mut Yue2Lm) -> gen_core::Result<()> {
        let device = lm.device().clone();
        let lt = Arc::new(CublasLt::new(&device).map_err(err("cuBLASLt handle"))?);
        let cap = lt.compute_cap().map_err(err("compute capability"))?;
        if !candle_quant_kernels::compute_cap_meets_fp8_floor(cap) {
            return Err(unsupported(format!(
                "it needs CUDA compute capability >= {:?}, this device is {cap:?}",
                candle_quant_kernels::FP8_COMPUTE_CAP_FLOOR
            )));
        }
        lm.fp8 = Some(Fp8State {
            originals: Vec::with_capacity(lm.layers().len()),
        });
        let swapped = swap_all(lm, &lt, &device);
        if let Err(e) = swapped {
            // Upstream: `except BaseException: restore_ar(model); raise`.
            return Err(match restore(lm) {
                Ok(()) => e,
                Err(r) => gen_core::Error::Msg(format!(
                    "{e}; restoring the BF16 AR projections afterwards also failed: {r}"
                )),
            });
        }
        Ok(())
    }

    fn swap_all(lm: &mut Yue2Lm, lt: &Arc<CublasLt>, device: &Device) -> gen_core::Result<()> {
        let layers = lm.layers().len();
        for i in 0..layers {
            lm.fp8
                .as_mut()
                .expect("prepared above")
                .originals
                .push(Vec::with_capacity(AR_LINEARS_PER_LAYER));
            for j in 0..AR_LINEARS_PER_LAYER {
                let weight = {
                    let p = lm.layers()[i].ar.projections()[j];
                    p.dense().expect("checked dense BF16").clone()
                };
                let host = weight
                    .to_device(&Device::Cpu)
                    .map_err(err("move original"))?;
                let (q, scale) = quantize_e4m3(&host).map_err(err("quantize weight"))?;
                let q = q.to_device(device).map_err(err("upload weight"))?;
                let staged = lt.stage_fp8(&q).map_err(err("stage weight"))?;
                let fp8 = Fp8Weight {
                    w: staged,
                    scale,
                    shape: weight.dims2().map_err(err("weight shape"))?,
                    device: device.clone(),
                    lt: Arc::clone(lt),
                };
                // The original is kept before the device copy is dropped.
                lm.fp8.as_mut().expect("prepared above").originals[i].push(host);
                *lm.layers_mut()[i].ar.projections_mut()[j] = Proj::Fp8(Box::new(fp8));
            }
        }
        Ok(())
    }

    pub(super) fn restore(lm: &mut Yue2Lm) -> gen_core::Result<()> {
        let Some(state) = lm.fp8.take() else {
            return Ok(());
        };
        let device = lm.device().clone();
        let mut result = Ok(());
        'outer: for (i, kept) in state.originals.iter().enumerate() {
            let mut projections = lm.layers_mut()[i].ar.projections_mut();
            for (j, original) in kept.iter().enumerate() {
                if matches!(*projections[j], Proj::Fp8(_)) {
                    match original.to_device(&device) {
                        Ok(t) => *projections[j] = Proj::Dense(t),
                        Err(e) => {
                            result = Err(err("restore original")(e));
                            break 'outer;
                        }
                    }
                }
            }
        }
        if result.is_err() {
            // Keep the originals so a retry can finish the restore.
            lm.fp8 = Some(state);
        }
        result
    }
}

#[cfg(test)]
mod tests;
