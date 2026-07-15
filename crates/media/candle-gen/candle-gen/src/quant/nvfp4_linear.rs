//! `Nvfp4Linear` — the NVFP4 (FP4) linear layer (sc-11041, epic 11037).
//!
//! Serves a packed [`Nvfp4Tensor`] weight (E2M1 nibbles + UE4M3 block scales + FP32 per-tensor scale,
//! ~4.5 effective bits/weight — sc-11040) resident in VRAM, forwarding through the sc-11039 cuBLASLt
//! `matmul_nvfp4_staged` on consumer Blackwell `sm_120`. This is the candle-gen linear layer that a
//! provider crate swaps in for an NVFP4 compute tier, the FP4 twin of the `Fp8Linear`/`Int8Linear`
//! layers (`super::eight_bit_linear`, cuda-only).
//!
//! # Mixed-precision policy (spike sc-11038 / sc-7702)
//!
//! Two activation regimes, selected per layer by [`ActPrecision`]:
//!
//! - **W4A4** (the default): both operands FP4. This is the *only* regime that lights up the FP4
//!   tensor cores for the ~2× compute win (SC#1) — the FP4 MMA requires both operands in E2M1, so a
//!   bf16 activation cannot feed it. The weight is staged resident on-device as a packed `DevNvfp4`;
//!   the activation is packed per forward and the GEMM runs on the FP4 cores. **This is the SC#6
//!   packed-forward path — the weight never full-dequants to bf16; resident VRAM stays at the NVFP4
//!   footprint** (`Nvfp4Linear::resident_device_bytes`).
//! - **W4A16** (the per-layer override for the outlier class): FP4 weight × bf16 activation. W4A4
//!   collapses on layers with dense activation outliers (the sc-7702 mechanism: an outlier sharing a
//!   16-block crushes its co-located channels to E2M1 zero), so the spike keeps bf16 activation on the
//!   outlier class — text→DiT `caption_projection`, cross-attn K/V, first & last DiT blocks
//!   ([`ActPrecision::for_outlier_layer`]). There is no FP4-weight×bf16-activation tensor-core MMA, so
//!   W4A16 is realized by **dequantizing the FP4 weight to bf16 and running a dense bf16 matmul** — a
//!   storage-parity tier with **no FP4 compute win** (throughput class of the existing dequant-dense
//!   path). Reports [`Nvfp4Regime::DequantBf16`].
//!
//! # Capability gate + fallback (sc-11041 AC)
//!
//! W4A4 requires the `cuda` feature, a CUDA device, `sm_120`+ (`CublasLt::meets_nvfp4_floor`), and a
//! shape the cuBLASLt FP4 path accepts (padded K a multiple of `NVFP4_K_ALIGN`, N a multiple of 16).
//! When any of these do not hold — a `<sm_120` GPU, a CPU device, a non-cuda build, an ineligible
//! shape, or an explicit W4A16 override — the layer **transparently falls back** to the
//! [`Nvfp4Regime::DequantBf16`] dense path (no crash). The non-cuda build compiles this whole module
//! (the FP4 compute leg is cfg-gated); it only ever takes the fallback.
//!
//! # M (token-row) alignment (sc-11039 handoff)
//!
//! sc-11039's `check_nvfp4_alignment` guards K/N but **not** M. cuBLASLt's FP4 block-scaled path can
//! return `NOT_SUPPORTED` for an unaligned free dimension, so the W4A4 forward **pads the token rows**
//! up to a multiple of [`NVFP4_M_ALIGN`] (zero rows, which contribute nothing to any real output) and
//! slices the result back to the logical `M`. Arbitrary token counts therefore run without a runtime
//! `NOT_SUPPORTED`.

use super::nvfp4::Nvfp4Tensor;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

/// Token-row (M) alignment the W4A4 forward pads to before the cuBLASLt FP4 GEMM (sc-11041). The
/// contraction K and output N are aligned by the packer / sc-11039 (`NVFP4_K_ALIGN` = 32, N a multiple
/// of 16); M is the free dimension the merged primitive does **not** guard, so the layer pads it here
/// and slices the padded rows back off. 16 matches the general cuBLASLt 8-bit leading-dim alignment
/// (`check_alignment`) and is the multiple the live-GPU M-alignment test confirms cuBLASLt accepts.
pub const NVFP4_M_ALIGN: usize = 16;

// Only the W4A4 FP4 forward (cuda-only) needs M-row rounding; dead on a non-cuda build.
#[cfg(feature = "cuda")]
#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// The activation-precision regime for an [`Nvfp4Linear`] — the mixed-precision policy flag
/// (sc-11041). Default **W4A4**; **W4A16** is the per-layer override for the outlier class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActPrecision {
    /// W4A4 — both operands FP4. Lights up the FP4 tensor cores (~2×). The compute default.
    #[default]
    W4A4,
    /// W4A16 — FP4 weight × bf16 activation. Dequantizes the weight to bf16 (no FP4 compute win); the
    /// per-layer override for the outlier class where W4A4 collapses (sc-7702 / spike sc-11038).
    W4A16,
}

impl ActPrecision {
    /// The **default per-layer policy** (spike sc-11038): the outlier-carrying layer class runs
    /// **W4A16** (bf16 activation), everything else **W4A4**. The outlier class — where a dense
    /// activation outlier collapses W4A4 — is the text→DiT `caption_projection`, the cross-attention
    /// K/V projections, and the first & last DiT blocks. Matched by substring on the layer path so a
    /// provider can thread its own dotted key names; callers wanting a blanket regime pass
    /// [`ActPrecision::W4A4`] / [`ActPrecision::W4A16`] directly instead of consulting this.
    pub fn for_outlier_layer(layer_name: &str) -> Self {
        let l = layer_name.to_ascii_lowercase();
        let is_outlier = l.contains("caption_projection")
            || l.contains("caption_proj")
            // cross-attention K/V (attn2 = cross-attn in the diffusers DiT naming); guard K/V only.
            || (l.contains("attn2") && (l.contains(".to_k") || l.contains(".to_v")))
            || (l.contains("cross") && (l.contains("_k") || l.contains("_v") || l.contains(".k") || l.contains(".v")))
            // first & last DiT blocks (blocks.0 / block_0 and an explicit last-block marker).
            || l.contains("blocks.0.")
            || l.contains("block_0.")
            || l.contains("last_block")
            || l.contains("final_block");
        if is_outlier {
            ActPrecision::W4A16
        } else {
            ActPrecision::W4A4
        }
    }

    /// Partition a set of layer names into the W4A4 (benign, FP4 activation) and W4A16 (outlier class,
    /// bf16 activation) classes per [`Self::for_outlier_layer`] — the **explicit, testable** form of
    /// the spike sc-11038 mixed-precision policy (sc-11044 AC 2). Returns each name paired with its
    /// regime plus the partition counts, so a loader/report can show exactly which projections light
    /// the FP4 cores and which stay bf16-activation.
    pub fn partition_layers<'a, I>(names: I) -> Nvfp4Partition
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut assignments = Vec::new();
        let (mut n_w4a4, mut n_w4a16) = (0usize, 0usize);
        for name in names {
            let p = Self::for_outlier_layer(name);
            match p {
                ActPrecision::W4A4 => n_w4a4 += 1,
                ActPrecision::W4A16 => n_w4a16 += 1,
            }
            assignments.push((name.to_string(), p));
        }
        Nvfp4Partition {
            assignments,
            n_w4a4,
            n_w4a16,
        }
    }
}

/// The result of [`ActPrecision::partition_layers`] — the mixed-precision policy applied to a concrete
/// set of layers (sc-11044). `n_w4a4` benign projections run the FP4 W4A4 compute path; `n_w4a16`
/// outlier-class projections keep bf16 activation (W4A16).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nvfp4Partition {
    /// Each layer name paired with the regime the policy assigns it.
    pub assignments: Vec<(String, ActPrecision)>,
    /// Count assigned W4A4 (benign, FP4 activation).
    pub n_w4a4: usize,
    /// Count assigned W4A16 (outlier class, bf16 activation).
    pub n_w4a16: usize,
}

impl Nvfp4Partition {
    /// Fraction of layers on the FP4 W4A4 compute path (the compute-bulk the ~2× win rides).
    pub fn w4a4_fraction(&self) -> f64 {
        let total = self.n_w4a4 + self.n_w4a16;
        if total == 0 {
            0.0
        } else {
            self.n_w4a4 as f64 / total as f64
        }
    }
}

/// Which compute path an [`Nvfp4Linear`] actually runs — surfaced so a bench/report can state whether
/// the FP4 cores are lit (sc-11041 AC: the report must name the A-precision regime).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Regime {
    /// W4A4 FP4 tensor-core GEMM on `sm_120` via cuBLASLt (`matmul_nvfp4_staged`) — the real FP4
    /// compute win, weight resident at the NVFP4 footprint (the SC#6 packed-forward path).
    Fp4W4A4,
    /// Weight dequantized to bf16, dense bf16 matmul (full-precision activation) — the W4A16 outlier
    /// override **or** the `<sm_120` / CPU / non-cuda / ineligible-shape capability fallback. **No FP4
    /// compute** (storage/parity tier).
    DequantBf16,
}

/// The FP4 compute leg — a resident on-device NVFP4 weight + a shared cuBLASLt handle. Cuda-only.
#[cfg(feature = "cuda")]
struct Fp4Resident {
    lt: std::sync::Arc<super::cublaslt::CublasLt>,
    w_staged: super::cublaslt::DevNvfp4,
}

/// An NVFP4 linear projection `y = x·Wᵀ (+ b)` over a packed [`Nvfp4Tensor`] weight (sc-11041).
///
/// Built from packed weights ([`Self::from_packed`]) or a dense weight ([`Self::from_dense`]). On
/// `sm_120` with the default W4A4 regime it serves the weight resident-packed and runs the cuBLASLt FP4
/// GEMM (the SC#6 packed-forward path); otherwise it transparently falls back to a dequant→bf16 dense
/// matmul (see the [module docs](self)). The packed [`Nvfp4Tensor`] is always retained (the source of
/// truth for the footprint accounting + a re-stage), so [`Self::nvfp4_footprint_bytes`] /
/// [`Self::resident_device_bytes`] report the NVFP4 footprint regardless of regime.
pub struct Nvfp4Linear {
    /// The packed NVFP4 weight (host container — the canonical ~4.5-bit representation). Retained in
    /// every regime for footprint accounting and to (re)stage the FP4 device weight.
    weight: Nvfp4Tensor,
    bias: Option<Tensor>,
    device: Device,
    act: ActPrecision,
    regime: Nvfp4Regime,
    /// The resident bf16 dense weight `[out, in]` for the [`Nvfp4Regime::DequantBf16`] path (dequantized
    /// once at construction). `None` in the FP4 regime.
    dequant_w: Option<Tensor>,
    /// The resident FP4 compute leg (staged device weight + handle) for [`Nvfp4Regime::Fp4W4A4`].
    #[cfg(feature = "cuda")]
    fp4: Option<Fp4Resident>,
}

impl Nvfp4Linear {
    /// Build from an already-packed [`Nvfp4Tensor`] weight, choosing the regime by the capability gate +
    /// the requested [`ActPrecision`]. `bias` is kept full-precision (added back in the activation dtype).
    ///
    /// W4A4 on a `sm_120` CUDA device with an eligible shape → the FP4 packed-forward path; anything
    /// else → the transparent dequant→bf16 fallback. Never panics on an ineligible device/shape.
    pub fn from_packed(
        weight: Nvfp4Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if act == ActPrecision::W4A4 {
                if let Some(built) = Self::try_build_fp4(&weight, &bias, device)? {
                    return Ok(built);
                }
            }
        }
        // W4A16 override, or the <sm_120 / CPU / non-cuda / ineligible-shape fallback.
        Self::new_dequant(weight, bias, device, act)
    }

    /// Pack a dense `[out, in]` weight (bf16/f32, any device) to NVFP4 and build (see [`Self::from_packed`]).
    pub fn from_dense(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        let packed = Nvfp4Tensor::pack(weight)?;
        Self::from_packed(packed, bias, device, act)
    }

    /// Build applying the **default per-layer policy** ([`ActPrecision::for_outlier_layer`]): the
    /// outlier class runs W4A16, everything else W4A4. The convenience entry a loader calls with the
    /// projection's dotted key name to get the mixed-precision default without classifying by hand.
    pub fn from_dense_for_layer(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        layer_name: &str,
    ) -> Result<Self> {
        Self::from_dense(weight, bias, device, ActPrecision::for_outlier_layer(layer_name))
    }

    /// Attempt to build the resident FP4 (W4A4) compute leg. `Ok(None)` when the device is not
    /// `sm_120`+, not CUDA, or the shape is ineligible for the cuBLASLt FP4 path (→ caller falls back).
    /// A hard cuBLASLt/allocation failure also degrades to `Ok(None)` (transparent fallback) with a note.
    #[cfg(feature = "cuda")]
    fn try_build_fp4(
        weight: &Nvfp4Tensor,
        bias: &Option<Tensor>,
        device: &Device,
    ) -> Result<Option<Self>> {
        use super::cublaslt::{CublasLt, NVFP4_K_ALIGN};
        if !matches!(device, Device::Cuda(_)) {
            return Ok(None);
        }
        // Shape gate: the cuBLASLt FP4 path needs padded-K a multiple of NVFP4_K_ALIGN and N a
        // multiple of 16 (sc-11039). An ineligible shape falls back rather than erroring at runtime.
        if !weight.cols_padded.is_multiple_of(NVFP4_K_ALIGN) || !weight.rows.is_multiple_of(16) {
            eprintln!(
                "[sc-11041] Nvfp4Linear: shape [{}, {} (K_pad {})] ineligible for the cuBLASLt FP4 \
                 path (need K_pad % {NVFP4_K_ALIGN} == 0 and N % 16 == 0); using bf16 fallback",
                weight.rows, weight.cols, weight.cols_padded
            );
            return Ok(None);
        }
        let lt = match CublasLt::new(device) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[sc-11041] Nvfp4Linear: cuBLASLt handle unavailable ({e}); bf16 fallback");
                return Ok(None);
            }
        };
        match lt.meets_nvfp4_floor() {
            Ok(true) => {}
            _ => return Ok(None), // <sm_120 → fallback
        }
        let w_staged = match lt.stage_nvfp4(weight) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[sc-11041] Nvfp4Linear: FP4 weight stage failed ({e}); bf16 fallback");
                return Ok(None);
            }
        };
        Ok(Some(Self {
            weight: weight.clone(),
            bias: bias.clone(),
            device: device.clone(),
            act: ActPrecision::W4A4,
            regime: Nvfp4Regime::Fp4W4A4,
            dequant_w: None,
            fp4: Some(Fp4Resident {
                lt: std::sync::Arc::new(lt),
                w_staged,
            }),
        }))
    }

    /// Build the [`Nvfp4Regime::DequantBf16`] path: dequantize the packed weight to a resident bf16
    /// `[out, in]` tensor on `device` **once**. The W4A16 override and the capability fallback share
    /// this (they differ only in *why* they are here — recorded in `act`).
    fn new_dequant(
        weight: Nvfp4Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        // `Nvfp4Tensor::dequantize` returns a CPU f32 [rows, cols]; store it resident as bf16 on device.
        let w = weight
            .dequantize()?
            .to_dtype(DType::BF16)?
            .to_device(device)?;
        Ok(Self {
            weight,
            bias,
            device: device.clone(),
            act,
            regime: Nvfp4Regime::DequantBf16,
            dequant_w: Some(w),
            #[cfg(feature = "cuda")]
            fp4: None,
        })
    }

    /// `y = x·Wᵀ (+ b)`. Routes to the FP4 W4A4 GEMM ([`Nvfp4Regime::Fp4W4A4`]) or the dequant→bf16
    /// dense matmul ([`Nvfp4Regime::DequantBf16`]). Accepts a rank-≥1 activation `[..., in]`; the output
    /// is `[..., out]` cast back to the input dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.regime {
            #[cfg(feature = "cuda")]
            Nvfp4Regime::Fp4W4A4 => self.forward_fp4(x),
            // The FP4 regime is only ever constructed under the `cuda` feature (`try_build_fp4`), so on
            // a non-cuda build this arm is unreachable — but the match must still compile.
            #[cfg(not(feature = "cuda"))]
            Nvfp4Regime::Fp4W4A4 => {
                unreachable!("Nvfp4Regime::Fp4W4A4 is only constructed with the cuda feature")
            }
            Nvfp4Regime::DequantBf16 => self.forward_dequant(x),
        }
    }

    /// The dequant→bf16 dense forward (W4A16 / fallback). The resident bf16 weight is cast to the
    /// activation dtype and run through a plain `candle_nn::Linear` (which broadcast-matmuls rank-≥2
    /// inputs), keeping the activation full-precision.
    fn forward_dequant(&self, x: &Tensor) -> Result<Tensor> {
        let w = self
            .dequant_w
            .as_ref()
            .expect("DequantBf16 regime holds a resident weight")
            .to_dtype(x.dtype())?;
        let bias = match &self.bias {
            Some(b) => Some(b.to_dtype(x.dtype())?),
            None => None,
        };
        Linear::new(w, bias).forward(x)
    }

    /// The W4A4 FP4 forward (cuda-only): flatten leading dims to `[M, K]`, pad M to [`NVFP4_M_ALIGN`],
    /// pack the activation to NVFP4, run the resident-weight cuBLASLt FP4 GEMM, slice the padded rows
    /// off, reshape back, add bias, cast to the input dtype. The weight never dequantizes — this is the
    /// SC#6 packed-forward path.
    #[cfg(feature = "cuda")]
    fn forward_fp4(&self, x: &Tensor) -> Result<Tensor> {
        let fp4 = self.fp4.as_ref().expect("Fp4W4A4 regime holds a staged weight");
        let dims = x.dims().to_vec();
        let k = *dims.last().expect("linear input has a last dim");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;

        // M-alignment (sc-11039 handoff): pad the token rows up to a multiple of NVFP4_M_ALIGN with
        // zero rows (they add nothing to any real output) so cuBLASLt does not return NOT_SUPPORTED,
        // then slice the padding back off the result.
        let m_pad = round_up(m.max(1), NVFP4_M_ALIGN);
        let x_pad = if m_pad != m {
            x2.pad_with_zeros(0, 0, m_pad - m)?
        } else {
            x2
        };

        // W4A4 (sc-11044): quantize the activation to NVFP4 **on-device** — no CPU round-trip — and run
        // the FP4 GEMM against the resident weight. `cols_padded` matches the resident weight so the two
        // operands share the padded contraction width.
        let x_stg = fp4
            .lt
            .quantize_nvfp4_activation(&x_pad, fp4.w_staged.shape_padded().1)?;
        let y = fp4.lt.matmul_nvfp4_staged(&fp4.w_staged, &x_stg)?; // [m_pad, N] bf16
        let y = if m_pad != m { y.narrow(0, 0, m)? } else { y };

        let n = y.dim(1)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(n);
        let mut y = y.reshape(out_shape)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&b.to_dtype(y.dtype())?)?;
        }
        y.to_dtype(x.dtype())
    }

    /// [`Self::forward`] plus a **NaN/inf guard** (sc-11044 AC): asserts the output is finite and
    /// **fails loud** rather than letting a collapsed/garbage tensor propagate silently through the
    /// denoise. The W4A4 quantizer itself is NaN-free by construction (E2M1 saturates, divisors are
    /// clamped positive — spike sc-11038), so this guards the *accuracy* failure mode (signal collapse
    /// over steps surfacing as an inf/NaN downstream), not the quantizer. Uses a single sum-of-squares
    /// scalar reduction (a NaN/inf in any element makes the sum non-finite), so it is cheap enough to
    /// leave on around a denoise step. Callers wanting max throughput use [`Self::forward`] directly.
    pub fn forward_checked(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.forward(x)?;
        let energy = y.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !energy.is_finite() {
            candle_core::bail!(
                "Nvfp4Linear::forward_checked: non-finite output (NaN/inf) from the {:?} regime — \
                 W4A4 signal collapse or a bad activation; failing loud (sc-11044 NaN guard)",
                self.regime
            );
        }
        Ok(y)
    }

    /// The active compute path (whether the FP4 cores are lit) — for a bench/report.
    pub fn regime(&self) -> Nvfp4Regime {
        self.regime
    }

    /// The requested activation-precision regime (W4A4 default / W4A16 override).
    pub fn act_precision(&self) -> ActPrecision {
        self.act
    }

    /// True iff this layer runs the real FP4 tensor-core GEMM (W4A4 on sm_120) — as opposed to the
    /// dequant→bf16 storage/fallback path.
    pub fn lights_up_fp4(&self) -> bool {
        matches!(self.regime, Nvfp4Regime::Fp4W4A4)
    }

    /// The **NVFP4 footprint** in bytes of the weight — E2M1 nibble bytes + UE4M3 block-scale bytes
    /// (the packed host container's actual size). The resident device weight in the FP4 regime matches
    /// this (see [`Self::resident_device_bytes`]); the SC#6 gate asserts this is ≈ the ~4.5-bit
    /// footprint and far below the bf16 size.
    pub fn nvfp4_footprint_bytes(&self) -> usize {
        self.weight.packed.len() + self.weight.scales.len()
    }

    /// The bf16 footprint the weight *would* occupy dense (`rows * cols * 2`) — the baseline the SC#6
    /// resident-VRAM assertion compares against.
    pub fn bf16_footprint_bytes(&self) -> usize {
        self.weight.rows * self.weight.cols * 2
    }

    /// The logical `[out, in]` weight shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.weight.rows, self.weight.cols)
    }

    /// The device the layer's resident weight lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resident **device** bytes of the FP4 weight when in the [`Nvfp4Regime::Fp4W4A4`] path — the
    /// E2M1 nibble buffer + UE4M3 block-scale buffer actually uploaded to VRAM (the SC#6 packed-forward
    /// proof). `None` in the dequant→bf16 regime (there the resident device weight is the bf16 tensor).
    #[cfg(feature = "cuda")]
    pub fn resident_device_bytes(&self) -> Option<usize> {
        self.fp4.as_ref().map(|f| f.w_staged.resident_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in ~[-1, 1) — no `rand` dep.
    fn prng(seed: &mut u64) -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((*x - *y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
        (num / (den + 1e-30)).sqrt() as f32
    }

    /// On a CPU device (no cuda / not sm_120) `from_packed` always selects the dequant→bf16 fallback,
    /// never crashes, and forwards a coherent result matching the packed weight's dequant reference.
    #[test]
    fn cpu_selects_dequant_fallback_and_forwards() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut seed = 0x11FED00Du64;
        let w: Vec<f32> = (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.4).collect();
        let w_t = Tensor::from_vec(w, (out_dim, in_dim), &dev)?;

        let lin = Nvfp4Linear::from_dense(&w_t, None, &dev, ActPrecision::W4A4)?;
        // No sm_120 here → transparent fallback (the AC), not a crash.
        assert_eq!(lin.regime(), Nvfp4Regime::DequantBf16);
        assert!(!lin.lights_up_fp4());

        // Forward matches x · (dequant W)ᵀ within a small bf16-cast tolerance.
        let x = Tensor::from_vec(
            (0..4 * in_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (4, in_dim),
            &dev,
        )?;
        let packed = Nvfp4Tensor::pack(&w_t)?;
        let w_dq = packed.dequantize()?; // [out, in] f32
        let reference = x.matmul(&w_dq.t()?)?;
        let got = lin.forward(&x)?;
        assert_eq!(got.dims(), &[4, out_dim]);
        let rr = rel_rms(
            &got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            &reference.flatten_all()?.to_vec1::<f32>()?,
        );
        assert!(rr < 0.02, "dequant fallback forward rel-RMS {rr} vs its own dequant ref");
        Ok(())
    }

    /// The dequant fallback forward preserves the input dtype and handles a rank-3 activation (leading
    /// dims broadcast through `candle_nn::Linear`), with a bias added back.
    #[test]
    fn dequant_forward_rank3_and_bias_and_dtype() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (48usize, 64usize);
        let mut seed = 0xABCDEF01u64;
        let w = Tensor::from_vec(
            (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.3).collect::<Vec<_>>(),
            (out_dim, in_dim),
            &dev,
        )?;
        let b = Tensor::from_vec(
            (0..out_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (out_dim,),
            &dev,
        )?;
        let lin = Nvfp4Linear::from_dense(&w, Some(b.clone()), &dev, ActPrecision::W4A16)?;
        assert_eq!(lin.regime(), Nvfp4Regime::DequantBf16);
        assert_eq!(lin.act_precision(), ActPrecision::W4A16);

        let x = Tensor::from_vec(
            (0..2 * 5 * in_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (2, 5, in_dim),
            &dev,
        )?;
        let y = lin.forward(&x)?;
        assert_eq!(y.dims(), &[2, 5, out_dim]);
        assert_eq!(y.dtype(), x.dtype());
        Ok(())
    }

    /// The per-layer policy defaults the outlier class to W4A16 and everything else to W4A4.
    #[test]
    fn outlier_policy_classification() {
        for name in [
            "transformer.caption_projection.linear_1",
            "blocks.0.attn.to_q",
            "transformer_blocks.12.attn2.to_k", // cross-attn K
            "transformer_blocks.7.attn2.to_v",  // cross-attn V
            "dit.final_block.proj",
        ] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A16,
                "{name} must be classified outlier → W4A16"
            );
        }
        for name in [
            "transformer_blocks.7.attn1.to_q", // self-attn
            "transformer_blocks.7.ff.net.0.proj",
            "blocks.12.attn2.to_out.0", // cross-attn OUTPUT proj is not K/V → benign
        ] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A4,
                "{name} must be classified benign → W4A4"
            );
        }
    }

    /// The explicit mixed-precision partition (sc-11044 AC 2) over a Sana-1.6B-DiT-style layer list:
    /// self-attn + FF (the compute bulk) → W4A4; caption_projection, cross-attn K/V, first & last
    /// blocks → W4A16. The partition is countable and the benign class dominates.
    #[test]
    fn partition_layers_matches_spike_policy() {
        let layers = [
            // benign compute bulk (W4A4)
            "transformer_blocks.4.attn1.to_q",
            "transformer_blocks.4.attn1.to_k",
            "transformer_blocks.4.attn1.to_v",
            "transformer_blocks.4.attn1.to_out.0",
            "transformer_blocks.4.ff.net.0.proj",
            "transformer_blocks.4.ff.net.2",
            "transformer_blocks.4.attn2.to_q",   // cross-attn Q is benign (not K/V)
            "transformer_blocks.4.attn2.to_out.0", // cross-attn output proj is benign
            // outlier class (W4A16)
            "transformer.caption_projection.linear_1",
            "transformer_blocks.4.attn2.to_k", // cross-attn K
            "transformer_blocks.4.attn2.to_v", // cross-attn V
            "transformer_blocks.0.attn1.to_q", // first block
            "final_block.proj",                // last block
        ];
        let part = ActPrecision::partition_layers(layers);
        assert_eq!(part.n_w4a16, 5, "5 outlier-class projections must be W4A16");
        assert_eq!(part.n_w4a4, 8, "8 benign projections must be W4A4");
        assert!(
            part.w4a4_fraction() > 0.6,
            "the FP4 compute bulk must dominate (got {:.2})",
            part.w4a4_fraction()
        );
        // Spot-check a couple assignments.
        let by_name: std::collections::HashMap<_, _> = part
            .assignments
            .iter()
            .map(|(n, p)| (n.as_str(), *p))
            .collect();
        assert_eq!(by_name["transformer_blocks.4.attn1.to_q"], ActPrecision::W4A4);
        assert_eq!(by_name["transformer_blocks.4.attn2.to_k"], ActPrecision::W4A16);
        assert_eq!(by_name["transformer.caption_projection.linear_1"], ActPrecision::W4A16);
    }

    /// Footprint accounting: the NVFP4 footprint is far below the bf16 size (~4.5 vs 16 bits/weight).
    #[test]
    fn footprint_is_far_below_bf16() -> Result<()> {
        let dev = Device::Cpu;
        // A realistically large weight so the 128-row scale-atom padding overhead is negligible.
        let (out_dim, in_dim) = (1024usize, 4096usize);
        let mut seed = 0x5EED_1234u64;
        let w = Tensor::from_vec(
            (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.2).collect::<Vec<_>>(),
            (out_dim, in_dim),
            &dev,
        )?;
        let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4)?;
        let nvfp4 = lin.nvfp4_footprint_bytes();
        let bf16 = lin.bf16_footprint_bytes();
        // ~4.5 bits/weight vs 16 → ratio ~0.281; assert comfortably below a third of bf16.
        let ratio = nvfp4 as f64 / bf16 as f64;
        assert!(
            ratio < 0.32,
            "NVFP4 footprint {nvfp4} B / bf16 {bf16} B = {ratio:.3}, expected ≈0.28 (~4.5 bit)"
        );
        // And close to the 4.5-bit ideal (nibble 0.5 B/wt + scale 1 B/16 wt = 0.5625 B/wt).
        let per_wt = nvfp4 as f64 / (out_dim * in_dim) as f64;
        assert!(
            (per_wt - 0.5625).abs() < 0.02,
            "bytes/weight {per_wt:.4}, expected ≈0.5625 (4.5 bits)"
        );
        Ok(())
    }
}
