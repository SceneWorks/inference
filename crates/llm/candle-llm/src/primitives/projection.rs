//! A linear projection that is dense, group-wise quantized, Prism-packed or NVFP4.
//!
//! The decoders hold their attention/MLP projections behind this so quantize-on-load is a load-time
//! choice with no decoder changes: a dense `[out, in]` weight either stays dense (a
//! [`candle_nn::Linear`]), is quantized to Q4/Q8 ([`QuantizedLinear`]) via Candle's quant, or — on
//! a CUDA sm_120 device — is quantized to NVFP4 ([`Nvfp4Weight`], sc-24135) through the shared
//! `candle-quant-kernels` codec and served by the cuBLASLt W4A4 FP4 GEMM.
//!
//! An NVFP4 projection's forward is one of two implementations over the same resident weight —
//! the fused decode GEMV for ≤ 8 bf16 token rows, cuBLASLt W4A4 otherwise — chosen per call in
//! [`nvfp4_path`](super::nvfp4_path) with the path recorded (sc-24136).
//!
//! [`ProjectionFormat`] is the load-time selector a loader threads to [`Projection::load_as`];
//! [`ProjectionCensus`] is the load telemetry that says which kind each projection actually became.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use candle_quant_kernels::{nvfp4_shape_refusal, Nvfp4Context, Nvfp4Weight};

use crate::error::Result;
use crate::primitives::prism::PrismPackedWeight;
use crate::primitives::quant::QuantizedLinear;

/// Group-wise quantization spec, mapped to a Candle GGML dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// The target GGML block-quant dtype.
    pub dtype: GgmlDType,
    /// Source group size when the checkpoint stores an MLX affine packed triple.
    /// Dense load-time quantization ignores this value.
    group_size: usize,
}

impl QuantSpec {
    /// 4-bit (GGML Q4_K).
    pub fn q4() -> Self {
        Self {
            dtype: GgmlDType::Q4K,
            group_size: 64,
        }
    }

    /// 8-bit (GGML Q8_0).
    pub fn q8() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
            group_size: 64,
        }
    }

    /// Map a persisted `quantization.bits` value to a spec: `4 → Q4_K`, `8 → Q8_0`. Any other width
    /// is unrecognized (`None`) — the snapshot writer only ever emits 4 or 8.
    pub fn from_bits(bits: u32) -> Option<Self> {
        Self::from_bits_and_group_size(bits, 64)
    }

    /// Map a persisted affine quantization block to its bit width and source group size.
    pub fn from_bits_and_group_size(bits: u32, group_size: usize) -> Option<Self> {
        if group_size == 0 {
            return None;
        }
        match bits {
            4 => Some(Self {
                dtype: GgmlDType::Q4K,
                group_size,
            }),
            8 => Some(Self {
                dtype: GgmlDType::Q8_0,
                group_size,
            }),
            _ => None,
        }
    }

    /// The bit width to persist in a snapshot's `quantization` block (`Q4_K → 4`, `Q8_0 → 8`).
    pub fn bits(&self) -> u32 {
        match self.dtype {
            GgmlDType::Q8_0 => 8,
            _ => 4,
        }
    }

    /// Group size declared by an MLX affine packed source.
    pub fn group_size(&self) -> usize {
        self.group_size
    }
}

/// The load-time storage format for a decoder's large projections — the weight-format selector
/// (`bf16 | Q4 | Q8 | NVFP4`; dense bf16 is `None` at the call sites).
///
/// The NVFP4 arm carries the shared per-device cuBLASLt context, which is why this is not `Copy`:
/// constructing it ([`ProjectionFormat::nvfp4`]) *is* the capability check, so a loader holding one
/// already knows the device can serve every NVFP4 projection it builds.
#[derive(Clone)]
pub enum ProjectionFormat {
    /// GGML block quantization (Q4_K / Q8_0) via Candle's quant.
    Ggml(QuantSpec),
    /// NVFP4, quantized on-device at load and served by the cuBLASLt W4A4 FP4 GEMM.
    Nvfp4(Nvfp4Context),
}

impl ProjectionFormat {
    /// The NVFP4 format for `device`, or the typed [`Nvfp4Refused`](crate::error::Error::Nvfp4Refused) naming
    /// why the device cannot serve it (not CUDA, below sm_120, no fused quantizer). This is the
    /// whole capability floor, so a loader calls it before reading any weights.
    pub fn nvfp4(device: &Device) -> Result<Self> {
        Ok(Self::Nvfp4(Nvfp4Context::require(device)?))
    }

    /// [`Self::nvfp4`] with the compute-capability probe injected
    /// ([`Nvfp4Context::require_with`]), so the gate can be exercised with a mocked capability.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_with_cap_probe(
        device: &Device,
        cap_probe: impl FnOnce(&candle_quant_kernels::CublasLt) -> candle_core::Result<(i32, i32)>,
    ) -> Result<Self> {
        Ok(Self::Nvfp4(Nvfp4Context::require_with(device, cap_probe)?))
    }

    /// The GGML spec, when this is a GGML format.
    pub fn ggml(&self) -> Option<QuantSpec> {
        match self {
            Self::Ggml(q) => Some(*q),
            Self::Nvfp4(_) => None,
        }
    }

    /// Whether this is NVFP4.
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4(_))
    }
}

impl From<QuantSpec> for ProjectionFormat {
    fn from(q: QuantSpec) -> Self {
        Self::Ggml(q)
    }
}

impl std::fmt::Debug for ProjectionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ggml(q) => f.debug_tuple("Ggml").field(q).finish(),
            Self::Nvfp4(_) => f.write_str("Nvfp4"),
        }
    }
}

/// A linear projection weight, dense or quantized.
pub enum Projection {
    /// A dense `[out, in]` weight wrapped in a Candle linear.
    Dense(Linear),
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
    /// A compact Prism/Bonsai affine-2 or native ternary weight.
    Prism(std::sync::Arc<PrismPackedWeight>),
    /// An NVFP4 weight quantized at load (sc-24135), resident as packed E2M1 + UE4M3 scales. Boxed:
    /// its device handles would otherwise grow every projection-holding enum in the decoders.
    Nvfp4(Box<Nvfp4Weight>),
}

/// Which representation a loaded [`Projection`] actually holds — the load telemetry's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    /// Dense float weight.
    Dense,
    /// GGML block-quantized (Q4_K / Q8_0 / GGUF block types).
    Ggml,
    /// Prism/Bonsai packed.
    Prism,
    /// NVFP4.
    Nvfp4,
}

impl ProjectionKind {
    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Ggml => "ggml",
            Self::Prism => "prism",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// Count, logical parameters and resident bytes of the projections of one [`ProjectionKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectionTally {
    /// Projections of this kind.
    pub count: u64,
    /// Logical weight elements (`Σ out · in`).
    pub params: u64,
    /// Resident weight (+ bias) bytes of the measured projections.
    pub resident_bytes: u64,
    /// Projections whose resident bytes could not be measured (Prism); excluded from
    /// `resident_bytes` rather than counted as zero.
    pub unmeasured: u64,
}

impl ProjectionTally {
    /// Resident bits per logical parameter. `None` when the tally is empty or any of its
    /// projections is unmeasured (a partial sum would understate the footprint).
    pub fn bits_per_param(&self) -> Option<f64> {
        (self.params > 0 && self.unmeasured == 0)
            .then(|| self.resident_bytes as f64 * 8.0 / self.params as f64)
    }

    fn add(&mut self, other: &Self) {
        self.count += other.count;
        self.params += other.params;
        self.resident_bytes += other.resident_bytes;
        self.unmeasured += other.unmeasured;
    }
}

/// Which projection kinds a loaded model holds, with per-kind parameter and residency totals — the
/// load telemetry that makes the served representation visible (epic sc-24128 E2): an NVFP4 request
/// reports every requested projection under `nvfp4`, never a dense projection under an NVFP4 label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectionCensus {
    /// Dense float projections.
    pub dense: ProjectionTally,
    /// GGML block-quantized projections.
    pub ggml: ProjectionTally,
    /// Prism/Bonsai packed projections.
    pub prism: ProjectionTally,
    /// NVFP4 projections.
    pub nvfp4: ProjectionTally,
}

impl ProjectionCensus {
    /// Add one projection.
    pub fn record(&mut self, p: &Projection) {
        let tally = match p.kind() {
            ProjectionKind::Dense => &mut self.dense,
            ProjectionKind::Ggml => &mut self.ggml,
            ProjectionKind::Prism => &mut self.prism,
            ProjectionKind::Nvfp4 => &mut self.nvfp4,
        };
        tally.count += 1;
        tally.params += p.params();
        match p.resident_bytes() {
            Some(bytes) => tally.resident_bytes += bytes,
            None => tally.unmeasured += 1,
        }
    }

    /// The tally for one kind.
    pub fn tally(&self, kind: ProjectionKind) -> &ProjectionTally {
        match kind {
            ProjectionKind::Dense => &self.dense,
            ProjectionKind::Ggml => &self.ggml,
            ProjectionKind::Prism => &self.prism,
            ProjectionKind::Nvfp4 => &self.nvfp4,
        }
    }

    /// All projections together.
    pub fn total(&self) -> ProjectionTally {
        let mut total = ProjectionTally::default();
        for t in [&self.dense, &self.ggml, &self.prism, &self.nvfp4] {
            total.add(t);
        }
        total
    }

    /// Merge another census into this one.
    pub fn merge(&mut self, other: &Self) {
        self.dense.add(&other.dense);
        self.ggml.add(&other.ggml);
        self.prism.add(&other.prism);
        self.nvfp4.add(&other.nvfp4);
    }
}

/// A loaded model's whole resident weight set: its [`ProjectionCensus`] plus every other weight
/// tensor (embeddings, norms, recurrent parameters) under `other`. The load telemetry the
/// NVFP4 evidence reads its bits/param from (sc-24135).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightCensus {
    /// The projections, by kind.
    pub projections: ProjectionCensus,
    /// Every non-projection weight tensor (`count` is tensors, not projections).
    pub other: ProjectionTally,
}

impl WeightCensus {
    /// Add a non-projection dense weight tensor.
    pub fn record_tensor(&mut self, t: &Tensor) {
        self.other.count += 1;
        self.other.params += t.elem_count() as u64;
        self.other.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
    }

    /// Add a non-projection weight whose storage is not measured here (a Prism embedding).
    pub fn record_unmeasured(&mut self, params: u64) {
        self.other.count += 1;
        self.other.params += params;
        self.other.unmeasured += 1;
    }

    /// Everything together.
    pub fn total(&self) -> ProjectionTally {
        let mut total = self.projections.total();
        total.add(&self.other);
        total
    }

    /// Merge another census into this one.
    pub fn merge(&mut self, other: &Self) {
        self.projections.merge(&other.projections);
        self.other.add(&other.other);
    }
}

impl Projection {
    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Tensor, quant: Option<QuantSpec>) -> Result<Self> {
        Self::load_with_bias(weight, None, quant)
    }

    /// Load from a dense `[out, in]` weight plus an optional `[out]` bias (Qwen2 attention carries
    /// q/k/v bias), quantizing the weight if `quant` is set. The bias is always applied dense.
    pub fn load_with_bias(
        weight: Tensor,
        bias: Option<Tensor>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense(Linear::new(weight, bias))),
            Some(q) => Ok(Projection::Quantized(QuantizedLinear::quantize(
                &weight, q.dtype, bias,
            )?)),
        }
    }

    /// Load from a dense `[out, in]` weight plus optional bias in the requested `format` (`None`
    /// keeps it dense). The one entry point a loader needs for every format, NVFP4 included.
    ///
    /// NVFP4 quantizes on the weight's device (it must be `format`'s device) and refuses an
    /// ineligible shape with the typed [`Nvfp4Refused`](crate::error::Error::Nvfp4Refused) — it never keeps a
    /// projection dense under an NVFP4 request.
    pub fn load_as(
        weight: Tensor,
        bias: Option<Tensor>,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        match format {
            None => Self::load_with_bias(weight, bias, None),
            Some(ProjectionFormat::Ggml(q)) => Self::load_with_bias(weight, bias, Some(*q)),
            Some(ProjectionFormat::Nvfp4(ctx)) => {
                let (rows, cols) = weight.dims2()?;
                nvfp4_shape_refusal(rows, cols)?;
                Ok(Self::Nvfp4(Box::new(Nvfp4Weight::quantize(
                    &weight, bias, ctx,
                )?)))
            }
        }
    }

    /// [`Self::load_as`] with the llama-family shape policy (sc-24140): under an NVFP4 `format`, a
    /// weight whose shape the FP4 GEMM cannot serve ([`nvfp4_shape_refusal`] — `N % 16 != 0`, or
    /// too large for the fused quantizer) stays **dense** instead of refusing the load, so a
    /// checkpoint with one odd projection (a vocabulary that is not a multiple of 16, say) still
    /// loads with every eligible projection NVFP4. The kept projection reports
    /// [`ProjectionKind::Dense`], so the load census shows it under `dense` — never under an NVFP4
    /// label. Every other format (and every eligible shape) behaves exactly as `load_as`.
    pub fn load_eligible(
        weight: Tensor,
        bias: Option<Tensor>,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        if let Some(ProjectionFormat::Nvfp4(_)) = format {
            let (rows, cols) = weight.dims2()?;
            if nvfp4_shape_refusal(rows, cols).is_err() {
                return Self::load_with_bias(weight, bias, None);
            }
        }
        Self::load_as(weight, bias, format)
    }

    /// Wrap a resident compact Prism weight. Prism projections do not carry an additive bias.
    pub fn load_prism(weight: std::sync::Arc<PrismPackedWeight>) -> Self {
        Self::Prism(weight)
    }

    /// Wrap a pre-quantized GGUF matrix without expanding it. Plain F16/BF16/F32 GGUF matrices are
    /// dequantized by the caller and use [`Self::load_with_bias`]; block-quantized matrices stay in
    /// this representation for their full resident lifetime.
    pub fn load_qtensor(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self::Quantized(QuantizedLinear::from_qtensor(
            weight, bias,
        )?))
    }

    /// Load a pre-quantized MLX affine Q8 triple without interpreting its shortened U32 code
    /// matrix as a dense projection. The source is converted once to the resident Q8_0 form used by
    /// the existing quantized forward.
    pub fn load_mlx_affine_q8(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        quant: QuantSpec,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if quant.bits() != 8 {
            return Err(crate::error::Error::Config(format!(
                "MLX affine projection requires Q8, got Q{}",
                quant.bits()
            )));
        }
        Ok(Projection::Quantized(QuantizedLinear::from_mlx_affine_q8(
            weight,
            scales,
            biases,
            bias,
            quant.group_size(),
            device,
        )?))
    }

    /// `x @ weightᵀ`.
    ///
    /// An NVFP4 projection dispatches between the fused decode GEMV (≤ 8 bf16 rows) and the
    /// cuBLASLt W4A4 GEMM, recording which ran ([`nvfp4_path`](super::nvfp4_path), sc-24136).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Projection::Dense(l) => Ok(l.forward(x)?),
            Projection::Quantized(q) => q.forward(x),
            Projection::Prism(weight) => weight.forward(x),
            Projection::Nvfp4(weight) => super::nvfp4_path::forward(weight, x),
        }
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Projection::Dense(_))
    }

    /// Which representation this projection holds.
    pub fn kind(&self) -> ProjectionKind {
        match self {
            Projection::Dense(_) => ProjectionKind::Dense,
            Projection::Quantized(_) => ProjectionKind::Ggml,
            Projection::Prism(_) => ProjectionKind::Prism,
            Projection::Nvfp4(_) => ProjectionKind::Nvfp4,
        }
    }

    /// Logical weight elements (`out · in`).
    pub fn params(&self) -> u64 {
        (match self {
            Projection::Dense(l) => l.weight().elem_count(),
            Projection::Quantized(q) => q.weight_elems(),
            Projection::Prism(w) => w.rows() * w.input_width(),
            Projection::Nvfp4(w) => {
                let (rows, cols) = w.shape();
                rows * cols
            }
        }) as u64
    }

    /// Resident weight (+ bias) bytes, or `None` for a representation whose storage is not
    /// measured here (Prism).
    pub fn resident_bytes(&self) -> Option<u64> {
        let dense = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        match self {
            Projection::Dense(l) => Some(dense(l.weight()) + l.bias().map_or(0, dense)),
            Projection::Quantized(q) => Some(q.resident_bytes() as u64),
            Projection::Prism(_) => None,
            Projection::Nvfp4(w) => Some(w.resident_bytes() as u64),
        }
    }
}

/// A layer's key **and** value projections, which may be one shared weight.
///
/// Gemma 4's `attention_k_eq_v` makes the `full_attention` layers reuse the key projection's output
/// as the value projection's — there is no `v_proj` weight in the checkpoint at all. That is a
/// projection-layer fact, not a decoder one: the value path still gets its own (scale-free) per-head
/// norm afterwards, so K and V remain different tensors; only the matmul and the weight are shared.
///
/// Holding it here keeps the saving real. A decoder that "supported" `k_eq_v` by running the same
/// weight through two projections would produce identical numbers while paying twice the matmul and
/// twice the (quantized) weight footprint — the whole point of the flag.
pub struct KvProjection {
    k: Projection,
    /// `None` means `attention_k_eq_v`: the value heads come from `k`'s output.
    v: Option<Projection>,
}

impl KvProjection {
    /// Independent key and value projections (every architecture before Gemma 4, and Gemma 4's
    /// `sliding_attention` layers).
    pub fn separate(k: Projection, v: Projection) -> Self {
        Self { k, v: Some(v) }
    }

    /// One shared projection feeding both key and value heads (`attention_k_eq_v: true`).
    pub fn shared(k: Projection) -> Self {
        Self { k, v: None }
    }

    /// Whether K and V share a projection.
    pub fn k_eq_v(&self) -> bool {
        self.v.is_none()
    }

    /// The key projection.
    pub fn key(&self) -> &Projection {
        &self.k
    }

    /// The value projection, or `None` when it is shared with the key's.
    pub fn value(&self) -> Option<&Projection> {
        self.v.as_ref()
    }

    /// Project `x` into the **raw** key and value tensors, before any per-head norm or RoPE.
    ///
    /// When shared, the key projection runs **once** and both returned handles reference that one
    /// result (Candle tensors are refcounted, so this is a handle clone, not a copy).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.k.forward(x)?;
        match &self.v {
            Some(v) => {
                let v = v.forward(x)?;
                Ok((k, v))
            }
            None => Ok((k.clone(), k)),
        }
    }

    /// Whether either half is quantized.
    pub fn is_quantized(&self) -> bool {
        self.k.is_quantized() || self.v.as_ref().is_some_and(Projection::is_quantized)
    }

    /// Add both halves (one when shared) to a load census.
    pub fn record(&self, census: &mut ProjectionCensus) {
        census.record(&self.k);
        if let Some(v) = &self.v {
            census.record(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use candle_core::DType;
    use candle_core::Device;

    fn ramp(rows: usize, cols: usize, seed: u64, device: &Device) -> Tensor {
        let mut x = seed;
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Tensor::from_vec(data, (rows, cols), device).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
        let num: f64 = got
            .iter()
            .zip(want)
            .map(|(g, w)| ((g - w) as f64).powi(2))
            .sum();
        let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
        (num / den.max(1e-30)).sqrt() as f32
    }

    /// sc-24135 AC2 at the primitive: an NVFP4 format for a CPU device is the typed refusal.
    #[test]
    fn nvfp4_format_on_cpu_is_the_typed_refusal() {
        match ProjectionFormat::nvfp4(&Device::Cpu) {
            Err(crate::Error::Nvfp4Refused(refusal)) => {
                assert!(matches!(
                    refusal,
                    candle_quant_kernels::Nvfp4Refusal::NotCudaDevice { .. }
                ));
                assert_eq!(refusal.capability(), "nvfp4");
            }
            Err(other) => panic!("expected Nvfp4Refused, got {other}"),
            Ok(_) => panic!("CPU cannot serve NVFP4"),
        }
    }

    #[test]
    fn census_counts_kinds_params_and_resident_bytes() {
        let dev = Device::Cpu;
        let dense = Projection::load(ramp(64, 32, 1, &dev), None).unwrap();
        let q8 = Projection::load(ramp(64, 64, 2, &dev), Some(QuantSpec::q8())).unwrap();
        assert_eq!(dense.kind(), ProjectionKind::Dense);
        assert_eq!(q8.kind(), ProjectionKind::Ggml);
        let mut census = ProjectionCensus::default();
        census.record(&dense);
        census.record(&q8);
        assert_eq!(census.dense.count, 1);
        assert_eq!(census.dense.params, 64 * 32);
        assert_eq!(census.dense.resident_bytes, 64 * 32 * 4, "f32 on CPU");
        assert_eq!(census.dense.bits_per_param(), Some(32.0));
        assert_eq!(census.ggml.count, 1);
        assert_eq!(census.ggml.params, 64 * 64);
        // Q8_0: 34 bytes per 32-element block = 8.5 bits/param.
        assert_eq!(census.ggml.bits_per_param(), Some(8.5));
        assert_eq!(census.nvfp4, ProjectionTally::default());
        assert_eq!(census.total().count, 2);
    }

    /// sc-24140: outside NVFP4, `load_eligible` is exactly `load_as` — whatever the shape.
    #[test]
    fn load_eligible_is_load_as_for_dense_and_ggml() {
        let dev = Device::Cpu;
        let q8 = ProjectionFormat::from(QuantSpec::q8());
        for (rows, cols) in [(64usize, 64usize), (50, 64)] {
            for format in [None, Some(&q8)] {
                let a = Projection::load_eligible(ramp(rows, cols, 3, &dev), None, format).unwrap();
                let b = Projection::load_as(ramp(rows, cols, 3, &dev), None, format).unwrap();
                assert_eq!(a.kind(), b.kind(), "{rows}x{cols} {format:?}");
                assert_eq!(a.resident_bytes(), b.resident_bytes());
            }
        }
    }

    /// A CUDA device that meets the NVFP4 floor, or `None` (the GPU tests then skip loudly).
    #[cfg(feature = "cuda")]
    fn nvfp4_format() -> Option<(Device, ProjectionFormat)> {
        let device = crate::device::new_cuda_for_test().ok()?;
        let format = ProjectionFormat::nvfp4(&device).ok()?;
        Some((device, format))
    }

    /// sc-24135: `Projection::Nvfp4` forward against the dequantize-then-matmul reference over the
    /// *same* quantized operands (the NVFP4 weight read back from the device, the activation packed
    /// by the CPU codec), across the decode (M=1), odd and multi-tile M, a K that needs padding to
    /// the GEMM's 32-alignment, rank-3 input and a bias.
    ///
    /// Declared tolerances: rel-RMS ≤ 1e-2 against that reference (the residual is the bf16 output
    /// rounding and f32 accumulation order), and ≤ 0.2 against the exact dense product (the W4A4
    /// quantization error itself on uniform random operands).
    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_forward_matches_the_dequantize_then_matmul_reference() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            return;
        };
        let (n, k) = (96usize, 80usize); // K=80 pads to 96 at quantization
        let w = ramp(n, k, 0x5c24_1350, &device)
            .to_dtype(DType::BF16)
            .unwrap();
        let bias = ramp(1, n, 0xb1a5, &device)
            .reshape(n)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let p = Projection::load_as(w.clone(), Some(bias.clone()), Some(&format)).unwrap();
        assert_eq!(p.kind(), ProjectionKind::Nvfp4);
        assert!(p.is_quantized());
        let Projection::Nvfp4(nv) = &p else {
            unreachable!()
        };
        let wq = nv.to_host().unwrap(); // [n, 96] incl. zero K padding
        let w_deq = wq
            .dequantize()
            .unwrap()
            .narrow(1, 0, k)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let bias_f = bias
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let w_exact = w
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        let _gemv_on = crate::primitives::nvfp4_path::nvfp4_gemv_policy_guard(Some(true));
        let flat_bf16 = |y: Tensor| {
            y.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for m in [1usize, 5, 17, 130] {
            let x = ramp(m, k, 0xacdc + m as u64, &device)
                .to_dtype(DType::BF16)
                .unwrap();
            // The cuBLASLt W4A4 path at every M (the projection's dispatch sends M <= 8 to the
            // fused GEMV, sc-24136; that path is checked below against its own reference).
            let y = nv
                .forward(&x.unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
            assert_eq!(y.dims(), &[m, n]);
            assert_eq!(y.dtype(), DType::BF16);
            let got = flat_bf16(y);
            // Through the projection: the GEMV for M <= 8 (activation NOT quantized, so its
            // reference is the unquantized x against the same dequantized weight), cuBLASLt above.
            let y_proj = p
                .forward(&x.unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
            assert_eq!(y_proj.dims(), &[m, n]);

            let x_host = x
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let xq = candle_quant_kernels::Nvfp4Tensor::pack(&x_host).unwrap();
            let x_deq = xq.dequantize().unwrap();
            let reference = x_deq
                .matmul(&w_deq.t().unwrap())
                .unwrap()
                .broadcast_add(&bias_f)
                .unwrap();
            let exact = x_host
                .matmul(&w_exact.t().unwrap())
                .unwrap()
                .broadcast_add(&bias_f)
                .unwrap();
            let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vs_ref = rel_rms(&got, &flat(reference));
            let vs_exact = rel_rms(&got, &flat(exact.clone()));
            assert!(
                vs_ref <= 1e-2,
                "M={m}: rel-RMS vs quantized reference {vs_ref}"
            );
            assert!(vs_exact <= 0.2, "M={m}: rel-RMS vs dense {vs_exact}");

            let got_proj = flat_bf16(y_proj);
            if m <= candle_quant_kernels::NVFP4_GEMV_MAX_ROWS {
                let unquantized_ref = x_host
                    .matmul(&w_deq.t().unwrap())
                    .unwrap()
                    .broadcast_add(&bias_f)
                    .unwrap();
                let vs_ref = rel_rms(&got_proj, &flat(unquantized_ref));
                assert!(
                    (vs_ref as f64) <= candle_quant_kernels::GEMV_REL_RMS_TOL,
                    "M={m}: GEMV rel-RMS vs unquantized-activation reference {vs_ref}"
                );
                // More accurate than W4A4 against the dense product (no activation quantization).
                assert!(rel_rms(&got_proj, &flat(exact)) < vs_exact, "M={m}");
            } else {
                assert_eq!(got_proj, got, "M={m}: the projection ran cuBLASLt");
            }
        }

        // Residency: packed nibbles + padded UE4M3 scales + the bf16 bias, far below bf16.
        let bytes = p.resident_bytes().unwrap();
        assert_eq!(bytes, (wq.packed.len() + wq.scales.len() + n * 2) as u64);
        assert!(bytes < (n * k * 2) as u64);
    }

    /// sc-24140: the llama-family loader's shape policy — under NVFP4, an eligible shape becomes
    /// NVFP4 and one the FP4 GEMM cannot serve (`N % 16 != 0`) stays dense, reported as `Dense`
    /// (so the census counts it under `dense`), bias included.
    #[cfg(feature = "cuda")]
    #[test]
    fn load_eligible_keeps_an_ineligible_nvfp4_shape_dense_and_visible() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            return;
        };
        let eligible = Projection::load_eligible(
            ramp(64, 32, 7, &device).to_dtype(DType::BF16).unwrap(),
            None,
            Some(&format),
        )
        .unwrap();
        assert_eq!(eligible.kind(), ProjectionKind::Nvfp4);
        let bias = ramp(1, 50, 9, &device).reshape(50).unwrap();
        let odd =
            Projection::load_eligible(ramp(50, 32, 7, &device), Some(bias), Some(&format)).unwrap();
        assert_eq!(odd.kind(), ProjectionKind::Dense);
        assert_eq!(odd.resident_bytes(), Some((50 * 32 + 50) * 4));
        let mut census = ProjectionCensus::default();
        census.record(&eligible);
        census.record(&odd);
        assert_eq!((census.nvfp4.count, census.dense.count), (1, 1));
        assert_eq!(census.dense.params, 50 * 32);
    }

    /// An NVFP4 request never keeps an ineligible projection dense: it is a typed refusal.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_ineligible_shape_is_refused_not_kept_dense() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            return;
        };
        let w = ramp(50, 32, 7, &device);
        match Projection::load_as(w, None, Some(&format)) {
            Err(crate::Error::Nvfp4Refused(
                candle_quant_kernels::Nvfp4Refusal::ShapeIneligible { rows: 50, cols: 32 },
            )) => {}
            Err(other) => panic!("expected ShapeIneligible, got {other}"),
            Ok(p) => panic!("an ineligible NVFP4 projection loaded as {:?}", p.kind()),
        }
    }
}
