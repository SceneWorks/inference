//! Krea 2 packed-load seam (sc-9411, sc-9089 umbrella) — the candle twin of the flux2-dev conversion
//! (sc-9087), z-image (sc-9408), flux-schnell (sc-9407), and boogu (sc-9410), built on the shared
//! [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! Krea ships a **pre-quantized** MLX tier (`SceneWorks/krea-2-turbo-mlx`, bf16/q4/q8) whose q4/q8
//! snapshots store each quantized `Linear` as the MLX packed triple `{base}.weight` (u32 codes) +
//! `{base}.scales` + `{base}.biases`. The group size is read from each component `config.json`'s
//! `quantization.group_size` ([`candle_gen::quant::PackedConfig`]). Component open prepares
//! content-addressed GGML sidecars (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0`) using that group size;
//! module construction maps the sidecar and transfers its bytes without re-reading or converting the
//! affine triple. **No dense bf16 weight is ever materialized** on the packed path.
//!
//! Absent `.scales` (a dense bf16 tier, or a projection MLX left dense) the **dense** path is taken
//! **unchanged** (`candle_nn::Linear` / `Embedding` from `{base}.weight`), so one crate serves both a
//! dense bf16 and a packed q4/q8 snapshot.
//!
//! Krea's loader is a thin `MmapedSafetensors` wrapper ([`crate::loader::Weights`]) with an
//! adapter-merge **overlay** (`set_overlay`, sc-7836), not a `VarBuilder` — so this seam builds the
//! quantized module from a device-format [`QTensor`] supplied by `Weights` rather than the
//! VarBuilder-detecting `candle_gen::quant::lin`. The compute path is identical (the shared
//! dequant-on-forward `QLinear`/`QEmbedding`, sc-7702 — *not* candle's int8 `QMatMul` fast path, whose
//! q8_1 activation quant NaNs on outlier text features).
//!
//! **Adapter compose (sc-9411).** The `krea_2_raw` LoRA/LoKr merge folds deltas into the DiT attention
//! /FFN Linears — which is exactly the packed surface. On a packed tier the merge reconstructs the
//! dense base from the packed parts on the CPU and installs the merged **dense** weight in the overlay
//! ([`crate::adapters::merge_into_weights`]); the overlay then shadows the packed path for those keys
//! ([`crate::loader::linear_detect`] reads the overlay first). So the packed base stays packed for
//! untargeted projections, while an adapted projection resolves to its correct merged dense weight —
//! there is no packed `quantize_onto` to no-op here (that is the VarBuilder crates' seam).

use candle_gen::candle_core::quantized::QTensor;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module};
use candle_gen::gen_core::Quant;
use candle_gen::quant::{self as shared, AdaptLinear, Int8Context};

/// A Linear projection that is **residual-capable** (dense or MLX-packed base + optional forward-time
/// additive LoRA/LoKr residuals — the shared [`candle_gen::quant::AdaptLinear`], sc-11105) or
/// **int8-ConvRot** (a community INT8-ConvRot checkpoint's per-output-channel int8 projection, sc-9300).
/// Built dense ([`Self::dense`]), packed ([`Self::packed`]), or int8 ([`Self::convrot_int8`]); every
/// forward computes `x·Wᵀ + b` (+ any additive residual).
///
/// The `Adapt` arm folds the former `Dense(Linear)` + `Packed(shared::QLinear)` cases into the one
/// shared `AdaptLinear` (sc-11105): a dense base (the loaded bf16 weight, possibly adapter-merged via
/// the overlay — [`crate::adapters::merge_into_weights`]) or an MLX-packed base loaded straight from the
/// packed parts (`Q4_1`/`Q8_0`, **dequantizes-on-forward** into a dense matmul — sc-7702, *not* the int8
/// `QMatMul` fast path). On a **packed** tier a user LoRA rides as a forward-time additive residual on
/// the packed base ([`crate::adapters::install_additive`]) — the packed base keeps its footprint.
pub enum QLinear {
    /// Dense or MLX-packed base + optional forward-time additive residuals (sc-11105).
    Adapt(AdaptLinear),
    /// An **NVFP4** projection (sc-12110, epic 11037): the weight packed to NVFP4 (E2M1 block-16 +
    /// UE4M3 scales) and served through [`candle_gen::quant::Nvfp4Linear`] — the FP4 tensor-core W4A4
    /// GEMM on `sm_120`, or a transparent dequant→bf16 fallback for the W4A16 outlier override and off
    /// Blackwell. Built only when a [`crate::nvfp4_dit::DitPlan`] asks for it
    /// ([`crate::loader::linear_detect_planned`]); the shipping `linear_detect` path never produces one.
    Nvfp4(crate::nvfp4_dit::Nvfp4Proj),
    /// A **probe-wrapped** baseline projection (sc-12110): records its input activation's outlier
    /// sparsity, then delegates to the wrapped [`Self::Adapt`] leg. This is how
    /// the partition gate measures the trunk's *unperturbed* real activations. Instrumentation only —
    /// never on a timed or shipping path.
    Probed(crate::nvfp4_dit::ProbedProj),
}

impl QLinear {
    /// Wrap an already-loaded dense [`Linear`] (the loader built it from `{base}.weight` [+ `.bias`],
    /// or from the adapter-merged overlay) as a dense-base [`AdaptLinear`] — the logical `[out, in]` dims
    /// are recovered from the weight shape (`[out, in]`).
    pub fn dense(linear: Linear) -> Self {
        // `weight()` is `[out, in]`; read the dims before moving `linear` into the adapter.
        let (out_dim, in_dim) = linear.weight_dims().expect("a projection has rank two");
        Self::Adapt(AdaptLinear::from_dense(linear, in_dim, out_dim))
    }

    /// Build a **packed** projection from the MLX packed triple (`wq` u32 codes + `scales` + `biases`,
    /// optional dense `bias`) at the tier's `group_size`, on `wq`'s device, via the shared group-size
    /// -aware loader, wrapped as a packed-base [`AdaptLinear`] (residual-capable — sc-11105). No dense
    /// weight is materialized. The logical `[out, in]` dims come from the `scales` shape (`[out,
    /// in/group_size]`).
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        let (out_dim, in_dim) = {
            let sd = scales.dims();
            (sd[0], sd[1] * group_size)
        };
        let q = shared::QLinear::from_packed_gs(wq, scales, biases, bias, group_size, &device)?;
        Ok(Self::Adapt(AdaptLinear::from_packed(q, in_dim, out_dim)))
    }

    /// Build a packed projection from an already-converted GGML `QTensor` loaded from the
    /// content-addressed sidecar cache (SC-16096). Unlike [`Self::packed`], this accepts no MLX
    /// affine triple and therefore cannot repeat format conversion inside a block-window loop.
    pub fn packed_device_format(qtensor: QTensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = qtensor.shape().dims();
        if dims.len() != 2 {
            candle_gen::candle_core::bail!(
                "Krea packed device-format projection must be rank 2, got {:?}",
                qtensor.shape()
            );
        }
        let (out_dim, in_dim) = (dims[0], dims[1]);
        let q = shared::QLinear::from_qtensor_dequant(std::sync::Arc::new(qtensor), bias);
        Ok(Self::Adapt(AdaptLinear::from_packed(q, in_dim, out_dim)))
    }

    /// Bind native ConvRot parts using a private workspace for a standalone projection.
    /// Normal checkpoint loads share their reader's workspace and rotation cache.
    pub fn convrot_int8(
        codes: Tensor,
        scales: Vec<f32>,
        group: usize,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        Self::convrot_int8_in(
            codes,
            scales,
            group,
            bias,
            device,
            &Int8Context::new(device)?,
        )
    }

    /// Compatibility constructor for callers that supply signed code values as integer tensors.
    /// The checkpoint reader binds the stored byte view directly and bypasses this conversion.
    pub fn convrot_int8_in(
        codes: Tensor,
        scales: Vec<f32>,
        group: usize,
        bias: Option<Tensor>,
        device: &Device,
        context: &Int8Context,
    ) -> Result<Self> {
        let (rows, cols) = codes.dims2()?;
        let descriptor = candle_gen::gen_core::convrot_int8::ConvRotInt8::from_geometry(
            group,
            &[rows, cols],
            &[scales.len(), 1],
        )
        .map_err(candle_gen::candle_core::Error::Msg)?;
        descriptor
            .validate_scales(&scales)
            .map_err(candle_gen::candle_core::Error::Msg)?;
        let codes = match codes.dtype() {
            DType::U8 => codes.to_device(device)?,
            DType::F32 => {
                let values = codes
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if values
                    .iter()
                    .any(|v| !v.is_finite() || v.fract() != 0. || !(-128.0..=127.0).contains(v))
                {
                    candle_gen::candle_core::bail!("ConvRot requires exact signed integer codes");
                }
                let bytes = values
                    .into_iter()
                    .map(|v| v as i8 as u8)
                    .collect::<Vec<_>>();
                Tensor::from_vec(bytes, (rows, cols), device)?
            }
            DType::I64 => {
                let values = codes
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<i64>()?;
                let bytes = values
                    .into_iter()
                    .map(|v| {
                        i8::try_from(v).map(|n| n as u8).map_err(|_| {
                            candle_gen::candle_core::Error::Msg(
                                "ConvRot code exceeds signed I8".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::from_vec(bytes, (rows, cols), device)?
            }
            _ => candle_gen::candle_core::bail!("ConvRot requires exact signed integer codes"),
        };
        let mut resources = candle_gen::weights::ProjectionResources::default();
        resources.set_int8_context(context.clone());
        let packed = candle_gen::quant::convrot_int8::ConvRotLinear::from_native_parts(
            &resources,
            descriptor,
            codes,
            Tensor::from_vec(scales, (rows, 1), device)?,
            bias,
        )
        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        Ok(Self::Adapt(AdaptLinear::from_convrot(packed)))
    }

    /// `x·Wᵀ + b` (+ any additive residual). The `Adapt` arm delegates to the shared [`AdaptLinear`]
    /// (dense `candle_nn::Linear`, or the packed dequant-on-forward `QLinear` — sc-7702 — plus its
    /// residuals); native ConvRot uses its shared cuBLASLt integer GEMM.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Adapt(l) => __model_proof_output!(l.forward(x)),
            Self::Nvfp4(l) => __model_proof_output!(l.forward(x)),
            Self::Probed(l) => __model_proof_output!(l.forward(x)),
        }
    }

    /// [`Self::forward`] for the Qwen3-VL TE's **bf16-store / f32-compute** regime (sc-12828): the
    /// `Adapt` dense base upcasts its weight to `x`'s dtype per matmul, so the bf16-resident projection
    /// runs against the f32 hidden state bit-identically to an f32 store — at half the resident TE
    /// footprint. The quant legs (NVFP4 / probed / int8-ConvRot) already dequantize to the activation
    /// dtype, so they delegate to [`Self::forward`] unchanged. Inert (an `Arc` clone) on the f32-store
    /// training / ControlNet paths, where `x` already matches the base dtype.
    pub fn forward_upcast(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Adapt(l) => l.forward_upcast(x),
            Self::Nvfp4(_) | Self::Probed(_) => self.forward(x),
        }
    }

    /// Fold a dense residual-capable base to Q4/Q8 while preserving its additive adapter stack.
    /// Specialized ConvRot/NVFP4/probe projections implement different numeric contracts and reject
    /// an affine fold instead of silently changing format.
    pub fn quantize(&mut self, quant: Quant) -> Result<()> {
        match self {
            Self::Adapt(linear) => linear.quantize(quant),
            Self::Nvfp4(_) => candle_gen::candle_core::bail!(
                "Krea NVFP4 projections cannot be re-quantized as {quant:?}"
            ),
            Self::Probed(_) => candle_gen::candle_core::bail!(
                "Krea activation-probe projections cannot be quantized"
            ),
        }
    }

    /// CPU-stage fold for imported dense checkpoints: quantized projections land directly on the
    /// compute device, so the full dense DiT never co-resides there with its Q4/Q8 replacement.
    pub fn quantize_onto(&mut self, quant: Quant, device: &Device) -> Result<()> {
        match self {
            Self::Adapt(linear) => {
                linear.quantize_dequant_onto(quant, device)?;
                linear.to_device(device)
            }
            Self::Nvfp4(_) => candle_gen::candle_core::bail!(
                "Krea NVFP4 projections cannot be re-quantized as {quant:?}"
            ),
            Self::Probed(_) => candle_gen::candle_core::bail!(
                "Krea activation-probe projections cannot be quantized"
            ),
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        match self {
            Self::Adapt(linear) => linear.to_device(device),
            // sc-21483: route the NVFP4 arm through the shared host so any installed residual
            // migrates with the projection, and a cross-device move is refused rather than silently
            // dropped (the packed weight is staged at construction and cannot be re-staged). A move
            // onto the device it already occupies stays the no-op it has always been.
            Self::Nvfp4(p) => p.adapt_mut().to_device(device),
            Self::Probed(_) => Ok(()),
        }
    }

    /// The NVFP4 leg, when this projection is served through [`candle_gen::quant::Nvfp4Linear`] — the
    /// accounting seam [`crate::transformer::Krea2Transformer::nvfp4_report`] walks for SC#6/SC#4.
    pub(crate) fn nvfp4(&self) -> Option<&candle_gen::quant::Nvfp4Linear> {
        match self {
            Self::Nvfp4(p) => Some(p.linear()),
            _ => None,
        }
    }

    /// Whether this projection loaded directly from the MLX-packed tier (its `Adapt` base is packed) —
    /// used by the loaders + tests to assert a packed tier fired the packed path (not a silent dense
    /// fallback).
    #[cfg(test)]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Adapt(a) if a.is_packed())
    }

    /// Whether this projection loaded as an INT8-ConvRot int8 layer (sc-9300) — the detect-arm assertion
    /// (a ConvRot checkpoint's quantized surface fired the int8 path, not a silent dense fallback).
    /// Public, not `#[cfg(test)]`: the CUDA integration test `tests/convrot_int8_gpu.rs` links the
    /// library built without `cfg(test)` and asserts on it.
    pub fn is_convrot_int8(&self) -> bool {
        matches!(self, Self::Adapt(linear) if linear.is_convrot_int8())
    }

    /// The inner shared [`AdaptLinear`] used by dense/MLX-packed projections. The shared host also owns additive stacks for native ConvRot bases.
    pub fn as_adapt_mut(&mut self) -> Option<&mut AdaptLinear> {
        match self {
            Self::Adapt(a) => Some(a),
            Self::Nvfp4(p) => Some(p.adapt_mut()),
            Self::Probed(_) => None,
        }
    }

    /// Whether this projection can host job-local additive LoRA/LoKr residuals. Shipping dense,
    /// MLX-packed, **NVFP4** (sc-21483), and INT8-ConvRot projections are all supported; only the
    /// probe leg is excluded, because it is measurement instrumentation and never on a shipping path.
    pub(crate) fn as_additive_mut(&mut self) -> Option<&mut Self> {
        match self {
            Self::Adapt(_) | Self::Nvfp4(_) => Some(self),
            Self::Probed(_) => None,
        }
    }

    /// Whether a factor this projection cannot host must be a hard **admission error** rather than a
    /// skipped key (sc-21483). True for NVFP4: it has no fallback — the base cannot be folded,
    /// dequantized, or re-quantized to accommodate a mismatched adapter — so silently skipping would
    /// render an unadapted base with no signal. The dense and MLX-packed arms keep their existing
    /// skip-and-report behavior, where a shape mismatch means the key targeted a different module.
    ///
    /// Strictness follows the **constructed representation, not the plan**, and that is deliberate.
    /// A row the plan marks NVFP4 is served DENSE whenever the FP4 representation is unavailable —
    /// on CPU, on a pre-`sm_120` device, or when the row's shape cannot be aligned to
    /// `NVFP4_K_ALIGN`/`NVFP4_N_ALIGN` — and such a row arrives here as `Self::Adapt`, so it keeps
    /// skip-and-report. That is the correct answer for it: a dense base genuinely *can* fold or
    /// absorb a delta, so the no-fallback premise that justifies refusing does not hold. The
    /// consequence is that admission strictness is device-dependent for a plan-NVFP4 checkpoint —
    /// intended, and the same rule `AdditiveProj for AdaptLinear::strict_admission`
    /// (`crate::adapters`) applies, so the two hosts of one projection never disagree.
    pub(crate) fn strict_adapter_admission(&self) -> bool {
        matches!(self, Self::Nvfp4(_)) || self.is_convrot_int8()
    }

    pub(crate) fn additive_shape(&self) -> (usize, usize) {
        match self {
            Self::Adapt(a) => a.base_shape(),
            Self::Nvfp4(p) => p.adapt().base_shape(),
            Self::Probed(_) => {
                unreachable!("probe projections are not exposed to adapter installation")
            }
        }
    }

    pub(crate) fn push_additive_lora(&mut self, a: Tensor, b: Tensor, scale: f64) -> Result<()> {
        match self {
            Self::Adapt(host) if host.is_convrot_int8() => host.push_lora_checked(a, b, scale),
            Self::Adapt(host) => host.push_lora(a, b, scale),
            // NVFP4 admits through the shared CHECKED push: a shape/dtype/device mismatch is a typed
            // error here — at install, before the first sampler step — and the packed base is never
            // converted to another regime to make the factor fit.
            Self::Nvfp4(host) => host.adapt_mut().push_lora_checked(a, b, scale),
            Self::Probed(_) => {
                unreachable!("probe projections are not exposed to adapter installation")
            }
        }
        .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))
    }

    pub(crate) fn push_additive_lokr(
        &mut self,
        factors: candle_gen::quant::LokrFactors,
    ) -> Result<()> {
        match self {
            Self::Adapt(host) if host.is_convrot_int8() => {
                host.push_lokr_structured_checked(factors)
            }
            Self::Adapt(host) => host.push_lokr_structured(factors),
            Self::Nvfp4(host) => host.adapt_mut().push_lokr_structured_checked(factors),
            Self::Probed(_) => {
                unreachable!("probe projections are not exposed to adapter installation")
            }
        }
        .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))
    }

    pub(crate) fn clear_adapters(&mut self) {
        match self {
            Self::Adapt(host) => host.clear_adapters(),
            Self::Nvfp4(host) => host.adapt_mut().clear_adapters(),
            Self::Probed(_) => {}
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

/// A token embedding that is **dense** (`candle_nn::Embedding`) or **packed** (loaded straight from the
/// MLX-packed tier's `embed_tokens` triple via the shared [`candle_gen::quant::QEmbedding`], sc-9411).
/// The Krea Qwen3-VL TE keeps `language_model.embed_tokens` **dense** in the hosted q4/q8 tiers (only
/// the layer projections are packed), so this closes the packed-detect surface for future-proofing +
/// the guard, while the hosted tier takes the dense arm.
pub enum QEmbedding {
    Dense(Embedding),
    Packed(shared::QEmbedding),
}

impl QEmbedding {
    /// Wrap an already-loaded dense [`Embedding`].
    pub fn dense(embedding: Embedding) -> Self {
        Self::Dense(embedding)
    }

    /// Build a **packed** embedding from the MLX packed triple at `group_size`, dequantized to
    /// `out_dtype` (the dense-path table dtype — dtype parity), on `wq`'s device.
    pub fn packed(
        wq: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        out_dtype: candle_gen::candle_core::DType,
        group_size: usize,
    ) -> Result<Self> {
        let device = wq.device().clone();
        Ok(Self::Packed(shared::QEmbedding::from_packed_dtype_gs(
            wq, scales, biases, &device, out_dtype, group_size,
        )?))
    }

    /// Build from already-device-format bytes loaded through the file-backed sidecar cache.
    pub fn packed_device_format(qtensor: QTensor, out_dtype: DType) -> Result<Self> {
        Ok(Self::Packed(shared::QEmbedding::from_qtensor(
            qtensor, out_dtype,
        )?))
    }

    /// Index-select the embedding rows for `indexes`.
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => __model_proof_output!(e.forward(indexes)),
            Self::Packed(e) => __model_proof_output!(e.forward(indexes)),
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if let Self::Dense(embedding) = self {
            *embedding = Embedding::new(
                embedding.embeddings().to_device(device)?,
                embedding.hidden_size(),
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};
    use candle_gen::testkit::tensor_cosine;

    /// The Krea tier's MLX quant group size (64, read from `config.json` at load — pinned here as the
    /// hosted value so the fixtures exercise the real tier's grouping).
    const G: usize = 64;

    /// Test-side MLX Q4 packer at group `G`: per-element 4-bit codes → MLX u32 words (LSB-first
    /// nibbles). Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine grid)` —
    /// the exact packed-parts fixture the loaders consume plus the affine grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / G;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / G;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A group-64 packed `QLinear` forward matches a dense linear built from the SAME affine grid the
    /// pack represents — bit-exact (the Q4 → Q4_1 repack is lossless, both forwards dequant-to-dense
    /// -matmul). Uses a `to_out.0`-style output-projection shape.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qlinear_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let packed = QLinear::packed(&wq, &s, &b, None, G)?;
        assert!(packed.is_packed(), "group-64 triple ⇒ packed load");
        let dense = QLinear::dense(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        ));
        assert!(!dense.is_packed());

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = tensor_cosine(&packed.forward(&x)?, &dense.forward(&x)?) as f32;
        assert!(
            cos > 0.99999,
            "group-64 packed vs affine-grid cosine {cos:.6}"
        );
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qlinear_matches_dense_grid"
        );
        Ok(())
    }

    /// A group-64 packed `QEmbedding` reproduces the affine grid rows exactly, dequantized to the
    /// requested dtype (the future-proof path for a tier that ever packs `embed_tokens`).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let packed = QEmbedding::packed(&wq, &s, &b, DType::F32, G)?;
        assert!(packed.is_packed(), "group-64 triple ⇒ packed embedding");
        let dense = QEmbedding::dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 63, 12, 5], (5,), &dev)?;
        let p = packed.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "group-64 packed embedding deviates from the grid"
        );
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qembedding_matches_dense_grid"
        );
        Ok(())
    }

    /// F-121 (sc-11208): `convrot_int8` now builds the cuBLASLt IGEMM leg eagerly (where `?` is
    /// available) rather than in a lazy `get_or_init(|| … .expect())` on the sampler thread. On CPU
    /// that eager leg is skipped, so a valid build must construct + forward without panicking, and
    /// malformed parts must be typed errors from the constructor (never a `.expect()`/panic).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn convrot_int8_constructor_is_typed_error_not_panic() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for j in 0..in_dim {
                wv[o * in_dim + j] = ((o * 7 + j * 3) % 51) as f32 / 25.0 - 1.0;
            }
        }
        // Canonical weight → rotate → per-row int8: the on-disk ConvRot granularity.
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &dev)?;
        let r = candle_gen::quant::regular_hadamard(G, &dev)?;
        let rw = candle_gen::quant::convrot_rotate(&w, &r)?;
        let pc = candle_gen::quant::quantize_weight_int8_per_channel(&rw)?;

        // The native constructor must fail early on malformed operands and run valid ones.
        let lin = QLinear::convrot_int8(pc.q.clone(), pc.scale.clone(), G, None, &dev)?;
        assert!(lin.is_convrot_int8());
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        assert_eq!(lin.forward(&x)?.dims(), &[4, out_dim]);

        // Malformed parts ⇒ typed errors from the constructor, not panics.
        assert!(
            QLinear::convrot_int8(pc.q.clone(), vec![1.0; out_dim + 1], G, None, &dev).is_err(),
            "scale-len mismatch must be a typed error"
        );
        assert!(
            QLinear::convrot_int8(pc.q.clone(), pc.scale.clone(), G + 1, None, &dev).is_err(),
            "non-power-of-four group_size must be a typed error"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::convrot_int8_constructor_is_typed_error_not_panic");
        Ok(())
    }

    /// User LoRA and structured LoKr remain forward-time residuals over the ConvRot int8 base. The
    /// base forward is identical in each pair, so subtraction isolates the adapter contribution and
    /// proves the INT8-ConvRot route does not silently drop either advertised adapter kind.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn convrot_int8_applies_lora_and_lokr_without_replacing_int8_base() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (out_dim, in_dim, group) = (16usize, 64usize, 64usize);
        let canonical = Tensor::ones((out_dim, in_dim), DType::F32, &dev)?;
        let rotation = candle_gen::quant::regular_hadamard(group, &dev)?;
        let rotated = candle_gen::quant::convrot_rotate(&canonical, &rotation)?;
        let parts = candle_gen::quant::quantize_weight_int8_per_channel(&rotated)?;
        let build =
            || QLinear::convrot_int8(parts.q.clone(), parts.scale.clone(), group, None, &dev);
        let x = Tensor::ones((2, in_dim), DType::F32, &dev)?;
        let bare = build()?;
        let mut lora = build()?;
        let a = Tensor::ones((in_dim, 1), DType::F32, &dev)?;
        let b = Tensor::ones((1, out_dim), DType::F32, &dev)?;
        lora.push_additive_lora(a.clone(), b.clone(), 0.75)?;
        let lora_actual = (lora.forward(&x)? - bare.forward(&x)?)?;
        let lora_expected = (x.matmul(&a)?.matmul(&b)? * 0.75)?;
        let lora_max = (lora_actual - lora_expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            lora_max < 1e-5,
            "ConvRot LoRA residual drifted ({lora_max})"
        );

        let w1 = Tensor::ones((4, 8), DType::F32, &dev)?;
        let w2 = Tensor::ones((4, 8), DType::F32, &dev)?;
        let factors = candle_gen::quant::LokrFactors::build(
            0.5,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("4x8 Kronecker factors fit a 16x64 projection");
        let expected = factors.residual(&x)?;
        let mut lokr = build()?;
        lokr.push_additive_lokr(factors)?;
        let lokr_actual = (lokr.forward(&x)? - bare.forward(&x)?)?;
        let lokr_max = (lokr_actual - expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            lokr_max < 1e-5,
            "ConvRot LoKr residual drifted ({lokr_max})"
        );
        assert!(lora.is_convrot_int8() && lokr.is_convrot_int8());
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::convrot_int8_applies_lora_and_lokr_without_replacing_int8_base");
        Ok(())
    }

    /// F-121 device regression (sc-11208): the production ConvRot path materializes the int8 codes on
    /// the **CPU** (the loader's `get_int8_codes`, to avoid 8× VRAM), so `convrot_int8` must build the
    /// cuBLASLt IGEMM leg on the model's resident **compute** device (the activation's CUDA device), NOT
    /// the CPU-resident codes' device. This reproduces that exact split — CPU codes + a CUDA compute
    /// device + a CUDA activation — and asserts the int8 IGEMM leg (not a cross-device dequant-dense
    /// fallback, which the buggy `w_i8.device().is_cuda()` gate would have hit) runs and matches the CPU
    /// dequant-dense reference within int8 tolerance. A CPU-only test cannot catch this — on CPU `lt` is
    /// legitimately `None`. Skips cleanly when no CUDA device is present.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn convrot_int8_cuda_forward_with_cpu_codes_matches_dequant_dense() -> Result<()> {
        let cuda = match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => d,
            _ => {
                eprintln!("[sc-11208] no CUDA device; skipping ConvRot int8 CUDA forward gate");
                #[cfg(feature = "model-inference-proof")]
                crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/src/quant.rs:870 (convrot_int8_cuda_forward_with_cpu_codes_matches_dequant_dense): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
                return Ok(());
            }
        };
        let cpu = Device::Cpu;
        // cuBLASLt int8 needs K and N multiples of 16; group_size (a power of four) must divide K.
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for j in 0..in_dim {
                wv[o * in_dim + j] = ((o * 7 + j * 3) % 51) as f32 / 25.0 - 1.0;
            }
        }
        // Canonical weight → rotate → per-row int8, codes left on the CPU exactly as the loader's
        // `get_int8_codes` materializes them in production.
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &cpu)?;
        let r = candle_gen::quant::regular_hadamard(G, &cpu)?;
        let rw = candle_gen::quant::convrot_rotate(&w, &r)?;
        let pc = candle_gen::quant::quantize_weight_int8_per_channel(&rw)?;
        let w_i8_cpu = pc.q;
        assert!(
            !w_i8_cpu.device().is_cuda(),
            "fixture must mirror production: int8 codes materialized on the CPU"
        );

        // Compute device = CUDA (the resident / activation device), codes on CPU: the production split.
        let lin_cuda = QLinear::convrot_int8(w_i8_cpu.clone(), pc.scale.clone(), G, None, &cuda)?;
        assert!(lin_cuda.is_convrot_int8());
        let x = Tensor::randn(0f32, 1f32, (8, in_dim), &cuda)?;
        // Must take the int8 IGEMM leg (built on `cuda`), NOT fall through to a cross-device
        // dequant-dense matmul (CPU weight × CUDA activation), which is the F-121 bug this guards.
        let y_cuda = lin_cuda
            .forward(&x)?
            .to_dtype(DType::F32)?
            .to_device(&cpu)?;

        // Reference: the SAME parts on a CPU compute device (dequant-dense leg), full-precision
        // activation — the int8-tolerance yardstick for the IGEMM path.
        let rotated = candle_gen::quant::convrot_rotate(&x.to_device(&cpu)?, &r)?;
        let row_scales = Tensor::from_vec(pc.scale, (out_dim, 1), &cpu)?;
        let reference = w_i8_cpu.to_dtype(DType::F32)?.broadcast_mul(&row_scales)?;
        let y_ref = rotated.matmul(&reference.t()?)?;

        let cos = tensor_cosine(&y_cuda, &y_ref) as f32;
        let num = (y_cuda.sub(&y_ref)?)
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let den = y_ref
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?
            .sqrt()
            .max(1e-6);
        let rel = num / den;
        assert!(
            cos > 0.99 && rel < 0.1,
            "CUDA int8 IGEMM vs CPU dequant-dense: cosine {cos:.5}, rel-RMS {rel:.4}"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::convrot_int8_cuda_forward_with_cpu_codes_matches_dequant_dense");
        Ok(())
    }
}
