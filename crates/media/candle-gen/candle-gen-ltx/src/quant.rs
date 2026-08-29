//! LTX-2.3 packed-load seam (sc-9417, sc-9089 umbrella — the **last** umbrella crate) — the candle
//! twin of the flux2-dev conversion (sc-9087), krea (sc-9411), qwen-image (sc-9415), and the other
//! merged crate conversions, built on the shared [`candle_gen::quant`] packed-load module (sc-9086).
//!
//! LTX ships a **pre-quantized** MLX tier (`SceneWorks/ltx-2.3-mlx`, q4/q8; +a dense `gemma/` TE; **no
//! bf16**) whose `transformer.safetensors` stores each quantized `AvDiT` attention / feed-forward
//! `Linear` as the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`
//! (the dense `{base}.bias` rides alongside). The group size is read from the component
//! `quantize_config.json`'s `quantization.group_size` ([`candle_gen::quant::PackedConfig`]) and threaded
//! through the shared group-size-aware loaders called directly here —
//! [`candle_gen::quant::QLinear::from_packed_gs`] for Linears and [`candle_gen::quant::embedding_gs`] for
//! `embed_tokens` (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0`) — the hosted tier packs
//! at group 64, but the loader honours whatever the config says (exactly as boogu threads its group 32).
//! **No dense bf16 weight is ever materialized** on the packed path.
//!
//! ## Per-component packed / dense split (hf-header audit of `SceneWorks/ltx-2.3-mlx` q4, sc-9417)
//!
//! | Component | File | Packed surface |
//! |---|---|---|
//! | **AvDiT** (video+audio+cross-modal attn/ff) | `transformer.safetensors` | **PACKED** (1344 `.scales`: `to_q/k/v/to_out` + `ff.proj_in/out`) |
//! | **Gemma-3-12B TE** | `gemma/*.safetensors` | dense in the tier (0 `.scales`, no `quantization` block) |
//! | connector, VAE (3D conv) enc/dec, audio VAE, vocoder, upsampler | separate files | dense (0 `.scales`) |
//!
//! So on this tier only the AvDiT attn/ff Linears carry `.scales`; every other component is dense
//! (3-D / audio-VAE / vocoder convs are not MLX-affine-packed). This seam routes the whole
//! **AvDiT + Gemma TE + connector Linear surface** (and Gemma's `embed_tokens`) through the shared
//! **packed-detect** loaders — [`qlinear`] / [`qembedding`] build a packed module when a `{base}.scales`
//! sibling is present, else the **dense** path is taken **unchanged** (`{base}.weight` [+ `.bias`], cast
//! to the vb dtype exactly as the legacy per-crate `linear` did). One crate serves both the current
//! dense single-file checkpoint (no `.scales` ⇒ every leaf dense, byte-identical to before) and a packed
//! tier. The gate projections (`to_gate_logits`) are dense **in the tier** but still routed through the
//! packed-detecting loader (dense-fallback superset), as are the Gemma / connector projections — the
//! detect is by `.scales` presence, not by a hardcoded per-key packed/dense list.
//!
//! The 3-D-conv VAEs, audio VAE and vocoder stay **dense** (their conv weights are never affine-packed),
//! loaded through [`guard_no_scales`]: it loads the dense weight and **errors loudly** if a `.scales`
//! sibling unexpectedly appears where we load dense (a tier that ever packs a conv would otherwise load
//! u32 codes as garbage silently). Dense parity is not uniform across the seam: the **DiT/Gemma** dense
//! arm ([`qlinear`]/[`qembedding`]) is byte-identical to the legacy read (both cast to the bf16 DiT/Gemma
//! builder dtype, a no-op), but [`guard_no_scales`] casts the weight to the passed `vb.dtype()` — **F32**
//! for the VAE/audio-VAE/vocoder builders — so a bf16 on-disk weight is upcast to F32 (lossless) where the
//! legacy per-crate read kept the on-disk dtype. The whole seam is the shared **dequant-on-forward** `QLinear`
//! (sc-7702) — *not* candle's int8 `QMatMul` fast path, whose q8_1 activation quant NaNs on outliers.
//!
//! ## Scope boundary — packed-detect seam only; tier *ingestion* is a follow-up (sc-9545)
//!
//! This seam makes every AvDiT/Gemma/connector Linear **packed-detect** on the crate's own key layout:
//! the current single-file dense checkpoint has no `.scales`, so every leaf takes the dense arm
//! **byte-identically** to before, and a `.scales` sibling at any of those keys fires the packed path.
//! **Actually loading the hosted `SceneWorks/ltx-2.3-mlx` q4/q8 tier is a separate loader effort**: that
//! tier ships one safetensors *per component* (not the crate's single bundled file) and its packed
//! `transformer.safetensors` uses **different key names** than the dense Lightricks checkpoint
//! (`to_out` not `to_out.0`, `ff.proj_in/out` not `net.0.proj`/`net.2`, `linear1/2` not `linear_1/2`).
//! Resolving the `q4/` subfolder + `gemma/` shards and remapping those keys (`to_out.0`↔`to_out` etc.) so
//! the `.scales` siblings are found — **and the real packed GPU video render** — are deferred to and
//! tracked by **sc-9545**; no real packed render has been run here. The tests below validate the wiring
//! with **synthetic** packed fixtures built on the **real** AvDiT block-0 key layout the hf-header audit
//! captured — they prove the packed-detect seam fires on that layout, not that a real tier was ingested.

use std::sync::OnceLock;

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_gen::quant as shared;
#[cfg(feature = "cuda")]
use candle_gen::quant::Int8Linear;
use candle_gen::quant::{ActPrecision, LokrFactors, Nvfp4Linear, Nvfp4Regime};
use candle_gen::train::lora::LoraLinear;

use crate::advanced_quant::{active_source, record_projection_execution, AdvancedOperatorKind};

/// The LTX MLX tier's quant group size (read from `quantize_config.json`'s `quantization.group_size`;
/// the hosted q4/q8 tiers pack at 64, MLX's default). Threaded through the shared group-size-aware
/// loaders so a future tier that packs at a different group is honoured, not silently mis-read.
pub const GROUP_SIZE: usize = shared::MLX_GROUP_SIZE; // 64

/// A Linear projection whose frozen base is **dense** (the loaded bf16 weight, the legacy per-crate
/// path) or **packed** (loaded straight from the MLX-packed tier via the shared
/// [`candle_gen::quant::QLinear`], sc-9417). The shared [`LoraLinear`] wrapper makes either base
/// inference-adaptable without changing its adapter-free forward.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum QLinearBase {
    Adapt(LoraLinear),
    ConvRot(ConvRotLinear),
    Nvfp4(Nvfp4Linear),
}

/// LTX projection dispatch. Advanced arms are materially different operators, not labels over the
/// dense/MLX-affine loader. They are only constructible from an active, descriptor-validated source.
pub struct QLinear {
    base: QLinearBase,
    path: String,
    in_features: usize,
    out_features: usize,
    attestation: String,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct ConvRotLinear {
    group_size: usize,
    rotation: OnceLock<Tensor>,
    #[cfg(feature = "cuda")]
    linear: Int8Linear,
}

impl ConvRotLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let rotation = if let Some(rotation) = self.rotation.get() {
                rotation
            } else {
                let rotation = shared::regular_hadamard(self.group_size, x.device())?;
                self.rotation.get_or_init(|| rotation)
            };
            let rotated = shared::convrot_rotate(x, rotation)?;
            self.linear.forward(&rotated)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = x;
            candle_gen::candle_core::bail!(
                "LTX INT8-ConvRot has no dense CPU fallback; build with cuda and run its IGEMM operator"
            )
        }
    }
}

impl QLinear {
    /// `x·Wᵀ + b` plus any inference residuals. The frozen dense/packed base dispatch is owned by the
    /// shared [`LoraLinear`].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = match &self.base {
            QLinearBase::Adapt(linear) => linear.forward(x),
            QLinearBase::ConvRot(linear) => linear.forward(x),
            QLinearBase::Nvfp4(linear) => linear.forward(x),
        }?;
        record_projection_execution(
            self.path.clone(),
            self.operator_kind(),
            self.attestation.clone(),
        );
        Ok(output)
    }

    /// Whether this projection loaded directly from the MLX-packed tier (the packed path) — used by the
    /// tests to assert a packed tier fired the packed path (not a silent dense fallback).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        matches!(&self.base, QLinearBase::Adapt(linear) if linear.is_packed())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn operator_kind(&self) -> AdvancedOperatorKind {
        match &self.base {
            QLinearBase::Adapt(linear) if linear.is_packed() => AdvancedOperatorKind::MlxAffine,
            QLinearBase::Adapt(_) => AdvancedOperatorKind::Dense,
            QLinearBase::ConvRot(_) => AdvancedOperatorKind::Int8ConvRotIgemm,
            QLinearBase::Nvfp4(_) => AdvancedOperatorKind::Nvfp4W4A4,
        }
    }

    /// Canonical PEFT path captured from the projection's loading builder.
    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    /// Logical `(out_features, in_features)` for adapter factor validation.
    pub(crate) fn base_shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// Attach a LoRA residual with one strength per distilled denoise pass.
    pub(crate) fn push_additive_lora_per_pass(
        &mut self,
        a: Tensor,
        b: Tensor,
        scales: Vec<f64>,
    ) -> Result<()> {
        match &mut self.base {
            QLinearBase::Adapt(linear) => linear
                .push_additive_lora_per_pass(a, b, scales)
                .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string())),
            QLinearBase::ConvRot(_) | QLinearBase::Nvfp4(_) => candle_gen::candle_core::bail!(
                "LTX advanced native quant projections refuse adapters; no dense merge/fallback is permitted"
            ),
        }
    }

    /// Attach a structured LoKr residual with one user strength per distilled denoise pass.
    pub(crate) fn push_additive_lokr_per_pass(
        &mut self,
        factors: LokrFactors,
        scales: Vec<f64>,
    ) -> Result<()> {
        match &mut self.base {
            QLinearBase::Adapt(linear) => linear
                .push_additive_lokr_per_pass(factors, scales)
                .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string())),
            QLinearBase::ConvRot(_) | QLinearBase::Nvfp4(_) => candle_gen::candle_core::bail!(
                "LTX advanced native quant projections refuse adapters; no dense merge/fallback is permitted"
            ),
        }
    }

    /// Select the active distilled denoise pass for this projection's additive residuals.
    pub(crate) fn set_additive_pass(&self, pass: usize) {
        if let QLinearBase::Adapt(linear) = &self.base {
            linear.set_additive_pass(pass);
        }
    }

    /// Wrap this frozen projection in the shared training-time LoRA seam without changing its
    /// forward. Dense bases remain dense; an MLX-packed base remains packed (and therefore
    /// inference-only, as required by [`candle_gen::train::lora::LoraLinear`]).
    pub(crate) fn into_lora(
        self,
        in_features: usize,
        out_features: usize,
        path: String,
    ) -> LoraLinear {
        debug_assert_eq!(self.in_features, in_features);
        debug_assert_eq!(self.out_features, out_features);
        debug_assert_eq!(self.path, path);
        match self.base {
            QLinearBase::Adapt(linear) => linear,
            QLinearBase::ConvRot(_) | QLinearBase::Nvfp4(_) => {
                panic!("advanced native quant projections are inference-only")
            }
        }
    }

    /// Expose the same trainable residual seam used by the standalone training DiT.  Keeping this
    /// on the packed-aware wrapper is what makes full AV QLoRA train the actual loaded projection
    /// instead of rebuilding a dense video-only shadow model.
    pub(crate) fn lora_mut(&mut self) -> Option<&mut LoraLinear> {
        match &mut self.base {
            QLinearBase::Adapt(linear) => Some(linear),
            QLinearBase::ConvRot(_) | QLinearBase::Nvfp4(_) => None,
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

/// **Packed-detecting** Linear loader for `{key}` under `vb` (sc-9417). If `{key}.scales` is present (a
/// pre-quantized MLX tier), build a packed frozen base straight from the parts on `vb`'s device
/// via the shared [`candle_gen::quant::QLinear::from_packed_gs`] at [`GROUP_SIZE`] — **no dense weight is
/// materialized**.
/// Otherwise the **dense** path is taken unchanged: `{key}.weight` [+ `{key}.bias` when `bias`], cast to
/// the vb dtype (bf16) exactly as the legacy per-crate `linear` did. `key` is the full dotted prefix
/// (e.g. `attn1.to_out.0`), so the `.scales`/`.biases` siblings survive any `to_out.0`-style nesting
/// (the sc-8670 remap trap: build the base string first, then detect — never `.pp()` past the sibling).
///
/// The dense fallback reads the weight shape from the file (`get_unchecked`), not threaded config dims,
/// so it drops in for the old `linear(vb, key) -> Linear` helpers without plumbing `in_dim`/`out_dim`.
#[allow(unused_variables)]
pub fn qlinear(vb: &VarBuilder, key: &str, bias: bool) -> Result<QLinear> {
    let path = vb.pp(key).prefix();
    if let Some(source) = active_source() {
        if source.is_advanced_projection(&path) {
            let bias_tensor = if bias {
                Some(
                    vb.get_unchecked(&format!("{key}.bias"))?
                        .to_dtype(vb.dtype())?,
                )
            } else {
                None
            };
            let loaded = source.load_projection(&path, vb.device())?;
            let (base, in_features, out_features, attestation) = match loaded {
                crate::advanced_quant::LoadedAdvancedProjection::Int8ConvRot {
                    codes,
                    scale,
                    group_size,
                    context,
                    attestation,
                } => {
                    if !vb.device().is_cuda() || !context.is_int8() {
                        candle_gen::candle_core::bail!(
                            "LTX INT8-ConvRot projection `{path}` requires a live CUDA cuBLASLt IGEMM context; dense fallback is forbidden"
                        );
                    }
                    let (out_features, in_features) = codes.dims2()?;
                    #[cfg(feature = "cuda")]
                    let linear = Int8Linear::from_per_channel_parts(
                        codes,
                        scale,
                        bias_tensor,
                        context.handle_for(vb.device())?.clone(),
                    )?;
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = (codes, scale, bias_tensor, context);
                        candle_gen::candle_core::bail!(
                            "LTX INT8-ConvRot requires a cuda-enabled executable"
                        );
                    }
                    #[cfg(feature = "cuda")]
                    {
                        (
                            QLinearBase::ConvRot(ConvRotLinear {
                                group_size,
                                rotation: OnceLock::new(),
                                linear,
                            }),
                            in_features,
                            out_features,
                            attestation,
                        )
                    }
                }
                crate::advanced_quant::LoadedAdvancedProjection::Nvfp4 {
                    packed,
                    context,
                    attestation,
                } => {
                    let out_features = packed.rows;
                    let in_features = packed.cols;
                    let linear = Nvfp4Linear::from_packed_in(
                        packed,
                        bias_tensor,
                        vb.device(),
                        ActPrecision::W4A4,
                        &context,
                    )?;
                    if linear.regime() != Nvfp4Regime::Fp4W4A4 {
                        candle_gen::candle_core::bail!(
                            "LTX NVFP4 projection `{path}` did not construct the native W4A4 operator ({:?}); dense BF16 fallback is forbidden",
                            linear.fallback_cause()
                        );
                    }
                    (
                        QLinearBase::Nvfp4(linear),
                        in_features,
                        out_features,
                        attestation,
                    )
                }
            };
            return Ok(QLinear {
                base,
                path,
                in_features,
                out_features,
                attestation,
            });
        }
        source.refuse_undeclared_advanced_tensor(&path)?;
    }
    let scales_key = format!("{key}.scales");
    if vb.contains_tensor(&scales_key) {
        let device = vb.device().clone();
        // Native `U32` for the packed codes (a float cast would reinterpret the bit-packed nibbles);
        // scales/biases upcast to f32 exactly; the dense `{key}.bias` (distinct from the packed
        // `{key}.biases`) rides at the vb dtype.
        let wq = vb.get_unchecked_dtype(&format!("{key}.weight"), DType::U32)?;
        let scales = vb.get_unchecked_dtype(&scales_key, DType::F32)?;
        let biases = vb.get_unchecked_dtype(&format!("{key}.biases"), DType::F32)?;
        let bias = if bias {
            Some(vb.get_unchecked_dtype(&format!("{key}.bias"), vb.dtype())?)
        } else {
            None
        };
        let out_features = scales.dim(0)?;
        let in_features = scales.dim(1)? * GROUP_SIZE;
        return Ok(QLinear {
            base: QLinearBase::Adapt(LoraLinear::from_qlinear(
                shared::QLinear::from_packed_gs(&wq, &scales, &biases, bias, GROUP_SIZE, &device)?,
                in_features,
                out_features,
                path.clone(),
            )),
            path,
            in_features,
            out_features,
            attestation: "mlx-affine-triple".into(),
        });
    }
    // Dense path, byte-identical to the legacy `linear`: read `{key}.weight` [+ `.bias`], cast to the
    // vb dtype (bf16). `get_unchecked` (no shape validation) matches the old helper's behavior.
    let w = vb
        .get_unchecked(&format!("{key}.weight"))?
        .to_dtype(vb.dtype())?;
    let b = if bias {
        Some(
            vb.get_unchecked(&format!("{key}.bias"))?
                .to_dtype(vb.dtype())?,
        )
    } else {
        None
    };
    let (out_features, in_features) = w.dims2()?;
    Ok(QLinear {
        base: QLinearBase::Adapt(LoraLinear::from_linear(
            Linear::new(w, b),
            in_features,
            out_features,
            path.clone(),
        )),
        path,
        in_features,
        out_features,
        attestation: "dense-weight".into(),
    })
}

/// A resolved token-embedding **table** (`[vocab, hidden]`), loaded either dense (`{key}.weight`, cast
/// to the vb dtype) or from the MLX-packed tier's `embed_tokens` triple (dequantized to the vb dtype
/// once at load, via the shared [`candle_gen::quant::embedding`]). The Gemma encoder index-selects this
/// table directly and applies its bespoke `× √hidden` scaling to the raw rows, so it needs the table
/// tensor (not a `candle_nn::Embedding` wrapper). The Gemma `embed_tokens` is **dense** in the hosted
/// tier — the packed arm only future-proofs the surface + closes the guard.
pub struct QEmbedding {
    /// The resolved `[vocab, hidden]` table at the vb dtype (dequantized once if the tier packs it).
    table: Tensor,
    packed: bool,
}

impl QEmbedding {
    /// The `[vocab, hidden]` table (the Gemma encoder scales + index-selects it directly).
    pub fn weight(&self) -> &Tensor {
        &self.table
    }

    /// Index-select the embedding rows for `indexes` (the `candle_nn::Embedding` contract, used by tests
    /// / any consumer that prefers the wrapped forward).
    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let hidden = self.table.dim(1)?;
        Embedding::new(self.table.clone(), hidden).forward(indexes)
    }

    /// Whether the table loaded from the MLX-packed tier (vs a dense `{key}.weight`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_packed(&self) -> bool {
        self.packed
    }
}

/// **Packed-detecting** embedding loader for `{key}` under `vb` (sc-9417): packed when `{key}.scales`
/// is present (dequantize the packed table to the vb dtype once via the shared
/// [`candle_gen::quant::embedding_gs`] at [`GROUP_SIZE`] — dtype parity with the dense table), else the
/// **dense** table (`{key}.weight`, cast to the vb dtype, exactly the legacy Gemma `embed_tokens` read).
/// `vocab`/`hidden` size the dense fallback's shape check.
pub fn qembedding(vb: &VarBuilder, key: &str, vocab: usize, hidden: usize) -> Result<QEmbedding> {
    if vb.contains_tensor(&format!("{key}.scales")) {
        // Build the shared packed embedding (dequantizes to vb.dtype() on the vb device), then
        // materialize its `[vocab, hidden]` table once for the Gemma encoder's direct index-select.
        let e = shared::embedding_gs(vb, key, vocab, hidden, GROUP_SIZE)?;
        let idx = Tensor::arange(0u32, vocab as u32, vb.device())?;
        let table = e.forward(&idx)?;
        return Ok(QEmbedding {
            table,
            packed: true,
        });
    }
    let table = vb
        .get_unchecked(&format!("{key}.weight"))?
        .to_dtype(vb.dtype())?;
    Ok(QEmbedding {
        table,
        packed: false,
    })
}

/// Load a **dense** weight for `{key}` (`{key}.weight`, cast to `dtype`), erroring loudly if a
/// `{key}.scales` sibling is present — the guard for components we deliberately keep dense (the 3-D-conv
/// VAEs, the audio VAE, the vocoder). Those weights are never MLX-affine-packed, so a `.scales` sibling
/// means the tier packed a leaf we would silently load as u32-code garbage. Fail with a clear message
/// instead (sc-9417). Returns the raw dense tensor for the caller to wrap (conv/norm/etc.).
pub fn guard_no_scales(vb: &VarBuilder, key: &str, dtype: DType) -> Result<Tensor> {
    if vb.contains_tensor(&format!("{key}.scales")) {
        candle_gen::candle_core::bail!(
            "ltx: `{key}.scales` present where a DENSE weight is expected — this component (VAE / \
             audio-VAE / vocoder conv) is not MLX-affine-packed, so a packed sibling would load as \
             u32-code garbage. Route it through `quant::qlinear` if it is genuinely a packed Linear, \
             or file a follow-up to pack this surface (sc-9417)."
        );
    }
    vb.get_unchecked(&format!("{key}.weight"))?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer at [`GROUP_SIZE`] (64): per-element 4-bit codes → MLX u32 words
    /// (LSB-first nibbles). Returns `(wq [out, in/8] u32, scales [out, in/G], biases [out, in/G], affine
    /// grid [out, in])` — the exact packed-parts fixture the loaders consume plus the grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / g;
        let scales: Vec<f32> = (0..groups).map(|k| 0.0625 * (k as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|k| -0.5 - 0.25 * k as f32).collect();
        let gpr = in_dim / g;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let k = row * gpr + col / g;
                scales[k] * codes[i] as f32 + biases[k]
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

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// Build a safetensors mimicking the **real LTX AvDiT block-0 key layout** — the exact
    /// `transformer_blocks.0.attn1.to_{q,out}` packed triples the hf-header audit found (`to_out` with
    /// **no** `.0` and a dense `.bias`, plus a dense `to_gate_logits`) — and load through the
    /// packed-detecting `qlinear`. The `.scales`/`.biases` siblings must fire the packed path (not a
    /// silent dense fallback), the dense gate must stay dense, and the packed forward must match the
    /// affine grid the pack represents (bit-exact repack + dequant-on-forward).
    #[test]
    fn qlinear_packed_detect_on_avdit_key_layout() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        // A dense bias for the packed to_out (the tier ships `to_out.bias` alongside the packed triple).
        let out_bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("transformer_blocks.0.attn1.to_out.weight".into(), wq);
        map.insert("transformer_blocks.0.attn1.to_out.scales".into(), s);
        map.insert("transformer_blocks.0.attn1.to_out.biases".into(), b);
        map.insert(
            "transformer_blocks.0.attn1.to_out.bias".into(),
            out_bias.clone(),
        );
        // Dense gate (`to_gate_logits`) — no `.scales`, the tier keeps it dense; must take dense arm.
        map.insert(
            "transformer_blocks.0.attn1.to_gate_logits.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        map.insert(
            "transformer_blocks.0.attn1.to_gate_logits.bias".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev)?,
        );

        let tmp = tmp.path().join("sc9417_avdit.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let blk = vb.pp("transformer_blocks.0.attn1");

        let mut packed = qlinear(&blk, "to_out", true)?;
        assert!(
            packed.is_packed(),
            "`.scales` under to_out ⇒ packed load (not a silent dense fallback)"
        );
        let gate = qlinear(&blk, "to_gate_logits", true)?;
        assert!(
            !gate.is_packed(),
            "no `.scales` ⇒ dense gate, path unchanged"
        );

        // The packed forward reproduces the affine grid (+ the dense bias) bit-exactly.
        let grid_lin = QLinear {
            base: QLinearBase::Adapt(LoraLinear::from_linear(
                Linear::new(
                    Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
                    Some(out_bias),
                ),
                in_dim,
                out_dim,
                "grid".into(),
            )),
            path: "grid".into(),
            in_features: in_dim,
            out_features: out_dim,
            attestation: "dense-weight".into(),
        };
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let base = packed.forward(&x)?;
        let cos = cosine(&base, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        // Inference LoRA rides through the shared additive core without unpacking the q4 base.
        let rank = 4;
        let a = Tensor::randn(0f32, 0.1f32, (in_dim, rank), &dev)?;
        let bfac = Tensor::randn(0f32, 0.1f32, (rank, out_dim), &dev)?;
        packed
            .push_additive_lora_per_pass(a.clone(), bfac.clone(), vec![0.7])
            .unwrap();
        assert!(
            packed.is_packed(),
            "adding LoRA must preserve packed residency"
        );
        let adapted = packed.forward(&x)?;
        let effect = (adapted - &base)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(effect > 1e-6, "nonzero LoRA must change packed output");

        let mut zero = qlinear(&blk, "to_out", true)?;
        zero.push_additive_lora_per_pass(a, bfac, vec![0.0])
            .unwrap();
        assert!(
            zero.is_packed(),
            "scale=0 LoRA must preserve packed residency"
        );
        let zero_diff = (zero.forward(&x)? - base)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(zero_diff, 0.0, "scale=0 must equal the packed base");

        Ok(())
    }

    /// The dense arm of `qlinear` is byte-identical to the legacy per-crate `linear` (read `{key}.weight`
    /// [+ `.bias`] at the vb dtype) — a dense checkpoint (no `.scales` anywhere) loads every leaf dense,
    /// unchanged. Confirms the current single-file LTX checkpoint path is untouched.
    #[test]
    fn qlinear_dense_path_unchanged() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("proj.weight".into(), w.clone());
        map.insert("proj.bias".into(), b.clone());
        let tmp = tmp.path().join("sc9417_dense.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        crate::advanced_quant::begin_operator_attestation(crate::quant_eval::Ltx25QuantMode::Bf16);
        let lin = qlinear(&vb, "proj", true)?;
        assert!(!lin.is_packed(), "no `.scales` ⇒ dense");
        let before_forward = crate::advanced_quant::finish_operator_attestation()
            .unwrap_err()
            .to_string();
        assert!(before_forward.contains("executed no"), "{before_forward}");
        // Reference: the exact legacy read.
        let ref_lin = Linear::new(w, Some(b));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let output = lin.forward(&x)?;
        let after_forward = crate::advanced_quant::finish_operator_attestation()?;
        assert_eq!(after_forward.executed_projection_count, 1);
        let dev_max = (output.sub(&ref_lin.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "dense arm deviates from the legacy linear read"
        );

        Ok(())
    }

    /// `quantize` is a **no-op** on a packed `shared::QLinear` — an MLX-packed weight must never be
    /// double-quantized (the LTX seam relies on the shared idempotence when composing loads). The stored
    /// `Q4_1` weight and the forward stay unchanged.
    #[test]
    fn packed_quantize_is_noop() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut packed = shared::QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = packed.forward(&x)?;
        packed.quantize(candle_gen::gen_core::Quant::Q4)?; // must no-op, not re-quantize
        let after = packed.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the packed forward");
        Ok(())
    }

    /// The packed-detecting `qembedding` fires on the Gemma `embed_tokens.scales` sibling and materializes
    /// a table that reproduces the affine grid rows exactly (dtype parity via the vb dtype). Also confirms
    /// the dense arm (no `.scales`) loads the raw `{key}.weight` table unchanged.
    #[test]
    fn qembedding_packed_detect_on_gemma_embed_tokens() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        // A dense sibling table (no `.scales`) — the hosted-tier arm.
        map.insert(
            "dense_embed.weight".into(),
            Tensor::from_vec(grid.clone(), (vocab, hidden), &dev)?,
        );
        let tmp = tmp.path().join("sc9417_emb.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = qembedding(&vb, "embed_tokens", vocab, hidden)?;
        assert!(packed.is_packed(), "`.scales` ⇒ packed embed_tokens");
        let dense = qembedding(&vb, "dense_embed", vocab, hidden)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense embed_tokens");

        // The packed table reproduces the affine grid (bit-exact repack).
        let grid_t = Tensor::from_vec(grid, (vocab, hidden), &dev)?;
        let dev_max = (packed.weight().sub(&grid_t)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "packed embed table deviates from the affine grid"
        );

        Ok(())
    }

    /// `guard_no_scales` errors loudly when a `.scales` sibling appears where a dense weight is expected
    /// (a VAE / audio-VAE / vocoder conv), and loads the dense weight cleanly otherwise. This is the
    /// guard the story requires so a tier that ever packs a conv doesn't silently load u32-code garbage.
    #[test]
    fn guard_no_scales_errors_on_packed_conv() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (wq, s, b, _grid) = q4_packed(16, 64);

        // A conv leaf that (wrongly) carries a `.scales` sibling.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("conv_in.conv.weight".into(), wq);
        map.insert("conv_in.conv.scales".into(), s);
        map.insert("conv_in.conv.biases".into(), b);
        // A clean dense conv leaf (no `.scales`).
        map.insert(
            "conv_out.conv.weight".into(),
            Tensor::randn(0f32, 1f32, (8, 8), &dev)?,
        );
        let tmp = tmp.path().join("sc9417_guard.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let err = guard_no_scales(&vb, "conv_in.conv", DType::F32);
        assert!(
            err.is_err(),
            "guard must error on a `.scales` sibling where a dense conv is expected"
        );
        // The clean dense leaf loads fine.
        let ok = guard_no_scales(&vb, "conv_out.conv", DType::F32)?;
        assert_eq!(ok.dims2()?, (8, 8));

        Ok(())
    }
}
