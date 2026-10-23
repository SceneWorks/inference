//! SDXL iOS execution quanta (sc-26270): the labels and the COUNTED scratch geometry every guarded
//! SDXL quantum is priced with.
//!
//! Every price is `live activations + kernel transients`, counted from the ops of that quantum:
//!
//! * **Activations** are counted from the code at each phase of the quantum (which tensors of which
//!   shape are alive together), read off the LOADED tensors (conv weight shapes, projection base
//!   shapes) and the request's actual activation shapes and dtype, never off production constants.
//! * **Kernel transients** are what an MLX kernel allocates internally for one call, modelled per
//!   backend from the pinned MLX source ([`Backend`]): a 3x3 convolution on Metal runs Winograd
//!   (padded input + transformed input/output/filter) when MLX's dispatch selects it, and copies a
//!   non-row-contiguous weight; on the CPU it is an explicit GEMM over an f32 im2col. Fused SDPA
//!   materializes no score tensor on Metal for head dims 64/80/128 (the SDXL U-Net and CLIP); the
//!   CPU always runs the unfused fallback (scaled q, scores, softmax).
//!
//! The device prices with [`Backend::Metal`]; each counted site is proven by a CPU test that
//! measures MLX's peak for a small shape and checks the SAME activation counts with the CPU's kernel
//! transients (`quantum_tests::proofs`). The guards themselves are `mlx_gen::ios_quantum::{run,
//! before, finish}`, which are pass-throughs whenever [`mlx_gen::ios_quantum::enabled`] is false
//! (macOS, Linux, the simulator), so nothing here can change host numerics or ordering.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::ios_quantum::{
    attention_block_bytes, scores_bytes, tensor_bytes, AttentionBlockGeometry, Scores,
};
use mlx_gen::Result;
// The AutoencoderKL convolution / GroupNorm / tiled-stage kernel accounting is shared with every
// diffusers VAE on iOS (sc-26433): it lives in `mlx_gen::ios_quantum` and is re-exported here so
// this crate's call sites and proofs keep their paths.
pub(crate) use mlx_gen::ios_quantum::{
    bytes, conv_channels, decoded_to_image_bytes, downsample_stage_bytes, dtype_bytes,
    element_bytes, elements, group_norm_buffers, norm_conv_bytes, norm_promotion_units, page_slack,
    resnet_bytes_from_metadata, single_conv_bytes, tile_bytes, tiled_stage_bytes,
    tiled_stage_f32_accum, vae_attention_bytes, Backend, ConvCall, TileSource,
    VAE_ATTENTION_SCORE_ELEMENTS,
};
#[cfg(test)]
pub(crate) use mlx_gen::ios_quantum::{
    downsample_tile_bytes, resnet_bytes, GROUP_NORM_BUFFERS, PAGE_BYTES,
};
// Additive VAE metadata bindings; current native TileSource/callers remain untouched.
pub(crate) use mlx_gen::ios_quantum::{
    decoded_to_image_bytes_from_metadata, downsample_tile_bytes_from_metadata,
    tile_bytes_from_metadata, tiled_stage_f32_accum_from_metadata,
};
use mlx_rs::{Array, Dtype};

#[cfg(test)]
pub(crate) fn trace_layout(label: &str, a: &Array) {
    use std::collections::HashSet;
    thread_local! { static SEEN: std::cell::RefCell<HashSet<String>> = Default::default(); }
    let signature = format!(
        "{label}: shape={:?} dtype={:?} strides={:?}",
        a.shape(),
        a.dtype(),
        a.strides()
    );
    SEEN.with(|seen| {
        if seen.borrow_mut().insert(signature.clone()) {
            eprintln!("LAYOUT {signature}");
        }
    });
}

fn u(v: i32) -> u64 {
    v.max(0) as u64
}

thread_local! {
    static PRICING_BACKEND: std::cell::Cell<Option<Backend>> = const { std::cell::Cell::new(None) };
}

/// The backend every counted SDXL quantum is priced for: the running device's
/// ([`Backend::current`]), except inside [`with_backend`], where header-only pricing
/// ([`crate::ios_pricing`]) prices the device's Metal kernels from any host.
pub(crate) fn backend() -> Backend {
    PRICING_BACKEND
        .with(std::cell::Cell::get)
        .unwrap_or_else(Backend::current)
}

/// Run `body` with every [`backend`] call answering `backend` (restored on return or unwind).
pub(crate) fn with_backend<T>(backend: Backend, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Backend>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRICING_BACKEND.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(PRICING_BACKEND.with(|b| b.replace(Some(backend))));
    body()
}

/// One forward module's output signature in a header-only pricing walk: `(module path, shape,
/// dtype)`. Recorded only inside [`with_module_trace`]; elsewhere [`trace_module`] is inert.
pub(crate) type ModuleOutput = (String, Vec<i32>, Dtype);

thread_local! {
    static MODULE_TRACE: std::cell::RefCell<Option<Vec<ModuleOutput>>> =
        const { std::cell::RefCell::new(None) };
}

/// Record a U-Net/VAE module output while a trace is active (pricing only, never native code).
pub(crate) fn trace_module(path: &str, output: &mlx_gen::adapters::facts::TensorFacts) {
    MODULE_TRACE.with(|trace| {
        if let Some(trace) = trace.borrow_mut().as_mut() {
            trace.push((path.to_owned(), output.shape.clone(), output.dtype));
        }
    });
}

/// Run `body` recording every [`trace_module`] call; the previous trace state is restored.
pub(crate) fn with_module_trace<T>(body: impl FnOnce() -> T) -> (T, Vec<ModuleOutput>) {
    struct Restore(Option<Vec<ModuleOutput>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODULE_TRACE.with(|trace| *trace.borrow_mut() = self.0.take());
        }
    }
    let restore = Restore(MODULE_TRACE.with(|trace| trace.borrow_mut().replace(Vec::new())));
    let result = body();
    let recorded = MODULE_TRACE.with(|trace| trace.borrow_mut().take().unwrap_or_default());
    drop(restore);
    (result, recorded)
}

/// Run `body` without recording (ControlNet branches reuse U-Net module paths).
pub(crate) fn without_module_trace<T>(body: impl FnOnce() -> T) -> T {
    let saved = MODULE_TRACE.with(|trace| trace.borrow_mut().take());
    struct Restore(Option<Vec<ModuleOutput>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MODULE_TRACE.with(|trace| *trace.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(saved);
    body()
}

/// The counted quanta one header-only pricing walk checks, in execution order: `(label, scratch)`.
pub(crate) type Quanta = Vec<(&'static str, u64)>;

/// `a` promoted with every array's dtype, as MLX promotes an elementwise chain over them.
#[cfg(test)]
pub(crate) fn promote_all(a: Dtype, arrays: &[&Array]) -> Dtype {
    arrays
        .iter()
        .fold(a, |d, x| Dtype::from_promoting_types(d, x.dtype()))
}

/// A shape-and-dtype stand-in for an activation in header-only pricing: a lazy MLX array that is
/// never evaluated, so it carries the `shape()` / `dtype()` every scratch formula reads and no
/// payload.
#[cfg(test)]
pub(crate) fn shaped(shape: &[i32], dtype: Dtype) -> mlx_gen::Result<Array> {
    Ok(mlx_rs::ops::zeros_dtype(shape, dtype)?)
}

/// U-Net stem: Kolors context projection, timestep/`text_time` embedding and `conv_in`.
/// Prefix of the native-order shape checks the metadata CLIP/VAE forwards reproduce.
pub(crate) const NATIVE_SHAPE_CHECK: &str = "sdxl native op shape validation: ";

/// The exact messages of metadata ops that reproduce a NATIVE op's own shape validation:
/// matmul/broadcast/add, convolution, reshape, take and quantize alignment/rank. A world that
/// fails one of these fails natively at that op, so its admitted prefix is a real native ledger.
/// Every other pricing error (Unsupported, missing tensors, pricer invariants) is not recoverable.
const NATIVE_SHAPE_FAILURES: &[&str] = &[
    NATIVE_SHAPE_CHECK,
    "sdxl adapter operands do not broadcast",
    "sdxl adapter matmul operand is scalar",
    "sdxl adapter matmul inner dimensions differ",
    "sdxl load-time affine input is not group-aligned",
    "sdxl selected convolution input/weight axes disagree",
    "sdxl metadata reshape element counts differ",
    "sdxl metadata take row index out of bounds",
    "sdxl resnet skip concat axes disagree",
    "sdxl text pack input is not group-aligned",
    "sdxl text pack requires at least two dimensions",
];

pub(crate) fn is_native_shape_failure(error: &mlx_gen::Error) -> bool {
    matches!(error, mlx_gen::Error::Msg(message)
        if NATIVE_SHAPE_FAILURES.iter().any(|known| message.starts_with(known)))
}

/// A native-order shape check the metadata forward reproduces (see [`NATIVE_SHAPE_CHECK`]).
pub(crate) fn native_shape_failure(detail: &str) -> mlx_gen::Error {
    mlx_gen::Error::Msg(format!("{NATIVE_SHAPE_CHECK}{detail}"))
}

/// Whether `a` and `b` broadcast under the native rule (trailing axes equal or one of them 1).
pub(crate) fn broadcasts(a: &[i32], b: &[i32]) -> bool {
    a.iter()
        .rev()
        .zip(b.iter().rev())
        .all(|(&x, &y)| x == y || x == 1 || y == 1)
}

/// Native dense GroupNorm broadcasts its affine operands; GlobalGroupNorm (tiled VAE)
/// requires exact channel vectors before the tiled convolution starts.
pub(crate) fn validate_group_norm(
    shape: &[i32],
    weight: &mlx_gen::adapters::facts::TensorFacts,
    bias: &mlx_gen::adapters::facts::TensorFacts,
    global: bool,
) -> Result<()> {
    let Some(&channels) = shape.last() else {
        return Err(native_shape_failure("GroupNorm input is scalar"));
    };
    if shape.len() < 2 || channels % 32 != 0 {
        return Err(native_shape_failure(
            "GroupNorm channel count is not divisible by groups",
        ));
    }
    for operand in [weight, bias] {
        if (global && operand.shape != [channels]) || !broadcasts(shape, &operand.shape) {
            return Err(native_shape_failure(
                "GroupNorm affine operands differ from its channels",
            ));
        }
    }
    Ok(())
}

pub(crate) const UNET_STEM: &str = "SDXL UNet stem";
/// One U-Net / ControlNet `ResnetBlock2D` (including its up-path skip concatenation).
pub(crate) const UNET_RESNET: &str = "SDXL UNet resnet";
/// A U-Net / ControlNet stride-2 downsample convolution.
pub(crate) const UNET_DOWNSAMPLE: &str = "SDXL UNet downsample";
/// A U-Net nearest-2x upsample plus its convolution.
pub(crate) const UNET_UPSAMPLE: &str = "SDXL UNet upsample";
/// A `Transformer2D`'s GroupNorm + `proj_in`.
pub(crate) const TRANSFORMER_PROJ_IN: &str = "SDXL transformer proj_in";
/// One spatial-transformer block (self-attn, cross-attn [+ IP branch], GEGLU FFN).
pub(crate) const TRANSFORMER_BLOCK: &str = "SDXL transformer block";
/// A `Transformer2D`'s `proj_out` + residual.
pub(crate) const TRANSFORMER_PROJ_OUT: &str = "SDXL transformer proj_out";
/// The ControlNet down residuals added into the U-Net skip stack.
pub(crate) const CONTROL_DOWN_INJECTION: &str = "SDXL ControlNet down injection";
/// The ControlNet mid residual added into the U-Net mid output.
pub(crate) const CONTROL_MID_INJECTION: &str = "SDXL ControlNet mid injection";
/// U-Net output GroupNorm + SiLU + `conv_out`.
pub(crate) const UNET_CONV_OUT: &str = "SDXL UNet conv_out";
/// ControlNet stem: context projection, embeddings, `conv_in` + conditioning embedding.
pub(crate) const CONTROL_STEM: &str = "SDXL ControlNet stem";
/// One ControlNet 1x1 zero-convolution head (down or mid), scaled.
pub(crate) const CONTROL_ZERO_CONV: &str = "SDXL ControlNet zero-conv";
/// One convolution of the ControlNet conditioning embedding.
pub(crate) const CONTROL_COND_CONV: &str = "SDXL ControlNet cond embedding";
/// CLIP text token + position embedding.
pub(crate) const CLIP_EMBEDDINGS: &str = "SDXL CLIP embeddings";
/// One CLIP text encoder layer.
pub(crate) const CLIP_LAYER: &str = "SDXL CLIP layer";
/// CLIP text final LayerNorm + pooled (projected) EOS token.
pub(crate) const CLIP_POOLED: &str = "SDXL CLIP final norm";
/// CLIP vision patch/class/position embedding + `pre_layrnorm`.
pub(crate) const VISION_EMBEDDINGS: &str = "SDXL CLIP vision embeddings";
/// One CLIP vision encoder layer.
pub(crate) const VISION_LAYER: &str = "SDXL CLIP vision layer";
/// IP-Adapter Resampler `proj_in`.
pub(crate) const RESAMPLER_PROJ_IN: &str = "SDXL IP resampler proj_in";
/// One IP-Adapter Resampler (perceiver attention + FFN) layer.
pub(crate) const RESAMPLER_LAYER: &str = "SDXL IP resampler layer";
/// IP-Adapter Resampler `proj_out` + `norm_out`.
pub(crate) const RESAMPLER_PROJ_OUT: &str = "SDXL IP resampler proj_out";
/// The IP image-prompt tokens' CFG pair: the zeros uncond row and the concatenation (epic 16774:
/// the last conditioning quantum before a staged request sheds the image encoder).
pub(crate) const IP_CFG_TOKENS: &str = "SDXL IP CFG tokens";
/// VAE decode de-normalization, `post_quant_conv` and decoder `conv_in`.
pub(crate) const VAE_DECODE_IN: &str = "SDXL VAE decode conv_in";
/// VAE encoder `conv_in`.
pub(crate) const VAE_ENCODE_IN: &str = "SDXL VAE encode conv_in";
/// One VAE (temb-free) resnet.
pub(crate) const VAE_RESNET: &str = "SDXL VAE resnet";
/// The VAE mid-block single-head attention.
pub(crate) const VAE_ATTENTION: &str = "SDXL VAE attention";
/// A VAE nearest-2x upsample plus its convolution.
pub(crate) const VAE_UPSAMPLE: &str = "SDXL VAE upsample";
/// A VAE asymmetric-pad stride-2 downsample convolution.
pub(crate) const VAE_DOWNSAMPLE: &str = "SDXL VAE downsample";
/// VAE decoder output GroupNorm + SiLU + `conv_out` (+ output cast).
pub(crate) const VAE_DECODE_OUT: &str = "SDXL VAE decode conv_out";
/// VAE encoder output GroupNorm + SiLU + `conv_out` + `quant_conv` + mean.
pub(crate) const VAE_ENCODE_OUT: &str = "SDXL VAE encode conv_out";
/// The decoded image's clamp / scale / round to 8-bit pixel values.
pub(crate) const DECODED_TO_IMAGE: &str = "SDXL decoded to image";
/// One denoise step's CFG combine + sampler update (+ inpaint blend), materialized.
pub(crate) const SAMPLER_STEP: &str = "SDXL sampler step";
/// One halo-expanded tile of a layer-wise tiled VAE convolution, checked before its graph exists.
pub(crate) const VAE_TILE: &str = "SDXL VAE tile";
/// The assembled output of one tiled VAE layer (resnet / upsample / output convolution).
pub(crate) const VAE_TILED_STAGE: &str = "SDXL VAE tiled stage";
/// One VAE ENCODER (temb-free) resnet — the encode twin of [`VAE_RESNET`], labelled apart so a
/// device capture attributes it to the encode phase ([`crate::ios_pricing::ios_quantum_phase`]).
pub(crate) const VAE_ENCODE_RESNET: &str = "SDXL VAE encode resnet";
/// The VAE ENCODER mid-block attention (the encode twin of [`VAE_ATTENTION`]).
pub(crate) const VAE_ENCODE_ATTENTION: &str = "SDXL VAE encode attention";
/// One halo-expanded tile of a bounded-encode convolution (the encode twin of [`VAE_TILE`]).
pub(crate) const VAE_ENCODE_TILE: &str = "SDXL VAE encode tile";
/// One assembled bounded-encode layer (the encode twin of [`VAE_TILED_STAGE`]).
pub(crate) const VAE_ENCODE_TILED_STAGE: &str = "SDXL VAE encode tiled stage";

/// The labels one VAE pass (encode or decode) checks its shared quanta under. The encoder and the
/// decoder run the same resnet / attention / tiled-convolution code, so the pass names its labels:
/// a label must say which phase it runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VaeLabels {
    pub resnet: &'static str,
    pub attention: &'static str,
    pub tiled_stage: &'static str,
    pub tile: &'static str,
}

/// The decode pass's labels.
pub(crate) const VAE_DECODE_LABELS: VaeLabels = VaeLabels {
    resnet: VAE_RESNET,
    attention: VAE_ATTENTION,
    tiled_stage: VAE_TILED_STAGE,
    tile: VAE_TILE,
};

/// The encode pass's labels.
pub(crate) const VAE_ENCODE_LABELS: VaeLabels = VaeLabels {
    resnet: VAE_ENCODE_RESNET,
    attention: VAE_ENCODE_ATTENTION,
    tiled_stage: VAE_ENCODE_TILED_STAGE,
    tile: VAE_ENCODE_TILE,
};

/// Component load floors (`ios_quantum::before` over a header-only floor before construction).
pub(crate) const TEXT_ENCODER_1_LOAD: &str = "SDXL text_encoder load";
pub(crate) const TEXT_ENCODER_2_LOAD: &str = "SDXL text_encoder_2 load";
pub(crate) const UNET_LOAD: &str = "SDXL UNet load";
pub(crate) const VAE_LOAD: &str = "SDXL VAE load";
pub(crate) const VAE_FP16_FIX_LOAD: &str = "SDXL fp16-fix VAE load";
pub(crate) const CONTROLNET_LOAD: &str = "SDXL ControlNet load";
pub(crate) const IP_IMAGE_ENCODER_LOAD: &str = "SDXL IP image encoder load";
pub(crate) const IP_ADAPTER_LOAD: &str = "SDXL IP adapter load";
/// One projection's load-time Q4/Q8 pack (`quant::quantize_linear`, epic 16774): a ControlNet
/// branch, the installed IP K/V pairs, or a dense U-Net / text encoder at a packed tier.
pub(crate) const LOAD_QUANTIZE: &str = "SDXL load-time quantize";
/// The fused single-file (LDM) checkpoint's phase loads.
pub(crate) const LDM_TEXT_LOAD: &str = "SDXL LDM text load";
pub(crate) const LDM_HEAVY_LOAD: &str = "SDXL LDM UNet+VAE load";

/// One adapter (LoRA / LoKr / LoHa) file's read, checked against its header floor before the read.
pub(crate) const ADAPTER_LOAD: &str = "SDXL adapter load";
/// One adapter target's delta reconstruction + merge into the base (or, on a packed base, the
/// install of its forward-time residual).
pub(crate) const ADAPTER_MERGE: &str = "SDXL adapter merge";

/// Borrowed-operator metadata for one actual Linear. Header-selected installs update these
/// existing checked facts; the native pricing model is never filled with invented adapter Arrays.
/// Declared tensor extents are not allocation identities or a physical-release certificate.
#[derive(Clone, Debug)]
pub(crate) struct LinearWork {
    pub base: mlx_gen::adapters::LinearFacts,
    pub weights: Vec<mlx_gen::adapters::facts::TensorFacts>,
    pub bias: Option<mlx_gen::adapters::facts::TensorFacts>,
    pub residuals: Vec<mlx_gen::adapters::facts::AdapterFacts>,
    pub group_size: i32,
    pub bits: Option<i32>,
}

impl LinearWork {
    pub(crate) fn of(linear: &AdaptableLinear) -> Self {
        use mlx_gen::adapters::facts::{AdapterFacts, TensorFacts};
        let mut base = mlx_gen::adapters::LinearFacts::of(linear);
        let (weights, bias, group_size, bits) =
            if let Some((w, scales, offsets, bias, group, bits)) = linear.quantized_params() {
                // Keep the actual packed operand geometry, including its logical input width.
                let mut logical = scales.shape().to_vec();
                if let Some(input) = logical.last_mut() {
                    *input = input.saturating_mul(group);
                }
                base.base_shape = logical;
                (
                    vec![
                        TensorFacts::of(w),
                        TensorFacts::of(scales),
                        TensorFacts::of(offsets),
                    ],
                    bias.map(TensorFacts::of),
                    group,
                    Some(bits),
                )
            } else {
                let (w, bias) = linear
                    .dense_weight()
                    .expect("LinearBase is dense or packed");
                (
                    vec![TensorFacts::of(w)],
                    bias.map(TensorFacts::of),
                    crate::quant::GROUP_SIZE,
                    None,
                )
            };
        Self {
            base,
            weights,
            bias,
            residuals: linear.adapters().iter().map(AdapterFacts::of).collect(),
            group_size,
            bits,
        }
    }

    /// The selected AdaptableLinear path passes x unchanged to the Rust primitive.
    /// Native affine qmm promotes scales + quantization offsets, then x; the ordinary
    /// output bias is added afterward and cannot affect this compute dtype.
    fn packed_compute_dtype(&self, input: Dtype) -> Dtype {
        let affine = Dtype::from_promoting_types(self.weights[1].dtype, self.weights[2].dtype);
        Dtype::from_promoting_types(input, affine)
    }

    pub(crate) fn output_dtype(&self, input: Dtype) -> Dtype {
        if self.bits.is_some() {
            let compute = self.packed_compute_dtype(input);
            return self.bias.as_ref().map_or(compute, |bias| {
                Dtype::from_promoting_types(compute, bias.dtype)
            });
        }
        let weight = self.weights[usize::from(self.bits.is_some())].dtype;
        let out = Dtype::from_promoting_types(input, weight);
        self.bias
            .as_ref()
            .map_or(out, |bias| Dtype::from_promoting_types(out, bias.dtype))
    }

    pub(crate) fn output_shape(&self, input: &[i32]) -> Result<Vec<i32>> {
        let mut shape = self.base_output_shape(input)?;
        for residual in self.residuals.iter().filter(|a| !a.disabled()) {
            shape = crate::adapters::broadcast_shape(
                &shape,
                &crate::adapters::adapter_residual_shape(residual, input)?,
            )?;
        }
        Ok(shape)
    }

    fn base_output_shape(&self, input: &[i32]) -> Result<Vec<i32>> {
        let mut transposed = self.base.base_shape.clone();
        if self.bits.is_some() {
            if transposed.len() < 2 {
                return Err(mlx_gen::Error::Msg(
                    "sdxl packed Linear rank is invalid".into(),
                ));
            }
            let n = transposed.len();
            transposed.swap(n - 1, n - 2);
        } else {
            transposed.reverse();
        }
        let shape = crate::adapters::matmul_shape(input, &transposed)?;
        match &self.bias {
            Some(bias) => crate::adapters::broadcast_shape(&shape, &bias.shape),
            None => Ok(shape),
        }
    }

    pub(crate) fn output_width(&self) -> u64 {
        let axis = if self.bits.is_some() {
            self.base.base_shape.len().saturating_sub(2)
        } else {
            0
        };
        self.base.base_shape.get(axis).copied().unwrap_or(0).max(0) as u64
    }

    pub(crate) fn operand_bytes(&self, shape: &[i32], input: Dtype) -> u64 {
        if self.bits.is_some() {
            return self.packed_operand_bytes(shape, input);
        }
        let output = self.output_dtype(input);
        let width = dtype_bytes(output);
        let activation = if input != output {
            shape.iter().fold(width, |n, &d| n.saturating_mul(u(d)))
        } else {
            0
        };
        let weight = if self.bits.is_none() && self.weights[0].dtype != output {
            self.weights[0].elements().saturating_mul(width)
        } else {
            0
        };
        let bias = self.bias.as_ref().map_or(0, |bias| {
            if bias.dtype != output {
                bias.elements().saturating_mul(width)
            } else {
                0
            }
        });
        let mut accumulated = self.base_output_shape(shape).ok();
        self.residuals.iter().fold(
            activation.saturating_add(weight).saturating_add(bias),
            |n, adapter| {
                if adapter.disabled() {
                    return n;
                }
                accumulated = accumulated.as_ref().and_then(|host| {
                    crate::adapters::adapter_residual_shape(adapter, shape)
                        .ok()
                        .and_then(|residual| crate::adapters::broadcast_shape(host, &residual).ok())
                });
                let add_elements = accumulated
                    .as_ref()
                    .map_or(0, |s| s.iter().fold(1u64, |n, &d| n.saturating_mul(u(d))));
                // Invalid native operands keep their original operator error priority; this numeric
                // guard helper adds no new runtime validation. Valid outputs carry the whole ordered
                // host-add geometry, including earlier residuals, not an independent residual maximum.
                n.saturating_add(crate::adapters::adapter_forward_scratch_at_output(
                    adapter,
                    shape,
                    input,
                    output,
                    add_elements,
                ))
            },
        )
    }

    fn packed_operand_bytes(&self, shape: &[i32], input: Dtype) -> u64 {
        let compute = self.packed_compute_dtype(input);
        let output = self.output_dtype(input);
        let compute_width = dtype_bytes(compute);
        let output_width = dtype_bytes(output);
        let activation = if input != compute {
            shape
                .iter()
                .fold(compute_width, |n, &d| n.saturating_mul(u(d)))
        } else {
            0
        };
        // Compare dtype identities, not widths. Native qmm casts both affine operands,
        // never its packed U32 codes. There is no caller input-to-scales precast.
        let affine = self.weights[1..3].iter().fold(0u64, |n, tensor| {
            n.saturating_add(if tensor.dtype != compute {
                tensor.elements().saturating_mul(compute_width)
            } else {
                0
            })
        });
        let bias = self.bias.as_ref().map_or(0, |bias| {
            if bias.dtype != output {
                bias.elements().saturating_mul(output_width)
            } else {
                0
            }
        });
        // A widening ordinary bias add casts the qmm result, BEFORE bias broadcasting.
        // Keep invalid native operands' error priority: this guard adds no validation.
        let post_bias = if compute != output {
            let mut transposed = self.base.base_shape.clone();
            if transposed.len() < 2 {
                0
            } else {
                let n = transposed.len();
                transposed.swap(n - 1, n - 2);
                crate::adapters::matmul_shape(shape, &transposed)
                    .ok()
                    .map_or(0, |dims| {
                        dims.iter()
                            .fold(output_width, |n, &d| n.saturating_mul(u(d)))
                    })
            }
        } else {
            0
        };
        let base_casts = activation
            .saturating_add(affine)
            .saturating_add(bias)
            .saturating_add(post_bias);
        let mut accumulated = self.base_output_shape(shape).ok();
        self.residuals.iter().fold(base_casts, |n, adapter| {
            if adapter.disabled() {
                return n;
            }
            accumulated = accumulated.as_ref().and_then(|host| {
                crate::adapters::adapter_residual_shape(adapter, shape)
                    .ok()
                    .and_then(|residual| crate::adapters::broadcast_shape(host, &residual).ok())
            });
            let add_elements = accumulated
                .as_ref()
                .map_or(0, |s| s.iter().fold(1u64, |n, &d| n.saturating_mul(u(d))));
            n.saturating_add(crate::adapters::adapter_forward_scratch_at_output(
                adapter,
                shape,
                input,
                output,
                add_elements,
            ))
        })
    }

    pub(crate) fn install(&mut self, work: &crate::adapters::AdapterInstallWork) -> Result<()> {
        use crate::adapters::AdapterInstallStorage as S;
        match work.storage {
            S::DenseLinear if self.bits.is_none() => {
                let weight = work.installed.first().ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl selected dense install has no declared output".into())
                })?;
                self.base.base_shape = weight.shape.clone();
                self.weights = vec![weight.clone()];
            }
            S::PackedLora | S::PackedDelta | S::StructuredLokr if self.bits.is_some() => {
                self.residuals.push(work.residual().ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "sdxl selected packed install has incomplete declared operands".into(),
                    )
                })?);
                self.base.adapter_count = self.residuals.len();
                self.base.has_live_adapters = self.residuals.iter().any(|a| !a.disabled());
            }
            _ => {
                return Err(mlx_gen::Error::Msg(
                    "sdxl selected install disagrees with actual target storage".into(),
                ))
            }
        }
        Ok(())
    }

    /// The SAME post-adapter base-only pack decision exposed by the actual private quant walk.
    /// No tensor is manufactured or evaluated. Existing stored packs remain untouched; residuals
    /// and their disabled owners are deliberately unchanged, exactly as AdaptableLinear::quantize.
    /// The native load-time pack guard's budget over this base, admitted BEFORE the quantize
    /// op validates rank/group alignment (`quant::quantize_linear`).
    pub(crate) fn pack_scratch(&self, bits: i32) -> u64 {
        let ordinary_bias = self
            .bias
            .as_ref()
            .map_or(0, |b| b.elements().saturating_mul(2));
        crate::quant::quantize_bytes_from_metadata(self.weights[0].elements(), ordinary_bias, bits)
    }

    pub(crate) fn pack(&mut self, bits: i32) -> Result<u64> {
        use mlx_gen::adapters::facts::TensorFacts;
        if self.bits.is_some() {
            return Ok(0);
        }
        let shape = &self.base.base_shape;
        if !matches!(bits, 4 | 8) {
            return Err(mlx_gen::Error::Msg(
                "sdxl load-time affine pack metadata is invalid".into(),
            ));
        }
        if shape.len() < 2 {
            return Err(native_shape_failure("quantize weight rank is below two"));
        }
        let input = *shape.last().unwrap();
        if input % crate::quant::GROUP_SIZE != 0 {
            return Err(mlx_gen::Error::Msg(
                "sdxl load-time affine input is not group-aligned".into(),
            ));
        }
        let mut codes = shape.clone();
        *codes.last_mut().unwrap() = input
            .checked_mul(bits)
            .and_then(|n| n.checked_div(32))
            .ok_or_else(|| mlx_gen::Error::Msg("sdxl packed column overflow".into()))?;
        let mut scales = shape.clone();
        *scales.last_mut().unwrap() = input / crate::quant::GROUP_SIZE;
        let ordinary_bias = self
            .bias
            .as_ref()
            .map_or(0, |b| b.elements().saturating_mul(2));
        let scratch = crate::quant::quantize_bytes_from_metadata(
            self.weights[0].elements(),
            ordinary_bias,
            bits,
        );
        self.weights = vec![
            TensorFacts {
                shape: codes,
                dtype: Dtype::Uint32,
                strides: Vec::new(),
            },
            TensorFacts {
                shape: scales.clone(),
                dtype: Dtype::Bfloat16,
                strides: Vec::new(),
            },
            TensorFacts {
                shape: scales,
                dtype: Dtype::Bfloat16,
                strides: Vec::new(),
            },
        ];
        if let Some(bias) = &mut self.bias {
            bias.dtype = Dtype::Bfloat16;
            bias.strides.clear();
        }
        self.bits = Some(bits);
        self.group_size = crate::quant::GROUP_SIZE;
        self.base.is_quantized = true;
        Ok(scratch)
    }

    pub(crate) fn declared_bytes(&self) -> u64 {
        let payload = |a: &mlx_gen::adapters::facts::TensorFacts| {
            mlx_gen::ios_quantum::allocated_bytes(a.bytes())
        };
        let base = self
            .weights
            .iter()
            .fold(0u64, |n, a| n.saturating_add(payload(a)))
            .saturating_add(self.bias.as_ref().map_or(0, payload));
        self.residuals.iter().fold(base, |n, a| {
            a.tensors()
                .iter()
                .fold(n, |n, tensor| n.saturating_add(payload(tensor)))
        })
    }
}

/// Explicit per-load projection facts. Keys come from the actual private quantization walk;
/// lookup never falls back to a dense projection when a required final effect is missing.
#[derive(Default)]
pub(crate) struct LinearEffects(
    pub std::collections::BTreeMap<String, LinearWork>,
    /// Changed dense convolution weights in their ACTUAL installed OHWI layout. Unchanged weights
    /// are borrowed from the native constructor; this map never contains synthetic Arrays.
    pub std::collections::BTreeMap<String, mlx_gen::adapters::facts::TensorFacts>,
);

#[derive(Clone)]
pub(crate) struct EffectScope<'a> {
    effects: Option<&'a LinearEffects>,
    prefix: String,
}

impl<'a> EffectScope<'a> {
    pub(crate) fn native() -> Self {
        Self {
            effects: None,
            prefix: String::new(),
        }
    }
    pub(crate) fn root(effects: &'a LinearEffects) -> Self {
        Self {
            effects: Some(effects),
            prefix: String::new(),
        }
    }
    /// This scope's module path (`""` at the root).
    pub(crate) fn path(&self) -> &str {
        &self.prefix
    }
    pub(crate) fn child(&self, name: &str) -> Self {
        Self {
            effects: self.effects,
            prefix: crate::quant::child_path(&self.prefix, name),
        }
    }
    pub(crate) fn conv(&self, name: &str, native: &mlx_gen::adapters::AdaptableConv2d) -> ConvWork {
        let path = crate::quant::child_path(&self.prefix, name);
        let weight = self
            .effects
            .and_then(|effects| effects.1.get(&path))
            .cloned()
            .unwrap_or_else(|| mlx_gen::adapters::facts::TensorFacts::of(native.weight()));
        ConvWork {
            weight,
            bias: native.bias().map(mlx_gen::adapters::facts::TensorFacts::of),
        }
    }
    pub(crate) fn linear(&self, name: &str, native: &AdaptableLinear) -> Result<LinearWork> {
        let path = crate::quant::child_path(&self.prefix, name);
        match self.effects {
            Some(effects) => effects.0.get(&path).cloned().ok_or_else(|| {
                mlx_gen::Error::Msg(format!("sdxl missing final projection effect: {path}"))
            }),
            None => Ok(LinearWork::of(native)),
        }
    }
}

/// The dtype `x`-typed activations leave `linear` at: MLX promotes a dense matmul to
/// `result_type(x, weight)` and a quantized one to `result_type(x, scales, quantization_offsets)`
/// before its ordinary output-bias add, so a pre-quantized
/// SceneWorks snapshot's BF16 `.scales` (every packed U-Net / CLIP projection) turn f16 activations
/// into f32 (measured: `quantum_tests::proofs::f16_transformer_quanta_track_their_measured_peaks`),
/// and everything downstream of that projection — residual sums, norms, convolutions — runs at f32.
#[cfg(test)]
pub(crate) fn linear_output_dtype(linear: &AdaptableLinear, x: Dtype) -> Dtype {
    LinearWork::of(linear).output_dtype(x)
}

/// Convolution's composed bias add: its promoted bias vector and, if the bias widens the
/// convolution result, the distinct wider output. The internal result cast may donate to add.
#[cfg(test)]
pub(crate) fn conv_bias_bytes(
    input: Dtype,
    weight: &Array,
    bias: Option<&Array>,
    output_elements: u64,
) -> u64 {
    ConvWork {
        weight: mlx_gen::adapters::facts::TensorFacts::of(weight),
        bias: bias.map(mlx_gen::adapters::facts::TensorFacts::of),
    }
    .bias_bytes(input, output_elements)
}

/// Dense convolution facts in native OHWI order, at their PRECAST element widths. Source quant
/// bits, total nbytes and a promoted activation width are never substituted for the weight width.
#[derive(Clone)]
pub(crate) struct ConvWork {
    pub weight: mlx_gen::adapters::facts::TensorFacts,
    pub bias: Option<mlx_gen::adapters::facts::TensorFacts>,
}

impl ConvWork {
    pub(crate) fn call(&self, x: &[i32], act: u64, stride: i32, pad: i32) -> ConvCall {
        ConvCall::from_metadata(
            x,
            act,
            &self.weight.shape,
            dtype_bytes(self.weight.dtype),
            stride,
            pad,
        )
    }
    pub(crate) fn output_shape(&self, x: &[i32], stride: i32, pad: i32) -> Result<Vec<i32>> {
        if x.len() != 4 || self.weight.shape.len() != 4 || x[3] != self.weight.shape[3] {
            return Err(mlx_gen::Error::Msg(
                "sdxl selected convolution input/weight axes disagree".into(),
            ));
        }
        let axis = |n: i32, k: i32| -> Result<i32> {
            let result = (i64::from(n) + 2 * i64::from(pad) - i64::from(k)) / i64::from(stride) + 1;
            i32::try_from(result).map_err(|_| {
                mlx_gen::Error::Msg("sdxl convolution output dimension overflow".into())
            })
        };
        let shape = vec![
            x[0],
            axis(x[1], self.weight.shape[1])?,
            axis(x[2], self.weight.shape[2])?,
            self.weight.shape[0],
        ];
        match &self.bias {
            Some(bias) => crate::adapters::broadcast_shape(&shape, &bias.shape),
            None => Ok(shape),
        }
    }
    pub(crate) fn output_dtype(&self, input: Dtype) -> Dtype {
        let convolution = Dtype::from_promoting_types(input, self.weight.dtype);
        self.bias.as_ref().map_or(convolution, |b| {
            Dtype::from_promoting_types(convolution, b.dtype)
        })
    }
    pub(crate) fn bias_bytes(&self, input: Dtype, output_elements: u64) -> u64 {
        let Some(bias) = &self.bias else {
            return 0;
        };
        let convolution = Dtype::from_promoting_types(input, self.weight.dtype);
        let output = Dtype::from_promoting_types(convolution, bias.dtype);
        let width = dtype_bytes(output);
        let cast = if bias.dtype != output {
            bias.elements() * width
        } else {
            0
        };
        cast + if convolution != output {
            output_elements * width
        } else {
            0
        }
    }
}

/// Operand casts of a dense fused addmm, separate from its single output. In particular an f32
/// bias promotes BOTH half-precision matrix operands; there is no second bias-add output.
/// Packed matmul casts its unchanged x, scales and quantization offsets to their common compute
/// dtype, never its U32 codes; a widening ordinary bias add casts the qmm output separately.
#[cfg(test)]
pub(crate) fn linear_operand_bytes(linear: &AdaptableLinear, shape: &[i32], input: Dtype) -> u64 {
    #[cfg(test)]
    if let Some((w, b)) = linear.dense_weight() {
        trace_layout("addmm weight", w);
        if let Some(b) = b {
            trace_layout("addmm bias", b);
        }
        eprintln!("OPERAND addmm input shape={shape:?} dtype={input:?}");
    }
    LinearWork::of(linear).operand_bytes(shape, input)
}

/// A final affine LayerNorm's peak, excluding its already-resident input. CPU fast.cpp's
/// fallback has two competing live sets: the f32 centering/variance graph, and normalization +
/// output cast + affine epilogue. Count the larger set, with the three row statistics and the
/// promoted affine operands, rather than adding every intermediate in the graph.
#[cfg(test)]
pub(crate) fn final_norm_bytes(shape: &[i32], input: Dtype, weight: &Array, bias: &Array) -> u64 {
    #[cfg(test)]
    {
        trace_layout("final norm weight", weight);
        trace_layout("final norm bias", bias);
        eprintln!("OPERAND final norm input shape={shape:?} dtype={input:?}");
    }
    final_norm_bytes_from_metadata(
        shape,
        input,
        &mlx_gen::adapters::facts::TensorFacts::of(weight),
        &mlx_gen::adapters::facts::TensorFacts::of(bias),
    )
}

/// The identical affine-norm formula over borrowed operand metadata. Native guards above and
/// header-only CLIP pricing both consume this; metadata carries no backing or allocation identity.
pub(crate) fn final_norm_bytes_from_metadata(
    shape: &[i32],
    input: Dtype,
    weight: &mlx_gen::adapters::facts::TensorFacts,
    bias: &mlx_gen::adapters::facts::TensorFacts,
) -> u64 {
    let output =
        Dtype::from_promoting_types(Dtype::from_promoting_types(input, weight.dtype), bias.dtype);
    let count = shape.iter().fold(1u64, |n, &v| n.saturating_mul(u(v)));
    let channels = u(*shape.last().expect("LayerNorm has a channel dimension")).max(1);
    let width = dtype_bytes(output);
    let affine = [weight, bias]
        .iter()
        .map(|a| {
            if a.dtype != output {
                a.elements() * width
            } else {
                0
            }
        })
        .sum::<u64>();
    match backend() {
        Backend::Metal => count * width + affine,
        Backend::Cpu => {
            let input_width = dtype_bytes(input);
            let centered = count * (2 * 4 + u64::from(input_width < 4) * 4);
            let normalized = count * (3 * width + u64::from(input_width < 4) * 3 * 4);
            // MLX v0.32.0 fast.cpp fallback constructs two mean normalizers (ops.cpp
            // number_of_elements) and epsilon as eager F32 scalar Arrays before evaluating
            // the lazy norm graph. Count all three source owners, not a measured byte gap.
            // Some can retire before the peak; retaining all is a source-derived upper.
            let scalars = tensor_bytes(dtype_bytes(Dtype::Float32), &[3]);
            centered.max(normalized) + 3 * (count / channels) * 4 + affine + scalars
        }
    }
}

/// Control residuals are caller-owned: their compute-dtype cast is lazy at the injection
/// boundary, and cannot donate its retained storage. Price that potential pending allocation
/// separately from the promoted sum (whose internal operand cast may donate).
pub(crate) fn control_injection_bytes(hidden: &Array, residual: &Array) -> u64 {
    #[cfg(test)]
    {
        trace_layout("control injection hidden", hidden);
        trace_layout("control injection residual", residual);
    }
    control_injection_bytes_metadata(&activation_facts(hidden), &activation_facts(residual))
}

/// Deferred MultiControlNet sums are graph work, not already-held early-forward tensors. Record
/// each actual add OUTPUT at its own width, and charge that logical workspace only at the existing
/// down/mid injection barriers that evaluate it. This raw guard upper is never a subtractable
/// physical lower or a fresh-allocation/cache certificate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ControlMergeWork {
    pub down: u64,
    pub mid: u64,
}

impl ControlMergeWork {
    pub fn record_outputs(&mut self, down: &[Array], mid: &Array) -> mlx_gen::Result<()> {
        let down: Vec<_> = down.iter().map(activation_facts).collect();
        self.record_outputs_metadata(&down, &activation_facts(mid))
    }
}

/// Scalar-only observation of genuinely borrowed owners at a source scope boundary. Counts are
/// logical holder sums (aliases may overlap), explicitly NOT native backing/footprint evidence.
pub(crate) fn diagnostic_owned_tensors<'a>(
    kind: &str,
    context: u64,
    tensors: impl Iterator<Item = &'a Array>,
) {
    let (mut count, mut available, mut pending) = (0u64, 0u64, 0u64);
    for tensor in tensors {
        count += 1;
        if mlx_gen::array::is_materialized(tensor) {
            available = available.saturating_add(bytes(tensor));
        } else {
            pending = pending.saturating_add(bytes(tensor));
        }
    }
    mlx_gen::ios_quantum::diagnostic_metadata(
        "sdxl",
        kind,
        &[
            ("schema", 1),
            ("context", context),
            ("holder_count", count),
            ("available_logical_holder_bytes", available),
            ("pending_logical_holder_bytes", pending),
            ("physical_certificate", 0),
        ],
    );
}

/// Owned inputs live DURING a model forward, excluding construction-only temporaries. `None`
/// scaled dtype is an alias of caller state; `Some` is a separately owned scaled input. Curated
/// paths cast before CFG, whereas ancestral paths preserve the sampler dtype through forward_eps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ForwardInputOwnership {
    pub state: Dtype,
    pub scaled: Option<Dtype>,
    pub cast_before_cfg: bool,
}

impl ForwardInputOwnership {
    pub(crate) fn bytes(self, elements: u64, compute: Dtype, cfg_rows: u64) -> u64 {
        self.unique_allocation_requests(elements, elements, compute, cfg_rows)
            .into_iter()
            .fold(0, u64::saturating_add)
    }

    /// Unique allocation REQUEST sizes, not physical backing/allocator certificates. State may
    /// be a slice/broadcast/reshape: its source-owned FULL parent request must be supplied, not
    /// the view's logical nbytes/span. Each subsequent nonzero entry is a fresh distinct output;
    /// aliases add no entry. Round/price these individually BEFORE taking any matched delta.
    pub(crate) fn unique_allocation_requests(
        self,
        elements: u64,
        state_parent_elements: u64,
        compute: Dtype,
        cfg_rows: u64,
    ) -> [u64; 5] {
        assert!(cfg_rows > 0, "an executed forward has at least one row");
        let tensor = |dtype, rows| tensor_bytes(dtype_bytes(dtype), &[elements, rows]);
        let mut owned = [0; 5];
        owned[0] = tensor_bytes(dtype_bytes(self.state), &[state_parent_elements]);
        let mut input = self.state;
        if let Some(scaled) = self.scaled {
            owned[1] = tensor(scaled, 1);
            input = scaled;
        }
        if self.cast_before_cfg && input != compute {
            owned[2] = tensor(compute, 1);
            input = compute;
        }
        if cfg_rows > 1 {
            owned[3] = tensor(input, cfg_rows);
        }
        // forward_eps preserves this input. Only the historical pre-CFG curated cast above
        // creates a compute-dtype owner; ancestral F32 state/CFG is never narrowed on iOS.
        owned
    }

    /// Native-counter charge of every nonzero request under `charge`: the exact fresh malloc
    /// length ([`mlx_gen::ios_quantum::fresh_metal_allocation_bytes`]) as a certain lower bound,
    /// or the Metal buffer-cache reuse upper ([`metal_reuse_upper_bytes`]). A pre-scope/cached/
    /// external parent cannot use this path merely because its current view is contiguous.
    pub(crate) fn native_bytes(
        self,
        elements: u64,
        state_parent_elements: u64,
        compute: Dtype,
        cfg_rows: u64,
        page_bytes: u64,
        charge: fn(u64, u64) -> Option<u64>,
    ) -> Option<u64> {
        if cfg_rows == 0 || (elements != 0 && state_parent_elements == 0) {
            return None;
        }
        self.unique_allocation_requests(elements, state_parent_elements, compute, cfg_rows)
            .into_iter()
            .try_fold(0u64, |sum, request| {
                sum.checked_add(charge(request, page_bytes)?)
            })
    }
}

/// MLX Metal `malloc` upper on `active_memory` for one request when the buffer cache may hold
/// buffers: `malloc` aligns a request above one page to whole pages (exact length at or below a
/// page), then `BufferCache::reuse_from_cache` may return a cached buffer of length L with
/// `size <= L < min(2*size, size + 2*page)` (immutable MLX `backend/metal/allocator.cpp` malloc,
/// `backend/common/buffer_cache.h` reuse_from_cache). A zero request allocates nothing.
pub(crate) fn metal_reuse_upper_bytes(request_bytes: u64, page_bytes: u64) -> Option<u64> {
    let size = mlx_gen::ios_quantum::fresh_metal_allocation_bytes(request_bytes, page_bytes)?;
    Some(
        size.checked_mul(2)?
            .min(size.checked_add(page_bytes.checked_mul(2)?)?),
    )
}

/// A matched execution-epoch envelope, not difference-of-maxima: max(H+request)-max(H+base)
/// <= max(request-base). The final base entry describes its proved steady suffix. An empty
/// request executes no model forward, so contributes no retained-forward excess.
#[cfg(test)]
pub(crate) fn forward_input_excess(
    elements: u64,
    compute: Dtype,
    cfg_rows: u64,
    base: &[ForwardInputOwnership],
    request: &[ForwardInputOwnership],
) -> u64 {
    assert!(
        !base.is_empty(),
        "a calibration baseline needs a model epoch"
    );
    request
        .iter()
        .enumerate()
        .map(|(i, request)| {
            let base = base
                .get(i)
                .unwrap_or_else(|| base.last().expect("nonempty baseline"));
            request
                .bytes(elements, compute, cfg_rows)
                .saturating_sub(base.bytes(elements, compute, cfg_rows))
        })
        .max()
        .unwrap_or(0)
}

/// Output-row count of one projection (the `out` of its logical `[out, in]` weight).
pub(crate) fn linear_out(linear: &AdaptableLinear) -> u64 {
    linear.base_shape().first().map_or(0, |d| u(*d))
}

/// How a call site's attention scores exist on `backend`: MLX's Metal SDPA is fused for a query
/// length above 8 at head dims 64/80/128 (`sdpa_full`), or at most 8 queries at 64/96/128/256
/// (`sdpa_vector`); every other shape — and every CPU call — runs the unfused fallback, which
/// materializes the scores and (for a multi-head call, measured) a second score-sized buffer for
/// the mask add / softmax. The single-head VAE attention prices its own (donated) softmax.
pub(crate) fn scores(backend: Backend, head_dim: i32, query_rows: i32, act: u64) -> Scores {
    let fused = match backend {
        Backend::Cpu => false,
        Backend::Metal if query_rows > 8 => matches!(head_dim, 64 | 80 | 128),
        Backend::Metal => matches!(head_dim, 64 | 96 | 128 | 256),
    };
    if fused {
        Scores::Fused
    } else {
        Scores::Materialized {
            live: 2,
            element_bytes: act,
        }
    }
}

/// Buffers of one `mlx_rs::fast::layer_norm` at its input's size: Metal runs the fused kernel (its
/// output only); MLX's CPU backend runs the composed fallback — the centred input, its square and
/// the output (measured 3.03 input-sized buffers at f32). Below f32 the CPU fallback also holds f32
/// working copies ([`norm_promotion_units`]); a U-Net transformer block's measured f16 peak is
/// already covered without them (1.09-1.14x: its scores dominate), a CLIP layer's is not
/// ([`clip_layer_bytes`] adds them).
pub(crate) fn layer_norm_buffers(backend: Backend) -> u64 {
    match backend {
        Backend::Metal => 1,
        Backend::Cpu => 3,
    }
}
/// Query-sized buffers the unfused SDPA fallback adds per self-attention when `scores`
/// materializes: the scaled query (row-contiguous), and the row-contiguous copies of the head-split
/// k/v views its batched matmuls take.
fn fallback_hidden(scores: Scores) -> u64 {
    3 * u64::from(matches!(scores, Scores::Materialized { .. }))
}

/// One CLIP text / vision encoder layer over `[B, N, D]` (pre-norm attention + pre-norm MLP, both
/// residual), FFN width `ff`; `mask` when the text encoder's causal additive mask `[1, 1, N, N]` is
/// applied. On iOS the attention half is materialized before the MLP is built, so the price is
/// [`attention_block_bytes`]'s wider phase, each counted as new buffers under [`conv_sum`]'s
/// retention model (a residual sum donates its projection operand; each LayerNorm is
/// [`layer_norm_buffers`]):
/// * attention — LayerNorm, q, k, v, the head-merged SDPA output and its reshape copy, the output
///   projection [+ the fallback's], and the scores (+ the mask);
/// * MLP — the materialized attention result, LayerNorm, `fc2`; FFN-sized: `fc1` and its GELU.
#[allow(clippy::too_many_arguments)]
pub(crate) fn clip_layer_bytes(
    b: i32,
    n: i32,
    dim: u64,
    ff: u64,
    heads: i32,
    head_dim: i32,
    mask: bool,
    act: u64,
    backend: Backend,
) -> u64 {
    let s = scores(backend, head_dim, n, act);
    // A 77-token layer's small score tensors leave the CPU fallback LayerNorm's f32 working copies
    // at the layer's peak below f32 (measured at f16: 0.86x without them).
    let ln = layer_norm_buffers(backend) + norm_promotion_units(backend, act);
    let attn_hidden_live = 6 + ln + fallback_hidden(s);
    let mlp_hidden_live = 2 + ln;
    let mask_bytes = if mask {
        tensor_bytes(act, &[u(n), u(n)])
    } else {
        0
    };
    attention_block_bytes(AttentionBlockGeometry {
        batch: u(b),
        tokens: u(n),
        dim,
        heads: u(heads),
        context_tokens: 0,
        ff_hidden: ff,
        query_rows: u(n),
        element_bytes: act,
        weight_bytes: 0,
        attn_hidden_live,
        context_live: 0,
        scores: s,
        mlp_hidden_live,
        ff_live: 2,
    })
    .saturating_add(mask_bytes)
    .saturating_add(page_slack(attn_hidden_live.max(mlp_hidden_live + 2) + 3))
}

/// One U-Net spatial-transformer block over `[B, N, D]`, cross-attending to `context_tokens` keys
/// (text memory plus any IP tokens) projected at `context_batch` rows (a sequential CFG half
/// projects the pair's conditioning, then keeps its row), FFN (GEGLU half) width `ff`, `query_rows`
/// queries per SDPA call, `ip` when the IP branch runs.
///
/// On iOS the block materializes each sub-layer before building the next (`forward_ip`'s barriers),
/// so the price is its widest sub-layer, each counted as new buffers under [`conv_sum`]'s retention
/// model (a residual sum donates its projection operand; each LayerNorm is
/// [`layer_norm_buffers`]):
/// * self-attention — hidden: LayerNorm, q, k, v, the SDPA output and its head-merge copy, the
///   output projection [+ the fallback's per attention; a bounded plan's chunk outputs and their
///   concatenation]; its scores;
/// * cross-attention — hidden: the materialized self-attention result, LayerNorm, q, the SDPA
///   output and its copy, the projection [+ the IP branch's output and its scaled copy]; context:
///   k and v at `context_batch` rows, their selected rows under sequential CFG [+ the fallback's
///   k/v copies]; its scores;
/// * FFN — hidden: the materialized attention result, LayerNorm, `linear3`; FFN-sized: `linear1`,
///   `linear2`, the GELU (the GEGLU product and the final sum donate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn transformer_block_bytes(
    batch: i32,
    context_batch: i32,
    tokens: i32,
    dim: u64,
    heads: i32,
    head_dim: i32,
    context_tokens: i32,
    ff: u64,
    query_rows: i32,
    ip: bool,
    act: u64,
    backend: Backend,
) -> u64 {
    let s = scores(backend, head_dim, query_rows, act);
    let fallback = u64::from(matches!(s, Scores::Materialized { .. }));
    let chunked = 2 * u64::from(query_rows < tokens);
    let (b, cb) = (u(batch).max(1), u(context_batch).max(u(batch)));
    let selected = u64::from(cb > b);
    let ln = layer_norm_buffers(backend);
    let geometry = AttentionBlockGeometry {
        batch: b,
        tokens: u(tokens),
        dim,
        heads: u(heads),
        context_tokens: 0,
        ff_hidden: ff,
        query_rows: u(query_rows),
        element_bytes: act,
        weight_bytes: 0,
        attn_hidden_live: 6 + ln + 3 * fallback + chunked,
        context_live: 0,
        scores: s,
        mlp_hidden_live: 2 + ln,
        ff_live: 3,
    };
    // Self-attention and FFN phases.
    let self_and_ffn = attention_block_bytes(geometry);
    // Cross-attention phase: k and v at the context batch (+ their selected rows, + the fallback's
    // k/v copies), in `[batch, S, D]` units. Its scores span the `S` context keys only: priced
    // here (`attention_block_bytes` would score `N + S` keys), the API pricing the buffers.
    let context_units = (2 * cb + 2 * selected * b + 2 * fallback * b).div_ceil(b);
    let cross = attention_block_bytes(AttentionBlockGeometry {
        context_tokens: u(context_tokens),
        attn_hidden_live: 5 + ln + fallback + chunked + 2 * u64::from(ip),
        context_live: context_units,
        scores: Scores::Fused,
        mlp_hidden_live: 0,
        ff_live: 0,
        ..geometry
    })
    .saturating_add(scores_bytes(
        s,
        b,
        u(heads),
        u(query_rows),
        u(context_tokens),
    ));
    let buffers = geometry.attn_hidden_live.max(5 + ln + 2 + context_units) + 2;
    self_and_ffn.max(cross).saturating_add(page_slack(buffers))
}

/// Check one halo-expanded tile of a layer-wise tiled VAE convolution BEFORE its norm/activation/
/// conv graph is constructed, priced at the tile's extent, its `source` (the activation's width,
/// whether the crop is normalized) and `weight`'s channel counts. Called from
/// the per-tile preprocess hook of `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc*`, which evaluates
/// each output core before the next tile is built.
pub(crate) fn before_tile(
    label: &str,
    tile: &Array,
    source: TileSource,
    weight: &Array,
    f32_accum: bool,
) -> mlx_gen::Result<()> {
    mlx_gen::ios_quantum::before(
        label,
        tile_bytes(
            tile.shape(),
            source,
            element_bytes(tile),
            weight,
            f32_accum,
            backend(),
        ),
    )
}

/// One ancestral denoise step's latent-sized work, materialized by its per-step eval
/// (`pipeline::denoise_core`), in f32 latent buffers (the widest the step runs at), counted as new
/// buffers (every other op of the chain donates its single-consumer operand): the CFG combine's
/// difference; the update's scaled sample, the f32 noise draw and its compute-dtype cast; and, with
/// an inpaint blend, the re-noised init, the kept and the repainted halves.
pub(crate) fn sampler_step_bytes(latents: &Array, inpaint: bool) -> u64 {
    sampler_step_bytes_metadata(latents.size() as u64, inpaint)
}

/// The IP tokens' CFG pair over the materialized `[1, T, D]` `tokens`
/// (`IpImageEncoder::zeros_like_tokens` + the concatenation): the f32 zeros, their cast to the
/// tokens' width (absent at f32) and the `[2, T, D]` pair, each its own buffer.
pub(crate) fn ip_cfg_tokens_bytes(tokens: &Array) -> u64 {
    ip_cfg_tokens_bytes_metadata(&activation_facts(tokens))
}

/// Same actual CFG zeros/cast/pair formula over token metadata, never a native tensor factory.
pub(crate) fn ip_cfg_tokens_bytes_metadata(tokens: &mlx_gen::adapters::facts::TensorFacts) -> u64 {
    let t = tokens.bytes();
    let f32_zeros = mlx_gen::ios_quantum::tensor_bytes(4, &[tokens.elements()]);
    let cast = if tokens.dtype == Dtype::Float32 { 0 } else { t };
    f32_zeros
        .saturating_add(cast)
        .saturating_add(2 * t)
        .saturating_add(page_slack(3))
}

/// One adapter target's install (`adapters.rs`), counted under MLX's retention (every buffer the
/// install creates stays live until its command buffer completes): `widths` lists the bytes per
/// element of each target-sized (`elements`) buffer the delta reconstruction and the merge create
/// (an elementwise op whose single-consumer operand has its width donates and is not listed),
/// `small` the factor-sized buffers (f32 casts, low-rank products, a scaled factor), and
/// `unread_base` the base weight's stored bytes when it has not been read yet — a lazily loaded
/// base is read inside the merge, and the merged weight replaces it only once the merge completes.
pub(crate) fn adapter_install_bytes(
    elements: u64,
    widths: &[u64],
    small: u64,
    unread_base: u64,
) -> u64 {
    widths
        .iter()
        .fold(0u64, |n, w| n.saturating_add(w.saturating_mul(elements)))
        .saturating_add(small)
        .saturating_add(unread_base)
        .saturating_add(page_slack(widths.len() as u64 + 2))
}

/// iOS pre-load floor of a component that [`crate::loader::cast_for_load`]-style casts after
/// reading, as `(resident bytes after the load, floor)`. Header-only.
///
/// Every KEPT tensor (`keep(name)`: a rung-4 streamed U-Net leaves its `transformer_blocks`
/// bodies to the block stream) ends resident at its stored bytes, or — when `casts(name, stored)`
/// (the loader's rule) — at its `target_bytes`-wide copy, each one page-rounded allocation
/// ([`mlx_gen::ios_quantum::allocated_bytes`]). The loader materializes the kept
/// tensors through `Weights::materialize_*`, which evaluates them in sorted-key batches of
/// [`mlx_gen::weights::Weights::MATERIALIZE_BATCH_BYTES`] resident bytes, each batch's safetensors
/// reads first: within a batch every read source and every cast copy are live together (the
/// device holds a command buffer's operands until it completes), on top of the batches already
/// resident. The floor is the worst batch, and at least the whole resident set.
pub(crate) fn cast_load_parts(
    files: &[std::path::PathBuf],
    target_bytes: u64,
    casts: impl Fn(&str, mlx_gen::gen_core::weightsmeta::Dtype) -> bool,
    keep: impl Fn(&str) -> bool,
) -> mlx_gen::Result<(u64, u64)> {
    let mut tensors: Vec<(String, u64, u64)> = Vec::new();
    for file in files {
        for header in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?
        {
            if !keep(&header.name) {
                continue;
            }
            let stored = mlx_gen::ios_quantum::allocated_bytes(header.data_bytes);
            let (resident, live) = if casts(&header.name, header.dtype) {
                let cast = mlx_gen::ios_quantum::allocated_bytes(
                    header
                        .shape
                        .iter()
                        .fold(target_bytes, |a, &d| a.saturating_mul(d as u64)),
                );
                // The loader holds every source until the batch completes
                // (`loader::hold_cast_sources`), so the copy is never donated the source's buffer:
                // it ends resident at its own width, the source and the copy live together.
                (cast, cast.saturating_add(stored))
            } else {
                (stored, stored)
            };
            tensors.push((header.name, resident, live));
        }
    }
    tensors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let batch = mlx_gen::weights::Weights::MATERIALIZE_BATCH_BYTES as u64;
    let (mut done, mut floor) = (0u64, 0u64);
    let (mut batch_resident, mut batch_live) = (0u64, 0u64);
    for (_, resident, live) in tensors {
        batch_resident = batch_resident.saturating_add(resident);
        batch_live = batch_live.saturating_add(live);
        if batch_resident >= batch {
            floor = floor.max(done.saturating_add(batch_live));
            done = done.saturating_add(batch_resident);
            (batch_resident, batch_live) = (0, 0);
        }
    }
    floor = floor.max(done.saturating_add(batch_live));
    done = done.saturating_add(batch_resident);
    Ok((done, floor.max(done)))
}

impl LinearWork {
    /// A constructor-owned dense metadata base. This does not manufacture a native parameter.
    pub(crate) fn dense(
        weight: mlx_gen::adapters::facts::TensorFacts,
        bias: Option<mlx_gen::adapters::facts::TensorFacts>,
    ) -> Self {
        Self {
            base: mlx_gen::adapters::LinearFacts {
                base_shape: weight.shape.clone(),
                is_quantized: false,
                has_live_adapters: false,
                adapter_count: 0,
                bias_shape: bias.as_ref().map(|b| b.shape.clone()),
            },
            weights: vec![weight],
            bias,
            residuals: Vec::new(),
            group_size: crate::quant::GROUP_SIZE,
            bits: None,
        }
    }
}
impl ConvWork {
    pub(crate) fn new(
        weight: mlx_gen::adapters::facts::TensorFacts,
        bias: Option<mlx_gen::adapters::facts::TensorFacts>,
    ) -> Self {
        Self { weight, bias }
    }
    pub(crate) fn weight(&self) -> &mlx_gen::adapters::facts::TensorFacts {
        &self.weight
    }
    pub(crate) fn bias(&self) -> Option<&mlx_gen::adapters::facts::TensorFacts> {
        self.bias.as_ref()
    }
}
impl EffectScope<'_> {
    /// Header-owned constructor defaults replace only the native fallback; selected effect lookup
    /// and missing-target errors remain the same as the native scope. No dense fallback on a map miss.
    pub(crate) fn linear_metadata(&self, name: &str, base: &LinearWork) -> Result<LinearWork> {
        let path = crate::quant::child_path(&self.prefix, name);
        match self.effects {
            Some(effects) => effects.0.get(&path).cloned().ok_or_else(|| {
                mlx_gen::Error::Msg(format!("sdxl missing final projection effect: {path}"))
            }),
            None => Ok(base.clone()),
        }
    }
    pub(crate) fn shortcut_metadata(&self, name: &str, base: &LinearWork) -> LinearWork {
        let path = crate::quant::child_path(&self.prefix, name);
        self.effects
            .and_then(|effects| effects.0.get(&path))
            .cloned()
            .unwrap_or_else(|| base.clone())
    }
    pub(crate) fn conv_metadata(&self, name: &str, base: &ConvWork) -> ConvWork {
        let path = crate::quant::child_path(&self.prefix, name);
        let weight = self
            .effects
            .and_then(|effects| effects.1.get(&path))
            .cloned()
            .unwrap_or_else(|| base.weight.clone());
        ConvWork {
            weight,
            bias: base.bias.clone(),
        }
    }
}

pub(crate) fn control_injection_bytes_metadata(
    hidden: &mlx_gen::adapters::facts::TensorFacts,
    residual: &mlx_gen::adapters::facts::TensorFacts,
) -> u64 {
    let output = Dtype::from_promoting_types(hidden.dtype, residual.dtype);
    residual
        .bytes()
        .saturating_add(hidden.elements() * dtype_bytes(output))
}
impl ControlMergeWork {
    pub(crate) fn record_outputs_metadata(
        &mut self,
        down: &[mlx_gen::adapters::facts::TensorFacts],
        mid: &mlx_gen::adapters::facts::TensorFacts,
    ) -> Result<()> {
        for output in down {
            self.down = self
                .down
                .checked_add(mlx_gen::ios_quantum::allocated_bytes(output.bytes()))
                .ok_or_else(|| {
                    mlx_gen::Error::Unsupported("sdxl control merge accounting overflow".into())
                })?;
        }
        self.mid = self
            .mid
            .checked_add(mlx_gen::ios_quantum::allocated_bytes(mid.bytes()))
            .ok_or_else(|| {
                mlx_gen::Error::Unsupported("sdxl control merge accounting overflow".into())
            })?;
        Ok(())
    }
}
pub(crate) fn sampler_step_bytes_metadata(elements: u64, inpaint: bool) -> u64 {
    mlx_gen::ios_quantum::elementwise_bytes(elements, 4 + 3 * u64::from(inpaint))
}

/// Borrow only shape/dtype for guard activations. Never query lazy strides, evaluate, retain a
/// native owner, or claim a layout/backing identity merely to calculate a header-equivalent guard.
pub(crate) fn activation_facts(array: &Array) -> mlx_gen::adapters::facts::TensorFacts {
    mlx_gen::adapters::facts::TensorFacts {
        shape: array.shape().to_vec(),
        dtype: array.dtype(),
        strides: Vec::new(),
    }
}

#[cfg(test)]
mod linear_dtype_cast_tests {
    use super::*;
    use mlx_gen::adapters::facts::TensorFacts;

    fn tensor(shape: &[i32], dtype: Dtype) -> TensorFacts {
        TensorFacts {
            shape: shape.to_vec(),
            dtype,
            strides: Vec::new(),
        }
    }

    #[test]
    fn packed_compute_and_postbias_casts_follow_independent_literals_host() {
        use Dtype::{Bfloat16 as B, Float16 as H, Float32 as F};
        // [3,64] activation =192 elements, affine operands [2,1] =2 each,
        // qmm result [3,2] =6. Expected byte totals below are literal operation sums,
        // not calls to the implementation's promotion or byte-budget helpers.
        // input, scales, quantization offsets, ordinary bias, compute, output, cast bytes.
        let cases = [
            (H, H, H, None, H, H, 0),
            (B, B, B, None, B, B, 0),
            (F, F, F, Some(F), F, F, 0),
            // Neither selected AdaptableLinear nor pinned QuantizedLinear Module precasts x.
            (F, H, H, None, F, F, 16),
            (B, H, H, None, F, F, 784),
            (H, H, B, Some(H), F, F, 792),
            (H, B, H, Some(H), F, F, 792),
            (B, H, B, Some(B), F, F, 792),
            (F, H, B, Some(H), F, F, 24),
            (H, F, H, Some(H), F, F, 784),
            (H, H, F, Some(H), F, F, 784),
            (H, H, H, Some(F), H, F, 24),
            (B, B, B, Some(H), B, F, 32),
            (H, H, H, Some(B), H, F, 32),
            (H, H, B, None, F, F, 784),
            (F, B, B, None, F, F, 16),
            (H, B, B, None, F, F, 784),
        ];
        for bits in [4, 8] {
            for (input, scales, offsets, bias, compute, output, cast_bytes) in cases {
                let work = LinearWork {
                    base: mlx_gen::adapters::LinearFacts {
                        base_shape: vec![2, 64],
                        is_quantized: true,
                        has_live_adapters: false,
                        adapter_count: 0,
                        bias_shape: bias.map(|_| vec![2]),
                    },
                    weights: vec![
                        tensor(&[2, 64 * bits / 32], Dtype::Uint32),
                        tensor(&[2, 1], scales),
                        tensor(&[2, 1], offsets),
                    ],
                    bias: bias.map(|dtype| tensor(&[2], dtype)),
                    residuals: Vec::new(),
                    group_size: 64,
                    bits: Some(bits),
                };
                assert_eq!(work.packed_compute_dtype(input), compute);
                assert_eq!(work.output_dtype(input), output);
                assert_eq!(
                    work.operand_bytes(&[3, 64], input),
                    cast_bytes,
                    "q{bits} input={input:?} scales={scales:?} offsets={offsets:?} bias={bias:?}"
                );
            }
            let work = LinearWork {
                base: mlx_gen::adapters::LinearFacts {
                    base_shape: vec![2, 64],
                    is_quantized: true,
                    has_live_adapters: false,
                    adapter_count: 0,
                    bias_shape: Some(vec![3, 2]),
                },
                weights: vec![
                    tensor(&[2, 64 * bits / 32], Dtype::Uint32),
                    tensor(&[2, 1], H),
                    tensor(&[2, 1], H),
                ],
                bias: Some(tensor(&[3, 2], F)),
                residuals: Vec::new(),
                group_size: 64,
                bits: Some(bits),
            };
            assert_eq!(work.output_shape(&[1, 64]).unwrap(), vec![3, 2]);
            assert_eq!(
                work.operand_bytes(&[1, 64], H),
                8,
                "only two unbroadcast qmm result elements are cast, not six add outputs"
            );
        }
    }

    #[test]
    fn dense_fused_addmm_casts_remain_independent_literals_host() {
        use Dtype::{Bfloat16 as B, Float16 as H, Float32 as F};
        // Dense [2,64] weights =128 elements, [3,64] input =192, bias =2.
        for (input, weight, bias, output, cast_bytes) in [
            (H, H, None, H, 0),
            (H, B, None, F, 1280),
            (H, H, Some(F), F, 1280),
            (F, H, Some(B), F, 520),
            (B, B, Some(H), F, 1288),
            (F, F, Some(F), F, 0),
        ] {
            let work = LinearWork::dense(tensor(&[2, 64], weight), bias.map(|d| tensor(&[2], d)));
            assert_eq!(work.output_dtype(input), output);
            assert_eq!(work.operand_bytes(&[3, 64], input), cast_bytes);
        }
    }
}
