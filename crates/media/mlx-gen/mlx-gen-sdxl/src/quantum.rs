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
    attention_block_bytes, conv_stage_bytes, scores_bytes, tensor_bytes, AttentionBlockGeometry,
    ConvStageGeometry, Scores,
};
use mlx_rs::{Array, Dtype};

/// U-Net stem: Kolors context projection, timestep/`text_time` embedding and `conv_in`.
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

/// One adapter (LoRA / LoKr / LoHa) file's read, checked against its header floor before the read.
pub(crate) const ADAPTER_LOAD: &str = "SDXL adapter load";
/// One adapter target's delta reconstruction + merge into the base (or, on a packed base, the
/// install of its forward-time residual).
pub(crate) const ADAPTER_MERGE: &str = "SDXL adapter merge";

/// The MLX backend whose kernels run a quantum, which decides the kernel transients priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    /// The device: Metal kernels (Winograd / implicit-GEMM convolution, fused SDPA).
    Metal,
    /// MLX's CPU backend: explicit-GEMM convolution over an f32 im2col, unfused SDPA.
    Cpu,
}

impl Backend {
    /// The backend of the current default device.
    pub(crate) fn current() -> Self {
        match mlx_rs::Device::try_default() {
            Ok(device) if device == mlx_rs::Device::cpu() => Backend::Cpu,
            _ => Backend::Metal,
        }
    }
}

fn u(v: i32) -> u64 {
    v.max(0) as u64
}

/// Bytes per element of `a`'s dtype.
pub(crate) fn element_bytes(a: &Array) -> u64 {
    a.item_size() as u64
}

/// Elements of `a`, as u64.
pub(crate) fn elements(a: &Array) -> u64 {
    a.shape().iter().fold(1u64, |n, d| n.saturating_mul(u(*d)))
}

/// Bytes of `a` as stored.
pub(crate) fn bytes(a: &Array) -> u64 {
    elements(a).saturating_mul(element_bytes(a))
}

/// Bytes per element of `dtype`.
pub(crate) fn dtype_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Uint16 | Dtype::Int16 => 2,
        Dtype::Float64 | Dtype::Int64 | Dtype::Uint64 | Dtype::Complex64 => 8,
        _ => 4,
    }
}

/// The dtype `x`-typed activations leave `linear` at: MLX promotes a dense matmul to
/// `result_type(x, weight)` and a quantized one to `result_type(x, scales)`, so a pre-quantized
/// SceneWorks snapshot's BF16 `.scales` (every packed U-Net / CLIP projection) turn f16 activations
/// into f32 (measured: `quantum_tests::proofs::f16_transformer_quanta_track_their_measured_peaks`),
/// and everything downstream of that projection — residual sums, norms, convolutions — runs at f32.
pub(crate) fn linear_output_dtype(linear: &AdaptableLinear, x: Dtype) -> Dtype {
    let weight = match linear.quantized_params() {
        Some((_, scales, ..)) => scales.dtype(),
        None => linear.weight_dtype().unwrap_or(x),
    };
    Dtype::from_promoting_types(x, weight)
}

/// Output-row count of one projection (the `out` of its logical `[out, in]` weight).
pub(crate) fn linear_out(linear: &AdaptableLinear) -> u64 {
    linear.base_shape().first().map_or(0, |d| u(*d))
}

/// Channel counts `(in, out)` of an NHWC conv weight `[out, kH, kW, in]`.
pub(crate) fn conv_channels(weight: &Array) -> (i32, i32) {
    let s = weight.shape();
    (s[s.len() - 1], s[0])
}

/// One 2-D NHWC convolution call, from its input shape, its LOADED weight and the call's
/// stride/padding, for its kernel-internal transients.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvCall {
    batch: u64,
    in_h: u64,
    in_w: u64,
    cin: u64,
    out_h: u64,
    out_w: u64,
    cout: u64,
    kh: u64,
    kw: u64,
    stride: u64,
    pad: u64,
    /// Activation element bytes.
    act: u64,
    /// Weight element bytes.
    weight: u64,
}

impl ConvCall {
    /// `x`: the convolution's NHWC input shape; `weight`: the `[out, kH, kW, in]` weight it runs.
    pub(crate) fn new(x: &[i32], act: u64, weight: &Array, stride: i32, pad: i32) -> Self {
        let w = weight.shape();
        let out = |n: i32, k: i32| u((n + 2 * pad - k) / stride.max(1) + 1);
        Self {
            batch: u(x[0]),
            in_h: u(x[1]),
            in_w: u(x[2]),
            cin: u(w[3]),
            out_h: out(x[1], w[1]),
            out_w: out(x[2], w[2]),
            cout: u(w[0]),
            kh: u(w[1]),
            kw: u(w[2]),
            stride: u(stride),
            pad: u(pad),
            act,
            weight: element_bytes(weight),
        }
    }

    /// Bytes of the call's input at its activation width.
    pub(crate) fn in_bytes(&self) -> u64 {
        tensor_bytes(self.act, &[self.batch, self.in_h, self.in_w, self.cin])
    }

    /// Bytes of the call's output at its activation width.
    pub(crate) fn out_bytes(&self) -> u64 {
        tensor_bytes(self.act, &[self.batch, self.out_h, self.out_w, self.cout])
    }

    fn weight_elements(&self) -> u64 {
        [self.cout, self.kh, self.kw, self.cin]
            .iter()
            .fold(1u64, |a, b| a.saturating_mul(*b))
    }

    #[cfg(test)]
    pub(crate) fn weight_elements_for_test(&self) -> u64 {
        self.weight_elements()
    }

    /// Whether MLX's Metal `dispatch_conv_2D_gpu` routes this call to `winograd_conv_2D_gpu`
    /// (a 3x3, stride-1, ungrouped, unflipped call with 32-aligned channels, `N·H·W ≥ 4096` and
    /// `C + O ≥ 256`).
    pub(crate) fn winograd(&self) -> bool {
        self.kh == 3
            && self.kw == 3
            && self.stride == 1
            && self.cin.is_multiple_of(32)
            && self.cout.is_multiple_of(32)
            && self.batch * self.in_h * self.in_w >= 4096
            && self.cin + self.cout >= 256
    }

    /// The bytes this call's kernel allocates beside its input and output, on `backend`:
    ///
    /// * Metal (`conv_2D_gpu`): `ensure_row_contiguous(wt)` copies the weight, which SDXL holds as
    ///   an NHWC transposed VIEW of its NCHW checkpoint tensor — at the activation width when the
    ///   activations are wider than the stored weight (the packed tier's f32 activations over f16
    ///   convs: `conv2d` promotes, and the cast copy is that row-contiguous copy); Winograd (`winograd_conv_2D_gpu`)
    ///   adds the zero-padded input `[N, 6⌈(H+2p−2)/6⌉+2, …, C]`, the transformed input
    ///   `[64, tiles, C]`, output `[64, tiles, O]` and filter `[64, C, O]` (`tiles = N⌈oH/6⌉⌈oW/6⌉`).
    /// * CPU (`explicit_gemm_conv_ND_cpu`): an f32 padded input, the f32 im2col
    ///   `[N·oH·oW, kH·kW·C]`, an f32 weight copy (non-contiguous or non-f32), and an f32 GEMM
    ///   output when the activation is narrower.
    pub(crate) fn transient(&self, backend: Backend) -> u64 {
        let weight_copy = |width: u64| self.weight_elements().saturating_mul(width);
        match backend {
            Backend::Metal => {
                let mut bytes = weight_copy(self.weight.max(self.act));
                if self.winograd() {
                    let padded = |n: u64| 6 * (n + 2 * self.pad).saturating_sub(2).div_ceil(6) + 2;
                    let tiles = self.batch * self.out_h.div_ceil(6) * self.out_w.div_ceil(6);
                    bytes = bytes
                        .saturating_add(tensor_bytes(
                            self.act,
                            &[self.batch, padded(self.in_h), padded(self.in_w), self.cin],
                        ))
                        .saturating_add(tensor_bytes(self.act, &[64, tiles, self.cin]))
                        .saturating_add(tensor_bytes(self.act, &[64, tiles, self.cout]))
                        .saturating_add(tensor_bytes(self.weight, &[64, self.cin, self.cout]));
                }
                bytes
            }
            Backend::Cpu => {
                let rows = self.batch * self.out_h * self.out_w;
                let mut bytes = tensor_bytes(
                    4,
                    &[
                        self.batch,
                        self.in_h + 2 * self.pad,
                        self.in_w + 2 * self.pad,
                        self.cin,
                    ],
                )
                .saturating_add(tensor_bytes(4, &[rows, self.kh * self.kw * self.cin]))
                .saturating_add(weight_copy(4));
                if self.act != 4 {
                    bytes = bytes.saturating_add(tensor_bytes(4, &[rows, self.cout]));
                }
                bytes
            }
        }
    }
}

/// Buffers of the counted ops of the composed `mlx_gen::nn::group_norm` over its input: the
/// grouped-transpose reshape copy, the LayerNorm output, the transposed-back reshape copy, the weight
/// product and the bias sum (neither affine op can donate: their operand is the reshape copy's view
/// with a second consumer pending). Measured: 5.00 input-sized buffers at f32 (and on Metal at any
/// width — see [`group_norm_buffers`]).
pub(crate) const GROUP_NORM_BUFFERS: u64 = 5;

/// Extra input-sized buffers, in units of the activation width `act`, that one `mlx_rs::fast::
/// layer_norm` (alone, or inside [`GROUP_NORM_BUFFERS`]' composed GroupNorm) holds on `backend`
/// beyond its f32 count when the activations are narrower than f32. MLX's CPU backend
/// runs the fast op's composed fallback in f32 — `astype(x, f32)` plus the centred input, its square
/// and the normalized product at f32 — so an f16 norm holds three f32 input-sized buffers more than
/// its width-scaled f32 count (measured at f16: GroupNorm 11.0, LayerNorm 9.06 input-sized buffers,
/// against 5.0 / 3.03 at f32). Metal's fused kernel keeps the f32 statistics in registers: none.
pub(crate) fn norm_promotion_units(backend: Backend, act: u64) -> u64 {
    match backend {
        Backend::Cpu if act < 4 => 3 * 4 / act.max(1),
        _ => 0,
    }
}

/// [`GROUP_NORM_BUFFERS`] on `backend` at activation width `act` (+ [`norm_promotion_units`]).
pub(crate) fn group_norm_buffers(backend: Backend, act: u64) -> u64 {
    GROUP_NORM_BUFFERS + norm_promotion_units(backend, act)
}

/// MLX's Metal allocator (also the macOS CPU backend's) rounds every buffer above one page up to
/// the 16 KiB `vm_page_size`: at most one page of slack per counted buffer.
pub(crate) const PAGE_BYTES: u64 = 16 * 1024;

/// The allocator's page rounding over `buffers` counted buffers.
pub(crate) fn page_slack(buffers: u64) -> u64 {
    buffers.saturating_mul(PAGE_BYTES)
}

/// A convolutional quantum, counted under MLX's retention: a quantum's graph is evaluated in one
/// `eval`, and every primitive's INPUT buffers stay referenced until its command buffer (Metal) or
/// queued task (CPU) completes, so the quantum's intermediates are live together — the peak is the
/// SUM of its new buffers (an elementwise op whose operand is a single-consumer temporary donates and
/// adds none) plus every kernel's transients. `in_units` / `out_units` count the new buffers at
/// `call`'s input / output extents; `convs` are the kernels whose transients the quantum holds.
pub(crate) fn conv_sum(
    call: &ConvCall,
    in_units: u64,
    out_units: u64,
    convs: &[&ConvCall],
    backend: Backend,
) -> u64 {
    let transients = convs
        .iter()
        .fold(0u64, |n, c| n.saturating_add(c.transient(backend)));
    // Each convolution's kernel allocates at most four transient buffers (Metal Winograd: padded,
    // transformed input/output/filter + the weight copy; CPU: padded, im2col, weight, GEMM out).
    let buffers = in_units + out_units + 5 * convs.len() as u64;
    conv_stage_bytes(ConvStageGeometry {
        batch: call.batch,
        height: call.out_h,
        width: call.out_w,
        in_channels: call.cin,
        out_channels: call.cout,
        element_bytes: call.act,
        in_live: 0,
        out_live: out_units,
        weight_bytes: transients,
    })
    .saturating_add(call.in_bytes().saturating_mul(in_units))
    .saturating_add(page_slack(buffers))
}

/// A resnet (`ResnetBlock2D::forward_core`, U-Net or VAE) over input `x` `[B,H,W,Cx]` at
/// `act_in` bytes per element, whose `conv1` takes `cin` (= Cx, or Cx + skip after the up path's
/// concat) to `cout`, and whose second half runs at `act_out` (wider than `act_in` when the
/// projected time embedding is: the packed tier's f32 `temb` promotes the sum, and GroupNorm 2,
/// SiLU, `conv2` and the residual follow it). Counted new buffers (see [`conv_sum`]): input width
/// — the skip concatenation (up path), GroupNorm 1 ([`group_norm_buffers`]), SiLU, `conv1` and
/// the 1x1 shortcut (when `cin != cout`); output width — the promoted time-embedding sum (only when
/// it widens: otherwise it donates `conv1`), GroupNorm 2, SiLU, `conv2`; the bias adds and the
/// residual sum donate. Plus both convolutions' kernel transients (`conv2`'s at `act_out`).
pub(crate) fn resnet_bytes(
    x: &[i32],
    act_in: u64,
    act_out: u64,
    conv1: &Array,
    conv2: &Array,
    backend: Backend,
) -> u64 {
    let (cin, cout) = conv_channels(conv1);
    let act_out = act_out.max(act_in);
    let input = [x[0], x[1], x[2], cin];
    let c1 = ConvCall::new(&input, act_in, conv1, 1, 1);
    let mid = [x[0], x[1], x[2], conv_channels(conv2).0];
    let c2 = ConvCall::new(&mid, act_out, conv2, 1, 1);
    let concat = u64::from(cin != x[3]);
    let shortcut = u64::from(cin != cout);
    let widened = u64::from(act_out > act_in);
    let in_units = concat + group_norm_buffers(backend, act_in) + 1;
    let narrow_out = 1 + shortcut;
    let wide_out = widened + group_norm_buffers(backend, act_out) + 1 + 1;
    let out = |width: u64, units: u64| {
        tensor_bytes(width, &[u(x[0]), u(x[1]), u(x[2]), u(cout)]).saturating_mul(units)
    };
    let buffers = in_units + narrow_out + wide_out + 10;
    c1.in_bytes()
        .saturating_mul(in_units)
        .saturating_add(out(act_in, narrow_out))
        .saturating_add(out(act_out, wide_out))
        .saturating_add(c1.transient(backend))
        .saturating_add(c2.transient(backend))
        .saturating_add(page_slack(buffers))
}

/// A single convolution quantum over an already-materialized input: the output, `extra_out`
/// output-width epilogue buffers (a SiLU; a bias add or scale donates), `in_units` input-width
/// buffers the quantum builds before the convolution (a pad, a nearest-2x copy, a lazy cast), plus
/// the convolution's transient.
pub(crate) fn single_conv_bytes(
    call: &ConvCall,
    in_units: u64,
    extra_out: u64,
    backend: Backend,
) -> u64 {
    conv_sum(call, in_units, 1 + extra_out, &[call], backend)
}

/// GroupNorm + SiLU + one convolution (the `conv_out` heads): GroupNorm and SiLU at input width,
/// the convolution's output and transient.
pub(crate) fn norm_conv_bytes(call: &ConvCall, backend: Backend) -> u64 {
    conv_sum(
        call,
        group_norm_buffers(backend, call.act) + 1,
        1,
        &[call],
        backend,
    )
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

/// Row count of one VAE mid-attention query chunk under the iOS bounded attention: the largest that
/// keeps one chunk's `[B, 1, rows, N]` scores within [`VAE_ATTENTION_SCORE_ELEMENTS`].
pub(crate) const VAE_ATTENTION_SCORE_ELEMENTS: u64 = 16 * 1024 * 1024;

/// The VAE mid-block single-head attention over `[B, N, C]`, `query_rows` queries per SDPA call
/// (the whole `N` on the host path, one chunk on iOS). Head dim = `C` (512 in SDXL): never a fused
/// Metal shape, so the scores materialize per call on every backend.
///
/// On iOS the projections are materialized before the attention (`VaeAttention::forward_planned`),
/// so the price is the wider of two phases, each counted as new buffers under [`conv_sum`]'s
/// retention model:
/// * projections — GroupNorm ([`group_norm_buffers`]), q, k, v (hidden-sized);
/// * attention — q, k, v, the fallback's scaled query, the SDPA output and, chunked, the
///   concatenation of the per-chunk outputs, the output projection (hidden-sized; the residual sum
///   donates); a chunk's scores (the single-head softmax donates them in place: measured one
///   score tensor) and its scaled query / output rows —
///   each chunk is evaluated before the next is built, which releases the previous chunk's scores.
pub(crate) fn vae_attention_bytes(
    b: i32,
    n: i32,
    c: i32,
    query_rows: i32,
    act: u64,
    backend: Backend,
) -> u64 {
    let rows = query_rows.clamp(1, n.max(1));
    let chunked = u64::from(rows < n);
    let hidden = tensor_bytes(act, &[u(b), u(n), u(c)]);
    let projections = group_norm_buffers(backend, act) + 3;
    let attention_units = 3 + 1 + 1 + chunked + 1;
    let attention = hidden
        .saturating_mul(attention_units)
        .saturating_add(tensor_bytes(act, &[2 * chunked, u(b), u(rows), u(c)]))
        .saturating_add(scores_bytes(
            Scores::Materialized {
                live: 1,
                element_bytes: act,
            },
            u(b),
            1,
            u(rows),
            u(n),
        ));
    hidden
        .saturating_mul(projections)
        .max(attention)
        .saturating_add(page_slack(projections.max(attention_units + 4)))
}

/// Check one halo-expanded tile of a layer-wise tiled VAE convolution BEFORE its norm/activation/
/// conv graph is constructed, priced at the tile's extent and `weight`'s channel counts. Called from
/// the per-tile preprocess hook of `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc*`, which evaluates
/// each output core before the next tile is built.
pub(crate) fn before_tile(tile: &Array, weight: &Array, f32_accum: bool) -> mlx_gen::Result<()> {
    mlx_gen::ios_quantum::before(
        VAE_TILE,
        tile_bytes(
            tile.shape(),
            element_bytes(tile),
            weight,
            f32_accum,
            Backend::current(),
        ),
    )
}

/// One tile of `mlx_gen::vae_tiling`'s tiled 3x3 convolution over a `[B, h, w, C]` crop (a strided
/// slice of the full activation): the GlobalGroupNorm apply's contiguous crop copy, centred and
/// affine buffers, the SiLU output (input width), the `f32_accum` promoted crop (f32), the
/// convolution output and its core copy (output width, f32 under `f32_accum`) + the transient.
pub(crate) fn tile_bytes(
    tile: &[i32],
    act: u64,
    weight: &Array,
    f32_accum: bool,
    backend: Backend,
) -> u64 {
    let conv_act = if f32_accum { 4 } else { act };
    let call = ConvCall::new(tile, conv_act, weight, 1, 1);
    let crop = tensor_bytes(act, &[u(tile[0]), u(tile[1]), u(tile[2]), u(tile[3])]);
    let promoted = if f32_accum { call.in_bytes() } else { 0 };
    // The norm phase holds 3 crop-width buffers; the convolution the SiLU output (+ promoted
    // copy), its output, the core and the transient.
    (3 * crop).max(
        crop.saturating_add(promoted)
            .saturating_add(2 * call.out_bytes())
            .saturating_add(call.transient(backend)),
    )
}

/// One layer-wise tiled VAE stage producing `[B, H, W, cout]` from a `[B, H, W, cin]` input: the
/// GlobalGroupNorm statistics' grouped copy of the input, the evaluated row strips plus their
/// concatenation (two output-width images), and (resnet) the first convolution's output, the
/// shortcut and the sum. `resnet` selects the resnet layer's extra full-resolution buffers; the
/// per-tile scratch is checked separately by [`before_tile`].
pub(crate) fn tiled_stage_bytes(out: &[i32], cin: i32, cout: i32, act: u64, resnet: bool) -> u64 {
    let plane = |c: i32| tensor_bytes(act, &[u(out[0]), u(out[1]), u(out[2]), u(c)]);
    let outputs = if resnet { 3 } else { 2 };
    plane(cin).saturating_add(plane(cout).saturating_mul(outputs))
}

/// The bounded encode's tiled stride-2 downsample stage (`vae::downsample_tiled`) producing
/// `out` `[B, oH, oW, C]`: the evaluated row strips and their concatenation — two output-width
/// images (its input is already materialized). Each tile is checked by [`downsample_tile_bytes`].
pub(crate) fn downsample_stage_bytes(out: &[i32], act: u64) -> u64 {
    tensor_bytes(act, &[2, u(out[0]), u(out[1]), u(out[2]), u(out[3])])
        .saturating_add(page_slack(2))
}

/// One tile of the bounded encode's stride-2 downsample over a `[B, h, w, C]` crop (`h`/`w` its
/// exact halo, `2·core + 1`): the bottom/right zero-padded crop copy, the stride-2 convolution's
/// output core (its bias add donates) and the kernel transient.
pub(crate) fn downsample_tile_bytes(
    crop: &[i32],
    act: u64,
    weight: &Array,
    backend: Backend,
) -> u64 {
    let call = ConvCall::new(crop, act, weight, 2, 0);
    single_conv_bytes(&call, 1, 0, backend)
}

/// [`tiled_stage_bytes`] for the f16-storage upsample convolution that accumulates in f32
/// (`tiled_conv2d_3x3_nhwc_f32_accum`, the fp16-fix VAE path): the f32 copies of the conv weight
/// and bias are built once and stay live across every tile of the stage, on top of the stage's
/// activations. Each tile's own promoted crop and f32 output are priced by [`before_tile`].
pub(crate) fn tiled_stage_f32_accum(out: &[i32], weight: &Array, bias: &Array, act: u64) -> u64 {
    let (cin, cout) = conv_channels(weight);
    tiled_stage_bytes(out, cin, cout, act, false)
        .saturating_add(elements(weight).saturating_mul(4))
        .saturating_add(elements(bias).saturating_mul(4))
}

/// One ancestral denoise step's latent-sized work, materialized by its per-step eval
/// (`pipeline::denoise_core`), in f32 latent buffers (the widest the step runs at), counted as new
/// buffers (every other op of the chain donates its single-consumer operand): the CFG combine's
/// difference; the update's scaled sample, the f32 noise draw and its compute-dtype cast; and, with
/// an inpaint blend, the re-noised init, the kept and the repainted halves.
pub(crate) fn sampler_step_bytes(latents: &Array, inpaint: bool) -> u64 {
    mlx_gen::ios_quantum::elementwise_bytes(latents.size() as u64, 4 + 3 * u64::from(inpaint))
}

/// The decoded image's pixel conversion (`pipeline::decoded_to_image`): one f32 image-sized buffer
/// (every op of the scale/clamp/round chain after the first donates; a half-precision VAE's lazy
/// f32 output cast is donated into by the first) and its allocator page.
pub(crate) fn decoded_to_image_bytes(decoded: &Array) -> u64 {
    mlx_gen::ios_quantum::elementwise_bytes(decoded.size() as u64, 1).saturating_add(page_slack(1))
}

/// `copies` buffers the size of `arrays` at their own widths: a small elementwise pass (residual
/// add, scale) over already-materialized tensors.
pub(crate) fn elementwise_bytes(arrays: &[&Array], copies: u64) -> u64 {
    arrays
        .iter()
        .fold(0u64, |n, a| n.saturating_add(bytes(a)))
        .saturating_mul(copies)
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

/// iOS pre-load floor for a component that [`crate::loader::cast_for_load`]-style casts after
/// reading: every tensor's RESIDENT bytes after the load (its `dtype`-wide copy when the loader casts
/// it, its stored bytes otherwise), plus the largest stored-or-cast tensor (a cast holds its source
/// and its copy together; a read holds one tensor). `casts(name, stored)` mirrors the loader's rule.
/// Header-only; a no-op off device.
pub(crate) fn before_cast_load(
    label: &str,
    files: &[std::path::PathBuf],
    target_bytes: u64,
    casts: impl Fn(&str, mlx_gen::gen_core::weightsmeta::Dtype) -> bool,
) -> mlx_gen::Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let (mut total, mut largest) = (0u64, 0u64);
    for file in files {
        for header in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?
        {
            let resident = if casts(&header.name, header.dtype) {
                let cast = header
                    .shape
                    .iter()
                    .fold(target_bytes, |a, &d| a.saturating_mul(d as u64));
                largest = largest.max(cast.saturating_add(header.data_bytes));
                cast
            } else {
                largest = largest.max(header.data_bytes);
                header.data_bytes
            };
            total = total.saturating_add(resident);
        }
    }
    mlx_gen::ios_quantum::before(label, total.saturating_add(largest))
}
