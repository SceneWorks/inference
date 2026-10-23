//! Counted kernel accounting shared by every family's iOS quanta (epic 16774): which MLX backend
//! runs a quantum, how its SDPA scores exist, and the AutoencoderKL convolution / GroupNorm /
//! layer-tiled VAE prices. Hoisted out of `mlx-gen-sdxl` and `mlx-gen-anima` (sc-26433) so a second
//! diffusers VAE or Qwen3 tower prices the SAME ops with the same counted formulas instead of a copy.
//! Each formula is proven against MLX's measured CPU peak by the call sites that use it (SDXL's
//! `quantum_tests::proofs`, Anima's and Z-Image's `quantum_tests`).

use super::{conv_stage_bytes, scores_bytes, tensor_bytes, ConvStageGeometry, Scores};
use mlx_rs::{Array, Dtype};

/// The MLX backend whose kernels run a quantum, which decides the kernel transients priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The device: Metal kernels (Winograd / implicit-GEMM convolution, fused SDPA).
    Metal,
    /// MLX's CPU backend: explicit-GEMM convolution over an f32 im2col, unfused SDPA.
    Cpu,
}

impl Backend {
    /// The backend of the current default device.
    pub fn current() -> Self {
        match mlx_rs::Device::try_default() {
            Ok(device) if device == mlx_rs::Device::cpu() => Backend::Cpu,
            _ => Backend::Metal,
        }
    }
}

fn u(v: i32) -> u64 {
    v.max(0) as u64
}

/// Bytes per element of `dtype` (the activation width a quantum is priced at).
pub fn dtype_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Uint16 | Dtype::Int16 => 2,
        Dtype::Float64 | Dtype::Int64 | Dtype::Uint64 | Dtype::Complex64 => 8,
        _ => 4,
    }
}

/// Bytes per element of `a`'s dtype.
pub fn element_bytes(a: &Array) -> u64 {
    a.item_size() as u64
}

/// Elements of `a`, as u64.
pub fn elements(a: &Array) -> u64 {
    a.shape().iter().fold(1u64, |n, d| n.saturating_mul(u(*d)))
}

/// Bytes of `a` as stored.
pub fn bytes(a: &Array) -> u64 {
    elements(a).saturating_mul(element_bytes(a))
}

/// How the scores of a quantum's `mlx_rs::fast::scaled_dot_product_attention` calls exist in
/// memory on `backend`.
///
/// `attentions` lists each attention as `(queries, rows_per_call, keys)`: a query-chunked
/// attention (`sdpa_budgeted_bhsd`) issues calls of `rows_per_call` queries plus one call over the
/// remainder; an unchunked one passes `rows_per_call = queries`. The scores are [`Scores::Fused`]
/// only when MLX 0.32's Metal kernel serves EVERY call (`ScaledDotProductAttention::use_fallback`,
/// mlx/backend/metal/scaled_dot_product_attention.cpp):
/// * `sdpa_full` — more than 8 query rows at head dims 64/80/128;
/// * `sdpa_vector` — at most 8 query rows, no more than the keys, at head dims 64/96/128/256
///   (and `rows x gqa <= 32`; every caller passes gqa 1 — a GQA tower repeats its kv heads before
///   the call);
/// * an element width the kernels are instantiated for: float32, float16, bfloat16 (4 or 2 bytes).
///
/// Anything else — the CPU, or a head dim / row count / width the kernels reject — runs MLX's
/// unfused fallback (`q·kᵀ → softmax → ·v`), which materializes the logits and their softmax at the
/// compute width: [`Scores::Materialized`] with 2 live. A single falling-back call prices them all
/// as materialized (an over-estimate, never an under-estimate).
pub fn sdpa_scores(
    backend: Backend,
    head_dim: u64,
    element_bytes: u64,
    attentions: &[(u64, u64, u64)],
) -> Scores {
    let served = |rows: u64, keys: u64| {
        if rows > 8 {
            matches!(head_dim, 64 | 80 | 128)
        } else {
            rows <= keys && matches!(head_dim, 64 | 96 | 128 | 256)
        }
    };
    let fused = backend == Backend::Metal
        && matches!(element_bytes, 2 | 4)
        && attentions.iter().all(|&(queries, rows, keys)| {
            let rows = rows.min(queries);
            rows == 0
                || (served(rows, keys) && (queries % rows == 0 || served(queries % rows, keys)))
        });
    if fused {
        Scores::Fused
    } else {
        Scores::Materialized {
            live: 2,
            element_bytes,
        }
    }
}

/// Channel counts `(in, out)` of an NHWC conv weight `[out, kH, kW, in]`.
pub fn conv_channels(weight: &Array) -> (i32, i32) {
    let s = weight.shape();
    (s[s.len() - 1], s[0])
}

/// One 2-D NHWC convolution call, from its input shape, its LOADED weight and the call's
/// stride/padding, for its kernel-internal transients.
#[derive(Clone, Copy, Debug)]
pub struct ConvCall {
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
    pub fn new(x: &[i32], act: u64, weight: &Array, stride: i32, pad: i32) -> Self {
        Self::from_metadata(x, act, weight.shape(), element_bytes(weight), stride, pad)
    }

    /// The same call geometry from actual OHWI axes and PRECAST weight bytes PER ELEMENT.
    /// This constructs no Array and preserves the native constructor's arithmetic/dispatch.
    pub fn from_metadata(
        x: &[i32],
        act: u64,
        w: &[i32],
        weight_element_bytes: u64,
        stride: i32,
        pad: i32,
    ) -> Self {
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
            weight: weight_element_bytes,
        }
    }

    /// Bytes of the call's input at its activation width.
    pub fn in_bytes(&self) -> u64 {
        tensor_bytes(self.act, &[self.batch, self.in_h, self.in_w, self.cin])
    }

    /// Bytes of the call's output at its activation width.
    pub fn out_bytes(&self) -> u64 {
        tensor_bytes(self.act, &[self.batch, self.out_h, self.out_w, self.cout])
    }

    /// Elements of the call's `[out, kH, kW, in]` weight.
    pub fn weight_elements(&self) -> u64 {
        [self.cout, self.kh, self.kw, self.cin]
            .iter()
            .fold(1u64, |a, b| a.saturating_mul(*b))
    }

    /// Whether MLX's Metal `dispatch_conv_2D_gpu` routes this call to `winograd_conv_2D_gpu`
    /// (a 3x3, stride-1, ungrouped, unflipped call with 32-aligned channels, `N·H·W ≥ 4096` and
    /// `C + O ≥ 256`).
    pub fn winograd(&self) -> bool {
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
    /// * Metal (`conv_2D_gpu`): `ensure_row_contiguous(wt)` copies the weight, which a diffusers
    ///   VAE holds as an NHWC transposed VIEW of its NCHW checkpoint tensor — at the activation
    ///   promoted width. The frontend dtype cast preserves dense transposed strides, so it is
    ///   counted separately from this layout preparation; Winograd (`winograd_conv_2D_gpu`) adds the
    ///   zero-padded input `[N, 6⌈(H+2p−2)/6⌉+2, …, C]`, the transformed input `[64, tiles, C]`,
    ///   output `[64, tiles, O]` and filter `[64, C, O]` (`tiles = N⌈oH/6⌉⌈oW/6⌉`).
    /// * CPU (`explicit_gemm_conv_ND_cpu`): an f32 padded input, the f32 im2col
    ///   `[N·oH·oW, kH·kW·C]`, an f32 weight copy (non-contiguous or non-f32), and an f32 GEMM
    ///   output when the activation is narrower.
    pub fn transient(&self, backend: Backend) -> u64 {
        let weight_copy = |width: u64| self.weight_elements().saturating_mul(width);
        let promoted = self.act.max(self.weight);
        // CPU and GPU AsType both preserve a dense transposed input's strides (Vector copy).
        // The frontend cast is therefore NOT the backend's later row-contiguous preparation.
        let frontend = if self.weight < promoted {
            weight_copy(promoted)
        } else {
            0
        } + if self.act < promoted {
            tensor_bytes(promoted, &[self.batch, self.in_h, self.in_w, self.cin])
        } else {
            0
        };
        let kernel = match backend {
            Backend::Metal => {
                let mut bytes = weight_copy(self.weight.max(self.act));
                if self.winograd() {
                    let padded = |n: u64| 6 * (n + 2 * self.pad).saturating_sub(2).div_ceil(6) + 2;
                    let tiles = self.batch * self.out_h.div_ceil(6) * self.out_w.div_ceil(6);
                    bytes = bytes
                        .saturating_add(tensor_bytes(
                            promoted,
                            &[self.batch, padded(self.in_h), padded(self.in_w), self.cin],
                        ))
                        .saturating_add(tensor_bytes(promoted, &[64, tiles, self.cin]))
                        .saturating_add(tensor_bytes(promoted, &[64, tiles, self.cout]))
                        .saturating_add(tensor_bytes(promoted, &[64, self.cin, self.cout]));
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
                if promoted != 4 {
                    bytes = bytes.saturating_add(tensor_bytes(4, &[rows, self.cout]));
                }
                bytes
            }
        };
        kernel.saturating_add(frontend)
    }
}

/// Buffers of the counted ops of the composed `mlx_gen::nn::group_norm` over its input: the
/// grouped-transpose reshape copy, the LayerNorm output, the transposed-back reshape copy, the weight
/// product and the bias sum (neither affine op can donate: their operand is the reshape copy's view
/// with a second consumer pending). Measured: 5.00 input-sized buffers at f32 (and on Metal at any
/// width — see [`group_norm_buffers`]).
pub const GROUP_NORM_BUFFERS: u64 = 5;

/// Extra input-sized buffers, in units of the activation width `act`, that one `mlx_rs::fast::
/// layer_norm` (alone, or inside [`GROUP_NORM_BUFFERS`]' composed GroupNorm) holds on `backend`
/// beyond its f32 count when the activations are narrower than f32. MLX's CPU backend
/// runs the fast op's composed fallback in f32 — `astype(x, f32)` plus the centred input, its square
/// and the normalized product at f32 — so an f16 norm holds three f32 input-sized buffers more than
/// its width-scaled f32 count (measured at f16: GroupNorm 11.0, LayerNorm 9.06 input-sized buffers,
/// against 5.0 / 3.03 at f32). Metal's fused kernel keeps the f32 statistics in registers: none.
pub fn norm_promotion_units(backend: Backend, act: u64) -> u64 {
    match backend {
        Backend::Cpu if act < 4 => 3 * 4 / act.max(1),
        _ => 0,
    }
}

/// [`GROUP_NORM_BUFFERS`] on `backend` at activation width `act` (+ [`norm_promotion_units`]).
pub fn group_norm_buffers(backend: Backend, act: u64) -> u64 {
    GROUP_NORM_BUFFERS + norm_promotion_units(backend, act)
}

/// MLX's Metal allocator (also the macOS CPU backend's) rounds every buffer above one page up to
/// the 16 KiB `vm_page_size`: at most one page of slack per counted buffer.
pub const PAGE_BYTES: u64 = super::budget::ALLOCATOR_PAGE_BYTES;

/// The allocator's page rounding over `buffers` counted buffers.
pub fn page_slack(buffers: u64) -> u64 {
    buffers.saturating_mul(PAGE_BYTES)
}

/// A convolutional quantum, counted under MLX's retention: a quantum's graph is evaluated in one
/// `eval`, and every primitive's INPUT buffers stay referenced until its command buffer (Metal) or
/// queued task (CPU) completes, so the quantum's intermediates are live together — the peak is the
/// SUM of its new buffers (an elementwise op whose operand is a single-consumer temporary donates and
/// adds none) plus every kernel's transients. `in_units` / `out_units` count the new buffers at
/// `call`'s input / output extents; `convs` are the kernels whose transients the quantum holds.
pub fn conv_sum(
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

/// A resnet (`ResnetBlock2D` forward, U-Net or VAE) over input `x` `[B,H,W,Cx]` at `act_in` bytes
/// per element, whose `conv1` takes `cin` (= Cx, or Cx + skip after a U-Net up path's concat) to
/// `cout`, and whose second half runs at `act_out` (wider than `act_in` when a projected time
/// embedding promotes the sum: GroupNorm 2, SiLU, `conv2` and the residual follow it). Counted new
/// buffers (see [`conv_sum`]): input width — the skip concatenation (up path), GroupNorm 1
/// ([`group_norm_buffers`]), SiLU, `conv1` and the 1x1 shortcut (when `cin != cout`); output width
/// — the promoted time-embedding sum (only when it widens: otherwise it donates `conv1`),
/// GroupNorm 2, SiLU, `conv2`; the bias adds and the residual sum donate. Plus both convolutions'
/// kernel transients (`conv2`'s at `act_out`).
pub fn resnet_bytes(
    x: &[i32],
    act_in: u64,
    act_out: u64,
    conv1: &Array,
    conv2: &Array,
    backend: Backend,
) -> u64 {
    resnet_bytes_from_metadata(
        x,
        act_in,
        act_out,
        ConvOperandMetadata {
            shape: conv1.shape(),
            element_bytes: element_bytes(conv1),
        },
        ConvOperandMetadata {
            shape: conv2.shape(),
            element_bytes: element_bytes(conv2),
        },
        backend,
    )
}

/// A convolution weight's actual OHWI geometry and PRECAST bytes per element (metadata only).
#[derive(Clone, Copy, Debug)]
pub struct ConvOperandMetadata<'a> {
    pub shape: &'a [i32],
    pub element_bytes: u64,
}

/// Exact resnet_bytes formula over actual OHWI geometry and PRECAST bytes PER ELEMENT.
/// Metadata is not backing identity, a physical/RSS bound or a new shape-validation policy.
pub fn resnet_bytes_from_metadata(
    x: &[i32],
    act_in: u64,
    act_out: u64,
    conv1: ConvOperandMetadata<'_>,
    conv2: ConvOperandMetadata<'_>,
    backend: Backend,
) -> u64 {
    let (conv1_shape, conv1_element_bytes) = (conv1.shape, conv1.element_bytes);
    let (conv2_shape, conv2_element_bytes) = (conv2.shape, conv2.element_bytes);
    let (cin, cout) = (conv1_shape[conv1_shape.len() - 1], conv1_shape[0]);
    let act_out = act_out.max(act_in);
    let input = [x[0], x[1], x[2], cin];
    let c1 = ConvCall::from_metadata(&input, act_in, conv1_shape, conv1_element_bytes, 1, 1);
    let mid = [x[0], x[1], x[2], conv2_shape[conv2_shape.len() - 1]];
    let c2 = ConvCall::from_metadata(&mid, act_out, conv2_shape, conv2_element_bytes, 1, 1);
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
pub fn single_conv_bytes(call: &ConvCall, in_units: u64, extra_out: u64, backend: Backend) -> u64 {
    conv_sum(call, in_units, 1 + extra_out, &[call], backend)
}

/// GroupNorm + SiLU + one convolution (the `conv_out` heads): GroupNorm and SiLU at input width,
/// the convolution's output and transient.
pub fn norm_conv_bytes(call: &ConvCall, backend: Backend) -> u64 {
    conv_sum(
        call,
        group_norm_buffers(backend, call.act) + 1,
        1,
        &[call],
        backend,
    )
}

/// Row count of one VAE mid-attention query chunk under the iOS bounded attention: the largest that
/// keeps one chunk's `[B, 1, rows, N]` scores within [`VAE_ATTENTION_SCORE_ELEMENTS`].
pub const VAE_ATTENTION_SCORE_ELEMENTS: u64 = 16 * 1024 * 1024;

/// The VAE mid-block single-head attention over `[B, N, C]`, `query_rows` queries per SDPA call
/// (the whole `N` on the host path, one chunk on iOS). Head dim = `C` (512 in an AutoencoderKL):
/// never a fused Metal shape, so the scores materialize per call on every backend.
///
/// On iOS the projections are materialized before the attention, so the price is the wider of two
/// phases, each counted as new buffers under [`conv_sum`]'s retention model:
/// * projections — GroupNorm ([`group_norm_buffers`]), q, k, v (hidden-sized);
/// * attention — q, k, v, the fallback's scaled query, the SDPA output and, chunked, the
///   concatenation of the per-chunk outputs, the output projection (hidden-sized; the residual sum
///   donates); a chunk's scores (the single-head softmax donates them in place: measured one
///   score tensor) and its scaled query / output rows —
///   each chunk is evaluated before the next is built, which releases the previous chunk's scores.
pub fn vae_attention_bytes(
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

/// Where one tile's `[B, h, w, C]` crop comes from in `mlx_gen::vae_tiling`'s tiled 3x3
/// convolution, and what runs on it before the convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSource {
    /// Width of the full activation the crop is sliced from (`x.shape()[2]` of the tiled call).
    pub full_width: i32,
    /// The crop is normalized by `vae_tiling::GlobalGroupNorm::apply` and SiLU-activated before
    /// the convolution (a resnet's conv1/conv2, a decoder's `conv_out`); `false` for a raw crop
    /// (`conv_in`, an upsample convolution).
    pub normalized: bool,
}

/// One tile of `mlx_gen::vae_tiling`'s tiled 3x3 convolution over a `[B, h, w, C]` crop, counted
/// under MLX's retention (the tile's graph is one `eval`, so its intermediates are live together —
/// see [`conv_sum`]):
///
/// * the crop's materialization: a ROW STRIP `[B, h, full_width, C]` (the halo crop is a strided
///   slice of the full activation; MLX materializes it through the full-width rows) and the
///   contiguous crop itself, at the input width;
/// * `normalized`: the GlobalGroupNorm apply's grouped copy, centred and transposed-back buffers
///   (three f32 crops), and below f32 also the f32 normalized product and the cast back to the
///   storage width;
/// * the `f32_accum` promoted crop (f32);
/// * the convolution output, its bias sum and the core copy (output width, f32 under
///   `f32_accum`), plus the core's cast back to the storage width under `f32_accum`;
/// * the kernel transient ([`ConvCall::transient`]) and an allocator page per buffer.
///
/// Measured (`vae_tiling::tests::tile_formula_tracks_the_measured_tile`, stalled runs on the CPU
/// backend): f32 and f16, normalized and raw, `f32_accum` on and off, crops from 23% to 100% of the
/// row width. The previous `max(norm phase, conv phase)` form under-counted every tile by the row
/// strip, the bias sum and (normalized) the norm buffers live beside the convolution — ~5 crops on a
/// 258² f32 crop of a 288-wide activation.
pub fn tile_bytes(
    tile: &[i32],
    source: TileSource,
    act: u64,
    weight: &Array,
    f32_accum: bool,
    backend: Backend,
) -> u64 {
    tile_bytes_from_metadata(
        tile,
        source,
        act,
        weight.shape(),
        element_bytes(weight),
        f32_accum,
        backend,
    )
}

/// The identical existing raw guard formula over exact operand metadata; no native graph,
/// new rank/cap/policy, allocation identity or physical-transfer certificate.
pub fn tile_bytes_from_metadata(
    tile: &[i32],
    source: TileSource,
    act: u64,
    weight_shape: &[i32],
    weight_element_bytes: u64,
    f32_accum: bool,
    backend: Backend,
) -> u64 {
    let conv_act = if f32_accum { 4 } else { act };
    let call = ConvCall::from_metadata(tile, conv_act, weight_shape, weight_element_bytes, 1, 1);
    let (b, h, w, c) = (u(tile[0]), u(tile[1]), u(tile[2]), u(tile[3]));
    let crop = tensor_bytes(act, &[b, h, w, c]);
    let strip = tensor_bytes(act, &[b, h, u(source.full_width).max(w), c]);
    let crop_f32 = tensor_bytes(4, &[b, h, w, c]);
    let norm = if source.normalized {
        let narrow = if act < 4 {
            crop_f32.saturating_add(crop)
        } else {
            0
        };
        crop_f32.saturating_mul(3).saturating_add(narrow)
    } else {
        0
    };
    let promoted = if f32_accum { call.in_bytes() } else { 0 };
    let cast_back = if f32_accum {
        tensor_bytes(act, &[call.batch, call.out_h, call.out_w, call.cout])
    } else {
        0
    };
    strip
        .saturating_add(crop)
        .saturating_add(norm)
        .saturating_add(promoted)
        .saturating_add(call.out_bytes().saturating_mul(3))
        .saturating_add(cast_back)
        .saturating_add(call.transient(backend))
        .saturating_add(page_slack(16))
}

/// One layer-wise tiled VAE stage producing `[B, H, W, cout]` from a `[B, H, W, cin]` input: the
/// GlobalGroupNorm statistics' grouped copy of the input, the evaluated row strips plus their
/// concatenation (two output-width images), and (resnet) the first convolution's output, the
/// shortcut and the sum. `resnet` selects the resnet layer's extra full-resolution buffers; the
/// per-tile scratch is checked separately ([`tile_bytes`]).
pub fn tiled_stage_bytes(out: &[i32], cin: i32, cout: i32, act: u64, resnet: bool) -> u64 {
    let plane = |c: i32| tensor_bytes(act, &[u(out[0]), u(out[1]), u(out[2]), u(c)]);
    let outputs = if resnet { 3 } else { 2 };
    plane(cin).saturating_add(plane(cout).saturating_mul(outputs))
}

/// A bounded encode's tiled stride-2 downsample stage producing `out` `[B, oH, oW, C]`: the
/// evaluated row strips and their concatenation — two output-width images (its input is already
/// materialized). Each tile is checked by [`downsample_tile_bytes`].
pub fn downsample_stage_bytes(out: &[i32], act: u64) -> u64 {
    tensor_bytes(act, &[2, u(out[0]), u(out[1]), u(out[2]), u(out[3])])
        .saturating_add(page_slack(2))
}

/// One tile of a bounded encode's stride-2 downsample over a `[B, h, w, C]` crop (`h`/`w` its exact
/// halo, `2·core + 1`): the bottom/right zero-padded crop copy, the stride-2 convolution's output
/// core (its bias add donates) and the kernel transient.
pub fn downsample_tile_bytes(crop: &[i32], act: u64, weight: &Array, backend: Backend) -> u64 {
    downsample_tile_bytes_from_metadata(crop, act, weight.shape(), element_bytes(weight), backend)
}

/// The identical existing raw guard formula over exact operand metadata; no native graph,
/// new rank/cap/policy, allocation identity or physical-transfer certificate.
pub fn downsample_tile_bytes_from_metadata(
    crop: &[i32],
    act: u64,
    weight_shape: &[i32],
    weight_element_bytes: u64,
    backend: Backend,
) -> u64 {
    let call = ConvCall::from_metadata(crop, act, weight_shape, weight_element_bytes, 2, 0);
    single_conv_bytes(&call, 1, 0, backend)
}

/// [`tiled_stage_bytes`] for an f16-storage upsample convolution that accumulates in f32
/// (`tiled_conv2d_3x3_nhwc_f32_accum`): the f32 copies of the conv weight and bias are built once
/// and stay live across every tile of the stage, on top of the stage's activations. Each tile's own
/// promoted crop and f32 output are priced by [`tile_bytes`].
pub fn tiled_stage_f32_accum(out: &[i32], weight: &Array, bias: &Array, act: u64) -> u64 {
    tiled_stage_f32_accum_from_metadata(out, weight.shape(), elements(weight), elements(bias), act)
}

/// The identical existing raw guard formula over exact operand metadata; no native graph,
/// new rank/cap/policy, allocation identity or physical-transfer certificate.
pub fn tiled_stage_f32_accum_from_metadata(
    out: &[i32],
    weight_shape: &[i32],
    weight_elements: u64,
    bias_elements: u64,
    act: u64,
) -> u64 {
    let (cin, cout) = (weight_shape[weight_shape.len() - 1], weight_shape[0]);
    tiled_stage_bytes(out, cin, cout, act, false)
        .saturating_add(weight_elements.saturating_mul(4))
        .saturating_add(bias_elements.saturating_mul(4))
}

/// The decoded image's pixel conversion (`mlx_gen::image::decoded_to_image`): one f32 image-sized
/// buffer (every op of the scale/clamp/round chain after the first donates; a half-precision VAE's
/// lazy f32 output cast is donated into by the first) and its allocator page.
pub fn decoded_to_image_bytes(decoded: &Array) -> u64 {
    decoded_to_image_bytes_from_metadata(decoded.size() as u64)
}

/// The identical existing raw guard formula over exact operand metadata; no native graph,
/// new rank/cap/policy, allocation identity or physical-transfer certificate.
pub fn decoded_to_image_bytes_from_metadata(decoded_elements: u64) -> u64 {
    super::elementwise_bytes(decoded_elements, 1).saturating_add(page_slack(1))
}

#[cfg(all(test, target_os = "macos"))]
mod metadata_constructor_tests {
    use super::*;
    // Independent OLD A6224 literal math, NOT the new helper used as its own oracle.
    const OLD_GROUP_NORM_BUFFERS: u64 = 5;
    const OLD_PAGE_BYTES: u64 = 16 * 1024;
    fn old_u(v: i32) -> u64 {
        v.max(0) as u64
    }

    fn old_product(values: &[u64]) -> u64 {
        values.iter().fold(1u64, |a, b| a.saturating_mul(*b))
    }

    pub fn old_tensor_bytes(element_bytes: u64, dims: &[u64]) -> u64 {
        old_product(&[old_product(dims), element_bytes])
    }

    pub fn old_norm_promotion_units(backend: Backend, act: u64) -> u64 {
        match backend {
            Backend::Cpu if act < 4 => 3 * 4 / act.max(1),
            _ => 0,
        }
    }

    pub fn old_group_norm_buffers(backend: Backend, act: u64) -> u64 {
        OLD_GROUP_NORM_BUFFERS + old_norm_promotion_units(backend, act)
    }

    pub fn old_page_slack(buffers: u64) -> u64 {
        buffers.saturating_mul(OLD_PAGE_BYTES)
    }

    fn old_conv_channels(s: &[i32]) -> (i32, i32) {
        (s[s.len() - 1], s[0])
    }

    #[derive(Clone, Copy)]
    struct OldConvCall {
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
        act: u64,
        weight: u64,
    }

    impl OldConvCall {
        /// `x`: the convolution's NHWC input shape; `weight`: the `[out, kH, kW, in]` weight it runs.
        fn new(x: &[i32], act: u64, w: &[i32], weight_width: u64, stride: i32, pad: i32) -> Self {
            let out = |n: i32, k: i32| old_u((n + 2 * pad - k) / stride.max(1) + 1);
            Self {
                batch: old_u(x[0]),
                in_h: old_u(x[1]),
                in_w: old_u(x[2]),
                cin: old_u(w[3]),
                out_h: out(x[1], w[1]),
                out_w: out(x[2], w[2]),
                cout: old_u(w[0]),
                kh: old_u(w[1]),
                kw: old_u(w[2]),
                stride: old_u(stride),
                pad: old_u(pad),
                act,
                weight: weight_width,
            }
        }

        /// Bytes of the call's input at its activation width.
        pub fn in_bytes(&self) -> u64 {
            old_tensor_bytes(self.act, &[self.batch, self.in_h, self.in_w, self.cin])
        }

        /// Bytes of the call's output at its activation width.
        pub fn out_bytes(&self) -> u64 {
            old_tensor_bytes(self.act, &[self.batch, self.out_h, self.out_w, self.cout])
        }

        /// Elements of the call's `[out, kH, kW, in]` weight.
        pub fn weight_elements(&self) -> u64 {
            [self.cout, self.kh, self.kw, self.cin]
                .iter()
                .fold(1u64, |a, b| a.saturating_mul(*b))
        }

        /// Whether MLX's Metal `dispatch_conv_2D_gpu` routes this call to `winograd_conv_2D_gpu`
        /// (a 3x3, stride-1, ungrouped, unflipped call with 32-aligned channels, `N·H·W ≥ 4096` and
        /// `C + O ≥ 256`).
        pub fn winograd(&self) -> bool {
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
        /// * Metal (`conv_2D_gpu`): `ensure_row_contiguous(wt)` copies the weight, which a diffusers
        ///   VAE holds as an NHWC transposed VIEW of its NCHW checkpoint tensor — at the activation
        ///   promoted width. The frontend dtype cast preserves dense transposed strides, so it is
        ///   counted separately from this layout preparation; Winograd (`winograd_conv_2D_gpu`) adds the
        ///   zero-padded input `[N, 6⌈(H+2p−2)/6⌉+2, …, C]`, the transformed input `[64, tiles, C]`,
        ///   output `[64, tiles, O]` and filter `[64, C, O]` (`tiles = N⌈oH/6⌉⌈oW/6⌉`).
        /// * CPU (`explicit_gemm_conv_ND_cpu`): an f32 padded input, the f32 im2col
        ///   `[N·oH·oW, kH·kW·C]`, an f32 weight copy (non-contiguous or non-f32), and an f32 GEMM
        ///   output when the activation is narrower.
        pub fn transient(&self, backend: Backend) -> u64 {
            let weight_copy = |width: u64| self.weight_elements().saturating_mul(width);
            let promoted = self.act.max(self.weight);
            // CPU and GPU AsType both preserve a dense transposed input's strides (Vector copy).
            // The frontend cast is therefore NOT the backend's later row-contiguous preparation.
            let frontend = if self.weight < promoted {
                weight_copy(promoted)
            } else {
                0
            } + if self.act < promoted {
                old_tensor_bytes(promoted, &[self.batch, self.in_h, self.in_w, self.cin])
            } else {
                0
            };
            let kernel = match backend {
                Backend::Metal => {
                    let mut bytes = weight_copy(self.weight.max(self.act));
                    if self.winograd() {
                        let padded =
                            |n: u64| 6 * (n + 2 * self.pad).saturating_sub(2).div_ceil(6) + 2;
                        let tiles = self.batch * self.out_h.div_ceil(6) * self.out_w.div_ceil(6);
                        bytes = bytes
                            .saturating_add(old_tensor_bytes(
                                promoted,
                                &[self.batch, padded(self.in_h), padded(self.in_w), self.cin],
                            ))
                            .saturating_add(old_tensor_bytes(promoted, &[64, tiles, self.cin]))
                            .saturating_add(old_tensor_bytes(promoted, &[64, tiles, self.cout]))
                            .saturating_add(old_tensor_bytes(promoted, &[64, self.cin, self.cout]));
                    }
                    bytes
                }
                Backend::Cpu => {
                    let rows = self.batch * self.out_h * self.out_w;
                    let mut bytes = old_tensor_bytes(
                        4,
                        &[
                            self.batch,
                            self.in_h + 2 * self.pad,
                            self.in_w + 2 * self.pad,
                            self.cin,
                        ],
                    )
                    .saturating_add(old_tensor_bytes(4, &[rows, self.kh * self.kw * self.cin]))
                    .saturating_add(weight_copy(4));
                    if promoted != 4 {
                        bytes = bytes.saturating_add(old_tensor_bytes(4, &[rows, self.cout]));
                    }
                    bytes
                }
            };
            kernel.saturating_add(frontend)
        }
    }
    struct OldResnetWidths {
        act_in: u64,
        act_out: u64,
        width1: u64,
        width2: u64,
    }
    fn old_resnet(
        x: &[i32],
        widths: OldResnetWidths,
        conv1: &[i32],
        conv2: &[i32],
        backend: Backend,
    ) -> u64 {
        let OldResnetWidths {
            act_in,
            act_out,
            width1,
            width2,
        } = widths;
        let (cin, cout) = old_conv_channels(conv1);
        let act_out = act_out.max(act_in);
        let input = [x[0], x[1], x[2], cin];
        let c1 = OldConvCall::new(&input, act_in, conv1, width1, 1, 1);
        let mid = [x[0], x[1], x[2], old_conv_channels(conv2).0];
        let c2 = OldConvCall::new(&mid, act_out, conv2, width2, 1, 1);
        let concat = u64::from(cin != x[3]);
        let shortcut = u64::from(cin != cout);
        let widened = u64::from(act_out > act_in);
        let in_units = concat + old_group_norm_buffers(backend, act_in) + 1;
        let narrow_out = 1 + shortcut;
        let wide_out = widened + old_group_norm_buffers(backend, act_out) + 1 + 1;
        let out = |width: u64, units: u64| {
            old_tensor_bytes(width, &[old_u(x[0]), old_u(x[1]), old_u(x[2]), old_u(cout)])
                .saturating_mul(units)
        };
        let buffers = in_units + narrow_out + wide_out + 10;
        c1.in_bytes()
            .saturating_mul(in_units)
            .saturating_add(out(act_in, narrow_out))
            .saturating_add(out(act_out, wide_out))
            .saturating_add(c1.transient(backend))
            .saturating_add(c2.transient(backend))
            .saturating_add(old_page_slack(buffers))
    }

    #[derive(Clone, Copy)]
    struct Case {
        x: [i32; 4],
        c1: [i32; 4],
        c2: [i32; 4],
    }
    const CASES: [Case; 5] = [
        Case {
            x: [1, 7, 9, 32],
            c1: [32, 3, 3, 32],
            c2: [32, 3, 3, 32],
        },
        Case {
            x: [1, 7, 9, 32],
            c1: [64, 3, 3, 64],
            c2: [64, 3, 3, 64],
        },
        Case {
            x: [1, 7, 9, 32],
            c1: [64, 3, 3, 32],
            c2: [64, 3, 3, 64],
        },
        Case {
            x: [1, 7, 9, 32],
            c1: [96, 3, 3, 64],
            c2: [96, 3, 3, 96],
        },
        Case {
            x: [1, 64, 64, 128],
            c1: [128, 3, 3, 128],
            c2: [128, 3, 3, 128],
        },
    ];
    const GEOMETRIES: [([i32; 4], [i32; 4], i32, i32); 3] = [
        ([1, 7, 9, 32], [64, 1, 1, 32], 1, 0),
        ([1, 7, 9, 32], [64, 3, 3, 32], 2, 1),
        ([1, 9, 11, 32], [64, 5, 3, 32], 2, 0),
    ];

    #[test]
    fn old_literal_formula_matches_metadata_and_preserves_last_axis() {
        for case in CASES {
            for (wi, wo) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
                for (ai, ao) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
                    for backend in [Backend::Cpu, Backend::Metal] {
                        // Defect map: lost concat/shortcut/widen/norm/transient/page formula term.
                        let expected = old_resnet(
                            &case.x,
                            OldResnetWidths {
                                act_in: ai,
                                act_out: ao,
                                width1: wi,
                                width2: wo,
                            },
                            &case.c1,
                            &case.c2,
                            backend,
                        );
                        assert_eq!(resnet_bytes_from_metadata(&case.x, ai, ao, ConvOperandMetadata { shape: &case.c1, element_bytes: wi }, ConvOperandMetadata { shape: &case.c2, element_bytes: wo }, backend), expected,
                            "ROOT_OLD_LITERAL_RESNET x={:?} weights={wi}/{wo} activation={ai}/{ao} backend={backend:?}", case.x);
                    }
                }
            }
        }
        for (x, w, stride, pad) in GEOMETRIES {
            for (act, width) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
                let old = OldConvCall::new(&x, act, &w, width, stride, pad);
                let actual = ConvCall::from_metadata(&x, act, &w, width, stride, pad);
                assert_eq!(
                    actual.in_bytes(),
                    old.in_bytes(),
                    "ROOT_OLD_LITERAL_CONV_INPUT"
                );
                assert_eq!(
                    actual.out_bytes(),
                    old.out_bytes(),
                    "ROOT_OLD_LITERAL_CONV_OUTPUT"
                );
                assert_eq!(
                    actual.weight_elements(),
                    old.weight_elements(),
                    "ROOT_OLD_LITERAL_CONV_WEIGHT"
                );
                assert_eq!(
                    actual.winograd(),
                    old.winograd(),
                    "ROOT_OLD_LITERAL_DISPATCH"
                );
                for backend in [Backend::Cpu, Backend::Metal] {
                    assert_eq!(
                        actual.transient(backend),
                        old.transient(backend),
                        "ROOT_OLD_LITERAL_TRANSIENT"
                    );
                }
            }
        }
        let x = [1, 7, 9, 32];
        let c1 = [64, 3, 3, 32, 64];
        let c2 = [64, 3, 3, 64, 32];
        assert_eq!(old_conv_channels(&c1).0, 64, "ROOT_OLD_LAST_AXIS");
        assert_eq!(
            ConvCall::from_metadata(&x, 2, &c1, 2, 1, 1).cin,
            32,
            "ROOT_CONVCALL_AXIS3"
        );
        for backend in [Backend::Cpu, Backend::Metal] {
            let expected = old_resnet(
                &x,
                OldResnetWidths {
                    act_in: 2,
                    act_out: 4,
                    width1: 2,
                    width2: 4,
                },
                &c1,
                &c2,
                backend,
            );
            assert_eq!(
                resnet_bytes_from_metadata(
                    &x,
                    2,
                    4,
                    ConvOperandMetadata {
                        shape: &c1,
                        element_bytes: 2
                    },
                    ConvOperandMetadata {
                        shape: &c2,
                        element_bytes: 4
                    },
                    backend
                ),
                expected,
                "ROOT_ODD_METADATA_LAST_AXIS_NOT_AXIS3"
            );
            let wrong1 = [64, 3, 3, 32, 32];
            let wrong2 = [64, 3, 3, 64, 64];
            assert_ne!(
                expected,
                old_resnet(
                    &x,
                    OldResnetWidths {
                        act_in: 2,
                        act_out: 4,
                        width1: 2,
                        width2: 4
                    },
                    &wrong1,
                    &wrong2,
                    backend
                ),
                "ROOT_ODD_CASE_MUST_DISCRIMINATE_AXIS3"
            );
        }
        let c = CASES[0];
        let correct = old_resnet(
            &c.x,
            OldResnetWidths {
                act_in: 4,
                act_out: 4,
                width1: 2,
                width2: 2,
            },
            &c.c1,
            &c.c2,
            Backend::Cpu,
        );
        assert_ne!(
            correct,
            resnet_bytes_from_metadata(
                &c.x,
                4,
                4,
                ConvOperandMetadata {
                    shape: &c.c1,
                    element_bytes: 4
                },
                ConvOperandMetadata {
                    shape: &c.c2,
                    element_bytes: 4
                },
                Backend::Cpu
            ),
            "ROOT_PRECAST_NOT_PROMOTED_WIDTH"
        );
        let nbytes = old_u(c.c1[0]) * old_u(c.c1[1]) * old_u(c.c1[2]) * old_u(c.c1[3]) * 2;
        assert_ne!(
            correct,
            resnet_bytes_from_metadata(
                &c.x,
                4,
                4,
                ConvOperandMetadata {
                    shape: &c.c1,
                    element_bytes: nbytes
                },
                ConvOperandMetadata {
                    shape: &c.c2,
                    element_bytes: nbytes
                },
                Backend::Cpu
            ),
            "ROOT_ITEM_WIDTH_NOT_NBYTES"
        );
    }

    const CHILD: &str = "MLX_ROOT_METADATA_CPU_CHILD";
    const DONE: &str = "MLX_ROOT_METADATA_CPU_DONE";
    const FQN: &str = "ios_quantum::kernels::metadata_constructor_tests::actual_precast_weights_match_old_literal_formula_cpu_child";
    const CASE_MASK: u64 = (1u64 << 36) - 1;
    struct Completion {
        cases: u64,
        geometry: u8,
    }
    impl Drop for Completion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(self.cases, CASE_MASK, "ROOT_NATIVE_CASE_MASK");
                assert_eq!(self.geometry, 7, "ROOT_NATIVE_GEOMETRY_MASK");
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "ROOT_METADATA_FINAL_CPU_SYNC"
                );
                eprintln!(
                    "ROOT_METADATA_BODY_COMPLETED_V1 native_cases=36 geometries=3 cpu_sync=0"
                );
                std::fs::write(
                    std::env::var(DONE).expect("ROOT_METADATA_PARENT_DONE"),
                    b"ok",
                )
                .unwrap();
            }
        }
    }
    fn strict_cpu_child() -> bool {
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            assert!(
                marker.is_absolute() && !marker.exists(),
                "ROOT_METADATA_FRESH_PARENT_DONE"
            );
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "ROOT_METADATA_CHILD_STATUS");
            assert!(marker.is_file(), "ROOT_METADATA_CHILD_BODY_MISSING");
            assert_eq!(
                std::fs::read(marker).unwrap().as_slice(),
                b"ok",
                "ROOT_METADATA_CHILD_DONE_CONTENT"
            );
            return false;
        }
        let marker = std::path::PathBuf::from(
            std::env::var(DONE).expect("ROOT_METADATA_REQUIRES_PARENT_DONE"),
        );
        assert!(marker.is_absolute(), "ROOT_METADATA_DONE_ABSOLUTE");
        assert!(!marker.exists(), "ROOT_METADATA_DONE_FRESH");
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu(),
            "ROOT_METADATA_CPU_DEVICE"
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0,
            "ROOT_METADATA_CPU_DEFAULT_STREAM"
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap(),
            "ROOT_METADATA_CPU_STREAM_READBACK"
        );
        assert!(
            crate::ios_quantum::quantum_metal_guard::forbid_metal(),
            "ROOT_METADATA_METAL_TRAP"
        );
        assert!(
            crate::ios_quantum::quantum_metal_guard::metal_forbidden(),
            "ROOT_METADATA_METAL_TRAP_READBACK"
        );
        eprintln!(
            "ROOT_METADATA_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1"
        );
        true
    }
    const MAX_ELEMENTS: usize = 200_000;
    const MAX_CASE_SOURCE_BYTES: usize = 16 * 1024 * 1024;
    fn count(shape: &[i32]) -> usize {
        shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(usize::try_from(*d).unwrap()))
            .unwrap()
    }
    fn fixture(shape: &[i32], dtype: Dtype) -> Array {
        let n = count(shape);
        assert!(n <= MAX_ELEMENTS, "ROOT_METADATA_STATIC_ELEMENT_BUDGET");
        let values = vec![0.125f32; n];
        let value = Array::from_slice(&values, shape).as_dtype(dtype).unwrap();
        value.eval().unwrap();
        value
    }
    fn expected_width(dtype: Dtype) -> u64 {
        match dtype {
            Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Float32 => 4,
            _ => panic!("ROOT_METADATA_UNEXPECTED_TEST_DTYPE"),
        }
    }
    #[test]
    fn actual_precast_weights_match_old_literal_formula_cpu_child() {
        let mut completion: Option<Completion> = None;
        if !strict_cpu_child() {
            return;
        }
        let mut cases = 0u64;
        let mut index = 0;
        {
            // Four SMALL valid rank4 geometries only. Winograd's threshold geometry is HOST math.
            for case in &CASES[..4] {
                for (d1, d2) in [
                    (Dtype::Float16, Dtype::Bfloat16),
                    (Dtype::Bfloat16, Dtype::Float32),
                    (Dtype::Float32, Dtype::Float16),
                ] {
                    for (ai, ao) in [
                        (Dtype::Float16, Dtype::Bfloat16),
                        (Dtype::Float16, Dtype::Float32),
                        (Dtype::Float32, Dtype::Bfloat16),
                    ] {
                        let mid_shape = [case.x[0], case.x[1], case.x[2], case.c2[3]];
                        let source_work = (count(&case.x)
                            + count(&mid_shape)
                            + count(&case.c1)
                            + count(&case.c2))
                        .checked_mul(12)
                        .unwrap();
                        assert!(
                            source_work <= MAX_CASE_SOURCE_BYTES,
                            "ROOT_METADATA_STATIC_CASE_BYTE_WORK"
                        );
                        let input = fixture(&case.x, ai);
                        let mid = fixture(&mid_shape, ao);
                        let w1 = fixture(&case.c1, d1);
                        let w2 = fixture(&case.c2, d2);
                        let wi = w1.item_size() as u64;
                        let wo = w2.item_size() as u64;
                        assert_eq!(
                            (wi, wo),
                            (expected_width(d1), expected_width(d2)),
                            "ROOT_NATIVE_ACTUAL_PRECAST_WIDTH"
                        );
                        assert_eq!(w1.shape(), &case.c1, "ROOT_NATIVE_WEIGHT1_SHAPE");
                        assert_eq!(w2.shape(), &case.c2, "ROOT_NATIVE_WEIGHT2_SHAPE");
                        let ain = input.item_size() as u64;
                        let aout = mid.item_size() as u64;
                        assert_eq!(
                            (ain, aout),
                            (expected_width(ai), expected_width(ao)),
                            "ROOT_NATIVE_ACTIVATION_WIDTH"
                        );
                        for backend in [Backend::Cpu, Backend::Metal] {
                            let expected = old_resnet(
                                input.shape(),
                                OldResnetWidths {
                                    act_in: ain,
                                    act_out: aout,
                                    width1: expected_width(d1),
                                    width2: expected_width(d2),
                                },
                                &case.c1,
                                &case.c2,
                                backend,
                            );
                            assert_eq!(
                                resnet_bytes(input.shape(), ain, aout, &w1, &w2, backend),
                                expected,
                                "ROOT_NATIVE_WRAPPER_OLD_LITERAL case={index} backend={backend:?}"
                            );
                            assert_eq!(
                                resnet_bytes_from_metadata(
                                    input.shape(),
                                    ain,
                                    aout,
                                    ConvOperandMetadata {
                                        shape: w1.shape(),
                                        element_bytes: wi
                                    },
                                    ConvOperandMetadata {
                                        shape: w2.shape(),
                                        element_bytes: wo
                                    },
                                    backend
                                ),
                                expected,
                                "ROOT_NATIVE_METADATA_OLD_LITERAL case={index} backend={backend:?}"
                            );
                        }
                        eprintln!("ROOT_METADATA_NATIVE_CASE_V1 index={index} weights={d1:?}/{d2:?} activation={ai:?}/{ao:?} source_work={source_work}");
                        cases |= 1u64 << index;
                        index += 1;
                    }
                }
            }
        }
        assert_eq!(index, 36, "ROOT_NATIVE_CASE_COUNT");
        let mut geometry = 0u8;
        {
            for (i, (x, shape, stride, pad)) in GEOMETRIES.into_iter().enumerate() {
                let input = fixture(&x, Dtype::Float32);
                let weight = fixture(&shape, [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32][i]);
                let width = weight.item_size() as u64;
                let old = OldConvCall::new(
                    input.shape(),
                    4,
                    &shape,
                    expected_width(weight.dtype()),
                    stride,
                    pad,
                );
                let actual = ConvCall::new(input.shape(), 4, &weight, stride, pad);
                assert_eq!(
                    actual.out_bytes(),
                    old.out_bytes(),
                    "ROOT_NATIVE_CONVCALL_OUTPUT_GEOMETRY"
                );
                assert_eq!(
                    actual.weight, width,
                    "ROOT_NATIVE_CONVCALL_PRECAST_NOT_NBYTES"
                );
                for backend in [Backend::Cpu, Backend::Metal] {
                    assert_eq!(
                        actual.transient(backend),
                        old.transient(backend),
                        "ROOT_NATIVE_CONVCALL_OLD_LITERAL_TRANSIENT"
                    );
                }
                geometry |= 1u8 << i;
            }
        }
        let _ = completion.replace(Completion { cases, geometry });
    }
}
