//! SW-6141 (feature SW-F287 M3): a **speed-only POC** of SageAttention-style INT8 `Q·Kᵀ`
//! attention as custom Metal kernels, measured against MLX's fused
//! `fast::scaled_dot_product_attention`.
//!
//! Nothing in this module is on a default path: no provider, attention route or engine option
//! calls it. It exists to answer one question with numbers — does INT8 `Q·Kᵀ` beat MLX's fused
//! SDPA on the Mac at DiT shapes — and the registered proof
//! `int8_attention_metal::tests::int8_attention_metal_parity_bench_and_go_no_go` prints the
//! answer (`SW6141_VERDICT`). MLX's fused SDPA never materializes the `[B, H, Sq, Sk]` scores
//! (see [`crate::attention`]), so there is no memory case: the only possible win is time, and
//! only where the GPU has INT8 matrix hardware. Per the story, engine wiring (an opt-in, off by
//! default) lands only on a measured GO.
//!
//! # Arithmetic (the CUDA twin's, SW-6131)
//!
//! 1. K is optionally **centered** by its per-channel mean over the sequence (`mean(K, axis=S)`,
//!    MLX's deterministic reduction): every score of a query row shifts by the same constant,
//!    which the softmax cancels exactly.
//! 2. Q and K rows are rotated by the same **signed Walsh–Hadamard** transform over the 128
//!    channels (the CUDA twin's sign diagonal, then the orthonormal `H_128`). The rotation is
//!    orthogonal, so `q·k` is unchanged while an outlier channel is spread over the row.
//! 3. Each block of [`Int8AttentionConfig::block_tokens`] rows (default 32: per-block, as the story
//!    specifies; 1 = per-token, the CUDA twin's granularity) shares one INT8 scale
//!    `max(amax / 127, 1e-12)`, codes rounded to nearest even and clamped to `±127`.
//! 4. The attention kernel computes `Q·Kᵀ` on the INT8 codes, dequantizes with the two row scales
//!    (the Q scale also carrying `1/sqrt(D) · log2(e)`), runs a streaming (online) softmax in FP32
//!    with `exp2`, and accumulates `P·V` in FP32 with V kept in its own bf16/f16 dtype. No score
//!    tensor is ever materialized.
//!
//! # Routes
//!
//! * [`QkRoute::Simd`] — every Apple GPU: `simdgroup_matrix` FP32 products over the INT8 codes as
//!   exact integers (`|Σ| <= 128 · 127² < 2²⁴`). Apple's SIMD-group matrix units have no INT8 mode,
//!   so this route can only save K/Q load bandwidth, never matrix throughput; it is the portable
//!   correctness baseline.
//! * [`QkRoute::Nax`] — the M5-generation neural accelerators (`mpp::tensor_ops::matmul2d`, Metal
//!   4), `Q·Kᵀ` as INT8 × INT8 → INT32. This is the only route on which INT8 could be faster than
//!   MLX's own NAX SDPA. [`attend_nax_unquantized`] is the same kernel on the bf16/f16 operands
//!   (FP32 accumulation, no quantization): the control that separates "INT8 helps" from "this
//!   kernel's tiling is better or worse than MLX's". [`nax_layout_probe`] checks the one assumption
//!   the INT8 NAX kernel makes beyond the in-tree 16-bit NAX kernels: that INT8 cooperative tensors
//!   use the 16-bit `BaseNAXFrag` element layout; [`nax_layout_probe_arrays`] runs the same INT8
//!   product over plain thread arrays, so a compile failure of the 8-wide INT8 vector form is not
//!   mistaken for "no INT8 `matmul2d`".
//!
//! # Envelope (checked by the pure [`check`] before anything launches)
//!
//! Rank-4 `[B, H, S, D]` row-contiguous operands (MLX makes a contiguous copy otherwise), `D = 128`,
//! one bf16 or f16 dtype for Q, K and V, matching `B`/`H`/`D` and K/V lengths (`Sq` may differ from
//! `Sk`), no mask, no causal mode, and every operand at most `i32::MAX` elements (E6). Typed
//! [`Refusal`]s, never a silent fallback: this module has no reference fallback because it has no
//! production caller.
//!
//! # Provenance (E10)
//!
//! Original Metal source written for this repository from the in-tree CUDA twin's arithmetic
//! (SW-6131, `candle_quant_kernels::sage_attention`) and the thu-ml SageAttention *algorithm*
//! (arXiv 2410.02367; no SageAttention or comfy-kitchen source is copied). The SIMD-group kernel
//! follows the in-tree Veda Metal attention (`mlx-gen-minimax-h3/src/veda/metal.rs`); the NAX
//! helpers are the in-tree SC-20676 / Veda `BaseNAXFrag` helpers (MLX's fragment layout),
//! generalised over the accumulator type. The sources are `include_str!`-ed `.metal` fragments in
//! `src/int8_attention_metal/`, JIT-compiled by MLX through `mlx_rs::fast::MetalKernel` (E12).

use std::cell::OnceCell;

use mlx_rs::fast::{scaled_dot_product_attention, MetalKernel};
use mlx_rs::{Array, Dtype};

use crate::{Error, Result};

const HEADER: &str = include_str!("int8_attention_metal/header.metal");
const NAX_HEADER: &str = include_str!("int8_attention_metal/nax_header.metal");
const QUANTIZE: &str = include_str!("int8_attention_metal/quantize.metal");
const ATTEND_SIMD: &str = include_str!("int8_attention_metal/attend_simd.metal");
const ATTEND_NAX: &str = include_str!("int8_attention_metal/attend_nax.metal");
const NAX_PROBE: &str = include_str!("int8_attention_metal/nax_probe.metal");
const NAX_ARRAY_HEADER: &str = include_str!("int8_attention_metal/nax_array_header.metal");
const NAX_PROBE_ARRAY: &str = include_str!("int8_attention_metal/nax_probe_array.metal");

/// The only head dimension the kernels are specialised for (every DiT this POC targets).
pub const HEAD_DIM: usize = 128;
/// Largest number of rows one quantization scale may span (one threadgroup of 32 SIMD groups).
pub const MAX_BLOCK_TOKENS: usize = 32;
/// Rows per quantization scale by default: per-block, as the story specifies.
pub const DEFAULT_BLOCK_TOKENS: usize = 32;
/// Query rows one SIMD-group-route threadgroup serves (four SIMD groups of eight rows).
pub const SIMD_QUERY_ROWS: usize = 32;
/// Threads per SIMD-group-route threadgroup.
pub const SIMD_THREADS: usize = 128;
/// Query rows one NAX-route threadgroup serves (eight SIMD groups of sixteen rows).
pub const NAX_QUERY_ROWS: usize = 128;
/// Threads per NAX-route threadgroup.
pub const NAX_THREADS: usize = 256;
/// The signed Walsh–Hadamard sign diagonal: channel `c` is negated when bit `c & 31` of word
/// `c >> 5` is clear. The CUDA twin's `SAGE_SIGN_WORDS` (SW-6131), mirrored in `header.metal`.
pub const SIGN_WORDS: [u32; 4] = [0x1035_997b, 0x8087_f5ee, 0xee2e_4e1a, 0x7113_2418];

/// Which matrix unit runs `Q·Kᵀ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkRoute {
    /// `simdgroup_matrix` FP32 products over exact-integer INT8 codes (every Apple GPU).
    Simd,
    /// INT8 × INT8 → INT32 `matmul2d` on the neural accelerators (M5 generation).
    Nax,
}

impl QkRoute {
    /// Stable name for tables and telemetry.
    pub const fn name(self) -> &'static str {
        match self {
            QkRoute::Simd => "simd",
            QkRoute::Nax => "nax",
        }
    }
}

/// Quantization choices of one INT8 attention call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8AttentionConfig {
    /// Rows sharing one INT8 scale, `1..=`[`MAX_BLOCK_TOKENS`].
    pub block_tokens: usize,
    /// Subtract K's per-channel sequence mean before quantizing (softmax-invariant).
    pub center_k: bool,
    /// Rotate Q (must equal `rotate_k` for a correct result; the flags exist for the mutation
    /// test).
    pub rotate_q: bool,
    /// Rotate K.
    pub rotate_k: bool,
}

impl Default for Int8AttentionConfig {
    fn default() -> Self {
        Self {
            block_tokens: DEFAULT_BLOCK_TOKENS,
            center_k: true,
            rotate_q: true,
            rotate_k: true,
        }
    }
}

/// Why a call is outside the kernels' envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An operand is not rank 4.
    Rank { operand: &'static str, rank: usize },
    /// Operand shapes disagree (or a dimension is negative).
    Shape(String),
    /// The head dimension is not [`HEAD_DIM`].
    HeadDim(usize),
    /// Operands are not one bf16/f16 dtype.
    Dtype(String),
    /// A zero-sized operand.
    Empty,
    /// An operand exceeds `i32::MAX` elements, or a dispatch dimension exceeds `i32` (E6).
    IndexOverflow,
    /// `block_tokens` outside `1..=MAX_BLOCK_TOKENS`.
    BlockTokens(usize),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Rank { operand, rank } => {
                write!(f, "{operand} must be [B, H, S, D], got rank {rank}")
            }
            Refusal::Shape(why) => write!(f, "operand shapes disagree: {why}"),
            Refusal::HeadDim(d) => write!(f, "head dim {d} is not {HEAD_DIM}"),
            Refusal::Dtype(why) => write!(f, "operand dtypes: {why}"),
            Refusal::Empty => write!(f, "empty operand"),
            Refusal::IndexOverflow => write!(f, "operand exceeds the i32 index range"),
            Refusal::BlockTokens(n) => {
                write!(f, "block_tokens {n} is outside 1..={MAX_BLOCK_TOKENS}")
            }
        }
    }
}

impl From<Refusal> for Error {
    fn from(refusal: Refusal) -> Self {
        Error::Unsupported(format!("SW-6141 INT8 attention: {refusal}"))
    }
}

/// The admitted geometry of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub batch: usize,
    pub heads: usize,
    pub q_len: usize,
    pub k_len: usize,
}

impl Plan {
    /// `(batch, head)` rows of the dispatch grid.
    pub const fn rows(&self) -> usize {
        self.batch * self.heads
    }
}

fn dims(shape: &[i32], operand: &'static str) -> std::result::Result<[usize; 4], Refusal> {
    let [b, h, s, d] = shape else {
        return Err(Refusal::Rank {
            operand,
            rank: shape.len(),
        });
    };
    let mut out = [0usize; 4];
    for (slot, &value) in out.iter_mut().zip([b, h, s, d]) {
        *slot = usize::try_from(value)
            .map_err(|_| Refusal::Shape(format!("{operand} has a negative dimension")))?;
    }
    Ok(out)
}

/// Validate one call (pure; E5): shapes, dtypes, block size and index ranges.
pub fn check(
    q: (&[i32], Dtype),
    k: (&[i32], Dtype),
    v: (&[i32], Dtype),
    config: &Int8AttentionConfig,
) -> std::result::Result<Plan, Refusal> {
    let [qb, qh, qs, qd] = dims(q.0, "q")?;
    let kd = dims(k.0, "k")?;
    let vd = dims(v.0, "v")?;
    if qd != HEAD_DIM {
        return Err(Refusal::HeadDim(qd));
    }
    if kd != vd {
        return Err(Refusal::Shape(format!("k {kd:?} vs v {vd:?}")));
    }
    if [kd[0], kd[1], kd[3]] != [qb, qh, qd] {
        return Err(Refusal::Shape(format!(
            "q {:?} vs k {kd:?} (batch, heads and head dim must match)",
            [qb, qh, qs, qd]
        )));
    }
    if q.1 != k.1 || q.1 != v.1 {
        return Err(Refusal::Dtype(format!(
            "q {:?}, k {:?}, v {:?} must be one dtype",
            q.1, k.1, v.1
        )));
    }
    if !matches!(q.1, Dtype::Bfloat16 | Dtype::Float16) {
        return Err(Refusal::Dtype(format!("{:?} is not bf16/f16", q.1)));
    }
    if !(1..=MAX_BLOCK_TOKENS).contains(&config.block_tokens) {
        return Err(Refusal::BlockTokens(config.block_tokens));
    }
    let plan = Plan {
        batch: qb,
        heads: qh,
        q_len: qs,
        k_len: kd[2],
    };
    if plan.rows() == 0 || plan.q_len == 0 || plan.k_len == 0 {
        return Err(Refusal::Empty);
    }
    let limit = i32::MAX as usize;
    let elements = |len: usize| {
        plan.rows()
            .checked_mul(len)
            .and_then(|x| x.checked_mul(HEAD_DIM))
    };
    for len in [plan.q_len, plan.k_len] {
        if elements(len).is_none_or(|n| n > limit) {
            return Err(Refusal::IndexOverflow);
        }
    }
    // Every dispatch dimension is an i32 thread count.
    let grid_x = |((x, _, _), _): Dispatch| x;
    let grids = [
        grid_x(quantize_dispatch(
            plan.rows(),
            plan.q_len.max(plan.k_len),
            1,
        )),
        grid_x(attend_dispatch(QkRoute::Simd, plan.rows(), plan.q_len)),
        grid_x(attend_dispatch(QkRoute::Nax, plan.rows(), plan.q_len)),
    ];
    if grids.iter().any(|&g| g <= 0) || plan.rows() > limit {
        return Err(Refusal::IndexOverflow);
    }
    Ok(plan)
}

/// `(grid, threadgroup)` thread counts of one dispatch, in MLX's `MetalKernel` convention.
pub type Dispatch = ((i32, i32, i32), (i32, i32, i32));

/// `usize -> i32` for dispatch sizes; an overflow becomes `-1`, which [`check`] refuses.
fn grid_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(-1)
}

/// `(grid, threadgroup)` of the quantizer: one threadgroup of `32 · block` threads per block of
/// `block` rows, blocks along `x`, `(batch, head)` rows along `y`.
pub fn quantize_dispatch(rows: usize, len: usize, block: usize) -> Dispatch {
    let threads = 32 * block;
    (
        (
            grid_i32(len.div_ceil(block).saturating_mul(threads)),
            grid_i32(rows),
            1,
        ),
        (grid_i32(threads), 1, 1),
    )
}

/// `(grid, threadgroup)` of an attention kernel: one threadgroup per query tile along `x`,
/// `(batch, head)` rows along `y`.
pub fn attend_dispatch(route: QkRoute, rows: usize, q_len: usize) -> Dispatch {
    let (tile, threads) = match route {
        QkRoute::Simd => (SIMD_QUERY_ROWS, SIMD_THREADS),
        QkRoute::Nax => (NAX_QUERY_ROWS, NAX_THREADS),
    };
    (
        (
            grid_i32(q_len.div_ceil(tile).saturating_mul(threads)),
            grid_i32(rows),
            1,
        ),
        (threads as i32, 1, 1),
    )
}

struct Kernels {
    quantize: MetalKernel,
    simd: MetalKernel,
    nax: MetalKernel,
    probe: MetalKernel,
    probe_array: MetalKernel,
}

thread_local! {
    // `MetalKernel` is thread-affine; each thread builds its own handles (the mlx-layers INT8 GEMM
    // and Veda convention). The handles capture no arrays and MLX caches the compiled pipelines
    // itself, keyed on name, source and template arguments, so a second thread does not recompile.
    // Construction does not compile Metal: MLX compiles each instantiation on its first dispatch,
    // so a GPU or compiler without INT8 `matmul2d` fails only the NAX dispatches.
    static KERNELS: OnceCell<std::result::Result<Kernels, String>> = const { OnceCell::new() };
}

fn with_kernels<T>(f: impl FnOnce(&Kernels) -> Result<T>) -> Result<T> {
    KERNELS.with(|cell| {
        let kernels = cell
            .get_or_init(|| {
                let nax_header = format!("{HEADER}{NAX_HEADER}");
                let array_header = format!("{nax_header}{NAX_ARRAY_HEADER}");
                let make =
                    |name: &str, inputs: &[&str], outputs: &[&str], body: &str, header: &str| {
                        MetalKernel::with_options(
                            &format!("sw6141_int8_attention_{name}_v1"),
                            inputs,
                            outputs,
                            body,
                            header,
                            true,
                            false,
                        )
                        .map_err(|e| e.to_string())
                    };
                const ATTEND_INPUTS: [&str; 6] = ["q", "k", "v", "sq", "sk", "params"];
                Ok(Kernels {
                    quantize: make(
                        "quantize",
                        &["x", "shift", "params"],
                        &["codes", "scales"],
                        QUANTIZE,
                        HEADER,
                    )?,
                    simd: make("attend_simd", &ATTEND_INPUTS, &["out"], ATTEND_SIMD, HEADER)?,
                    nax: make(
                        "attend_nax",
                        &ATTEND_INPUTS,
                        &["out"],
                        ATTEND_NAX,
                        &nax_header,
                    )?,
                    probe: make(
                        "nax_probe",
                        &["pa", "pb", "ha", "hb"],
                        &["ci", "cf"],
                        NAX_PROBE,
                        &nax_header,
                    )?,
                    probe_array: make(
                        "nax_probe_array",
                        &["pa", "pb"],
                        &["ci"],
                        NAX_PROBE_ARRAY,
                        &array_header,
                    )?,
                })
            })
            .as_ref()
            .map_err(|e| Error::Msg(format!("SW-6141 Metal kernels: {e}")))?;
        f(kernels)
    })
}

fn params(values: &[i32]) -> Array {
    Array::from_slice(values, &[values.len() as i32])
}

fn shape4(a: &Array) -> Result<[usize; 4]> {
    dims(a.shape(), "operand").map_err(Error::from)
}

/// Whether MLX runs its neural-accelerator kernels in this process (MLX's own predicate, via
/// [`mlx_llm::primitives::mlx_nax_available`]); `false` when the device probe fails.
pub fn nax_available() -> bool {
    mlx_llm::primitives::mlx_nax_available().unwrap_or(false)
}

/// K's per-channel mean over the sequence, `[B, H, S, D] -> f32 [B, H, D]` (MLX's deterministic
/// mean). Any per-(batch, head) vector is softmax-invariant to subtract; the mean is the one that
/// shrinks the quantization range.
pub fn k_channel_mean(k: &Array) -> Result<Array> {
    Ok(k.mean_axis(2, false)?.as_dtype(Dtype::Float32)?)
}

/// Rotate (optionally), subtract `mean` and quantize `x: [B, H, S, 128]` to INT8 with one scale per
/// `block_tokens` rows: `(codes I8 [B, H, S, 128], scales F32 [B, H, S])` (the block scale repeated
/// on each of its rows). `mean` is F32 `[B, H, 128]`.
pub fn quantize_rows(
    x: &Array,
    mean: &Array,
    block_tokens: usize,
    rotate: bool,
) -> Result<(Array, Array)> {
    let [b, h, s, d] = shape4(x)?;
    if d != HEAD_DIM {
        return Err(Refusal::HeadDim(d).into());
    }
    if !(1..=MAX_BLOCK_TOKENS).contains(&block_tokens) {
        return Err(Refusal::BlockTokens(block_tokens).into());
    }
    if mean.shape() != [b as i32, h as i32, d as i32] || mean.dtype() != Dtype::Float32 {
        return Err(Refusal::Shape(format!(
            "mean must be f32 [{b}, {h}, {d}], got {:?} {:?}",
            mean.dtype(),
            mean.shape()
        ))
        .into());
    }
    let ((gx, gy, gz), (tx, ty, tz)) = quantize_dispatch(b * h, s, block_tokens);
    let (bi, hi, si, di) = (b as i32, h as i32, s as i32, d as i32);
    with_kernels(|kernels| {
        let mut out = kernels
            .quantize
            .apply()
            .inputs([x, mean, &params(&[si])])
            .output_shape([bi, hi, si, di], Dtype::Int8)
            .output_shape([bi, hi, si], Dtype::Float32)
            .template_arg("QB", block_tokens as i32)
            .template_arg("ROTATE", rotate)
            .grid(gx, gy, gz)
            .thread_group(tx, ty, tz)
            .run()?
            .into_iter();
        let codes = out
            .next()
            .ok_or_else(|| Error::Msg("SW-6141 quantizer returned no codes".into()))?;
        let scales = out
            .next()
            .ok_or_else(|| Error::Msg("SW-6141 quantizer returned no scales".into()))?;
        Ok((codes, scales))
    })
}

/// INT8 codes and per-row scales of Q and K. Holds no reference to the floating-point Q/K, so a
/// caller may drop them before [`attend_quantized`] runs.
pub struct QuantizedQk {
    pub q_codes: Array,
    pub q_scales: Array,
    pub k_codes: Array,
    pub k_scales: Array,
    pub plan: Plan,
}

/// Quantize Q and K for [`attend_quantized`] (validates the call first).
pub fn prequantize(
    q: &Array,
    k: &Array,
    v: &Array,
    config: &Int8AttentionConfig,
) -> Result<QuantizedQk> {
    let plan = check(
        (q.shape(), q.dtype()),
        (k.shape(), k.dtype()),
        (v.shape(), v.dtype()),
        config,
    )?;
    let zeros = Array::zeros::<f32>(&[plan.batch as i32, plan.heads as i32, HEAD_DIM as i32])?;
    let k_mean = if config.center_k {
        k_channel_mean(k)?
    } else {
        zeros.clone()
    };
    let (q_codes, q_scales) = quantize_rows(q, &zeros, config.block_tokens, config.rotate_q)?;
    let (k_codes, k_scales) = quantize_rows(k, &k_mean, config.block_tokens, config.rotate_k)?;
    Ok(QuantizedQk {
        q_codes,
        q_scales,
        k_codes,
        k_scales,
        plan,
    })
}

fn dispatch_attend(
    route: QkRoute,
    quantized: bool,
    inputs: [&Array; 5],
    plan: Plan,
    out_dtype: Dtype,
) -> Result<Array> {
    let [q, k, v, sq, sk] = inputs;
    let p = params(&[plan.q_len as i32, plan.k_len as i32]);
    let ((gx, gy, gz), (tx, ty, tz)) = attend_dispatch(route, plan.rows(), plan.q_len);
    let shape = [
        plan.batch as i32,
        plan.heads as i32,
        plan.q_len as i32,
        HEAD_DIM as i32,
    ];
    with_kernels(|kernels| {
        let dispatch = match route {
            QkRoute::Simd => kernels.simd.apply(),
            QkRoute::Nax => kernels
                .nax
                .apply()
                .template_arg("QUANT", i32::from(quantized)),
        };
        let mut out = dispatch
            .inputs([q, k, v, sq, sk, &p])
            .output_shape(shape, out_dtype)
            .grid(gx, gy, gz)
            .thread_group(tx, ty, tz)
            .run()?;
        Ok(out.remove(0))
    })
}

/// Attention over prequantized Q/K: `[B, H, Sq, 128]` in V's dtype.
pub fn attend_quantized(qk: &QuantizedQk, v: &Array, route: QkRoute) -> Result<Array> {
    dispatch_attend(
        route,
        true,
        [&qk.q_codes, &qk.k_codes, v, &qk.q_scales, &qk.k_scales],
        qk.plan,
        v.dtype(),
    )
}

/// The NAX kernel on the 16-bit operands themselves (FP32 accumulation, no rotation, no
/// quantization): the control row of the bench.
pub fn attend_nax_unquantized(q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let plan = check(
        (q.shape(), q.dtype()),
        (k.shape(), k.dtype()),
        (v.shape(), v.dtype()),
        &Int8AttentionConfig::default(),
    )?;
    let unit = Array::from_slice(&[1.0_f32], &[1]);
    dispatch_attend(
        QkRoute::Nax,
        false,
        [q, k, v, &unit, &unit],
        plan,
        v.dtype(),
    )
}

/// The whole INT8 attention call: center/rotate/quantize Q and K, then attend.
pub fn int8_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    config: &Int8AttentionConfig,
    route: QkRoute,
) -> Result<Array> {
    let qk = prequantize(q, k, v, config)?;
    attend_quantized(&qk, v, route)
}

/// MLX's fused SDPA at the same shape and scale: the baseline every row is measured against.
pub fn fused_sdpa(q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let scale = (HEAD_DIM as f32).powf(-0.5);
    Ok(scaled_dot_product_attention(q, k, v, scale, None, None)?)
}

/// `(column, row)` of lane `lane`'s first element of a 16×16 `BaseNAXFrag` fragment (the lane
/// holds rows `row` and `row + 8`, columns `column..column + 4`), as `sw6141_nax_coord`.
pub const fn nax_coord(lane: usize) -> (usize, usize) {
    let qid = lane >> 2;
    let row = (qid & 4) | ((lane >> 1) & 3);
    let column = ((qid & 2) | (lane & 1)) * 4;
    (column, row)
}

/// `(row, column)` of element `i < 8` of lane `lane`'s 16×16 fragment.
pub const fn nax_element(lane: usize, i: usize) -> (usize, usize) {
    let (column, row) = nax_coord(lane);
    (row + 8 * (i / 4), column + i % 4)
}

/// The probe product under the assumed `BaseNAXFrag` layout, lane-major like the kernel's output:
/// `A [16, 16]` from `pa` (8 per lane), `Bᵀ [32, 16]` from `pb` (16 per lane: rows `0..16` then
/// `16..32`), `C = A · B` with `C[m][n] = Σ_k A[m][k] · Bᵀ[n][k]`.
pub fn assumed_probe_product(pa: &[i8], pb: &[i8]) -> Vec<f32> {
    assert_eq!((pa.len(), pb.len()), (256, 512));
    let mut a = [[0f32; 16]; 16];
    let mut bt = [[0f32; 16]; 32];
    for lane in 0..32 {
        for i in 0..8 {
            let (r, c) = nax_element(lane, i);
            a[r][c] = f32::from(pa[lane * 8 + i]);
            bt[r][c] = f32::from(pb[lane * 16 + i]);
            bt[16 + r][c] = f32::from(pb[lane * 16 + 8 + i]);
        }
    }
    let mut out = vec![0f32; 512];
    for lane in 0..32 {
        for i in 0..16 {
            let (m, n) = nax_element(lane, i % 8);
            let n = n + 16 * (i / 8);
            out[lane * 16 + i] = (0..16).map(|kk| a[m][kk] * bt[n][kk]).sum();
        }
    }
    out
}

/// What [`nax_layout_probe`] observed.
#[derive(Clone, Debug)]
pub struct LayoutProbe {
    /// The INT8 × INT8 → INT32 product, lane-major.
    pub int8: Vec<i32>,
    /// The bf16 × bf16 → FP32 product of the same per-lane values.
    pub sixteen_bit: Vec<f32>,
    /// [`assumed_probe_product`] of the inputs.
    pub assumed: Vec<f32>,
}

impl LayoutProbe {
    /// The INT8 cooperative tensors use the 16-bit element layout.
    pub fn int8_matches_sixteen_bit(&self) -> bool {
        self.int8
            .iter()
            .zip(&self.sixteen_bit)
            .all(|(&i, &f)| i as f32 == f)
    }

    /// The 16-bit product matches the host's `BaseNAXFrag` model (validates the model itself).
    pub fn sixteen_bit_matches_assumed(&self) -> bool {
        self.sixteen_bit == self.assumed
    }
}

/// Deterministic probe values in `[-8, 8]` (every product and 16-term sum exact in bf16/FP32).
pub fn probe_values(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 17) as i8 - 8
        })
        .collect()
}

/// The probe's `(pa, pb)` values for `seed`.
pub fn probe_inputs(seed: u64) -> (Vec<i8>, Vec<i8>) {
    (
        probe_values(256, seed),
        probe_values(512, seed ^ 0x9e37_79b9_7f4a_7c15),
    )
}

/// Run one SIMD group's INT8 and 16-bit `matmul2d` on the same per-lane values (NAX devices).
pub fn nax_layout_probe(seed: u64) -> Result<LayoutProbe> {
    let (pa, pb) = probe_inputs(seed);
    let as_bf16 = |values: &[i8]| -> Result<Array> {
        let floats: Vec<f32> = values.iter().map(|&x| f32::from(x)).collect();
        Ok(Array::from_slice(&floats, &[floats.len() as i32]).as_dtype(Dtype::Bfloat16)?)
    };
    let (ha, hb) = (as_bf16(&pa)?, as_bf16(&pb)?);
    let (ia, ib) = (
        Array::from_slice(&pa, &[256]),
        Array::from_slice(&pb, &[512]),
    );
    let mut out = with_kernels(|kernels| {
        Ok(kernels
            .probe
            .apply()
            .inputs([&ia, &ib, &ha, &hb])
            .output_shape([512], Dtype::Int32)
            .output_shape([512], Dtype::Float32)
            .grid(32, 1, 1)
            .thread_group(32, 1, 1)
            .run()?)
    })?;
    let cf = out
        .pop()
        .ok_or_else(|| Error::Msg("probe: no f32".into()))?;
    let ci = out
        .pop()
        .ok_or_else(|| Error::Msg("probe: no i32".into()))?;
    mlx_rs::transforms::eval([&ci, &cf])?;
    Ok(LayoutProbe {
        int8: ci.as_slice::<i32>().to_vec(),
        sixteen_bit: cf.as_slice::<f32>().to_vec(),
        assumed: assumed_probe_product(&pa, &pb),
    })
}

/// The INT8 `matmul2d` of [`nax_layout_probe`] over plain thread arrays instead of 8-wide INT8
/// vectors (a diagnostic: separates "no INT8 cooperative tensors" from "no 8-wide INT8 vectors").
/// Returns the lane-major INT32 product, comparable with [`assumed_probe_product`] of
/// [`probe_inputs`].
pub fn nax_layout_probe_arrays(seed: u64) -> Result<Vec<i32>> {
    let (pa, pb) = probe_inputs(seed);
    let (ia, ib) = (
        Array::from_slice(&pa, &[256]),
        Array::from_slice(&pb, &[512]),
    );
    let ci = with_kernels(|kernels| {
        Ok(kernels
            .probe_array
            .apply()
            .inputs([&ia, &ib])
            .output_shape([512], Dtype::Int32)
            .grid(32, 1, 1)
            .thread_group(32, 1, 1)
            .run()?
            .remove(0))
    })?;
    mlx_rs::transforms::eval([&ci])?;
    Ok(ci.as_slice::<i32>().to_vec())
}

#[cfg(test)]
mod tests;
