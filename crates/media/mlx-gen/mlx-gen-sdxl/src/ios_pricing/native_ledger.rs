//! SDXL cumulative native-allocator backing primitives; not RSS or physical footprint.
//!
//! Immutable authorities: inference SDXL 212a6d6 (`authority`), MLX ITSNSQ/mlx-src,
//! mlx-rs fe22998. Paths in citations below are relative to those roots. The allocator
//! law is fused-native-backing-authority-v2.json, independently visible in
//! mlx/backend/metal/allocator.cpp:107-231 and backend/common/buffer_cache.h:30-46.
//! It applies to the witnessed iOS build's CPU stream too: its global allocator is
//! the linked Metal allocator, NOT a hypothetical malloc-only CPU build.
//!
//! This is a cumulative upper: charge each possible request independently through U.
//! Never retire storage for eval, a barrier, a drop, command completion, or donation.
//! Views do not allocate; copy-producing consumers do. Include conditional layout
//! repairs even when one actual execution could omit them. CPU and Metal workspace
//! laws stay separate. Without an architecture witness, charge both applicable
//! Metal Split-K alternatives (never an invented blanket workspace multiplier).
//! No threadgroup/kernel-local storage, BLAS internals, host Vec, or encoder metadata
//! is MLX allocator backing. No native-to-process transfer is asserted here.
//!
//! Shared reference: Iris ios_pricing/native_ledger.rs:29-219,251-308,714-733,
//! 1184-1205. SDXL additions: mixed-affine LayerNorm, composed/fused GroupNorm,
//! native affine quantization/dequantization, and CPU/Metal convolution lowerings.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeBackend {
    Cpu,
    Metal,
}

fn overflow() -> Error {
    Error::Unsupported("SDXL cumulative native allocation arithmetic overflow".into())
}
fn invalid(what: &str) -> Error {
    Error::Unsupported(format!("SDXL native ledger: {what}"))
}
fn product(dims: &[u64]) -> Result<u64> {
    dims.iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b).ok_or_else(overflow))
}
fn plus(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(overflow)
}
fn ceil(n: u64, d: u64) -> Result<u64> {
    if d == 0 {
        return Err(overflow());
    }
    plus(n / d, u64::from(!n.is_multiple_of(d)))
}
macro_rules! p {
    ($($d:expr),+ $(,)?) => {
        product(&[$(u64::try_from($d).map_err(|_| overflow())?),+])?
    };
}

/// U(0)=0; otherwise r + min(r,2*page), with r=n at/below one page and
/// r=ceil(n/page)*page above it. Cache reuse is operand-local, not a universal
/// round of the final cumulative sum. Every intermediate overflow fails closed.
fn backing_upper(n: u64, page: u64) -> Result<u64> {
    if !page.is_power_of_two() {
        return Err(invalid("invalid Metal VM page"));
    }
    if n == 0 {
        return Ok(0);
    }
    let r = if n <= page {
        n
    } else {
        p!(ceil(n, page)?, page)
    };
    let slack = product(&[2, page])?;
    plus(r, r.min(slack))
}

pub(super) struct NativeLedger {
    pub(super) page: u64,
    pub(super) backend: NativeBackend,
    pub(super) total: u64,
}
impl NativeLedger {
    pub(super) fn new(page: u64, backend: NativeBackend) -> Result<Self> {
        // Reject even empty/non-numeric/non-positive values: no execution witness
        // binds this process-global input (metal/scaled_dot_product_attention.cpp:477).
        if backend == NativeBackend::Metal && std::env::var_os("MLX_SDPA_BLOCKS").is_some() {
            return Err(Error::Unsupported("SDXL native certificate: MLX_SDPA_BLOCKS override is not bound by any execution witness".into()));
        }
        backing_upper(0, page)?;
        Ok(Self {
            page,
            backend,
            total: 0,
        })
    }
    pub(super) fn bytes(&mut self, bytes: u64) -> Result<()> {
        self.total = plus(self.total, backing_upper(bytes, self.page)?)?;
        Ok(())
    }
    pub(super) fn array(&mut self, width: u64, elements: u64) -> Result<()> {
        self.bytes(p!(width, elements))
    }
    pub(super) fn repeated(&mut self, width: u64, elements: u64, count: u64) -> Result<()> {
        let one = backing_upper(p!(width, elements), self.page)?;
        self.total = plus(self.total, p!(one, count))?;
        Ok(())
    }
    pub(super) fn scalar(&mut self, width: u64) -> Result<()> {
        self.array(width, 1)
    }
    /// Dtype equality, not merely width equality, is the caller's responsibility:
    /// an F16<->BF16 same-width conversion must instead call array(2,n).
    pub(super) fn cast(&mut self, n: u64, from: u64, to: u64) -> Result<()> {
        if from != to {
            self.array(to, n)?;
        }
        Ok(())
    }
    /// Non-donated output; broadcasts/views themselves allocate nothing.
    pub(super) fn op(&mut self, width: u64, n: u64) -> Result<()> {
        self.array(width, n)
    }
    /// common/common.cpp:190-199: possible logical-shape copy, not a retirement.
    pub(super) fn reshape_copy(&mut self, width: u64, n: u64) -> Result<()> {
        self.array(width, n)
    }
    pub(super) fn concat(&mut self, width: u64, n: u64, parts: u64) -> Result<()> {
        if parts == 0 {
            return Err(invalid("concatenate has no inputs"));
        }
        // ops.cpp concatenate returns its sole input verbatim.
        if parts > 1 {
            self.array(width, n)?;
        }
        Ok(())
    }

    /// Dense floating Matmul / AddMM A,B work; caller charges C/bias repair and
    /// eager bias epilogue. `width` is 2 or 4, not complex64. For rank-2 weights
    /// ops.cpp:3363-3366 folds all input rows into M; callers must reflect that.
    /// CPU matmul.cpp:69-162; Metal matmul.cpp:25-40,1248-1318,1359-1420.
    pub(super) fn matmul(&mut self, width: u64, batch: u64, m: u64, n: u64, k: u64) -> Result<()> {
        float_width(width)?;
        self.array(width, p!(batch, m, n))?;
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            // Metal's empty-input path creates a typed zero before fill_gpu.
            if self.backend == NativeBackend::Metal {
                self.scalar(width)?;
            }
            return Ok(());
        }
        self.array(width, p!(batch, m, k))?; // possible exact-shape A repair
        self.array(width, p!(batch, k, n))?; // possible exact-shape B repair
        if self.backend == NativeBackend::Metal {
            // With batch>1 the noncollapsed branch has no Split-K. Conditional
            // broadcast-batch collapse may still route to a single GEMM.
            self.metal_mm_workspace(p!(batch, m), n, k)?;
        }
        Ok(())
    }
    fn metal_mm_workspace(&mut self, m: u64, n: u64, k: u64) -> Result<()> {
        // Matmul::eval_gpu takes GEMV before calling steel_matmul for vector shapes.
        if m <= 1 || n <= 1 {
            return Ok(());
        }
        self.metal_steel_workspace(m, n, k)
    }
    fn metal_steel_workspace(&mut self, m: u64, n: u64, k: u64) -> Result<()> {
        // Convolution's direct steel_matmul call does NOT take the Matmul GEMV
        // shortcut, so even M=1 or N=1 must retain applicable Split-K scratch.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        // SIMD Split-K, matmul.cpp:501-543,911-942. Architecture threshold max=2048.
        let tiles16 = p!(ceil(m, 16)?, ceil(n, 16)?);
        if tiles16 <= 2048 && k / 16 >= 8 && k >= m.max(n) {
            let tiles32 = p!(ceil(m, 32)?, ceil(n, 32)?);
            let split = (k / 16 / tiles32)
                .checked_next_power_of_two()
                .ok_or_else(overflow)?
                .clamp(2, 32);
            self.array(4, p!(split, m, n))?;
        }
        // NAX Split-K, matmul.cpp:658-703,946-949. Distinct possible hardware path.
        let edge = m.max(n);
        if k >= p!(3, edge) || (edge <= 1024 && k > p!(2, edge)) {
            let step = if k <= 1024 {
                k / 2
            } else if k <= 2048 {
                1024
            } else if k <= 4096 {
                2048
            } else {
                4096
            };
            self.array(4, p!(ceil(k, step)?, m, n))?;
        }
        Ok(())
    }

    /// Affine, transpose=true Q4/Q8 linear (not arbitrary QVM/QQMatmul). `codes`
    /// counts stored U32 words; `scales` counts each of the two affine parameters.
    /// Caller owns dtype casts/padding and any separate output-bias epilogue.
    /// CPU quantized.cpp:924-970; Metal quantized.cpp:53-125,891-923,1501-1585.
    pub(super) fn qmm(
        &mut self,
        width: u64,
        rows: u64,
        input: u64,
        out: u64,
        packed: [u64; 3],
    ) -> Result<()> {
        let [codes, scales, group] = packed;
        float_width(width)?;
        if input == 0 || out == 0 || group == 0 || !input.is_multiple_of(group) {
            return Err(invalid("invalid affine QMM geometry"));
        }
        self.array(width, p!(rows, out))?;
        self.array(width, p!(rows, input))?;
        self.array(4, codes)?;
        self.repeated(width, scales, 2)?;
        if self.backend == NativeBackend::Metal && rows >= 6 {
            // Every architecture's transposed QMV threshold is >=6. Include
            // QMM splitk whenever it can be selected; no scratch for QMV itself.
            let denom = p!(ceil(out, 32)?, ceil(rows, 32)?);
            let mut parts = (512 / denom).max(1).min(input / group);
            while parts > 1 && !input.is_multiple_of(p!(parts, group)) {
                parts -= 1;
            }
            if parts > 1 {
                self.array(width, p!(parts, rows, out))?;
            }
        }
        Ok(())
    }

    /// Native affine Quantize: one possible input repair and three distinct sibling
    /// outputs. CPU quantized.cpp:1233-1285, Metal quantized.cpp:1801-1827. No eager
    /// fallback graph is selected on these two backends. SDXL AdaptableLinear first
    /// casts weight/bias to BF16 (mlx-gen/src/adapters.rs:842-874); caller charges it.
    pub(super) fn quantize(
        &mut self,
        width: u64,
        rows: u64,
        input: u64,
        group: u64,
        bits: u64,
    ) -> Result<()> {
        float_width(width)?;
        let (n, words, groups) = quant_grid(rows, input, group, bits)?;
        self.array(width, n)?;
        self.array(4, words)?;
        self.repeated(width, groups, 2)
    }
    /// Last-axis softmax: CPU softmax.cpp:117-145, Metal softmax.cpp:24-44.
    /// Non-donated output OR layout repair shared by output, not both. Floating
    /// inputs only; caller handles non-last-axis explicit softmax graphs.
    pub(super) fn softmax(&mut self, width: u64, n: u64) -> Result<()> {
        float_width(width)?;
        self.array(width, n)
    }
    /// Same-dtype floating or 32-bit reduction. `last_axis=true` promises a
    /// row-contiguous last-axis input, not merely an axis index. Width-changing
    /// integer reductions must be represented with explicit input/output widths
    /// by the caller, not fed here as a same-dtype operation.
    /// CPU reduce.cpp:452; Metal reduce.cpp:328-361,640-655,818-826,931-1045.
    pub(super) fn reduce(
        &mut self,
        width: u64,
        input: u64,
        output: u64,
        extent: u64,
        last_axis: bool,
    ) -> Result<()> {
        float_width(width)?;
        if output == 0 || extent == 0 || p!(output, extent) != input {
            return Err(invalid("invalid reduction geometry"));
        }
        if self.backend == NativeBackend::Cpu {
            return self.array(width, output);
        }
        self.bytes(p!(width, output).max(4))?;
        if output == 1 && input > 4096 {
            let rows = if p!(width, input) <= (1 << 26) {
                128
            } else {
                4096
            };
            self.array(width, rows)?;
        }
        if !last_axis {
            self.array(width, input)?; // possible GeneralReduce layout repair
            if extent >= 1024 {
                let outer = if extent >= 32768 { 128 } else { 32 };
                self.array(width, p!(outer, output))?;
            }
            if extent > 256 && output / 32 < 1024 {
                self.array(width, p!(32, output))?;
            }
        }
        Ok(())
    }

    /// Mixed-affine fast::LayerNorm. Width 0 denotes absent gain/bias. Caller
    /// computes promoted output dtype (F16 + BF16 is F32 despite equal widths).
    /// fast.cpp:200-285, Metal normalization.cpp:209-300. CPU fallback never uses
    /// the Metal fused norm's buffer law: its moments and arithmetic are explicit.
    pub(super) fn layernorm(
        &mut self,
        n: u64,
        rows: u64,
        input_width: u64,
        output_width: u64,
        gain_width: u64,
        bias_width: u64,
    ) -> Result<()> {
        float_width(input_width)?;
        float_width(output_width)?;
        if rows == 0 || n == 0 || !n.is_multiple_of(rows) {
            return Err(invalid("invalid LayerNorm rows"));
        }
        let dim = n / rows;
        for w in [gain_width, bias_width] {
            if w == 0 {
                self.scalar(output_width)?;
            } else {
                float_width(w)?;
                self.cast(dim, w, output_width)?;
            }
        }
        match self.backend {
            NativeBackend::Metal => {
                self.cast(n, input_width, output_width)?;
                self.array(output_width, n)?; // output OR shared layout repair
            }
            NativeBackend::Cpu => {
                self.cast(n, input_width, 4)?;
                self.mean(4, n, rows, dim, true)?;
                self.op(4, n)?; // centered x
                self.op(4, n)?; // square
                self.mean(4, n, rows, dim, true)?;
                self.scalar(4)?; // epsilon
                self.repeated(4, rows, 2)?; // epsilon add, rsqrt
                self.op(4, n)?; // normalized product
                self.cast(n, 4, output_width)?;
                if gain_width != 0 {
                    self.op(output_width, n)?;
                }
                if bias_width != 0 {
                    self.op(output_width, n)?;
                }
            }
        }
        Ok(())
    }
    fn mean(
        &mut self,
        width: u64,
        input: u64,
        output: u64,
        extent: u64,
        last_axis: bool,
    ) -> Result<()> {
        self.reduce(width, input, output, extent, last_axis)?;
        self.scalar(width)?; // ops.cpp:2201-2218 number_of_elements normalizer
        self.op(width, output)
    }
    /// NHWC shared nn::group_norm, not a hypothetical native CPU GroupNorm.
    /// mlx-gen/src/nn.rs:1031-1095 transposes/reshapes to [B,G,H*W*C/G], calls
    /// unaffined fast::LayerNorm, reshapes back, multiplies gain and adds bias.
    /// Both conditional input/output reshape materializations count independently.
    /// Caller supplies actual affine widths and owns dtype promotion. Equal
    /// widths must denote the same floating dtype; for an F16/BF16 mixture the
    /// caller first charges/converts affine leaves to F32 so max-width promotion
    /// below represents the real graph. This is the same width-only contract as cast.
    pub(super) fn groupnorm(
        &mut self,
        width: u64,
        shape: [u64; 4],
        groups: u64,
        affine: [u64; 2],
    ) -> Result<()> {
        let [b, h, w, channels] = shape;
        let [gain_width, bias_width] = affine;
        float_width(width)?;
        float_width(gain_width)?;
        float_width(bias_width)?;
        if groups == 0 || channels == 0 || !channels.is_multiple_of(groups) {
            return Err(invalid("invalid GroupNorm channel groups"));
        }
        let n = p!(b, h, w, channels);
        self.reshape_copy(width, n)?; // x->[B,HW,G,C/G]
        self.reshape_copy(width, n)?; // transposed->[B,G,-1]
        self.layernorm(n, p!(b, groups), width, width, 0, 0)?;
        self.reshape_copy(width, n)?; // normalized transpose back to NHWC
        let mul_width = width.max(gain_width);
        self.cast(n, width, mul_width)?;
        self.cast(channels, gain_width, mul_width)?;
        self.op(mul_width, n)?;
        let out_width = mul_width.max(bias_width);
        self.cast(n, mul_width, out_width)?;
        self.cast(channels, bias_width, out_width)?;
        self.op(out_width, n)?;
        if self.backend == NativeBackend::Metal {
            // Optional exact epilogue is a different native path. Source:
            // fast.cpp:368-449; normalization.cpp:302-405. Even when the
            // composed path is not selected, its sum above cannot underprice it.
            // Include wrapper contiguous repair, kernel repairs for x/gain/bias,
            // and non-donated result separately (safe overcount of shared copies).
            self.repeated(width, n, 3)?;
            self.array(gain_width, channels)?;
            self.array(bias_width, channels)?;
        }
        Ok(())
    }
    /// mlx-rs nn/activation.rs:876-880; pre-fusion sigmoid + multiply envelope.
    pub(super) fn silu(&mut self, width: u64, n: u64) -> Result<()> {
        self.repeated(width, n, 2)
    }
    /// fast::SDPA inference, equal Q/K/V head dimension, no sinks/logsumexp.
    /// Metal support predicate and workspaces: scaled_dot_product_attention.cpp:
    /// 440-504,591-641,658-786. CPU always uses fast.cpp:880-954's fallback;
    /// unsupported Metal geometry uses that graph with METAL primitive laws.
    /// `mask` denotes an additive floating array in the same dtype as Q/K/V;
    /// caller owns its cast/upload. Boolean masks require a scalar minimum in
    /// the fallback; this bound charges that scalar even for additive masks.
    pub(super) fn sdpa(&mut self, shape: [u64; 7], mask: bool, causal: bool) -> Result<()> {
        let [width, b, heads, kv_heads, q, k, dim] = shape;
        float_width(width)?;
        if b == 0
            || heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || q == 0
            || k == 0
            || dim == 0
        {
            return Err(invalid("invalid SDPA head geometry"));
        }
        let qn = p!(b, heads, q, dim);
        let kn = p!(b, kv_heads, k, dim);
        let scores = p!(b, heads, q, k);
        let full_mask = !causal || mask || q <= k;
        let fused = self.backend == NativeBackend::Metal
            && ((q > 8 && matches!(dim, 64 | 80 | 128) && full_mask)
                || (q <= 8
                    && q <= k
                    && matches!(dim, 64 | 96 | 128 | 256)
                    && p!(q, heads / kv_heads) <= 32));
        if fused {
            self.array(width, qn)?; // possible Q copy
            self.repeated(width, kn, 2)?; // separate K/V copies, no GQA expansion
            if mask {
                self.array(width, scores)?;
            } // wrapper-broadcast mask repair
            self.array(width, qn)?; // non-donated result
            if q <= 8 && k >= 1024 {
                let simds = p!(heads / kv_heads, q);
                let s_blocks: u64 = if k > 1024 && simds > 4 {
                    if k <= 8192 {
                        128
                    } else if k <= 32768 {
                        256
                    } else if k <= 65536 {
                        512
                    } else {
                        1024
                    }
                } else {
                    64
                };
                let d_blocks = if simds <= 2 && k > 8192 {
                    256
                } else if simds >= 6 && k >= 65536 {
                    1024
                } else if simds >= 6 && k >= 16384 {
                    512
                } else {
                    128
                };
                let blocks = s_blocks.max(d_blocks).max(if simds >= 4 { 64 } else { 32 });
                let partial = p!(b, heads, q, blocks);
                self.array(width, p!(partial, dim))?;
                self.repeated(4, partial, 2)?; // distinct sums/maxs
            }
            return Ok(());
        }
        self.scalar(width)?;
        self.op(width, qn)?; // scale Q
        self.matmul(width, p!(b, heads), q, k, dim)?;
        if causal {
            self.array(4, k)?;
            self.array(4, q)?;
            self.op(1, p!(q, k))?;
            self.scalar(width)?;
            self.op(width, scores)?; // where
        } else if mask {
            self.scalar(width)?; // covers a bool mask's min scalar as well
            self.op(width, scores)?;
        }
        self.softmax(width, scores)?;
        self.matmul(width, p!(b, heads), q, dim, k)
    }

    /// NHWC/OHWI, symmetric padding, input dilation=1, flip=false, no bias.
    /// Caller owns type promotion and bias (including possible fused bias copy).
    /// CPU conv.cpp:1004-1175,1218-1253,1298-1358; Metal conv.cpp:25-190,
    /// 766-960,1024-1103,1239-1284. Kernel-local/threadgroup storage is excluded.
    pub(super) fn conv2d(
        &mut self,
        shape: [u64; 10],
        stride: [u64; 2],
        pad: [u64; 2],
        dilation: [u64; 2],
        groups: u64,
    ) -> Result<()> {
        let [width, b, ih, iw, cin, cout, kh, kw, oh, ow] = shape;
        let [stride_h, stride_w] = stride;
        let [pad_h, pad_w] = pad;
        let [dilation_h, dilation_w] = dilation;
        float_width(width)?;
        if b == 0
            || ih == 0
            || iw == 0
            || cin == 0
            || cout == 0
            || kh == 0
            || kw == 0
            || stride_h == 0
            || stride_w == 0
            || dilation_h == 0
            || dilation_w == 0
            || groups == 0
            || !cin.is_multiple_of(groups)
            || !cout.is_multiple_of(groups)
        {
            return Err(invalid("invalid conv2d geometry"));
        }
        let ph = plus(ih, p!(2, pad_h))?;
        let pw = plus(iw, p!(2, pad_w))?;
        let effective_h = plus(p!(kh - 1, dilation_h), 1)?;
        let effective_w = plus(p!(kw - 1, dilation_w), 1)?;
        let eh = ph
            .checked_sub(effective_h)
            .ok_or_else(|| invalid("conv2d kernel exceeds padded input"))?;
        let ew = pw
            .checked_sub(effective_w)
            .ok_or_else(|| invalid("conv2d kernel exceeds padded input"))?;
        if oh != plus(eh / stride_h, 1)? || ow != plus(ew / stride_w, 1)? {
            return Err(invalid("conv2d output geometry mismatch"));
        }
        let output = p!(b, oh, ow, cout);
        let weight = p!(cout, kh, kw, cin / groups);
        self.array(width, output)?;
        if self.backend == NativeBackend::Cpu {
            // Dilation/grouped path is slow_conv_2D: no MLX scratch Arrays.
            if dilation_h == 1 && dilation_w == 1 && groups == 1 {
                self.scalar(4)?;
                self.array(4, p!(b, ph, pw, cin))?; // padded F32 input
                self.array(4, p!(b, oh, ow, kh, kw, cin))?; // materialized im2col
                self.array(4, weight)?; // F32/layout weight conversion (conditional)
                if width != 4 {
                    self.array(4, output)?;
                } // F32 gemm_out
            }
            return Ok(());
        }
        self.array(width, p!(b, ih, iw, cin))?; // possible input repair
        self.array(width, weight)?; // possible weight repair
        let cpg = cin / groups;
        let opg = cout / groups;
        let implicit_group =
            (cpg <= 4 || cpg.is_multiple_of(16)) && (opg <= 16 || opg.is_multiple_of(16));
        if groups > 1 {
            // Row-contiguous weight makes depthwise predicate deterministic;
            // depthwise/implicit kernels have no allocator-owned workspaces.
            let depthwise = cpg == 1
                && opg == 1
                && dilation_h == 1
                && dilation_w == 1
                && kh <= 7
                && kw <= 7
                && stride_h <= 2
                && stride_w <= 2
                && cin.is_multiple_of(16)
                && cin == cout;
            if !depthwise && !implicit_group {
                self.array(width, p!(b, oh, ow, kh, kw, cin))?; // unfold all groups
                self.array(width, weight)?; // grouped weight transpose
                                            // steel_matmul_regular has batch_size=groups>1: no Split-K.
            }
            return Ok(());
        }
        let winograd = stride_h == 1
            && stride_w == 1
            && dilation_h == 1
            && dilation_w == 1
            && kh == 3
            && kw == 3
            && cin.is_multiple_of(32)
            && cout.is_multiple_of(32)
            && p!(b, ih, iw) >= 4096
            && plus(cin, cout)? >= 256;
        if winograd {
            // conv.cpp:774-910, transformed 6x6 output tiles with 8x8 backing.
            let padded_h = plus(p!(6, ceil(ph.checked_sub(2).ok_or_else(overflow)?, 6)?), 2)?;
            let padded_w = plus(p!(6, ceil(pw.checked_sub(2).ok_or_else(overflow)?, 6)?), 2)?;
            self.scalar(width)?;
            self.array(width, p!(b, padded_h, padded_w, cin))?;
            self.array(width, p!(64, cin, cout))?;
            let tiles = p!(b, ceil(oh, 6)?, ceil(ow, 6)?);
            self.array(width, p!(64, tiles, cin))?;
            self.array(width, p!(64, tiles, cout))?;
            // Batched steel_matmul uses batch=64: no Split-K allocation.
            return Ok(());
        }
        let implicit =
            (cin <= 4 || cin.is_multiple_of(16)) && (cout <= 16 || cout.is_multiple_of(16));
        let general = (cin.is_multiple_of(16) && cout.is_multiple_of(16)) || p!(b, oh, ow) >= 256;
        if !implicit && !general {
            let m = p!(b, oh, ow);
            let k = p!(kh, kw, cin);
            self.array(width, p!(m, k))?; // explicit unfold
                                          // Explicit GEMM's weight is a view, operands already normalized;
                                          // only its actual hardware-dependent Split-K buffers remain.
            self.metal_steel_workspace(m, cout, k)?;
        }
        Ok(())
    }
}

fn float_width(width: u64) -> Result<()> {
    if matches!(width, 2 | 4) {
        Ok(())
    } else {
        Err(invalid("expected F16/BF16/F32 byte width"))
    }
}
fn quant_grid(rows: u64, input: u64, group: u64, bits: u64) -> Result<(u64, u64, u64)> {
    if rows == 0
        || input == 0
        || group == 0
        || !input.is_multiple_of(group)
        || !matches!(bits, 4 | 8)
        || !input.is_multiple_of(32 / bits)
    {
        return Err(invalid("invalid Q4/Q8 affine pack grid"));
    }
    let n = p!(rows, input);
    Ok((n, p!(rows, input / (32 / bits)), n / group))
}

#[cfg(test)]
mod primitive_tests {
    use super::*;

    /// HOST arithmetic only; no Array construction, MLX execution or native query.
    #[test]
    fn allocator_operand_upper_matches_request_and_reuse_law() {
        for (n, upper) in [
            (0, 0),
            (1, 2),
            (4, 8),
            (16_384, 32_768),
            (16_385, 65_536),
            (32_768, 65_536),
            (32_769, 81_920),
        ] {
            assert_eq!(backing_upper(n, 16_384).unwrap(), upper, "U({n})");
        }
        assert!(backing_upper(1, 1u64 << 63).is_err());
        // Construct directly: these tests must not depend on or mutate process env.
        let mut l = NativeLedger {
            page: 16_384,
            backend: NativeBackend::Cpu,
            total: 0,
        };
        l.repeated(1, 16_385, 2).unwrap();
        assert_eq!(l.total, 131_072);
        assert_eq!(backing_upper(2 * 16_385, 16_384).unwrap(), 81_920);
    }
    #[test]
    fn checked_arithmetic_and_ledger_overflow_fail_closed() {
        assert_eq!(product(&[]).unwrap(), 1);
        assert_eq!(product(&[0, u64::MAX]).unwrap(), 0);
        assert!(product(&[u64::MAX, 2]).is_err());
        assert!(plus(u64::MAX, 1).is_err());
        assert_eq!(ceil(u64::MAX, u64::MAX).unwrap(), 1);
        assert_eq!(ceil(u64::MAX, 2).unwrap(), 1u64 << 63);
        assert!(ceil(1, 0).is_err());
        assert!(backing_upper(0, 0).is_err());
        assert!(backing_upper(1, 3).is_err());
        assert!(backing_upper(u64::MAX, 16_384).is_err());
        assert!(backing_upper(1u64 << 63, 1u64 << 63).is_err());
        let mut l = NativeLedger {
            page: 16_384,
            backend: NativeBackend::Cpu,
            total: 0,
        };
        assert!(l.array(4, u64::MAX).is_err());
        assert_eq!(l.total, 0);
        assert!(l.repeated(4, 1, u64::MAX).is_err());
        assert_eq!(l.total, 0);
        l.total = u64::MAX;
        assert!(l.bytes(1).is_err());
        assert_eq!(l.total, u64::MAX);
    }
    #[test]
    fn checked_shape_macro_rejects_negative_dimensions() {
        fn dimensions(d: i32) -> Result<u64> {
            Ok(p!(4u64, d))
        }
        assert_eq!(dimensions(3).unwrap(), 12);
        assert!(dimensions(-1).is_err());
    }
}

impl NativeLedger {
    /// Float32 random::normal with no loc/scale arguments. Source random.cpp:
    /// 20-34,36-83,95-141,174-204. Includes a conservative fresh key upload and
    /// default-KeySequence split; an explicit caller key can only remove those
    /// requests. Split/reshape key views add no backing. Separate U32->F32
    /// conversion is charged despite equal byte widths. RandomBits has only its
    /// result allocation on CPU primitives.cpp:272-294 and Metal primitives.cpp:
    /// 159-169; no RNG threadgroup storage is priced.
    pub(super) fn random_normal(&mut self, elements: u64) -> Result<()> {
        self.array(4, 2)?; // fresh seed/root key, never a retirement credit
        self.array(4, 4)?; // KeySequence::next -> split(key,2)
        self.repeated(4, 1, 6)?; // low, high, hi-low, upper, maxval, sqrt(2)
        self.array(4, elements)?; // RandomBits U32
        self.array(4, elements)?; // U32 -> F32 for divide
        self.repeated(4, elements, 6) // divide, min, range product, low add, erfinv, scale
    }
    /// Same-dtype float cumulative product (also bounds a float cumulative sum).
    /// CPU scan.cpp:245-265 allocates a possible input repair AND distinct output.
    /// Metal scan.cpp:122-143 shares the possible repair with output, OR allocates
    /// a non-donated output. Kernel-local scan scratch is not allocator backing.
    pub(super) fn cumprod(&mut self, width: u64, elements: u64) -> Result<()> {
        float_width(width)?;
        self.array(width, elements)?;
        if self.backend == NativeBackend::Cpu {
            self.array(width, elements)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod direct_steel_arithmetic_tests {
    use super::*;

    /// HOST recipe arithmetic only: convolution directly invokes steel_matmul,
    /// whereas ordinary Matmul takes its vector shortcut before steel dispatch.
    #[test]
    fn direct_convolution_gemm_keeps_vector_shape_split_k_scratch() {
        let mut l = NativeLedger {
            page: 16_384,
            backend: NativeBackend::Metal,
            total: 0,
        };
        l.metal_mm_workspace(1, 1, 256).unwrap();
        assert_eq!(l.total, 0);
        l.metal_steel_workspace(1, 1, 256).unwrap();
        assert_eq!(l.total, 144); // U(16*1*1*4) + U(2*1*1*4)
    }
}

// Source-derived component graphs, cumulative native active backing; never RSS/footprint.
// Authority: immutable inference 212a6d6, mlx ITSNSQ, mlx-rs fe22998.
// No eval/drop/barrier/window/donation retirement credit. Each node is passed separately to U.
// Primitive API is supplied by the parent native_ledger and norm/conv component.
//
// Citation map (relative to crates/media/mlx-gen):
// mlx-gen-sdxl/src/unet/mod.rs:756-900,1145-1184 (stem, skip stack, control injection, head);
// unet/block.rs:181-244; resnet.rs:188-233,343-407; transformer.rs:202-273,584-640,
// 830-923; embeddings.rs:106-124,203-208,300-336; controlnet.rs:125-135,338-509.
// mlx-gen-sdxl/src/vae.rs:232-257,301-353,541-595,714-737,858-997,1044-1181,
// 1229-1420; src/vae_tiling.rs:48-184,225-350,792-860 (exact halo/core loops).
// mlx-gen-sdxl/src/vision_encoder.rs:168-211,255-281,351-361;
// ip_adapter.rs:40-62,264-290,311-316,377-412 (penultimate tower + perceiver resampler).
// src/attention.rs:107-198; src/nn.rs:411-430,847-857,1098-1110;
// src/array.rs:52-76 (reshape/copy consumers). Eager nonlinear expansions bound compiled fusion.
//
// Integration: parent owns linear(work,input_width,rows)->Result<u64> INCLUDING residuals,
// primitive array/op/cast/scalar/matmul/qmm/softmax/reduce/sdpa, layernorm/groupnorm/conv2d.
// Stream constructor/reread recipes, raw weights, pixel host preprocessing, sampler, load and
// post-decode image conversion are deliberately caller-owned. control_condition_graph is called
// ONCE per image/control branch, control_graph ONCE per actual denoise invocation. Parent invokes
// stream_reread separately for each streamed UNet forward; this component never silently reloads.

use crate::ip_adapter::pricing::IpImageEncoderFacts;
use crate::quantum as q;
use crate::unet::pricing::{
    AttentionFacts, BlockFacts, ControlFacts, ConvFacts, ResnetFacts, TimeFacts, TransformerFacts,
    UNetFacts,
};
use crate::vae::pricing::AutoencoderFacts;
use mlx_gen::adapters::facts::TensorFacts;
use mlx_gen::attention::AttentionPlan;
use mlx_rs::Dtype;

#[derive(Clone, Debug)]
pub(super) struct GraphTensor {
    pub(super) shape: Vec<u64>,
    pub(super) dtype: Dtype,
}
impl GraphTensor {
    pub(super) fn new(shape: &[u64], dtype: Dtype) -> Result<Self> {
        // Native Array axes are i32; validation here mirrors that surface, not a memory policy.
        for &d in shape {
            if d > i32::MAX as u64 {
                return Err(cg_error("native axis exceeds i32"));
            }
        }
        product(shape)?;
        Ok(Self {
            shape: shape.to_vec(),
            dtype,
        })
    }
    pub(super) fn elements(&self) -> Result<u64> {
        product(&self.shape)
    }
    pub(super) fn width(&self) -> u64 {
        cg_width(self.dtype)
    }
    fn axis(&self, i: usize) -> Result<u64> {
        self.shape
            .get(i)
            .copied()
            .ok_or_else(|| cg_error("graph rank"))
    }
    fn channels(&self) -> Result<u64> {
        self.shape
            .last()
            .copied()
            .ok_or_else(|| cg_error("graph rank"))
    }
    fn rows(&self) -> Result<u64> {
        product(&self.shape[..self.shape.len().saturating_sub(1)])
    }
}
#[derive(Clone, Copy)]
pub(super) struct GraphRegime<'a> {
    /// Actual source plan; query_block is reused, not a hand-selected chunk size.
    pub(super) attention: AttentionPlan<'a>,
    pub(super) conditioning_rows: u64,
    pub(super) cfg_row: bool,
}
#[derive(Clone)]
pub(super) struct ControlGraphOutput {
    pub(super) down: Vec<GraphTensor>,
    pub(super) mid: GraphTensor,
}
fn cg_error(s: &str) -> Error {
    q::native_shape_failure(s)
}
fn cg_width(d: Dtype) -> u64 {
    q::dtype_bytes(d)
}
fn cg_promote(a: Dtype, b: Dtype) -> Dtype {
    Dtype::from_promoting_types(a, b)
}
fn cg_fact_width(t: &TensorFacts) -> u64 {
    cg_width(t.dtype)
}
fn cg_fact_elements(t: &TensorFacts) -> Result<u64> {
    product(
        &t.shape
            .iter()
            .map(|&d| cg_dim(d))
            .collect::<Result<Vec<_>>>()?,
    )
}
fn cg_dim(d: i32) -> Result<u64> {
    u64::try_from(d).map_err(|_| cg_error("negative parameter axis"))
}
fn cg_shape(t: &TensorFacts, i: usize) -> Result<u64> {
    cg_dim(*t.shape.get(i).ok_or_else(|| cg_error("parameter rank"))?)
}
// Final-effect projections reuse existing factory facts; no parallel constructor or parameter graph.
fn cg_effect_time(f: &TimeFacts, e: &q::EffectScope<'_>) -> Result<TimeFacts> {
    Ok(TimeFacts {
        linear1: e.linear_metadata("linear_1", &f.linear1)?,
        linear2: e.linear_metadata("linear_2", &f.linear2)?,
    })
}
fn cg_effect_resnet(f: &ResnetFacts, e: &q::EffectScope<'_>) -> Result<ResnetFacts> {
    let mut out = f.clone();
    out.conv1 = e.conv_metadata("conv1", &f.conv1);
    out.conv2 = e.conv_metadata("conv2", &f.conv2);
    if let Some(p) = &f.time_emb_proj {
        out.time_emb_proj = Some(e.linear_metadata("time_emb_proj", p)?);
    }
    if let Some(p) = &f.shortcut {
        out.shortcut = Some(e.shortcut_metadata("conv_shortcut", p));
    }
    Ok(out)
}
fn cg_effect_attention(f: &AttentionFacts, e: &q::EffectScope<'_>) -> Result<AttentionFacts> {
    let mut out = f.clone();
    out.q = e.linear_metadata("to_q", &f.q)?;
    out.k = e.linear_metadata("to_k", &f.k)?;
    out.v = e.linear_metadata("to_v", &f.v)?;
    out.out = e.linear_metadata("to_out.0", &f.out)?;
    if let Some(p) = &f.to_k_ip {
        out.to_k_ip = Some(e.linear_metadata("to_k_ip", p)?);
    }
    if let Some(p) = &f.to_v_ip {
        out.to_v_ip = Some(e.linear_metadata("to_v_ip", p)?);
    }
    Ok(out)
}
fn cg_effect_transformer(f: &TransformerFacts, e: &q::EffectScope<'_>) -> Result<TransformerFacts> {
    let mut out = f.clone();
    out.proj_in = e.linear_metadata("proj_in", &f.proj_in)?;
    out.proj_out = e.linear_metadata("proj_out", &f.proj_out)?;
    for (i, b) in out.blocks.iter_mut().enumerate() {
        let e = e.child(&format!("transformer_blocks.{i}"));
        b.attn1 = cg_effect_attention(&b.attn1, &e.child("attn1"))?;
        b.attn2 = cg_effect_attention(&b.attn2, &e.child("attn2"))?;
        b.linear1 = e.linear_metadata("ff.linear1", &b.linear1)?;
        b.linear2 = e.linear_metadata("ff.linear2", &b.linear2)?;
        b.linear3 = e.linear_metadata("ff.linear3", &b.linear3)?;
    }
    Ok(out)
}
fn cg_effect_block(f: &BlockFacts, e: &q::EffectScope<'_>) -> Result<BlockFacts> {
    let mut out = f.clone();
    for (i, r) in out.resnets.iter_mut().enumerate() {
        *r = cg_effect_resnet(r, &e.child(&format!("resnets.{i}")))?;
    }
    if let Some(attn) = &mut out.attentions {
        for (i, a) in attn.iter_mut().enumerate() {
            *a = cg_effect_transformer(a, &e.child(&format!("attentions.{i}")))?;
        }
    }
    if let Some(c) = &f.downsample {
        out.downsample = Some(e.conv_metadata("downsamplers.0.conv", c));
    }
    if let Some(c) = &f.upsample {
        out.upsample = Some(e.conv_metadata("upsamplers.0.conv", c));
    }
    Ok(out)
}
fn cg_float_width(d: Dtype, width: u64) -> Dtype {
    // The projection helper includes adapter promotions. F32 has no narrower interpretation;
    // a two-byte result retains the actual f16/bf16 base dtype rather than conflating the two.
    if width == 4 {
        Dtype::Float32
    } else {
        d
    }
}
impl NativeLedger {
    fn cg_cast(&mut self, x: &GraphTensor, dtype: Dtype) -> Result<GraphTensor> {
        // cast helper is width-based; equal-width f16<->bf16 is still a real AsType allocation.
        if x.dtype != dtype {
            self.array(cg_width(dtype), x.elements()?)?;
        }
        Ok(GraphTensor {
            shape: x.shape.clone(),
            dtype,
        })
    }
    fn cg_parameter_cast(&mut self, t: &TensorFacts, d: Dtype) -> Result<()> {
        if t.dtype != d {
            self.array(cg_width(d), cg_fact_elements(t)?)?;
        }
        Ok(())
    }
    fn cg_linear(&mut self, w: &q::LinearWork, x: &GraphTensor) -> Result<GraphTensor> {
        let weight_dtype = if w.bits.is_some() {
            cg_promote(
                w.weights
                    .get(1)
                    .ok_or_else(|| cg_error("packed scales"))?
                    .dtype,
                w.weights
                    .get(2)
                    .ok_or_else(|| cg_error("packed offsets"))?
                    .dtype,
            )
        } else {
            w.weights
                .first()
                .ok_or_else(|| cg_error("linear weight"))?
                .dtype
        };
        let compute = cg_promote(x.dtype, weight_dtype);
        // A byte-width-only primitive cannot discover f16+bf16 -> f32. Materialize the actual
        // input cast in that case; its wider input tells the primitive to cast parameter operands.
        let raw_weight_width = if w.bits.is_some() {
            cg_fact_width(&w.weights[1]).max(cg_fact_width(&w.weights[2]))
        } else {
            cg_fact_width(&w.weights[0])
        };
        let mixed_compute = cg_width(compute) > x.width().max(raw_weight_width);
        let input_width = if mixed_compute {
            self.cg_cast(x, compute)?;
            cg_width(compute)
        } else {
            x.width()
        };
        let width = self.linear(w, input_width, x.rows()?)?;
        // Opposite f16/bf16 in the separately-added output bias widens AFTER matmul. Preserve its
        // source ordering rather than pretending the matmul itself computed at the bias dtype.
        if let Some(bias) = &w.bias {
            let out = cg_promote(compute, bias.dtype);
            if cg_width(out) > cg_width(compute).max(cg_fact_width(bias)) {
                let n = p!(x.rows()?, w.output_width());
                self.array(cg_width(out), n)?;
                self.cg_parameter_cast(bias, out)?;
                self.op(cg_width(out), n)?;
            }
        }
        // output_shape also checks active residual broadcasting; do not truncate adapters to base.
        let input = x
            .shape
            .iter()
            .map(|&d| i32::try_from(d).map_err(|_| cg_error("linear axis")))
            .collect::<Result<Vec<_>>>()?;
        let output = w.output_shape(&input)?;
        let shape = output.into_iter().map(cg_dim).collect::<Result<Vec<_>>>()?;
        GraphTensor::new(&shape, cg_float_width(w.output_dtype(x.dtype), width))
    }
    fn cg_add(&mut self, a: &GraphTensor, b: &GraphTensor) -> Result<GraphTensor> {
        let ai = a.shape.iter().map(|&n| n as i32).collect::<Vec<_>>();
        let bi = b.shape.iter().map(|&n| n as i32).collect::<Vec<_>>();
        let shape = crate::adapters::broadcast_shape(&ai, &bi)?
            .into_iter()
            .map(cg_dim)
            .collect::<Result<Vec<_>>>()?;
        let d = cg_promote(a.dtype, b.dtype);
        self.cg_cast(a, d)?;
        self.cg_cast(b, d)?;
        let out = GraphTensor::new(&shape, d)?;
        self.op(out.width(), out.elements()?)?;
        Ok(out)
    }
    fn cg_concat(&mut self, tensors: &[GraphTensor], axis: usize) -> Result<GraphTensor> {
        let first = tensors.first().ok_or_else(|| cg_error("empty concat"))?;
        if tensors.len() == 1 {
            return Ok(first.clone());
        }
        let mut shape = first.shape.clone();
        let mut d = first.dtype;
        shape[axis] = 0;
        for t in tensors {
            if t.shape.len() != shape.len()
                || t.shape
                    .iter()
                    .enumerate()
                    .any(|(i, &n)| i != axis && n != first.shape[i])
            {
                return Err(cg_error("concat geometry"));
            }
            shape[axis] = plus(shape[axis], t.shape[axis])?;
            d = cg_promote(d, t.dtype);
        }
        for t in tensors {
            self.cg_cast(t, d)?;
        }
        let out = GraphTensor::new(&shape, d)?;
        self.array(out.width(), out.elements()?)?;
        Ok(out)
    }
    fn cg_silu(&mut self, x: &GraphTensor) -> Result<GraphTensor> {
        // Sigmoid primitive plus Multiply; retained/compiled exact epilogues cannot need more
        // graph output arrays than these eager nodes. No output donation assumed.
        self.silu(x.width(), x.elements()?)?;
        Ok(x.clone())
    }
    fn cg_gelu(&mut self, x: &GraphTensor, quick: bool) -> Result<GraphTensor> {
        let constants = if quick { 1 } else { 3 };
        for _ in 0..constants {
            self.scalar(4)?;
            if x.dtype != Dtype::Float32 {
                self.scalar(x.width())?;
            }
        }
        self.repeated(x.width(), x.elements()?, if quick { 3 } else { 5 })?;
        Ok(x.clone())
    }
    fn cg_nearest(&mut self, x: &GraphTensor) -> Result<GraphTensor> {
        let mut out = x.clone();
        out.shape[1] = p!(2, x.axis(1)?);
        out.shape[2] = p!(2, x.axis(2)?);
        // Broadcast is a view. Its final reshape merges broadcast axes and may copy.
        self.array(out.width(), out.elements()?)?;
        Ok(out)
    }
    fn cg_norm(
        &mut self,
        x: &GraphTensor,
        gain: &TensorFacts,
        bias: &TensorFacts,
        group: bool,
    ) -> Result<GraphTensor> {
        let d = cg_promote(cg_promote(x.dtype, gain.dtype), bias.dtype);
        if group {
            // Primitive knows widths, not equal-width f16/bf16 promotion. In that mixed case
            // materialize the actual promoted affine leaves explicitly before passing widths.
            let mixed = cg_width(d) > x.width().max(cg_fact_width(gain)).max(cg_fact_width(bias));
            let (gw, bw) = if mixed {
                self.cg_parameter_cast(gain, d)?;
                self.cg_parameter_cast(bias, d)?;
                (cg_width(d), cg_width(d))
            } else {
                (cg_fact_width(gain), cg_fact_width(bias))
            };
            self.groupnorm(
                x.width(),
                [x.axis(0)?, x.axis(1)?, x.axis(2)?, x.axis(3)?],
                32,
                [gw, bw],
            )?;
        } else {
            self.layernorm(
                x.elements()?,
                x.rows()?,
                x.width(),
                cg_width(d),
                cg_fact_width(gain),
                cg_fact_width(bias),
            )?;
        }
        Ok(GraphTensor {
            shape: x.shape.clone(),
            dtype: d,
        })
    }
    fn cg_conv(
        &mut self,
        x: &GraphTensor,
        w: &TensorFacts,
        bias: Option<&TensorFacts>,
        stride: u64,
        pad: u64,
    ) -> Result<GraphTensor> {
        let (b, h, wi, ci) = (x.axis(0)?, x.axis(1)?, x.axis(2)?, x.axis(3)?);
        let (co, kh, kw, wci) = (
            cg_shape(w, 0)?,
            cg_shape(w, 1)?,
            cg_shape(w, 2)?,
            cg_shape(w, 3)?,
        );
        if ci != wci || stride == 0 {
            return Err(cg_error("convolution geometry"));
        }
        let ph = plus(h, p!(2, pad))?;
        let pw = plus(wi, p!(2, pad))?;
        let oh = plus(
            ph.checked_sub(kh)
                .ok_or_else(|| cg_error("convolution height"))?
                / stride,
            1,
        )?;
        let ow = plus(
            pw.checked_sub(kw)
                .ok_or_else(|| cg_error("convolution width"))?
                / stride,
            1,
        )?;
        let compute = cg_promote(x.dtype, w.dtype);
        self.cg_cast(x, compute)?;
        self.cg_parameter_cast(w, compute)?;
        self.conv2d(
            [cg_width(compute), b, h, wi, ci, co, kh, kw, oh, ow],
            [stride, stride],
            [pad, pad],
            [1, 1],
            1,
        )?;
        let mut out = GraphTensor::new(&[b, oh, ow, co], compute)?;
        if let Some(bias) = bias {
            let d = cg_promote(compute, bias.dtype);
            self.cg_cast(&out, d)?;
            self.cg_parameter_cast(bias, d)?;
            // Covers Metal's exact convolution epilogue conditional contiguous bias operand.
            self.array(cg_width(d), cg_fact_elements(bias)?)?;
            self.op(cg_width(d), out.elements()?)?;
            out.dtype = d;
        }
        Ok(out)
    }
    fn cg_conv_work(
        &mut self,
        x: &GraphTensor,
        w: &q::ConvWork,
        stride: u64,
        pad: u64,
    ) -> Result<GraphTensor> {
        self.cg_conv(x, &w.weight, w.bias.as_ref(), stride, pad)
    }
    fn cg_conv_fact(&mut self, x: &GraphTensor, w: &ConvFacts) -> Result<GraphTensor> {
        self.cg_conv(
            x,
            &w.weight,
            Some(&w.bias),
            cg_dim(w.stride)?,
            cg_dim(w.pad)?,
        )
    }
    fn cg_select(
        &mut self,
        x: GraphTensor,
        batch: u64,
        regime: GraphRegime<'_>,
    ) -> Result<GraphTensor> {
        if regime.cfg_row && x.axis(0)? > batch {
            let mut out = x;
            out.shape[0] = 1;
            self.array(4, 1)?;
            self.array(out.width(), out.elements()?)?;
            Ok(out)
        } else {
            Ok(x)
        }
    }
    fn cg_sdpa(&mut self, shape: [u64; 6], plan: AttentionPlan<'_>) -> Result<()> {
        let [width, b, heads, q, k, dim] = shape;
        let block = cg_dim(
            plan.budget
                .query_block(b as i32, heads as i32, q as i32, k as i32),
        )?;
        if block == 0 {
            return Err(cg_error("zero attention block"));
        }
        let mut start = 0;
        let mut count = 0;
        while start < q {
            let len = block.min(q - start);
            self.sdpa([width, b, heads, heads, len, k, dim], false, false)?;
            start = plus(start, len)?;
            count = plus(count, 1)?;
        }
        if count > 1 {
            self.array(width, p!(b, heads, q, dim))?;
        }
        Ok(())
    }
    fn cg_attention(
        &mut self,
        f: &AttentionFacts,
        x: &GraphTensor,
        context: &GraphTensor,
        ip: Option<&GraphTensor>,
        r: GraphRegime<'_>,
    ) -> Result<GraphTensor> {
        let b = x.axis(0)?;
        let nq = x.axis(1)?;
        let nk = context.axis(1)?;
        let q = self.cg_linear(&f.q, x)?;
        let k = self.cg_linear(&f.k, context)?;
        let k = self.cg_select(k, b, r)?;
        let v = self.cg_linear(&f.v, context)?;
        let v = self.cg_select(v, b, r)?;
        let d = cg_promote(cg_promote(q.dtype, k.dtype), v.dtype);
        self.cg_cast(&q, d)?;
        self.cg_cast(&k, d)?;
        self.cg_cast(&v, d)?;
        let heads = cg_dim(f.num_heads)?;
        let dim = cg_dim(f.head_dim)?;
        self.cg_sdpa([cg_width(d), b, heads, nq, nk, dim], r.attention)?;
        let mut o = GraphTensor::new(&[b, nq, p!(heads, dim)], d)?;
        if let (Some(ki), Some(vi), Some(tokens)) = (&f.to_k_ip, &f.to_v_ip, ip) {
            let k = self.cg_linear(ki, tokens)?;
            let k = self.cg_select(k, b, r)?;
            let v = self.cg_linear(vi, tokens)?;
            let v = self.cg_select(v, b, r)?;
            let di = cg_promote(cg_promote(q.dtype, k.dtype), v.dtype);
            self.cg_cast(&q, di)?;
            self.cg_cast(&k, di)?;
            self.cg_cast(&v, di)?;
            self.cg_sdpa(
                [cg_width(di), b, heads, nq, tokens.axis(1)?, dim],
                r.attention,
            )?;
            let oi = GraphTensor::new(&o.shape, di)?;
            self.scalar(4)?;
            if di != Dtype::Float32 {
                self.scalar(cg_width(di))?;
            }
            self.op(oi.width(), oi.elements()?)?;
            o = self.cg_add(&o, &oi)?;
        }
        // transpose -> reshape can require a logical output copy before the linear consumer.
        self.array(o.width(), o.elements()?)?;
        self.cg_linear(&f.out, &o)
    }
    fn cg_resnet(
        &mut self,
        f: &ResnetFacts,
        x: &GraphTensor,
        temb: Option<&GraphTensor>,
        r: GraphRegime<'_>,
    ) -> Result<GraphTensor> {
        let n = self.cg_norm(x, &f.norm1_w, &f.norm1_b, true)?;
        let n = self.cg_silu(&n)?;
        let mut y = self.cg_conv_work(&n, &f.conv1, 1, 1)?;
        if let (Some(proj), Some(t)) = (&f.time_emb_proj, temb) {
            let t = self.cg_silu(t)?;
            let t = self.cg_linear(proj, &t)?;
            let mut t = self.cg_select(t, x.axis(0)?, r)?;
            t.shape = vec![t.axis(0)?, 1, 1, t.channels()?];
            y = self.cg_add(&y, &t)?;
        }
        let n = self.cg_norm(&y, &f.norm2_w, &f.norm2_b, true)?;
        let n = self.cg_silu(&n)?;
        let y = self.cg_conv_work(&n, &f.conv2, 1, 1)?;
        let skip = match &f.shortcut {
            Some(sc) => self.cg_linear(sc, x)?,
            None => x.clone(),
        };
        self.cg_add(&skip, &y)
    }
    fn cg_transformer(
        &mut self,
        f: &TransformerFacts,
        x: &GraphTensor,
        memory: &GraphTensor,
        ip: Option<&GraphTensor>,
        r: GraphRegime<'_>,
    ) -> Result<GraphTensor> {
        let n = self.cg_norm(x, &f.norm_w, &f.norm_b, true)?;
        let mut n = GraphTensor::new(
            &[x.axis(0)?, p!(x.axis(1)?, x.axis(2)?), x.axis(3)?],
            n.dtype,
        )?;
        n = self.cg_linear(&f.proj_in, &n)?;
        for block in &f.blocks {
            let y = self.cg_norm(&n, &block.norm1_w, &block.norm1_b, false)?;
            let y = self.cg_attention(&block.attn1, &y, &y, None, r)?;
            n = self.cg_add(&n, &y)?;
            let y = self.cg_norm(&n, &block.norm2_w, &block.norm2_b, false)?;
            let y = self.cg_attention(&block.attn2, &y, memory, ip, r)?;
            n = self.cg_add(&n, &y)?;
            let y = self.cg_norm(&n, &block.norm3_w, &block.norm3_b, false)?;
            let value = self.cg_linear(&block.linear1, &y)?;
            let gate = self.cg_linear(&block.linear2, &y)?;
            let gate = self.cg_gelu(&gate, false)?;
            let mixed = self.cg_add_product(&value, &gate)?;
            let y = self.cg_linear(&block.linear3, &mixed)?;
            n = self.cg_add(&n, &y)?;
        }
        let mut n = self.cg_linear(&f.proj_out, &n)?;
        n.shape = x.shape.clone();
        self.cg_add(&n, x)
    }
    // Same broadcast/promotion graph as add, but one Multiply node (not an additional add).
    fn cg_add_product(&mut self, a: &GraphTensor, b: &GraphTensor) -> Result<GraphTensor> {
        self.cg_add(a, b)
    }
    fn cg_block(
        &mut self,
        f: &BlockFacts,
        mut x: GraphTensor,
        conditioning: [&GraphTensor; 2],
        mut skips: Option<&mut Vec<GraphTensor>>,
        ip: Option<&GraphTensor>,
        r: GraphRegime<'_>,
    ) -> Result<(GraphTensor, Vec<GraphTensor>)> {
        let [memory, temb] = conditioning;
        let mut outputs = Vec::new();
        for (i, resnet) in f.resnets.iter().enumerate() {
            if let Some(skips) = skips.as_deref_mut() {
                let skip = skips.pop().ok_or_else(|| cg_error("skip underflow"))?;
                x = self.cg_concat(&[x, skip], 3)?;
            }
            x = self.cg_resnet(resnet, &x, Some(temb), r)?;
            if let Some(attn) = &f.attentions {
                x = self.cg_transformer(
                    attn.get(i).ok_or_else(|| cg_error("attention count"))?,
                    &x,
                    memory,
                    ip,
                    r,
                )?;
            }
            outputs.push(x.clone());
        }
        if let Some(c) = &f.downsample {
            x = self.cg_conv_work(&x, c, 2, 1)?;
            outputs.push(x.clone());
        }
        if let Some(c) = &f.upsample {
            x = self.cg_nearest(&x)?;
            x = self.cg_conv_work(&x, c, 1, 1)?;
            outputs.push(x.clone());
        }
        Ok((x, outputs))
    }
    fn cg_time(&mut self, f: &TimeFacts, x: &GraphTensor) -> Result<GraphTensor> {
        let x = self.cg_linear(&f.linear1, x)?;
        let x = self.cg_silu(&x)?;
        self.cg_linear(&f.linear2, &x)
    }
    fn cg_sinusoid(&mut self, rows: u64, dim: u64, dtype: Dtype) -> Result<GraphTensor> {
        // multiply, cos, sin, concatenate; table construction belongs to load_recipe.
        let actual = p!(2, dim / 2);
        self.repeated(4, p!(rows, dim / 2), 3)?;
        self.array(4, p!(rows, actual))?;
        let x = GraphTensor::new(&[rows, actual], Dtype::Float32)?;
        self.cg_cast(&x, dtype)
    }
    fn cg_temb(
        &mut self,
        time: &TimeFacts,
        add: &TimeFacts,
        channels: u64,
        conditioning: [&GraphTensor; 2],
        dtype: Dtype,
        r: GraphRegime<'_>,
    ) -> Result<GraphTensor> {
        let [pooled, ids] = conditioning;
        self.array(4, r.conditioning_rows)?;
        let t = self.cg_sinusoid(r.conditioning_rows, channels, dtype)?;
        let t = self.cg_time(time, &t)?;
        let per_id = cg_dim(add.linear1.base.base_shape[1])?
            .checked_sub(pooled.channels()?)
            .ok_or_else(|| cg_error("add embedding width"))?
            / ids.axis(1)?;
        let id = self.cg_sinusoid(p!(ids.axis(0)?, ids.axis(1)?), per_id, dtype)?;
        let id = GraphTensor::new(&[ids.axis(0)?, p!(ids.axis(1)?, per_id)], id.dtype)?;
        let joined = self.cg_concat(&[pooled.clone(), id], 1)?;
        let add = self.cg_time(add, &joined)?;
        self.cg_add(&t, &add)
    }
    pub(super) fn unet_graph(
        &mut self,
        f: &UNetFacts,
        inputs: [GraphTensor; 4],
        r: GraphRegime<'_>,
        ip: Option<GraphTensor>,
        control: Option<&ControlGraphOutput>,
        effects: &q::EffectScope<'_>,
    ) -> Result<GraphTensor> {
        let [input, mut memory, pooled, ids] = inputs;
        if let Some(p) = &f.encoder_hid_proj {
            let p = effects.linear_metadata("encoder_hid_proj", p)?;
            memory = self.cg_linear(&p, &memory)?;
        }
        let time = cg_effect_time(&f.time_embedding, &effects.child("time_embedding"))?;
        let add = cg_effect_time(&f.add_embedding, &effects.child("add_embedding"))?;
        let conv_in = effects.conv_metadata("conv_in", &f.conv_in);
        let t = self.cg_temb(
            &time,
            &add,
            cg_shape(&conv_in.weight, 0)?,
            [&pooled, &ids],
            input.dtype,
            r,
        )?;
        let mut x = self.cg_conv_work(&input, &conv_in, 1, 1)?;
        let mut skips = vec![x.clone()];
        for (i, b) in f.down_blocks.iter().enumerate() {
            let b = cg_effect_block(b, &effects.child(&format!("down_blocks.{i}")))?;
            let (next, out) = self.cg_block(&b, x, [&memory, &t], None, ip.as_ref(), r)?;
            x = next;
            skips.extend(out);
        }
        if let Some(c) = control {
            if c.down.len() != skips.len() {
                return Err(cg_error("control down pairing"));
            }
            for (s, c) in skips.iter_mut().zip(&c.down) {
                *s = self.cg_add(s, c)?;
            }
        }
        let mid0 = cg_effect_resnet(&f.mid_resnet0, &effects.child("mid_block.resnets.0"))?;
        let mid_attn =
            cg_effect_transformer(&f.mid_transformer, &effects.child("mid_block.attentions.0"))?;
        let mid1 = cg_effect_resnet(&f.mid_resnet1, &effects.child("mid_block.resnets.1"))?;
        x = self.cg_resnet(&mid0, &x, Some(&t), r)?;
        x = self.cg_transformer(&mid_attn, &x, &memory, ip.as_ref(), r)?;
        x = self.cg_resnet(&mid1, &x, Some(&t), r)?;
        if let Some(c) = control {
            x = self.cg_add(&x, &c.mid)?;
        }
        for (i, b) in f.up_blocks.iter().enumerate() {
            let b = cg_effect_block(b, &effects.child(&format!("up_blocks.{i}")))?;
            x = self
                .cg_block(&b, x, [&memory, &t], Some(&mut skips), ip.as_ref(), r)?
                .0;
        }
        let x = self.cg_norm(&x, &f.conv_norm_out_w, &f.conv_norm_out_b, true)?;
        let x = self.cg_silu(&x)?;
        self.cg_conv_work(&x, &effects.conv_metadata("conv_out", &f.conv_out), 1, 1)
    }
    pub(super) fn control_condition_graph(
        &mut self,
        f: &ControlFacts,
        pixels: GraphTensor,
    ) -> Result<GraphTensor> {
        let x = self.cg_conv_fact(&pixels, &f.cond_embedding.conv_in)?;
        let mut x = self.cg_silu(&x)?;
        for c in &f.cond_embedding.blocks {
            let y = self.cg_conv_fact(&x, c)?;
            x = self.cg_silu(&y)?;
        }
        self.cg_conv_fact(&x, &f.cond_embedding.conv_out)
    }
    pub(super) fn control_graph(
        &mut self,
        f: &ControlFacts,
        inputs: [GraphTensor; 5],
        r: GraphRegime<'_>,
        effects: &q::EffectScope<'_>,
    ) -> Result<ControlGraphOutput> {
        let [input, mut memory, pooled, ids, condition] = inputs;
        if let Some(p) = &f.encoder_hid_proj {
            let p = effects.linear_metadata("encoder_hid_proj", p)?;
            memory = self.cg_linear(&p, &memory)?;
        }
        let time = cg_effect_time(&f.time_embedding, &effects.child("time_embedding"))?;
        let add = cg_effect_time(&f.add_embedding, &effects.child("add_embedding"))?;
        let t = self.cg_temb(
            &time,
            &add,
            cg_shape(&f.conv_in.weight, 0)?,
            [&pooled, &ids],
            input.dtype,
            r,
        )?;
        let x = self.cg_conv_fact(&input, &f.conv_in)?;
        let mut x = self.cg_add(&x, &condition)?;
        let mut skips = vec![x.clone()];
        for (i, b) in f.down_blocks.iter().enumerate() {
            let b = cg_effect_block(b, &effects.child(&format!("down_blocks.{i}")))?;
            let (next, out) = self.cg_block(&b, x, [&memory, &t], None, None, r)?;
            x = next;
            skips.extend(out);
        }
        let mid0 = cg_effect_resnet(&f.mid_resnet0, &effects.child("mid_block.resnets.0"))?;
        let mid_attn =
            cg_effect_transformer(&f.mid_transformer, &effects.child("mid_block.attentions.0"))?;
        let mid1 = cg_effect_resnet(&f.mid_resnet1, &effects.child("mid_block.resnets.1"))?;
        x = self.cg_resnet(&mid0, &x, Some(&t), r)?;
        x = self.cg_transformer(&mid_attn, &x, &memory, None, r)?;
        x = self.cg_resnet(&mid1, &x, Some(&t), r)?;
        if skips.len() != f.down_zero.len() {
            return Err(cg_error("zero head pairing"));
        }
        self.scalar(4)?;
        if input.dtype != Dtype::Float32 {
            self.scalar(input.width())?;
        }
        let mut down = Vec::new();
        for (s, z) in skips.iter().zip(&f.down_zero) {
            let y = self.cg_conv_fact(s, z)?;
            let d = cg_promote(y.dtype, input.dtype);
            self.cg_cast(&y, d)?;
            if d != input.dtype {
                self.scalar(cg_width(d))?;
            }
            self.op(cg_width(d), y.elements()?)?;
            down.push(GraphTensor {
                shape: y.shape,
                dtype: d,
            });
        }
        let mut mid = self.cg_conv_fact(&x, &f.mid_zero)?;
        let d = cg_promote(mid.dtype, input.dtype);
        self.cg_cast(&mid, d)?;
        if d != input.dtype {
            self.scalar(cg_width(d))?;
        }
        self.op(cg_width(d), mid.elements()?)?;
        mid.dtype = d;
        Ok(ControlGraphOutput { down, mid })
    }
    pub(super) fn control_merge_graph(
        &mut self,
        a: ControlGraphOutput,
        b: ControlGraphOutput,
    ) -> Result<ControlGraphOutput> {
        if a.down.len() != b.down.len() {
            return Err(cg_error("multi-control pairing"));
        }
        let down = a
            .down
            .iter()
            .zip(&b.down)
            .map(|(a, b)| self.cg_add(a, b))
            .collect::<Result<Vec<_>>>()?;
        let mid = self.cg_add(&a.mid, &b.mid)?;
        Ok(ControlGraphOutput { down, mid })
    }
}

/// Saved global statistics dtype; affine tensors remain borrowed from the real header facts.
struct CgGlobalNorm<'a> {
    mean_dtype: Dtype,
    inv_dtype: Dtype,
    gain: &'a TensorFacts,
    bias: &'a TensorFacts,
}
impl NativeLedger {
    fn cg_scale(&mut self, x: &GraphTensor, scalar_dtype: Dtype) -> Result<GraphTensor> {
        self.scalar(4)?;
        if scalar_dtype != Dtype::Float32 {
            self.scalar(cg_width(scalar_dtype))?;
        }
        let d = cg_promote(x.dtype, scalar_dtype);
        let out = self.cg_cast(x, d)?;
        if d != scalar_dtype {
            self.scalar(cg_width(d))?;
        }
        self.op(out.width(), out.elements()?)?;
        Ok(out)
    }
    fn cg_mean(&mut self, width: u64, input: u64, output: u64, extent: u64) -> Result<()> {
        // ops.cpp:2201-2218: sum * number_of_elements(..., reciprocal=true).
        self.reduce(width, input, output, extent, false)?;
        self.scalar(width)?;
        self.op(width, output)
    }
    fn cg_global_stats<'a>(
        &mut self,
        x: &GraphTensor,
        gain: &'a TensorFacts,
        bias: &'a TensorFacts,
    ) -> Result<CgGlobalNorm<'a>> {
        let (b, h, w, c) = (x.axis(0)?, x.axis(1)?, x.axis(2)?, x.axis(3)?);
        if !c.is_multiple_of(32) || h == 0 {
            return Err(cg_error("global group norm geometry"));
        }
        let out = p!(b, 32);
        let extent = p!(h, w, c / 32);
        if x.dtype == Dtype::Float16 {
            let row = p!(b, w, c);
            let rows = ((1u64 << 24) / row.max(1)).max(1).min(h);
            let chunks = ceil(h, rows)?;
            for pass in 0..2 {
                let mut y = 0;
                let mut index = 0;
                while y < h {
                    let len = rows.min(h - y);
                    let n = p!(b, len, w, c);
                    self.array(4, n)?; // actual per-crop AsType f16 -> f32
                    if pass == 1 {
                        self.repeated(4, n, 2)?;
                    } // deviation and square
                    self.reduce(4, n, out, p!(len, w, c / 32), false)?;
                    if index > 0 {
                        self.op(4, out)?;
                    }
                    y = plus(y, len)?;
                    index = plus(index, 1)?;
                }
                debug_assert_eq!(index, chunks);
                self.scalar(4)?;
                self.op(4, out)?; // mean or variance divide
            }
            self.scalar(4)?;
            self.repeated(4, out, 2)?; // epsilon add, rsqrt
            Ok(CgGlobalNorm {
                mean_dtype: Dtype::Float32,
                inv_dtype: Dtype::Float32,
                gain,
                bias,
            })
        } else {
            // grouped reshape may require an exact logical-shape copy; transpose is a view.
            self.array(x.width(), x.elements()?)?;
            self.cg_mean(x.width(), x.elements()?, out, extent)?; // explicit mean
            self.cg_mean(x.width(), x.elements()?, out, extent)?; // var's own mean
            self.repeated(x.width(), x.elements()?, 2)?; // centred square
            self.reduce(x.width(), x.elements()?, out, extent, false)?;
            self.scalar(x.width())?;
            self.op(x.width(), out)?; // variance normalizer
            self.scalar(4)?;
            let d = cg_promote(x.dtype, Dtype::Float32);
            if d != x.dtype {
                self.array(cg_width(d), out)?;
            }
            self.repeated(cg_width(d), out, 2)?;
            Ok(CgGlobalNorm {
                mean_dtype: x.dtype,
                inv_dtype: d,
                gain,
                bias,
            })
        }
    }
    fn cg_global_apply(&mut self, x: &GraphTensor, norm: &CgGlobalNorm<'_>) -> Result<GraphTensor> {
        self.array(x.width(), x.elements()?)?; // conditional reshape of a noncontiguous spatial crop
        let sub = cg_promote(x.dtype, norm.mean_dtype);
        self.cg_cast(x, sub)?;
        self.op(cg_width(sub), x.elements()?)?; // subtract mean
        let d = cg_promote(sub, norm.inv_dtype);
        if d != sub {
            self.array(cg_width(d), x.elements()?)?;
        }
        self.op(cg_width(d), x.elements()?)?; // multiply inv_std
        self.array(cg_width(d), x.elements()?)?; // transpose+reshape conditional logical copy
        let affine = cg_promote(cg_promote(d, norm.gain.dtype), norm.bias.dtype);
        if affine != d {
            self.array(cg_width(affine), x.elements()?)?;
        }
        self.cg_parameter_cast(norm.gain, affine)?;
        self.cg_parameter_cast(norm.bias, affine)?;
        self.repeated(cg_width(affine), x.elements()?, 2)?; // affine mul+add
        let out = GraphTensor::new(&x.shape, affine)?;
        if x.dtype == Dtype::Float16 {
            self.cg_cast(&out, Dtype::Float16)
        } else {
            Ok(out)
        }
    }
    fn cg_tiled_conv(
        &mut self,
        x: &GraphTensor,
        w: &TensorFacts,
        bias: &TensorFacts,
        edge: u64,
        norm: Option<&CgGlobalNorm<'_>>,
        activation: [bool; 2],
    ) -> Result<GraphTensor> {
        let [activate, wide] = activation;
        let (b, h, wi, c) = (x.axis(0)?, x.axis(1)?, x.axis(2)?, x.axis(3)?);
        let edge = edge.max(3);
        let co = cg_shape(w, 0)?;
        let mut ww = w.clone();
        let mut wb = bias.clone();
        if wide {
            self.cg_parameter_cast(w, Dtype::Float32)?;
            self.cg_parameter_cast(bias, Dtype::Float32)?;
            ww.dtype = Dtype::Float32;
            wb.dtype = Dtype::Float32;
        }
        let tiled = h > edge || wi > edge;
        let core = if tiled { edge - 2 } else { h.max(wi).max(1) };
        let mut rows = Vec::new();
        let mut y = 0;
        while y < h {
            let y1 = plus(y, core)?.min(h);
            let iy = y.saturating_sub(1);
            let ey = plus(y1, 1)?.min(h);
            let mut cores = Vec::new();
            let mut col = 0;
            while col < wi {
                let x1 = plus(col, core)?.min(wi);
                let ix = col.saturating_sub(1);
                let ex = plus(x1, 1)?.min(wi);
                let mut tile = GraphTensor::new(&[b, ey - iy, ex - ix, c], x.dtype)?;
                if let Some(norm) = norm {
                    tile = self.cg_global_apply(&tile, norm)?;
                }
                if activate {
                    tile = self.cg_silu(&tile)?;
                }
                if wide {
                    tile = self.cg_cast(&tile, Dtype::Float32)?;
                }
                let out = self.cg_conv(&tile, &ww, Some(&wb), 1, 1)?;
                let out = GraphTensor::new(&[b, y1 - y, x1 - col, co], out.dtype)?;
                let out = if wide {
                    self.cg_cast(&out, x.dtype)?
                } else {
                    out
                };
                cores.push(out);
                col = x1;
                if !tiled {
                    break;
                }
            }
            rows.push(self.cg_concat(&cores, 2)?);
            y = y1;
            if !tiled {
                break;
            }
        }
        let out = self.cg_concat(&rows, 1)?;
        if tiled {
            // array::contiguous: at most a reshape-copy and explicit Contiguous copy. These are
            // layout-conditional source nodes, charged independently; no generic layout multiplier.
            self.repeated(out.width(), out.elements()?, 2)?;
        }
        Ok(out)
    }
    fn cg_tiled_resnet(
        &mut self,
        f: &ResnetFacts,
        x: &GraphTensor,
        edge: u64,
    ) -> Result<GraphTensor> {
        let norm = self.cg_global_stats(x, &f.norm1_w, &f.norm1_b)?;
        let bias = f
            .conv1
            .bias
            .as_ref()
            .ok_or_else(|| cg_error("VAE conv1 bias"))?;
        let y = self.cg_tiled_conv(x, &f.conv1.weight, bias, edge, Some(&norm), [true, false])?;
        self.cg_trace(&y)?;
        let norm = self.cg_global_stats(&y, &f.norm2_w, &f.norm2_b)?;
        let bias = f
            .conv2
            .bias
            .as_ref()
            .ok_or_else(|| cg_error("VAE conv2 bias"))?;
        let y = self.cg_tiled_conv(&y, &f.conv2.weight, bias, edge, Some(&norm), [true, false])?;
        self.cg_trace(&y)?;
        let residual = match &f.shortcut {
            Some(sc) => self.cg_linear(sc, x)?,
            None => x.clone(),
        };
        self.cg_add(&residual, &y)
    }
    fn cg_tiled_down(
        &mut self,
        x: &GraphTensor,
        w: &TensorFacts,
        bias: &TensorFacts,
        edge: u64,
    ) -> Result<GraphTensor> {
        let (b, h, wi, c) = (x.axis(0)?, x.axis(1)?, x.axis(2)?, x.axis(3)?);
        // Source's exact integer formula ((h-2)/2+1), not a rounded latent-grid estimate.
        let oh = plus(
            h.checked_sub(2)
                .ok_or_else(|| cg_error("downsample height"))?
                / 2,
            1,
        )?;
        let ow = plus(
            wi.checked_sub(2)
                .ok_or_else(|| cg_error("downsample width"))?
                / 2,
            1,
        )?;
        let core = edge.max(1);
        let mut y = 0;
        let mut rows = Vec::new();
        while y < oh {
            let y1 = plus(y, core)?.min(oh);
            let mut col = 0;
            let mut cores = Vec::new();
            while col < ow {
                let x1 = plus(col, core)?.min(ow);
                let need_h = plus(p!(2, y1 - y), 1)?;
                let need_w = plus(p!(2, x1 - col), 1)?;
                let _have_h = need_h.min(h - p!(2, y));
                let _have_w = need_w.min(wi - p!(2, col));
                let crop = GraphTensor::new(&[b, need_h, need_w, c], x.dtype)?;
                // Source always calls Pad, even for an interior crop with zero pad widths.
                // mlx-rs/shapes.rs:781-784 constructs I32 zero then casts to input dtype.
                self.scalar(4)?;
                self.scalar(crop.width())?;
                self.array(crop.width(), crop.elements()?)?;
                cores.push(self.cg_conv(&crop, w, Some(bias), 2, 0)?);
                col = x1;
            }
            rows.push(self.cg_concat(&cores, 2)?);
            y = y1;
        }
        self.cg_concat(&rows, 1)
    }
    fn cg_vae_attn(
        &mut self,
        f: &crate::vae::pricing::AttentionFacts,
        x: &GraphTensor,
    ) -> Result<GraphTensor> {
        let y = self.cg_norm(x, &f._gn_w, &f._gn_b, true)?;
        let q = self.cg_linear(&f._q, &y)?;
        let k = self.cg_linear(&f._k, &y)?;
        let v = self.cg_linear(&f._v, &y)?;
        let d = cg_promote(cg_promote(q.dtype, k.dtype), v.dtype);
        self.cg_cast(&q, d)?;
        self.cg_cast(&k, d)?;
        self.cg_cast(&v, d)?;
        let tokens = p!(x.axis(1)?, x.axis(2)?);
        self.cg_sdpa(
            [cg_width(d), x.axis(0)?, 1, tokens, tokens, x.axis(3)?],
            crate::vae::pricing::AttentionFacts::bounded_plan(),
        )?;
        let y = GraphTensor::new(&x.shape, d)?;
        let y = self.cg_linear(&f.out, &y)?;
        self.cg_add(x, &y)
    }
    /// Source optional trace is conservatively included whenever its call site is reachable.
    /// Its environment switch is not silently used to omit allocations from a certificate.
    fn cg_trace(&mut self, x: &GraphTensor) -> Result<()> {
        let row = p!(x.axis(0)?, x.axis(2)?, x.axis(3)?);
        let h = x.axis(1)?;
        if h == 0 {
            return Ok(());
        }
        let rows = ((1u64 << 20) / row.max(1)).max(1).min(h);
        let mut y = 0;
        while y < h {
            let len = rows.min(h - y);
            let n = p!(row, len);
            // ops.cpp:1919-1953: isnan=not_equal (1); isinf=two equals + OR (3);
            // isfinite=NOT(OR(isinf,isnan)) (6). Each isinf also has two dtype scalars.
            self.repeated(1, n, 10)?;
            self.repeated(x.width(), 1, 4)?;
            self.repeated(4, n, 2)?; // count bool->u32
            self.reduce(4, n, 1, n, false)?;
            self.reduce(4, n, 1, n, false)?;
            // where(scalar f32 infinity,part) promotes half part to f32 at each where.
            self.repeated(4, 1, 2)?;
            if x.dtype != Dtype::Float32 {
                self.repeated(4, n, 2)?;
            }
            self.repeated(4, n, 2)?;
            self.reduce(4, n, 1, n, false)?;
            self.reduce(4, n, 1, n, false)?;
            y = plus(y, len)?;
        }
        Ok(())
    }
    pub(super) fn vae_decode_graph(
        &mut self,
        f: &AutoencoderFacts,
        z: GraphTensor,
        tile: Option<u64>,
    ) -> Result<GraphTensor> {
        let d = &f.decoder;
        let r = GraphRegime {
            attention: AttentionPlan::UNBOUNDED,
            conditioning_rows: z.axis(0)?,
            cfg_row: false,
        };
        if tile.is_some() {
            self.cg_trace(&z)?;
        }
        let z = if f.dtype != Dtype::Float32 {
            self.cg_cast(&z, f.dtype)?
        } else {
            z
        };
        let z = self.cg_scale(&z, f.dtype)?;
        let z = self.cg_linear(&f.post_quant_proj, &z)?;
        let mut x = self.cg_conv(&z, &d.conv_in_w, Some(&d.conv_in_b), 1, 1)?;
        x = self.cg_resnet(&d.mid_resnet0, &x, None, r)?;
        x = self.cg_vae_attn(&d.mid_attn, &x)?;
        x = self.cg_resnet(&d.mid_resnet1, &x, None, r)?;
        if tile.is_some() {
            self.cg_trace(&x)?;
        }
        let mut remaining = 1u64;
        for block in &d.up_blocks {
            if block.upsample.is_some() {
                remaining = p!(remaining, 2);
            }
        }
        for block in &d.up_blocks {
            if tile.is_some() {
                self.cg_trace(&x)?;
            }
            let edge = tile
                .map(|t| ceil(t, remaining).map(|n| n.max(3)))
                .transpose()?;
            for resnet in &block.resnets {
                x = match edge {
                    Some(edge) => self.cg_tiled_resnet(resnet, &x, edge)?,
                    None => self.cg_resnet(resnet, &x, None, r)?,
                };
                if tile.is_some() {
                    self.cg_trace(&x)?;
                }
            }
            if let Some((w, b)) = &block.upsample {
                x = self.cg_nearest(&x)?;
                if tile.is_some() {
                    self.cg_trace(&x)?;
                }
                let next = (remaining / 2).max(1);
                x = match tile {
                    Some(t) => self.cg_tiled_conv(
                        &x,
                        w,
                        b,
                        ceil(t, next)?.max(3),
                        None,
                        [false, x.dtype == Dtype::Float16],
                    )?,
                    None => self.cg_conv(&x, w, Some(b), 1, 1)?,
                };
                remaining = next;
            }
            if tile.is_some() {
                self.cg_trace(&x)?;
            }
        }
        x = match tile {
            Some(t) => {
                let norm = self.cg_global_stats(&x, &d._norm_out_w, &d._norm_out_b)?;
                self.cg_tiled_conv(
                    &x,
                    &d.conv_out_w,
                    &d.conv_out_b,
                    t.max(3),
                    Some(&norm),
                    [true, false],
                )?
            }
            None => {
                let n = self.cg_norm(&x, &d._norm_out_w, &d._norm_out_b, true)?;
                let n = self.cg_silu(&n)?;
                self.cg_conv(&n, &d.conv_out_w, Some(&d.conv_out_b), 1, 1)?
            }
        };
        if tile.is_some() {
            self.cg_trace(&x)?;
        }
        if f.dtype != Dtype::Float32 {
            x = self.cg_cast(&x, Dtype::Float32)?;
        }
        if tile.is_some() {
            self.cg_trace(&x)?;
        }
        Ok(x)
    }
    pub(super) fn vae_encode_graph(
        &mut self,
        f: &AutoencoderFacts,
        image: GraphTensor,
        tile: Option<u64>,
    ) -> Result<GraphTensor> {
        let e = &f.encoder;
        // Actual source fallback: a fitting image uses dense, not a one-tile assembly.
        let tile = tile.filter(|&t| image.shape[1] > t || image.shape[2] > t);
        let r = GraphRegime {
            attention: AttentionPlan::UNBOUNDED,
            conditioning_rows: image.axis(0)?,
            cfg_row: false,
        };
        let image = if f.dtype != Dtype::Float32 {
            self.cg_cast(&image, f.dtype)?
        } else {
            image
        };
        let mut x = match tile {
            Some(t) => self.cg_tiled_conv(
                &image,
                &e.conv_in_w,
                &e.conv_in_b,
                t.max(3),
                None,
                [false, false],
            )?,
            None => self.cg_conv(&image, &e.conv_in_w, Some(&e.conv_in_b), 1, 1)?,
        };
        let mut scale = 1;
        for block in &e.down_blocks {
            let edge = tile.map(|t| ceil(t, scale).map(|n| n.max(3))).transpose()?;
            for resnet in &block.resnets {
                x = match edge {
                    Some(edge) => self.cg_tiled_resnet(resnet, &x, edge)?,
                    None => self.cg_resnet(resnet, &x, None, r)?,
                };
            }
            if let Some((w, b)) = &block.downsample {
                scale = p!(scale, 2);
                x = match tile {
                    Some(t) => self.cg_tiled_down(&x, w, b, ceil(t, scale)?.max(1))?,
                    None => {
                        let mut pad = x.clone();
                        pad.shape[1] = plus(pad.shape[1], 1)?;
                        pad.shape[2] = plus(pad.shape[2], 1)?;
                        self.scalar(4)?;
                        self.scalar(pad.width())?;
                        self.array(pad.width(), pad.elements()?)?;
                        self.cg_conv(&pad, w, Some(b), 2, 0)?
                    }
                };
            }
        }
        x = self.cg_resnet(&e.mid_resnet0, &x, None, r)?;
        x = self.cg_vae_attn(&e.mid_attn, &x)?;
        x = self.cg_resnet(&e.mid_resnet1, &x, None, r)?;
        let n = self.cg_norm(&x, &e._norm_out_w, &e._norm_out_b, true)?;
        let n = self.cg_silu(&n)?;
        let x = self.cg_conv(&n, &e.conv_out_w, Some(&e._conv_out_b), 1, 1)?;
        let mut moments = self.cg_linear(&f._quant_proj, &x)?;
        let half = moments.channels()? / 2;
        self.array(4, half)?;
        *moments
            .shape
            .last_mut()
            .ok_or_else(|| cg_error("moments rank"))? = half;
        self.array(moments.width(), moments.elements()?)?; // gather mean half, not a view/split
        moments = self.cg_scale(&moments, f.dtype)?;
        if f.dtype != Dtype::Float32 {
            moments = self.cg_cast(&moments, Dtype::Float32)?;
        }
        Ok(moments)
    }
    pub(super) fn ip_image_graph(
        &mut self,
        f: &IpImageEncoderFacts,
        pixels: GraphTensor,
        quick_gelu: bool,
    ) -> Result<GraphTensor> {
        let e = &f.encoder;
        let b = pixels.axis(0)?;
        let patch = cg_dim(e.patch)?;
        let patches = self.cg_conv(&pixels, &e.patch_embedding, None, patch, 0)?;
        let n = p!(patches.axis(1)?, patches.axis(2)?);
        let patches = GraphTensor::new(&[b, n, cg_dim(e.hidden)?], patches.dtype)?;
        let cls = GraphTensor::new(&[b, 1, cg_dim(e.hidden)?], e.class_embedding.dtype)?;
        let mut x = self.cg_concat(&[cls, patches], 1)?;
        let position = GraphTensor::new(
            &[1, cg_shape(&e.position_embedding, 0)?, cg_dim(e.hidden)?],
            e.position_embedding.dtype,
        )?;
        x = self.cg_add(&x, &position)?;
        x = self.cg_norm(&x, &e.pre_ln_w, &e.pre_ln_b, false)?;
        // Penultimate: exactly num_layers-1. Last-layer unused parameters are not forward calls.
        for l in e.layers.iter().take(e.layers.len().saturating_sub(1)) {
            let y = self.cg_norm(&x, &l.ln1_w, &l._ln1_b, false)?;
            let q = self.cg_linear(&l.q, &y)?;
            let k = self.cg_linear(&l._k, &y)?;
            let v = self.cg_linear(&l._v, &y)?;
            let d = cg_promote(cg_promote(q.dtype, k.dtype), v.dtype);
            self.cg_cast(&q, d)?;
            self.cg_cast(&k, d)?;
            self.cg_cast(&v, d)?;
            self.sdpa(
                [
                    cg_width(d),
                    b,
                    cg_dim(l.num_heads)?,
                    cg_dim(l.num_heads)?,
                    x.axis(1)?,
                    x.axis(1)?,
                    cg_dim(l.head_dim)?,
                ],
                false,
                false,
            )?;
            let o = GraphTensor::new(&x.shape, d)?;
            self.array(o.width(), o.elements()?)?;
            let o = self.cg_linear(&l.out, &o)?;
            x = self.cg_add(&x, &o)?;
            let y = self.cg_norm(&x, &l._ln2_w, &l._ln2_b, false)?;
            let y = self.cg_linear(&l.fc1, &y)?;
            let y = self.cg_gelu(&y, quick_gelu)?;
            let y = self.cg_linear(&l.fc2, &y)?;
            x = self.cg_add(&x, &y)?;
        }
        let f = &f.resampler;
        let mut latents = GraphTensor::new(
            &[b, cg_dim(f.num_queries)?, cg_dim(f.dim)?],
            f.latents.dtype,
        )?;
        let x = self.cg_linear(&f.proj_in, &x)?;
        for (attn, ff) in &f.layers {
            let image = self.cg_norm(&x, &attn._norm1_w, &attn._norm1_b, false)?;
            let latent = self.cg_norm(&latents, &attn._norm2_w, &attn._norm2_b, false)?;
            let q = self.cg_linear(&attn.to_q, &latent)?;
            let kv = self.cg_concat(&[image, latent], 1)?;
            let kv = self.cg_linear(&attn._to_kv, &kv)?;
            let d = cg_promote(q.dtype, kv.dtype);
            self.cg_cast(&q, d)?;
            self.cg_cast(&kv, d)?;
            self.sdpa(
                [
                    cg_width(d),
                    b,
                    cg_dim(attn.heads)?,
                    cg_dim(attn.heads)?,
                    latents.axis(1)?,
                    kv.axis(1)?,
                    cg_dim(attn.dim_head)?,
                ],
                false,
                false,
            )?;
            let o = GraphTensor::new(
                &[
                    b,
                    latents.axis(1)?,
                    p!(cg_dim(attn.heads)?, cg_dim(attn.dim_head)?),
                ],
                d,
            )?;
            self.array(o.width(), o.elements()?)?;
            let o = self.cg_linear(&attn._to_out, &o)?;
            latents = self.cg_add(&latents, &o)?;
            let y = self.cg_norm(&latents, &ff._ln_w, &ff._ln_b, false)?;
            let y = self.cg_linear(&ff.fc1, &y)?;
            let y = self.cg_gelu(&y, false)?;
            let y = self.cg_linear(&ff._fc2, &y)?;
            latents = self.cg_add(&latents, &y)?;
        }
        let out = self.cg_linear(&f.proj_out, &latents)?;
        self.cg_norm(&out, &f.norm_out_w, &f.norm_out_b, false)
    }
    pub(super) fn ip_cfg_graph(
        &mut self,
        tokens: GraphTensor,
        guided: bool,
    ) -> Result<GraphTensor> {
        if !guided {
            return Ok(tokens);
        }
        // zeros::<f32> constructs a scalar and full f32 output, then as_dtype(tokens.dtype).
        self.scalar(4)?;
        self.array(4, tokens.elements()?)?;
        if tokens.dtype != Dtype::Float32 {
            self.array(tokens.width(), tokens.elements()?)?;
        }
        let zero = tokens.clone();
        self.cg_concat(&[tokens, zero], 0)
    }
}

// Source-derived cumulative native load ledger. Concatenate inside native_ledger.rs.
// Authority: SDXL loader.rs:39-84,319-350,403-439,478-505,557-592;
// model.rs:842-930,1035-1168; quant.rs:173-202; text_encoder.rs:38-52;
// adapters.rs:614-905,990-1037,1262-1280,2640-3110; shared adapters.rs:96-157,
// 215-322,443-520,842-876. Native: common/load.cpp:30-61, common/common.cpp
// (shared-buffer views), gpu/primitives.cpp (AsType/copy), metal/indexing.cpp
// (Take), mlx-rs nn/quantized.rs:254-278,344-352, mlx/einsum.cpp:350-414.
// Every term below is an individual source allocation passed through bytes/array/
// repeated or a native operator recipe. No logical-byte multiplier, retirement,
// cache credit, source-parent subtraction, or process-footprint claim is used.

type LrHeader = mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
type LrTensor = mlx_gen::adapters::facts::TensorFacts;
type LrComponent = gen_core_family::sdxl_ldm::SdxlComponent;

fn lr_error(message: &str) -> Error {
    Error::Msg(format!("sdxl native load: {message}"))
}
fn lr_dim(value: i32) -> Result<u64> {
    u64::try_from(value).map_err(|_| lr_error("negative native dimension"))
}
fn lr_elements(shape: &[i32]) -> Result<u64> {
    shape.iter().try_fold(1, |n, &d| Ok(p!(n, lr_dim(d)?)))
}
fn lr_header_elements(h: &LrHeader) -> Result<u64> {
    h.shape.iter().try_fold(1, |n, &d| {
        let d = u64::try_from(d).map_err(|_| lr_error("header dimension overflow"))?;
        Ok(p!(n, d))
    })
}
fn lr_width(dtype: Dtype) -> u64 {
    q::dtype_bytes(dtype)
}
fn lr_headers(path: &Path) -> Result<Vec<LrHeader>> {
    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(|e| Error::Msg(e.to_string()))
}
fn lr_component_headers(inputs: &Inputs, component: LrComponent) -> Result<Vec<LrHeader>> {
    if let Some(fused) = &inputs.fused {
        return Ok(fused.source.headers_with(component, &fused.world));
    }
    lr_headers(match component {
        LrComponent::ClipL => &inputs.te1,
        LrComponent::ClipBigG => &inputs.te2,
        LrComponent::Unet => &inputs.unet,
        LrComponent::Vae => &inputs.vae.0,
    })
}
fn lr_json_tensor(value: &serde_json::Value) -> Result<LrTensor> {
    let shape = value["shape"]
        .as_array()
        .ok_or_else(|| lr_error("missing semantic shape"))?
        .iter()
        .map(|v| {
            let d = v
                .as_i64()
                .ok_or_else(|| lr_error("invalid semantic dimension"))?;
            i32::try_from(d).map_err(|_| lr_error("semantic dimension exceeds i32"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dtype = match value["dtype"].as_str().unwrap_or("") {
        "Bool" => Dtype::Bool,
        "Uint8" => Dtype::Uint8,
        "Int8" => Dtype::Int8,
        "Uint16" => Dtype::Uint16,
        "Int16" => Dtype::Int16,
        "Uint32" => Dtype::Uint32,
        "Int32" => Dtype::Int32,
        "Uint64" => Dtype::Uint64,
        "Int64" => Dtype::Int64,
        "Float16" => Dtype::Float16,
        "Bfloat16" => Dtype::Bfloat16,
        "Float32" => Dtype::Float32,
        "Float64" => Dtype::Float64,
        "Complex64" => Dtype::Complex64,
        _ => return Err(lr_error("unknown semantic native dtype")),
    };
    Ok(LrTensor {
        shape,
        dtype,
        strides: Vec::new(),
    })
}
fn lr_factor(factors: &serde_json::Value, name: &str) -> Result<LrTensor> {
    lr_json_tensor(
        factors
            .get(name)
            .ok_or_else(|| lr_error("missing semantic factor"))?,
    )
}

impl NativeLedger {
    fn lr_tensor(&mut self, tensor: &LrTensor) -> Result<()> {
        self.array(lr_width(tensor.dtype), lr_elements(&tensor.shape)?)
    }
    fn lr_cast_tensor(&mut self, tensor: &LrTensor, dtype: Dtype) -> Result<()> {
        // Equal byte width is not equal dtype: f16 -> bf16 is still AsType.
        if tensor.dtype != dtype {
            self.array(lr_width(dtype), lr_elements(&tensor.shape)?)?;
        }
        Ok(())
    }
    fn lr_raw(&mut self, header: &LrHeader) -> Result<()> {
        crate::quant::header_dtype(header.dtype)?;
        let expected = p!(lr_header_elements(header)?, header.dtype.size() as u64);
        if expected != header.data_bytes {
            return Err(lr_error("header byte/shape mismatch"));
        }
        self.bytes(header.data_bytes)
    }
    fn lr_header_cast(&mut self, header: &LrHeader, dtype: Dtype) -> Result<()> {
        if crate::quant::header_dtype(header.dtype)? != dtype {
            self.array(lr_width(dtype), lr_header_elements(header)?)?;
        }
        Ok(())
    }
    fn lr_file_load(
        &mut self,
        headers: &[LrHeader],
        forced: Option<Dtype>,
        select: impl Fn(&str) -> bool,
    ) -> Result<()> {
        for h in headers {
            if !select(&h.name) {
                continue;
            }
            self.lr_raw(h)?;
            let dtype = match forced {
                Some(dtype) => dtype,
                None => crate::quant::loaded_header_dtype(headers, h)?,
            };
            self.lr_header_cast(h, dtype)?;
        }
        Ok(())
    }
    fn lr_fused_load(&mut self, inputs: &Inputs, components: &[LrComponent]) -> Result<()> {
        // Two native fullsplits (text/heavy) create separate lazy Load graphs. Within
        // ONE split, charge a mapped raw source once, however many Q/K/V views it writes.
        // Union over all mapped writers is intentionally conservative for overwritten
        // aliases: it covers every realizable source-order winner without pretending
        // target views are independent raw backings. Ignored/unconstructed sources cost 0.
        let fused = inputs
            .fused
            .as_ref()
            .ok_or_else(|| lr_error("missing fused source"))?;
        for raw in &fused.source.raw {
            let mut selected = Vec::new();
            crate::ldm::pricing::visit_transformed_header(raw, |component, h| {
                if components.contains(&component) {
                    selected.push((component, h));
                }
                Ok(())
            })?;
            if selected.is_empty() {
                continue;
            }
            self.lr_raw(raw)?;
            for (component, header) in selected {
                let dtype = if component == LrComponent::Vae {
                    inputs.vae.1
                } else {
                    crate::quant::loaded_dtype_in_component(
                        fused.source.component_packed(component),
                        &header,
                    )?
                };
                self.lr_header_cast(&header, dtype)?;
            }
        }
        Ok(())
    }
    fn lr_sinusoids(&mut self, cfg: &crate::config::UNetConfig) -> Result<()> {
        // unet/embeddings.rs:42-66: host arange + 4 f32 scalars, then
        // divide/subtract/multiply/add/exp. No donation or lazy-eval retirement credit.
        for dims in [
            cfg.block_out_channels[0],
            cfg.addition_time_embed_dim.unwrap_or(256),
        ] {
            let half = lr_dim(dims / 2)?;
            self.array(4, half)?;
            self.repeated(4, 1, 4)?;
            self.repeated(4, half, 5)?;
        }
        Ok(())
    }
    fn lr_geglu(
        &mut self,
        headers: &[LrHeader],
        select: impl Fn(&str) -> bool,
        materialized: impl Fn(&str) -> bool,
    ) -> Result<()> {
        // quant::lin_geglu_half creates two host i32 selections. Take is NOT a
        // split/view: each half of each weight/bias/affine sibling gets its own output.
        for h in headers
            .iter()
            .filter(|h| h.name.ends_with(".ff.net.0.proj.weight"))
        {
            if !select(&h.name) {
                continue;
            }
            let base = h.name.strip_suffix(".weight").expect("selected suffix");
            let rows = *h.shape.first().ok_or_else(|| lr_error("scalar GEGLU"))?;
            if !rows.is_multiple_of(2) {
                return Err(lr_error("odd GEGLU row count"));
            }
            self.repeated(4, rows as u64 / 2, 2)?;
            if !materialized(&h.name) {
                continue;
            }
            for suffix in ["weight", "bias", "scales", "biases"] {
                if let Some(part) = headers
                    .iter()
                    .find(|p| p.name == format!("{base}.{suffix}"))
                {
                    let dtype = crate::quant::loaded_header_dtype(headers, part)?;
                    if part.shape.is_empty() {
                        return Err(lr_error("scalar GEGLU operand"));
                    }
                    let trailing = part.shape[1..].iter().try_fold(1u64, |n, &d| {
                        Ok::<u64, Error>(p!(
                            n,
                            u64::try_from(d).map_err(|_| lr_error("GEGLU axis overflow"))?
                        ))
                    })?;
                    // take_axis uses the WEIGHT's row index length, including for
                    // nonstandard ordinary-bias shapes; never divide a sibling's own rows.
                    self.repeated(lr_width(dtype), p!(rows as u64 / 2, trailing), 2)?;
                }
            }
        }
        Ok(())
    }
    fn lr_pack(&mut self, linear: &q::LinearWork, bits: i32) -> Result<()> {
        if linear.bits.is_some() {
            return Ok(());
        } // Actual already-packed no-op.
        let shape = &linear.base.base_shape;
        if shape.len() < 2 {
            return Err(q::native_shape_failure("quantize weight rank is below two"));
        }
        let (&input, leading) = shape.split_last().ok_or_else(|| lr_error("scalar pack"))?;
        let input = lr_dim(input)?;
        let rows = lr_elements(leading)?;
        let weight = linear
            .weights
            .first()
            .ok_or_else(|| lr_error("pack without weight"))?;
        self.lr_cast_tensor(weight, Dtype::Bfloat16)?;
        if let Some(bias) = &linear.bias {
            self.lr_cast_tensor(bias, Dtype::Bfloat16)?;
        }
        // Unlike Iris, SDXL uses ops::quantize directly; there is NO stored-grid
        // re-encoding graph. Native primitive owns all three affine sibling outputs.
        let group = lr_dim(linear.group_size)?;
        if group == 0 || input == 0 || !input.is_multiple_of(group) {
            return Err(q::native_shape_failure(
                "quantize input is not group-aligned",
            ));
        }
        self.quantize(2, rows, input, group, lr_dim(bits)?)
    }
    fn lr_text_pack(
        &mut self,
        headers: &[LrHeader],
        cfg: &crate::config::ClipTextConfig,
        bits: Option<i32>,
    ) -> Result<()> {
        let Some(bits) = bits else {
            return Ok(());
        };
        for (path, bias) in crate::text_encoder::quantizable_text_paths(cfg) {
            let linear = crate::text_encoder::clip_header_linear(headers, &path, bias)?;
            self.lr_pack(&linear, bits)?;
        }
        Ok(())
    }
    fn lr_heavy_pack(
        &mut self,
        inputs: &Inputs,
        unet_headers: &[LrHeader],
        work: &[crate::adapters::AdapterWork],
        ip: Option<&[LrHeader]>,
    ) -> Result<()> {
        let Some(bits) = inputs.quant_bits else {
            return Ok(());
        };
        let mut unet =
            crate::unet::pricing::UNetFacts::from_headers(unet_headers, &inputs.arch.unet)?;
        if let Some(ip) = ip {
            unet.install_ip_adapter(crate::unet::pricing::ip_pairs_from_headers(ip)?)?;
        }
        let mut effects = std::collections::BTreeMap::new();
        unet.quantize_using(bits, &mut |path, linear, _| {
            effects.insert(path.to_owned(), linear.clone());
            Ok(())
        })?;
        for item in work {
            let crate::adapters::AdapterWork::Install(install) = item else {
                continue;
            };
            // Dense convs and shortcut-as-linear are intentionally outside quant walk.
            if let Some(linear) = effects.get_mut(&install.target) {
                linear.install(install)?;
            }
        }
        unet.quantize_using(bits, &mut |path, _, bits| {
            self.lr_pack(
                effects
                    .get(path)
                    .ok_or_else(|| lr_error("missing pack target"))?,
                bits,
            )
        })?;
        for file in &inputs.controls {
            let headers = lr_headers(file)?;
            let mut control =
                crate::unet::pricing::ControlFacts::from_headers(&headers, &inputs.arch.unet)?;
            control.quantize_using(bits, &mut |_, linear, bits| self.lr_pack(linear, bits))?;
        }
        Ok(())
    }
    /// One complete native load, including the configured adapter outcome. The selected
    /// Inputs carries the parent's TE alias witness. Staging sheds handles, not cumulative
    /// allocations, so both staged and resident policies are charged without release credits.
    fn load_recipe_selected(
        &mut self,
        inputs: &Inputs,
        _regime: &Regime,
        spec: Option<&LoadSpec>,
        outcome: Option<&crate::adapters::AdapterOutcome<'_>>,
    ) -> Result<()> {
        let te1 = lr_component_headers(inputs, LrComponent::ClipL)?;
        let te2 = lr_component_headers(inputs, LrComponent::ClipBigG)?;
        if inputs.fused.is_some() {
            self.lr_fused_load(inputs, &[LrComponent::ClipL, LrComponent::ClipBigG])?;
        } else {
            self.lr_file_load(&te1, None, |_| true)?;
            self.lr_file_load(&te2, None, |_| true)?;
        }
        self.lr_text_pack(&te1, &inputs.arch.te1, inputs.text_quant_bits)?;
        self.lr_text_pack(&te2, &inputs.arch.te2, inputs.text_quant_bits)?;
        let unet = lr_component_headers(inputs, LrComponent::Unet)?;
        let external_vae = spec
            .and_then(|s| s.components.get(crate::loader::VAE_FP16_FIX_COMPONENT))
            .or_else(|| inputs.fused.as_ref().and_then(|f| f.vae_fix.as_ref()));
        if inputs.fused.is_some() {
            if external_vae.is_some() {
                self.lr_fused_load(inputs, &[LrComponent::Unet])?;
            } else {
                self.lr_fused_load(inputs, &[LrComponent::Unet, LrComponent::Vae])?;
            }
        } else {
            self.lr_file_load(&unet, None, |name| {
                !(inputs.streamable && crate::loader::is_streamed_block_tensor(name))
            })?;
        }
        self.lr_sinusoids(&inputs.arch.unet)?;
        self.lr_geglu(
            &unet,
            |_| true,
            |name| !(inputs.streamable && crate::loader::is_streamed_block_tensor(name)),
        )?;
        // Component-level packed detection does not imply every projection is packed.
        // A mixed snapshot's initial heavy pack can read/evaluate dense streamed targets.
        // Charge the whole block read/gather/pack union, retaining every operand separately.
        if inputs.streamable && inputs.quant_bits.is_some() {
            self.stream_reread(inputs)?;
        }
        let vae_load = |ledger: &mut Self| -> Result<()> {
            if let Some(source) = external_vae {
                let file = crate::loader::resolve_vae_fp16_fix_file(source)?;
                ledger.lr_file_load(
                    &lr_headers(&file)?,
                    Some(crate::loader::VAE_FP16_FIX_DTYPE),
                    |_| true,
                )
            } else if inputs.fused.is_none() {
                ledger.lr_file_load(&lr_headers(&inputs.vae.0)?, Some(inputs.vae.1), |_| true)
            } else {
                Ok(())
            }
        };
        // Fused constructs/materializes VAE before installs; Directory does it after installs.
        if inputs.fused.is_some() {
            vae_load(self)?;
        }
        if let Some(outcome) = outcome {
            self.lr_adapter_outcome(outcome)?;
        } else if spec.is_some_and(|s| !s.adapters.is_empty()) {
            return Err(lr_error("selected adapter outcome required"));
        }
        if outcome.is_some_and(|o| o.refusal.is_some()) {
            return Ok(());
        }
        if inputs.fused.is_none() {
            vae_load(self)?;
        }
        for file in &inputs.controls {
            let headers = lr_headers(file)?;
            self.lr_file_load(&headers, None, |_| true)?;
            self.lr_sinusoids(&inputs.arch.unet)?;
            self.lr_geglu(&headers, |_| true, |_| true)?;
        }
        let ip_source = spec
            .and_then(|s| s.ip_adapter.as_ref())
            .or_else(|| inputs.fused.as_ref().and_then(|f| f.ip_source.as_ref()));
        let ip_files = match ip_source {
            Some(WeightsSource::Dir(dir)) => Some(crate::loader::resolve_ip_adapter_files(dir)?),
            Some(WeightsSource::File(_)) => {
                return Err(lr_error("IP adapter source must be directory"))
            }
            None => inputs.ip.clone(),
        };
        let ip_headers = if let Some((encoder, adapter)) = &ip_files {
            self.lr_file_load(&lr_headers(encoder)?, None, |_| true)?;
            let headers = lr_headers(adapter)?;
            self.lr_file_load(&headers, None, |_| true)?;
            Some(headers)
        } else {
            None
        };
        self.lr_heavy_pack(
            inputs,
            &unet,
            outcome.map_or(&[], |o| o.work),
            ip_headers.as_deref(),
        )
    }
    /// Convenience envelope. Whole-branch orchestration should instead call the selected
    /// interface in the SAME AdapterOutcome visitor as its forward facts.
    fn load_recipe(
        &mut self,
        inputs: &Inputs,
        regime: &Regime,
        spec: Option<&LoadSpec>,
    ) -> Result<()> {
        let Some(configured) = spec.filter(|s| !s.adapters.is_empty()) else {
            return self.load_recipe_selected(inputs, regime, spec, None);
        };
        let headers = lr_component_headers(inputs, LrComponent::Unet)?;
        let start = self.total;
        let mut upper = start;
        crate::adapters::visit_configured_adapter_outcomes(
            configured,
            &headers,
            &inputs.arch.unet,
            |outcome| {
                self.total = start;
                self.load_recipe_selected(inputs, regime, spec, Some(&outcome))?;
                upper = upper.max(self.total);
                Ok(())
            },
        )?;
        self.total = upper;
        Ok(())
    }
    /// Exactly one complete U-Net forward's block-stream rereads. Caller owns count,
    /// CFG-by-row multiplication and sampler callback multiplicity. Captured IP pairs
    /// and adapter payloads are handle clones, NOT fresh file reads or installs.
    fn stream_reread(&mut self, inputs: &Inputs) -> Result<()> {
        if !inputs.streamable {
            return Ok(());
        }
        let headers = lr_headers(&inputs.unet)?;
        self.lr_file_load(&headers, None, crate::loader::is_streamed_block_tensor)?;
        self.lr_geglu(&headers, crate::loader::is_streamed_block_tensor, |_| true)?;
        // block_stream.rs:350-352 quantizes every reconstructed base. Marker-level
        // stream eligibility can coexist with dense targets; only actual packed targets alias.
        if let Some(bits) = inputs.quant_bits {
            let mut unet =
                crate::unet::pricing::UNetFacts::from_headers(&headers, &inputs.arch.unet)?;
            unet.quantize_using(bits, &mut |path, linear, bits| {
                if crate::loader::is_streamed_block_tensor(path) {
                    self.lr_pack(linear, bits)?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    fn lr_mm(&mut self, a: &LrTensor, b: &LrTensor) -> Result<LrTensor> {
        let out = crate::adapters::matmul_shape(&a.shape, &b.shape)?;
        let dtype = Dtype::from_promoting_types(a.dtype, b.dtype);
        self.lr_cast_tensor(a, dtype)?;
        self.lr_cast_tensor(b, dtype)?;
        // matmul receives arbitrary header-selected layouts. The primitive includes
        // conditional operand copies/workspaces; these reshape candidates cover the
        // preceding noncontiguous flattening in einsum/conv reconstructions too.
        let av = a.shape.len() == 1;
        let bv = b.shape.len() == 1;
        let n = if bv {
            1
        } else {
            lr_dim(
                *b.shape
                    .last()
                    .ok_or_else(|| lr_error("scalar matrix operand"))?,
            )?
        };
        let m = if av {
            1
        } else {
            lr_dim(a.shape[a.shape.len() - 2])?
        };
        let matrix_axes = usize::from(!av) + usize::from(!bv);
        let batch = lr_elements(&out[..out.len() - matrix_axes])?;
        let k = lr_dim(
            *a.shape
                .last()
                .ok_or_else(|| lr_error("scalar matrix operand"))?,
        )?;
        self.matmul(lr_width(dtype), batch, m, n, k)?;
        Ok(LrTensor {
            shape: out,
            dtype,
            strides: Vec::new(),
        })
    }
    fn lr_tucker(&mut self, t: &LrTensor, a: &LrTensor, b: &LrTensor) -> Result<LrTensor> {
        if t.shape.len() != 4 || a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(lr_error("invalid Tucker geometry"));
        }
        let i = lr_dim(t.shape[0].max(a.shape[0]))?;
        let j = lr_dim(t.shape[1].max(b.shape[0]))?;
        let hw = p!(lr_dim(t.shape[2])?, lr_dim(t.shape[3])?);
        let p = lr_dim(a.shape[1])?;
        let r = lr_dim(b.shape[1])?;
        let dtype =
            Dtype::from_promoting_types(t.dtype, Dtype::from_promoting_types(a.dtype, b.dtype));
        let width = lr_width(dtype);
        self.lr_cast_tensor(t, dtype)?;
        self.lr_cast_tensor(a, dtype)?;
        self.lr_cast_tensor(b, dtype)?;
        // einsum.cpp's optimizer may select either first contraction or the outer
        // product. Charge the UNION of these three finite native contraction graphs,
        // including transpose+reshape copy candidates. This is not an empirical factor.
        self.array(width, p!(i, j, hw))?;
        self.array(width, p!(i, p))?;
        self.matmul(width, 1, p!(j, hw), p, i)?;
        self.array(width, p!(j, hw, p))?;
        self.array(width, p!(j, r))?;
        self.matmul(width, 1, p!(p, hw), r, j)?;
        self.array(width, p!(i, j, hw))?;
        self.array(width, p!(j, r))?;
        self.matmul(width, 1, p!(i, hw), r, j)?;
        self.array(width, p!(i, hw, r))?;
        self.array(width, p!(i, p))?;
        self.matmul(width, 1, p!(r, hw), p, i)?;
        self.array(width, p!(i, p, j, r))?; // a outer b multiply
        self.array(width, p!(i, p, j, r))?;
        self.array(width, p!(i, j, hw))?;
        self.matmul(width, 1, p!(p, r), hw, p!(i, j))?;
        self.array(width, p!(p, r, hw))?; // final optimizer layout normalization candidate
        Ok(LrTensor {
            shape: vec![a.shape[1], b.shape[1], t.shape[2], t.shape[3]],
            dtype,
            strides: Vec::new(),
        })
    }
    fn lr_rebuild_factor(
        &mut self,
        factors: &serde_json::Value,
        full: &str,
        a: &str,
        b: &str,
        t: Option<&str>,
    ) -> Result<LrTensor> {
        if let Some(full) = factors.get(full) {
            return lr_json_tensor(full);
        }
        let a = lr_factor(factors, a)?;
        let b = lr_factor(factors, b)?;
        if let Some(t) = t.and_then(|name| factors.get(name)) {
            self.lr_tucker(&lr_json_tensor(t)?, &a, &b)
        } else {
            self.lr_mm(&a, &b)
        }
    }
    fn lr_adapter_semantic_graph(&mut self, install: &serde_json::Value) -> Result<()> {
        let factors = &install["factor_layouts"];
        let kind = install["kind"]
            .as_str()
            .ok_or_else(|| lr_error("missing adapter kind"))?;
        if kind == "lora" || kind == "conv_lora" {
            let down = lr_factor(factors, "down")?;
            let mut up = lr_factor(factors, "up")?;
            if let Some(rows) = install.get("up_rows").and_then(|v| v.as_array()) {
                let lo = rows[0]
                    .as_i64()
                    .ok_or_else(|| lr_error("invalid row begin"))?;
                let hi = rows[1]
                    .as_i64()
                    .ok_or_else(|| lr_error("invalid row end"))?;
                up.shape[0] =
                    i32::try_from(hi - lo).map_err(|_| lr_error("row selection overflow"))?;
                self.array(4, lr_dim(up.shape[0])?)?;
                self.lr_tensor(&up)?;
            }
            if install["installed"].as_str() == Some("factor_residual") {
                self.scalar(4)?;
                if up.dtype != Dtype::Float32 {
                    self.scalar(lr_width(up.dtype))?;
                }
                self.lr_tensor(&up)?; // alpha-fold multiply; A/B transposes are views
            } else {
                self.lr_cast_tensor(&up, Dtype::Float32)?;
                self.lr_cast_tensor(&down, Dtype::Float32)?;
                let rank = lr_dim(
                    *down
                        .shape
                        .first()
                        .ok_or_else(|| lr_error("missing LoRA rank"))?,
                )?;
                let input = lr_elements(&down.shape[1..])?;
                let out = lr_dim(
                    *up.shape
                        .first()
                        .ok_or_else(|| lr_error("missing LoRA output"))?,
                )?;
                self.matmul(4, 1, out, input, rank)?;
                if up.dtype != Dtype::Float32 {
                    self.array(lr_width(up.dtype), p!(out, input))?;
                    self.array(4, p!(out, input))?;
                }
                self.scalar(4)?;
                self.op(4, p!(out, input))?;
            }
            return Ok(());
        }
        let (one, two, structured) = if kind == "Lokr" || kind == "ThirdPartyLokr" {
            let one = self.lr_rebuild_factor(factors, "lokr_w1", "lokr_w1_a", "lokr_w1_b", None)?;
            let two = self.lr_rebuild_factor(
                factors,
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                if kind == "ThirdPartyLokr" {
                    Some("lokr_t2")
                } else {
                    None
                },
            )?;
            (
                one,
                two,
                install["installed"].as_str() == Some("structured_residual"),
            )
        } else if kind == "ThirdPartyLoha" {
            let tucker = factors.get("hada_t1").is_some() && factors.get("hada_t2").is_some();
            let one = self.lr_rebuild_factor(
                factors,
                "hada_w1",
                "hada_w1_a",
                "hada_w1_b",
                tucker.then_some("hada_t1"),
            )?;
            let two = self.lr_rebuild_factor(
                factors,
                "hada_w2",
                "hada_w2_a",
                "hada_w2_b",
                tucker.then_some("hada_t2"),
            )?;
            (one, two, false)
        } else {
            return Err(lr_error("unsupported semantic adapter install"));
        };
        if structured {
            self.scalar(4)?;
            let dtype = Dtype::from_promoting_types(two.dtype, Dtype::Float32);
            self.lr_cast_tensor(&two, dtype)?;
            self.array(lr_width(dtype), lr_elements(&two.shape)?)?;
            if dtype != Dtype::Float32 {
                self.array(4, lr_elements(&two.shape)?)?;
            }
            self.lr_cast_tensor(&one, Dtype::Float32)?;
        } else {
            let dtype = Dtype::from_promoting_types(one.dtype, two.dtype);
            self.lr_cast_tensor(&one, dtype)?;
            self.lr_cast_tensor(&two, dtype)?;
            let n = if kind == "ThirdPartyLoha" {
                lr_elements(&crate::adapters::broadcast_shape(&one.shape, &two.shape)?)?
            } else {
                p!(lr_elements(&one.shape)?, lr_elements(&two.shape)?)
            };
            self.op(lr_width(dtype), n)?; // kron multiply / Hadamard multiply
            self.scalar(4)?;
            let scaled = Dtype::from_promoting_types(dtype, Dtype::Float32);
            if dtype != scaled {
                self.array(lr_width(scaled), n)?;
            }
            self.op(lr_width(scaled), n)?;
            self.reshape_copy(lr_width(scaled), n)?; // noncontiguous reshape candidate
            if scaled != Dtype::Float32 {
                self.array(4, n)?;
            }
            // PEFT and TP SDXL callers apply user strength after reconstruction.
            self.scalar(4)?;
            self.op(4, n)?;
        }
        Ok(())
    }
    fn lr_candidate_factor(
        &mut self,
        factors: &std::collections::BTreeMap<String, LrTensor>,
        full: &str,
        a: &str,
        b: &str,
        t: Option<&str>,
    ) -> Result<Option<LrTensor>> {
        if let Some(full) = factors.get(full) {
            return Ok(Some(full.clone()));
        }
        let (Some(a), Some(b)) = (factors.get(a), factors.get(b)) else {
            return Ok(None);
        };
        let dtype = Dtype::from_promoting_types(a.dtype, b.dtype);
        self.lr_cast_tensor(a, dtype)?;
        self.lr_cast_tensor(b, dtype)?;
        if let Some(t) = t.and_then(|name| factors.get(name)) {
            if t.shape.len() != 4 || a.shape.len() != 2 || b.shape.len() != 2 {
                return Ok(None);
            }
            // Invalid singleton contractions fail before native einsum allocation.
            let compatible = |x: i32, y: i32| x == y || (x == 1 && y > 0) || (y == 1 && x > 0);
            if !compatible(t.shape[0], a.shape[0]) || !compatible(t.shape[1], b.shape[0]) {
                return Ok(None);
            }
            return Ok(Some(self.lr_tucker(t, a, b)?));
        }
        if crate::adapters::matmul_shape(&a.shape, &b.shape).is_err() {
            return Ok(None);
        }
        Ok(Some(self.lr_mm(a, b)?))
    }
    fn lr_candidate_lora(&mut self, down: &LrTensor, up: &LrTensor) -> Result<Option<LrTensor>> {
        // Union of the two executable storage branches, not a logical-byte factor.
        self.scalar(4)?;
        if up.dtype != Dtype::Float32 {
            self.scalar(lr_width(up.dtype))?;
        }
        self.lr_tensor(up)?; // packed-host alpha-fold multiply
        self.lr_cast_tensor(up, Dtype::Float32)?;
        self.lr_cast_tensor(down, Dtype::Float32)?;
        if crate::adapters::matmul_shape(&up.shape, &down.shape).is_err() {
            return Ok(None);
        }
        let mut u = up.clone();
        u.dtype = Dtype::Float32;
        let mut d = down.clone();
        d.dtype = Dtype::Float32;
        let product = self.lr_mm(&u, &d)?;
        let n = lr_elements(&product.shape)?;
        if up.dtype != Dtype::Float32 {
            self.array(lr_width(up.dtype), n)?;
            self.array(4, n)?;
        }
        self.scalar(4)?;
        self.op(4, n)?;
        Ok(Some(product))
    }
    fn lr_candidate_merge(&mut self, delta: &LrTensor, targets: &serde_json::Value) -> Result<()> {
        // A failed reshape can precede an install, so no installed result is needed.
        // Enumerate actual declared target geometries, applying the real broadcast
        // operator. Incompatible candidates have no sum output allocation.
        if let Some(targets) = targets.as_object() {
            for target in targets.values() {
                let Some(shape) = target["shape"].as_array() else {
                    continue;
                };
                let shape = shape
                    .iter()
                    .map(|d| {
                        i32::try_from(d.as_i64().ok_or_else(|| lr_error("target axis"))?)
                            .map_err(|_| lr_error("target axis exceeds i32"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let dtype_value = serde_json::json!({"shape":shape,"dtype":target["weight_dtype"]});
                let base = lr_json_tensor(&dtype_value)?;
                self.lr_cast_tensor(delta, base.dtype)?;
                if let Ok(merged) = crate::adapters::broadcast_shape(&base.shape, &delta.shape) {
                    self.array(lr_width(base.dtype), lr_elements(&merged)?)?;
                }
                // Conv hosts transpose NCHW delta to OHWI before the same add.
                if delta.shape.len() == 4 {
                    let ohwi = vec![
                        delta.shape[0],
                        delta.shape[2],
                        delta.shape[3],
                        delta.shape[1],
                    ];
                    if let Ok(merged) = crate::adapters::broadcast_shape(&base.shape, &ohwi) {
                        self.array(lr_width(base.dtype), lr_elements(&merged)?)?;
                    }
                }
            }
        }
        Ok(())
    }
    fn lr_candidate_graph(
        &mut self,
        kind: &str,
        raw: &str,
        factors: &std::collections::BTreeMap<String, LrTensor>,
        targets: &serde_json::Value,
    ) -> Result<()> {
        if kind == "Lora" {
            let (Some(down), Some(up)) = (factors.get("down"), factors.get("up")) else {
                return Ok(());
            };
            if down.shape.len() == 2 && up.shape.len() == 2 {
                for (_, rows) in crate::adapters::lora_output_rows(raw, up.shape[0]) {
                    let mut selected = up.clone();
                    if let Some((lo, hi)) = rows {
                        selected.shape[0] = hi - lo;
                        self.array(4, lr_dim(hi - lo)?)?;
                        self.lr_tensor(&selected)?;
                    }
                    if let Some(delta) = self.lr_candidate_lora(down, &selected)? {
                        self.lr_candidate_merge(&delta, targets)?;
                    }
                }
            }
            return Ok(()); // rank-four pre-routing graph is lr_conv_candidates below
        }
        let (one, two, loha) = if kind == "Lokr" || kind == "ThirdPartyLokr" {
            let Some(one) =
                self.lr_candidate_factor(factors, "lokr_w1", "lokr_w1_a", "lokr_w1_b", None)?
            else {
                return Ok(());
            };
            let Some(two) = self.lr_candidate_factor(
                factors,
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                (kind == "ThirdPartyLokr").then_some("lokr_t2"),
            )?
            else {
                return Ok(());
            };
            (one, two, false)
        } else if kind == "ThirdPartyLoha" {
            let tucker = factors.contains_key("hada_t1") && factors.contains_key("hada_t2");
            let Some(one) = self.lr_candidate_factor(
                factors,
                "hada_w1",
                "hada_w1_a",
                "hada_w1_b",
                tucker.then_some("hada_t1"),
            )?
            else {
                return Ok(());
            };
            let Some(two) = self.lr_candidate_factor(
                factors,
                "hada_w2",
                "hada_w2_a",
                "hada_w2_b",
                tucker.then_some("hada_t2"),
            )?
            else {
                return Ok(());
            };
            (one, two, true)
        } else {
            return Ok(());
        };
        if kind == "Lokr" {
            // Structured build computes/scales both small factors before checking
            // whether the dense target reshape/merge could succeed.
            self.scalar(4)?;
            let scaled = Dtype::from_promoting_types(two.dtype, Dtype::Float32);
            self.lr_cast_tensor(&two, scaled)?;
            self.array(lr_width(scaled), lr_elements(&two.shape)?)?;
            if scaled != Dtype::Float32 {
                self.array(4, lr_elements(&two.shape)?)?;
            }
            self.lr_cast_tensor(&one, Dtype::Float32)?;
        }
        let dtype = Dtype::from_promoting_types(one.dtype, two.dtype);
        self.lr_cast_tensor(&one, dtype)?;
        self.lr_cast_tensor(&two, dtype)?;
        let shape = if loha {
            let Ok(shape) = crate::adapters::broadcast_shape(&one.shape, &two.shape) else {
                return Ok(());
            };
            shape
        } else {
            let rank = one.shape.len().max(two.shape.len());
            let mut left = one.shape.clone();
            left.resize(rank, 1);
            let mut right = vec![1; rank - two.shape.len()];
            right.extend_from_slice(&two.shape);
            let mut result = Vec::with_capacity(rank);
            for (a, b) in left.iter().zip(right) {
                let d = p!(lr_dim(*a)?, lr_dim(b)?);
                let Ok(d) = i32::try_from(d) else {
                    return Ok(());
                };
                result.push(d);
            }
            result
        };
        let n = lr_elements(&shape)?;
        self.op(lr_width(dtype), n)?;
        self.scalar(4)?;
        let scaled = Dtype::from_promoting_types(dtype, Dtype::Float32);
        if scaled != dtype {
            self.array(lr_width(scaled), n)?;
        }
        self.op(lr_width(scaled), n)?;
        self.reshape_copy(lr_width(scaled), n)?;
        if scaled != Dtype::Float32 {
            self.array(4, n)?;
        }
        self.scalar(4)?;
        self.op(4, n)?;
        // Native delta_at reshapes to the selected base BEFORE its merge. Cover
        // every actual target with equal element count, including failed later adds.
        if let Some(targets_map) = targets.as_object() {
            for target in targets_map.values() {
                let value = serde_json::json!({"shape":target["shape"],"dtype":"Float32"});
                let delta = lr_json_tensor(&value)?;
                if lr_elements(&delta.shape)? == n {
                    let base_value =
                        serde_json::json!({"shape":target["shape"],"dtype":target["weight_dtype"]});
                    let base = lr_json_tensor(&base_value)?;
                    self.lr_cast_tensor(&delta, base.dtype)?;
                    self.array(lr_width(base.dtype), n)?;
                }
            }
        }
        Ok(())
    }
    fn lr_raw_adapter_candidates(
        &mut self,
        adapter: &serde_json::Value,
        file: &std::collections::BTreeMap<String, LrTensor>,
        targets: &serde_json::Value,
    ) -> Result<()> {
        let Some(kind) = adapter["kind"].as_str() else {
            return Ok(());
        };
        let Some(recipe) = adapter["winner_recipe"].as_array() else {
            return Ok(());
        };
        let mut groups = std::collections::BTreeMap::<
            String,
            std::collections::BTreeMap<String, Vec<LrTensor>>,
        >::new();
        for writer in recipe {
            let (Some(raw), Some(role), Some(key)) = (
                writer["raw_group"].as_str(),
                writer["role"].as_str(),
                writer["source_key"].as_str(),
            ) else {
                continue;
            };
            if role == "alpha" {
                continue;
            }
            if let Some(tensor) = file.get(key) {
                groups
                    .entry(raw.to_owned())
                    .or_default()
                    .entry(role.to_owned())
                    .or_default()
                    .push(tensor.clone());
            }
        }
        for (raw, roles) in groups {
            let roles = roles.into_iter().collect::<Vec<_>>();
            if roles.is_empty() {
                continue;
            }
            let mut picks = vec![0usize; roles.len()];
            loop {
                let factors = roles
                    .iter()
                    .zip(&picks)
                    .map(|((role, candidates), &pick)| (role.clone(), candidates[pick].clone()))
                    .collect();
                self.lr_candidate_graph(kind, &raw, &factors, targets)?;
                let mut advance = false;
                for i in (0..picks.len()).rev() {
                    picks[i] = picks[i]
                        .checked_add(1)
                        .ok_or_else(|| lr_error("role index overflow"))?;
                    if picks[i] < roles[i].1.len() {
                        advance = true;
                        break;
                    }
                    picks[i] = 0;
                }
                if !advance {
                    break;
                }
            }
        }
        Ok(())
    }
    fn lr_conv_candidates(
        &mut self,
        adapter: &serde_json::Value,
        file: &std::collections::BTreeMap<String, LrTensor>,
        targets: &serde_json::Value,
    ) -> Result<()> {
        if adapter["kind"].as_str() != Some("Lora") {
            return Ok(());
        }
        let Some(recipe) = adapter["winner_recipe"].as_array() else {
            return Ok(());
        };
        // Native conv_lora_delta runs before target routing. Its matmul therefore
        // exists even when no Install semantic record follows. Union only down/up
        // writers in the SAME classified raw group, never unrelated file factors.
        for down in recipe.iter().filter(|r| r["role"].as_str() == Some("down")) {
            for up in recipe
                .iter()
                .filter(|r| r["role"].as_str() == Some("up") && r["raw_group"] == down["raw_group"])
            {
                let Some(d) = down["source_key"].as_str().and_then(|k| file.get(k)) else {
                    continue;
                };
                let Some(u) = up["source_key"].as_str().and_then(|k| file.get(k)) else {
                    continue;
                };
                if d.shape.len() != 4
                    || u.shape.len() != 4
                    || d.shape[0] != u.shape[1]
                    || p!(lr_dim(u.shape[2])?, lr_dim(u.shape[3])?) != 1
                {
                    continue;
                }
                self.lr_cast_tensor(d, Dtype::Float32)?;
                self.lr_cast_tensor(u, Dtype::Float32)?;
                let out = lr_dim(u.shape[0])?;
                let input = lr_elements(&d.shape[1..])?;
                self.matmul(4, 1, out, input, lr_dim(d.shape[0])?)?;
                if u.dtype != Dtype::Float32 {
                    self.array(lr_width(u.dtype), p!(out, input))?;
                    self.array(4, p!(out, input))?;
                }
                self.scalar(4)?;
                self.op(4, p!(out, input))?;
                let delta = LrTensor {
                    shape: vec![u.shape[0], d.shape[1], d.shape[2], d.shape[3]],
                    dtype: Dtype::Float32,
                    strides: Vec::new(),
                };
                self.lr_candidate_merge(&delta, targets)?;
                if d.shape[2] == 1 && d.shape[3] == 1 {
                    let shortcut = LrTensor {
                        shape: vec![u.shape[0], d.shape[1]],
                        dtype: Dtype::Float32,
                        strides: Vec::new(),
                    };
                    self.lr_candidate_merge(&shortcut, targets)?;
                }
            }
        }
        Ok(())
    }
    fn lr_adapter_outcome(&mut self, outcome: &crate::adapters::AdapterOutcome<'_>) -> Result<()> {
        use crate::adapters::AdapterWork as W;
        let mut file = std::collections::BTreeMap::<String, LrTensor>::new();
        let mut file_index = 0usize;
        for work in outcome.work {
            match work {
                W::FileMaterialized(tensors) => {
                    file.clear();
                    for (name, tensor) in tensors {
                        self.lr_tensor(tensor)?;
                        file.insert(name.clone(), tensor.clone());
                    }
                    if let Some(adapter) = outcome.facts["adapters"].get(file_index) {
                        self.lr_conv_candidates(adapter, &file, &outcome.facts["targets"])?;
                        self.lr_raw_adapter_candidates(adapter, &file, &outcome.facts["targets"])?;
                    }
                    file_index = file_index
                        .checked_add(1)
                        .ok_or_else(|| lr_error("adapter file index overflow"))?;
                }
                W::AlphaRead { key, .. } => {
                    let tensor = file
                        .get(key)
                        .ok_or_else(|| lr_error("alpha without source"))?;
                    self.lr_cast_tensor(tensor, Dtype::Float32)?;
                }
                W::Preparation(tensors) => {
                    for tensor in tensors {
                        self.lr_tensor(tensor)?;
                    }
                }
                W::GuardAttempt(install) => {
                    // These are source-declared arrays, not install.scratch/pre_guard
                    // aggregate budgets. Some parents are views/already charged; counting
                    // them again is conservative, and never a release assertion.
                    for tensor in install.parents.iter().chain(&install.installed) {
                        self.lr_tensor(tensor)?;
                    }
                    // Dense merges cast delta then add into the broadcast base shape.
                    if matches!(
                        install.storage,
                        crate::adapters::AdapterInstallStorage::DenseLinear
                            | crate::adapters::AdapterInstallStorage::DenseConv
                    ) {
                        if let (Some(base), Some(result)) =
                            (install.parents.first(), install.installed.first())
                        {
                            self.array(lr_width(base.dtype), lr_elements(&result.shape)?)?;
                            self.array(lr_width(base.dtype), lr_elements(&result.shape)?)?;
                        }
                    }
                }
                W::Install(_) | W::Selected(_) | W::FileScopeEnd => {}
            }
        }
        if let Some(adapters) = outcome.facts["adapters"].as_array() {
            for adapter in adapters {
                if let Some(installed) = adapter["installed"].as_array() {
                    for install in installed {
                        self.lr_adapter_semantic_graph(install)?;
                    }
                }
            }
        }
        // Raw-group candidate graphs above precede semantic success and include
        // every role winner's reconstruction prefix, even when native validation
        // refuses before an Install record exists. No opcode-IR refusal is needed.
        Ok(())
    }
    /// Native Adapter::residual followed by AdaptableLinear's ordered narrowing/add.
    /// Width-only callers do not distinguish f16 from bf16. Promotion below therefore
    /// unions both 16-bit activation possibilities rather than assuming equal widths
    /// imply equal dtypes. The host output remains base_width after every residual.
    fn linear_residuals(
        &mut self,
        work: &q::LinearWork,
        input_width: u64,
        rows: u64,
        base_width: u64,
    ) -> Result<u64> {
        use mlx_gen::adapters::facts::AdapterFacts as A;
        let input = lr_dim(
            *work
                .base
                .base_shape
                .last()
                .ok_or_else(|| lr_error("missing linear input"))?,
        )?;
        let mut host = vec![
            i32::try_from(rows).map_err(|_| lr_error("row count exceeds i32"))?,
            i32::try_from(work.output_width()).map_err(|_| lr_error("output exceeds i32"))?,
        ];
        for residual in work.residuals.iter().filter(|a| !a.disabled()) {
            let (n, width, scaled) = match residual {
                A::Lora { a, b, .. } => {
                    let first = lr_promoted_input_width(input_width, a.dtype);
                    let second = lr_promoted_input_width(first, b.dtype);
                    let rank = lr_dim(*a.shape.last().ok_or_else(|| lr_error("LoRA A rank"))?)?;
                    let out = lr_dim(*b.shape.last().ok_or_else(|| lr_error("LoRA B output"))?)?;
                    // Possible equal-width dtype conversion must still be charged.
                    self.array(first, p!(rows, input))?;
                    self.array(first, lr_elements(&a.shape)?)?;
                    self.matmul(first, 1, rows, rank, input)?;
                    self.array(second, p!(rows, rank))?;
                    self.array(second, lr_elements(&b.shape)?)?;
                    self.matmul(second, 1, rows, out, rank)?;
                    (p!(rows, out), second, true)
                }
                A::Materialized { delta, .. } => {
                    let out = lr_dim(
                        *delta
                            .shape
                            .first()
                            .ok_or_else(|| lr_error("delta output"))?,
                    )?;
                    self.array(input_width, lr_elements(&delta.shape)?)?;
                    self.matmul(input_width, 1, rows, out, input)?;
                    (p!(rows, out), input_width, true)
                }
                A::Structured { w1, w2, .. } => {
                    if w1.shape.len() != 2 || w2.shape.len() != 2 {
                        return Err(lr_error("structured rank"));
                    }
                    let a = lr_dim(w1.shape[0])?;
                    let c = lr_dim(w1.shape[1])?;
                    let b = lr_dim(w2.shape[0])?;
                    let d = lr_dim(w2.shape[1])?;
                    if p!(c, d) != input {
                        return Err(lr_error("structured input mismatch"));
                    }
                    self.array(input_width, lr_elements(&w1.shape)?)?;
                    self.array(input_width, lr_elements(&w2.shape)?)?;
                    self.reshape_copy(input_width, p!(rows, input))?;
                    self.matmul(input_width, rows, a, d, c)?;
                    self.matmul(input_width, rows, a, b, d)?;
                    self.reshape_copy(input_width, p!(rows, a, b))?;
                    (p!(rows, a, b), input_width, false)
                }
            };
            if scaled {
                self.scalar(4)?;
                if width != 4 {
                    self.scalar(width)?;
                }
                self.op(width, n)?;
            }
            // Same-width narrowing can still be f16/bf16: charge the candidate cast.
            self.array(base_width, n)?;
            let input_shape = [
                i32::try_from(rows).map_err(|_| lr_error("rows exceed i32"))?,
                i32::try_from(input).map_err(|_| lr_error("input exceeds i32"))?,
            ];
            let shape = crate::adapters::adapter_residual_shape(residual, &input_shape)?;
            host = crate::adapters::broadcast_shape(&host, &shape)?;
            self.op(base_width, lr_elements(&host)?)?;
        }
        Ok(base_width)
    }
}
fn lr_promoted_input_width(input: u64, factor: Dtype) -> u64 {
    if input == 2 {
        lr_width(Dtype::from_promoting_types(Dtype::Float16, factor)).max(lr_width(
            Dtype::from_promoting_types(Dtype::Bfloat16, factor),
        ))
    } else {
        lr_width(Dtype::from_promoting_types(Dtype::Float32, factor)).max(input)
    }
}

type Header = mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
fn facts_elements(f: &PriceTensor) -> Result<u64> {
    f.shape.iter().try_fold(1u64, |n, &d| {
        n.checked_mul(u64::try_from(d).map_err(|_| overflow())?)
            .ok_or_else(overflow)
    })
}
fn native_dtype(width: u64) -> Dtype {
    if width == 2 {
        Dtype::Float16
    } else {
        Dtype::Float32
    }
}
impl NativeLedger {
    /// AdaptableLinear::forward's base, followed by the source-owned residual walk.
    /// Each operand conversion/copy is retained, including the ordinary post-QMM bias.
    fn linear(&mut self, work: &q::LinearWork, input_width: u64, rows: u64) -> Result<u64> {
        let k = u64::try_from(*work.base.base_shape.last().ok_or_else(overflow)?)
            .map_err(|_| overflow())?;
        let out = work.output_width();
        let input_dtype = native_dtype(input_width);
        let compute = if work.bits.is_some() {
            Dtype::from_promoting_types(
                input_dtype,
                Dtype::from_promoting_types(work.weights[1].dtype, work.weights[2].dtype),
            )
        } else {
            Dtype::from_promoting_types(input_dtype, work.weights[0].dtype)
        };
        let cw = q::dtype_bytes(compute);
        self.cast(p!(rows, k), input_width, cw)?;
        if work.bits.is_some() {
            for weight in &work.weights[1..] {
                self.cast(facts_elements(weight)?, q::dtype_bytes(weight.dtype), cw)?;
            }
            self.qmm(
                cw,
                rows,
                k,
                out,
                [
                    facts_elements(&work.weights[0])?,
                    facts_elements(&work.weights[1])?,
                    u64::try_from(work.group_size).map_err(|_| overflow())?,
                ],
            )?;
        } else {
            self.cast(
                facts_elements(&work.weights[0])?,
                q::dtype_bytes(work.weights[0].dtype),
                cw,
            )?;
            self.matmul(cw, 1, rows, out, k)?;
        }
        let bw = q::dtype_bytes(work.output_dtype(input_dtype));
        if let Some(bias) = &work.bias {
            self.cast(p!(rows, out), cw, bw)?;
            self.cast(facts_elements(bias)?, q::dtype_bytes(bias.dtype), bw)?;
            self.op(bw, p!(rows, out))?;
        }
        self.linear_residuals(work, input_width, rows, bw)
    }
    /// fast::LayerNorm: fused Metal primitive or the source fallback graph, NEVER CPU lowering
    /// substituted for Metal. Full gain/bias casts remain distinct allocation requests.
    fn text_norm(
        &mut self,
        rows: u64,
        dim: u64,
        input: Dtype,
        gain: Dtype,
        bias: Dtype,
    ) -> Result<Dtype> {
        let output = Dtype::from_promoting_types(Dtype::from_promoting_types(input, gain), bias);
        let n = p!(rows, dim);
        let iw = q::dtype_bytes(input);
        let ow = q::dtype_bytes(output);
        // Equal-size F16/BF16 AsType is still an allocation; widths alone cannot elide it.
        if input != output && iw == ow {
            self.array(ow, n)?;
        }
        if gain != output && q::dtype_bytes(gain) == ow {
            self.array(ow, dim)?;
        }
        if bias != output && q::dtype_bytes(bias) == ow {
            self.array(ow, dim)?;
        }
        self.layernorm(n, rows, iw, ow, q::dtype_bytes(gain), q::dtype_bytes(bias))?;
        Ok(output)
    }
    /// Source pre-compile envelopes for mlx_gen::nn::{gelu_exact,gelu_quick}; compiled fusion
    /// cannot add native graph operands beyond this envelope. Constants are dtype-weak casts.
    fn text_activation(
        &mut self,
        width: u64,
        n: u64,
        act: crate::config::ClipActivation,
    ) -> Result<()> {
        match act {
            crate::config::ClipActivation::QuickGelu => {
                self.scalar(4)?;
                self.cast(1, 4, width)?;
                self.repeated(width, n, 3)?;
            }
            crate::config::ClipActivation::Gelu => {
                for _ in 0..3 {
                    self.scalar(4)?;
                    self.cast(1, 4, width)?;
                }
                self.repeated(width, n, 5)?;
            }
        }
        Ok(())
    }
    fn text_encoder_graph(
        &mut self,
        headers: &[Header],
        cfg: &ClipTextConfig,
        quant: Option<i32>,
        b: u64,
        n: u64,
    ) -> Result<(Dtype, Dtype)> {
        let tensor = |name: &str| crate::text_encoder::clip_header_tensor(headers, name);
        let token = tensor("text_model.embeddings.token_embedding.weight")?;
        let pos = tensor("text_model.embeddings.position_embedding.weight")?;
        let dim = u64::try_from(token.shape[1]).map_err(|_| overflow())?;
        let rows = p!(b, n);
        self.array(q::dtype_bytes(token.dtype), p!(rows, dim))?;
        self.array(4, n)?; // actual arange host i32 position indices
        self.array(q::dtype_bytes(pos.dtype), p!(n, dim))?;
        let mut dt = Dtype::from_promoting_types(token.dtype, pos.dtype);
        self.cast(
            p!(rows, dim),
            q::dtype_bytes(token.dtype),
            q::dtype_bytes(dt),
        )?;
        self.cast(p!(n, dim), q::dtype_bytes(pos.dtype), q::dtype_bytes(dt))?;
        self.op(q::dtype_bytes(dt), p!(rows, dim))?;
        self.array(4, p!(n, n))?;
        self.cast(p!(n, n), 4, q::dtype_bytes(dt))?;
        let mut penultimate = dt;
        for layer in 0..cfg.num_layers {
            let prefix = format!("text_model.encoder.layers.{layer}");
            let norm = |suffix: &str| tensor(&format!("{prefix}.{suffix}"));
            let ln1w = norm("layer_norm1.weight")?;
            let ln1b = norm("layer_norm1.bias")?;
            let ln2w = norm("layer_norm2.weight")?;
            let ln2b = norm("layer_norm2.bias")?;
            let projection = |name: &str| -> Result<q::LinearWork> {
                let mut work = crate::text_encoder::clip_header_linear(
                    headers,
                    &format!("{prefix}.{name}"),
                    true,
                )?;
                if let Some(bits) = quant {
                    work.pack(bits)?;
                }
                Ok(work)
            };
            let ndt = self.text_norm(rows, dim, dt, ln1w.dtype, ln1b.dtype)?;
            let query = projection("self_attn.q_proj")?;
            let key = projection("self_attn.k_proj")?;
            let value = projection("self_attn.v_proj")?;
            let qdt = query.output_dtype(ndt);
            let kdt = key.output_dtype(ndt);
            let vdt = value.output_dtype(ndt);
            self.linear(&query, q::dtype_bytes(ndt), rows)?;
            self.linear(&key, q::dtype_bytes(ndt), rows)?;
            self.linear(&value, q::dtype_bytes(ndt), rows)?;
            let adt = Dtype::from_promoting_types(Dtype::from_promoting_types(qdt, kdt), vdt);
            for from in [qdt, kdt, vdt] {
                self.cast(p!(rows, dim), q::dtype_bytes(from), q::dtype_bytes(adt))?;
            }
            self.sdpa(
                [
                    q::dtype_bytes(adt),
                    b,
                    cfg.num_heads as u64,
                    cfg.num_heads as u64,
                    n,
                    n,
                    (cfg.model_dims / cfg.num_heads) as u64,
                ],
                true,
                false,
            )?;
            self.reshape_copy(q::dtype_bytes(adt), p!(rows, dim))?;
            let out = projection("self_attn.out_proj")?;
            self.linear(&out, q::dtype_bytes(adt), rows)?;
            let attention_dt = out.output_dtype(adt);
            let residual = Dtype::from_promoting_types(dt, attention_dt);
            self.cast(p!(rows, dim), q::dtype_bytes(dt), q::dtype_bytes(residual))?;
            self.cast(
                p!(rows, dim),
                q::dtype_bytes(attention_dt),
                q::dtype_bytes(residual),
            )?;
            self.op(q::dtype_bytes(residual), p!(rows, dim))?;
            let ndt = self.text_norm(rows, dim, residual, ln2w.dtype, ln2b.dtype)?;
            let fc1 = projection("mlp.fc1")?;
            let fc2 = projection("mlp.fc2")?;
            self.linear(&fc1, q::dtype_bytes(ndt), rows)?;
            let hidden = fc1.output_dtype(ndt);
            self.text_activation(
                q::dtype_bytes(hidden),
                p!(rows, fc1.output_width()),
                cfg.hidden_act,
            )?;
            self.linear(&fc2, q::dtype_bytes(hidden), rows)?;
            let mlp = fc2.output_dtype(hidden);
            dt = Dtype::from_promoting_types(residual, mlp);
            self.cast(p!(rows, dim), q::dtype_bytes(residual), q::dtype_bytes(dt))?;
            self.cast(p!(rows, dim), q::dtype_bytes(mlp), q::dtype_bytes(dt))?;
            self.op(q::dtype_bytes(dt), p!(rows, dim))?;
            if layer + 2 == cfg.num_layers {
                penultimate = dt;
            }
        }
        let gain = tensor("text_model.final_layer_norm.weight")?;
        let bias = tensor("text_model.final_layer_norm.bias")?;
        let final_dt = self.text_norm(rows, dim, dt, gain.dtype, bias.dtype)?;
        self.array(4, b)?; // actual host EOS gather indices, NOT an argmax tensor graph
        self.array(q::dtype_bytes(final_dt), p!(b, dim))?;
        let pooled = if cfg.projection_dim.is_some() {
            let mut projection =
                crate::text_encoder::clip_header_linear(headers, "text_projection", false)?;
            if let Some(bits) = quant {
                projection.pack(bits)?;
            }
            self.linear(&projection, q::dtype_bytes(final_dt), b)?;
            projection.output_dtype(final_dt)
        } else {
            final_dt
        };
        Ok((penultimate, pooled))
    }
    fn text_recipe(&mut self, inputs: &Inputs, regime: &Regime) -> Result<(Dtype, Dtype)> {
        use gen_core_family::sdxl_ldm::SdxlComponent as C;
        let headers = |path: &Path, component| -> Result<Vec<Header>> {
            match &inputs.fused {
                Some(f) => Ok(f.source.headers_with(component, &f.world)),
                None => mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
                    .map_err(|e| Error::Msg(e.to_string())),
            }
        };
        let h1 = headers(&inputs.te1, C::ClipL)?;
        let h2 = headers(&inputs.te2, C::ClipBigG)?;
        let b = u64::from(regime.sampling.guidance() > 1.0) + 1;
        let chunks = ceil(
            u64::try_from(regime.context_tokens).map_err(|_| overflow())?,
            inputs.arch.text_tokens as u64,
        )?;
        let n = inputs.arch.text_tokens as u64;
        let mut conditioning = Dtype::Float16;
        let mut pooled = Dtype::Float16;
        for _ in 0..chunks {
            self.array(4, p!(b, n))?; // long_prompt::legacy_batch / window host token tensor
            let (d1, _) =
                self.text_encoder_graph(&h1, &inputs.arch.te1, inputs.text_quant_bits, b, n)?;
            let (d2, p2) =
                self.text_encoder_graph(&h2, &inputs.arch.te2, inputs.text_quant_bits, b, n)?;
            conditioning = Dtype::from_promoting_types(d1, d2);
            pooled = p2;
            self.cast(
                p!(b, n, inputs.arch.te1.model_dims),
                q::dtype_bytes(d1),
                q::dtype_bytes(conditioning),
            )?;
            self.cast(
                p!(b, n, inputs.arch.te2.model_dims),
                q::dtype_bytes(d2),
                q::dtype_bytes(conditioning),
            )?;
            self.concat(
                q::dtype_bytes(conditioning),
                p!(b, n, inputs.arch.context_dim),
                2,
            )?;
        }
        self.concat(
            q::dtype_bytes(conditioning),
            p!(b, chunks, n, inputs.arch.context_dim),
            chunks,
        )?;
        self.array(4, p!(b, 6))?; // text_time_ids
        Ok((conditioning, pooled))
    }
}

// Full-request native certificates. Each delta is a cumulative request-only upper;
// the anchor supplies a measured nonnegative addend, never a subtractable lower.
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn certified_request_bounds(
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    certified_request_bounds_with_replay(
        anchor_request,
        None,
        anchor_pricing,
        request,
        request_pricing,
    )
}

pub(super) fn certified_request_bounds_with_replay(
    anchor_request: &IosPricingRequest<'_>,
    anchor_replay: Option<super::SdxlReplay<'_>>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    if anchor_request.provider != request.provider || request.provider != crate::model::MODEL_ID {
        return Err(Error::Unsupported(
            "SDXL native certificate needs one provider family".into(),
        ));
    }
    let anchor = resolve_bound_side(anchor_request, anchor_replay)?;
    let requested = resolve_bound_side(request, None)?;
    let mut anchor_inputs = Inputs::from_spec(anchor_request.spec, &anchor)?;
    let mut request_inputs = Inputs::from_spec(request.spec, &requested)?;
    certificate_core(
        CertificateSide {
            inputs: &mut anchor_inputs,
            regime: &anchor,
            pricing: anchor_pricing,
            spec: Some(anchor_request.spec),
            native_context_tokens: None,
        },
        CertificateSide {
            inputs: &mut request_inputs,
            regime: &requested,
            pricing: request_pricing,
            spec: Some(request.spec),
            native_context_tokens: Some(actual_context_tokens(request)?),
        },
    )
}

fn actual_context_tokens(req: &IosPricingRequest<'_>) -> Result<i32> {
    let generation = req.generation.ok_or_else(|| {
        Error::Unsupported(
            "SDXL native certificate: requested side requires its actual generation request".into(),
        )
    })?;
    let sampling = ResolvedSampling::for_pricing(Some(generation))?;
    let negative =
        (sampling.guidance() > 1.0).then(|| generation.negative_prompt.as_deref().unwrap_or(""));
    let windows = crate::model::conditioning_window_count(req.spec, &generation.prompt, negative)?;
    let tokens = windows.checked_mul(77).ok_or_else(overflow)?;
    i32::try_from(tokens).map_err(|_| overflow())
}

fn resolve_bound_side(
    req: &IosPricingRequest<'_>,
    replay: Option<super::SdxlReplay<'_>>,
) -> Result<Regime> {
    if req.generation.is_some() {
        return Regime::from_request(req);
    }
    let replay = replay.ok_or_else(|| {
        Error::Unsupported(
            "SDXL native certificate: payload-free side requires an SDXL sampling replay token"
                .into(),
        )
    })?;
    let recipe = SamplingRecipe::from_token(replay.token)?;
    let mut work = SamplingWorkFacts {
        reference_count: replay.true_reference_count,
        mask_count: replay.mask_count,
        control_count: replay.control_count,
        reference_image: None,
    };
    if let Some(source) = replay.source {
        source.validate_roles()?;
        let recorded = source.work();
        if recorded.reference_count != work.reference_count
            || recorded.mask_count != work.mask_count
            || recorded.control_count != work.control_count
        {
            return Err(Error::Unsupported(
                "SDXL native certificate: replay roles disagree with SamplingSourceFacts".into(),
            ));
        }
        work = recorded;
    }
    // from_recipe resolves actual factory/CFG/strength and validates exact aggregate roles.
    // IP needs its real source image facts: never substitute crop or output geometry.
    Regime::from_recipe(req, &recipe, work)
}
#[cfg(test)]
pub(crate) fn resolve_bound_side_for_test(
    req: &IosPricingRequest<'_>,
    replay: Option<super::SdxlReplay<'_>>,
) -> Result<Regime> {
    resolve_bound_side(req, replay)
}

fn same_component_graph(a: &Inputs, b: &Inputs) -> bool {
    a.fused.is_some() == b.fused.is_some()
        && a.te1 == b.te1
        && a.te2 == b.te2
        && a.unet == b.unet
        && a.vae == b.vae
        && a.quant_bits == b.quant_bits
        && a.text_quant_bits == b.text_quant_bits
        && a.streamable == b.streamable
        && a.fused.as_ref().map(|f| &f.vae_fix) == b.fused.as_ref().map(|f| &f.vae_fix)
}

fn source_profile(
    inputs: &mut Inputs,
    regime: &Regime,
    spec: Option<&LoadSpec>,
) -> Result<IosPricing> {
    q::with_backend(Backend::Metal, || price_inputs(inputs, regime, spec))
}

fn matches_source(expected: &IosPricing, supplied: &IosPricing) -> bool {
    expected.phases == supplied.phases
        && expected.windows == supplied.windows
        && expected.absent_phases == supplied.absent_phases
}

struct CertificateSide<'a> {
    inputs: &'a mut Inputs,
    regime: &'a Regime,
    pricing: &'a IosPricing,
    spec: Option<&'a LoadSpec>,
    native_context_tokens: Option<i32>,
}
fn certificate_core(
    anchor_side: CertificateSide<'_>,
    request_side: CertificateSide<'_>,
) -> Result<Vec<IosWindowBound>> {
    let CertificateSide {
        inputs: anchor_inputs,
        regime: anchor,
        pricing: anchor_pricing,
        spec: anchor_spec,
        ..
    } = anchor_side;
    let CertificateSide {
        inputs,
        regime: request,
        pricing: request_pricing,
        spec: request_spec,
        native_context_tokens,
    } = request_side;
    if !same_component_graph(anchor_inputs, inputs) {
        return Err(Error::Unsupported(
            "sdxl common-forward transfer needs the same source component graph".into(),
        ));
    }
    if !matches_source(
        &source_profile(anchor_inputs, anchor, anchor_spec)?,
        anchor_pricing,
    ) || !matches_source(
        &source_profile(inputs, request, request_spec)?,
        request_pricing,
    ) {
        return Err(Error::Unsupported(
            "sdxl transfer pricing is not the source-generated profile".into(),
        ));
    }
    let (page, witnesses) = anchor_witnesses(anchor_inputs, anchor, anchor_pricing)?;
    let mut native_request = request.clone();
    if let Some(tokens) = native_context_tokens {
        native_request.context_tokens = native_request.context_tokens.max(tokens);
    }
    let cpu = cumulative_native_upper(
        inputs,
        &native_request,
        request_spec,
        page,
        NativeBackend::Cpu,
    )?;
    let metal = cumulative_native_upper(
        inputs,
        &native_request,
        request_spec,
        page,
        NativeBackend::Metal,
    )?;
    let delta = i128::from(cpu.max(metal));
    let mut candidates = Vec::new();
    let mut anchor_classes = BTreeSet::new();
    for window in &anchor_pricing.windows {
        let Some(class) = window.profile_class() else {
            continue;
        };
        if !anchor_classes.insert(class) {
            continue;
        }
        for ((label, phase), found) in &witnesses {
            if label != &window.label || phase.is_some_and(|p| p != window.phase) {
                continue;
            }
            for witness in found {
                candidates.push(IosAnchorCandidate {
                    anchor_class: class.into(),
                    delta_growth_upper: delta,
                    physical_anchor_lower_bound: None,
                    witnesses: vec![witness.clone()],
                });
            }
        }
    }
    if candidates.is_empty() {
        return Err(Error::Unsupported(
            "SDXL native certificate has no actual anchor window witness".into(),
        ));
    }
    let mut classes = BTreeSet::new();
    Ok(request_pricing
        .windows
        .iter()
        .filter_map(|w| w.profile_class())
        .filter(|class| classes.insert(*class))
        .map(|class| IosWindowBound::Transfer {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: class.into(),
            anchor_candidates: candidates.clone(),
        })
        .collect())
}

type Witnesses = BTreeMap<(String, Option<IosPhase>), Vec<IosAnchorWitness>>;
fn anchor_witnesses(
    inputs: &Inputs,
    anchor: &Regime,
    pricing: &IosPricing,
) -> Result<(u64, Witnesses)> {
    let mut pages = BTreeSet::new();
    let mut witnesses: Witnesses = BTreeMap::new();
    let cfg = anchor.sampling.guidance() > 1.0;
    let latent_h =
        u64::try_from(anchor.height / inputs.arch.latent_scale).map_err(|_| overflow())?;
    let latent_w =
        u64::try_from(anchor.width / inputs.arch.latent_scale).map_err(|_| overflow())?;
    let elements = p!(latent_h, latent_w, 4);
    let steps = execution_steps(anchor)?;
    for (ci, capture) in pricing.observations.iter().enumerate() {
        if !capture.has_source_witness_coordinate() {
            continue;
        }
        for (ki, check) in capture.checks.iter().enumerate() {
            let marks = || {
                check
                    .context
                    .iter()
                    .filter_map(|&i| capture.metadata.get(i as usize))
            };
            let find = |kind: &str| marks().find(|m| m.provider == "sdxl" && m.kind == kind);
            let is = |fields: &BTreeMap<String, u64>, key: &str, expected| {
                fields.get(key) == Some(&expected)
            };
            // Actual production request metadata, not caller-invented class-string parsing.
            // pipeline.rs:899-923 (discrete),1297-1320 (curated); sampler.rs:302-330 tensor geometry.
            let identity = match anchor.sampling.route() {
                SamplingRoute::Curated(_) | SamplingRoute::CfgPp(_) => {
                    match (find("curated-callback"), find("curated-compute-input")) {
                        (Some(epoch), Some(input)) => {
                            is(&epoch.fields, "cfg_on", u64::from(cfg))
                                && is(
                                    &epoch.fields,
                                    "cfg_batched",
                                    u64::from(anchor.cfg_batching.is_batched()),
                                )
                                && is(
                                    &epoch.fields,
                                    "cfg_pp",
                                    u64::from(matches!(
                                        anchor.sampling.route(),
                                        SamplingRoute::CfgPp(_)
                                    )),
                                )
                                && epoch.fields.get("evaluation").is_some_and(|e| *e < steps)
                                && tensor_identity(&input.fields, latent_h, latent_w, elements)
                        }
                        _ => false,
                    }
                }
                _ => match (find("discrete-epoch"), find("discrete-state")) {
                    (Some(epoch), Some(input)) => {
                        is(&epoch.fields, "steps", steps)
                            && is(&epoch.fields, "cfg_on", u64::from(cfg))
                            && is(
                                &epoch.fields,
                                "cfg_batched",
                                u64::from(anchor.cfg_batching.is_batched()),
                            )
                            && is(
                                &epoch.fields,
                                "inpaint",
                                u64::from(anchor.route == Route::Mask),
                            )
                            && is(
                                &epoch.fields,
                                "control_branches",
                                inputs.controls.len() as u64,
                            )
                            && is(
                                &epoch.fields,
                                "ip",
                                u64::from(anchor.route == Route::IpAdapter),
                            )
                            && epoch.fields.get("evaluation").is_some_and(|e| *e < steps)
                            && tensor_identity(&input.fields, latent_h, latent_w, elements)
                    }
                    _ => false,
                },
            };
            let serial = find("serial-output-start");
            let serial_matches = serial.is_some_and(|mark| {
                is(&mark.fields, "count", u64::from(anchor.count))
                    && mark
                        .fields
                        .get("output")
                        .is_some_and(|output| *output < u64::from(anchor.count))
            });
            let epoch_matches = match anchor.sampling.route() {
                SamplingRoute::Curated(solver) | SamplingRoute::CfgPp(solver) => {
                    let matched = match (find("curated-callback"), find("curated-compute-input")) {
                        (Some(epoch), Some(input)) => epoch
                            .fields
                            .get("evaluation")
                            .copied()
                            .filter(|evaluation| {
                                input.fields.get("evaluation") == Some(evaluation)
                            }),
                        _ => None,
                    };
                    if let Some(evaluation) = matched {
                        let alpha = alpha_schedule();
                        let sigmas = anchor.sampling.curated_sigmas(
                            &alpha,
                            anchor
                                .sampling
                                .effective_init_strength(anchor.has_init, false),
                        )?;
                        let actual_solver = if matches!(
                            anchor.sampling.route(),
                            SamplingRoute::CfgPp(mlx_gen::Solver::Ddim)
                        ) {
                            mlx_gen::Solver::Euler
                        } else {
                            solver
                        };
                        let epochs = actual_solver.logical_epochs(&sigmas).map_err(Error::from)?;
                        epochs
                            .get(usize::try_from(evaluation).map_err(|_| overflow())?)
                            .is_some_and(|epoch| {
                                use mlx_gen::gen_core::sampling::solvers::SolverEvaluation;
                                let kind = match epoch.evaluation {
                                    SolverEvaluation::Primary => "solver_primary",
                                    SolverEvaluation::HeunCorrector => "solver_heun_corrector",
                                    SolverEvaluation::SdeMidpoint => "solver_sde_midpoint",
                                };
                                marks().any(|mark| {
                                    mark.provider == "curated"
                                        && mark.kind == kind
                                        && is(&mark.fields, "solver_code", actual_solver as u64)
                                        && is(&mark.fields, "step", epoch.step as u64)
                                        && is(
                                            &mark.fields,
                                            "sigma_bits",
                                            u64::from(epoch.sigma.to_bits()),
                                        )
                                        && is(
                                            &mark.fields,
                                            "next_sigma_bits",
                                            u64::from(epoch.next_sigma.to_bits()),
                                        )
                                })
                            })
                    } else {
                        false
                    }
                }
                _ => match (find("discrete-epoch"), find("discrete-state")) {
                    (Some(epoch), Some(input)) => {
                        epoch.fields.get("evaluation").is_some_and(|evaluation| {
                            input.fields.get("evaluation") == Some(evaluation)
                        })
                    }
                    _ => false,
                },
            };
            if !identity || !serial_matches || !epoch_matches {
                continue;
            }
            let Some(scope) = marks().find(|m| m.provider == "mlx" && m.kind == "allocation-scope")
            else {
                continue;
            };
            if scope.fields.get("setter_succeeded") != Some(&1)
                || scope.fields.get("cache_query_succeeded") != Some(&1)
            {
                continue;
            }
            let Some(page) = scope
                .fields
                .get("page_bytes")
                .copied()
                .filter(|p| p.is_power_of_two())
            else {
                continue;
            };
            pages.insert(page);
            witnesses
                .entry((check.label.clone(), check.phase))
                .or_default()
                .push(IosAnchorWitness {
                    capture: u32::try_from(ci).map_err(|_| overflow())?,
                    check: u32::try_from(ki).map_err(|_| overflow())?,
                });
        }
    }
    let page = match pages.into_iter().collect::<Vec<_>>()[..] {
        [page] => page,
        [] => {
            return Err(Error::Unsupported(
                "SDXL native certificate has no actual source-matched anchor witness".into(),
            ))
        }
        _ => {
            return Err(Error::Unsupported(
                "SDXL anchor witnesses disagree on the native allocator page".into(),
            ))
        }
    };
    Ok((page, witnesses))
}
fn tensor_identity(fields: &BTreeMap<String, u64>, h: u64, w: u64, n: u64) -> bool {
    fields.get("rank") == Some(&4)
        && fields.get("dim0") == Some(&1)
        && fields.get("dim1") == Some(&h)
        && fields.get("dim2") == Some(&w)
        && fields.get("dim3") == Some(&4)
        && fields.get("elements") == Some(&n)
}

#[cfg(test)]
pub(crate) fn source_pricing_for_test(inputs: &mut Inputs, regime: &Regime) -> Result<IosPricing> {
    source_profile(inputs, regime, None)
}
#[cfg(test)]
pub(crate) fn certified_request_bounds_for_test(
    anchor_inputs: &Inputs,
    anchor: &Regime,
    anchor_pricing: &IosPricing,
    inputs: &Inputs,
    request: &Regime,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    let copy = |i: &Inputs| -> Result<Inputs> {
        Ok(Inputs {
            fused: match &i.fused {
                Some(f) => Some(FusedInputs {
                    source: crate::ldm::pricing::FusedHeaderSource::read(&i.unet)?,
                    vae_fix: f.vae_fix.clone(),
                    ip_source: f.ip_source.clone(),
                    world: f.world.clone(),
                }),
                None => None,
            },
            arch: i.arch.clone(),
            te1: i.te1.clone(),
            te2: i.te2.clone(),
            unet: i.unet.clone(),
            vae: i.vae.clone(),
            controls: i.controls.clone(),
            ip: i.ip.clone(),
            quant_bits: i.quant_bits,
            text_quant_bits: i.text_quant_bits,
            streamable: i.streamable,
        })
    };
    certificate_core(
        CertificateSide {
            inputs: &mut copy(anchor_inputs)?,
            regime: anchor,
            pricing: anchor_pricing,
            spec: None,
            native_context_tokens: None,
        },
        CertificateSide {
            inputs: &mut copy(inputs)?,
            regime: request,
            pricing: request_pricing,
            spec: None,
            native_context_tokens: None,
        },
    )
}

/// Enumerate correlated source winners, not independently maximized tensor dimensions. The
/// window-pricing text DP is deliberately NOT reused: it optimizes individual windows, whereas
/// this certificate maximizes a complete cumulative graph. Each assignment remains whole.
pub(super) fn native_worlds(inputs: &Inputs) -> Result<Vec<text_alias::Picks>> {
    let Some(f) = &inputs.fused else {
        return Ok(vec![Vec::new()]);
    };
    let mut worlds = vec![Vec::new()];
    for (group, alias) in f.source.aliases.iter().enumerate() {
        let count = worlds
            .len()
            .checked_mul(alias.assignments.len())
            .ok_or_else(overflow)?;
        let mut next = Vec::new();
        next.try_reserve(count)
            .map_err(|e| Error::Msg(e.to_string()))?;
        for world in worlds {
            for assignment in 0..alias.assignments.len() {
                let mut selected = world.clone();
                selected.push((group, assignment));
                next.push(selected);
            }
        }
        worlds = next;
    }
    Ok(worlds)
}
/// AE's source pricer keeps native-shape failure prefixes. Preserve their reached
/// native allocations too; a complete source graph, overflow, environment override,
/// or pricer-invariant error cannot borrow this exception.
fn retain_source_prefix(
    ledger: &NativeLedger,
    result: Result<()>,
    source_complete: bool,
) -> Result<u64> {
    match result {
        Ok(()) => Ok(ledger.total),
        Err(error) if !source_complete && q::is_native_shape_failure(&error) => Ok(ledger.total),
        Err(error) => Err(error),
    }
}

fn cumulative_native_upper(
    inputs: &mut Inputs,
    r: &Regime,
    spec: Option<&LoadSpec>,
    page: u64,
    backend: NativeBackend,
) -> Result<u64> {
    let worlds = native_worlds(inputs)?;
    let mut maximum = 0;
    for world in worlds {
        if let Some(f) = &mut inputs.fused {
            f.world = world;
        }
        let mut priced = match spec {
            Some(s) => Priced::build_configured(inputs, s)?,
            None => Priced::build(inputs)?,
        };
        let selected_source = q::with_backend(Backend::Metal, || price_world(inputs, r, spec))?;
        let source_complete = selected_source
            .windows
            .iter()
            .any(|w| w.label == q::DECODED_TO_IMAGE);
        if let Some(s) = spec.filter(|s| !s.adapters.is_empty()) {
            // A failed alternate source order still allocates its reached install prefix.
            // Keep that whole prefix, plus possible staged text work, alongside successes.
            let mut prefix = NativeLedger::new(page, backend)?;
            prefix.ancestral_constructor()?;
            let prefix_result = (|| {
                prefix.load_recipe(inputs, r, Some(s))?;
                prefix.text_recipe(inputs, r)?;
                Ok(())
            })();
            maximum = maximum.max(retain_source_prefix(
                &prefix,
                prefix_result,
                source_complete,
            )?);
            q::with_backend(Backend::Metal, || {
                priced.price_configured_observing(
                    s,
                    &inputs.unet,
                    inputs.quant_bits,
                    r,
                    |outcome, selected, _, selected_profile| {
                        let mut l = NativeLedger::new(page, backend)?;
                        l.ancestral_constructor()?;
                        let result = (|| {
                            l.load_recipe_selected(inputs, r, Some(s), Some(outcome))?;
                            l.request_graph(inputs, r, selected)
                        })();
                        let complete = selected_profile
                            .windows
                            .iter()
                            .any(|w| w.label == q::DECODED_TO_IMAGE);
                        maximum = maximum.max(retain_source_prefix(&l, result, complete)?);
                        Ok(())
                    },
                )
            })?;
        } else {
            let mut l = NativeLedger::new(page, backend)?;
            // A staged model may construct the default sampler before a later
            // text/heavy failure. Retain that possible earlier construction too.
            l.ancestral_constructor()?;
            let result = (|| {
                l.load_recipe(inputs, r, spec)?;
                l.request_graph(inputs, r, &priced)
            })();
            maximum = maximum.max(retain_source_prefix(&l, result, source_complete)?);
        }
    }
    if let Some(f) = &mut inputs.fused {
        f.world = Vec::new();
    }
    Ok(maximum)
}

/// A complete request-initialization + ONE-forward native allocation upper. The legacy
/// synchronized-retirement transfer cannot cancel an img2img first-use prefix against a
/// text anchor's steady F32 body merely because the forward dtypes agree. Reuse the same
/// source-operation ledgers as the full-request certificate, with no retirement/donation
/// credit for this prefix. Work is bounded by architecture, not requested step count.
pub(super) fn uncancelled_forward_prefix_upper(
    inputs: &mut Inputs,
    r: &Regime,
    page: u64,
) -> Result<u64> {
    if r.sampling.route() != SamplingRoute::Ancestral
        || !matches!(r.route, Route::Text | Route::Reference | Route::Mask)
        || !matches!(inputs.quant_bits, Some(4 | 8))
    {
        return Err(invalid(
            "uncancelled prefix requires the packed ancestral forward domain",
        ));
    }
    let steps = execution_steps(r)?;
    let nodes = plus(steps, 1)?;
    let schedules = if r.route == Route::Mask { 2 } else { 1 };
    let rounded_nodes = product(&[nodes, schedules])?;
    let worlds = native_worlds(inputs)?;
    let mut maximum = 0;
    for world in worlds {
        if let Some(f) = &mut inputs.fused {
            f.world = world;
        }
        let priced = Priced::build(inputs)?;
        let (first, steady) = q::with_backend(Backend::Metal, || priced.ancestral_states(r))?;
        for state in [first, steady]
            .into_iter()
            .enumerate()
            .filter_map(|(index, state)| (index == 0 || state != first).then_some(state))
        {
            for backend in [NativeBackend::Cpu, NativeBackend::Metal] {
                let mut ledger = NativeLedger::new(page, backend)?;
                ledger.ancestral_constructor()?;
                ledger.load_recipe(inputs, r, None)?;
                let (memory_dtype, pooled_dtype) = ledger.text_recipe(inputs, r)?;
                let rows = r.conditioning_rows() as u64;
                let chunks = ceil(r.context_tokens as u64, inputs.arch.text_tokens as u64)?;
                let context = ForwardContext {
                    inputs,
                    priced: &priced,
                    memory: GraphTensor::new(
                        &[
                            rows,
                            p!(chunks, inputs.arch.text_tokens),
                            inputs.arch.context_dim as u64,
                        ],
                        memory_dtype,
                    )?,
                    pooled: GraphTensor::new(
                        &[rows, priced.te.1.pooled_width() as u64],
                        pooled_dtype,
                    )?,
                    time_ids: GraphTensor::new(&[rows, 6], Dtype::Float32)?,
                    controls: Vec::new(),
                    ip: None,
                };
                let h = r.height as u64 / inputs.arch.latent_scale as u64;
                let w = r.width as u64 / inputs.arch.latent_scale as u64;
                let elements = p!(h, w, 4);
                if r.has_init {
                    let image =
                        GraphTensor::new(&[1, r.height as u64, r.width as u64, 3], Dtype::Float32)?;
                    ledger.array(4, image.elements()?)?;
                    let tile = r
                        .tiling
                        .as_ref()
                        .and_then(|t| t.spatial.as_ref())
                        .map(|s| s.tile_px as u64);
                    ledger.vae_encode_graph(&priced.vae, image, tile)?;
                }
                if r.route == Route::Mask {
                    ledger.array(4, elements / 4)?;
                }
                // Actual f16 schedule nodes: F32 scalar -> F16 -> F32. Count them
                // multiplicatively (including the mask's separate schedule), never build N nodes.
                ledger.repeated(4, 1, p!(rounded_nodes, 2))?;
                ledger.repeated(2, 1, rounded_nodes)?;
                ledger.initial_latent(r, elements)?;
                let prior = r
                    .count
                    .checked_sub(1)
                    .ok_or_else(|| invalid("zero serial count"))?;
                ledger.repeated(q::dtype_bytes(steady), elements, prior as u64)?;
                ledger.forward_request(&context, r, state)?;
                maximum = maximum.max(ledger.total);
            }
        }
    }
    if let Some(f) = &mut inputs.fused {
        f.world = Vec::new();
    }
    Ok(maximum)
}

struct ForwardContext<'a> {
    inputs: &'a Inputs,
    priced: &'a Priced,
    memory: GraphTensor,
    pooled: GraphTensor,
    time_ids: GraphTensor,
    controls: Vec<GraphTensor>,
    ip: Option<GraphTensor>,
}
impl NativeLedger {
    fn request_graph(&mut self, inputs: &Inputs, r: &Regime, p: &Priced) -> Result<()> {
        // The per-world caller already charged the default sampler constructor,
        // including a possible earlier construction on a later-failing staged request.
        let (memory_dt, pooled_dt) = self.text_recipe(inputs, r)?;
        let b = 1 + u64::from(r.sampling.guidance() > 1.0);
        let chunks = ceil(r.context_tokens as u64, inputs.arch.text_tokens as u64)?;
        let memory = GraphTensor::new(
            &[
                b,
                p!(chunks, inputs.arch.text_tokens),
                inputs.arch.context_dim as u64,
            ],
            memory_dt,
        )?;
        let pooled = GraphTensor::new(&[b, p.te.1.pooled_width() as u64], pooled_dt)?;
        let time_ids = GraphTensor::new(&[b, 6], Dtype::Float32)?;
        let mut controls = Vec::new();
        if r.route == Route::Control {
            for f in &p.controls {
                let mut pixels =
                    GraphTensor::new(&[1, r.height as u64, r.width as u64, 3], Dtype::Float32)?;
                self.array(4, pixels.elements()?)?;
                if b == 2 {
                    pixels = self.cg_concat(&[pixels.clone(), pixels], 0)?;
                }
                controls.push(self.control_condition_graph(f, pixels)?);
            }
        }
        let ip = if r.route == Route::IpAdapter {
            let f =
                p.ip.as_ref()
                    .ok_or_else(|| Error::MissingTensor("SDXL IP image encoder".into()))?;
            let size = inputs.arch.vision.image_size as u64;
            // Preprocessing is host resize/crop/normalize, followed by ONE native F32 array.
            let pixels = GraphTensor::new(&[1, size, size, 3], Dtype::Float32)?;
            self.array(4, pixels.elements()?)?;
            let pixels = self.cg_cast(&pixels, f.resampler.dtype())?;
            let tokens = self.ip_image_graph(f, pixels, inputs.arch.vision.quick_gelu)?;
            Some(self.ip_cfg_graph(tokens, b == 2)?)
        } else {
            None
        };
        let tile = r
            .tiling
            .as_ref()
            .and_then(|t| t.spatial.as_ref())
            .map(|s| s.tile_px as u64);
        if r.has_init {
            let image = GraphTensor::new(&[1, r.height as u64, r.width as u64, 3], Dtype::Float32)?;
            self.array(4, image.elements()?)?;
            self.vae_encode_graph(&p.vae, image, tile)?;
        }
        if r.route == Route::Mask {
            // preprocess_mask stores [1,H/8,W/8,1] F32 after HOST luma/resize/normalization.
            self.array(
                4,
                p!(
                    r.height as u64 / inputs.arch.latent_scale as u64,
                    r.width as u64 / inputs.arch.latent_scale as u64
                ),
            )?;
        }
        let context = ForwardContext {
            inputs,
            priced: p,
            memory,
            pooled,
            time_ids,
            controls,
            ip,
        };
        let h = r.height as u64 / inputs.arch.latent_scale as u64;
        let w = r.width as u64 / inputs.arch.latent_scale as u64;
        let n = p!(h, w, 4);
        let mut finals = Vec::new();
        // EXACT serial output count: only seed-independent conditioning/init is outside this loop.
        for _ in 0..r.count {
            let latent = self.sampler_request(&context, r, n)?;
            finals.push(GraphTensor::new(&[1, h, w, 4], native_dtype(latent.width))?);
        }
        for latent in finals {
            let decode_tile = r
                .tiling
                .as_ref()
                .filter(|cfg| {
                    cfg.needs_tiling(
                        mlx_gen::tiling::VaeTiling {
                            spatial_scale: inputs.arch.latent_scale,
                            temporal_scale: 1,
                            causal_temporal: false,
                            full_res_channels: inputs.arch.vae.block_out_channels[0],
                        },
                        1,
                        h as i32,
                        w as i32,
                    )
                })
                .and(tile);
            let image = self.vae_decode_graph(&p.vae, latent, decode_tile)?;
            let image = self.cg_cast(&image, Dtype::Float32)?;
            self.rgb_graph(image.elements()?)?;
        }
        Ok(())
    }
    fn forward_request(
        &mut self,
        c: &ForwardContext<'_>,
        r: &Regime,
        input_dtype: Dtype,
    ) -> Result<GraphTensor> {
        let h = r.height as u64 / c.inputs.arch.latent_scale as u64;
        let w = r.width as u64 / c.inputs.arch.latent_scale as u64;
        let cfg = r.sampling.guidance() > 1.0;
        let sequential = cfg && !r.cfg_batching.is_batched();
        let rows = if cfg && !sequential { 2 } else { 1 };
        let input = GraphTensor::new(&[rows, h, w, 4], input_dtype)?;
        if cfg && !sequential {
            self.concat(input.width(), input.elements()?, 2)?;
        }
        // Preserve the real sampler dtype; caller-owned historical casts are counted there.
        let halves = if sequential { 2 } else { 1 };
        let mut output = None;
        for _ in 0..halves {
            if c.inputs.streamable {
                self.stream_reread(c.inputs)?;
            }
            let regime = GraphRegime {
                attention: mlx_gen::attention::AttentionPlan::UNBOUNDED,
                conditioning_rows: c.memory.shape[0],
                cfg_row: sequential,
            };
            let mut residual = None;
            for (index, (f, embed)) in c.priced.controls.iter().zip(&c.controls).enumerate() {
                let embed = if sequential && embed.shape[0] == 2 {
                    self.array(4, 1)?;
                    let mut row = embed.clone();
                    row.shape[0] = 1;
                    self.array(row.width(), row.elements()?)?;
                    row
                } else {
                    embed.clone()
                };
                let effects = c.priced.control_effects[index]
                    .as_ref()
                    .map_or_else(q::EffectScope::native, q::EffectScope::root);
                let branch = self.control_graph(
                    f,
                    [
                        input.clone(),
                        c.memory.clone(),
                        c.pooled.clone(),
                        c.time_ids.clone(),
                        embed,
                    ],
                    regime,
                    &effects,
                )?;
                residual = Some(match residual {
                    None => branch,
                    Some(prev) => self.control_merge_graph(prev, branch)?,
                });
            }
            let effects = c
                .priced
                .unet_effects
                .as_ref()
                .map_or_else(q::EffectScope::native, q::EffectScope::root);
            output = Some(self.unet_graph(
                &c.priced.unet,
                [
                    input.clone(),
                    c.memory.clone(),
                    c.pooled.clone(),
                    c.time_ids.clone(),
                ],
                regime,
                c.ip.clone(),
                residual.as_ref(),
                &effects,
            )?);
        }
        let mut eps = output.take().ok_or_else(overflow)?;
        if cfg {
            if !sequential {
                eps.shape[0] = 1;
                for _ in 0..2 {
                    self.array(4, 1)?;
                    self.array(eps.width(), eps.elements()?)?;
                }
            }
            self.op(eps.width(), eps.elements()?)?; // cond-uncond
            self.typed_scalar(eps.width())?;
            self.repeated(eps.width(), eps.elements()?, 2)?; // cfg product and sum
        }
        Ok(eps)
    }
    fn rgb_graph(&mut self, n: u64) -> Result<()> {
        // pipeline.rs:1435-1455; NO single-consumer donation credit.
        self.scalar(4)?;
        self.op(4, n)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.op(4, n)
    }
}

/// Give every source window an address for this FULL-request native domain. An address is not
/// an allocation proof: certified_request_bounds separately validates the source profile and
/// computes its request-only CPU/Metal ledgers. Legacy exact forward classes remain unchanged.
pub(super) fn classify_windows(pricing: &mut IosPricing, r: &Regime) -> Result<()> {
    for window in &mut pricing.windows {
        if window.class.is_none() {
            window.class = Some(format!(
                "sdxl/full-request-native-v1/{:?}/{}",
                window.phase, window.label
            ));
        }
        if window.basis.is_none() {
            let rows = match window.phase {
                IosPhase::Load => 0,
                IosPhase::Conditioning => 1 + u64::from(r.sampling.guidance() > 1.0),
                IosPhase::Denoise => r.model_rows() as u64,
                _ => 1,
            };
            window.basis = Some(IosWorkBasis {
                forward_rows: rows,
                token_count: r.context_tokens as u64,
                model_width: r.width as u32,
                model_height: r.height as u32,
                ..IosWorkBasis::default()
            });
        }
        if window.owner_domain.is_empty() {
            window.owner_domain = "sdxl/ios_pricing/price_inputs/request-native-active-v1".into();
        }
    }
    Ok(())
}

fn alpha_schedule() -> mlx_gen::AlphaSchedule {
    let c = crate::config::DiffusionConfig::sdxl_base();
    mlx_gen::AlphaSchedule::scaled_linear(c.num_train_steps, c.beta_start, c.beta_end)
}
fn execution_steps(r: &Regime) -> Result<u64> {
    let a = alpha_schedule();
    let n = match r.sampling.route() {
        SamplingRoute::Ancestral => {
            r.sampling
                .ancestral_window(
                    crate::config::DiffusionConfig::sdxl_base().num_train_steps as f32,
                    r.sampling
                        .effective_init_strength(r.has_init, r.route == Route::Mask),
                )?
                .0
        }
        SamplingRoute::Curated(_) | SamplingRoute::CfgPp(_) => {
            let sigmas = r
                .sampling
                .curated_sigmas(&a, r.sampling.effective_init_strength(r.has_init, false))?;
            r.sampling.logical_epochs(&sigmas)?.len()
        }
        _ => r.sampling.acceleration_coefficients(&a)?.len(),
    };
    u64::try_from(n).map_err(|_| overflow())
}

#[derive(Clone)]
struct SymbolicLatent {
    n: u64,
    width: u64,
}
struct CumulativeMlxOps {
    ledger: std::cell::RefCell<NativeLedger>,
}
impl mlx_gen::gen_core::sampling::LatentOps for CumulativeMlxOps {
    type Latent = SymbolicLatent;
    fn scale(&self, x: &SymbolicLatent, c: f32) -> mlx_gen::gen_core::Result<SymbolicLatent> {
        if c == 1.0 {
            return Ok(x.clone());
        }
        let mut l = self.ledger.borrow_mut();
        l.scalar(4)?;
        l.cast(1, 4, x.width)?;
        l.op(x.width, x.n)?;
        Ok(x.clone())
    }
    fn add(
        &self,
        a: &SymbolicLatent,
        b: &SymbolicLatent,
    ) -> mlx_gen::gen_core::Result<SymbolicLatent> {
        let width = a.width.max(b.width);
        let mut l = self.ledger.borrow_mut();
        l.cast(a.n, a.width, width)?;
        l.cast(b.n, b.width, width)?;
        l.op(width, a.n)?;
        Ok(SymbolicLatent { n: a.n, width })
    }
    fn sub(
        &self,
        a: &SymbolicLatent,
        b: &SymbolicLatent,
    ) -> mlx_gen::gen_core::Result<SymbolicLatent> {
        self.add(a, b)
    }
    fn axpy(
        &self,
        a: f32,
        x: &SymbolicLatent,
        _b: f32,
        y: &SymbolicLatent,
    ) -> mlx_gen::gen_core::Result<SymbolicLatent> {
        // MlxLatentOps override (sampler.rs:346-360), not LatentOps' alias-eliding default.
        let mut l = self.ledger.borrow_mut();
        l.scalar(4)?;
        l.cast(1, 4, y.width)?;
        l.op(y.width, y.n)?;
        if a != 1.0 {
            l.scalar(4)?;
            l.cast(1, 4, x.width)?;
            l.op(x.width, x.n)?;
        }
        let width = x.width.max(y.width);
        l.cast(x.n, x.width, width)?;
        l.cast(y.n, y.width, width)?;
        l.op(width, x.n)?;
        Ok(SymbolicLatent { n: x.n, width })
    }
    fn randn_like(
        &self,
        x: &SymbolicLatent,
        _seed: u64,
        _step: usize,
    ) -> mlx_gen::gen_core::Result<SymbolicLatent> {
        self.ledger.borrow_mut().random_normal(x.n)?;
        Ok(SymbolicLatent { n: x.n, width: 4 })
    }
}

impl NativeLedger {
    fn sigma_interpolation(&mut self, round_half: bool) -> Result<()> {
        // sigma_arr: four F32 source scalars, two products and their sum. The schedule's
        // half-rounding performs two independent F32->F16->F32 scalar round trips.
        if round_half {
            for _ in 0..2 {
                self.scalar(4)?;
                self.scalar(2)?;
                self.scalar(4)?;
            }
        }
        self.repeated(4, 1, 7)
    }
    fn typed_scalar(&mut self, width: u64) -> Result<()> {
        self.scalar(4)?;
        self.cast(1, 4, width)
    }
    fn ancestral_constructor(&mut self) -> Result<()> {
        let cfg = crate::config::DiffusionConfig::sdxl_base();
        let n = cfg.num_train_steps as u64;
        // sampler.rs:69-97: arange/divide; scaled-linear mul/add/square; subtract,
        // cumprod, subtract/divide/sqrt; zero then concatenate. Scalar arguments separate.
        self.array(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.op(4, n)?;
        self.scalar(4)?;
        self.op(4, n)?;
        self.cumprod(4, n)?;
        self.scalar(4)?;
        self.repeated(4, n, 3)?;
        self.array(4, 1)?;
        self.concat(4, plus(n, 1)?, 2)
    }
    fn ancestral_update(&mut self, n: u64, state_width: u64, eps_width: u64) -> Result<u64> {
        for _ in 0..2 {
            self.sigma_interpolation(true)?;
            self.cast(1, 4, eps_width)?;
        }
        // sigma2,prev2; typed one,two; renorm add+rsqrt; sigma_up sub/mul/div/sqrt;
        // sigma_down power/sub/sqrt; dt subtraction; sqrt(sigma2+1).
        self.repeated(eps_width, 1, 2)?;
        self.typed_scalar(eps_width)?;
        self.typed_scalar(eps_width)?;
        self.repeated(eps_width, 1, 12)?;
        let out_width = state_width.max(eps_width);
        self.cast(n, state_width, out_width)?;
        self.cast(n, eps_width, out_width)?;
        self.op(out_width, n)?; // x_t * sqrt(sigma2+1)
        self.op(eps_width, n)?; // eps * dt
        self.cast(n, eps_width, out_width)?;
        self.op(out_width, n)?; // scaled_x sum
        self.random_normal(n)?;
        self.cast(n, 4, out_width)?;
        self.repeated(out_width, n, 3)?; // noise multiply, sum, renormalize; no final-zero shortcut
        Ok(out_width)
    }
    fn initial_latent(&mut self, r: &Regime, n: u64) -> Result<SymbolicLatent> {
        self.random_normal(n)?;
        match r.sampling.route() {
            SamplingRoute::Ancestral => {
                if r.has_init {
                    self.sigma_interpolation(false)?;
                    self.repeated(4, 1, 4)?; // square, one, add, rsqrt
                    self.repeated(4, n, 3)?; // noise*s, init+..., renorm
                    Ok(SymbolicLatent { n, width: 4 })
                } else {
                    self.scalar(4)?;
                    self.scalar(4)?;
                    self.repeated(4, 1, 3)?;
                    self.repeated(4, n, 2)?;
                    self.cast(n, 4, 2)?;
                    Ok(SymbolicLatent { n, width: 2 })
                }
            }
            SamplingRoute::Curated(_) | SamplingRoute::CfgPp(_) => {
                self.scalar(4)?;
                self.op(4, n)?;
                if r.has_init {
                    self.op(4, n)?;
                }
                Ok(SymbolicLatent { n, width: 4 })
            }
            _ => {
                // LCM/TCD scale_initial=1 aliases; Lightning's nonunity scale allocates.
                if r.sampling.route() == SamplingRoute::Lightning {
                    self.scalar(4)?;
                    self.op(4, n)?;
                }
                Ok(SymbolicLatent { n, width: 4 })
            }
        }
    }
    fn mask_blend(&mut self, n: u64, width: u64) -> Result<u64> {
        self.sigma_interpolation(false)?;
        self.repeated(4, 1, 4)?;
        self.repeated(4, n, 3)?;
        self.scalar(4)?;
        self.op(4, n / 4)?;
        self.cast(n, width, 4)?;
        self.repeated(4, n, 3)?;
        Ok(4)
    }
    fn preview_graph(&mut self, n: u64, width: u64, ve: bool) -> Result<()> {
        // A source upper over enabled AND inert sinks: charge every actual callback candidate.
        // Never depend on a preview flag absent from Regime. Native host RGB storage is excluded.
        self.cast(n, width, 4)?;
        if ve {
            self.scalar(4)?;
            self.op(4, n)?;
        }
        self.array(4, 12)?;
        self.array(4, 3)?;
        self.matmul(4, 1, n / 4, 3, 4)?;
        self.op(4, p!(n / 4, 3))?; // bias
        self.scalar(4)?;
        self.op(4, p!(n / 4, 3))?;
        self.scalar(4)?;
        self.op(4, p!(n / 4, 3))?;
        self.scalar(4)?;
        self.op(4, p!(n / 4, 3))?;
        self.op(4, p!(n / 4, 3))
    }
    /// The shared solver factories require an owned/static operation type. Restore
    /// its ledger on BOTH success and error so a native failure prefix cannot lose
    /// allocations when a callback returns early through `?`.
    fn replay_owned<T>(&mut self, run: impl FnOnce(&CumulativeMlxOps) -> Result<T>) -> Result<T> {
        let empty = NativeLedger::new(self.page, self.backend)?;
        let owned = std::mem::replace(self, empty);
        let ops = CumulativeMlxOps {
            ledger: std::cell::RefCell::new(owned),
        };
        let result = run(&ops);
        *self = ops.ledger.into_inner();
        result
    }
    fn sampler_request(
        &mut self,
        c: &ForwardContext<'_>,
        r: &Regime,
        n: u64,
    ) -> Result<SymbolicLatent> {
        let initial = self.initial_latent(r, n)?;
        let alpha = alpha_schedule();
        match r.sampling.route() {
            SamplingRoute::Curated(solver) | SamplingRoute::CfgPp(solver) => {
                use mlx_gen::gen_core::sampling::{
                    cfgpp_denoise, cfgpp_sampler_for, denoise, LatentOps,
                };
                let sigmas = r.sampling.curated_sigmas(
                    &alpha,
                    r.sampling.effective_init_strength(r.has_init, false),
                )?;
                solver.logical_epochs(&sigmas).map_err(Error::from)?;
                let ms = mlx_gen::DiscreteModelSampling::sdxl(&alpha);
                self.replay_owned(|ops| {
                    let mut predict =
                        |input: &SymbolicLatent,
                         _time: f32|
                         -> mlx_gen::gen_core::Result<SymbolicLatent> {
                            let mut l = ops.ledger.borrow_mut();
                            l.preview_graph(input.n, input.width, true)?;
                            l.cast(n, input.width, 2)?;
                            let eps = l.forward_request(c, r, crate::model::DTYPE)?;
                            Ok(SymbolicLatent {
                                n,
                                width: q::dtype_bytes(eps.dtype),
                            })
                        };
                    let result = if matches!(r.sampling.route(), SamplingRoute::CfgPp(_)) {
                        let sampler =
                            cfgpp_sampler_for::<CumulativeMlxOps>(solver).ok_or_else(|| {
                                Error::Unsupported(
                                    "SDXL selected solver has no production CFG++ factory".into(),
                                )
                            })?;
                        let mut callback = |x: &SymbolicLatent, sigma: f32| {
                            cfgpp_denoise(ops, &ms, x, sigma, |input, time| {
                                let guided = predict(input, time)?;
                                Ok((guided.clone(), guided))
                            })
                        };
                        sampler
                            .sample(ops, &mut callback, initial, &sigmas)
                            .map_err(Error::from)?
                    } else {
                        let mut callback = |x: &SymbolicLatent, sigma: f32| {
                            denoise(ops, &ms, x, sigma, &mut predict)
                        };
                        solver
                            .boxed::<CumulativeMlxOps>()
                            .sample(ops, &ms, &mut callback, initial, &sigmas, 0)
                            .map_err(Error::from)?
                    };
                    // Explicitly use the trait here so replay cannot silently drift to a tensor backend.
                    let _ = ops.scale(&result, 1.0).map_err(Error::from)?;
                    Ok(result)
                })
            }
            SamplingRoute::Ancestral => {
                let steps = execution_steps(r)?;
                // AncestralEuler::new builds a half-rounded host step vector, with one
                // F32->F16->F32 native round trip at EACH actual schedule node.
                for _ in 0..if r.route == Route::Mask { 2 } else { 1 } {
                    for _ in 0..plus(steps, 1)? {
                        self.scalar(4)?;
                        self.scalar(2)?;
                        self.scalar(4)?;
                    }
                }
                let mut state = initial;
                for _ in 0..steps {
                    self.preview_graph(n, state.width, false)?;
                    let eps = self.forward_request(c, r, native_dtype(state.width))?;
                    state.width =
                        self.ancestral_update(n, state.width, q::dtype_bytes(eps.dtype))?;
                    if r.route == Route::Mask {
                        state.width = self.mask_blend(n, state.width)?;
                    }
                }
                Ok(state)
            }
            _ => {
                let mut state = initial;
                for coeff in r.sampling.acceleration_coefficients(&alpha)? {
                    if coeff.c_in != 1.0 {
                        self.scalar(4)?;
                        self.op(state.width, n)?;
                    }
                    self.cast(n, state.width, 2)?;
                    self.preview_graph(n, 2, false)?;
                    let eps = self.forward_request(c, r, crate::model::DTYPE)?;
                    self.cast(n, state.width, 4)?;
                    self.cast(n, q::dtype_bytes(eps.dtype), 4)?;
                    if coeff.uses_unity_noiseless_update() {
                        self.scalar(4)?;
                        self.repeated(4, n, 2)?;
                    } else {
                        self.repeated(4, 1, 2)?;
                        self.repeated(4, n, 3)?;
                        if coeff.has_noise() {
                            self.random_normal(n)?;
                            self.scalar(4)?;
                            self.repeated(4, n, 2)?;
                        }
                    }
                    state.width = 4;
                }
                Ok(state)
            }
        }
    }
}

#[cfg(test)]
mod source_prefix_tests {
    use super::*;

    #[test]
    fn failed_symbolic_replay_restores_owned_ledger() {
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        ledger.array(4, 1).unwrap();
        let before = ledger.total;
        let result: Result<()> = ledger.replay_owned(|ops| {
            ops.ledger.borrow_mut().array(4, 1024)?;
            Err(crate::adapters::matmul_shape(&[1, 2], &[3, 1]).unwrap_err())
        });
        assert_eq!(ledger.total, before + backing_upper(4096, 16_384).unwrap());
        assert_eq!(
            retain_source_prefix(&ledger, result, false).unwrap(),
            ledger.total
        );
    }

    #[test]
    fn failing_alias_world_retains_its_cumulative_prefix() {
        // Two source alternatives: the second produces both operands before native
        // matmul validation rejects their incompatible inner axes. No native arrays.
        let mut successful = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        successful.array(4, 1).unwrap();
        let success = retain_source_prefix(&successful, Ok(()), true).unwrap();
        let mut failing = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        failing.array(4, 2).unwrap();
        failing.array(4, 3).unwrap();
        let source_error = crate::adapters::matmul_shape(&[1, 2], &[3, 1]).unwrap_err();
        assert!(q::is_native_shape_failure(&source_error));
        let prefix = failing.total;
        let failed = retain_source_prefix(&failing, Err(source_error), false).unwrap();
        assert_eq!(failed, prefix);
        assert!(success.max(failed) >= prefix);
        assert!(retain_source_prefix(&failing, Err(overflow()), false).is_err());
        assert!(retain_source_prefix(
            &failing,
            Err(Error::Unsupported("environment/invariant".into())),
            false
        )
        .is_err());
        let source_error = crate::adapters::matmul_shape(&[1, 2], &[3, 1]).unwrap_err();
        assert!(retain_source_prefix(&failing, Err(source_error), true).is_err());
    }
}
