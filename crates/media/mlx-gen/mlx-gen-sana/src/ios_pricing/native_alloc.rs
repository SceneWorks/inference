//! Cumulative MLX allocator operand law for SANA's closed source graph.
//!
//! Source authority is the same pinned MLX allocator/lowerings as SDXL's
//! `ios_pricing/native_ledger.rs`: each request is rounded separately, with the
//! buffer-cache reuse upper. These are native backing bytes, NOT process footprint.
//! CPU and Metal matrix/convolution/reduction workspaces are distinct.
//!
//! No allocation is retired for eval, synchronization, owner drop, donation or
//! command completion. Recipes must enumerate the whole reached source prefix,
//! including constructor, copied-layout and conditional lowering allocations.
//! This arithmetic alone certifies no request: `native_graph` owns source coverage.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeBackend {
    Cpu,
    Metal,
}

fn overflow() -> Error {
    Error::Unsupported("SANA cumulative native allocation arithmetic overflow".into())
}
fn invalid(what: &str) -> Error {
    Error::Unsupported(format!("SANA native ledger: {what}"))
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocator_rounds_every_operand_before_sum() {
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        ledger.repeated(1, 16_385, 2).unwrap();
        assert_eq!(ledger.total, 131_072);
        assert_eq!(backing_upper(32_770, 16_384).unwrap(), 81_920);
        assert_ne!(ledger.total, backing_upper(32_770, 16_384).unwrap());
    }
    #[test]
    fn cache_reuse_upper_keeps_small_and_page_sized_requests_distinct() {
        for (request, expected) in [
            (0, 0),
            (1, 2),
            (4, 8),
            (16_384, 32_768),
            (16_385, 65_536),
            (32_769, 81_920),
        ] {
            assert_eq!(backing_upper(request, 16_384).unwrap(), expected);
        }
    }
    #[test]
    fn checked_failures_do_not_wrap_or_commit_a_partial_operand() {
        assert!(NativeLedger::new(3, NativeBackend::Metal).is_err());
        assert!(backing_upper(u64::MAX, 16_384).is_err());
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        assert!(ledger.array(4, u64::MAX).is_err());
        assert_eq!(ledger.total, 0);
        assert!(ledger.repeated(4, 1, u64::MAX).is_err());
        assert_eq!(ledger.total, 0);
        ledger.total = u64::MAX;
        assert!(ledger.scalar(4).is_err());
        assert_eq!(ledger.total, u64::MAX);
    }
    #[test]
    fn direct_convolution_gemm_keeps_vector_split_k_workspace() {
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Metal).unwrap();
        ledger.metal_mm_workspace(1, 1, 256).unwrap();
        assert_eq!(ledger.total, 0);
        ledger.metal_steel_workspace(1, 1, 256).unwrap();
        assert_eq!(ledger.total, 144);
    }
    #[test]
    fn cpu_and_metal_convolution_have_distinct_source_workspaces() {
        let mut cpu = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        let mut metal = NativeLedger::new(16_384, NativeBackend::Metal).unwrap();
        let shape = [4, 1, 4, 4, 3, 8, 3, 3, 4, 4];
        cpu.conv2d(shape, [1, 1], [1, 1], [1, 1], 1).unwrap();
        metal.conv2d(shape, [1, 1], [1, 1], [1, 1], 1).unwrap();
        assert!(
            cpu.total > metal.total,
            "CPU im2col is not a Metal implicit convolution workspace"
        );
        assert!(cpu.conv2d(shape, [0, 1], [1, 1], [1, 1], 1).is_err());
    }
}
