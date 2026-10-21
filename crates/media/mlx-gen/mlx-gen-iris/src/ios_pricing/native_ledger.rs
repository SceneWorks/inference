//! Cumulative MLX native-allocator Active backing recipe for Iris generation and depth.
//!
//! Source authority: inference canonical2acdde0dc; MLX immutable source ITSNSQ; mlx-rs fe22998.
//! This is NOT process/RSS/physical-memory pricing. Every possible allocation is retained in this
//! ledger until the request ends. Eval, barriers, scopes, donation, cache eviction and Rust drops
//! grant NO retirement credit. Views grant no allocations, but a copy-producing consumer does.
//! Each operand request is charged independently through U, before multiplication by call count.
//!
//! Native source citation table (paths relative to the immutable MLX `mlx/` directory):
//! CPU: backend/cpu/matmul.cpp:69-162 (one exact-shape copy per input, output),
//! quantized.cpp:924-970,1233-1285 (qmm/quantize copies and outputs), softmax.cpp:117-145,
//! indexing.cpp:177-194, reduce.cpp:452; fast.cpp:91-121,880-954 (norm/SDPA fallbacks).
//! Metal: backend/metal/matmul.cpp:25-40,521-543,680-703,911-949,1248-1318,1359-1420;
//! quantized.cpp:53-125,891-923,1501-1585,1801-1827; scaled_dot_product_attention.cpp:
//! 440-504,658-786; normalization.cpp:25-51,84-92; softmax.cpp:24-44;
//! indexing.cpp:57-89,440-454; reduce.cpp:328-361,640-655,818-826,931-1014.
//! Both: array.cpp:176-216 and backend/common/common.cpp:190-304 (view aliasing),
//! backend/common/load.cpp:30-61; backend/metal/allocator.cpp:107-231 and
//! backend/common/buffer_cache.h:30-46 (request rounding, bounded reuse, Active counter).
//!
//! Conditional layout copies are conservatively included even for a contiguous request. Mutually
//! exclusive hardware-dependent workspace branches are each charged separately when the request
//! does not identify a GPU architecture; their source predicates still restrict applicable shapes.
//! Threadgroup, BLAS, host Vec and command-encoder metadata are not MLX allocator backing.

use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeBackend {
    Cpu,
    Metal,
}

fn overflow() -> Error {
    Error::Unsupported("Iris cumulative native allocation arithmetic overflow".into())
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
    (n / d)
        .checked_add(u64::from(!n.is_multiple_of(d)))
        .ok_or_else(overflow)
}
macro_rules! p {
    ($($d:expr),+ $(,)?) => { product(&[$(($d) as u64),+])? };
}

/// The allocator's exact request/reuse upper, NOT a universal page round.
fn backing_upper(n: u64, page: u64) -> Result<u64> {
    if !page.is_power_of_two() {
        return Err(Error::Unsupported("invalid Metal VM page".into()));
    }
    if n == 0 {
        return Ok(0);
    }
    let r = if n <= page {
        n
    } else {
        product(&[ceil(n, page)?, page])?
    };
    // Evaluate min without overflowing the branch that is not the minimum.
    let slack = product(&[2, page])?;
    let extra = r.min(slack);
    plus(r, extra)
}

/// A quantized projection's packed codes, scale count and affine group width.
#[derive(Clone, Copy)]
struct QmmStorage {
    codes: u64,
    scales: u64,
    group: u64,
}

/// One scaled-dot-product attention call's batch, head counts, query/key rows and head width.
#[derive(Clone, Copy)]
struct SdpaShape {
    b: u64,
    heads: u64,
    kv_heads: u64,
    q: u64,
    k: u64,
    dim: u64,
}

/// An attention block's query and key/value head counts and per-head width.
#[derive(Clone, Copy)]
struct AttentionHeads {
    heads: u64,
    kv_heads: u64,
    dim: u64,
}

/// An attention block's element widths: residual stream, query, key and value.
#[derive(Clone, Copy)]
struct AttentionWidths {
    width: u64,
    qwidth: u64,
    kwidth: u64,
    vwidth: u64,
}

/// A self-attention block's model width, head count and input element width.
#[derive(Clone, Copy)]
struct SelfAttentionShape {
    dim: u64,
    heads: u64,
    input_width: u64,
}

/// A self-attention block's QK-norm, rotary and mask options.
#[derive(Clone, Copy)]
struct SelfAttentionOptions {
    qk: bool,
    rope: bool,
    mask: bool,
}

/// A text block's head count and residual-stream element width.
#[derive(Clone, Copy)]
struct TextBlockWidth {
    heads: u64,
    stream_width: u64,
}

/// A backbone forward's row-major order, first-call and streamed-residency flags.
#[derive(Clone, Copy)]
struct ForwardMode {
    by_row: bool,
    first: bool,
    streamed: bool,
}

struct NativeLedger {
    page: u64,
    backend: NativeBackend,
    total: u64,
}
impl NativeLedger {
    fn new(page: u64, backend: NativeBackend) -> Result<Self> {
        backing_upper(0, page)?;
        Ok(Self {
            page,
            backend,
            total: 0,
        })
    }
    fn bytes(&mut self, bytes: u64) -> Result<()> {
        self.total = plus(self.total, backing_upper(bytes, self.page)?)?;
        Ok(())
    }
    fn array(&mut self, width: u64, elements: u64) -> Result<()> {
        self.bytes(product(&[width, elements])?)
    }
    fn repeated(&mut self, width: u64, elements: u64, count: u64) -> Result<()> {
        let one = backing_upper(product(&[width, elements])?, self.page)?;
        self.total = plus(self.total, product(&[one, count])?)?;
        Ok(())
    }
    fn scalar(&mut self, width: u64) -> Result<()> {
        self.array(width, 1)
    }
    fn cast(&mut self, n: u64, from: u64, to: u64) -> Result<()> {
        if from != to {
            self.array(to, n)?;
        }
        Ok(())
    }
    /// Unary/binary node's non-donated output. Broadcasts/views themselves allocate nothing.
    fn op(&mut self, width: u64, n: u64) -> Result<()> {
        self.array(width, n)
    }
    /// A reshape can be a view or one logical-shape contiguous copy (common/common.cpp).
    fn reshape_copy(&mut self, width: u64, n: u64) -> Result<()> {
        self.array(width, n)
    }
    fn concat(&mut self, width: u64, n: u64, parts: u64) -> Result<()> {
        // ops.cpp concatenate returns its sole input verbatim.
        if parts > 1 {
            self.array(width, n)?;
        }
        Ok(())
    }
    /// Dense Matmul/AddMM. Linear's 2-D weight folds ALL input rows into M (ops.cpp:3363-3366).
    /// For an attention matmul, batches and broadcast-expanded operand requests are explicit.
    fn matmul(&mut self, width: u64, batch: u64, m: u64, n: u64, k: u64) -> Result<()> {
        self.array(width, p!(batch, m, n))?;
        self.array(width, p!(batch, m, k))?; // conditional A exact-shape copy
        self.array(width, p!(batch, k, n))?; // conditional B exact-shape copy
        if self.backend == NativeBackend::Metal {
            if batch == 1 {
                self.metal_mm_workspace(m, n, k)?;
            } else {
                // matmul.cpp:1278-1287 (AddMM:1408-1419) may collapse broadcast batch into M
                // before dispatch. Layout is conditional; include the possible collapsed branch
                // separately. The ordinary multi-batch branch has no Split-K allocation.
                self.metal_mm_workspace(p!(batch, m), n, k)?;
            }
        }
        Ok(())
    }
    fn metal_mm_workspace(&mut self, m: u64, n: u64, k: u64) -> Result<()> {
        if m > 1 && n > 1 {
            // SIMD Split-K: hardware thresholds are <=2048; use largest applicable threshold.
            let tiles16 = p!(ceil(m, 16)?, ceil(n, 16)?);
            if tiles16 <= 2048 && k / 16 >= 8 && k >= m.max(n) {
                let tiles32 = p!(ceil(m, 32)?, ceil(n, 32)?);
                let split = (k / 16 / tiles32)
                    .checked_next_power_of_two()
                    .ok_or_else(overflow)?
                    .clamp(2, 32);
                self.array(4, p!(split, m, n))?;
            }
            // NAX Split-K is a distinct possible architecture branch, not the CPU lowering.
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
        }
        Ok(())
    }
    fn qmm(
        &mut self,
        width: u64,
        rows: u64,
        input: u64,
        out: u64,
        storage: QmmStorage,
    ) -> Result<()> {
        let QmmStorage {
            codes,
            scales,
            group,
        } = storage;
        self.array(width, p!(rows, out))?;
        // CPU ensure_row_contiguous / Metal copy_unless_contiguous: at most one per operand.
        self.array(width, p!(rows, input))?;
        self.array(4, codes)?;
        self.repeated(width, scales, 2)?;
        if self.backend == NativeBackend::Metal && rows >= 6 {
            // All transposed-Iris architecture vector thresholds are >=6. For 6..31 the QMM
            // path is conditional on architecture; charging its actual split law is conservative.
            let denom = p!(ceil(out, 32)?, ceil(rows, 32)?);
            let mut parts = (512 / denom).max(1).min(input / group);
            while parts > 1 && !input.is_multiple_of(p!(parts, group)) {
                parts -= 1;
            }
            if parts > 1 {
                self.array(width, p!(parts, rows, out))?;
            }
        }
        // Iris always invokes transpose=true: nontransposed QVM/QQMatmul cannot be selected.
        Ok(())
    }
    fn softmax(&mut self, width: u64, n: u64) -> Result<()> {
        // Contiguous non-donated result OR conditional copy shared by result, never both.
        self.array(width, n)
    }
    /// Reduction output plus every shape-compatible conditional Metal source workspace branch.
    /// Row-contiguous last-axis calls can set `last_axis=true`; no layout repair then exists.
    fn reduce(
        &mut self,
        width: u64,
        input: u64,
        output: u64,
        extent: u64,
        last_axis: bool,
    ) -> Result<()> {
        if self.backend == NativeBackend::Cpu {
            return self.array(width, output);
        }
        self.bytes(p!(width, output).max(4))?;
        // A rank-one last-axis reduction is also an all-reduce. Its two-pass scratch
        // must not disappear merely because layout repair is impossible.
        if output == 1 && input > 4096 {
            let rows = if p!(width, input) <= (1 << 26) {
                128
            } else {
                4096
            };
            self.array(width, rows)?;
        }
        if !last_axis {
            self.array(width, input)?; // GeneralReduce contiguous repair
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
    /// fast.cpp:91-121. Iris gains are already f32 (Loader::norm -> f32, charged at load);
    /// decoder gains/input/output are all bf16. Thus fast.cpp's learned-gain astype is identity.
    /// This primitive is source-specific, NOT a generic mixed-gain RMSNorm bound.
    fn norm(&mut self, n: u64, rows: u64, input_width: u64, output_width: u64) -> Result<()> {
        if self.backend == NativeBackend::Metal {
            self.cast(n, input_width, output_width)?;
            return self.array(output_width, n); // fused output OR contiguous-copy backing
        }
        self.cast(n, input_width, 4)?;
        self.op(4, n)?; // square
        self.reduce(4, n, rows, n.checked_div(rows).ok_or_else(overflow)?, true)?; // sum (mean's first node)
        self.scalar(4)?; // mean normalizer, ops.cpp:2201-2218
        self.op(4, rows)?; // multiply by normalizer
        self.scalar(4)?; // epsilon
        self.op(4, rows)?; // epsilon add
        self.op(4, rows)?; // rsqrt
        self.op(4, n)?; // normalized product
        self.cast(n, 4, output_width)?;
        self.op(output_width, n) // learned gain product; NEVER donation credit
    }
    fn silu(&mut self, width: u64, n: u64) -> Result<()> {
        // mlx-rs nn/activation.rs:876-880, pre-fusion graph envelope.
        self.op(width, n)?; // sigmoid
        self.op(width, n) // product
    }
    fn gelu(&mut self, input_width: u64, n: u64) -> Result<()> {
        // mlx-rs nn/activation.rs:890-898: x * (1 + erf(x/sqrt(2))) / 2.
        self.scalar(4)?; // sqrt(2) float32
        self.cast(n, input_width, 4)?;
        self.op(4, n)?; // divide
        self.op(4, n)?; // erf
        self.scalar(4)?; // int32 one
        self.scalar(4)?; // int32->float32 one
        self.op(4, n)?; // add one
        self.cast(n, input_width, 4)?; // other original-x edge (no common-cast credit)
        self.op(4, n)?; // multiply
        self.scalar(4)?; // float32 two
        self.op(4, n)
    }
    fn modulate(&mut self, n: u64, modulation: u64, xw: u64, mw: u64) -> Result<u64> {
        // nn.rs:493-496. The one scalar is first f32 then cast to scale dtype.
        self.scalar(4)?;
        self.cast(1, 4, mw)?;
        self.op(mw, modulation)?; // 1 + scale
        let outw = xw.max(mw);
        self.cast(n, xw, outw)?;
        self.cast(modulation, mw, outw)?;
        self.op(outw, n)?; // scale product
        self.op(outw, n)?; // shift add
        Ok(outw)
    }
    fn rope(&mut self, n: u64, width: u64, neox: bool) -> Result<()> {
        if neox {
            // mlx_llm/primitives/rope.rs:416-455: -half, cat, two products, sum.
            self.op(width, n / 2)?;
            self.concat(width, n, 2)?;
            self.repeated(width, n, 3)?;
        } else {
            // Iris nn.rs:551-563: explicit adjacent-pair graph even on Metal, NOT fast::RoPE.
            self.cast(n, width, 4)?;
            self.repeated(4, n / 2, 6)?; // four products plus difference and sum
            self.concat(4, n, 2)?; // stack's concat
            self.cast(n, 4, width)?;
        }
        Ok(())
    }
    fn rope_tables(&mut self, positions: u64, dim: u64) -> Result<()> {
        self.repeated(4, p!(positions, dim / 2), 2)
    }
    /// Concrete fast::SDPA call, before the consumer's BHSD->BSHD reshape.
    fn sdpa(&mut self, width: u64, shape: SdpaShape, mask: bool, causal: bool) -> Result<()> {
        let SdpaShape {
            b,
            heads,
            kv_heads,
            q,
            k,
            dim,
        } = shape;
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || dim == 0 {
            return Err(Error::Unsupported(
                "Iris native SDPA has an invalid head geometry".into(),
            ));
        }
        let qn = p!(b, heads, q, dim);
        let kn = p!(b, kv_heads, k, dim);
        let scores = p!(b, heads, q, k);
        // Metal support is shape-specific (scaled_dot_product_attention.cpp:591-641).
        // In particular head80 with <=8 query rows and tiny-fixture head widths use the shared
        // unfused graph, but its primitive allocations still use the METAL backend below.
        let fused = self.backend == NativeBackend::Metal
            && ((q > 8 && matches!(dim, 64 | 80 | 128))
                || (q <= 8
                    && q <= k
                    && matches!(dim, 64 | 96 | 128 | 256)
                    && p!(q, heads / kv_heads) <= 32));
        if fused {
            // Each conditional copy has its original input shape, not a GQA expansion.
            self.array(width, qn)?;
            self.repeated(width, kn, 2)?;
            if mask {
                self.array(width, p!(b, heads, q, k))?;
            } // native wrapper broadcasts first
            self.array(width, qn)?; // fused result, no donation credit
            if q <= 8 && k >= 1024 {
                // Architecture-specific branches can select vector two-pass. Largest actual
                // default is 1024; environment override is a source input, not a magic reserve.
                // metal/scaled_dot_product_attention.cpp:440-479: take the maximum of
                // the s/d/other architecture branches, evaluating each sequence-length predicate.
                let simds = p!(heads / kv_heads, q);
                let s_blocks = if k > 1024 && simds > 4 {
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
                self.repeated(4, partial, 2)?; // sums, maxs distinct operands
            }
            return Ok(());
        }
        // fast.cpp:880-954 shared fallback graph. Each Matmul/softmax uses this ledger's
        // actual backend; CPU layout laws are NEVER substituted for Metal primitive laws.
        self.scalar(width)?;
        self.op(width, qn)?;
        // GQA broadcasting occurs before matmul_general; exact expanded operands have Hq heads.
        self.matmul(width, p!(b, heads), q, k, dim)?;
        if causal {
            self.repeated(4, k, 1)?; // k_idx
            self.repeated(4, q, 1)?; // q_idx
            self.op(1, p!(q, k))?; // comparison
            self.scalar(width)?; // masked min scalar
            self.op(width, scores)?; // where
        } else if mask {
            self.op(width, scores)?;
        }
        self.softmax(width, scores)?;
        self.matmul(width, p!(b, heads), q, dim, k)
    }
    /// Iris nn::attention_budgeted. All chunks count, including evaluated chunks and tail.
    fn attention(
        &mut self,
        b: u64,
        n: u64,
        heads: AttentionHeads,
        widths: AttentionWidths,
        mask: bool,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let AttentionHeads {
            heads,
            kv_heads,
            dim,
        } = heads;
        let AttentionWidths {
            width,
            qwidth,
            kwidth,
            vwidth,
        } = widths;
        self.cast(p!(b, n, heads, dim), qwidth, width)?;
        self.cast(p!(b, n, kv_heads, dim), kwidth, width)?;
        self.cast(p!(b, n, kv_heads, dim), vwidth, width)?;
        if mask {
            self.cast(p!(b, n, n), 4, width)?;
        }
        let bi = i32::try_from(b).map_err(|_| overflow())?;
        let hi = i32::try_from(heads).map_err(|_| overflow())?;
        let ni = i32::try_from(n).map_err(|_| overflow())?;
        let block =
            u64::try_from(plan.budget.query_block(bi, hi, ni, ni)).map_err(|_| overflow())?;
        if block == 0 {
            return Err(overflow());
        }
        let mut start = 0;
        let mut calls = 0;
        while start < n {
            let count = block.min(n - start);
            self.sdpa(
                width,
                SdpaShape {
                    b,
                    heads,
                    kv_heads,
                    q: count,
                    k: n,
                    dim,
                },
                mask,
                false,
            )?;
            start = plus(start, count)?;
            calls = plus(calls, 1)?;
        }
        self.concat(width, p!(b, heads, n, dim), calls)?;
        self.reshape_copy(width, p!(b, heads, n, dim)) // possible merge re-layout
    }
}

/// Header-driven backbone graph. No Retention scratch prices, maxima, scalar allowances or held
/// ledgers are used here. Methods mirror the function bodies in nn.rs and dit.rs.
struct Graph<'a> {
    l: &'a mut NativeLedger,
    bb: &'a Backbone,
}
impl Graph<'_> {
    fn linear(&mut self, path: &str, rows: u64, input: u64, from: u64) -> Result<u64> {
        let cw = self.bb.cw;
        let out = self.bb.rows(path)?;
        self.l.cast(p!(rows, input), from, cw)?;
        if let Some(scales) = self.bb.headers.get(&format!("{path}.scales")) {
            let padded = p!(scales.shape[1], QUANT_GROUP_SIZE);
            if input > padded {
                return Err(Error::Unsupported(
                    "Iris packed input exceeds stored pack".into(),
                ));
            }
            if padded != input {
                self.l.scalar(4)?; // mlx-rs ops/shapes.rs:783-785: Array::from_int(0)
                self.l.scalar(cw)?; // int32 -> compute-dtype fill scalar
                self.l.array(cw, p!(rows, padded))?;
            }
            let codes = header_elements(self.bb.get(&format!("{path}.weight"))?)?;
            self.l.qmm(
                cw,
                rows,
                padded,
                out,
                QmmStorage {
                    codes,
                    scales: header_elements(scales)?,
                    group: QUANT_GROUP_SIZE as u64,
                },
            )?;
            if self.bb.headers.contains_key(&format!("{path}.bias")) {
                // Portable separate epilogue is an upper for Metal's fused output-bias variant.
                self.l.array(cw, out)?; // conditional output-bias layout copy
                self.l.op(cw, p!(rows, out))?;
            }
        } else {
            if input != self.bb.input(path)? {
                return Err(Error::Unsupported(format!(
                    "Iris projection {path}: source/input width mismatch"
                )));
            }
            self.l.matmul(cw, 1, rows, out, input)?;
        }
        if let Some(adapters) = self.bb.adapter_facts.get(path) {
            let compute = if cw == 4 {
                mlx_rs::Dtype::Float32
            } else {
                mlx_rs::Dtype::Bfloat16
            };
            installed_adapters::forward(self.l, adapters, rows, input, out, compute)?;
        }
        Ok(out)
    }
    fn norm(&mut self, path: &str, rows: u64, input_width: u64) -> Result<u64> {
        let dim = self.bb.norm(path)?;
        self.l.norm(p!(rows, dim), rows, input_width, 4)?;
        Ok(dim)
    }
    fn swiglu(&mut self, path: &str, rows: u64, input: u64, width: u64) -> Result<()> {
        let ff = self.linear(&format!("{path}.w1"), rows, input, width)?;
        self.l.silu(self.bb.cw, p!(rows, ff))?;
        self.linear(&format!("{path}.w3"), rows, input, width)?;
        self.l.op(self.bb.cw, p!(rows, ff))?;
        self.linear(&format!("{path}.w2"), rows, ff, self.bb.cw)?;
        Ok(())
    }
    fn qkv(
        &mut self,
        path: &str,
        side: &str,
        rows: u64,
        dim: u64,
        from: u64,
        split: bool,
    ) -> Result<()> {
        if split {
            for role in ["q_proj", "k_proj", "v_proj"] {
                self.linear(&format!("{path}{role}{side}"), rows, dim, from)?;
            }
        } else {
            self.linear(&format!("{path}qkv{side}"), rows, dim, from)?;
        }
        Ok(())
    }
    fn self_attention(
        &mut self,
        path: &str,
        b: u64,
        n: u64,
        shape: SelfAttentionShape,
        options: SelfAttentionOptions,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let SelfAttentionShape {
            dim,
            heads,
            input_width,
        } = shape;
        let SelfAttentionOptions { qk, rope, mask } = options;
        let rows = p!(b, n);
        self.linear(&format!("{path}.qkv"), rows, dim, input_width)?;
        if qk {
            self.norm(&format!("{path}.q_norm"), p!(rows, heads), self.bb.cw)?;
            self.norm(&format!("{path}.k_norm"), p!(rows, heads), self.bb.cw)?;
        }
        let qw = if qk { 4 } else { self.bb.cw };
        if rope {
            self.l.rope(p!(rows, dim), qw, false)?;
            self.l.rope(p!(rows, dim), qw, false)?;
        }
        self.l.attention(
            b,
            n,
            AttentionHeads {
                heads,
                kv_heads: heads,
                dim: dim / heads,
            },
            AttentionWidths {
                width: self.bb.cw,
                qwidth: qw,
                kwidth: qw,
                vwidth: self.bb.cw,
            },
            mask,
            plan,
        )?;
        self.linear(&format!("{path}.proj"), rows, dim, self.bb.cw)?;
        Ok(())
    }
    fn text_block(
        &mut self,
        path: &str,
        b: u64,
        n: u64,
        width: TextBlockWidth,
        refiner: bool,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let TextBlockWidth {
            heads,
            stream_width,
        } = width;
        let rows = p!(b, n);
        let dim = self.norm(&format!("{path}.norm1"), rows, stream_width)?;
        self.self_attention(
            &format!("{path}.attn"),
            b,
            n,
            SelfAttentionShape {
                dim,
                heads,
                input_width: 4,
            },
            SelfAttentionOptions {
                qk: refiner && self.bb.cfg.qk_norm,
                rope: false,
                mask: refiner,
            },
            plan,
        )?;
        // Residual promotion (layer stack f32; refiner compute-width).
        let residual = stream_width.max(self.bb.cw);
        self.l.cast(p!(rows, dim), self.bb.cw, residual)?;
        self.l.op(residual, p!(rows, dim))?;
        self.norm(&format!("{path}.norm2"), rows, residual)?;
        if refiner {
            self.swiglu(&format!("{path}.mlp"), rows, dim, 4)?;
        } else {
            let ff = self.linear(&format!("{path}.mlp.0"), rows, dim, 4)?;
            self.l.silu(self.bb.cw, p!(rows, ff))?;
            self.linear(&format!("{path}.mlp.2"), rows, ff, self.bb.cw)?;
        }
        self.l.cast(p!(rows, dim), self.bb.cw, residual)?;
        self.l.op(residual, p!(rows, dim))
    }
    fn trunk(
        &mut self,
        i: usize,
        b: u64,
        n: u64,
        t: u64,
        image_width: u64,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let m = self.bb.cfg.clone();
        let d = m.hidden_size as u64;
        let heads = m.num_heads as u64;
        let kvheads = m.num_kv_heads.unwrap_or(m.num_heads) as u64;
        let kvw = p!(d / heads, kvheads);
        let split = heads != kvheads;
        let text_out = m.final_block_text == "keep" || i + 1 < m.depth;
        let path = format!("blocks.{i}");
        if i < m.dual_depth {
            // dit.rs::dual: core+bias, norms/modulations, QKV, QK norms, RoPE, joint concat.
            self.l.repeated(4, p!(b, 6, d), 2)?;
            self.l.cast(p!(b, 6, d), self.bb.cw, 4)?; // image core -> f32 bias sum
            self.l.cast(p!(b, 6, d), self.bb.cw, 4)?; // text core -> f32 bias sum
            for (side, tokens, width) in [("x", n, image_width), ("y", t, 4)] {
                let rows = p!(b, tokens);
                self.norm(&format!("{path}.norm_{side}1"), rows, width)?;
                self.l.modulate(p!(rows, d), p!(b, d), 4, 4)?;
                self.qkv(
                    &format!("{path}.attn."),
                    &format!("_{side}"),
                    rows,
                    d,
                    4,
                    split,
                )?;
                self.norm(
                    &format!("{path}.attn.q_norm_{side}"),
                    p!(rows, heads),
                    self.bb.cw,
                )?;
                self.norm(
                    &format!("{path}.attn.k_norm_{side}"),
                    p!(rows, kvheads),
                    self.bb.cw,
                )?;
                self.l.rope(p!(rows, d), 4, false)?;
                self.l.rope(p!(rows, kvw), 4, false)?;
            }
            let nt = plus(n, t)?;
            self.l.concat(4, p!(b, nt, d), 2)?;
            self.l.concat(4, p!(b, nt, kvw), 2)?;
            self.l.concat(self.bb.cw, p!(b, nt, kvw), 2)?;
            self.l.attention(
                b,
                nt,
                AttentionHeads {
                    heads,
                    kv_heads: kvheads,
                    dim: d / heads,
                },
                AttentionWidths {
                    width: self.bb.cw,
                    qwidth: 4,
                    kwidth: 4,
                    vwidth: self.bb.cw,
                },
                false,
                plan,
            )?;
            for (side, tokens, width) in [("x", n, image_width), ("y", t, 4)] {
                if side == "y" && !text_out {
                    continue;
                }
                let rows = p!(b, tokens);
                self.linear(&format!("{path}.attn_gate_{side}"), rows, d, 4)?;
                self.l.op(self.bb.cw, p!(rows, d))?; // sigmoid gate
                self.l.op(self.bb.cw, p!(rows, d))?; // attention*gate
                self.linear(&format!("{path}.attn.proj_{side}"), rows, d, self.bb.cw)?;
                self.norm(&format!("{path}.attn_post_norm_{side}"), rows, self.bb.cw)?;
                self.l.op(4, p!(rows, d))?; // modulation gate product
                self.l.cast(p!(rows, d), width, 4)?;
                self.l.op(4, p!(rows, d))?; // residual
                self.norm(&format!("{path}.norm_{side}2"), rows, 4)?;
                self.l.modulate(p!(rows, d), p!(b, d), 4, 4)?;
                self.swiglu(&format!("{path}.mlp_{side}"), rows, d, 4)?;
                self.norm(&format!("{path}.mlp_post_norm_{side}"), rows, self.bb.cw)?;
                self.l.op(4, p!(rows, d))?;
                self.l.op(4, p!(rows, d))?;
            }
        } else {
            let nt = plus(n, t)?;
            let rows = p!(b, nt);
            self.l.cast(p!(b, 6, d), self.bb.cw, 4)?;
            self.l.op(4, p!(b, 6, d))?; // modulation core+bias
            self.l.cast(p!(b, n, d), image_width, 4)?;
            self.l.concat(4, p!(rows, d), 2)?;
            self.norm(&format!("{path}.norm1"), rows, 4)?;
            self.l.modulate(p!(rows, d), p!(b, d), 4, 4)?;
            self.qkv(&format!("{path}."), "", rows, d, 4, split)?;
            self.norm(&format!("{path}.q_norm"), p!(rows, heads), self.bb.cw)?;
            self.norm(&format!("{path}.k_norm"), p!(rows, kvheads), self.bb.cw)?;
            for tokens in [t, n] {
                self.l.rope(p!(b, tokens, d), 4, false)?;
                self.l.rope(p!(b, tokens, kvw), 4, false)?;
            }
            self.l.concat(4, p!(rows, d), 2)?;
            self.l.concat(4, p!(rows, kvw), 2)?;
            self.l.attention(
                b,
                nt,
                AttentionHeads {
                    heads,
                    kv_heads: kvheads,
                    dim: d / heads,
                },
                AttentionWidths {
                    width: self.bb.cw,
                    qwidth: 4,
                    kwidth: 4,
                    vwidth: self.bb.cw,
                },
                false,
                plan,
            )?;
            let outrows = p!(b, if text_out { nt } else { n });
            self.linear(&format!("{path}.attn_gate"), outrows, d, 4)?;
            self.l.op(self.bb.cw, p!(outrows, d))?;
            self.l.op(self.bb.cw, p!(outrows, d))?;
            self.linear(&format!("{path}.attn_proj"), outrows, d, self.bb.cw)?;
            self.norm(&format!("{path}.attn_post_norm"), outrows, self.bb.cw)?;
            self.l.op(4, p!(outrows, d))?;
            self.l
                .cast(p!(outrows, d), if text_out { 4 } else { image_width }, 4)?;
            self.l.op(4, p!(outrows, d))?;
            self.norm(&format!("{path}.norm2"), outrows, 4)?;
            self.l.modulate(p!(outrows, d), p!(b, d), 4, 4)?;
            self.swiglu(&format!("{path}.mlp"), outrows, d, 4)?;
            self.norm(&format!("{path}.mlp_post_norm"), outrows, self.bb.cw)?;
            self.l.op(4, p!(outrows, d))?;
            self.l.op(4, p!(outrows, d))?;
        }
        Ok(())
    }
    fn pixel(
        &mut self,
        i: usize,
        b: u64,
        n: u64,
        pp: u64,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let path = format!("pixel_blocks.{i}");
        let rows = p!(b, n);
        let pix = self.bb.norm(&format!("{path}.norm1"))?;
        let plane = p!(rows, pp, pix);
        self.linear(
            &format!("{path}.adaln"),
            rows,
            self.bb.input(&format!("{path}.adaln"))?,
            4,
        )?;
        self.norm(&format!("{path}.norm1"), p!(rows, pp), self.bb.cw)?;
        let compact = self.linear(&format!("{path}.compress"), rows, p!(pp, pix), 4)?;
        self.self_attention(
            &format!("{path}.attn"),
            b,
            n,
            SelfAttentionShape {
                dim: compact,
                heads: self.bb.cfg.pixel.num_heads as u64,
                input_width: self.bb.cw,
            },
            SelfAttentionOptions {
                qk: self.bb.cfg.qk_norm,
                rope: true,
                mask: false,
            },
            plan,
        )?;
        self.linear(&format!("{path}.expand"), rows, compact, self.bb.cw)?;
        self.l.modulate(plane, plane, self.bb.cw, self.bb.cw)?;
        self.l.op(self.bb.cw, plane)?;
        self.norm(&format!("{path}.norm2"), p!(rows, pp), self.bb.cw)?;
        let ff = self.linear(&format!("{path}.mlp.fc1"), p!(rows, pp), pix, 4)?;
        self.l.gelu(self.bb.cw, p!(rows, pp, ff))?;
        self.linear(&format!("{path}.mlp.fc2"), p!(rows, pp), ff, 4)?;
        self.l.modulate(plane, plane, self.bb.cw, self.bb.cw)?;
        self.l.op(self.bb.cw, plane)
    }
    fn output(&mut self, b: u64, n: u64, pp: u64) -> Result<()> {
        let rows = p!(b, n, pp);
        let pix = self.norm("final_layer.norm", rows, self.bb.cw)?;
        let out = self.linear("final_layer.linear", rows, pix, 4)?;
        self.l.reshape_copy(self.bb.cw, p!(rows, out))?;
        self.l.cast(p!(rows, out), self.bb.cw, 4)
    }
    fn forward(
        &mut self,
        b: u64,
        h: u64,
        w: u64,
        mode: ForwardMode,
        plan: mlx_gen::AttentionPlan<'_>,
    ) -> Result<()> {
        let ForwardMode {
            by_row,
            first,
            streamed,
        } = mode;
        let m = self.bb.cfg.clone();
        let patch = m.patch_size as u64;
        if patch == 0
            || !h.is_multiple_of(patch)
            || !w.is_multiple_of(patch)
            || m.num_heads == 0
            || m.pixel.num_heads == 0
        {
            return Err(Error::Unsupported(
                "Iris native graph is not on its actual patch/head grid".into(),
            ));
        }
        let n = p!(h / patch, w / patch);
        let pp = p!(patch, patch);
        let t = m.text_len as u64;
        let d = m.hidden_size as u64;
        let channels = self.bb.input_channels;
        let cw = self.bb.cw;
        // dit.rs:2745-2778 embed_body / timestep_embedding (nn.rs, every source output).
        self.l.reshape_copy(4, p!(b, h, w, channels))?;
        self.linear("s_embedder.proj", p!(b, n), p!(channels, pp), 4)?;
        let sinusoid = self.bb.input("t_embedder.mlp.0")?;
        self.l.array(4, sinusoid / 2)?; // uploaded frequencies
        self.l.repeated(4, p!(b, sinusoid / 2), 3)?; // phase, cosine, sine
        self.l.concat(4, p!(b, sinusoid), 2)?;
        let th = self.linear("t_embedder.mlp.0", b, sinusoid, 4)?;
        self.l.silu(cw, p!(b, th))?;
        let td = self.linear("t_embedder.mlp.2", b, th, cw)?;
        self.l.silu(cw, p!(b, td))?;
        self.linear("modulation_cores.adaln_img", b, td, cw)?;
        if m.dual_depth > 0 {
            self.linear("modulation_cores.adaln_txt", b, td, cw)?;
        }
        // Every streamed stack tensor is re-read once PER forward, not once per request/window.
        if streamed {
            load_backbone_headers(self.l, self.bb, true)?;
        }
        // text_adapter_exec: layerwise stack, pool, mask, refiner, final norm.
        for i in 0..Stack::Layer.depth(&m) {
            self.text_block(
                &format!("y_embedder.layer_blocks.{i}"),
                p!(b, t),
                self.bb.layers,
                TextBlockWidth {
                    heads: m.text_lap_num_heads as u64,
                    stream_width: 4,
                },
                false,
                plan,
            )?;
        }
        let dt = m.text_dim as u64;
        self.linear("y_embedder.layer_pool", p!(b, t, dt), self.bb.layers, 4)?;
        self.linear("y_embedder.refiner.proj", p!(b, t), dt, cw)?;
        self.l.array(4, p!(b, t, t))?; // key_padding_mask from_slice, host Vec excluded
        for i in 0..Stack::Refiner.depth(&m) {
            self.text_block(
                &format!("y_embedder.refiner.blocks.{i}"),
                b,
                t,
                TextBlockWidth {
                    heads: m.num_heads as u64,
                    stream_width: cw,
                },
                true,
                plan,
            )?;
        }
        self.norm("y_embedder.refiner.norm", p!(b, t), cw)?;
        self.l.op(4, p!(b, t, d))?; // text_positions
        self.l.rope_tables(n, d / m.num_heads as u64)?;
        self.l.rope_tables(t, d / m.num_heads as u64)?;
        for i in 0..m.depth {
            self.trunk(i, b, n, t, if i == 0 { cw } else { 4 }, plan)?;
        }
        // pixel_embed_body: timestep re-fusion, pixel projection/position add, patch re-layout.
        let sw = if m.depth == 0 { cw } else { 4 };
        let fusedw = sw.max(cw);
        self.l.cast(p!(b, td), cw, fusedw)?;
        self.l.op(fusedw, p!(b, n, d))?;
        self.l.silu(fusedw, p!(b, n, d))?;
        let pix = self.linear("pixel_embedder.proj", p!(b, h, w), channels, 4)?;
        if first {
            self.l.array(4, p!(h, w, pix))?;
        } // one table cached for this size
        self.l.cast(p!(h, w, pix), 4, cw)?; // per-call position-table cast
        self.l.op(cw, p!(b, h, w, pix))?;
        self.l.reshape_copy(cw, p!(b, h, w, pix))?;
        self.l
            .rope_tables(n, (m.pixel.attn_hidden_size / m.pixel.num_heads) as u64)?;
        let row_batch = if by_row && b > 1 { 1 } else { b };
        let iterations = if row_batch == 1 { b } else { 1 };
        for i in 0..m.pixel.depth {
            for _ in 0..iterations {
                self.pixel(i, row_batch, n, pp, plan)?;
            }
        }
        for _ in 0..iterations {
            self.output(row_batch, n, pp)?;
        }
        if iterations > 1 {
            self.l.concat(4, p!(b, m.in_channels, h, w), iterations)?;
        }
        Ok(())
    }
}

fn header_elements(h: &Header) -> Result<u64> {
    h.shape.iter().try_fold(1u64, |a, &d| {
        product(&[a, u64::try_from(d).map_err(|_| overflow())?])
    })
}
fn stored_width(dtype: Stored) -> Result<u64> {
    Ok(dtype.size() as u64)
}
fn raw_header(l: &mut NativeLedger, h: &Header) -> Result<()> {
    let expected = p!(header_elements(h)?, stored_width(h.dtype)?);
    if expected != h.data_bytes {
        return Err(Error::Unsupported(format!(
            "Iris header byte/shape mismatch: {}",
            h.name
        )));
    }
    l.bytes(h.data_bytes)
}
fn cast_header(l: &mut NativeLedger, h: &Header, target: mlx_rs::Dtype) -> Result<()> {
    let same = matches!(
        (h.dtype, target),
        (Stored::F32, mlx_rs::Dtype::Float32)
            | (Stored::BF16, mlx_rs::Dtype::Bfloat16)
            | (Stored::F16, mlx_rs::Dtype::Float16)
            | (Stored::U32, mlx_rs::Dtype::Uint32)
            | (Stored::I32, mlx_rs::Dtype::Int32)
    );
    if !same {
        l.array(
            mlx_llm::primitives::nn::dtype_bytes(target),
            header_elements(h)?,
        )?;
    }
    Ok(())
}

/// Constructor casts act on the post-read dtype, not the original header: a rank-two f32
/// adaLN bias is first read/cast to bf16 and then Loader::f32 widens it back to f32.
fn constructor_cast(
    l: &mut NativeLedger,
    h: &Header,
    matrix_cast: bool,
    compute: mlx_rs::Dtype,
    target: mlx_rs::Dtype,
) -> Result<()> {
    if matrix_cast {
        if compute != target {
            l.array(
                mlx_llm::primitives::nn::dtype_bytes(target),
                header_elements(h)?,
            )?;
        }
        Ok(())
    } else {
        cast_header(l, h, target)
    }
}

/// Read/cast/constructor graph of the actual selected header subset. `.weight` matrices are
/// projections; 1-D gains, y_pos_embedding and adaLN biases are f32. A packed marker is a raw
/// native Load read by `.item::<i32>()`, not a free host metadata value.
fn load_backbone_set(
    l: &mut NativeLedger,
    headers: &HashMap<String, Header>,
    cw: u64,
    prefix: &str,
    select: impl Fn(&str) -> bool,
    quantize_dense: Option<BackboneQuant>,
) -> Result<()> {
    let compute = if cw == 4 {
        mlx_rs::Dtype::Float32
    } else {
        mlx_rs::Dtype::Bfloat16
    };
    for (name, h) in headers {
        let name = name.strip_prefix(prefix).unwrap_or(name);
        if !select(name) {
            continue;
        }
        raw_header(l, h)?;
        let packed = crate::stream::is_packed_part_header(h);
        let matrix_cast = h.shape.len() >= 2 && name != "y_pos_embedding" && !packed;
        if matrix_cast {
            cast_header(l, h, compute)?;
        }
        if let Some(base) = name.strip_suffix(".weight") {
            if h.shape.len() == 1 {
                constructor_cast(l, h, matrix_cast, compute, mlx_rs::Dtype::Float32)?;
            } else if h.shape.len() == 2 && !packed {
                // Loader::linear's same-dtype cast aliases the preceding read cast.
                if let ProjectionLayout::Quantized { bits, padded_in } =
                    ProjectionLayout::of(h.shape[1], quantize_dense)
                {
                    if padded_in != h.shape[1] {
                        l.scalar(4)?; // pad's int32 zero
                        l.scalar(cw)?; // compute-dtype scalar cast
                        l.array(cw, p!(h.shape[0], padded_in))?;
                    }
                    l.cast(p!(h.shape[0], padded_in), cw, 2)?;
                    quantize_stored_grid(
                        l,
                        h.shape[0] as u64,
                        padded_in as u64,
                        QUANT_GROUP_SIZE as u64,
                        bits as u64,
                    )?;
                }
            }
            // `base` used only to distinguish the `.weight` suffix above.
            let _ = base;
        } else if let Some(base) = name.strip_suffix(".bias") {
            let sibling = format!("{prefix}{base}.weight");
            let linear = headers
                .get(&sibling)
                .or_else(|| headers.get(&format!("{base}.weight")))
                .is_some_and(|w| w.shape.len() == 2);
            if linear {
                let scales = format!("{prefix}{base}.scales");
                let packed_linear = headers.contains_key(&scales)
                    || headers.contains_key(&format!("{base}.scales"));
                constructor_cast(
                    l,
                    h,
                    matrix_cast,
                    compute,
                    if packed_linear {
                        mlx_rs::Dtype::Bfloat16
                    } else {
                        compute
                    },
                )?;
                if quantize_dense.is_some() {
                    l.cast(header_elements(h)?, cw, 2)?;
                }
            } else {
                constructor_cast(l, h, matrix_cast, compute, mlx_rs::Dtype::Float32)?;
            }
        } else if name == "y_pos_embedding" {
            cast_header(l, h, mlx_rs::Dtype::Float32)?;
        }
    }
    Ok(())
}
fn load_backbone_headers(l: &mut NativeLedger, bb: &Backbone, stacks: bool) -> Result<()> {
    load_backbone_set(
        l,
        &bb.headers,
        bb.cw,
        "",
        |name| Stack::of(name).is_some() == stacks,
        None,
    )
}

/// Native fast::Quantize (three sibling outputs + conditional input copy) followed by the exact
/// quantize_on_stored_grid graph in mlx_llm/primitives/quant.rs:127-175. All outputs count even
/// when the preliminary integer-code output is discarded or fusion would eliminate a node.
fn quantize_stored_grid(
    l: &mut NativeLedger,
    rows: u64,
    input: u64,
    group: u64,
    bits: u64,
) -> Result<()> {
    if !input.is_multiple_of(group) || !matches!(bits, 4 | 8) {
        return Err(overflow());
    }
    let n = p!(rows, input);
    let groups = n / group;
    let words = p!(n, bits) / 32;
    l.array(2, n)?; // conditional source contiguity copy for native Quantize
    l.array(4, words)?;
    l.repeated(2, groups, 2)?; // stored scales and biases
    l.array(4, n)?; // bf16 weight -> f32
    l.repeated(4, groups, 2)?; // scales, biases -> f32
    l.repeated(4, n, 5)?; // subtract, divide, round, max(0), min(top)
    l.repeated(4, 1, 2)?; // clip scalar endpoints
    l.array(4, n)?; // uint32 cast
    l.array(4, 32 / bits)?; // place from_slice
    l.op(4, n)?; // disjoint-field multiplication
    l.reduce(4, n, words, 32 / bits, true)?; // sum over word fields
    Ok(())
}

struct TowerFacts {
    cfg: mlx_llm::ModelConfig,
    /// Real stored headers, NOT a requantized resident-byte estimate.
    real: HashMap<String, Header>,
    built: HashMap<String, Header>,
    quant: Option<Quant>,
}
impl TowerFacts {
    fn read(dir: &Path, requested: Option<Quant>, streamed: bool) -> Result<Self> {
        let cfg = mlx_llm::ModelConfig::from_dir(dir).map_err(|e| Error::Msg(e.to_string()))?;
        if !cfg.architecture.is_qwen3_vl() {
            return Err(Error::Unsupported(
                "Iris native recipe requires the source-selected Qwen3-VL tower".into(),
            ));
        }
        let packed = cfg.quantization.is_some();
        let artifact = if packed {
            crate::packed::text_encoder_tier(dir)?
        } else {
            None
        };
        crate::packed::check_requested("the text encoder", artifact, requested)?;
        if streamed {
            crate::stream::refuse_stream_requantization(
                crate::model::MODEL_ID,
                "the text encoder",
                artifact,
                requested,
            )?;
        }
        let mut real = Vec::new();
        for file in crate::text_encoder::weight_shards(dir, !cfg.tie_word_embeddings)? {
            real.extend(crate::stream::headers(&file)?.into_iter().filter(|h| {
                crate::text_encoder::is_text_tower_tensor(&h.name, !cfg.tie_word_embeddings)
            }));
        }
        if streamed
            && real
                .iter()
                .any(|h| h.name == format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.scales"))
        {
            // text_encoder.rs:807-813: the streamed tower explicitly requires dense embedding.
            return Err(Error::Unsupported(
                "Iris streamed tower reads a dense token embedding, not a stored packed embedding"
                    .into(),
            ));
        }
        let built = if !packed {
            if let Some(q) = requested {
                emulate_packed_tower(&real, &cfg, q)?
            } else {
                real.clone()
            }
        } else {
            real.clone()
        };
        Ok(Self {
            cfg,
            real: real.into_iter().map(|h| (h.name.clone(), h)).collect(),
            built: built.into_iter().map(|h| (h.name.clone(), h)).collect(),
            quant: if packed { None } else { requested },
        })
    }
    fn load(&self, l: &mut NativeLedger, select: impl Fn(&str) -> bool) -> Result<()> {
        let spec = crate::text_encoder::text_encoder_quant_spec(&self.cfg, self.quant)?;
        for h in self.real.values().filter(|h| select(&h.name)) {
            raw_header(l, h)?;
            if matches!(h.dtype, Stored::F16 | Stored::F32 | Stored::BF16) {
                cast_header(l, h, mlx_rs::Dtype::Bfloat16)?;
            }
            let projection = h.shape.len() == 2
                && h.name.ends_with(".weight")
                && !h.name.ends_with("embed_tokens.weight");
            if projection {
                if let Some(spec) = spec {
                    quantize_stored_grid(
                        l,
                        h.shape[0] as u64,
                        h.shape[1] as u64,
                        spec.group_size as u64,
                        spec.bits as u64,
                    )?;
                }
            }
        }
        Ok(())
    }
    fn outside(&self, name: &str) -> bool {
        name == format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight")
            || name == format!("{LANGUAGE_TOWER_PREFIX}norm.weight")
    }
    fn initial_load(&self, l: &mut NativeLedger, streamed: bool) -> Result<()> {
        self.load(l, |name| !streamed || self.outside(name))?;
        l.array(4, self.cfg.head_dim as u64) // build_rope's device-resident frequency row
    }
    fn projection(
        &self,
        l: &mut NativeLedger,
        path: &str,
        rows: u64,
        input: u64,
        output: u64,
    ) -> Result<()> {
        let key = format!("{path}.weight");
        let weight = self.built.get(&key).ok_or(Error::MissingTensor(key))?;
        if let Some(scales) = self.built.get(&format!("{path}.scales")) {
            let group = input
                .checked_div(scales.shape[1] as u64)
                .filter(|&g| g > 0)
                .ok_or_else(overflow)?;
            l.qmm(
                2,
                rows,
                input,
                output,
                QmmStorage {
                    codes: header_elements(weight)?,
                    scales: header_elements(scales)?,
                    group,
                },
            )?;
        } else {
            l.matmul(2, 1, rows, output, input)?;
        }
        if self.built.contains_key(&format!("{path}.bias")) {
            l.op(2, p!(rows, output))?;
        }
        Ok(())
    }
    fn layer(&self, l: &mut NativeLedger, i: usize, n: u64, streamed: bool) -> Result<()> {
        let c = &self.cfg;
        let hidden = u64::try_from(c.hidden_size).map_err(|_| overflow())?;
        let heads = u64::try_from(c.num_heads).map_err(|_| overflow())?;
        let kvheads = u64::try_from(c.num_kv_heads).map_err(|_| overflow())?;
        let dim = u64::try_from(c.head_dim).map_err(|_| overflow())?;
        let ff = u64::try_from(c.intermediate_size).map_err(|_| overflow())?;
        if hidden == 0 || heads == 0 || kvheads == 0 || dim == 0 {
            return Err(overflow());
        }
        let stem = format!("{LANGUAGE_TOWER_PREFIX}layers.{i}");
        l.norm(p!(n, hidden), n, 2, 2)?;
        for (name, out) in [
            ("q", p!(heads, dim)),
            ("k", p!(kvheads, dim)),
            ("v", p!(kvheads, dim)),
        ] {
            self.projection(l, &format!("{stem}.self_attn.{name}_proj"), n, hidden, out)?;
        }
        l.norm(p!(n, heads, dim), p!(n, heads), 2, 2)?;
        l.norm(p!(n, kvheads, dim), p!(n, kvheads), 2, 2)?;
        l.rope(p!(n, heads, dim), 2, true)?;
        l.rope(p!(n, kvheads, dim), 2, true)?;
        if !streamed {
            let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS as u64;
            let capacity = p!(ceil(n, block)?, block, kvheads, dim);
            l.repeated(2, 1, 2)?; // zero scalar for each K/V cache block
            l.repeated(2, capacity, 2)?; // zero-fill K and V backing
            l.repeated(2, capacity, 2)?; // slice-update destinations; no donation credit
        }
        if c.attn_logit_softcap.is_some() {
            // Qwen3-VL product configs do not set a softcap, but the generic supported decoder
            // permits one. Its sdpa_eager branch is f32 with explicitly repeated KV.
            l.cast(p!(n, heads, dim), 2, 4)?;
            l.repeated(4, p!(n, kvheads, dim), 2)?;
            if heads != kvheads {
                l.repeated(4, p!(n, heads, dim), 2)?;
            }
            l.matmul(4, heads, n, n, dim)?;
            l.scalar(4)?;
            l.op(4, p!(heads, n, n))?; // scale scores
            l.repeated(4, 1, 2)?;
            l.repeated(4, p!(heads, n, n), 3)?; // /cap, tanh, *cap
            l.array(4, p!(n, n))?; // causal_mask host-built additive upload
            l.op(4, p!(heads, n, n))?;
            l.softmax(4, p!(heads, n, n))?;
            l.matmul(4, heads, n, dim, n)?;
            l.cast(p!(n, heads, dim), 4, 2)?;
        } else {
            decoder_attention(l, n, heads, kvheads, dim)?;
        }
        l.reshape_copy(2, p!(n, heads, dim))?;
        self.projection(
            l,
            &format!("{stem}.self_attn.o_proj"),
            n,
            p!(heads, dim),
            hidden,
        )?;
        l.op(2, p!(n, hidden))?;
        l.norm(p!(n, hidden), n, 2, 2)?;
        self.projection(l, &format!("{stem}.mlp.gate_proj"), n, hidden, ff)?;
        l.silu(2, p!(n, ff))?;
        self.projection(l, &format!("{stem}.mlp.up_proj"), n, hidden, ff)?;
        l.op(2, p!(n, ff))?;
        self.projection(l, &format!("{stem}.mlp.down_proj"), n, ff, hidden)?;
        l.op(2, p!(n, hidden))
    }
    fn encode(
        &self,
        l: &mut NativeLedger,
        n: u64,
        real: u64,
        config: &IrisConfig,
        streamed: bool,
    ) -> Result<()> {
        let hidden = u64::try_from(self.cfg.hidden_size).map_err(|_| overflow())?;
        let dim = u64::try_from(self.cfg.head_dim).map_err(|_| overflow())?;
        l.array(4, n)?; // actual token ids, resident_states / streamed_states
        let embed = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens");
        if let Some(scales) = self.built.get(&format!("{embed}.scales")) {
            let weight = self
                .built
                .get(&format!("{embed}.weight"))
                .ok_or_else(|| Error::MissingTensor(format!("{embed}.weight")))?;
            let words = weight.shape[1] as u64;
            let groups = scales.shape[1] as u64;
            let bits = p!(words, 32)
                .checked_div(hidden)
                .filter(|&v| matches!(v, 4 | 8))
                .ok_or_else(overflow)?;
            l.array(4, p!(n, words))?; // packed vocabulary-row gather
            l.repeated(2, p!(n, groups), 2)?; // scales/biases gathers
            dequantize_embedding(l, n, hidden, words, groups, bits)?;
        } else {
            l.array(2, p!(n, hidden))?; // dense embedding Gather
        }
        l.array(4, n)?; // positions arange (rope::cos_sin)
        l.array(4, n)?; // position cast
        l.repeated(4, p!(n, dim), 3)?; // angles, cos, sin
        l.repeated(2, p!(n, dim), 2)?; // table casts
        let deepest = *config.text_encoder.hidden_layers.last().unwrap_or(&0);
        let run = if streamed {
            deepest.min(self.cfg.num_layers)
        } else {
            self.cfg.num_layers
        };
        for i in 0..run {
            if streamed {
                self.load(l, |name| {
                    crate::text_encoder::tower_layer_of(name) == Some(i)
                })?;
            }
            self.layer(l, i, n, streamed)?;
        }
        if run == self.cfg.num_layers && run != 0 {
            l.norm(p!(n, hidden), n, 2, 2)?;
        }
        // conditioning_body: exact actual real slice extents from assemble_window; one f32
        // cast per picked state, stack, and (only on the short-window branch) zero pad + concat.
        let max = config.text_encoder.max_length as u64;
        let layers = config.text_encoder.hidden_layers.len() as u64;
        let d = config.text_encoder.dim as u64;
        if real > max {
            return Err(overflow());
        }
        l.repeated(4, p!(real, d), layers)?;
        l.concat(4, p!(real, layers, d), layers)?;
        if real < max {
            l.scalar(4)?;
            l.array(4, p!(max - real, layers, d))?;
            l.concat(4, p!(max, layers, d), 2)?;
        }
        Ok(())
    }
}

fn noise(l: &mut NativeLedger, rows: u64, channels: u64, h: u64, w: u64) -> Result<()> {
    // pipeline::noise_batch -> random.cpp:30-34,95-141,174-204. Explicit per-row keys.
    for _ in 0..rows {
        l.array(4, 2)?; // random::key
        l.repeated(4, 1, 6)?; // low, high, hi-low, upper, maxval, sqrt(2)
        let n = p!(channels, h, w);
        l.array(4, n)?; // RandomBits u32
        l.array(4, n)?; // u32 -> f32 for divide
        l.op(4, n)?; // divide by maxval
        l.op(4, n)?; // min upper
        l.op(4, n)?; // uniform range product
        l.op(4, n)?; // uniform low add
        l.op(4, n)?; // erfinv
        l.op(4, n)?; // normal scale product
    }
    l.concat(4, p!(rows, channels, h, w), rows)
}
fn solver_update(
    l: &mut NativeLedger,
    n: u64,
    prediction: Prediction,
    update: &mlx_gen::gen_core::iris::DpmUpdate,
    guided: bool,
) -> Result<()> {
    if guided {
        l.scalar(4)?; // CFG scale
        l.repeated(4, n, 3)?; // difference, scale, unconditional sum
    }
    if matches!(prediction, Prediction::Velocity) {
        l.scalar(4)?; // step.s
        l.repeated(4, n, 2)?; // s*out, x-(s*out)
    }
    match update {
        mlx_gen::gen_core::iris::DpmUpdate::First { .. } => {
            l.repeated(4, 1, 2)?; // cx, c0
            l.repeated(4, n, 3)?; // cx*x, c0*x0, difference
        }
        mlx_gen::gen_core::iris::DpmUpdate::Second { .. } => {
            l.repeated(4, 1, 4)?; // r0, cx, c0, c1
            l.repeated(4, n, 7)?; // x0-prev, /r0, cx*x, c0*x0, subtract, c1*d, subtract
        }
    }
    Ok(())
}
fn image(l: &mut NativeLedger, n: u64) -> Result<()> {
    // mlx-gen/src/image.rs:85-119. Scalars: half(shared by two operations), 0, 1, 255.
    l.repeated(4, 1, 4)?;
    l.repeated(4, n, 6)?; // x*half,+half,maximum,minimum,*255,round
    l.reshape_copy(4, n) // NHWC logical readback; host u8 Vec is outside Active
}
fn preview(l: &mut NativeLedger, h: u64, w: u64, patch: u64) -> Result<()> {
    let pixels = p!(h / patch, w / patch);
    // preview_image mean over patch axes, then preview::project_latents (first batch row).
    l.reduce(4, p!(3, h, w), p!(3, pixels), p!(patch, patch), false)?;
    l.scalar(4)?; // mean normalizer
    l.op(4, p!(3, pixels))?;
    l.array(4, 9)?; // exact identity factors * 0.5
    l.array(4, 3)?; // bias
    l.reshape_copy(4, p!(3, pixels))?;
    l.matmul(4, 1, pixels, 3, 3)?;
    l.op(4, p!(3, pixels))?; // bias add
    l.repeated(4, 1, 3)?; // zero, one, 255
    l.repeated(4, p!(3, pixels), 4)?; // max,min,*255,round
    Ok(())
}

fn generation_upper(
    req: &IosPricingRequest<'_>,
    token: Option<&SamplingToken>,
    l: &mut NativeLedger,
) -> Result<()> {
    let resources =
        GenerationResources::from_spec(req.spec, crate::model::MODEL_ID).map_err(Error::from)?;
    let streamed = req.spec.load_shape == LoadShape::DeferredMaterialization;
    // The SAME side resolution as the pricing: an actual request, or a recorded token replay.
    let work = generation_work(req, token, &resources, streamed)?;
    let config = &work.config;
    let plan = work.envelope.plan;
    let b = work.envelope.rows;
    let count = u64::from(work.envelope.count);
    let h = u64::from(req.geometry.height);
    let w = u64::from(req.geometry.width);
    if b == 0 || count == 0 {
        return Err(Error::Unsupported(
            "Iris native generation work/geometry mismatch".into(),
        ));
    }
    let staged =
        req.execution.stage_residency || req.spec.offload_policy == OffloadPolicy::Sequential;
    let path = resources
        .backbone_dir
        .join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
    crate::memory_strategy::refuse_streamed_adapters(crate::model::MODEL_ID, req.spec)
        .map_err(Error::from)?;
    let mut bb = priced_backbone(
        config,
        &path,
        "",
        req.spec.precision,
        req.spec.quantize,
        streamed,
        config.model.in_channels as u64,
        Pricing::default(),
    )?;
    let raw: HashMap<String, Header> = crate::stream::headers(&path)?
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
    let artifact = crate::packed::artifact_tier(&path)?;
    let quantize = if artifact.is_none() {
        BackboneQuant::for_load(req.spec.quantize)?
    } else {
        None
    };
    let tower = TowerFacts::read(&resources.text_encoder_dir, req.spec.quantize, streamed)?;
    let (adapter_inventory, packed_hosts) = if req.spec.adapters.is_empty() {
        (crate::adapters::AdapterHeaderInventory::default(), 0)
    } else {
        let (projections, packed) = crate::adapters::projection_shapes_from_headers(&path)?;
        (
            crate::adapters::header_inventory(
                &req.spec.adapters,
                &projections,
                compute_dtype(req.spec.precision),
                IrisTask::Generation,
                crate::model::MODEL_ID,
            )?,
            packed,
        )
    };
    bb.facts.adapter_facts = adapter_inventory.projections.clone();
    // Residency::from_policy(Resident) loads both initially; run_request_scoped(staged) evicts
    // that pair and loads both again. Sequential starts cold and loads once inside the request.
    // Counting both pairs never credits their drops (gen-core/residency.rs:192-210,423-485).
    let loads = 1 + u64::from(staged && req.spec.offload_policy == OffloadPolicy::Resident);
    for _ in 0..loads {
        tower.initial_load(l, streamed)?;
        load_backbone_set(
            l,
            &raw,
            bb.facts.cw,
            "",
            |name| !streamed || Stack::of(name).is_none(),
            quantize,
        )?;
        installed_adapters::load(l, &adapter_inventory, packed_hosts)?;
    }
    let tokenizer = mlx_gen::gen_core::tokenizer::TextTokenizer::from_file(
        resources.text_encoder_dir.join("tokenizer.json"),
        mlx_gen::gen_core::tokenizer::TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: mlx_gen::gen_core::tokenizer::ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(Error::from)?;
    let prefix = tokenizer
        .encode_ids(PROMPT_PREFIX, false)
        .map_err(Error::from)?;
    let suffix = tokenizer
        .encode_ids(mlx_gen::gen_core::iris::PROMPT_SUFFIX, false)
        .map_err(Error::from)?;
    let encodes = p!(b, 1) + u64::from(work.envelope.uses_cfg());
    match &work.prompts {
        // Tokenize the ACTUAL prompts/negative independently: no guessed full-window or
        // longest-row multiplication hides per-call work. Pure host operation, no MLX execution.
        Some((prompts, negative)) => {
            for prompt in prompts
                .iter()
                .map(String::as_str)
                .chain(work.envelope.uses_cfg().then_some(negative.as_str()))
            {
                let caption = tokenizer.encode_ids(prompt, false).map_err(Error::from)?;
                let window = mlx_gen::gen_core::iris::assemble_window(
                    &prefix,
                    &caption,
                    &suffix,
                    config.text_encoder.max_length,
                )
                .map_err(Error::from)?;
                tower.encode(
                    l,
                    window.input_ids.len() as u64,
                    window.window_tokens() as u64,
                    config,
                    streamed,
                )?;
            }
        }
        // A payload-free token replay records rows/CFG but no caption tokens. `encode` is NOT
        // monotone in the window (a short window adds a zero pad and a full-width concat, and
        // the decoder attention route changes with the token count), so no single "largest"
        // window is assumed. Every caption length assemble_window can produce (0 through the
        // budget `max_length - suffix`) is priced exactly on a fresh ledger, and the MAXIMUM is
        // charged for each encode: the actual unrecorded caption is one of these windows.
        None => {
            let max_length = config.text_encoder.max_length;
            let budget = max_length
                .checked_sub(suffix.len())
                .filter(|&b| b > 0)
                .ok_or_else(|| {
                    Error::Unsupported("Iris text window leaves no room for a caption".into())
                })?;
            let mut window_upper = 0u64;
            for caption in 0..=budget {
                let real = plus(caption as u64, suffix.len() as u64)?;
                let mut window = NativeLedger::new(l.page, l.backend)?;
                tower.encode(
                    &mut window,
                    plus(prefix.len() as u64, real)?,
                    real,
                    config,
                    streamed,
                )?;
                window_upper = window_upper.max(window.total);
            }
            l.total = plus(l.total, p!(window_upper, encodes))?;
        }
    }
    let cond = p!(
        config.text_encoder.max_length,
        config.text_encoder.hidden_layers.len(),
        config.text_encoder.dim
    );
    let state = p!(b, config.model.in_channels, h, w);
    let attention = plan_of(req.execution);
    let rungs = work.rungs;
    let mut first_forward = true;
    for _serial in 0..count {
        noise(l, b, config.model.in_channels as u64, h, w)?;
        l.concat(4, p!(b, cond), b)?;
        if work.envelope.uses_cfg() {
            l.concat(4, p!(2, b, cond), p!(2, b))?;
        }
        // Full indexed plan, not GenerationEnvelope's finite transition representatives.
        for index in 0..plan.len() {
            let step = plan
                .step(index)
                .ok_or_else(|| Error::Unsupported("invalid Iris actual solver epoch".into()))?;
            let guided = work.envelope.cfg_at(&step);
            let rows = p!(b, if guided { 2 } else { 1 });
            if guided {
                l.concat(4, p!(2, state), 2)?;
            }
            l.array(4, rows)?; // model time Array::from_slice
            Graph {
                l: &mut *l,
                bb: &bb.facts,
            }
            .forward(
                rows,
                h,
                w,
                ForwardMode {
                    by_row: rungs.pixel_rows,
                    first: first_forward,
                    streamed,
                },
                attention,
            )?;
            first_forward = false;
            solver_update(l, state, work.prediction, &step.update, guided)?;
            // Preview throttling is temporal, not a recipe proof. Charge the actual preview graph
            // at EVERY eligible solver callback when its sink is active (source-conditional upper).
            // A token replay does not record its preview sink: charge it (upper).
            if work.preview_active.unwrap_or(true) {
                preview(l, h, w, config.model.patch_size as u64)?;
            }
        }
        l.repeated(4, 1, 2)?; // final clamp bounds
        l.repeated(4, state, 2)?; // maximum, minimum
    }
    for _ in 0..p!(b, count) {
        image(l, p!(3, h, w))?;
    }
    Ok(())
}

fn depth_upper(req: &IosPricingRequest<'_>, l: &mut NativeLedger) -> Result<()> {
    // depth.rs:126-143 refuses installed adapters on the task itself.
    if !req.spec.adapters.is_empty() {
        return Err(Error::Unsupported(format!(
            "{}: LoadSpec::adapters is not supported by the Iris-3B depth task",
            crate::depth::DEPTH_MODEL_ID
        )));
    }
    let actual = req.depth.ok_or_else(|| {
        Error::Unsupported("Iris native depth recipe requires actual source/resolution".into())
    })?;
    let export = TaskExport::from_spec(req.spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
        .map_err(Error::from)?;
    let config = &export.config;
    let input = depth_input_plan(req.spec, actual)?;
    let h = input.model_height as u64;
    let w = input.model_width as u64;
    let channels = config.model.in_channels as u64;
    let streamed = req.spec.load_shape == LoadShape::DeferredMaterialization;
    let bb = priced_backbone(
        config,
        &export.weights_path(),
        "pixel.",
        req.spec.precision,
        req.spec.quantize,
        streamed,
        plus(channels, DEPTH_EXTRA_INPUT_CHANNELS as u64)?,
        Pricing::default(),
    )?;
    crate::packed::refuse_dense_requested_as_packed(
        crate::depth::DEPTH_MODEL_ID,
        &export.weights_path(),
        req.spec.quantize,
    )?;
    let headers: HashMap<String, Header> = crate::stream::headers(&export.weights_path())?
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
    load_backbone_set(
        l,
        &headers,
        bb.facts.cw,
        "pixel.",
        |name| !name.starts_with("depth_reducer.") && (!streamed || Stack::of(name).is_none()),
        None,
    )?;
    for name in ["depth_reducer.weight", "depth_reducer.bias"] {
        let h = headers
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        raw_header(l, h)?;
        cast_header(l, h, compute_dtype(req.spec.precision))?;
    }
    // EmptyPrompt::from_path decodes via safetensors host bytes; only its f32 embeddings from_slice
    // is a native allocation. No raw MLX Load exists for embeddings OR mask on this path.
    let empty = crate::stream::headers(&export.empty_prompt_path())?;
    let embeddings = empty
        .iter()
        .find(|h| h.name == "embeddings")
        .ok_or_else(|| Error::MissingTensor("empty-prompt embeddings".into()))?;
    l.array(4, header_elements(embeddings)?)?;
    l.array(4, p!(3, h, w))?; // predict RGB from_slice
    l.scalar(4)?; // zeros fill scalar
    l.array(4, p!(DEPTH_EXTRA_INPUT_CHANNELS, h, w))?;
    l.concat(
        4,
        p!(plus(channels, DEPTH_EXTRA_INPUT_CHANNELS as u64)?, h, w),
        2,
    )?;
    l.array(4, 1)?; // t_model from_slice
                    // Empty prompt broadcast is a view. Host preprocessing, map readback and source resize use
                    // only Rust Vec storage and intentionally contribute zero to MLX Active.
    Graph {
        l: &mut *l,
        bb: &bb.facts,
    }
    .forward(
        1,
        h,
        w,
        ForwardMode {
            by_row: false,
            first: true,
            streamed,
        },
        mlx_gen::AttentionPlan::UNBOUNDED,
    )?;
    // depth_reducer_body is addmm, NOT a convolution lowering despite its checkpoint key.
    l.reshape_copy(4, p!(channels, h, w))?;
    l.cast(p!(channels, h, w), 4, bb.facts.cw)?;
    l.matmul(bb.facts.cw, 1, p!(h, w), 1, channels)?;
    l.cast(p!(h, w), bb.facts.cw, 4)?;
    Ok(())
}

/// Complete source-derived cumulative native Active upper. Parent owns profile validation,
/// source identity/witness validation and certificate issuance. Host payload is a separate domain.
pub(super) fn cumulative_native_upper(
    req: &IosPricingRequest<'_>,
    page: u64,
    backend: NativeBackend,
) -> Result<u64> {
    req.validate_work_context()?;
    if backend == NativeBackend::Metal && std::env::var_os("MLX_SDPA_BLOCKS").is_some() {
        // metal/scaled_dot_product_attention.cpp:477 is the sole internal override. No inference
        // doc/runbook/supported product path exposes it; no execution witness binds its value.
        return Err(Error::Unsupported("Iris native certificate: MLX_SDPA_BLOCKS override is not bound by any execution witness".into()));
    }
    let mut ledger = NativeLedger::new(page, backend)?;
    match req.provider {
        crate::model::MODEL_ID => generation_upper(req, None, &mut ledger)?,
        crate::depth::DEPTH_MODEL_ID => depth_upper(req, &mut ledger)?,
        crate::restoration::MODEL_ID => restoration_upper(req, &mut ledger)?,
        _ => {
            return Err(Error::Unsupported(
                "Iris native certificate has no recipe for this route".into(),
            ))
        }
    }
    Ok(ledger.total)
}

/// A source-matched capture that does not attest its stream can only use the larger backend law.
pub(super) fn cumulative_native_upper_any(req: &IosPricingRequest<'_>, page: u64) -> Result<u64> {
    Ok(
        cumulative_native_upper(req, page, NativeBackend::Cpu)?.max(cumulative_native_upper(
            req,
            page,
            NativeBackend::Metal,
        )?),
    )
}

/// mlx_llm/primitives/attention.rs:147-265,582-641,693-734, with the frozen
/// core-llm/defaults.rs:200 `MLX.sdpa_kernel_routing=true`. This is the ACTUAL tiling schedule,
/// not the admission transient-price helper. Causal tiles attend their growing key prefix; every
/// tile's copies and every within-group/final concat count cumulatively on both backends.
fn decoder_attention(
    l: &mut NativeLedger,
    n: u64,
    heads: u64,
    kvheads: u64,
    dim: u64,
) -> Result<()> {
    if kvheads == 0 || heads == 0 || !heads.is_multiple_of(kvheads) {
        return Err(overflow());
    }
    let gqa = heads / kvheads;
    let vector = matches!(dim, 64 | 96 | 128 | 256) && gqa <= 32;
    let vector_rows = (32 / gqa).clamp(1, 8);
    let route = if n <= 8 {
        (vector && n > vector_rows).then_some((vector_rows, 256))
    } else if matches!(dim, 64 | 80 | 128) {
        Some((2048, 2048))
    } else if vector {
        Some((vector_rows, 256))
    } else {
        // These are the producer's tile-selection constants, NOT an invented native-op budget.
        let per_row = p!(heads.max(1), n.max(1), 4, 3);
        let rows = ((256u64 << 20) / per_row).clamp(8, i32::MAX as u64);
        (rows < n).then_some((rows, rows))
    };
    let Some((rows, eval_rows)) = route else {
        return l.sdpa(
            2,
            SdpaShape {
                b: 1,
                heads,
                kv_heads: kvheads,
                q: n,
                k: n,
                dim,
            },
            false,
            true,
        );
    };
    let checkpoint = n > eval_rows;
    let (mut start, mut group_rows, mut group_calls, mut groups, mut calls) = (0, 0, 0, 0, 0);
    while start < n {
        let count = rows.min(n - start);
        let end = plus(start, count)?;
        // sdpa_tile's q/K/V slicing is view-only. Its concrete primitive normalizers are in sdpa.
        l.sdpa(
            2,
            SdpaShape {
                b: 1,
                heads,
                kv_heads: kvheads,
                q: count,
                k: end,
                dim,
            },
            false,
            true,
        )?;
        group_rows = plus(group_rows, count)?;
        group_calls = plus(group_calls, 1)?;
        calls = plus(calls, 1)?;
        if checkpoint && (group_rows >= eval_rows || end == n) {
            l.concat(2, p!(heads, group_rows, dim), group_calls)?;
            groups = plus(groups, 1)?;
            group_rows = 0;
            group_calls = 0;
        }
        start = end;
    }
    l.concat(
        2,
        p!(heads, n, dim),
        if checkpoint { groups } else { calls },
    )
}

/// Resident QuantizedEmbedding::forward gathers codes/scales/biases before this call
/// (mlx_llm/primitives/quant.rs:80-94). CPU dequantization is the explicit ops.cpp:5715-5768
/// shift/concat/product/add graph; Metal fast::Quantize::eval_gpu:1801-1827 is a distinct kernel.
fn dequantize_embedding(
    l: &mut NativeLedger,
    rows: u64,
    hidden: u64,
    words: u64,
    groups: u64,
    bits: u64,
) -> Result<()> {
    let n = p!(rows, hidden);
    let packed = p!(rows, words);
    if l.backend == NativeBackend::Metal {
        l.array(2, n)?;
        l.array(4, packed)?; // conditional codes copy
        l.repeated(2, p!(rows, groups), 2)?; // conditional affine-parameter copies
    } else {
        let fields = 32 / bits;
        for _ in 0..fields {
            l.repeated(4, 1, 2)?; // left/right shift scalar counts
            l.repeated(4, packed, 2)?; // left and right shift outputs
        }
        l.concat(4, n, fields)?;
        l.cast(n, 4, 2)?; // U32 -> bf16 for affine product
        l.op(2, n)?; // scales product
        l.op(2, n)?; // biases add
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HOST: allocator-law arithmetic only (no Arrays, MLX evaluation or counter queries).
    /// U(n) = 0 at 0, else min(2*r(n), r(n) + 2*page), r = Metal page round above one page.
    #[test]
    fn allocator_operand_upper_includes_bounded_cache_reuse() {
        for (operand, upper) in [
            (0, 0),
            (1, 2),
            (4, 8),
            (16_384, 32_768),
            (16_385, 65_536),
            (32_768, 65_536),
            (32_769, 81_920),
        ] {
            assert_eq!(
                backing_upper(operand, 16_384).unwrap(),
                upper,
                "U({operand})"
            );
        }
        // Each operand is charged before multiplication, never as one combined request.
        let mut separate = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        separate.repeated(1, 16_385, 2).unwrap();
        let mut combined = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
        combined.array(1, 2 * 16_385).unwrap();
        assert_eq!((separate.total, combined.total), (131_072, 81_920));
    }

    /// HOST: invalid pages and overflow fail closed instead of becoming a certificate.
    #[test]
    fn allocator_arithmetic_fails_closed() {
        assert!(backing_upper(1, 0).is_err());
        assert!(backing_upper(1, 3).is_err());
        assert!(backing_upper(u64::MAX, 16_384).is_err());
        let mut ledger = NativeLedger::new(16_384, NativeBackend::Metal).unwrap();
        assert!(ledger.array(4, u64::MAX).is_err());
        assert!(ledger.repeated(4, 1, u64::MAX).is_err());
    }
}

// Source-derived adapter reconstruction: the strict loader's scalar/cast/matmul/Kronecker/
// Hadamard/Tucker graph. Greedy contraction selection mirrors the immutable adapter_load.rs;
// native requests below are independently charged through this ledger's per-backend laws.
mod installed_adapters {
    use super::*;
    use crate::adapters::{AdapterHeaderInventory, AdapterReconstructionFacts};
    use mlx_gen::adapters::facts::{dtype_width, AdapterFacts, TensorFacts};
    use mlx_rs::Dtype;
    use std::collections::{BTreeMap, BTreeSet};

    struct AdapterGraph<'a> {
        l: &'a mut NativeLedger,
    }

    impl AdapterGraph<'_> {
        fn output(
            &mut self,
            operation: &'static str,
            shape: Vec<i32>,
            dtype: Dtype,
        ) -> Result<TensorFacts> {
            let _ = operation; // Source-site name, not a reserve or opcode budget.
            self.l
                .array(dtype_width(dtype), checked_elements(&shape)?)?;
            contiguous(shape, dtype)
        }

        fn cast(&mut self, input: &TensorFacts, dtype: Dtype) -> Result<TensorFacts> {
            checked_tensor_bytes(input)?;
            if input.dtype == dtype {
                Ok(input.clone())
            } else {
                self.output("astype", input.shape.clone(), dtype)
            }
        }

        fn scalar(&mut self) -> Result<()> {
            // crate::array::scalar / Array::from_slice(&[alpha/rank], &[1]): allocated when the
            // reconstruction expression is constructed, not when its lazy outputs are evaluated.
            self.l.scalar(4)?;
            Ok(())
        }

        fn matmul(&mut self, a: &TensorFacts, b: &TensorFacts) -> Result<TensorFacts> {
            if a.shape.is_empty() || b.shape.is_empty() {
                return Err(Error::Msg(
                    "iris: adapter matmul factors cannot be scalars".into(),
                ));
            }
            let avector = a.shape.len() == 1;
            let bvector = b.shape.len() == 1;
            let ashape = if avector {
                vec![1, a.shape[0]]
            } else {
                a.shape.clone()
            };
            let bshape = if bvector {
                vec![b.shape[0], 1]
            } else {
                b.shape.clone()
            };
            let (m, k) = (ashape[ashape.len() - 2], ashape[ashape.len() - 1]);
            let (kb, n) = (bshape[bshape.len() - 2], bshape[bshape.len() - 1]);
            if k != kb {
                return Err(Error::Msg(
                    "iris: adapter factor matmul contraction widths disagree".into(),
                ));
            }
            let mut shape = broadcast(&ashape[..ashape.len() - 2], &bshape[..bshape.len() - 2])?;
            if !avector {
                shape.push(m);
            }
            if !bvector {
                shape.push(n);
            }
            let dtype = matmul_dtype(a.dtype, b.dtype)?;
            self.cast(a, dtype)?;
            self.cast(b, dtype)?;
            // Native matmul's promoted operands and each backend's layout/workspace branches.
            // ops.cpp flattens A's leading rows when the original B is at most rank two.
            let batch = checked_elements(&broadcast(
                &ashape[..ashape.len() - 2],
                &bshape[..bshape.len() - 2],
            )?)?;
            let width = dtype_width(dtype);
            if a.shape.len() > 2 && b.shape.len() <= 2 {
                self.l
                    .matmul(width, 1, product(&[batch, m as u64])?, n as u64, k as u64)?;
            } else {
                self.l.matmul(width, batch, m as u64, n as u64, k as u64)?;
            }
            contiguous(shape, dtype)
        }

        fn multiply(
            &mut self,
            a: &TensorFacts,
            b: &TensorFacts,
            operation: &'static str,
        ) -> Result<TensorFacts> {
            let dtype = Dtype::from_promoting_types(a.dtype, b.dtype);
            self.cast(a, dtype)?;
            self.cast(b, dtype)?;
            self.output(operation, broadcast(&a.shape, &b.shape)?, dtype)
        }

        fn scale(&mut self, input: &TensorFacts) -> Result<TensorFacts> {
            self.scalar()?;
            let dtype = Dtype::from_promoting_types(input.dtype, Dtype::Float32);
            self.cast(input, dtype)?;
            if dtype != Dtype::Float32 {
                self.output("reconstruction scale scalar cast", vec![1], dtype)?;
            }
            self.output("reconstruction scale multiply", input.shape.clone(), dtype)
        }

        fn kron(&mut self, a: &TensorFacts, b: &TensorFacts) -> Result<TensorFacts> {
            // reconstruct_lokr_delta_scaled appends singleton dimensions to w1 until its rank reaches
            // w2's; native kron PREPENDS any remaining singleton dimensions to the shorter operand.
            if checked_elements(&a.shape)? == 0 || checked_elements(&b.shape)? == 0 {
                return Err(Error::Msg(
                    "iris: native Kronecker factors cannot be empty".into(),
                ));
            }
            let mut a = a.clone();
            a.shape.resize(a.shape.len().max(b.shape.len()), 1);
            let rank = a.shape.len().max(b.shape.len());
            let mut sa = vec![1; rank - a.shape.len()];
            sa.extend(&a.shape);
            let mut sb = vec![1; rank - b.shape.len()];
            sb.extend(&b.shape);
            let shape: Vec<i32> = sa
                .iter()
                .zip(&sb)
                .map(|(&x, &y)| {
                    x.checked_mul(y).ok_or_else(|| {
                        Error::Msg("iris: adapter Kronecker dimension overflows MLX shape".into())
                    })
                })
                .collect::<Result<_>>()?;
            let dtype = Dtype::from_promoting_types(a.dtype, b.dtype);
            self.cast(&a, dtype)?;
            self.cast(b, dtype)?;
            // Native kron inserts singleton axes (views) and broadcasts the multiply; no expanded
            // operand copy. Its final reshape may require a copy if a scalar-factor fast path kept a
            // Tucker factor's column-major orientation. Count that explicit native reshape branch as
            // an upper, without pretending that an ordinary row-major reshape actually allocates.
            self.output("Kronecker broadcast multiply", shape.clone(), dtype)?;
            self.output("Kronecker reshape copy upper", shape, dtype)
        }

        fn tucker(
            &mut self,
            t: &TensorFacts,
            a: &TensorFacts,
            b: &TensorFacts,
            loha_order: bool,
        ) -> Result<TensorFacts> {
            if t.shape.len() != 4 || a.shape.len() != 2 || b.shape.len() != 2 {
                return Err(Error::Msg(
                    "iris: Tucker reconstruction requires t[i,j,h,w], a[i,p], b[j,r]".into(),
                ));
            }
            // MLX permits broadcasting a contracted dimension of size one. The dimensions in the
            // native einsum map are the maximum across occurrences; incompatible non-singletons error.
            broadcast_dim(t.shape[0], a.shape[0])?;
            broadcast_dim(t.shape[1], b.shape[0])?;
            let i = t.shape[0].max(a.shape[0]);
            let j = t.shape[1].max(b.shape[0]);
            let dims = BTreeMap::from([
                ('i', i),
                ('j', j),
                ('h', t.shape[2]),
                ('w', t.shape[3]),
                ('p', a.shape[1]),
                ('r', b.shape[1]),
            ]);
            let output = vec!['p', 'r', 'h', 'w'];
            let mut terms = vec![
                Term {
                    axes: vec!['i', 'j', 'h', 'w'],
                    tensor: t.clone(),
                },
                Term {
                    axes: vec!['i', 'p'],
                    tensor: a.clone(),
                },
                Term {
                    axes: vec!['j', 'r'],
                    tensor: b.clone(),
                },
            ];
            // LoKr calls ijhw,ip,jr; LoHa calls ijhw,jr,ip. Operand order breaks greedy cost ties.
            if loha_order {
                terms.swap(1, 2);
            }
            let naive_cost = checked_mul(
                axis_elements(&dims.keys().copied().collect::<Vec<_>>(), &dims)?,
                3,
                "Tucker naive cost",
            )?;
            let mut max_size = axis_elements(&output, &dims)?;
            for term in &terms {
                max_size = max_size.max(axis_elements(&term.axes, &dims)?);
            }
            let mut path_cost = 0_u64;
            while terms.len() > 1 {
                let mut candidates = Vec::new();
                for outer in [false, true] {
                    for x in 0..terms.len() {
                        for y in x + 1..terms.len() {
                            let union: BTreeSet<char> = terms[x]
                                .axes
                                .iter()
                                .chain(&terms[y].axes)
                                .copied()
                                .collect();
                            let shared = terms[x].axes.iter().any(|c| terms[y].axes.contains(c));
                            if !outer && !shared {
                                continue;
                            }
                            let mut kept: BTreeSet<char> = output
                                .iter()
                                .copied()
                                .filter(|c| union.contains(c))
                                .collect();
                            for (index, term) in terms.iter().enumerate() {
                                if index != x && index != y {
                                    kept.extend(
                                        term.axes.iter().copied().filter(|c| union.contains(c)),
                                    );
                                }
                            }
                            let mut axes: Vec<char> = kept.into_iter().collect();
                            // Native output-axis ties come from an unordered_set. Copy upper bounds below
                            // do not depend on a tie's incidental order.
                            axes.sort_by_key(|c| dims[c]);
                            let size = axis_elements(&axes, &dims)?;
                            let cost = checked_mul(
                                axis_elements(&union.iter().copied().collect::<Vec<_>>(), &dims)?,
                                if union.len() > axes.len() { 2 } else { 1 },
                                "Tucker contraction cost",
                            )?;
                            if size > max_size
                                || checked_add(path_cost, cost, "Tucker candidate path cost")?
                                    > naive_cost
                            {
                                continue;
                            }
                            let removed = axis_elements(&terms[x].axes, &dims)? as i128
                                + axis_elements(&terms[y].axes, &dims)? as i128
                                - size as i128;
                            candidates.push((removed, cost, x, y, axes));
                        }
                    }
                    if !candidates.is_empty() {
                        break;
                    }
                }
                if candidates.is_empty() {
                    let term = self.naive_einsum(&terms, &output, &dims)?;
                    terms = vec![term];
                    break;
                }
                candidates.sort_by(|a, b| {
                    b.0.cmp(&a.0)
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                        .then(a.3.cmp(&b.3))
                });
                let (_, cost, x, y, axes) = candidates.remove(0);
                let union: BTreeSet<_> = terms[x]
                    .axes
                    .iter()
                    .chain(&terms[y].axes)
                    .copied()
                    .collect();
                let contracts = union.len() > axes.len();
                let tensor = if contracts {
                    let dtype = matmul_dtype(terms[x].tensor.dtype, terms[y].tensor.dtype)?;
                    for term in [&terms[x], &terms[y]] {
                        // Native batch_tensordot broadcasts contracted dimensions before its
                        // transpose_reshape. Count the broadcast size for a possible normalization copy.
                        let shape: Vec<i32> = term.axes.iter().map(|c| dims[c]).collect();
                        // Broadcast views preserve source data_size. The native cast can keep that
                        // smaller backing, while a required pre-cast transpose_reshape copies the
                        // expanded operand in its ORIGINAL dtype. A post-cast matmul normalization
                        // instead copies the promoted dtype. The one copy upper uses the wider
                        // actual branch; it must not assume promotion always widens (I64+F16->F16).
                        self.cast(&term.tensor, dtype)?;
                        let copy_dtype = if dtype_width(term.tensor.dtype) > dtype_width(dtype) {
                            term.tensor.dtype
                        } else {
                            dtype
                        };
                        self.output(
                            "Tucker transpose_reshape/matmul operand copy upper",
                            shape,
                            copy_dtype,
                        )?;
                    }
                    self.contraction(&terms[x], &terms[y], &axes, &dims, dtype)?
                } else {
                    self.naive_einsum(&[terms[x].clone(), terms[y].clone()], &axes, &dims)?
                        .tensor
                };
                terms.remove(y);
                terms.remove(x);
                terms.push(Term { axes, tensor });
                path_cost = checked_add(path_cost, cost, "Tucker selected path cost")?;
            }
            let term = terms
                .pop()
                .ok_or_else(|| Error::Msg("iris: empty Tucker graph".into()))?;
            // The final transpose is a view. Describe its requested shape, retaining the same payload.
            contiguous(output.iter().map(|c| dims[c]).collect(), term.tensor.dtype)
        }

        /// Native einsum batch_tensordot: common kept labels form the batch, remaining kept
        /// A/B labels form M/N, and removed labels form K. Transpose/reshape views are zero;
        /// conditional pre-normalization copies were charged at the source sites above.
        fn contraction(
            &mut self,
            a: &Term,
            b: &Term,
            output: &[char],
            dims: &BTreeMap<char, i32>,
            dtype: Dtype,
        ) -> Result<TensorFacts> {
            let batch_axes: Vec<char> = output
                .iter()
                .copied()
                .filter(|c| a.axes.contains(c) && b.axes.contains(c))
                .collect();
            let left: Vec<char> = output
                .iter()
                .copied()
                .filter(|c| a.axes.contains(c) && !batch_axes.contains(c))
                .collect();
            let right: Vec<char> = output
                .iter()
                .copied()
                .filter(|c| b.axes.contains(c) && !batch_axes.contains(c))
                .collect();
            let contracted: BTreeSet<char> = a
                .axes
                .iter()
                .chain(&b.axes)
                .copied()
                .filter(|c| !output.contains(c))
                .collect();
            self.l.matmul(
                dtype_width(dtype),
                axis_elements(&batch_axes, dims)?,
                axis_elements(&left, dims)?,
                axis_elements(&right, dims)?,
                axis_elements(&contracted.into_iter().collect::<Vec<_>>(), dims)?,
            )?;
            contiguous(output.iter().map(|c| dims[c]).collect(), dtype)
        }

        fn naive_einsum(
            &mut self,
            terms: &[Term],
            output: &[char],
            dims: &BTreeMap<char, i32>,
        ) -> Result<Term> {
            let first = terms
                .first()
                .ok_or_else(|| Error::Msg("iris: empty Tucker fallback".into()))?;
            let mut axes = first.axes.clone();
            let mut tensor = first.tensor.clone();
            for term in &terms[1..] {
                for c in &term.axes {
                    if !axes.contains(c) {
                        axes.push(*c);
                    }
                }
                let dtype = Dtype::from_promoting_types(tensor.dtype, term.tensor.dtype);
                self.cast(&tensor, dtype)?;
                self.cast(&term.tensor, dtype)?;
                tensor = self.output(
                    "Tucker naive broadcast multiply",
                    axes.iter().map(|c| dims[c]).collect(),
                    dtype,
                )?;
            }
            if axes.iter().any(|c| !output.contains(c)) {
                let dtype = match tensor.dtype {
                    Dtype::Bool | Dtype::Int8 | Dtype::Int16 => Dtype::Int32,
                    Dtype::Uint8 | Dtype::Uint16 => Dtype::Uint32,
                    dtype => dtype,
                };
                let out_shape: Vec<i32> = output.iter().map(|c| dims[c]).collect();
                let input = checked_elements(&tensor.shape)?;
                let out = checked_elements(&out_shape)?;
                let extent = input.checked_div(out).unwrap_or(0);
                adapter_reduce(
                    self.l,
                    dtype_width(tensor.dtype),
                    dtype_width(dtype),
                    input,
                    out,
                    extent,
                )?;
                tensor = contiguous(out_shape, dtype)?;
            }
            Ok(Term {
                axes: output.to_vec(),
                tensor,
            })
        }
    }

    #[derive(Clone)]
    struct Term {
        axes: Vec<char>,
        tensor: TensorFacts,
    }

    fn checked_add(a: u64, b: u64, what: &str) -> Result<u64> {
        a.checked_add(b)
            .ok_or_else(|| Error::Unsupported(format!("iris: {what} overflow")))
    }
    fn checked_mul(a: u64, b: u64, what: &str) -> Result<u64> {
        a.checked_mul(b)
            .ok_or_else(|| Error::Unsupported(format!("iris: {what} overflow")))
    }
    fn checked_elements(shape: &[i32]) -> Result<u64> {
        if shape.iter().any(|&d| d < 0) {
            return Err(Error::Unsupported(
                "iris: negative adapter shape dimension".into(),
            ));
        }
        if shape.contains(&0) {
            return Ok(0);
        }
        shape.iter().try_fold(1, |total, &d| {
            checked_mul(total, d as u64, "adapter shape product")
        })
    }
    fn checked_tensor_bytes(t: &TensorFacts) -> Result<u64> {
        checked_mul(
            checked_elements(&t.shape)?,
            dtype_width(t.dtype),
            "adapter tensor byte product",
        )
    }
    fn axis_elements(axes: &[char], dims: &BTreeMap<char, i32>) -> Result<u64> {
        let shape = axes
            .iter()
            .map(|c| {
                dims.get(c)
                    .copied()
                    .ok_or_else(|| Error::Msg("iris: unknown Tucker axis".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        checked_elements(&shape)
    }
    fn matmul_dtype(a: Dtype, b: Dtype) -> Result<Dtype> {
        let dtype = Dtype::from_promoting_types(a, b);
        if matches!(
            dtype,
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64 | Dtype::Complex64
        ) {
            Ok(dtype)
        } else {
            Err(Error::Msg(format!(
                "iris: native adapter matmul requires inexact types, got {a:?}/{b:?}"
            )))
        }
    }
    fn contiguous(shape: Vec<i32>, dtype: Dtype) -> Result<TensorFacts> {
        checked_elements(&shape)?;
        let mut stride = 1_usize;
        let mut strides = vec![0; shape.len()];
        for (i, &d) in shape.iter().enumerate().rev() {
            strides[i] = stride;
            stride = stride.checked_mul(d as usize).ok_or_else(|| {
                Error::Unsupported("iris: adapter stride product overflow".into())
            })?;
        }
        Ok(TensorFacts {
            shape,
            dtype,
            strides,
        })
    }
    fn broadcast_dim(a: i32, b: i32) -> Result<i32> {
        if a == b {
            Ok(a)
        } else if a == 1 {
            Ok(b)
        } else if b == 1 {
            Ok(a)
        } else {
            Err(Error::Msg(format!(
                "iris: adapter dimensions {a} and {b} do not broadcast"
            )))
        }
    }
    fn broadcast(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
        checked_elements(a)?;
        checked_elements(b)?;
        let n = a.len().max(b.len());
        (0..n)
            .map(|i| {
                // Missing leading axes are the native broadcast's literal singleton dimensions.
                let da = if i >= n - a.len() {
                    a[i - (n - a.len())]
                } else {
                    1
                };
                let db = if i >= n - b.len() {
                    b[i - (n - b.len())]
                } else {
                    1
                };
                broadcast_dim(da, db)
            })
            .collect()
    }

    pub(super) fn load(
        l: &mut NativeLedger,
        inventory: &AdapterHeaderInventory,
        packed_hosts: u64,
    ) -> Result<()> {
        // ProjectionHost's scalar-backed logical placeholders never materialize a dense base.
        l.repeated(4, 1, packed_hosts)?;
        for file in &inventory.files {
            for source in file.sources.values() {
                l.array(dtype_width(source.dtype), checked_elements(&source.shape)?)?;
            }
            for key in &file.scalar_reads {
                let alpha = file
                    .sources
                    .get(key)
                    .ok_or_else(|| Error::MissingTensor(key.clone()))?;
                if alpha.dtype != Dtype::Float32 {
                    l.array(4, checked_elements(&alpha.shape)?)?;
                }
            }
        }
        for file in &inventory.files {
            for plan in &file.reconstructions {
                let mut graph = AdapterGraph { l: &mut *l };
                let (path, delta) = match plan {
                    AdapterReconstructionFacts::Lora {
                        projection,
                        up,
                        alpha_fold,
                        ..
                    } => {
                        if *alpha_fold {
                            let source = &file.sources[up];
                            let transposed = TensorFacts {
                                shape: source.shape.iter().copied().rev().collect(),
                                dtype: source.dtype,
                                strides: source.strides.iter().copied().rev().collect(),
                            };
                            graph.scale(&transposed)?;
                        }
                        (projection, None)
                    }
                    AdapterReconstructionFacts::Lokr {
                        projection,
                        factors,
                    } => {
                        let mut factor = |full: &str,
                                          a: &str,
                                          b: &str,
                                          t: Option<&str>|
                         -> Result<TensorFacts> {
                            if let Some(key) = factors.get(full) {
                                return Ok(file.sources[key].clone());
                            }
                            let wa = factors
                                .get(a)
                                .and_then(|k| file.sources.get(k))
                                .ok_or_else(|| {
                                    Error::Msg(format!("iris: {projection} missing {a}"))
                                })?;
                            let wb = factors
                                .get(b)
                                .and_then(|k| file.sources.get(k))
                                .ok_or_else(|| {
                                    Error::Msg(format!("iris: {projection} missing {b}"))
                                })?;
                            if let Some(t) = t
                                .filter(|_| {
                                    !factors.contains_key("lokr_w2")
                                        && factors.contains_key("lokr_t2")
                                })
                                .and_then(|t| factors.get(t))
                                .and_then(|k| file.sources.get(k))
                            {
                                graph.tucker(t, wa, wb, false)
                            } else {
                                graph.matmul(wa, wb)
                            }
                        };
                        let w1 = factor("lokr_w1", "lokr_w1_a", "lokr_w1_b", None)?;
                        let w2 = factor("lokr_w2", "lokr_w2_a", "lokr_w2_b", Some("lokr_t2"))?;
                        let product = graph.kron(&w1, &w2)?;
                        let delta = graph.scale(&product)?;
                        (projection, Some(delta))
                    }
                    AdapterReconstructionFacts::Loha {
                        projection,
                        factors,
                    } => {
                        // Shared reconstruct_loha_delta uses Tucker ONLY when both t1 and t2 exist;
                        // a lone t tensor is parsed/read but both sides still use their matmuls.
                        let tucker =
                            factors.contains_key("hada_t1") && factors.contains_key("hada_t2");
                        let mut factor = |a: &str, b: &str, t: &str| -> Result<TensorFacts> {
                            let wa = factors
                                .get(a)
                                .and_then(|k| file.sources.get(k))
                                .ok_or_else(|| {
                                    Error::Msg(format!("iris: {projection} missing {a}"))
                                })?;
                            let wb = factors
                                .get(b)
                                .and_then(|k| file.sources.get(k))
                                .ok_or_else(|| {
                                    Error::Msg(format!("iris: {projection} missing {b}"))
                                })?;
                            if let Some(t) = factors
                                .get(t)
                                .filter(|_| tucker)
                                .and_then(|k| file.sources.get(k))
                            {
                                graph.tucker(t, wa, wb, true)
                            } else {
                                graph.matmul(wa, wb)
                            }
                        };
                        let w1 = factor("hada_w1_a", "hada_w1_b", "hada_t1")?;
                        let w2 = factor("hada_w2_a", "hada_w2_b", "hada_t2")?;
                        let product = graph.multiply(&w1, &w2, "LoHa Hadamard multiply")?;
                        let delta = graph.scale(&product)?;
                        (projection, Some(delta))
                    }
                };
                if let Some(delta) = delta {
                    let expected=inventory.projections.get(path).ok_or_else(|| Error::Msg(format!("missing adapter projection {path}")))?.iter().find_map(|a|match a {AdapterFacts::Materialized {delta,..}=>Some(delta),_=>None})
                    .ok_or_else(||Error::Msg(format!("iris: materialized reconstruction has no delta inventory at {path}")))?;
                    let actual_elements = checked_elements(&delta.shape)?;
                    let expected_elements = checked_elements(&expected.shape)?;
                    if actual_elements != expected_elements {
                        return Err(Error::Msg(format!("iris: {path} reconstructed delta has {actual_elements} elements, target has {expected_elements}")));
                    }
                    // reconstruct_* ends with reshape(base_shape).as_dtype(compute). The reshape is
                    // a view when the factor path already produced a compatible contiguous order;
                    // Tucker/alias ordering can require one copy at this exact source site.
                    let reshaped = graph.output(
                        "delta reshape copy upper",
                        expected.shape.clone(),
                        delta.dtype,
                    )?;
                    graph.cast(&reshaped, expected.dtype)?;
                }
            }
        }

        Ok(())
    }

    /// Native reduction may promote an integer input without allocating a cast of that input.
    /// Keep input-copy width separate from output/partial-array width (metal/reduce.cpp).
    fn adapter_reduce(
        l: &mut NativeLedger,
        input_width: u64,
        output_width: u64,
        input: u64,
        output: u64,
        extent: u64,
    ) -> Result<()> {
        if l.backend == NativeBackend::Cpu {
            return l.array(output_width, output);
        }
        l.bytes(product(&[output_width, output])?.max(4))?;
        l.array(input_width, input)?; // conditional GeneralReduce normalization copy
        if output == 1 && input > 4096 {
            let rows = if product(&[input_width, input])? <= (1 << 26) {
                128
            } else {
                4096
            };
            l.array(output_width, rows)?;
        }
        if extent >= 1024 {
            let outer = if extent >= 32768 { 128 } else { 32 };
            l.array(output_width, product(&[outer, output])?)?;
        }
        if extent > 256 && output / 32 < 1024 {
            l.array(output_width, product(&[32, output])?)?;
        }
        Ok(())
    }

    fn cast_dtype(l: &mut NativeLedger, n: u64, from: Dtype, to: Dtype) -> Result<()> {
        if from != to {
            l.array(dtype_width(to), n)?;
        }
        Ok(())
    }

    /// AdaptableLinear::forward/forward_split -> apply_adapters -> Adapter::residual.
    /// Input is the compute-dtype UNPADDED logical x even when the packed base uses padding.
    /// Disabled adapters are skipped before any forward cast, scalar, product or sum is built.
    pub(super) fn forward(
        l: &mut NativeLedger,
        adapters: &[AdapterFacts],
        rows: u64,
        input: u64,
        output: u64,
        compute: Dtype,
    ) -> Result<()> {
        for adapter in adapters.iter().filter(|a| !a.disabled()) {
            let dtype = match adapter {
                AdapterFacts::Lora { a, b, .. } => {
                    if a.shape.len() != 2 || b.shape.len() != 2 {
                        return Err(overflow());
                    }
                    checked_elements(&a.shape)?;
                    checked_elements(&b.shape)?;
                    let rank = u64::try_from(a.shape[1]).map_err(|_| overflow())?;
                    if a.shape[0] as u64 != input
                        || b.shape[0] as u64 != rank
                        || b.shape[1] as u64 != output
                    {
                        return Err(Error::Unsupported(
                            "Iris LoRA residual width mismatch".into(),
                        ));
                    }
                    let first = Dtype::from_promoting_types(compute, a.dtype);
                    cast_dtype(l, product(&[rows, input])?, compute, first)?;
                    cast_dtype(l, checked_elements(&a.shape)?, a.dtype, first)?;
                    l.matmul(dtype_width(first), 1, rows, rank, input)?;
                    let second = Dtype::from_promoting_types(first, b.dtype);
                    cast_dtype(l, product(&[rows, rank])?, first, second)?;
                    cast_dtype(l, checked_elements(&b.shape)?, b.dtype, second)?;
                    l.matmul(dtype_width(second), 1, rows, output, rank)?;
                    second
                }
                AdapterFacts::Materialized { delta, .. } => {
                    if delta.shape.len() != 2
                        || delta.shape[0] as u64 != output
                        || delta.shape[1] as u64 != input
                    {
                        return Err(Error::Unsupported(
                            "Iris materialized adapter width mismatch".into(),
                        ));
                    }
                    cast_dtype(l, checked_elements(&delta.shape)?, delta.dtype, compute)?;
                    l.matmul(dtype_width(compute), 1, rows, output, input)?;
                    compute
                }
                AdapterFacts::Structured { w1, w2, .. } => {
                    if w1.shape.len() != 2 || w2.shape.len() != 2 {
                        return Err(overflow());
                    }
                    let (a, c, b, d) = (
                        w1.shape[0] as u64,
                        w1.shape[1] as u64,
                        w2.shape[0] as u64,
                        w2.shape[1] as u64,
                    );
                    if product(&[c, d])? != input || product(&[a, b])? != output {
                        return Err(Error::Unsupported(
                            "Iris structured adapter width mismatch".into(),
                        ));
                    }
                    cast_dtype(l, checked_elements(&w1.shape)?, w1.dtype, compute)?;
                    cast_dtype(l, checked_elements(&w2.shape)?, w2.dtype, compute)?;
                    l.reshape_copy(dtype_width(compute), product(&[rows, input])?)?;
                    l.matmul(dtype_width(compute), rows, a, d, c)?;
                    l.matmul(dtype_width(compute), rows, a, b, d)?;
                    l.reshape_copy(dtype_width(compute), product(&[rows, output])?)?;
                    // LokrFactors::residual returns directly: scale is baked into w2 at install.
                    l.op(dtype_width(compute), product(&[rows, output])?)?;
                    continue;
                }
            };
            // scalar(scale) is first an f32 Array, then cast to the residual's exact dtype.
            l.scalar(4)?;
            cast_dtype(l, 1, Dtype::Float32, dtype)?;
            l.op(dtype_width(dtype), product(&[rows, output])?)?;
            cast_dtype(l, product(&[rows, output])?, dtype, compute)?;
            l.op(dtype_width(compute), product(&[rows, output])?)?;
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn matrix(rows: i32, cols: i32, dtype: Dtype) -> TensorFacts {
            TensorFacts::matrix(rows, cols, dtype, false)
        }
        /// HOST: exact LoRA operand/scalar graph, including separate A/B matrix requests.
        #[test]
        fn lora_forward_prices_each_operand_and_disabled_is_zero_work() {
            let live = AdapterFacts::Lora {
                a: matrix(4, 2, Dtype::Float32),
                b: matrix(2, 3, Dtype::Float32),
                disabled: false,
            };
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            forward(&mut l, std::slice::from_ref(&live), 2, 4, 3, Dtype::Float32).unwrap();
            let expected = [16, 32, 32, 24, 16, 24, 4, 24, 24]
                .into_iter()
                .map(|n| backing_upper(n, 16_384).unwrap())
                .sum::<u64>();
            assert_eq!(l.total, expected);
            let disabled = AdapterFacts::Lora {
                a: matrix(4, 2, Dtype::Float32),
                b: matrix(2, 3, Dtype::Float32),
                disabled: true,
            };
            let mut off = NativeLedger::new(16_384, NativeBackend::Metal).unwrap();
            forward(&mut off, &[disabled], 2, 4, 3, Dtype::Float32).unwrap();
            assert_eq!(off.total, 0);
            let before = l.total;
            forward(&mut l, &[live], 2, 4, 3, Dtype::Float32).unwrap();
            assert_eq!(l.total, before.checked_mul(2).unwrap());
        }
        /// HOST: BF16+F16 promotes to f32 despite equal element widths, then narrows before add.
        #[test]
        fn mixed_lora_promotions_and_materialized_and_structured_are_counted() {
            let mixed = AdapterFacts::Lora {
                a: matrix(4, 2, Dtype::Float16),
                b: matrix(2, 3, Dtype::Float32),
                disabled: false,
            };
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            forward(&mut l, &[mixed], 2, 4, 3, Dtype::Bfloat16).unwrap();
            let expected = [32, 32, 16, 32, 32, 24, 16, 24, 4, 24, 12, 12]
                .into_iter()
                .map(|n| backing_upper(n, 16_384).unwrap())
                .sum::<u64>();
            assert_eq!(l.total, expected);
            for (adapter, requests) in [
                (
                    AdapterFacts::Materialized {
                        delta: matrix(3, 4, Dtype::Bfloat16),
                        disabled: false,
                    },
                    vec![48, 24, 32, 48, 4, 24, 24],
                ),
                (
                    AdapterFacts::Structured {
                        w1: matrix(1, 2, Dtype::Float32),
                        w2: matrix(3, 2, Dtype::Float32),
                        disabled: false,
                    },
                    vec![32, 16, 16, 32, 24, 16, 48, 24, 24],
                ),
            ] {
                let mut one = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
                forward(&mut one, &[adapter], 2, 4, 3, Dtype::Float32).unwrap();
                let exact = requests
                    .into_iter()
                    .map(|n| backing_upper(n, 16_384).unwrap())
                    .sum::<u64>();
                assert_eq!(one.total, exact);
            }
        }
        /// HOST: load-time Kronecker outputs/scaling remain charged; no release at materialize.
        #[test]
        fn reconstruction_graph_is_cumulative_and_backend_workspace_is_distinct() {
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            {
                let mut g = AdapterGraph { l: &mut l };
                let k = g
                    .kron(&matrix(2, 2, Dtype::Float32), &matrix(3, 2, Dtype::Float32))
                    .unwrap();
                assert_eq!(k.shape, [6, 4]);
                g.scale(&k).unwrap();
            }
            let expected = [96, 96, 4, 96]
                .into_iter()
                .map(|n| backing_upper(n, 16_384).unwrap())
                .sum::<u64>();
            assert_eq!(l.total, expected);
            let mut cpu = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            let mut metal = NativeLedger::new(16_384, NativeBackend::Metal).unwrap();
            for ledger in [&mut cpu, &mut metal] {
                AdapterGraph { l: ledger }
                    .matmul(
                        &matrix(2, 4096, Dtype::Float32),
                        &matrix(4096, 2, Dtype::Float32),
                    )
                    .unwrap();
            }
            assert!(metal.total > cpu.total);
        }
        /// HOST: real inventory traversal charges each file operand, alpha cast, host scalar and
        /// alpha-fold output even for an adapter disabled only on the forward path.
        #[test]
        fn installed_file_load_prices_raw_alpha_fold_and_retained_outputs() {
            use crate::adapters::AdapterFileFacts;
            use mlx_gen::adapters::loader::UnfusedAdapterFormat;
            let inventory = AdapterHeaderInventory {
                projections: BTreeMap::from([(
                    "p".into(),
                    vec![AdapterFacts::Lora {
                        a: matrix(4, 2, Dtype::Bfloat16),
                        b: matrix(2, 3, Dtype::Float32),
                        disabled: true,
                    }],
                )]),
                files: vec![AdapterFileFacts {
                    path: std::path::PathBuf::new(), // metadata-only: load never opens this path
                    format: UnfusedAdapterFormat::PeftLora,
                    sources: BTreeMap::from([
                        ("a".into(), matrix(2, 4, Dtype::Bfloat16)),
                        ("b".into(), matrix(3, 2, Dtype::Bfloat16)),
                        (
                            "alpha".into(),
                            contiguous(vec![1], Dtype::Bfloat16).unwrap(),
                        ),
                    ]),
                    scalar_reads: BTreeSet::from(["alpha".into()]),
                    reconstructions: vec![AdapterReconstructionFacts::Lora {
                        projection: "p".into(),
                        down: "a".into(),
                        up: "b".into(),
                        alpha_fold: true,
                    }],
                }],
            };
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            load(&mut l, &inventory, 2).unwrap();
            let expected = [4, 4, 16, 12, 2, 4, 4, 24, 24]
                .into_iter()
                .map(|n| backing_upper(n, 16_384).unwrap())
                .sum::<u64>();
            assert_eq!(l.total, expected);
        }
        /// HOST: actual greedy Tucker contraction arithmetic and its naive reduction fallback
        /// use per-operand native requests, not the old adapter scratch maxima.
        #[test]
        fn tucker_contraction_and_naive_reduction_price_actual_requests() {
            let mut totals = Vec::new();
            for loha_order in [false, true] {
                let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
                let out = AdapterGraph { l: &mut l }
                    .tucker(
                        &contiguous(vec![1, 1, 1, 1], Dtype::Float32).unwrap(),
                        &matrix(1, 2, Dtype::Float32),
                        &matrix(1, 3, Dtype::Float32),
                        loha_order,
                    )
                    .unwrap();
                assert_eq!(out.shape, [2, 3, 1, 1]);
                let exact = [4, 8, 8, 4, 8, 12, 8, 24, 12, 8]
                    .into_iter()
                    .map(|n| backing_upper(n, 16_384).unwrap())
                    .sum::<u64>();
                assert_eq!(l.total, exact);
                totals.push(l.total);
            }
            assert_eq!(totals[0], totals[1]);
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            let terms = [
                Term {
                    axes: vec!['i'],
                    tensor: contiguous(vec![2], Dtype::Float32).unwrap(),
                },
                Term {
                    axes: vec!['p'],
                    tensor: contiguous(vec![3], Dtype::Float32).unwrap(),
                },
            ];
            let out = AdapterGraph { l: &mut l }
                .naive_einsum(&terms, &['p'], &BTreeMap::from([('i', 2), ('p', 3)]))
                .unwrap();
            assert_eq!(out.tensor.shape, [3]);
            assert_eq!(
                l.total,
                backing_upper(24, 16_384)
                    .unwrap()
                    .checked_add(backing_upper(12, 16_384).unwrap())
                    .unwrap()
            );
        }
        /// HOST: reconstruction dimensions fail closed, without tensor construction/evaluation.
        #[test]
        fn adapter_shape_arithmetic_fails_closed() {
            assert!(checked_elements(&[-1, 2]).is_err());
            assert!(checked_elements(&[i32::MAX, i32::MAX, i32::MAX]).is_err());
            let mut l = NativeLedger::new(16_384, NativeBackend::Cpu).unwrap();
            let bad = AdapterFacts::Materialized {
                delta: matrix(3, 5, Dtype::Float32),
                disabled: false,
            };
            assert!(forward(&mut l, &[bad], 2, 4, 3, Dtype::Float32).is_err());
        }
    }
}

/// Repeat one source-identical call graph, applying U to each operand BEFORE repetition. The
/// first call can create a position cache; every subsequent call is the same steady graph. This
/// is exact call multiplicity, not a global allocation multiplier or an assumed scratch budget.
fn repeated_native_calls(
    l: &mut NativeLedger,
    calls: u64,
    mut call: impl FnMut(&mut NativeLedger, bool) -> Result<()>,
) -> Result<()> {
    if calls == 0 {
        return Err(Error::Unsupported(
            "Iris restoration has no tile forwards".into(),
        ));
    }
    call(l, true)?;
    if calls > 1 {
        let mut steady = NativeLedger::new(l.page, l.backend)?;
        call(&mut steady, false)?;
        l.total = plus(l.total, product(&[steady.total, calls - 1])?)?;
    }
    Ok(())
}

fn restoration_upper(req: &IosPricingRequest<'_>, l: &mut NativeLedger) -> Result<()> {
    restoration_upper_replay(req, None, l)
}

/// restoration.rs:150-196,242-264: export/prompt load, then one upload+time+DiT per actual tile.
/// All image/canvas/fusion/wavelet/RGB assembly is the neutral HOST driver, not an MLX graph.
fn restoration_upper_replay(
    req: &IosPricingRequest<'_>,
    token: Option<&RestorationToken>,
    l: &mut NativeLedger,
) -> Result<()> {
    req.validate_work_context()?;
    if req.provider != crate::restoration::MODEL_ID {
        return Err(Error::Unsupported(
            "not an Iris restoration native replay".into(),
        ));
    }
    crate::memory_strategy::refuse_unloadable_quant(crate::restoration::MODEL_ID, req.spec)
        .map_err(Error::from)?;
    for (field, set) in [
        ("adapters", !req.spec.adapters.is_empty()),
        ("text_encoder", req.spec.text_encoder.is_some()),
        ("control", req.spec.control.is_some()),
        ("extra_controls", !req.spec.extra_controls.is_empty()),
        ("ip_adapter", req.spec.ip_adapter.is_some()),
        ("pid", req.spec.pid.is_some()),
        ("identity", req.spec.identity.is_some()),
    ] {
        if set {
            return Err(Error::Unsupported(format!(
                "{}: LoadSpec::{field} is not supported by the Iris-3B restoration route",
                crate::restoration::MODEL_ID
            )));
        }
    }
    let resources = RestorationResources::from_spec(req.spec, crate::restoration::MODEL_ID)
        .map_err(Error::from)?;
    let config = &resources.export.config;
    let plan = restoration_priced_plan(req, &resources, token)?;
    crate::packed::refuse_dense_requested_as_packed(
        crate::restoration::MODEL_ID,
        &resources.export.weights_path(),
        req.spec.quantize,
    )?;
    let artifact = crate::packed::artifact_tier(&resources.export.weights_path())?;
    if artifact.is_some() && req.spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(format!(
            "iris: {} is a packed {artifact:?} artifact, which runs under the bf16 compute policy",
            resources.export.weights_path().display()
        )));
    }
    let streamed = req.spec.load_shape == LoadShape::DeferredMaterialization;
    let bb = priced_backbone(
        config,
        &resources.export.weights_path(),
        "",
        req.spec.precision,
        req.spec.quantize,
        streamed,
        config.model.in_channels as u64,
        Pricing::default(),
    )?;
    let raw: HashMap<String, Header> = crate::stream::headers(&resources.export.weights_path())?
        .into_iter()
        .map(|h| (h.name.clone(), h))
        .collect();
    load_backbone_set(
        l,
        &raw,
        bb.facts.cw,
        "",
        |name| !streamed || Stack::of(name).is_none(),
        None,
    )?;
    let prompt = crate::stream::headers(&resources.export.empty_prompt_path())?;
    let embeddings = prompt
        .iter()
        .find(|h| h.name == "embeddings")
        .ok_or_else(|| Error::MissingTensor("empty-prompt embeddings".into()))?;
    // EmptyPrompt::read uses host safetensors decoding; only this f32 from_slice is native.
    l.array(4, header_elements(embeddings)?)?;
    let (h, w) = if plan.single_pass() {
        (u64::from(plan.padded.height), u64::from(plan.padded.width))
    } else {
        (u64::from(plan.tile), u64::from(plan.tile))
    };
    repeated_native_calls(
        l,
        plan.forward_count().map_err(Error::from)?,
        |ledger, first| {
            ledger.array(4, p!(3, h, w))?; // velocity_body tile upload
            ledger.array(4, 1)?; // model time from_slice
            Graph {
                l: &mut *ledger,
                bb: &bb.facts,
            }
            .forward(
                1,
                h,
                w,
                ForwardMode {
                    by_row: false,
                    first,
                    streamed,
                },
                mlx_gen::AttentionPlan::UNBOUNDED,
            )
            // velocity's f32 as_dtype is identity; its flatten/readback creates no new native backing.
        },
    )
}

/// Native recipe replay only. This does NOT itself grant a restoration certificate or combine
/// native Active with host payload/process heap; the parent bounder owns that separate law.
pub(super) fn cumulative_native_upper_with_restoration_token(
    req: &IosPricingRequest<'_>,
    token: &RestorationToken,
    page: u64,
    backend: NativeBackend,
) -> Result<u64> {
    if backend == NativeBackend::Metal && std::env::var_os("MLX_SDPA_BLOCKS").is_some() {
        return Err(Error::Unsupported("Iris native certificate: MLX_SDPA_BLOCKS override is not bound by any execution witness".into()));
    }
    let mut l = NativeLedger::new(page, backend)?;
    restoration_upper_replay(req, Some(token), &mut l)?;
    Ok(l.total)
}

/// Opaque reserved proof input. No constructor is exposed while aa011's COMPLETE cold
/// tokenizer/regex construction and first-search/cache allocation upper remains unproved.
/// A future source-owned proof, not a caller-provided allowance, must construct this term.
pub(super) struct TokenizerHostBound {
    cumulative_requested_bytes: u64,
}

/// Opaque reserved proof input for complete parser/key/container requested layouts, including
/// EACH repeated streamed header/Weights parse, config validator formatting, and scoped-thread
/// requested bookkeeping layouts. No constructor is exposed without source proof. RawVec growth
/// alone is NOT a bound for serde scratch, HashMap/BTree nodes or std thread-spawn packets.
pub(super) struct MetadataHostBound {
    cumulative_requested_bytes: u64,
}

const TOKENIZER_HOST_GAP: &str =
    "tokenizer/regex cold-construction and first-search host allocation upper unproved (aa011)";
const METADATA_HOST_GAP: &str =
    "Iris metadata host allocation upper unproved: exact libc0.2.183 Apple cfg-chain / pthread_mutex_t requested layout missing";

fn required_tokenizer_host_bytes(
    provider: &str,
    bound: Option<&TokenizerHostBound>,
) -> Result<u64> {
    if provider == crate::model::MODEL_ID {
        bound
            .map(|b| b.cumulative_requested_bytes)
            .ok_or_else(|| Error::Unsupported(TOKENIZER_HOST_GAP.into()))
    } else {
        Ok(0)
    } // depth/restoration load a shipped EmptyPrompt and NEVER tokenize
}

/// Finite complete requested HOST-layout bound only when every input term is proved. No
/// request-scaled parser/tokenizer/plan/container allocation may hide in a fixed empirical reserve.
/// Today generation fails at aa011, and depth/restoration retain the named exact-version
/// Apple Mutex ABI gap. Recovered parser/table components are independently source-priced below.
/// The plane/tap/axis payload recipes below remain available separately, but are NOT certificates.
pub(super) fn host_growth_upper(
    req: &IosPricingRequest<'_>,
    callback: Option<mlx_gen::ios_pricing::IosCallbackHost>,
) -> Result<u64> {
    host_growth_upper_with_bounds(req, None, None, callback)
}

pub(super) fn host_growth_upper_with_bounds(
    req: &IosPricingRequest<'_>,
    metadata: Option<&MetadataHostBound>,
    tokenizer: Option<&TokenizerHostBound>,
    callback: Option<mlx_gen::ios_pricing::IosCallbackHost>,
) -> Result<u64> {
    host_growth_upper_checked(req, metadata, tokenizer, callback, |_| Ok(()))
}

/// Capacity seam for the SDK caller: supply runtime_ios::media::source_metadata_capacity here
/// BEFORE this helper parses configs/tokenizer files or constructs new source-context payloads.
/// runtime-ios/src/lib.rs:217-227,338-344 checks requested bytes at a fresh boundary; it does NOT
/// itself prove a parser/tokenizer malloc or process bound. The provider cannot depend upward on
/// the runtime bundle. Passing the checked byte count never grants a physical-charge assumption.
pub(super) fn host_growth_upper_checked(
    req: &IosPricingRequest<'_>,
    metadata: Option<&MetadataHostBound>,
    tokenizer: Option<&TokenizerHostBound>,
    callback: Option<mlx_gen::ios_pricing::IosCallbackHost>,
    mut source_metadata_capacity: impl FnMut(u64) -> Result<()>,
) -> Result<u64> {
    req.validate_work_context()?;
    let callback = callback.ok_or_else(|| Error::Unsupported(CALLBACK_HOST_GAP.into()))?;
    let tokenizer = required_tokenizer_host_bytes(req.provider, tokenizer)?;
    let metadata = metadata
        .ok_or_else(|| Error::Unsupported(METADATA_HOST_GAP.into()))?
        .cumulative_requested_bytes;
    let source_context = plus(metadata, tokenizer)?;
    source_metadata_capacity(source_context)?;
    let payload = match req.provider {
        crate::restoration::MODEL_ID => restoration_host_upper(req)?,
        crate::depth::DEPTH_MODEL_ID => host_payload::depth(req)?,
        crate::model::MODEL_ID => host_payload::generation(req)?,
        _ => {
            return Err(Error::Unsupported(
                "Iris has no host layout recipe for this provider".into(),
            ))
        }
    };
    let callbacks = match callback {
        mlx_gen::ios_pricing::IosCallbackHost::NonAllocating => 0,
        mlx_gen::ios_pricing::IosCallbackHost::Bounded {
            fixed_bytes,
            per_progress_event_bytes,
            per_preview_event_bytes,
            per_quantum_mark_bytes,
            per_quantum_label_byte_bytes,
        } => callback_event_upper(req)?.priced(
            fixed_bytes,
            per_progress_event_bytes,
            per_preview_event_bytes,
            per_quantum_mark_bytes,
            per_quantum_label_byte_bytes,
        )?,
    };
    plus(plus(source_context, payload)?, callbacks)
}

/// Known requested image/table/axis/plan payload upper ONLY. It excludes unproved parser/key and
/// tokenizer terms and MUST NOT be substituted for host_growth_upper in a live certificate.
pub(super) fn host_payload_growth_upper(req: &IosPricingRequest<'_>) -> Result<u64> {
    req.validate_work_context()?;
    required_tokenizer_host_bytes(req.provider, None)?; // refuse BEFORE invoking an unproved tokenizer
    match req.provider {
        crate::restoration::MODEL_ID => restoration_host_upper(req),
        crate::depth::DEPTH_MODEL_ID => host_payload::depth(req),
        _ => Err(Error::Unsupported(
            "Iris has no supplied host payload context".into(),
        )),
    }
}

/// Known restoration producer PAYLOAD only, not complete requested metadata or process heap.
/// Live certificates must use host_growth_upper and its required complete MetadataHostBound.
pub(super) fn restoration_host_upper(req: &IosPricingRequest<'_>) -> Result<u64> {
    host_payload::restoration(req, None)
}

/// Payload-only replay counterpart; the same missing-metadata qualification applies.
pub(super) fn restoration_host_upper_with_token(
    req: &IosPricingRequest<'_>,
    token: &RestorationToken,
) -> Result<u64> {
    host_payload::restoration(req, Some(token))
}

mod host_payload {
    use super::*;
    use mlx_gen::gen_core::iris::restoration::{Dims, RestorationPlanMetadata};

    /// TOOLCHAIN-POLICY ASSUMPTION, NOT MLX/Iris source authority: RawVec's amortized growth
    /// requests new capacity <= max(2*old, required, min_non_zero_cap), min_non_zero_cap <= 8.
    /// Retaining all old allocations yields a cumulative capacity upper (4*len+8) elements.
    /// Used ONLY at source growth-policy sites: fs::read bytes, Result<Vec<_>> collections and
    /// scoped-worker/result containers. Allocator overhead is NOT included or claimed bounded.
    const ASSUMED_GROWTH_FACTOR: u64 = 4;
    const ASSUMED_MIN_CAPACITY_SLACK: u64 = 8;

    #[derive(Default)]
    struct Payload {
        bytes: u64,
    }
    impl Payload {
        fn bytes(&mut self, n: u64) -> Result<()> {
            self.bytes = plus(self.bytes, n)?;
            Ok(())
        }
        fn vec<T>(&mut self, n: u64) -> Result<()> {
            self.bytes(product(&[std::mem::size_of::<T>() as u64, n])?)
        }
        fn growth<T>(&mut self, n: u64) -> Result<()> {
            if n == 0 {
                return Ok(());
            }
            let capacity = plus(
                product(&[ASSUMED_GROWTH_FACTOR, n])?,
                ASSUMED_MIN_CAPACITY_SLACK,
            )?;
            self.vec::<T>(capacity)
        }
        fn repeat(&mut self, n: u64, one: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
            if n == 0 {
                return Ok(());
            }
            let mut call = Self::default();
            one(&mut call)?;
            self.bytes(product(&[n, call.bytes])?)
        }
    }
    fn pixels(d: Dims) -> Result<u64> {
        product(&[u64::from(d.width), u64::from(d.height)])
    }
    fn rgb(p: &mut Payload, d: Dims) -> Result<()> {
        p.vec::<f32>(product(&[3, pixels(d)?])?)
    }

    /// imageops.rs:74-145: each axis creates both f64 and quantized i64 coefficient tables.
    /// Each inner Vec has explicit/ExactSize capacity; the source's clipped support-window bound
    /// applies separately to each coefficient row (same rule as resize_lanczos_u8_host_bytes).
    fn lanczos_axis(p: &mut Payload, input: u32, output: u32) -> Result<()> {
        if input == 0 || output == 0 {
            return Err(overflow());
        }
        let support = 3.0 * (f64::from(input) / f64::from(output)).max(1.0);
        let taps = plus((2.0 * support).ceil() as u64, 2)?.min(u64::from(input));
        p.vec::<(usize, Vec<f64>)>(u64::from(output))?;
        p.vec::<(usize, Vec<i64>)>(u64::from(output))?;
        p.vec::<f64>(product(&[taps, u64::from(output)])?)?;
        p.vec::<i64>(product(&[taps, u64::from(output)])?)
    }
    /// imageops.rs:162-231: two axis table sets, one horizontal plane, one output plane.
    /// Sum ALL producers; do not reuse the source helper's coincident peak as a cumulative price.
    fn lanczos(p: &mut Payload, source: Dims, output: Dims) -> Result<()> {
        lanczos_axis(p, source.width, output.width)?;
        lanczos_axis(p, source.height, output.height)?;
        p.vec::<f32>(product(&[
            3,
            u64::from(source.height),
            u64::from(output.width),
        ])?)?;
        rgb(p, output)
    }
    /// restoration.rs:849-882: one outer tuple Vec and clipped-support f32 weights per axis row.
    fn aa_axis(p: &mut Payload, input: u32, output: u32) -> Result<()> {
        if input == 0 || output == 0 {
            return Err(overflow());
        }
        let scale = input as f32 / output as f32;
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let taps = plus(product(&[support.ceil() as u64, 2])?, 1)?.min(u64::from(input));
        p.vec::<(usize, Vec<f32>)>(u64::from(output))?;
        p.vec::<f32>(product(&[taps, u64::from(output)])?)
    }
    /// restoration.rs:941-982: unconditional clone, then every changed horizontal/vertical pass.
    fn aa_resize(p: &mut Payload, source: Dims, output: Dims) -> Result<()> {
        rgb(p, source)?;
        if source.width != output.width {
            aa_axis(p, source.width, output.width)?;
            p.vec::<f32>(product(&[
                3,
                u64::from(source.height),
                u64::from(output.width),
            ])?)?;
        }
        if source.height != output.height {
            aa_axis(p, source.height, output.height)?;
            rgb(p, output)?;
        }
        Ok(())
    }
    /// tables.rs:44-90 + nn.rs:511-519: frequencies/coordinates/angles and BOTH host cos/sin Vecs.
    fn grid_rope(p: &mut Payload, h: u64, w: u64, head: u64) -> Result<()> {
        let pairs = product(&[2, head / 4])?;
        p.vec::<f32>(head / 4)?;
        p.vec::<f32>(h)?;
        p.vec::<f32>(w)?;
        p.repeat(3, |one| one.vec::<f32>(product(&[h, w, pairs])?))
    }
    /// Actual request-scaled host table/mask producers inside ONE DiT forward. Scalar tensor
    /// constructors use native allocator backing and are already charged in NativeLedger.
    fn backbone(p: &mut Payload, m: &IrisModel, b: u64, h: u64, w: u64, first: bool) -> Result<()> {
        let patch = m.patch_size as u64;
        if patch == 0 || m.num_heads == 0 || m.pixel.num_heads == 0 {
            return Err(overflow());
        }
        let (hp, wp) = (h / patch, w / patch);
        let t = m.text_len as u64;
        p.vec::<f32>(mlx_gen::gen_core::iris::TIMESTEP_FREQUENCY_DIM as u64 / 2)?;
        p.vec::<f32>(product(&[b, t, t])?)?; // key_padding_additive vec![0;B*T*T]
        let head = (m.hidden_size / m.num_heads) as u64;
        grid_rope(p, hp, wp, head)?;
        p.vec::<f32>(head / 2)?; // text rope inverse frequencies
        p.repeat(3, |one| one.vec::<f32>(product(&[t, head / 2])?))?; // angles/cos/sin
        if first {
            p.vec::<f64>(m.pixel.hidden_size as u64 / 4)?; // pixel_sincos omega
            p.vec::<f32>(product(&[h, w, m.pixel.hidden_size as u64])?)?;
        }
        grid_rope(
            p,
            hp,
            wp,
            (m.pixel.attn_hidden_size / m.pixel.num_heads) as u64,
        )
    }
    /// downstream.rs:280-326: file bytes, decoded f32 embeddings and i32 mask. Parser/map/name
    /// bookkeeping is a SEPARATE UNPROVED MetadataHostBound, never a fixed-overhead credit.
    fn empty_prompt(p: &mut Payload, export: &TaskExport, outer_mask: bool) -> Result<()> {
        let path = export.empty_prompt_path();
        let len = std::fs::metadata(&path)
            .map_err(|e| Error::Msg(e.to_string()))?
            .len();
        p.growth::<u8>(len)?; // TOOLCHAIN assumption for fs::read/Read::read_to_end
        let headers = crate::stream::headers(&path)?;
        let emb = headers
            .iter()
            .find(|h| h.name == "embeddings")
            .ok_or_else(|| Error::MissingTensor("embeddings".into()))?;
        p.vec::<f32>(header_elements(emb)?)?;
        p.vec::<i32>(export.config.model.text_len as u64)?;
        p.vec::<i32>(4)?; // load's shape.iter().map(...).collect
        if outer_mask {
            p.vec::<Vec<i32>>(1)?;
        }
        Ok(())
    }
    /// Full neutral restoration driver payload graph, restoration.rs:1193-1344. This also covers
    /// every owner in restoration_decode_host_bytes/host_plane_bytes_metadata, but NEVER adds a
    /// live-set/peak value again as if it were a newly allocated plane.
    fn restoration_driver(
        p: &mut Payload,
        plan: &RestorationPlanMetadata,
        caller: u64,
    ) -> Result<()> {
        p.bytes(caller)?;
        p.bytes(plan.axis_bytes().map_err(Error::from)?)?; // collect's explicit axis capacities
        if plan.budget_applied {
            lanczos(p, plan.source, plan.input)?;
            p.vec::<u8>(product(&[3, pixels(plan.input)?])?)?; // fit_budget_image f32->u8 collect
        } else {
            p.vec::<u8>(product(&[3, pixels(plan.source)?])?)?;
        } // image.clone
        rgb(p, plan.input)?; // Planes::from_rgb8
        p.vec::<([usize; 4], [f32; 4])>(plus(
            u64::from(plan.output.width),
            u64::from(plan.output.height),
        )?)?;
        rgb(p, plan.output)?; // bicubic_scale_factor reference
        if plan.enlarged {
            aa_resize(p, plan.output, plan.processing)?;
        } else {
            rgb(p, plan.output)?;
        } // reference.clone
        rgb(p, plan.padded)?; // normalized padded input
        let tiles = plan.forward_count().map_err(Error::from)?;
        if plan.single_pass() {
            rgb(p, plan.padded)?; // velocity readback to_vec
            rgb(p, plan.padded)?; // restore_tile's x-sigma*v collect
        } else {
            p.repeat(2, |one| one.vec::<f32>(u64::from(plan.tile)))?; // Gaussian axes
            p.vec::<f32>(product(&[u64::from(plan.tile), u64::from(plan.tile)])?)?;
            rgb(p, plan.padded)?; // fused canvas
            p.vec::<f32>(pixels(plan.padded)?)?; // fusion weights
            p.repeat(tiles, |one| {
                one.repeat(3, |buffer| {
                    buffer.vec::<f32>(product(&[3, u64::from(plan.tile), u64::from(plan.tile)])?)
                })
                // source crop, velocity readback, restored crop are DISTINCT Vecs per tile
            })?;
        }
        rgb(p, plan.processing)?; // fused.crop
        if plan.enlarged {
            aa_resize(p, plan.processing, plan.output)?;
        }
        if plan.color_fix {
            let o = pixels(plan.output)?;
            let dilations = mlx_gen::gen_core::iris::restoration::COLOR_FIX_DILATIONS.len() as u64;
            p.repeat(3, |worker| {
                // EACH split has high+low and one blur output PER dilation, no release credit.
                worker.repeat(2, |split| {
                    split.repeat(plus(2, dilations)?, |plane| plane.vec::<f32>(o))
                })?;
                worker.vec::<f32>(o) // high+low output for this channel
            })?;
            // TOOLCHAIN containers: scoped handles, then fallible worker-result Vec collection.
            p.growth::<std::thread::ScopedJoinHandle<'static, mlx_gen::gen_core::Result<Vec<f32>>>>(3)?;
            p.growth::<Vec<f32>>(3)?;
            p.bytes(provider_scoped_thread_heap_upper(3)?)?;
            rgb(p, plan.output)?; // planes.concat
        }
        p.vec::<u8>(product(&[3, pixels(plan.output)?])?) // Planes::to_rgb8, host only
    }

    pub(super) fn restoration(
        req: &IosPricingRequest<'_>,
        token: Option<&RestorationToken>,
    ) -> Result<u64> {
        if req.provider != crate::restoration::MODEL_ID {
            return Err(Error::Unsupported(
                "not an Iris restoration host replay".into(),
            ));
        }
        req.validate_work_context()?;
        let resources =
            RestorationResources::from_spec(req.spec, req.provider).map_err(Error::from)?;
        let plan = restoration_priced_plan(req, &resources, token)?;
        let caller = match token {
            Some(token) => token.wire()?.source_capacity_bytes,
            None => req
                .transform
                .ok_or_else(|| Error::Unsupported("missing restoration host source".into()))?
                .image
                .pixels
                .capacity() as u64,
        };
        let mut p = Payload::default();
        empty_prompt(&mut p, &resources.export, true)?;
        restoration_driver(&mut p, &plan, caller)?;
        let (h, w) = if plan.single_pass() {
            (u64::from(plan.padded.height), u64::from(plan.padded.width))
        } else {
            (u64::from(plan.tile), u64::from(plan.tile))
        };
        let tiles = plan.forward_count().map_err(Error::from)?;
        // velocity_body's &[f32;1] model time is a stack literal, NOT a host Vec.
        backbone(&mut p, &resources.export.config.model, 1, h, w, true)?;
        if tiles > 1 {
            p.repeat(tiles - 1, |one| {
                backbone(one, &resources.export.config.model, 1, h, w, false)
            })?;
        }
        Ok(p.bytes)
    }

    fn depth_input_planes(p: &mut Payload, source: Dims, model: Dims, resized: bool) -> Result<()> {
        if resized {
            lanczos(p, source, model)?;
        } else {
            rgb(p, model)?;
        } // HWC u8->f32 exact-size collect
        rgb(p, model) // CHW vec![0;3HW]
    }
    /// model.rs:483-489,523-539 + pipeline.rs:310-331 + image.rs:108-119.
    fn generation_outputs(p: &mut Payload, b: u64, count: u64, h: u64, w: u64) -> Result<()> {
        p.vec::<mlx_rs::Array>(product(&[b, count])?)?;
        p.vec::<mlx_gen::Image>(product(&[b, count])?)?;
        p.repeat(count, |batch| {
            if b == 1 {
                batch.vec::<mlx_gen::Image>(1)?;
            } else {
                batch.growth::<mlx_gen::Image>(b)?;
            }
            batch.repeat(b, |row| row.vec::<u8>(product(&[3, h, w])?))
        })
    }

    pub(super) fn depth(req: &IosPricingRequest<'_>) -> Result<u64> {
        let actual = req.depth.ok_or_else(|| {
            Error::Unsupported("depth host payload needs actual source/resolution".into())
        })?;
        let export = TaskExport::from_spec(req.spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
            .map_err(Error::from)?;
        let plan = depth_input_plan(req.spec, actual)?;
        let source = Dims::new(plan.source_width, plan.source_height);
        let model = Dims::new(plan.model_width, plan.model_height);
        let mut p = Payload::default();
        p.bytes(actual.image.pixels.capacity() as u64)?;
        empty_prompt(&mut p, &export, false)?;
        depth_input_planes(&mut p, source, model, plan.resized())?;
        p.vec::<f32>(1)?; // depth_input_body vec![t_model;b], b=1
        p.vec::<i32>(export.config.model.text_len as u64)?;
        p.vec::<Vec<i32>>(1)?; // depth_forward_with mask clone + outer mask
        backbone(
            &mut p,
            &export.config.model,
            1,
            u64::from(plan.model_height),
            u64::from(plan.model_width),
            true,
        )?;
        p.vec::<f32>(pixels(model)?)?; // native map readback to_vec
        if plan.resized() {
            p.vec::<(usize, usize, f32, f32)>(plus(
                u64::from(plan.source_height),
                u64::from(plan.source_width),
            )?)?;
            p.vec::<f32>(pixels(source)?)?; // interpolate_bilinear with_capacity
        }
        Ok(p.bytes)
    }

    pub(super) fn generation(req: &IosPricingRequest<'_>) -> Result<u64> {
        // A sampling-only replay records no raw caption lengths. Source wrapper encode_ids
        // copies ALL ids before truncating. It cannot be bounded from token_window alone.
        let actual = req.generation.ok_or_else(|| Error::Unsupported(
            "Iris host payload needs actual generation captions: sampling replay omits pre-truncation tokenizer-id payload lengths".into()))?;
        let resources = GenerationResources::from_spec(req.spec, crate::model::MODEL_ID)
            .map_err(Error::from)?;
        let config = IrisConfig::from_dir(&resources.backbone_dir).map_err(Error::from)?;
        let params = resolved_generation_controls(&config, actual, actual.seed.unwrap_or(0))?;
        let plan = params.plan_metadata().map_err(Error::from)?;
        let b = params.prompts.len() as u64;
        let count = u64::from(actual.count);
        let (h, w) = (u64::from(params.height), u64::from(params.width));
        let max = config.text_encoder.max_length as u64;
        let mut p = Payload::default();
        // Actual caller capacities once, plus GenerationParams::resolve's cloned prompt payloads.
        p.bytes(actual.prompt.capacity() as u64)?;
        p.vec::<String>(actual.prompt_batch.capacity() as u64)?;
        for prompt in &actual.prompt_batch {
            p.bytes(prompt.capacity() as u64)?;
        }
        if let Some(negative) = &actual.negative_prompt {
            p.bytes(negative.capacity() as u64)?;
        }
        // model::generate_impl first validate_impl -> validate_generation_request resolves
        // controls once, then resolves them again for execution (model.rs:436-450,553-560;
        // gen-core/iris.rs:1411-1413). Both cloned payloads remain charged cumulatively.
        p.repeat(2, |resolved| {
            resolved.vec::<String>(b)?;
            for prompt in &params.prompts {
                resolved.vec::<u8>(prompt.len() as u64)?;
            }
            resolved.vec::<u8>(params.negative_prompt.len() as u64)
        })?;
        let tokenizer = mlx_gen::gen_core::tokenizer::TextTokenizer::from_file(
            resources.text_encoder_dir.join("tokenizer.json"),
            mlx_gen::gen_core::tokenizer::TokenizerConfig {
                max_length: usize::MAX,
                pad_token_id: 0,
                chat_template: mlx_gen::gen_core::tokenizer::ChatTemplate::None,
                pad_to_max_length: false,
            },
        )
        .map_err(Error::from)?;
        let prefix = tokenizer
            .encode_ids(PROMPT_PREFIX, false)
            .map_err(Error::from)?;
        let suffix = tokenizer
            .encode_ids(mlx_gen::gen_core::iris::PROMPT_SUFFIX, false)
            .map_err(Error::from)?;
        let staged =
            req.execution.stage_residency || req.spec.offload_policy == OffloadPolicy::Sequential;
        let loads = plus(
            1,
            u64::from(staged && req.spec.offload_policy == OffloadPolicy::Resident),
        )?;
        let cfg = mlx_llm::ModelConfig::from_dir(&resources.text_encoder_dir)
            .map_err(|e| Error::Msg(e.to_string()))?;
        let head = u64::try_from(cfg.head_dim).map_err(|_| overflow())?;
        p.repeat(loads, |load| {
            load.vec::<i32>(prefix.len() as u64)?;
            load.vec::<i32>(suffix.len() as u64)?;
            load.vec::<f32>(head / 2)?; // Rope::standard inverse frequencies retained on host
            load.vec::<f32>(head) // Rope::new expanded frequency row before native upload
        })?;
        let mut warnings = 0u64;
        for prompt in params
            .prompts
            .iter()
            .map(String::as_str)
            .chain(params.uses_cfg().then_some(params.negative_prompt.as_str()))
        {
            let caption = tokenizer.encode_ids(prompt, false).map_err(Error::from)?;
            let window = mlx_gen::gen_core::iris::assemble_window(
                &prefix,
                &caption,
                &suffix,
                config.text_encoder.max_length,
            )
            .map_err(Error::from)?;
            p.vec::<i32>(caption.len() as u64)?; // wrapper encode_ids map/collect, BEFORE truncation
            p.vec::<i32>(window.input_ids.len() as u64)?; // explicit assemble_window capacity
            p.vec::<i32>(max)?; // window.mask vec![0;max]
            p.vec::<i32>(max)?; // TextConditioning.mask clone
            if req.spec.load_shape != LoadShape::DeferredMaterialization {
                p.vec::<u64>(plus(cfg.num_layers as u64, 1)?)?; // resident iOS layer-gate scratch Vec
            }
            if matches!(
                params.caption_overflow,
                mlx_gen::gen_core::CaptionOverflowPolicy::Warn
            ) {
                if let Some(warning) = mlx_gen::gen_core::iris::caption_overflow_warning(
                    &window,
                    config.text_encoder.max_length,
                    suffix.len(),
                ) {
                    p.growth::<u8>(warning.len() as u64)?; // TOOLCHAIN formatter String growth
                    warnings = plus(warnings, 1)?;
                }
            }
        }
        p.growth::<String>(warnings)?; // pipeline::encode_in warnings.extend
        p.growth::<mlx_gen::gen_core::GenerationWarning>(warnings)?; // final warning report collect
        p.repeat(warnings, |warning| {
            warning.vec::<u8>("caption_truncated".len() as u64)
        })?;
        generation_outputs(&mut p, b, count, h, w)?;
        let mut first = true;
        for _ in 0..count {
            p.bytes(plan.host_bytes().map_err(Error::from)?)?; // DpmStep plan + sigma Vec, not a peak
            p.vec::<u64>(b)?; // independent row seeds
            p.growth::<mlx_rs::Array>(b)?; // noise_batch fallible row collection
            p.vec::<&mlx_rs::Array>(b)?; // noise_batch refs
            p.vec::<Vec<i32>>(b)?; // cond_mask outer Vec
            p.vec::<i32>(product(&[b, max])?)?; // one cloned mask per prompt
            if params.uses_cfg() {
                p.vec::<Vec<i32>>(product(&[2, b])?)?;
                p.vec::<i32>(product(&[2, b, max])?)?; // repeat_n + cond-mask clones
            }
            for index in 0..plan.len() {
                let step = plan.step(index).ok_or_else(overflow)?;
                let rows = product(&[b, if params.cfg_at(&step) { 2 } else { 1 }])?;
                p.vec::<f32>(rows)?; // pipeline's vec![model_time;rows]
                backbone(&mut p, &config.model, rows, h, w, first)?;
                first = false;
                if actual.preview.is_active() {
                    p.vec::<f32>(9)?; // preview::project_latents factor_values
                    p.vec::<u8>(product(&[
                        3,
                        h / config.model.patch_size as u64,
                        w / config.model.patch_size as u64,
                    ])?)?;
                }
            }
        }
        // Adapter tensors/alpha-fold scalars are MLX Arrays, not host plane Vecs. Their native
        // constructors are charged in installed_adapters::load. Projection/factor-name maps and
        // safetensors parsing require the separate MetadataHostBound, never a fixed-overhead credit.
        Ok(p.bytes)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use mlx_gen::gen_core::iris::restoration::{
            plan_metadata, RestorationOptions, TileGeometry,
        };
        fn plan(w: u32, h: u32, scale: f64, color_fix: bool) -> RestorationPlanMetadata {
            plan_metadata(
                Dims::new(w, h),
                &RestorationOptions {
                    scale,
                    input_sizing: mlx_gen::InputSizing::Original,
                    color_fix,
                },
                TileGeometry {
                    tile: 16,
                    patch: 16,
                },
                crate::restoration::MODEL_ID,
            )
            .unwrap()
        }
        /// HOST: direct Vec capacities and the separately labelled toolchain policy are distinct.
        #[test]
        fn payload_capacity_arithmetic_and_growth_assumption_are_explicit() {
            let mut p = Payload::default();
            p.vec::<f32>(3).unwrap();
            assert_eq!(p.bytes, 12);
            p.growth::<f32>(3).unwrap();
            assert_eq!(p.bytes, 12 + (4 * 3 + 8) * 4);
            assert!(p.vec::<f32>(u64::MAX).is_err());
            assert!(p.growth::<f32>(u64::MAX).is_err());
        }
        /// HOST: exact legal single-pass path. short<=tile sets enlarged=true even at ratio 1,
        /// so the two identity aa-resize calls each clone once and MUST still be charged.
        #[test]
        fn restoration_single_pass_is_cumulative_and_rgb_is_host_only() {
            let plan = plan(16, 16, 1.0, false);
            assert!(plan.single_pass());
            assert!(plan.enlarged);
            let caller = 3 * 16 * 16;
            let mut p = Payload::default();
            restoration_driver(&mut p, &plan, caller).unwrap();
            let taps = (plan.output.height as u64 + plan.output.width as u64)
                * std::mem::size_of::<([usize; 4], [f32; 4])>() as u64;
            // caller + RGB clone + eight f32 RGB planes + RGB8 output + bicubic taps.
            let exact = 3 * caller + 8 * (3 * 16 * 16 * 4) + taps;
            assert_eq!(p.bytes, exact);
            assert!(
                p.bytes
                    >= crate::restoration::host_plane_bytes_metadata(&plan)
                        .unwrap()
                        .peak()
            );
        }
        /// HOST: tiled work retains ALL crop/readback/restored-crop requests, not one tile peak.
        #[test]
        fn restoration_tiles_and_wavelet_dilations_all_accumulate() {
            let plain = plan(32, 16, 1.0, false);
            assert!(!plain.single_pass());
            let mut colored = plain.clone();
            colored.color_fix = true;
            let mut bare = Payload::default();
            restoration_driver(&mut bare, &plain, 3 * 32 * 16).unwrap();
            let mut fixed = Payload::default();
            restoration_driver(&mut fixed, &colored, 3 * 32 * 16).unwrap();
            let o = pixels(colored.output).unwrap();
            let planes = (2
                * (2 + mlx_gen::gen_core::iris::restoration::COLOR_FIX_DILATIONS.len() as u64)
                + 1)
                * 3
                + 3;
            let mut workers = Payload::default();
            workers.growth::<std::thread::ScopedJoinHandle<'static, mlx_gen::gen_core::Result<Vec<f32>>>>(3).unwrap();
            workers.growth::<Vec<f32>>(3).unwrap();
            assert_eq!(
                fixed.bytes - bare.bytes,
                planes * o * 4 + workers.bytes + provider_scoped_thread_heap_upper(3).unwrap()
            );
        }
        /// HOST: native Active requests and host Vec capacities have independent arithmetic.
        #[test]
        fn repeated_native_tile_calls_apply_operand_u_before_call_count() {
            for backend in [NativeBackend::Cpu, NativeBackend::Metal] {
                let mut l = NativeLedger::new(16_384, backend).unwrap();
                repeated_native_calls(&mut l, 3, |one, first| {
                    one.array(4, 5)?;
                    one.array(4, 1)?;
                    if first {
                        one.array(4, 7)?;
                    }
                    Ok(())
                })
                .unwrap();
                assert_eq!(
                    l.total,
                    3 * (backing_upper(20, 16_384).unwrap() + backing_upper(4, 16_384).unwrap())
                        + backing_upper(28, 16_384).unwrap()
                );
            }
        }
        /// HOST: equal-size resize clones once; no unchanged-axis tap/plane producer is invented.
        #[test]
        fn aa_resize_counts_only_actual_axis_producers() {
            let mut identity = Payload::default();
            aa_resize(&mut identity, Dims::new(16, 16), Dims::new(16, 16)).unwrap();
            assert_eq!(identity.bytes, 3 * 16 * 16 * 4);
            let mut resized = Payload::default();
            aa_resize(&mut resized, Dims::new(16, 16), Dims::new(32, 16)).unwrap();
            let mut taps = Payload::default();
            aa_axis(&mut taps, 16, 32).unwrap();
            assert_eq!(resized.bytes, identity.bytes + taps.bytes + 3 * 16 * 32 * 4);
        }
        /// HOST: depth native-size input allocates HWC and CHW independently, not one peak plane.
        #[test]
        fn depth_payload_counts_both_input_layouts_and_resize_tables() {
            let mut native = Payload::default();
            depth_input_planes(&mut native, Dims::new(16, 16), Dims::new(16, 16), false).unwrap();
            assert_eq!(native.bytes, 2 * 3 * 16 * 16 * 4);
            let mut resized = Payload::default();
            depth_input_planes(&mut resized, Dims::new(32, 32), Dims::new(16, 16), true).unwrap();
            let mut exact = Payload::default();
            lanczos(&mut exact, Dims::new(32, 32), Dims::new(16, 16)).unwrap();
            exact.vec::<f32>(3 * 16 * 16).unwrap();
            assert_eq!(resized.bytes, exact.bytes);
            assert!(resized.bytes > native.bytes);
        }
        /// HOST: generation's RGB and per-batch collections retain all serial and prompt rows.
        #[test]
        fn generation_output_payload_counts_all_rows_and_serials() {
            let (b, count, h, w) = (2, 3, 16, 16);
            let mut p = Payload::default();
            generation_outputs(&mut p, b, count, h, w).unwrap();
            let exact = b
                * count
                * (std::mem::size_of::<mlx_rs::Array>() as u64
                    + std::mem::size_of::<mlx_gen::Image>() as u64)
                + count * (4 * b + 8) * std::mem::size_of::<mlx_gen::Image>() as u64
                + b * count * 3 * h * w;
            assert_eq!(p.bytes, exact);
        }
        /// HOST: live host-layout API refuses missing proof terms BEFORE config/tokenizer parsing.
        /// Supplied test-only metadata values exercise payload-plus-bound arithmetic, not a
        /// production proof constructor or an approved physical-charge premise.
        #[test]
        fn complete_host_bound_refuses_named_gaps_and_adds_test_only_terms() {
            use mlx_gen::gen_core::{
                MemoryGeometry, MemoryNumericTier, MemorySelection, MemoryStrategy,
                MemoryStrategyParameters,
            };
            let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
            let restore_spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(
                fixtures.join("tiny-snapshot/upscaler"),
            ));
            let depth_spec =
                mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(fixtures.join("tiny-depth")));
            let selection = MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
            };
            let memory = GenerationMemory::default();
            let mode = crate::memory_strategy::memory_mode(IrisTask::Restoration);
            let transform = mlx_gen::TransformRequest {
                image: mlx_gen::Image {
                    width: 16,
                    height: 16,
                    pixels: vec![128; 3 * 16 * 16],
                },
                target: mlx_gen::TargetSize::Scale(1.0),
                color_fix: Some(false),
                input_sizing: mlx_gen::InputSizing::Original,
                ..Default::default()
            };
            let req = IosPricingRequest {
                provider: crate::restoration::MODEL_ID,
                spec: &restore_spec,
                mode: &mode,
                selection: &selection,
                execution: &memory,
                geometry: MemoryGeometry {
                    width: 16,
                    height: 16,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                generation: None,
                transform: Some(&transform),
                depth: None,
                token_window: 0,
            };
            match host_growth_upper(&req, None) {
                Err(Error::Unsupported(reason)) => assert_eq!(reason, CALLBACK_HOST_GAP),
                other => panic!("missing callback host term accepted: {other:?}"),
            }
            match host_growth_upper(
                &req,
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
            ) {
                Err(Error::Unsupported(reason)) => assert_eq!(reason, METADATA_HOST_GAP),
                other => panic!("missing metadata proof accepted: {other:?}"),
            }
            let supplied = MetadataHostBound {
                cumulative_requested_bytes: 777,
            };
            let known = restoration_host_upper(&req).unwrap();
            let mut checks = 0;
            let complete = host_growth_upper_checked(
                &req,
                Some(&supplied),
                None,
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
                |bytes| {
                    assert_eq!(bytes, 777);
                    checks += 1;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(checks, 1);
            assert_eq!(complete, known.checked_add(777).unwrap());
            let depth_work = mlx_gen::gen_core::iris::depth::DepthRequest {
                image: transform.image.clone(),
                ..Default::default()
            };
            let depth_mode = crate::memory_strategy::memory_mode(IrisTask::Depth);
            let depth_req = IosPricingRequest {
                provider: crate::depth::DEPTH_MODEL_ID,
                spec: &depth_spec,
                mode: &depth_mode,
                transform: None,
                depth: Some(&depth_work),
                ..req
            };
            match host_growth_upper(
                &depth_req,
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
            ) {
                Err(Error::Unsupported(reason)) => assert_eq!(reason, METADATA_HOST_GAP),
                other => panic!("depth metadata proof accepted: {other:?}"),
            }
            assert_eq!(
                host_growth_upper_with_bounds(
                    &depth_req,
                    Some(&supplied),
                    None,
                    Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating)
                )
                .unwrap(),
                depth(&depth_req).unwrap().checked_add(777).unwrap()
            );
            let generation = GenerationRequest::default();
            let generation_req = IosPricingRequest {
                provider: crate::model::MODEL_ID,
                mode: &MemoryMode::TextToImage,
                generation: Some(&generation),
                transform: None,
                depth: None,
                ..req
            };
            match host_growth_upper(
                &generation_req,
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
            ) {
                Err(Error::Unsupported(reason)) => assert_eq!(reason, TOKENIZER_HOST_GAP),
                other => panic!("unproved tokenizer accepted: {other:?}"),
            }
            let tokenizer = TokenizerHostBound {
                cumulative_requested_bytes: 123,
            };
            assert_eq!(
                required_tokenizer_host_bytes(crate::model::MODEL_ID, Some(&tokenizer)).unwrap(),
                123
            );
            // Even a supplied tokenizer term cannot bypass the missing metadata-layout term.
            match host_growth_upper_with_bounds(
                &generation_req,
                None,
                Some(&tokenizer),
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
            ) {
                Err(Error::Unsupported(reason)) => assert_eq!(reason, METADATA_HOST_GAP),
                other => panic!("tokenizer-only bound accepted: {other:?}"),
            }
            // Full generation payload arithmetic with TEST-ONLY proof terms. This creates no
            // MLX Array or tensor evaluation; the fixture tokenizer is pure Rust host work.
            let mut generation_spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(
                fixtures.join("tiny-snapshot/iris"),
            ));
            generation_spec.components.insert(
                "text_encoder".into(),
                mlx_gen::WeightsSource::Dir(fixtures.join("tiny-snapshot/text_encoder")),
            );
            let generation_work = GenerationRequest {
                prompt: "a lighthouse".into(),
                width: 16,
                height: 16,
                steps: Some(1),
                guidance: Some(1.0),
                ..Default::default()
            };
            let generation_req = IosPricingRequest {
                spec: &generation_spec,
                generation: Some(&generation_work),
                ..generation_req
            };
            let generation_payload = super::generation(&generation_req).unwrap();
            let complete = host_growth_upper_with_bounds(
                &generation_req,
                Some(&supplied),
                Some(&tokenizer),
                Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
            )
            .unwrap();
            assert_eq!(
                complete,
                generation_payload
                    .checked_add(777)
                    .unwrap()
                    .checked_add(123)
                    .unwrap()
            );
        }
    }
}

/// Actual byte-size facts captured under the existing retained SourceInventory. These are NOT
/// parser/HashMap/BTree allocation bounds and cannot construct MetadataHostBound on their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CapturedMetadataBytes {
    pub(super) file_bytes: u64,
    pub(super) header_bytes: Option<u64>,
}

/// Reuse SourceInventory's before/after identity checks instead of treating a fresh path stat as
/// immutable proof. Reads a stack-sized safetensors prefix only, never a tensor or serde object.
/// Existing callers can capture these sizes before invoking source_metadata_capacity; that hook
/// guards the requested input buffers, not the still-unproved parser/scratch allocation graph.
pub(super) fn captured_metadata_bytes(
    req: &IosPricingRequest<'_>,
    path: &Path,
    safetensors: bool,
) -> Result<CapturedMetadataBytes> {
    use std::io::Read;
    let inventory = req.spec.source_inventory().ok_or_else(|| {
        Error::Unsupported(
            "Iris metadata size capture requires the retained SourceInventory".into(),
        )
    })?;
    inventory.verify_for(req.spec).map_err(Error::from)?;
    // SourceInventory verifies configured roots, not an arbitrary caller-supplied path. Require
    // exact configured File membership or a nonhidden, traversal-free descendant of a captured
    // directory. Its before/after directory-membership checks then cover this file as well.
    // No canonicalization: the existing pin identity is keyed on the literal loader path.
    let absolute = std::path::absolute(path).map_err(|e| Error::Msg(e.to_string()))?;
    let exact_file = req
        .spec
        .file_source_paths()
        .into_iter()
        .any(|file| std::path::absolute(file).is_ok_and(|file| file == absolute));
    let in_directory = req.spec.directory_source_paths().into_iter().any(|dir| {
        std::path::absolute(dir)
            .ok()
            .and_then(|dir| absolute.strip_prefix(dir).ok().map(Path::to_path_buf))
            .is_some_and(|tail| {
                !tail.as_os_str().is_empty()
                    && tail.components().all(|part| match part {
                        std::path::Component::Normal(name) => {
                            !name.to_string_lossy().starts_with('.')
                        }
                        _ => false,
                    })
            })
    });
    if !exact_file && !in_directory {
        return Err(Error::Unsupported(
            "Iris metadata size path is outside the retained SourceInventory".into(),
        ));
    }
    inventory.read_unchanged(|| -> Result<_> {
        let mut file = std::fs::File::open(path).map_err(|e| Error::Msg(e.to_string()))?;
        let file_bytes = file
            .metadata()
            .map_err(|e| Error::Msg(e.to_string()))?
            .len();
        let header_bytes = if safetensors {
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix)
                .map_err(|e| Error::Msg(e.to_string()))?;
            let header = u64::from_le_bytes(prefix);
            if plus(8, header)? > file_bytes {
                return Err(Error::Unsupported(
                    "Iris captured safetensors header extends past file".into(),
                ));
            }
            Some(header)
        } else {
            None
        };
        Ok(CapturedMetadataBytes {
            file_bytes,
            header_bytes,
        })
    })
}

/// Finite, independently composable metadata-layout FRAGMENTS whose producers ARE in the
/// snapshots. Each read/view multiplicity is an actual source call count, not an empirical factor.
///
/// This excludes serde_json/safetensors deserializer scratch and HashMap/BTree bucket/node
/// requests. Their authority is outside the supplied snapshots, so this cannot close the full
/// MetadataHostBound and must NOT be substituted for it in host_growth_upper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct KnownMetadataLayouts {
    pub(super) direct_input_buffers: u64,
    pub(super) normalized_header_vectors: u64,
    pub(super) config_parser_vecs_and_strings: u64,
    pub(super) streamed_key_string_payloads: u64,
}
impl KnownMetadataLayouts {
    pub(super) fn total(self) -> Result<u64> {
        plus(
            plus(self.direct_input_buffers, self.normalized_header_vectors)?,
            plus(
                self.config_parser_vecs_and_strings,
                self.streamed_key_string_payloads,
            )?,
        )
    }
}

/// Finite custom-YAML TREE allocation bound from actual input bytes (iris.rs:706-863).
/// There are at most B nonempty lines, entries, scalar/list items and nonempty containers in
/// B bytes. For each family sum the DECLARED RawVec policy, 4*total_items+8*containers;
/// no constant parser multiplier or arbitrary input cap is used. Source unquote never expands
/// input bytes; empty map values produce literal "null" (four bytes per entry). Flow-list cur
/// strings are a separate growth family, and unquote creates its own final scalar String.
/// This tree bound does not assert a serde_json/safetensors allocation policy.
fn known_yaml_tree_layout(input_bytes: u64) -> Result<u64> {
    fn bytes(width: u64, elements: u64) -> Result<u64> {
        product(&[width, elements])
    }
    // The ACTUAL parser element sizes, measured on gen-core's private types (no mirror).
    let layout = mlx_gen::gen_core::iris::yaml_parser_layouts();
    let slots = plus(product(&[4, input_bytes])?, product(&[8, input_bytes])?)?;
    let lines = if input_bytes == 0 {
        0
    } else {
        bytes(layout.line as u64, plus(product(&[4, input_bytes])?, 8)?)?
    };
    let maps = bytes(layout.map_entry as u64, slots)?;
    let lists = bytes(layout.value as u64, slots)?;
    let flow_items = bytes(std::mem::size_of::<String>() as u64, slots)?;
    let flow_cur_strings = slots; // one-byte String growth, same total bytes/nonempty count bound
    let unquoted = input_bytes;
    let null_literals = product(&["null".len() as u64, input_bytes])?;
    [
        lines,
        maps,
        lists,
        flow_items,
        flow_cur_strings,
        unquoted,
        null_literals,
    ]
    .into_iter()
    .try_fold(0, plus)
}

/// TaskExport::from_dir (downstream.rs:135-144) reads a file Vec and clones it into the UTF-8
/// String. The fs::read growth term uses only the explicitly labelled RawVec policy assumption.
/// Safetensors header-only readers (weightsmeta.rs:621-625) allocate exactly header_len bytes.
/// Locations/shapes use Result<Vec<_>> growth; the header-only final projection is ExactSize.
/// Streamed StackInventory key pairs (dit.rs:500-548) clone a file key and its stripped built key;
/// each key's byte length is bounded by the ACTUAL captured header name, not a fixed reserve.
/// The same two-key upper covers a full-view Rust key copy (mlx-rs/utils/io.rs:49-76) and a
/// selected built-key clone (stream.rs:234-265); initial inventory-pair construction is another
/// separately counted source call. HashMap buckets themselves remain in MetadataHostBound.
pub(super) fn known_metadata_layouts(
    config_bytes: u64,
    config_reads: u64,
    header_bytes: u64,
    header_reads: u64,
    headers: &[Header],
    window_key_builds: u64,
) -> Result<KnownMetadataLayouts> {
    fn assumed_vec_capacity_bytes(width: u64, len: u64) -> Result<u64> {
        // Same declared RawVec policy as host_payload; no allocator metadata/usable-size claim.
        if len == 0 {
            return Ok(0);
        }
        product(&[width, plus(product(&[4, len])?, 8)?])
    }
    let config_buffer = plus(assumed_vec_capacity_bytes(1, config_bytes)?, config_bytes)?;
    let direct_input_buffers = plus(
        product(&[config_buffer, config_reads])?,
        product(&[header_bytes, header_reads])?,
    )?;
    let count = headers.len() as u64;
    let mut vectors = assumed_vec_capacity_bytes(
        std::mem::size_of::<mlx_gen::gen_core::weightsmeta::SafetensorsTensorLocation>() as u64,
        count,
    )?;
    vectors = plus(
        vectors,
        product(&[std::mem::size_of::<Header>() as u64, count])?,
    )?;
    let mut names = 0;
    for h in headers {
        vectors = plus(
            vectors,
            assumed_vec_capacity_bytes(std::mem::size_of::<usize>() as u64, h.shape.len() as u64)?,
        )?;
        names = plus(names, h.name.len() as u64)?;
    }
    Ok(KnownMetadataLayouts {
        direct_input_buffers,
        normalized_header_vectors: product(&[vectors, header_reads])?,
        // TaskExport::from_dir calls yaml::parse twice: task settings, then IrisConfig::parse.
        config_parser_vecs_and_strings: product(&[
            known_yaml_tree_layout(config_bytes)?,
            2,
            config_reads,
        ])?,
        streamed_key_string_payloads: product(&[2, names, window_key_builds])?,
    })
}

#[cfg(test)]
mod metadata_payload_tests {
    use super::*;
    /// HOST: actual source bytes/cardinalities scale the known fragments; no arbitrary header cap
    /// or fixed metadata reserve is used. Missing map/parser authority remains a separate gap.
    #[test]
    fn captured_sizes_and_tensor_counts_price_only_source_known_layouts() {
        let h = Header {
            name: "blocks.0.q.weight".into(),
            dtype: Stored::F32,
            shape: vec![2, 3],
            data_bytes: 24,
        };
        let one = known_metadata_layouts(100, 1, 200, 1, std::slice::from_ref(&h), 1).unwrap();
        assert_eq!(one.direct_input_buffers, (4 * 100 + 8) + 100 + 200);
        assert_eq!(one.streamed_key_string_payloads, 2 * h.name.len() as u64);
        let repeated = known_metadata_layouts(100, 3, 200, 3, &[h], 3).unwrap();
        assert_eq!(
            repeated.total().unwrap(),
            one.total().unwrap().checked_mul(3).unwrap()
        );
        assert!(known_metadata_layouts(u64::MAX, 1, 0, 0, &[], 0).is_err());
    }
}

const CALLBACK_HOST_GAP: &str =
    "caller callback host allocation bound not supplied (on_progress / quantum listener)";

/// Source-derived event-count UPPER. Conditional load/check/synchronization branches are charged
/// even if disabled/off-device; one supplied callback term must cover all of its callback kinds.
/// Progress enum values and borrowed QuantumMark fields are stack values; there is no provider
/// boxing at those call sites. The execution owner supplies callback-body allocation requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CallbackEventUpper {
    progress: u64,
    preview: u64,
    metadata: u64,
    checked: u64,
    synchronized: u64,
}
impl CallbackEventUpper {
    fn add(&mut self, rhs: Self) -> Result<()> {
        self.progress = plus(self.progress, rhs.progress)?;
        self.preview = plus(self.preview, rhs.preview)?;
        self.metadata = plus(self.metadata, rhs.metadata)?;
        self.checked = plus(self.checked, rhs.checked)?;
        self.synchronized = plus(self.synchronized, rhs.synchronized)?;
        Ok(())
    }
    fn times(self, n: u64) -> Result<Self> {
        Ok(Self {
            progress: product(&[self.progress, n])?,
            preview: product(&[self.preview, n])?,
            metadata: product(&[self.metadata, n])?,
            checked: product(&[self.checked, n])?,
            synchronized: product(&[self.synchronized, n])?,
        })
    }
    fn priced(
        self,
        fixed: u64,
        progress: u64,
        preview: u64,
        quantum: u64,
        label_byte: u64,
    ) -> Result<u64> {
        let marks = plus(plus(self.metadata, self.checked)?, self.synchronized)?;
        // Checked/Synchronized marks carry one of this crate's source labels
        // (`quantum_labels`, the complete coverage set of `ios_quantum_phase`): each is at most
        // the longest of them.
        let labelled = plus(self.checked, self.synchronized)?;
        let longest = super::quantum_labels()
            .iter()
            .map(|label| label.len() as u64)
            .max()
            .unwrap_or(0);
        [
            fixed,
            product(&[self.progress, progress])?,
            product(&[self.preview, preview])?,
            product(&[marks, quantum])?,
            product(&[labelled, longest, label_byte])?,
        ]
        .into_iter()
        .try_fold(0, plus)
    }
}

fn stack_windows(depth: u64, window: Option<usize>) -> Result<u64> {
    if depth == 0 {
        return Ok(0);
    }
    ceil(
        depth,
        window.map(|w| w as u64).unwrap_or(depth).clamp(1, depth),
    )
}

/// dit.rs:501-582,589-665,2890-3150: every block, allocation-region mark, output row, stream
/// window read, and BOTH scopes (forward_impl and text_adapter_exec). Each scope emits its own
/// entry and restoration Metadata mark (ios_quantum.rs:295-325,342-382), including nested scopes.
fn backbone_callback_events(
    m: &IrisModel,
    rows: u64,
    by_row: bool,
    stream: Option<usize>,
) -> Result<CallbackEventUpper> {
    let layer = Stack::Layer.depth(m) as u64;
    let refiner = Stack::Refiner.depth(m) as u64;
    let trunk = m.depth as u64;
    let pixel = m.pixel.depth as u64;
    let row_bound = by_row && rows > 1;
    let iterations = if row_bound { rows } else { 1 };
    let mut windows = 0;
    if let Some(window) = stream {
        for depth in [layer, refiner, trunk, pixel] {
            windows = plus(windows, stack_windows(depth, Some(window))?)?;
        }
    }
    let quanta = [
        7,
        layer,
        refiner,
        trunk,
        product(&[plus(pixel, 1)?, iterations])?,
        u64::from(row_bound),
        windows,
    ]
    .into_iter()
    .try_fold(0, plus)?;
    let metadata = [
        3,
        layer,
        refiner,
        trunk,
        product(&[plus(pixel, 1)?, iterations])?,
        windows,
        4,
    ]
    .into_iter()
    .try_fold(0, plus)?;
    Ok(CallbackEventUpper {
        metadata,
        checked: quanta,
        synchronized: quanta,
        ..Default::default()
    })
}

fn callback_event_upper(req: &IosPricingRequest<'_>) -> Result<CallbackEventUpper> {
    match req.provider {
        crate::restoration::MODEL_ID => {
            let resources =
                RestorationResources::from_spec(req.spec, req.provider).map_err(Error::from)?;
            let plan = restoration_priced_plan(req, &resources, None)?;
            let tiles = plan.forward_count().map_err(Error::from)?;
            let stream = (req.spec.load_shape == LoadShape::DeferredMaterialization)
                .then_some(crate::stream::TRANSFORMER_WINDOW_SIZE as usize);
            let mut count =
                backbone_callback_events(&resources.export.config.model, 1, false, stream)?
                    .times(tiles)?;
            // Load checks: restoration LOAD plus model BACKBONE_LOAD. Plan axes (when nonempty)
            // have before/finish; HOST_PLANES and POST_TILE only have before, not finish.
            count.add(CallbackEventUpper {
                progress: plus(tiles, 1)?,
                checked: plus(4, u64::from(plan.axis_bytes().map_err(Error::from)? != 0))?,
                synchronized: u64::from(plan.axis_bytes().map_err(Error::from)? != 0),
                metadata: plus(plus(4, 2)?, tiles)?,
                ..Default::default()
            })?;
            // Every velocity has a separate TILE before/finish outside the DiT's own quanta.
            count.add(CallbackEventUpper {
                checked: tiles,
                synchronized: tiles,
                ..Default::default()
            })?;
            Ok(count)
        }
        crate::depth::DEPTH_MODEL_ID => {
            let actual = req.depth.ok_or_else(|| {
                Error::Unsupported("depth callback counts need source/resolution".into())
            })?;
            let export = TaskExport::from_spec(req.spec, IrisTask::Depth, req.provider)
                .map_err(Error::from)?;
            let plan = depth_input_plan(req.spec, actual)?;
            let stream = (req.spec.load_shape == LoadShape::DeferredMaterialization)
                .then_some(crate::stream::TRANSFORMER_WINDOW_SIZE as usize);
            let mut count = backbone_callback_events(&export.config.model, 1, false, stream)?;
            // One LOAD check; PREP,RGB,INPUT,REDUCER,HOST_MAP pairs; optional HOST_POST pair.
            count.add(CallbackEventUpper {
                progress: 1,
                metadata: 3,
                checked: plus(6, u64::from(plan.resized()))?,
                synchronized: plus(5, u64::from(plan.resized()))?,
                ..Default::default()
            })?;
            Ok(count)
        }
        crate::model::MODEL_ID => {
            let actual = req.generation.ok_or_else(|| {
                Error::Unsupported(
                    "generation callback counts need actual sampling/preview context".into(),
                )
            })?;
            let resources =
                GenerationResources::from_spec(req.spec, req.provider).map_err(Error::from)?;
            let config = IrisConfig::from_dir(&resources.backbone_dir).map_err(Error::from)?;
            let params = resolved_generation_controls(&config, actual, actual.seed.unwrap_or(0))?;
            let plan = params.plan_metadata().map_err(Error::from)?;
            let streamed = req.spec.load_shape == LoadShape::DeferredMaterialization;
            let rungs = crate::model::request_rungs(actual, streamed)?;
            let staged = req.execution.stage_residency
                || req.spec.offload_policy == OffloadPolicy::Sequential;
            let loads = plus(
                1,
                u64::from(staged && req.spec.offload_policy == OffloadPolicy::Resident),
            )?;
            let rows = params.prompts.len() as u64;
            let serials = u64::from(actual.count);
            let mut count = CallbackEventUpper {
                progress: plus(product(&[serials, plan.len() as u64])?, 3)?,
                preview: if actual.preview.is_active() {
                    product(&[serials, plan.len() as u64])?
                } else {
                    0
                },
                metadata: 3,
                checked: product(&[2, loads])?,
                ..Default::default()
            };
            // The progress +3 is Decoding plus at most TWO conditional Loading events. A warm
            // retained pair emits neither Loading event; source residency.rs:449-465,662-682.
            let cfg = mlx_llm::ModelConfig::from_dir(&resources.text_encoder_dir)
                .map_err(|e| Error::Msg(e.to_string()))?;
            let deepest = *config.text_encoder.hidden_layers.last().unwrap_or(&0) as u64;
            let encodes = plus(rows, u64::from(params.uses_cfg()))?;
            let text = if streamed {
                let layers = deepest.min(cfg.num_layers as u64);
                let windows = stack_windows(layers, rungs.text_window)?;
                CallbackEventUpper {
                    metadata: 2,
                    checked: plus(plus(2, layers)?, windows)?,
                    synchronized: plus(plus(2, layers)?, windows)?,
                    ..Default::default()
                }
            } else {
                CallbackEventUpper {
                    metadata: 2,
                    checked: plus(plus(3, deepest)?, cfg.num_layers as u64)?,
                    synchronized: plus(3, deepest)?,
                    ..Default::default()
                }
            };
            count.add(text.times(encodes)?)?;
            for _ in 0..serials {
                // serial, solve_complete; PLAN, FINALIZE, IMAGE pairs.
                count.add(CallbackEventUpper {
                    metadata: 2,
                    checked: 3,
                    synchronized: 3,
                    ..Default::default()
                })?;
                for index in 0..plan.len() {
                    let step = plan.step(index).ok_or_else(overflow)?;
                    let b = product(&[rows, if params.cfg_at(&step) { 2 } else { 1 }])?;
                    let stream = streamed.then_some(rungs.dit_window.unwrap_or(usize::MAX));
                    let mut epoch =
                        backbone_callback_events(&config.model, b, rungs.pixel_rows, stream)?;
                    // solver epoch metadata, pipeline forward metadata; STEP before/finish.
                    epoch.add(CallbackEventUpper {
                        metadata: 2,
                        checked: 1,
                        synchronized: 1,
                        ..Default::default()
                    })?;
                    count.add(epoch)?;
                }
            }
            if !req.spec.adapters.is_empty() {
                let path = resources
                    .backbone_dir
                    .join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
                let (projections, _) = crate::adapters::projection_shapes_from_headers(&path)?;
                let inventory = crate::adapters::header_inventory(
                    &req.spec.adapters,
                    &projections,
                    compute_dtype(req.spec.precision),
                    IrisTask::Generation,
                    req.provider,
                )?;
                let targets = inventory.projections.len() as u64;
                // HOST before once; Iris file before AND strict shared-loader before per file;
                // materialization before+finish for every adapted projection, disabled included.
                count.add(CallbackEventUpper {
                    checked: product(&[
                        loads,
                        plus(
                            plus(1, product(&[2, req.spec.adapters.len() as u64])?)?,
                            targets,
                        )?,
                    ])?,
                    synchronized: product(&[loads, targets])?,
                    ..Default::default()
                })?;
            }
            Ok(count)
        }
        _ => Err(Error::Unsupported(
            "Iris callback counts have no provider source".into(),
        )),
    }
}

/// Allocation-law components re-audited from Cargo.lock-checksum-authenticated dependency bytes
/// and exact ac68 std bytes. These are requested-layout bounds, never malloc/RSS charges. Their
/// source file manifest is supplied with the delta; no alternate package version is authority.
mod authenticated_metadata {
    use super::*;
    fn align_up(n: u64, a: u64) -> Result<u64> {
        if !a.is_power_of_two() {
            return Err(overflow());
        }
        product(&[ceil(n, a)?, a])
    }
    fn growth(width: u64, len: u64, containers: u64) -> Result<u64> {
        product(&[
            width,
            plus(product(&[4, len])?, product(&[8, containers])?)?,
        ])
    }
    fn group_width() -> Result<u64> {
        if cfg!(target_arch = "aarch64") {
            Ok(8)
        } else if cfg!(target_arch = "x86_64") {
            Ok(16)
        } else {
            Err(Error::Unsupported(
                "authenticated std/hashbrown group layout requires aarch64 or x86_64".into(),
            ))
        }
    }
    /// hashbrown 0.16.1 raw/mod.rs:105-162,218-239; 0.17.1 raw.rs:104-160,216-237.
    /// All maps here have non-ZST elements of >=one pointer, so the small-table minimum is 3.
    fn buckets(cap: u64) -> Result<u64> {
        if cap == 0 {
            return Ok(0);
        }
        if cap < 15 {
            return Ok(if cap < 4 {
                4
            } else if cap < 8 {
                8
            } else {
                16
            });
        }
        (product(&[cap, 8])? / 7)
            .checked_next_power_of_two()
            .ok_or_else(overflow)
    }
    fn table_request(width: u64, align: u64, count: u64) -> Result<u64> {
        if count == 0 {
            return Ok(0);
        }
        let group = group_width()?;
        plus(
            align_up(product(&[width, count])?, align.max(group))?,
            plus(count, group)?,
        )
    }
    /// Include EVERY geometrically grown backing. reserve_rehash_inner grows only while the
    /// current capacity is <2*new_items; once sparse enough it rehashes in place without a backing.
    /// Therefore buckets(2*max_items) bounds the last table even under remove/reinsert churn.
    fn hashmap<K, V>(items: u64) -> Result<u64> {
        if items == 0 {
            return Ok(0);
        }
        let last = buckets(product(&[2, items])?)?;
        let (width, align) = (
            std::mem::size_of::<(K, V)>() as u64,
            std::mem::align_of::<(K, V)>() as u64,
        );
        if width < std::mem::size_of::<usize>() as u64 {
            return Err(overflow());
        }
        let (mut current, mut total) = (4, 0);
        while current <= last {
            total = plus(total, table_request(width, align, current)?)?;
            current = current.checked_mul(2).ok_or_else(overflow)?;
        }
        Ok(total)
    }
    /// Exact source fields of ac68 btree/node.rs:42-69,102-114: parent pointer, two u16s,
    /// 11 key/value slots; internal node adds 12 pointers. Rounding EACH field to max alignment
    /// is an upper independent of private Rust field reordering, not a fitted per-node reserve.
    fn btree_node<K, V>() -> Result<u64> {
        let a = (std::mem::align_of::<K>() as u64)
            .max(std::mem::align_of::<V>() as u64)
            .max(std::mem::align_of::<usize>() as u64);
        [
            std::mem::size_of::<usize>() as u64,
            2,
            2,
            product(&[11, std::mem::size_of::<K>() as u64])?,
            product(&[11, std::mem::size_of::<V>() as u64])?,
            product(&[12, std::mem::size_of::<usize>() as u64])?,
        ]
        .into_iter()
        .try_fold(0, |sum, field| plus(sum, align_up(field, a)?))
    }
    /// Insert-only deserialization never frees/replaces tree nodes. Every allocated stable node
    /// contains a key; count + one root-insertion transient per nonempty map is a source upper.
    fn btrees<K, V>(entries: u64, maps: u64) -> Result<u64> {
        product(&[plus(entries, maps)?, btree_node::<K, V>()?])
    }
    /// indexmap2.14 inner.rs:228-265,480-498: entries reserve_exact to index-table capacity.
    /// Sum table AND entry buffers at every possible table size, no release/resize credit.
    fn indexmaps<K, V>(entries: u64) -> Result<u64> {
        if entries == 0 {
            return Ok(0);
        }
        let last = buckets(product(&[2, entries])?)?;
        let mut current = 4;
        let mut total = 0;
        while current <= last {
            total = plus(
                total,
                table_request(
                    std::mem::size_of::<usize>() as u64,
                    std::mem::align_of::<usize>() as u64,
                    current,
                )?,
            )?;
            total = plus(
                total,
                product(&[current, std::mem::size_of::<(usize, K, V)>() as u64])?,
            )?;
            current = current.checked_mul(2).ok_or_else(overflow)?;
        }
        Ok(total)
    }

    /// Insert-only map families: capacity_to_buckets(n) <=4*n; a geometric backing sequence
    /// sums to <2*last_buckets, hence <=8*n bucket slots across ALL independent maps too.
    /// There are <=n growths. Charge each table's real control group/alignment padding separately.
    /// This avoids assuming nonlinear table envelopes distribute over nested object cardinalities.
    fn indexmap_family<K, V>(entries: u64) -> Result<u64> {
        if entries == 0 {
            return Ok(0);
        }
        let group = group_width()?;
        let slots = product(&[8, entries])?;
        let indices = product(&[slots, std::mem::size_of::<usize>() as u64])?;
        let controls = slots;
        let tail_padding = product(&[entries, plus(group - 1, group)?])?;
        let values = product(&[slots, std::mem::size_of::<(usize, K, V)>() as u64])?;
        [indices, controls, tail_padding, values]
            .into_iter()
            .try_fold(0, plus)
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(super) struct JsonCounts {
        pub(super) bytes: u64,
        pub(super) objects: u64,
        pub(super) object_entries: u64,
        pub(super) arrays: u64,
        pub(super) array_elements: u64,
        pub(super) strings: u64,
        pub(super) string_source_bytes: u64,
        pub(super) numbers: u64,
        pub(super) number_source_bytes: u64,
        pub(super) floats: u64,
        pub(super) raw_value_key: bool,
    }
    struct Scan<'a> {
        data: &'a [u8],
        at: usize,
        counts: JsonCounts,
    }
    impl Scan<'_> {
        fn ws(&mut self) {
            while self.data.get(self.at).is_some_and(u8::is_ascii_whitespace) {
                self.at += 1;
            }
        }
        fn bad<T>() -> Result<T> {
            Err(Error::Unsupported(
                "invalid JSON for metadata layout scan".into(),
            ))
        }
        fn byte(&mut self) -> Result<u8> {
            let b = *self
                .data
                .get(self.at)
                .ok_or_else(|| Error::Unsupported("truncated metadata JSON".into()))?;
            self.at += 1;
            Ok(b)
        }
        fn string(&mut self, key: bool) -> Result<()> {
            if self.byte()? != b'"' {
                return Self::bad();
            }
            let start = self.at;
            let magic: [&[u8]; 2] = [
                b"$serde_json::private::RawValue",
                b"$serde_json::private::Number",
            ];
            let (mut matched, mut exact) = ([0usize; 2], [true; 2]);
            loop {
                let b = self.byte()?;
                if b == b'"' {
                    break;
                }
                let decoded = if b == b'\\' {
                    match self.byte()? {
                        b'u' => {
                            let mut c = 0u16;
                            for _ in 0..4 {
                                let b = self.byte()?;
                                let digit = match b {
                                    b'0'..=b'9' => b - b'0',
                                    b'a'..=b'f' => b - b'a' + 10,
                                    b'A'..=b'F' => b - b'A' + 10,
                                    _ => return Self::bad(),
                                };
                                c = (c << 4) | u16::from(digit);
                            }
                            u8::try_from(c).ok().filter(u8::is_ascii)
                        }
                        b'"' => Some(b'"'),
                        b'\\' => Some(b'\\'),
                        b'/' => Some(b'/'),
                        b'b' => Some(8),
                        b'f' => Some(12),
                        b'n' => Some(10),
                        b'r' => Some(13),
                        b't' => Some(9),
                        _ => return Self::bad(),
                    }
                } else {
                    if b < 0x20 {
                        return Self::bad();
                    }
                    Some(b)
                };
                for which in 0..magic.len() {
                    if exact[which] && magic[which].get(matched[which]).copied() == decoded {
                        matched[which] += 1;
                    } else {
                        exact[which] = false;
                    }
                }
            }
            self.counts.strings = plus(self.counts.strings, 1)?;
            self.counts.string_source_bytes = plus(
                self.counts.string_source_bytes,
                (self.at - start - 1) as u64,
            )?;
            self.counts.raw_value_key |= key
                && (0..magic.len())
                    .any(|which| exact[which] && matched[which] == magic[which].len());
            Ok(())
        }
        fn value(&mut self, depth: u16) -> Result<()> {
            if depth > 128 {
                return Err(Error::Unsupported(
                    "metadata JSON exceeds serde default recursion depth".into(),
                ));
            }
            self.ws();
            match self.data.get(self.at).copied() {
                Some(b'{') => {
                    self.at += 1;
                    self.counts.objects = plus(self.counts.objects, 1)?;
                    self.ws();
                    if self.data.get(self.at) == Some(&b'}') {
                        self.at += 1;
                        return Ok(());
                    }
                    loop {
                        self.ws();
                        self.string(true)?;
                        self.ws();
                        if self.byte()? != b':' {
                            return Self::bad();
                        }
                        self.counts.object_entries = plus(self.counts.object_entries, 1)?;
                        self.value(depth + 1)?;
                        self.ws();
                        match self.byte()? {
                            b'}' => break,
                            b',' => {}
                            _ => return Self::bad(),
                        }
                    }
                }
                Some(b'[') => {
                    self.at += 1;
                    self.counts.arrays = plus(self.counts.arrays, 1)?;
                    self.ws();
                    if self.data.get(self.at) == Some(&b']') {
                        self.at += 1;
                        return Ok(());
                    }
                    loop {
                        self.counts.array_elements = plus(self.counts.array_elements, 1)?;
                        self.value(depth + 1)?;
                        self.ws();
                        match self.byte()? {
                            b']' => break,
                            b',' => {}
                            _ => return Self::bad(),
                        }
                    }
                }
                Some(b'"') => self.string(false)?,
                Some(b't' | b'f' | b'n') => {
                    let literal: &[u8] = match self.data[self.at] {
                        b't' => b"true",
                        b'f' => b"false",
                        _ => b"null",
                    };
                    if self.data.get(self.at..self.at + literal.len()) != Some(literal) {
                        return Self::bad();
                    }
                    self.at += literal.len();
                }
                Some(b'-' | b'0'..=b'9') => {
                    let start = self.at;
                    let mut float = false;
                    while self.data.get(self.at).is_some_and(|b| {
                        matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                    }) {
                        float |= matches!(self.data[self.at], b'.' | b'e' | b'E');
                        self.at += 1;
                    }
                    self.counts.numbers = plus(self.counts.numbers, 1)?;
                    self.counts.number_source_bytes =
                        plus(self.counts.number_source_bytes, (self.at - start) as u64)?;
                    self.counts.floats = plus(self.counts.floats, u64::from(float))?;
                }
                _ => return Self::bad(),
            }
            Ok(())
        }
    }
    /// Allocation-free lexical cardinality scan. It does not replace serde validation, expand
    /// strings, allocate an AST, or trust an attacker-supplied cardinality field.
    pub(super) fn scan_json(data: &[u8]) -> Result<JsonCounts> {
        let mut s = Scan {
            data,
            at: 0,
            counts: JsonCounts {
                bytes: data.len() as u64,
                ..Default::default()
            },
        };
        s.value(0)?;
        s.ws();
        if s.at != data.len() {
            return Scan::bad();
        }
        Ok(s.counts)
    }

    /// serde_core/private/content.rs and serde/private/de.rs:10-39,313-510: the private
    /// `Content` enum's widest variant payload is a three-word String/Vec (no i128/u128 variant
    /// in this source). Its size is an UPPER from that payload plus a full-word discriminant,
    /// rounded to the word, under the labelled TOOLCHAIN-LAYOUT assumption that a repr(Rust)
    /// enum is no larger than one word of tag plus its widest payload (no type is mirrored and no
    /// equality with the private layout is claimed). Never below the public JSON Value cell.
    fn content_cell() -> u64 {
        let word = std::mem::size_of::<usize>() as u64;
        let widest_payload = 3 * word;
        (widest_payload + word).max(std::mem::size_of::<serde_json::Value>() as u64)
    }
    /// Complete generic JSON Value/Content TREE layouts and shared scanner scratch, excluding
    /// numeric bigint fallback (separate below). Cfg-exclusive BTree and IndexMap implementations
    /// are both charged so global serde_json feature unification is not silently guessed.
    pub(super) fn json_tree_layout(c: JsonCounts) -> Result<u64> {
        let value = std::mem::size_of::<serde_json::Value>() as u64;
        let mut total = growth(1, plus(c.bytes, 3)?, 1)?; // shared unescape scratch + reserve(4)
        total = plus(total, c.string_source_bytes)?; // owned strings in Value visitor
        total = plus(total, growth(value, c.array_elements, c.arrays)?)?;
        total = plus(
            total,
            btrees::<String, serde_json::Value>(c.object_entries, c.objects)?,
        )?;
        total = plus(
            total,
            indexmap_family::<String, serde_json::Value>(c.object_entries)?,
        )?;
        // Serde #[flatten] Content buffer, then typed leaves: real separate source allocations.
        total = plus(total, growth(content_cell(), c.array_elements, c.arrays)?)?;
        total = plus(
            total,
            growth(product(&[2, content_cell()])?, c.object_entries, c.objects)?,
        )?;
        total = plus(total, c.string_source_bytes)?; // Content owned-string branch
                                                     // arbitrary_precision starts a 16-byte String for EACH number; normalized f64 uses
                                                     // stack zmij Buffer[24] then one owned Number string. Both source cfg branches count.
        total = plus(total, product(&[16, c.numbers])?)?;
        total = plus(total, growth(1, c.number_source_bytes, c.numbers)?)?;
        total = plus(total, product(&[24, c.numbers])?)?;
        // Typed String/String metadata and String/usize index maps (both SafeTensors versions).
        total = plus(total, hashmap::<String, String>(c.object_entries)?)?;
        total = plus(total, hashmap::<String, usize>(c.object_entries)?)?;
        total = plus(total, product(&[2, c.string_source_bytes])?)?;
        Ok(total)
    }

    /// serde_json lexical/math.rs:661-775 recurrence over EVERY source Vec producer.
    /// Each long product creates y.len() small-product Vecs and grows its accumulating result.
    /// Balanced Karatsuba creates two sum Vecs, three recursive products and a result reserve;
    /// uneven Karatsuba creates one result and ceil(y/x) recursive products. Max is over actual
    /// mutually exclusive algorithm paths, NOT a guard scratch maximum or an operation budget.
    fn lexical_multiply_layout(max_words: usize, limb: u64) -> Result<u64> {
        const MAX: usize = ((4 * 769 + 3 * ((1usize << 14) - 1) + 64 + 31) / 32) + 2;
        if max_words > MAX {
            return Err(overflow());
        }
        let mut cumulative = [0u64; MAX + 1]; // fixed STACK memo, no host allocation
        for n in 1..=max_words {
            let out = plus(product(&[2, n as u64])?, 2)?;
            cumulative[n] = if n <= 32 {
                plus(
                    product(&[n as u64, growth(limb, plus(n as u64, 1)?, 1)?])?,
                    growth(limb, out, 1)?,
                )?
            } else {
                let half = n.div_ceil(2) + 1;
                let balanced = [
                    growth(limb, half as u64, 1)?,
                    growth(limb, half as u64, 1)?,
                    cumulative[n / 2],
                    cumulative[half],
                    cumulative[n.div_ceil(2)],
                    product(&[limb, out])?,
                ]
                .into_iter()
                .try_fold(0, plus)?;
                let mut uneven = 0;
                for x in 1..n.div_ceil(2) {
                    let path = plus(
                        growth(limb, out, 1)?,
                        product(&[ceil(n as u64, x as u64)?, cumulative[x]])?,
                    )?;
                    uneven = uneven.max(path);
                }
                balanced.max(uneven)
            };
        }
        Ok(cumulative[max_words])
    }
    /// Recovered lexical sources: Bigint defaults to 20 limbs; f64 MAX_DIGITS=769;
    /// POW5 has 14 static tables; each pow5 loop selects at most 14 products. Before any
    /// potentially enormous binary shift, bits <=4*769+3*(2^14-1)+64. Shift count is bounded
    /// by the source i32 exponent domain, <=2^31. Both Bigint owners and both source paths
    /// remain charged; no cap is invented from a test size. Both 32/64-limb cfgs are covered.
    fn lexical_float_layout() -> Result<u64> {
        let before_bits = plus(
            plus(product(&[4, 769])?, product(&[3, (1u64 << 14) - 1])?)?,
            64,
        )?;
        let mut total = 0;
        for limb in [4u64, 8] {
            let words = plus(ceil(before_bits, product(&[8, limb])?)?, 2)?;
            let shifted = plus(
                ceil(plus(before_bits, 1u64 << 31)?, product(&[8, limb])?)?,
                2,
            )?;
            let defaults = product(&[2, 20, limb])?;
            let owner_growth = product(&[2, growth(limb, shifted, 1)?])?;
            let products = product(&[2, 14, lexical_multiply_layout(words as usize, limb)?])?;
            total = plus(total, plus(plus(defaults, owner_growth)?, products)?)?;
        }
        Ok(total)
    }

    /// Finite complete parser-layout component, including optional raw-value/number reparsing
    /// and float_roundtrip Bigints. Each recursive private-marker reparse consumes a nested
    /// encoded string and never duplicates source bytes; there are <=B parser passes and <=B
    /// bytes per pass in B initial bytes. Normal headers without private keys use ONE pass.
    /// This function does not price the provider's execution schedule or std thread/IO wrappers.
    pub(super) fn json_parse_layout(data: &[u8]) -> Result<u64> {
        let c = scan_json(data)?;
        let (one, passes) = if c.raw_value_key {
            let b = c.bytes;
            (
                JsonCounts {
                    bytes: b,
                    objects: b,
                    object_entries: b,
                    arrays: b,
                    array_elements: b,
                    strings: b,
                    string_source_bytes: b,
                    numbers: b,
                    number_source_bytes: b,
                    floats: b,
                    raw_value_key: true,
                },
                plus(b, 1)?,
            )
        } else {
            (c, 1)
        };
        let mut parsed = json_tree_layout(one)?;
        if one.floats != 0 {
            parsed = plus(parsed, product(&[one.floats, lexical_float_layout()?])?)?;
        }
        product(&[passes, parsed])
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn json_scan_tracks_real_cardinalities_without_parsing_allocations() {
            let c = scan_json(br#"{"a":[1,2],"b":{"x":"hello"}}"#).unwrap();
            assert_eq!(
                (
                    c.objects,
                    c.object_entries,
                    c.arrays,
                    c.array_elements,
                    c.numbers,
                    c.strings
                ),
                (2, 3, 1, 2, 2, 4)
            );
            assert!(!c.raw_value_key);
            assert!(
                scan_json(br#"{"$serde_json::private::RawValue":"{}"}"#)
                    .unwrap()
                    .raw_value_key
            );
            assert!(scan_json(b"{").is_err());
        }
        #[test]
        fn recovered_json_layouts_and_small_bigint_recurrence_are_finite() {
            let raw = br#"{"shape":[1,2],"dtype":"F32","data_offsets":[0,8]}"#;
            assert!(json_parse_layout(raw).unwrap() > raw.len() as u64);
            let exact = 2 * growth(8, 3, 1).unwrap() + growth(8, 6, 1).unwrap();
            assert_eq!(lexical_multiply_layout(2, 8).unwrap(), exact);
        }
        #[test]
        fn authenticated_table_layouts_include_every_grown_backing() {
            let group = group_width().unwrap();
            assert_eq!(
                table_request(32, 8, 4).unwrap(),
                align_up(128, group).unwrap() + 4 + group
            );
            assert!(hashmap::<String, usize>(8).unwrap() > table_request(32, 8, 4).unwrap());
            assert!(
                btree_node::<String, serde_json::Value>().unwrap()
                    > std::mem::size_of::<serde_json::Value>() as u64
            );
            assert!(hashmap::<String, usize>(u64::MAX).is_err());
        }
    }
}

/// Finite source-known Vec/PathBuf part of SourceInventory::verify_for + verify_unchanged.
/// Uses actual retained cardinalities, not directory guesses. Overlapping roots may visit a
/// member once per root; old buffers remain charged. This does NOT include std ReadDir's private
/// OS-wrapper layouts or the remaining complete MetadataHostBound, and cannot replace that term.
pub(super) fn known_inventory_verify_vectors(req: &IosPricingRequest<'_>) -> Result<u64> {
    let inventory = req.spec.source_inventory().ok_or_else(|| {
        Error::Unsupported(
            "Iris host metadata needs retained SourceInventory allocation facts".into(),
        )
    })?;
    let f = inventory.allocation_facts().map_err(Error::from)?;
    fn grown(width: u64, elements: u64) -> Result<u64> {
        if elements == 0 {
            return Ok(0);
        }
        product(&[width, plus(product(&[4, elements])?, 8)?])
    }
    let paths = std::mem::size_of::<std::path::PathBuf>() as u64;
    let refs = std::mem::size_of::<&std::path::Path>() as u64;
    let entries = product(&[f.roots_len, f.entries_len])?;
    let files = product(&[f.roots_len, f.pins_len])?;
    let member_bytes = product(&[f.roots_len, f.entries_path_bytes])?;
    let member_growth = plus(product(&[4, member_bytes])?, product(&[8, entries])?)?;
    [
        grown(refs, f.roots_len)?,
        grown(refs, f.roots_len)?, // configured File/Dir path lists
        grown(paths, f.roots_len)?,
        f.roots_path_bytes, // absolute roots and their path clones
        grown(paths, entries)?,
        grown(paths, files)?,
        member_growth,
        product(&[2, member_bytes])?,
    ] // DirEntry.path + retained entry/file clones
    .into_iter()
    .try_fold(0, plus)
}

#[cfg(test)]
mod callback_host_tests {
    use super::*;
    #[test]
    fn callback_cost_keeps_progress_preview_and_all_mark_kinds_separate() {
        let events = CallbackEventUpper {
            progress: 2,
            preview: 3,
            metadata: 5,
            checked: 7,
            synchronized: 11,
        };
        let longest = super::super::quantum_labels()
            .iter()
            .map(|label| label.len() as u64)
            .max()
            .unwrap();
        assert!(longest > 0);
        assert_eq!(
            events.priced(13, 17, 19, 23, 29).unwrap(),
            13 + 2 * 17 + 3 * 19 + (5 + 7 + 11) * 23 + (7 + 11) * longest * 29
        );
        assert!(events.priced(u64::MAX, 1, 1, 1, 0).is_err());
        assert!(events.times(u64::MAX).is_err());
    }
    #[test]
    fn source_forward_event_formulas_cover_scope_and_window_marks() {
        let config = IrisConfig::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot/iris"),
        )
        .unwrap();
        let m = &config.model;
        let dense = backbone_callback_events(m, 1, false, None).unwrap();
        let blocks = m.depth as u64 + m.pixel.depth as u64;
        assert_eq!(dense.checked, 12 + blocks);
        assert_eq!(dense.synchronized, 12 + blocks);
        assert_eq!(dense.metadata, 12 + blocks); // allocation-region/block/output metadata +4 scope marks
        let streamed = backbone_callback_events(m, 1, false, Some(1)).unwrap();
        let windows = 4 + blocks;
        assert_eq!(streamed.checked - dense.checked, windows);
        assert_eq!(streamed.synchronized - dense.synchronized, windows);
        assert_eq!(streamed.metadata - dense.metadata, windows);
        let rows = backbone_callback_events(m, 2, true, None).unwrap();
        assert_eq!(rows.checked - dense.checked, m.pixel.depth as u64 + 2);
    }
}

/// Source-private aggregate-layout UPPER (never an equality claim about a private type): each
/// real source field rounded up to the maximum field alignment. Labelled TOOLCHAIN-LAYOUT
/// assumption: a repr(Rust) struct is no larger than that sum (rustc inserts no padding beyond
/// alignment and may only reorder or niche-pack fields, which can only shrink it).
fn aggregate_requested_layout(fields: &[(u64, u64)], minimum_alignment: u64) -> Result<(u64, u64)> {
    let align = fields
        .iter()
        .map(|&(_, align)| align)
        .fold(minimum_alignment, u64::max);
    if !align.is_power_of_two() {
        return Err(overflow());
    }
    let bytes = fields.iter().try_fold(0, |sum, &(size, _)| {
        plus(sum, product(&[ceil(size, align)?, align])?)
    })?;
    Ok((bytes, align))
}
fn arc_requested_layout(payload: (u64, u64)) -> Result<u64> {
    // Exact ac68 alloc/sync.rs ArcInner repr(C,align(2)), two AtomicUsize header words.
    let word = std::mem::size_of::<usize>() as u64;
    let align = payload.1.max(std::mem::align_of::<usize>() as u64).max(2);
    let offset = product(&[ceil(product(&[2, word])?, payload.1)?, payload.1])?;
    product(&[ceil(plus(offset, payload.0)?, align)?, align])
}

/// Provider-owned Rust heap around RGB wavelet worker bodies (already separately priced).
/// Source: exact ac68 thread/{scoped:148,lifecycle:56-57,113,121,148-151,180-184,
/// thread:49-107,current:241-250,284-300}, Darwin Parker:39-43. ScopeData/Packet/closure/ThreadInit
/// use real source fields; Arc clones allocate nothing. The cold parent Thread handle is included.
///
/// Spawn hooks are source-proven empty: stable1.96.0 cannot call the gated unstable API, and the
/// immutable workspace + every recovered dependency have no caller/feature. See scan manifest.
/// NATIVE libdispatch semaphore objects, OS thread stacks/guards/TLS backing and allocator charge
/// are NOT Rust requested layouts and remain explicitly unapproved physical quantities.
/// The paired producer change uses Builder.stack_size(COLOR_FIX_WORKER_STACK_BYTES), which
/// bypasses lifecycle.rs:29-44's ENTIRE cache/env closure. Request-time RUST_MIN_STACK bytes are
/// therefore source-proven zero for that producer, not a cache-warm assumption or env change.
fn provider_scoped_thread_heap_upper(workers: u64) -> Result<u64> {
    if workers == 0 {
        return Ok(0);
    }
    if !cfg!(all(
        target_pointer_width = "64",
        any(target_vendor = "apple", target_os = "linux")
    )) {
        return Err(Error::Unsupported(
            "scoped-thread requested layouts require authenticated 64-bit Apple/Linux target"
                .into(),
        ));
    }
    let word = std::mem::size_of::<usize>() as u64;
    let word_align = std::mem::align_of::<usize>() as u64;
    let thread = (
        std::mem::size_of::<std::thread::Thread>() as u64,
        std::mem::align_of::<std::thread::Thread>() as u64,
    );
    let id = (
        std::mem::size_of::<std::thread::ThreadId>() as u64,
        std::mem::align_of::<std::thread::ThreadId>() as u64,
    );
    let parker = if cfg!(target_vendor = "apple") {
        aggregate_requested_layout(&[(word, word_align), (1, 1)], word_align)? // dispatch ptr +AtomicI8
    } else {
        (
            std::mem::size_of::<std::sync::atomic::AtomicU32>() as u64,
            std::mem::align_of::<std::sync::atomic::AtomicU32>() as u64,
        )
    }; // Linux futex Parker
    let inner = aggregate_requested_layout(
        &[
            (
                std::mem::size_of::<Option<std::ffi::CString>>() as u64,
                std::mem::align_of::<Option<std::ffi::CString>>() as u64,
            ),
            id,
            parker,
        ],
        8,
    )?;
    let scope = aggregate_requested_layout(&[(word, word_align), (1, 1), thread], word_align)?;
    type WorkerReturn = mlx_gen::gen_core::Result<Vec<f32>>;
    type PacketResult = Option<std::thread::Result<WorkerReturn>>;
    let packet = aggregate_requested_layout(
        &[
            (word, word_align),
            (
                std::mem::size_of::<PacketResult>() as u64,
                std::mem::align_of::<PacketResult>() as u64,
            ),
        ],
        word_align,
    )?;
    // Iris worker F captures cp,rp:&[f32],h,w:usize,cancel:&CancelFlag; all are source-visible.
    let f = aggregate_requested_layout(
        &[
            (std::mem::size_of::<&[f32]>() as u64, word_align),
            (std::mem::size_of::<&[f32]>() as u64, word_align),
            (word, word_align),
            (word, word_align),
            (word, word_align),
        ],
        word_align,
    )?;
    let hooks = aggregate_requested_layout(
        &[
            (word, word_align),
            (
                std::mem::size_of::<Vec<Box<dyn FnOnce() + Send>>>() as u64,
                word_align,
            ),
        ],
        word_align,
    )?;
    let start = aggregate_requested_layout(&[f, hooks, (word, word_align)], word_align)?;
    let init = aggregate_requested_layout(
        &[
            thread,
            (
                std::mem::size_of::<Box<dyn FnOnce() + Send>>() as u64,
                word_align,
            ),
        ],
        word_align,
    )?;
    let each = [
        arc_requested_layout(inner)?,
        arc_requested_layout(packet)?,
        start.0,
        init.0,
    ]
    .into_iter()
    .try_fold(0, plus)?;
    plus(
        plus(arc_requested_layout(scope)?, arc_requested_layout(inner)?)?,
        product(&[workers, each])?,
    )
}

/// Separate, UNAPPROVED PHYSICAL-domain quantity: sum of BUILDER-REQUESTED worker stack bytes.
/// Uses the actual explicit source constant used by Builder::stack_size, never an inferred RSS
/// cap. pthread minimum/page rules, stack guards, TLS and libdispatch are additional
/// physical questions. This is deliberately not added to native Active OR Rust heap totals.
pub(super) fn restoration_worker_os_stack_request_bytes(
    req: &IosPricingRequest<'_>,
) -> Result<u64> {
    let resources = RestorationResources::from_spec(req.spec, crate::restoration::MODEL_ID)
        .map_err(Error::from)?;
    let plan = restoration_priced_plan(req, &resources, None)?;
    if plan.color_fix {
        product(&[
            3,
            mlx_gen::gen_core::iris::restoration::COLOR_FIX_WORKER_STACK_BYTES as u64,
        ])
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod scoped_thread_layout_tests {
    use super::*;
    #[test]
    fn explicit_worker_heap_excludes_os_stack_and_environment_bytes() {
        assert_eq!(provider_scoped_thread_heap_upper(0).unwrap(), 0);
        let one = provider_scoped_thread_heap_upper(1).unwrap();
        let three = provider_scoped_thread_heap_upper(3).unwrap();
        assert!(three > one);
        assert_eq!(
            three - one,
            2 * (provider_scoped_thread_heap_upper(2).unwrap() - one)
        );
        let payload = aggregate_requested_layout(&[(8, 8), (1, 1), (8, 8)], 8).unwrap();
        assert_eq!(payload, (24, 8));
        assert_eq!(arc_requested_layout(payload).unwrap(), 40);
    }
}
