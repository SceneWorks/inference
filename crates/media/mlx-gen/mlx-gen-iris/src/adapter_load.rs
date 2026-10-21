//! Source-counted RAW adapter normalization/reconstruction upper bounds.
//!
//! This is not an allocator/physical-footprint certificate. It sums the newly produced buffers
//! in the actual loader graph without depending on donation or early queued-task completion.
//! Source parameters are held outside this scratch. Matrix transpose/reshape views are not a
//! second retained parameter. The only optional copies counted below are the native
//! transpose_reshape / matmul operand-normalization branches; at most one copy per operand is
//! needed (a reshape copy is row-contiguous, so the following matmul cannot copy it again).
//!
//! Tucker follows MLX `einsum.cpp`'s three-term greedy contraction selection, including the
//! cost-limited remaining-term naive fallback. It does not assume ranks are smaller than widths.

use std::collections::{BTreeMap, BTreeSet};

use mlx_gen::adapters::facts::{dtype_width, AdapterFacts, TensorFacts};
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use super::{AdapterHeaderInventory, AdapterReconstructionFacts};
use crate::retention::PAGE_BYTES;

/// One concrete possible allocation site in the native reconstruction graph. Optional layout
/// normalization is an upper, not a claim that an alias actually allocates on this request.
#[derive(Clone, Debug)]
pub struct AdapterLoadAllocation {
    pub operation: &'static str,
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    /// A native kernel's minimum allocation, when larger than its logical result.
    pub minimum_bytes: u64,
    /// This graph's final newly produced parameter, as opposed to a reconstruction temporary.
    /// Source-file aliases (LoRA A and an unscaled B) are held inputs, not allocations in this list.
    pub retained_output: bool,
}

impl AdapterLoadAllocation {
    pub fn bytes(&self) -> Result<u64> {
        checked_allocated(
            checked_mul(
                checked_elements(&self.shape)?,
                dtype_width(self.dtype),
                self.operation,
            )?
            .max(self.minimum_bytes),
        )
    }
}

/// One projection's materialization window. Inputs (including earlier file reads), the base and
/// prior materialized adapters are ambient held memory, NOT newly allocated by this window.
#[derive(Clone, Debug, Default)]
pub struct ProjectionAdapterLoadFacts {
    pub retained: Vec<AdapterFacts>,
    pub allocations: Vec<AdapterLoadAllocation>,
    /// Host scalar arrays created when the strict loader BUILDS the graph, before evaluation.
    pub graph_constructor_bytes: u64,
}

impl ProjectionAdapterLoadFacts {
    pub fn materialize_scratch_bytes(&self) -> Result<u64> {
        self.allocations.iter().try_fold(0, |total, a| {
            checked_add(total, a.bytes()?, "materialization allocation sum")
        })
    }

    pub fn retained_bytes(&self) -> Result<u64> {
        self.retained
            .iter()
            .flat_map(AdapterFacts::tensors)
            .try_fold(0, |total, t| {
                checked_add(
                    total,
                    checked_allocated(checked_tensor_bytes(t)?)?,
                    "retained payload sum",
                )
            })
    }

    /// New retained allocation sites inside this evaluation, not the source-file aliases that
    /// were already held. Compose these plus transient_extra_bytes at the load window; do not add
    /// this projection's after-load retained_bytes again to that same window.
    pub fn new_retained_bytes(&self) -> Result<u64> {
        self.allocations
            .iter()
            .filter(|a| a.retained_output)
            .try_fold(0, |total, a| {
                checked_add(total, a.bytes()?, "new retained allocation sum")
            })
    }

    pub fn transient_extra_bytes(&self) -> Result<u64> {
        self.allocations
            .iter()
            .filter(|a| !a.retained_output)
            .try_fold(0, |total, a| {
                checked_add(total, a.bytes()?, "transient allocation sum")
            })
    }
}

/// One file's read/graph-construction quantum, before its source tensors are opened by MLX.
#[derive(Clone, Debug)]
pub struct AdapterFileReadFacts {
    pub path: std::path::PathBuf,
    pub source_bytes: u64,
    pub file_floor_bytes: u64,
    pub alpha_read_scratch_bytes: u64,
    pub graph_constructor_bytes: u64,
}

impl AdapterFileReadFacts {
    pub fn read_guard_bytes(&self) -> Result<u64> {
        checked_add(
            checked_add(
                self.file_floor_bytes.max(self.source_bytes),
                self.alpha_read_scratch_bytes,
                "file read plus alpha cast",
            )?,
            self.graph_constructor_bytes,
            "file read graph constructor sum",
        )
    }
}

pub(super) fn lokr_tucker_active(factors: &BTreeMap<String, String>) -> bool {
    !factors.contains_key("lokr_w2") && factors.contains_key("lokr_t2")
}

pub(super) fn loha_tucker_active(factors: &BTreeMap<String, String>) -> bool {
    factors.contains_key("hada_t1") && factors.contains_key("hada_t2")
}

#[derive(Default)]
struct Graph {
    allocations: Vec<AdapterLoadAllocation>,
    constructors: u64,
}

impl Graph {
    fn output(
        &mut self,
        operation: &'static str,
        shape: Vec<i32>,
        dtype: Dtype,
    ) -> Result<TensorFacts> {
        let allocation = AdapterLoadAllocation {
            operation,
            shape: shape.clone(),
            dtype,
            minimum_bytes: 0,
            retained_output: false,
        };
        allocation.bytes()?;
        let tensor = contiguous(shape, dtype)?;
        self.allocations.push(allocation);
        Ok(tensor)
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
        self.constructors = checked_add(
            self.constructors,
            checked_allocated(4)?,
            "adapter graph scalar sum",
        )?;
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
        if !matrix_layout(a)? {
            self.output("matmul operand layout copy", a.shape.clone(), dtype)?;
        }
        if !matrix_layout(b)? {
            self.output("matmul operand layout copy", b.shape.clone(), dtype)?;
        }
        self.output("matmul", shape, dtype)
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
                self.output(
                    "Tucker greedy contraction matmul",
                    axes.iter().map(|c| dims[c]).collect(),
                    dtype,
                )?
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
            // Native metal/reduce.cpp has one optional contiguous input copy, then at most one
            // partial array. all_reduce uses 32*REDUCE_N_READS or 1024*REDUCE_N_READS entries;
            // column/strided two-pass kernels use 32 or 128 rows times the output. REDUCE_N_READS
            // is 4 in kernels/defines.h. Take the max of these real mutually exclusive branches,
            // not a fitted multiplicative reserve. CPU needs no larger partial array.
            self.output(
                "Tucker reduction input normalization upper",
                tensor.shape.clone(),
                tensor.dtype,
            )?;
            let dtype = match tensor.dtype {
                Dtype::Bool | Dtype::Int8 | Dtype::Int16 => Dtype::Int32,
                Dtype::Uint8 | Dtype::Uint16 => Dtype::Uint32,
                dtype => dtype,
            };
            let out_shape: Vec<i32> = output.iter().map(|c| dims[c]).collect();
            let out_elements = checked_elements(&out_shape)?;
            let in_elements = checked_elements(&tensor.shape)?;
            // A genuinely empty output needs no partial reduction; this is an explicit native
            // empty-shape case, not an overflow/default fallback.
            let reduction = in_elements.checked_div(out_elements).unwrap_or(0);
            let all_partials = if in_elements > 4 * 1024 {
                if checked_tensor_bytes(&tensor)? <= 1 << 26 {
                    32 * 4
                } else {
                    1024 * 4
                }
            } else {
                0
            };
            let outer_blocks = if reduction >= 32768 { 128_i32 } else { 32_i32 };
            let column_partials = if in_elements == 0 {
                0
            } else {
                checked_mul(
                    outer_blocks as u64,
                    out_elements,
                    "Tucker reduction partial rows",
                )?
            };
            let partial_shape = if all_partials >= column_partials {
                vec![all_partials as i32]
            } else {
                std::iter::once(outer_blocks)
                    .chain(out_shape.iter().copied())
                    .collect()
            };
            self.output(
                "Tucker reduction partial-array branch upper",
                partial_shape,
                dtype,
            )?;
            tensor = self.output("Tucker naive reduction", out_shape, dtype)?;
            self.allocations
                .last_mut()
                .expect("just emitted reduction")
                .minimum_bytes = 4;
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
fn checked_allocated(bytes: u64) -> Result<u64> {
    if bytes > PAGE_BYTES {
        checked_mul(
            bytes.div_ceil(PAGE_BYTES),
            PAGE_BYTES,
            "adapter allocator rounding",
        )
    } else {
        Ok(bytes)
    }
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
        stride = stride
            .checked_mul(d as usize)
            .ok_or_else(|| Error::Unsupported("iris: adapter stride product overflow".into()))?;
    }
    Ok(TensorFacts {
        shape,
        dtype,
        strides,
    })
}
fn matrix_layout(t: &TensorFacts) -> Result<bool> {
    checked_tensor_bytes(t)?;
    let n = t.shape.len();
    if n == 2 && t.strides.is_empty() {
        return Ok(true);
    } // normalized native matrix-layout class
    if t.strides.len() != n {
        return Err(Error::Unsupported(
            "iris: adapter matrix stride/shape arity mismatch".into(),
        ));
    }
    Ok((n == 1 && t.strides[0] == 1)
        || (n >= 2
            && ((t.strides[n - 1] == 1 && t.strides[n - 2] == t.shape[n - 1] as usize)
                || (t.strides[n - 2] == 1 && t.strides[n - 1] == t.shape[n - 2] as usize))))
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

impl AdapterHeaderInventory {
    /// File sources are sized separately from retained deltas. Scalar alpha is read synchronously
    /// in scalar_alpha: at most one full F32 cast of a selected alpha tensor is live at a time.
    /// Graph scalar constants, in contrast, remain referenced by every lazy residual graph.
    pub fn file_read_facts(&self) -> Result<Vec<AdapterFileReadFacts>> {
        self.files
            .iter()
            .map(|file| {
                let one = AdapterHeaderInventory {
                    projections: self.projections.clone(),
                    files: vec![file.clone()],
                };
                let loads = one.projection_loads()?;
                let constructors = loads.values().try_fold(0, |total, l| {
                    checked_add(
                        total,
                        l.graph_constructor_bytes,
                        "file scalar constructor sum",
                    )
                })?;
                let mut alpha_scratch = 0;
                for key in &file.scalar_reads {
                    let alpha = &file.sources[key];
                    let count = checked_elements(&alpha.shape)?;
                    if count == 0 {
                        return Err(Error::Msg(format!(
                            "iris: alpha tensor {key} is present but empty"
                        )));
                    }
                    if alpha.dtype != Dtype::Float32 {
                        alpha_scratch = alpha_scratch.max(checked_allocated(checked_mul(
                            count,
                            4,
                            "alpha read cast bytes",
                        )?)?);
                    }
                }
                let read = AdapterFileReadFacts {
                    path: file.path.clone(),
                    source_bytes: file.sources.values().try_fold(0, |total, s| {
                        checked_add(
                            total,
                            checked_allocated(checked_tensor_bytes(s)?)?,
                            "file source allocation sum",
                        )
                    })?,
                    file_floor_bytes: mlx_gen::ios_quantum::load_floor_bytes(
                        std::slice::from_ref(&file.path),
                        |_| true,
                    )?,
                    alpha_read_scratch_bytes: alpha_scratch,
                    graph_constructor_bytes: constructors,
                };
                read.read_guard_bytes()?;
                Ok(read)
            })
            .collect()
    }

    /// Normalize the actual loader operations into per-projection evaluation windows. No MLX
    /// arrays are created. Same source dimensions/dtypes as the attached forward inventory.
    pub fn projection_loads(&self) -> Result<BTreeMap<String, ProjectionAdapterLoadFacts>> {
        let mut loads: BTreeMap<String, ProjectionAdapterLoadFacts> = self
            .projections
            .iter()
            .map(|(path, retained)| {
                (
                    path.clone(),
                    ProjectionAdapterLoadFacts {
                        retained: retained.clone(),
                        ..Default::default()
                    },
                )
            })
            .collect();
        for file in &self.files {
            for plan in &file.reconstructions {
                let mut graph = Graph::default();
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
                                .filter(|_| lokr_tucker_active(factors))
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
                        let tucker = loha_tucker_active(factors);
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
                let load = loads.get_mut(path).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris: reconstruction has no retained projection {path}"
                    ))
                })?;
                if let Some(delta) = delta {
                    let expected=load.retained.iter().find_map(|a|match a {AdapterFacts::Materialized {delta,..}=>Some(delta),_=>None})
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
                if let Some(final_output) = graph.allocations.last_mut() {
                    final_output.retained_output = true;
                }
                load.graph_constructor_bytes = checked_add(
                    load.graph_constructor_bytes,
                    graph.constructors,
                    "projection graph constructor sum",
                )?;
                load.allocations.extend(graph.allocations);
            }
        }
        // Never publish a clipped projection inventory, even when each individual site fits.
        for load in loads.values() {
            load.materialize_scratch_bytes()?;
            load.retained_bytes()?;
            load.new_retained_bytes()?;
            load.transient_extra_bytes()?;
        }
        Ok(loads)
    }
}
