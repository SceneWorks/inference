//! Metal-stream allocator-request lowering for the immutable native096 source.
//!
//! Citations below are relative to mlx-src/mlx/, or the canonical inference snapshot.
//! Each explicit backing is charged separately through ActiveCounterRequests::charge (U).
//! Alternatives are evaluated on cloned counters; only the greatest cumulative prefix survives.
//! NO donation credit, retirement, logical-price, RSS, or physical-footprint inference is used.
//! Kernel registers/threadgroup memory are NOT allocator::Buffer allocations and are excluded.
//! Host constructors and CPU-default safetensors loads share source laws with cpu.rs
//! (array.cpp/array.h; mlx-rs ops/io.rs:50,67; backend/common/load.cpp:30). A GPU task-local
//! override can force Load::eval_gpu, which throws before allocation (metal/primitives.cpp:155);
//! the successful CPU-default load's U(nbytes) dominates that failed branch. Descriptor
//! views use backend/common rules. No Metal primitive delegates to a CPU primitive recipe.
use super::cpu::{broadcast_shape, promote};
use super::*;
use mlx_gen::adapters::facts::dtype_width;

fn unsupported(contract: &str) -> Error {
    Error::Unsupported(format!("Z native Metal graph: {contract}"))
}
fn ceil(n: u64, d: u64) -> Result<u64> {
    if d == 0 {
        return Err(unsupported("zero divisor in source geometry"));
    }
    add(n / d, u64::from(n % d != 0))
}
fn floating(x: &Tensor) -> Result<()> {
    if matches!(x.dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32) {
        Ok(())
    } else {
        Err(unsupported(
            "served Metal floating primitive requires f16/bf16/f32",
        ))
    }
}
fn nonempty(x: &Tensor) -> Result<()> {
    if x.shape.contains(&0)
        || x.shape.len() != x.strides.len()
        || x.shape.iter().any(|&d| d > i32::MAX as u64)
    {
        return Err(unsupported(
            "nonempty native int32 dimensions and complete nonnegative strides required",
        ));
    }
    Ok(())
}

/// Counter maxima retain the UNION of output descriptors. A later primitive evaluates every
/// possible layout from the same conservative cumulative prefix, so selecting a large request
/// never erases another branch's potentially more expensive downstream contiguity copy.
pub(super) trait MetalBranchResult {
    fn merge_layouts(&mut self, other: Self) -> Result<()>;
}
impl MetalBranchResult for Tensor {
    fn merge_layouts(&mut self, other: Self) -> Result<()> {
        Tensor::merge_layouts(self, other)
    }
}
impl MetalBranchResult for (Tensor, Tensor) {
    fn merge_layouts(&mut self, other: Self) -> Result<()> {
        self.0.merge_layouts(other.0)?;
        self.1.merge_layouts(other.1)
    }
}
impl MetalBranchResult for [Tensor; 3] {
    fn merge_layouts(&mut self, other: Self) -> Result<()> {
        for (a, b) in self.iter_mut().zip(other) {
            a.merge_layouts(b)?;
        }
        Ok(())
    }
}

impl Graph {
    /// Both closures start at the SAME prefix. Never sum mutually exclusive algorithms, and
    /// never max raw bytes before page/cache rounding. Both output layouts survive.
    pub(super) fn metal_max<T: MetalBranchResult>(
        &mut self,
        a: impl FnOnce(&mut Graph) -> Result<T>,
        b: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        let mut ga = self.clone();
        let mut gb = self.clone();
        let mut va = a(&mut ga)?;
        let vb = b(&mut gb)?;
        va.merge_layouts(vb)?;
        *self = if ga.upper() >= gb.upper() { ga } else { gb };
        Ok(va)
    }
    fn metal_over_inputs<T: MetalBranchResult>(
        &mut self,
        inputs: &[&Tensor],
        f: impl Fn(&mut Graph, &[Tensor]) -> Result<T>,
    ) -> Result<T> {
        let mut combinations: Vec<Vec<Tensor>> = vec![Vec::new()];
        for input in inputs {
            let mut expanded = Vec::new();
            for prefix in combinations {
                for layout in input.layouts() {
                    let mut row = prefix.clone();
                    row.push(layout);
                    expanded.push(row);
                }
            }
            combinations = expanded;
        }
        let mut result: Option<T> = None;
        let mut best = self.clone();
        for inputs in combinations {
            let mut branch = self.clone();
            let out = f(&mut branch, &inputs)?;
            if let Some(result) = &mut result {
                result.merge_layouts(out)?;
            } else {
                result = Some(out);
            }
            if branch.upper() >= best.upper() {
                best = branch;
            }
        }
        *self = best;
        result.ok_or_else(|| unsupported("nonempty set of Metal descriptor alternatives"))
    }
    fn metal_copy(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        self.allocation(
            &x.shape,
            x.dtype,
            "backend/gpu/copy.cpp:37-40 contiguous_copy_gpu / metal/copy.cpp",
            site,
        )
    }
    fn metal_row(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if !x.alternatives.is_empty() {
            return self.metal_over_inputs(&[x], |g, v| g.metal_row(&v[0], site));
        }
        if x.row_contiguous() {
            Ok(x.clone())
        } else {
            self.metal_copy(x, site)
        }
    }
    pub(super) fn metal_reshape(
        &mut self,
        x: &Tensor,
        shape: &[u64],
        site: &str,
    ) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_reshape(&v[0], shape, site));
        }
        if shape.iter().any(|&d| d > i32::MAX as u64) {
            return Err(unsupported("Reshape native int32 target dimensions"));
        }
        if count(shape)? != x.elements()? {
            return Err(unsupported("Reshape equal element counts"));
        }
        // ops.cpp identity reshape is eliminated before creating the primitive.
        if shape == x.shape {
            return Ok(x.clone());
        }
        if x.row_contiguous() || x.elements()? == 0 {
            let mut out = Tensor::row(shape, x.dtype)?;
            out.data_size = x.data_size;
            out.contiguous = x.contiguous;
            return Ok(out);
        }
        if x.shape.is_empty() {
            return Ok(Tensor {
                shape: shape.into(),
                strides: vec![0; shape.len()],
                ..x.clone()
            });
        }
        if shape.contains(&0) {
            return Err(unsupported("nonempty reshape target dimensions"));
        }
        // Exact common/utils.cpp:88-111 collapse_contiguous_dims(in, INT32_MAX), then
        // common/common.cpp:148-198 prepare_reshape/shared_buffer_reshape. Respect size_cap;
        // singleton axes do not erase the source's first chunk/stride.
        let mut chunks = vec![(x.shape[0], x.strides[0])];
        for i in 1..x.shape.len() {
            if x.shape[i] == 1 {
                continue;
            }
            let last = chunks
                .last_mut()
                .ok_or_else(|| unsupported("reshape chunk"))?;
            let combined = mul(last.0, x.shape[i])?;
            if mul(x.strides[i], x.shape[i])? != last.1 || combined > i32::MAX as u64 {
                chunks.push((x.shape[i], x.strides[i]));
            } else {
                *last = (combined, x.strides[i]);
            }
        }
        let mut j = 0;
        let mut strides = Vec::new();
        for &d in shape {
            if j < chunks.len() && chunks[j].0 % d == 0 {
                chunks[j].0 /= d;
                strides.push(mul(chunks[j].0, chunks[j].1)?);
                if chunks[j].0 == 1 {
                    j += 1;
                }
            } else if d == 1 {
                strides.push(
                    *strides
                        .last()
                        .ok_or_else(|| unsupported("reshape singleton stride"))?,
                );
            } else {
                return self.allocation(
                    shape,
                    x.dtype,
                    "metal/copy.cpp:239-255 reshape General copy",
                    site,
                );
            }
        }
        Ok(Tensor {
            shape: shape.into(),
            strides,
            ..x.clone()
        })
    }
    pub(super) fn metal_contiguous(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_contiguous(&v[0], site));
        }
        // inference src/array.rs:52-70,113-130: contiguous two-factor regrouping,
        // and explicit Contiguous if b>1 or the original shape already equals [a,1].
        let mut a = 1;
        for &d in &x.shape {
            let next = mul(a, d)?;
            if next > i32::MAX as u64 {
                break;
            }
            a = next;
        }
        a = a.max(1);
        let b = (x.elements()? / a).max(1);
        if b > i32::MAX as u64 {
            return Err(unsupported(
                "array::contiguous residual flat_2d factor must fit int32",
            ));
        }
        let flat = self.reshape(x, &[a, b], site)?;
        let out = self.reshape(&flat, &x.shape, site)?;
        if b == 1 && x.shape != [a, 1] {
            return Ok(out);
        }
        // gpu/primitives.cpp:51-62: even row-contiguous inputs may be copied when the
        // cached MTLBuffer capacity exceeds nbytes+16384. Capacity is runtime-only;
        // choose the non-donated copy alternative, never assume exact backing capacity.
        let mut copied = self.metal_copy(&out, site)?;
        if out.row_contiguous() {
            copied.merge_layouts(out)?;
        }
        Ok(copied)
    }
    pub(super) fn metal_cast(&mut self, x: &Tensor, dtype: Dtype, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_cast(&v[0], dtype, site));
        }
        if x.dtype == dtype {
            return Ok(x.clone());
        }
        // gpu/primitives.cpp:28-33 -> gpu/copy.cpp:15-35 -> common/copy.h.
        if x.contiguous {
            self.counter.charge(
                "gpu/primitives.cpp:28-33 AsType Vector, non-donated",
                site,
                mul(x.data_size, dtype_width(dtype))?,
            )?;
            Ok(Tensor { dtype, ..x.clone() })
        } else {
            self.allocation(
                &x.shape,
                dtype,
                "gpu/primitives.cpp:28-33 AsType General",
                site,
            )
        }
    }
    pub(super) fn metal_unary(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_unary(&v[0], site));
        }
        // metal/unary.cpp calls common/unary.h::set_unary_output_data.
        if x.contiguous {
            self.counter.charge(
                "metal/unary.cpp / common/unary.h non-donated data_size",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x.clone())
        } else {
            self.allocation(&x.shape, x.dtype, "metal/unary.cpp General output", site)
        }
    }
    pub(super) fn metal_binary(&mut self, a: &Tensor, b: &Tensor, site: &str) -> Result<Tensor> {
        if [a, b].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[a, b], |g, v| g.metal_binary(&v[0], &v[1], site));
        }
        let shape = broadcast_shape(&a.shape, &b.shape)?;
        let dtype = promote(a.dtype, b.dtype)?;
        let a = self.cast(a, dtype, site)?.broadcast(&shape)?;
        let b = self.cast(b, dtype, site)?.broadcast(&shape)?;
        // metal/binary.cpp -> common/binary.h:21-110. These are shared backend laws,
        // not an assumption that Metal arithmetic lowers to CPU arithmetic.
        let col = |x: &Tensor| -> Result<bool> {
            let mut s = 1;
            for (&d, &st) in x.shape.iter().zip(&x.strides) {
                if d != 1 && st != s {
                    return Ok(false);
                }
                s = mul(s, d)?;
            }
            Ok(true)
        };
        let model = if a.data_size == 1 && b.data_size == 1 {
            Some(&a)
        } else if a.data_size == 1 && b.contiguous {
            Some(&b)
        } else if b.data_size == 1 && a.contiguous {
            Some(&a)
        } else if (a.row_contiguous() && b.row_contiguous()) || (col(&a)? && col(&b)?) {
            Some(&a)
        } else {
            None
        };
        if let Some(x) = model {
            self.counter.charge(
                "metal/binary.cpp / common/binary.h non-donated output",
                site,
                mul(x.data_size, dtype_width(dtype))?,
            )?;
            Ok(x.clone())
        } else {
            self.allocation(&shape, dtype, "metal/binary.cpp General output", site)
        }
    }
    pub(super) fn metal_concat(
        &mut self,
        xs: &[Tensor],
        axis: usize,
        site: &str,
    ) -> Result<Tensor> {
        if xs.iter().any(|x| !x.alternatives.is_empty()) {
            let inputs: Vec<_> = xs.iter().collect();
            return self.metal_over_inputs(&inputs, |g, v| g.metal_concat(v, axis, site));
        }
        let first = xs.first().ok_or_else(|| unsupported("empty Concatenate"))?;
        if xs.len() == 1 {
            return Ok(first.clone());
        }
        let mut shape = first.shape.clone();
        let extent = shape
            .get_mut(axis)
            .ok_or_else(|| unsupported("Concatenate axis"))?;
        *extent = 0;
        let mut dtype = first.dtype;
        for x in xs {
            if x.shape.len() != shape.len()
                || x.shape
                    .iter()
                    .enumerate()
                    .any(|(i, d)| i != axis && *d != shape[i])
            {
                return Err(unsupported("Concatenate incompatible dimensions"));
            }
            shape[axis] = add(shape[axis], x.shape[axis])?;
            dtype = promote(dtype, x.dtype)?;
        }
        for x in xs {
            self.cast(x, dtype, site)?;
        }
        self.allocation(
            &shape,
            dtype,
            "gpu/primitives.cpp:45-48 / gpu/slicing.cpp concatenate_gpu output",
            site,
        )
    }
    pub(super) fn metal_reduce(
        &mut self,
        x: &Tensor,
        axes: &[usize],
        site: &str,
    ) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_reduce(&v[0], axes, site));
        }
        nonempty(x)?;
        floating(x)?; // All reductions selected by Z are floating (statistics/normalization).
        if axes.iter().any(|&a| a >= x.shape.len()) || axes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(unsupported("Reduce sorted unique in-range axes"));
        }
        // ops.cpp reduction identity is removed before constructing a primitive.
        if axes.iter().all(|&a| x.shape[a] == 1) {
            return Ok(x.clone());
        }
        let mut shape = x.shape.clone();
        for &a in axes {
            shape[a] = 1;
        }
        let out = Tensor::row(&shape, x.dtype)?;
        self.counter.charge(
            "metal/reduce.cpp:968-971 minimum four-byte Reduce output",
            site,
            out.nbytes()?.max(4),
        )?;
        self.metal_reduce_work(x, axes, &out, site)?;
        Ok(out)
    }
    fn metal_reduce_work(
        &mut self,
        x: &Tensor,
        axes: &[usize],
        out: &Tensor,
        site: &str,
    ) -> Result<()> {
        // Port of common/reduce.cpp:29-171. Row reductions have no allocator temporary;
        // all-reduce and strided reductions have exactly the multipass backings below.
        if x.elements()? == x.data_size && axes.len() == x.shape.len() && x.contiguous {
            if x.elements()? > 4096 {
                let rows = if x.nbytes()? <= 1 << 26 { 128 } else { 4096 };
                self.allocation(
                    &[rows],
                    out.dtype,
                    "metal/reduce.cpp:344-361 all_reduce intermediate",
                    site,
                )?;
            }
            return Ok(());
        }
        let mut reductions: Vec<(u64, u64)> = Vec::new();
        if x.row_contiguous() {
            let mut previous = None;
            for &a in axes {
                if previous.is_some() && previous == a.checked_sub(1) && x.shape[a] > 1 {
                    if let Some(last) = reductions.last_mut() {
                        last.0 = mul(last.0, x.shape[a])?;
                        last.1 = x.strides[a];
                    }
                } else {
                    reductions.push((x.shape[a], x.strides[a]));
                }
                previous = Some(a);
            }
            reductions.retain(|&(d, _)| d != 1);
        } else {
            reductions = axes
                .iter()
                .filter(|&&a| x.shape[a] > 1)
                .map(|&a| (x.shape[a], x.strides[a]))
                .collect();
            reductions.sort_by(|a, b| match (a.1 == 0, b.1 == 0) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b.1.cmp(&a.1),
            });
            for i in (1..reductions.len()).rev() {
                let a = reductions[i];
                let b = reductions[i - 1];
                if b.1 == mul(a.0, a.1)? {
                    reductions[i - 1] = (mul(a.0, b.0)?, a.1);
                    reductions.remove(i);
                }
            }
        }
        let &(_, stride) = reductions
            .last()
            .ok_or_else(|| unsupported("Reduce nontrivial reduction plan"))?;
        if stride == 1 {
            return Ok(());
        }
        let mut strided = x.row_contiguous();
        if !strided && stride > 1 {
            let mut size = 1;
            let mut expand = false;
            for i in (0..x.shape.len()).rev() {
                if axes.last() == Some(&i) {
                    continue;
                }
                if x.strides[i] == 0 {
                    if x.shape[i] == 1 {
                        continue;
                    }
                    expand = true;
                    break;
                }
                if x.strides[i] != size && x.shape[i] != 1 {
                    break;
                }
                size = mul(size, x.shape[i])?;
            }
            strided = size > stride || (size == stride && !expand);
        }
        if !strided {
            let copied = self.metal_copy(x, site)?;
            return self.metal_reduce_work(&copied, axes, out, site);
        }
        let total = reductions.iter().try_fold(1, |n, &(d, _)| mul(n, d))?;
        let blocks = if total < 32 {
            0
        } else if stride < 32 && total >= 1024 {
            if total >= 32768 {
                128
            } else {
                32
            }
        } else if total > 256 && out.elements()? / 32 < 1024 {
            32
        } else {
            0
        };
        if blocks != 0 {
            self.allocation(
                &[blocks, out.elements()?],
                out.dtype,
                "metal/reduce.cpp:636-654,804-825,919-948 column multipass intermediate",
                site,
            )?;
        }
        Ok(())
    }
    pub(super) fn metal_softmax(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_softmax(&v[0], site));
        }
        nonempty(x)?;
        floating(x)?;
        if x.contiguous && x.strides.last() == Some(&1) {
            self.counter.charge(
                "metal/softmax.cpp:25-39 non-donated data_size output",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x.clone())
        } else {
            // The contiguous copy IS the output; do not allocate it twice.
            self.metal_copy(x, site)
        }
    }
    pub(super) fn metal_matrix_product(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        let mut inputs = vec![a, b];
        if let Some(extra) = bias {
            inputs.push(extra);
        }
        if inputs.iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&inputs, |g, v| {
                g.metal_matrix_product(
                    &v[0],
                    &v[1],
                    if bias.is_some() { Some(&v[2]) } else { None },
                    site,
                )
            });
        }
        nonempty(a)?;
        nonempty(b)?;
        if a.shape.len() < 2
            || b.shape.len() < 2
            || a.shape[a.shape.len() - 1] != b.shape[b.shape.len() - 2]
        {
            return Err(unsupported(
                "Matmul compatible matrix ranks/inner dimensions",
            ));
        }
        let mut dtype = promote(a.dtype, b.dtype)?;
        if let Some(c) = bias {
            dtype = promote(dtype, c.dtype)?;
        }
        let mut a = self.cast(a, dtype, site)?;
        let mut b = self.cast(b, dtype, site)?;
        floating(&a)?;
        let original = a.shape.clone();
        let flatten =
            a.shape.len() > 2 && b.shape.len() == 2 && bias.is_none_or(|c| c.shape.len() <= 1);
        if flatten {
            a = self.reshape(
                &a,
                &[
                    count(&original[..original.len() - 1])?,
                    original[original.len() - 1],
                ],
                site,
            )?;
        }
        let ar = a.shape.len();
        let br = b.shape.len();
        let m = a.shape[ar - 2];
        let k = a.shape[ar - 1];
        let n = b.shape[br - 1];
        let mut shape = broadcast_shape(&a.shape[..ar - 2], &b.shape[..br - 2])?;
        let batch = count(&shape)?;
        shape.extend([m, n]);
        let c = if let Some(c) = bias {
            let c = self.cast(c, dtype, site)?;
            if broadcast_shape(&c.shape, &shape)? != shape {
                return Err(unsupported("AddMM broadcast bias"));
            }
            Some(c.broadcast(&shape)?)
        } else {
            None
        };
        // ops.cpp:3365-3367,6293-6310 broadcasts BEFORE check_transpose. A general
        // copy must therefore cover the expanded logical batch, not the compact source shape.
        if b.shape.len() > 2 || (!flatten && bias.is_some() && a.shape.len() > 2) {
            let mut ashape = shape.clone();
            *ashape
                .last_mut()
                .ok_or_else(|| unsupported("Matmul broadcast rank"))? = k;
            a = a.broadcast(&ashape)?;
            let mut bshape = shape.clone();
            let matrix_axis = bshape.len() - 2;
            bshape[matrix_axis] = k;
            b = b.broadcast(&bshape)?;
        }
        let out = self.allocation(
            &shape,
            dtype,
            "metal/matmul.cpp:1256,1376 Matmul/AddMM output",
            site,
        )?;
        let mut prepared = Vec::new();
        for (x, vector) in [(&a, m == 1), (&b, n == 1)] {
            let r = x.shape.len();
            let sx = x.strides[r - 2];
            let sy = x.strides[r - 1];
            if !((sy == 1 && (!vector || sx == x.shape[r - 1]))
                || (sx == 1 && (!vector || sy == x.shape[r - 2])))
            {
                prepared.push(self.metal_copy(x, site)?);
            } else {
                prepared.push(x.clone());
            }
        }
        // Frontend flatten above covers the served unbatched-weight path. Nontrivial batched
        // products in Z SDPA have batch>1 and cannot use split-K (weights carry nonzero strides).
        if batch == 1 && m.min(n) > 1 {
            self.metal_splitk(m, n, k, site)?;
        } else if batch > 1 {
            // matmul.cpp:1274-1295,1398-1426 collapse_batches: all leading dimensions
            // jointly collapse only when A advances by M*K, B is broadcast, and C (if
            // present) advances by M*its row stride. Use the AFTER-copy descriptors.
            let a = &prepared[0];
            let b = &prepared[1];
            let b_broadcast = b.shape[..b.shape.len() - 2]
                .iter()
                .zip(&b.strides)
                .all(|(&d, &st)| d == 1 || st == 0);
            let mut c_collapsible = true;
            if let Some(c) = &c {
                let mut expected = mul(m, c.strides[c.shape.len() - 2])?;
                for i in (0..c.shape.len() - 2).rev() {
                    if c.shape[i] != 1 {
                        c_collapsible &= c.strides[i] == expected;
                        expected = mul(expected, c.shape[i])?;
                    }
                }
            }
            if a.row_contiguous() && b_broadcast && c_collapsible && mul(batch, m)?.min(n) > 1 {
                self.metal_splitk(mul(batch, m)?, n, k, site)?;
            }
        }
        if flatten {
            let mut restored = original;
            let last = restored
                .last_mut()
                .ok_or_else(|| unsupported("Matmul restored rank"))?;
            *last = n;
            self.reshape(&out, &restored, site)
        } else {
            Ok(out)
        }
    }
    /// Source-possible SIMD/NAX split-K MAX; regular GEMM/GEMV have no Array workspace.
    fn metal_splitk(&mut self, m: u64, n: u64, k: u64, site: &str) -> Result<()> {
        let mut partitions = 0;
        if mul(ceil(m, 16)?, ceil(n, 16)?)? <= 2048 && k / 16 >= 8 && k >= m.max(n) {
            let tiles = mul(ceil(m, 32)?, ceil(n, 32)?)?;
            let target = (k / 16) / tiles;
            partitions = target
                .checked_next_power_of_two()
                .ok_or_else(|| unsupported("SIMD split-K power overflow"))?
                .clamp(2, 32);
        }
        if k >= mul(3, m.max(n))? || (m.max(n) <= 1024 && k > mul(2, m.max(n))?) {
            let size = if k <= 1024 {
                k / 2
            } else if k <= 2048 {
                1024
            } else if k <= 4096 {
                2048
            } else {
                4096
            };
            partitions = partitions.max(ceil(k, size)?);
        }
        if partitions > 0 {
            self.allocation(
                &[partitions, m, n],
                Dtype::Float32,
                "metal/matmul.cpp:501-542,658-702,915-980 max SIMD/NAX C_split",
                site,
            )?;
        }
        Ok(())
    }
    pub(super) fn metal_qmm(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        scales: &Tensor,
        offsets: &Tensor,
        bias: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        let mut inputs = vec![x, weight, scales, offsets];
        if let Some(extra) = bias {
            inputs.push(extra);
        }
        if inputs.iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&inputs, |g, v| {
                g.metal_qmm(
                    &v[0],
                    &v[1],
                    &v[2],
                    &v[3],
                    if bias.is_some() { Some(&v[4]) } else { None },
                    site,
                )
            });
        }
        nonempty(x)?;
        nonempty(weight)?;
        if weight.shape.len() != 2 || x.shape.is_empty() {
            return Err(unsupported("Z QMM rank-2 weights, nonempty x"));
        }
        let dtype = promote(x.dtype, promote(scales.dtype, offsets.dtype)?)?;
        let x = self.cast(x, dtype, site)?;
        floating(&x)?;
        let scales = self.cast(scales, dtype, site)?;
        let offsets = self.cast(offsets, dtype, site)?;
        let mut prepared = Vec::new();
        for input in [&x, weight, &scales, &offsets] {
            let r = input.shape.len();
            if r == 0 {
                return Err(unsupported("QMM matrix input rank"));
            }
            let contiguous = if r == 1 {
                input.strides[0] == 1
            } else {
                input.strides[r - 2] == input.shape[r - 1] && input.strides[r - 1] == 1
            };
            prepared.push(if contiguous {
                input.clone()
            } else {
                self.metal_copy(input, site)?
            });
        }
        let x = &prepared[0];
        let k = *x
            .shape
            .last()
            .ok_or_else(|| unsupported("QMM x inner dimension"))?;
        let n = weight.shape[0];
        if weight.dtype != Dtype::Uint32
            || scales.shape != offsets.shape
            || scales.shape.len() != 2
            || scales.shape[0] != n
            || k == 0
            || mul(weight.shape[1], 32)? % k != 0
            || !matches!(mul(weight.shape[1], 32)? / k, 4 | 8)
        {
            return Err(unsupported(
                "QMM uint32 packed rank-two weights, matching affine scale/offset rows, bits4/8",
            ));
        }
        let sc = *scales
            .shape
            .last()
            .ok_or_else(|| unsupported("QMM scales group dimension"))?;
        if sc == 0 || k % sc != 0 || k / sc != 64 {
            return Err(unsupported("QMM affine group size 64 descriptor contract"));
        }
        let mut shape = x.shape.clone();
        *shape
            .last_mut()
            .ok_or_else(|| unsupported("QMM output rank"))? = n;
        // QuantizedMatmul::eval_gpu, NOT QQMatmul/GatherQMM: frontend transpose=true,
        // rank-two weights, no index inputs. qvm_split_k is transpose=false and unreachable.
        let normal = |g: &mut Graph| -> Result<Tensor> {
            let out = g.allocation(
                &shape,
                dtype,
                "metal/quantized.cpp:1501 QuantizedMatmul output",
                site,
            )?;
            let m = if x.row_contiguous() {
                x.elements()? / k
            } else {
                x.shape[x.shape.len() - 2]
            };
            let batches = out.elements()? / mul(m, n)?;
            // Across all architectures get_qmv_batch_limit minimum is 14/10/6.
            let limit = if k <= 2048 && n <= 2048 {
                14
            } else if k <= 4096 && n <= 4096 {
                10
            } else {
                6
            };
            if m >= limit && batches == 1 {
                let tiles = mul(ceil(n, 32)?, ceil(m, 32)?)?;
                let mut splits = (512 / tiles).max(1).min(k / 64);
                while splits > 1 && k % mul(splits, 64)? != 0 {
                    splits -= 1;
                }
                if splits > 1 {
                    let intermediate = g.allocation(
                        &[splits, out.elements()?],
                        dtype,
                        "metal/quantized.cpp:891-923 qmm_splitk intermediate",
                        site,
                    )?;
                    // qmm_splitk reuses out, but its reduction can have its OWN second-pass backing.
                    g.metal_reduce_work(&intermediate, &[0], &out, site)?;
                }
            }
            match bias {
                Some(b) => g.binary(&out, b, site),
                None => Ok(out),
            }
        };
        if let Some(b) = bias.filter(|b| b.dtype == dtype) {
            self.metal_max(normal, |g| {
                g.metal_row(b, site)?;
                g.allocation(
                    &shape,
                    dtype,
                    "metal/quantized.cpp:1501-1539 fused output-bias QMM (no split-K)",
                    site,
                )
            })
        } else {
            normal(self)
        }
    }
    pub(super) fn metal_rms(&mut self, x: &Tensor, weight: &Tensor, site: &str) -> Result<Tensor> {
        if [x, weight].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x, weight], |g, v| g.metal_rms(&v[0], &v[1], site));
        }
        nonempty(x)?;
        floating(x)?;
        let dtype = promote(x.dtype, weight.dtype)?;
        if weight.shape
            != [*x
                .shape
                .last()
                .ok_or_else(|| unsupported("RMSNorm input rank"))?]
        {
            return Err(unsupported(
                "RMSNorm rank-one weight matching normalized dimension",
            ));
        }
        let x = self.cast(x, dtype, site)?;
        self.cast(weight, dtype, site)?;
        self.metal_norm_output(&x, site)
    }
    fn metal_norm_output(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        let r = x.shape.len();
        if r == 0 {
            return Err(unsupported("RMSNorm/LayerNorm positive rank"));
        }
        let mut no_copy = x.contiguous && x.strides[r - 1] == 1;
        if r > 1 {
            no_copy &=
                x.strides[r - 2] == 0 || x.strides[r - 2] == x.shape[r - 1] || x.shape[r - 2] == 1;
        }
        if no_copy {
            self.counter.charge(
                "metal/normalization.cpp:18-50,213-245 non-donated fused norm output",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x.clone())
        } else {
            self.metal_copy(x, site)
        } // copy shared with output, no second request
    }
    pub(super) fn metal_layer_norm(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_layer_norm(&v[0], site));
        }
        nonempty(x)?;
        floating(x)?;
        // fast.cpp creates absent affine scalar arrays before constructing the fused primitive.
        self.scalar(x.dtype, site)?;
        self.scalar(x.dtype, site)?;
        self.metal_norm_output(x, site)
    }
    pub(super) fn metal_silu(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_silu(&v[0], site));
        }
        // nn.rs:847-860 exact-epilogue toggle; failed native selection falls back to eager.
        // fast.cpp:459-488 rejects empty, >UINT32_MAX, and non-floating exact activations.
        if x.elements()? == 0
            || x.elements()? > u32::MAX as u64
            || !matches!(x.dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
        {
            let sigmoid = self.unary(x, site)?;
            return self.binary(x, &sigmoid, site);
        }
        self.metal_max(
            |g| {
                let sigmoid = g.unary(x, site)?;
                g.binary(x, &sigmoid, site)
            },
            |g| {
                let x = if x.contiguous && x.elements()? == x.data_size {
                    x.clone()
                } else {
                    g.metal_copy(x, site)?
                };
                g.counter.charge(
                    "metal/normalization.cpp:391-410 ExactActivation non-donated output",
                    site,
                    mul(x.data_size, dtype_width(x.dtype))?,
                )?;
                Ok(x)
            },
        )
    }
    pub(super) fn metal_group_norm(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        groups: u64,
        site: &str,
    ) -> Result<Tensor> {
        if [x, weight, bias].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x, weight, bias], |g, v| {
                g.metal_group_norm(&v[0], &v[1], &v[2], groups, site)
            });
        }
        nonempty(x)?;
        let r = x.shape.len();
        if r < 2 || groups == 0 || x.shape[r - 1] % groups != 0 {
            return Err(unsupported("GroupNorm channels/groups"));
        }
        let batch = x.shape[0];
        let channels = x.shape[r - 1];
        let spatial = count(&x.shape[1..r - 1])?;
        let eager = |g: &mut Graph| -> Result<Tensor> {
            let grouped = g
                .reshape(x, &[batch, spatial, groups, channels / groups], site)?
                .transpose(&[0, 2, 1, 3])?;
            let grouped = g.reshape(
                &grouped,
                &[batch, groups, mul(spatial, channels / groups)?],
                site,
            )?;
            let grouped = g.layer_norm(&grouped, site)?;
            let grouped = g
                .reshape(&grouped, &[batch, groups, spatial, channels / groups], site)?
                .transpose(&[0, 2, 1, 3])?;
            let normalized = g.reshape(&grouped, &x.shape, site)?;
            let out = g.binary(&normalized, weight, site)?;
            g.binary(&out, bias, site)
        };
        if x.dtype != weight.dtype
            || x.dtype != bias.dtype
            || mul(batch, groups)? > u32::MAX as u64
            || mul(spatial, channels / groups)? > i32::MAX as u64
        {
            return eager(self);
        }
        self.metal_max(eager, |g| {
            // nn.rs:1058-1072 normalizes input layout BEFORE native fused GroupNormAffine.
            let normalized = g.metal_contiguous(x, site)?;
            let x = g.metal_row(&normalized, site)?;
            g.metal_row(weight, site)?;
            g.metal_row(bias, site)?;
            g.counter.charge(
                "metal/normalization.cpp:302-328 GroupNormAffine non-donated output",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x)
        })
    }
    fn metal_compiled_outputs(
        &mut self,
        inputs: &[&Tensor],
        shape: &[u64],
        dtype: Dtype,
        outputs: usize,
        site: &str,
    ) -> Result<Vec<Tensor>> {
        // metal/compiled.cpp:257-418 delegates to common/compiled.cpp:93-179.
        // Tape temporaries live in kernel registers, not allocator::Buffer backings.
        // Rank-zero alone is scalar (common/compiled.h:65); broadcast [1,...] is NOT scalar.
        let non_scalar: Vec<_> = inputs
            .iter()
            .copied()
            .filter(|x| !x.shape.is_empty())
            .collect();
        let all_row = non_scalar
            .iter()
            .all(|x| x.shape == shape && x.row_contiguous());
        let mut all_col = true;
        for x in &non_scalar {
            let mut stride = 1;
            all_col &= x.shape == shape;
            for (&d, &st) in x.shape.iter().zip(&x.strides) {
                all_col &= d == 1 || st == stride;
                stride = mul(stride, d)?;
            }
        }
        let contiguous = match non_scalar.len() {
            0 => shape.is_empty(),
            1 => non_scalar[0].shape == shape && non_scalar[0].contiguous,
            _ => all_row || all_col,
        };
        let representative = inputs.iter().copied().find(|x| x.shape == shape);
        let mut result = Vec::new();
        for _ in 0..outputs {
            if contiguous {
                let x = representative.ok_or_else(|| {
                    unsupported(
                        "Compiled contiguous output requires a same-shape representative input",
                    )
                })?;
                self.counter.charge(
                    "metal/compiled.cpp:418 / common/compiled.cpp:155-160 non-donated output",
                    site,
                    mul(x.data_size, dtype_width(dtype))?,
                )?;
                result.push(Tensor { dtype, ..x.clone() });
            } else {
                result.push(self.allocation(
                    shape,
                    dtype,
                    "metal/compiled.cpp:418 / common/compiled.cpp:179 strided output",
                    site,
                )?);
            }
        }
        Ok(result)
    }
    pub(super) fn metal_gated(
        &mut self,
        x: &Tensor,
        gate: &Tensor,
        normed: &Tensor,
        site: &str,
    ) -> Result<Tensor> {
        if [x, gate, normed].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x, gate, normed], |g, v| {
                g.metal_gated(&v[0], &v[1], &v[2], site)
            });
        }
        // z-image/transformer_block.rs:23-71: retained/one-shot have the SAME Compiled law.
        let shape = broadcast_shape(&x.shape, &broadcast_shape(&gate.shape, &normed.shape)?)?;
        let dtype = promote(x.dtype, promote(gate.dtype, normed.dtype)?)?;
        self.metal_max(
            |g| {
                let product = g.binary(gate, normed, site)?;
                g.binary(x, &product, site)
            },
            |g| {
                g.metal_compiled_outputs(&[x, gate, normed], &shape, dtype, 1, site)?
                    .pop()
                    .ok_or_else(|| unsupported("Compiled gated output"))
            },
        )
    }
    pub(super) fn metal_rope_rotate(
        &mut self,
        real: &Tensor,
        imag: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        site: &str,
    ) -> Result<(Tensor, Tensor)> {
        if [real, imag, cos, sin]
            .iter()
            .any(|x| !x.alternatives.is_empty())
        {
            return self.metal_over_inputs(&[real, imag, cos, sin], |g, v| {
                g.metal_rope_rotate(&v[0], &v[1], &v[2], &v[3], site)
            });
        }
        // Z does NOT call fast::rope: TE RoPE is composed; DiT attention.rs:250-294
        // compiles exactly these six elementwise ops with TWO escaping outputs.
        let shape = broadcast_shape(&real.shape, &cos.shape)?;
        let dtype = promote(real.dtype, cos.dtype)?;
        self.metal_max(
            |g| {
                let ac = g.binary(real, cos, site)?;
                let bs = g.binary(imag, sin, site)?;
                let r = g.binary(&ac, &bs, site)?;
                let as_ = g.binary(real, sin, site)?;
                let bc = g.binary(imag, cos, site)?;
                let i = g.binary(&as_, &bc, site)?;
                Ok((r, i))
            },
            |g| {
                let outputs =
                    g.metal_compiled_outputs(&[real, imag, cos, sin], &shape, dtype, 2, site)?;
                let [r, i]: [Tensor; 2] = outputs
                    .try_into()
                    .map_err(|_| unsupported("Compiled RoPE two-output contract"))?;
                Ok((r, i))
            },
        )
    }
    pub(super) fn metal_sdpa(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        let mut inputs = vec![q, k, v];
        if let Some(extra) = mask {
            inputs.push(extra);
        }
        if inputs.iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&inputs, |g, v| {
                g.metal_sdpa(
                    &v[0],
                    &v[1],
                    &v[2],
                    if mask.is_some() { Some(&v[3]) } else { None },
                    site,
                )
            });
        }
        for x in [q, k, v] {
            nonempty(x)?;
            floating(x)?;
        }
        if q.shape.len() != 4
            || k.shape.len() != 4
            || v.shape.len() != 4
            || q.shape[0] != k.shape[0]
            || k.shape[0] != v.shape[0]
            || q.shape[1] != k.shape[1]
            || k.shape[1] != v.shape[1]
            || q.shape[3] != k.shape[3]
            || k.shape[2] != v.shape[2]
        {
            return Err(unsupported(
                "SDPA provider-expanded equal heads and compatible [B,H,L,D] descriptors",
            ));
        }
        let dtype = promote(promote(q.dtype, k.dtype)?, v.dtype)?;
        let q = self.cast(q, dtype, site)?;
        let k = self.cast(k, dtype, site)?;
        let v = self.cast(v, dtype, site)?;
        let mask = if let Some(mask) = mask {
            if !matches!(
                mask.dtype,
                Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32
            ) {
                return Err(unsupported("Z SDPA additive floating mask; boolean/causal/sink graphs are not selected by this frontend"));
            }
            if promote(mask.dtype, dtype)? != dtype {
                return Err(unsupported("SDPA mask must promote to QKV dtype"));
            }
            Some(
                self.cast(mask, dtype, site)?
                    .broadcast(&[q.shape[0], q.shape[1], q.shape[2], k.shape[2]])?,
            )
        } else {
            None
        };
        let mask = mask.as_ref();
        let d = q.shape[3];
        let vd = v.shape[3];
        let l = q.shape[2];
        let vector = l <= 8
            && l <= k.shape[2]
            && ((d == vd && matches!(d, 64 | 96 | 128 | 256)) || (d == 192 && vd == 128));
        let full = l > 8 && d == vd && matches!(d, 64 | 80 | 128);
        if !vector && !full {
            // fast.cpp SDPA fallback, evaluated entirely with Metal primitives.
            let scale = self.scalar(dtype, site)?;
            let qs = self.binary(&q, &scale, site)?;
            let mut scores = self.matmul(&qs, &k.transpose(&[0, 1, 3, 2])?, site)?;
            if let Some(mask) = mask {
                scores = self.binary(&scores, mask, site)?;
            }
            let probs = self.softmax(&scores, site)?;
            return self.matmul(&probs, &v, site);
        }
        let shape = [q.shape[0], q.shape[1], l, vd];
        if vector {
            let q_ok = q.row_contiguous()
                || ((q.shape[0] == 1 || q.shape[1] == 1) && {
                    let bidx = if q.shape[0] == 1 { 1 } else { 0 };
                    q.strides[3] == 1
                        && q.strides[2] == mul(q.shape[3], q.shape[bidx])?
                        && q.strides[bidx] == q.shape[3]
                });
            if !q_ok {
                self.metal_copy(&q, site)?;
            }
            for x in [&k, &v] {
                let ok = x.strides[3] == 1
                    && (x.shape[0] == 1
                        || x.shape[1] == 1
                        || x.strides[0] == mul(x.strides[1], x.shape[1])?);
                if !ok {
                    self.metal_copy(x, site)?;
                }
            }
            if let Some(mask) = mask {
                if mask.shape.len() != 4 {
                    return Err(unsupported("SDPA broadcast rank-four mask descriptor"));
                }
                if !(mask.row_contiguous()
                    || q.shape[0] == 1
                    || q.shape[1] == 1
                    || mask.strides[0] == mul(mask.strides[1], mask.shape[1])?)
                {
                    self.metal_copy(mask, site)?;
                }
            }
            let out = self.allocation(
                &shape,
                dtype,
                "metal/scaled_dot_product_attention.cpp:724-730 vector output",
                site,
            )?;
            // Architecture chooses one or two passes. The environment override is a positive
            // C++ int, not a measured/default tuning value. MAX is INT_MAX, not 32/64/256.
            if k.shape[2] >= 1024 {
                self.metal_sdpa_intermediates(&out, i32::MAX as u64, site)?;
            }
            Ok(out)
        } else {
            for x in [&q, &k, &v] {
                if x.strides[3] != 1 {
                    self.metal_copy(x, site)?;
                }
            }
            if let Some(mask) = mask {
                if mask.strides.last() != Some(&1) {
                    self.metal_copy(mask, site)?;
                }
            }
            let mut out = self.allocation(
                &shape,
                dtype,
                "metal/scaled_dot_product_attention.cpp:759-780 full output",
                site,
            )?;
            // Full SDPA writes sequence-major backing [B,L,H,D], exposed as [B,H,L,D].
            out.strides = vec![mul(mul(l, q.shape[1])?, vd)?, vd, mul(q.shape[1], vd)?, 1];
            Ok(out)
        }
    }
    fn metal_sdpa_intermediates(&mut self, out: &Tensor, blocks: u64, site: &str) -> Result<()> {
        let r = out.shape.len();
        if r == 0 {
            return Err(unsupported("SDPA intermediate positive output rank"));
        }
        let mut shape = out.shape[..r - 1].to_vec();
        shape.extend([blocks, out.shape[r - 1]]);
        self.allocation(
            &shape,
            out.dtype,
            "metal/scaled_dot_product_attention.cpp:488-498 intermediate",
            site,
        )?;
        shape.pop();
        self.allocation(
            &shape,
            Dtype::Float32,
            "metal/scaled_dot_product_attention.cpp:496-499 sums",
            site,
        )?;
        self.allocation(
            &shape,
            Dtype::Float32,
            "metal/scaled_dot_product_attention.cpp:497-500 maxs",
            site,
        )?;
        Ok(())
    }
    pub(super) fn metal_normal_f32(&mut self, shape: &[u64], site: &str) -> Result<Tensor> {
        // random.cpp:95-140,174-203. Five host constants and an evaluated scalar range;
        // U(RandomBits), U(uint32->f32), and six floating expression outputs.
        self.immediate(&[2], Dtype::Uint32, site)?;
        let low = self.scalar(Dtype::Float32, site)?;
        let high = self.scalar(Dtype::Float32, site)?;
        let upper = self.scalar(Dtype::Float32, site)?;
        let maxval = self.scalar(Dtype::Float32, site)?;
        let scale = self.scalar(Dtype::Float32, site)?;
        let range = self.binary(&high, &low, site)?;
        let bits = self.allocation(
            shape,
            Dtype::Uint32,
            "metal/primitives.cpp:169 RandomBits output",
            site,
        )?;
        let bits = self.cast(&bits, Dtype::Float32, site)?;
        let uniform = self.binary(&bits, &maxval, site)?;
        let uniform = self.binary(&uniform, &upper, site)?;
        let uniform = self.binary(&range, &uniform, site)?;
        let uniform = self.binary(&uniform, &low, site)?;
        let samples = self.unary(&uniform, site)?;
        self.binary(&scale, &samples, site)
    }

    pub(super) fn metal_affine_quantize(
        &mut self,
        x: &Tensor,
        bits: u64,
        site: &str,
    ) -> Result<[Tensor; 3]> {
        if [x].iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&[x], |g, v| g.metal_affine_quantize(&v[0], bits, site));
        }
        floating(x)?;
        let mut packed = x.shape.clone();
        let mut groups = packed.clone();
        let last = packed
            .last_mut()
            .ok_or_else(|| unsupported("Quantize positive rank"))?;
        if !matches!(bits, 4 | 8) || *last == 0 || *last % 64 != 0 {
            return Err(unsupported(
                "Quantize affine group64, bits4/8 aligned descriptor",
            ));
        }
        *groups
            .last_mut()
            .ok_or_else(|| unsupported("Quantize group rank"))? = *last / 64;
        *last = mul(*last, bits)? / 32;
        self.metal_row(x, site)?;
        Ok([
            self.allocation(
                &packed,
                Dtype::Uint32,
                "metal/quantized.cpp:1806 packed output",
                site,
            )?,
            self.allocation(
                &groups,
                x.dtype,
                "metal/quantized.cpp:1824 scales output",
                site,
            )?,
            self.allocation(
                &groups,
                x.dtype,
                "metal/quantized.cpp:1827 biases output",
                site,
            )?,
        ])
    }
    pub(super) fn metal_affine_dequantize(
        &mut self,
        packed: &Tensor,
        scales: &Tensor,
        offsets: &Tensor,
        bits: u64,
        site: &str,
    ) -> Result<Tensor> {
        if [packed, scales, offsets]
            .iter()
            .any(|x| !x.alternatives.is_empty())
        {
            return self.metal_over_inputs(&[packed, scales, offsets], |g, v| {
                g.metal_affine_dequantize(&v[0], &v[1], &v[2], bits, site)
            });
        }
        if !matches!(bits, 4 | 8) {
            return Err(unsupported("Dequantize affine bits4/8"));
        }
        let mut shape = packed.shape.clone();
        let last = shape
            .last_mut()
            .ok_or_else(|| unsupported("Dequantize positive rank"))?;
        *last = mul(*last, 32)? / bits;
        floating(scales)?;
        if packed.dtype != Dtype::Uint32
            || scales.dtype != offsets.dtype
            || scales.shape != offsets.shape
        {
            return Err(unsupported(
                "Dequantize uint32 packed input and same-dtype/same-shape floating scales/biases",
            ));
        }
        let mut group_shape = shape.clone();
        let group_last = group_shape
            .last_mut()
            .ok_or_else(|| unsupported("Dequantize group rank"))?;
        if *group_last % 64 != 0 {
            return Err(unsupported("Dequantize group64 aligned output"));
        }
        *group_last /= 64;
        if group_shape != scales.shape {
            return Err(unsupported(
                "Dequantize scale/bias shape must match group64 output groups",
            ));
        }
        let dtype = scales.dtype;
        for x in [packed, scales, offsets] {
            self.metal_row(x, site)?;
        }
        self.allocation(
            &shape,
            dtype,
            "metal/quantized.cpp:1806-1820 affine dequantize output",
            site,
        )
    }
    pub(super) fn metal_gather_rows(
        &mut self,
        x: &Tensor,
        rows: u64,
        site: &str,
    ) -> Result<Tensor> {
        let mut shape = x.shape.clone();
        *shape
            .first_mut()
            .ok_or_else(|| unsupported("Gather source positive rank"))? = rows;
        // Gather (not Scatter): no mask/offset intermediate, even on JIT fallback branches.
        self.allocation(
            &shape,
            x.dtype,
            "metal/indexing.cpp:57-68 Gather output",
            site,
        )
    }
    pub(super) fn metal_conv2d(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: u64,
        padding: u64,
        site: &str,
    ) -> Result<Tensor> {
        let mut inputs = vec![x, weight];
        if let Some(extra) = bias {
            inputs.push(extra);
        }
        if inputs.iter().any(|x| !x.alternatives.is_empty()) {
            return self.metal_over_inputs(&inputs, |g, v| {
                g.metal_conv2d(
                    &v[0],
                    &v[1],
                    if bias.is_some() { Some(&v[2]) } else { None },
                    stride,
                    padding,
                    site,
                )
            });
        }
        nonempty(x)?;
        nonempty(weight)?;
        if x.shape.len() != 4
            || weight.shape.len() != 4
            || x.shape[3] != weight.shape[3]
            || stride == 0
        {
            return Err(unsupported(
                "Z convolution NHWC/OHWI, groups1, dilation1, flip=false geometry",
            ));
        }
        let dtype = promote(x.dtype, weight.dtype)?;
        let x = self.cast(x, dtype, site)?;
        let w = self.cast(weight, dtype, site)?;
        let ph = add(x.shape[1], mul(2, padding)?)?;
        let pw = add(x.shape[2], mul(2, padding)?)?;
        if ph < w.shape[1] || pw < w.shape[2] {
            return Err(unsupported("convolution kernel exceeds padded input"));
        }
        let oh = add((ph - w.shape[1]) / stride, 1)?;
        let ow = add((pw - w.shape[2]) / stride, 1)?;
        let shape = [x.shape[0], oh, ow, w.shape[0]];
        // nn.rs exact output-bias variant rejects Winograd/explicit branches before evaluation;
        // supported implicit branch has the same output, with optional contiguous bias copy.
        let winograd = stride == 1
            && w.shape[1] == 3
            && w.shape[2] == 3
            && x.shape[3] % 32 == 0
            && w.shape[0] % 32 == 0
            && mul(mul(x.shape[0], x.shape[1])?, x.shape[2])? >= 4096
            && add(x.shape[3], w.shape[0])? >= 256;
        let implicit =
            (x.shape[3] <= 4 || x.shape[3] % 16 == 0) && (w.shape[0] <= 16 || w.shape[0] % 16 == 0);
        let general =
            (x.shape[3] % 16 == 0 && w.shape[0] % 16 == 0) || mul(mul(x.shape[0], oh)?, ow)? >= 256;
        let core = |g: &mut Graph| -> Result<Tensor> {
            let out = g.allocation(&shape, dtype, "metal/conv.cpp:1254 conv_2D output", site)?;
            g.metal_row(&x, site)?;
            g.metal_row(&w, site)?;
            if winograd {
                g.metal_winograd(&x, &w, &out, padding, site)?;
            } else if !implicit && !general {
                let m = mul(mul(x.shape[0], oh)?, ow)?;
                let k = mul(mul(x.shape[3], w.shape[1])?, w.shape[2])?;
                g.allocation(
                    &[m, k],
                    dtype,
                    "metal/conv.cpp:34-49 explicit unfolded input",
                    site,
                )?;
                g.metal_splitk(m, w.shape[0], k, site)?;
            }
            Ok(out)
        };
        let eager = |g: &mut Graph| -> Result<Tensor> {
            let out = core(g)?;
            match bias {
                Some(b) => g.binary(&out, b, site),
                None => Ok(out),
            }
        };
        if let Some(b) = bias.filter(|b| {
            b.dtype == dtype && b.shape == [w.shape[0]] && !winograd && (implicit || general)
        }) {
            self.metal_max(eager, |g| {
                g.metal_row(b, site)?;
                core(g)
            })
        } else {
            eager(self)
        }
    }
    fn metal_winograd(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        out: &Tensor,
        padding: u64,
        site: &str,
    ) -> Result<()> {
        // metal/conv.cpp:765-949 F(6,3): padding backing, zero host scalar, filter/input/output
        // transforms. Transform GEMM has batch_size_out=64, excluding Split-K workspaces.
        let padded = |d| -> Result<u64> {
            let n = add(d, mul(2, padding)?)?;
            let inner = n
                .checked_sub(2)
                .ok_or_else(|| unsupported("Winograd padding extent"))?;
            add(mul(6, ceil(inner, 6)?)?, 2)
        };
        self.scalar(x.dtype, site)?;
        self.allocation(
            &[
                x.shape[0],
                padded(x.shape[1])?,
                padded(x.shape[2])?,
                x.shape[3],
            ],
            x.dtype,
            "metal/conv.cpp:773-788 Winograd in_padded fill_gpu",
            site,
        )?;
        let tiles = mul(
            mul(x.shape[0], ceil(out.shape[1], 6)?)?,
            ceil(out.shape[2], 6)?,
        )?;
        self.allocation(
            &[64, x.shape[3], w.shape[0]],
            w.dtype,
            "metal/conv.cpp:844-846 Winograd filt_wg",
            site,
        )?;
        self.allocation(
            &[64, tiles, x.shape[3]],
            x.dtype,
            "metal/conv.cpp:876-878 Winograd inp_wg",
            site,
        )?;
        self.allocation(
            &[64, tiles, w.shape[0]],
            x.dtype,
            "metal/conv.cpp:908-910 Winograd out_wg",
            site,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph() -> Graph {
        Graph::with_lowering(16_384, Lowering::Metal).unwrap()
    }
    #[test]
    fn split_k_is_one_f32_backing_not_output_dtype_or_sum_of_algorithms() {
        let mut g = graph();
        g.metal_splitk(32, 32, 4096, "host Split-K").unwrap();
        let mut expected = graph();
        // SIMD gives 32 partitions; NAX gives 2. Choose MAX U, not 34 partitions.
        expected
            .allocation(&[32, 32, 32], Dtype::Float32, "expected", "host")
            .unwrap();
        assert_eq!(g.upper(), expected.upper());
    }
    #[test]
    fn sdpa_two_pass_charges_three_separately_rounded_backings() {
        let mut g = graph();
        let out = Tensor::row(&[1, 4, 2, 64], Dtype::Float16).unwrap();
        g.metal_sdpa_intermediates(&out, 32, "host SDPA").unwrap();
        let mut expected = graph();
        for (shape, dtype) in [
            (&[1, 4, 2, 32, 64][..], Dtype::Float16),
            (&[1, 4, 2, 32][..], Dtype::Float32),
            (&[1, 4, 2, 32][..], Dtype::Float32),
        ] {
            expected
                .allocation(shape, dtype, "expected", "host")
                .unwrap();
        }
        assert_eq!(g.upper(), expected.upper());
    }
    #[test]
    fn convolution_epilogue_alternatives_take_max_not_sum() {
        let x = Tensor::row(&[1, 8, 8, 16], Dtype::Float16).unwrap();
        let w = Tensor::row(&[16, 3, 3, 16], Dtype::Float16).unwrap();
        let b = Tensor::row(&[16], Dtype::Float16).unwrap();
        let mut g = graph();
        g.conv2d(&x, &w, Some(&b), 1, 1, "host conv").unwrap();
        let mut expected = graph();
        let y = expected
            .allocation(&[1, 8, 8, 16], Dtype::Float16, "expected", "host")
            .unwrap();
        expected.binary(&y, &b, "host").unwrap();
        assert_eq!(g.upper(), expected.upper());
    }
    #[test]
    fn quantized_split_k_includes_the_nested_reduction_workspace() {
        let mut g = graph();
        let x = Tensor::row(&[16, 32768], Dtype::Float16).unwrap();
        let w = Tensor::row(&[32, 4096], Dtype::Uint32).unwrap();
        let scales = Tensor::row(&[32, 512], Dtype::Float16).unwrap();
        g.qmm(&x, &w, &scales, &scales, None, "host qmm").unwrap();
        let mut expected = graph();
        expected
            .allocation(&[16, 32], Dtype::Float16, "expected", "host")
            .unwrap();
        expected
            .allocation(&[512, 16 * 32], Dtype::Float16, "expected", "host")
            .unwrap();
        expected
            .allocation(&[32, 16 * 32], Dtype::Float16, "expected", "host")
            .unwrap();
        assert_eq!(g.upper(), expected.upper());
    }
    #[test]
    fn differing_silu_layouts_survive_for_downstream_copy_lowering() {
        let mut g = graph();
        let scalar = Tensor::row(&[], Dtype::Float16).unwrap();
        let x = scalar.broadcast(&[2, 3]).unwrap();
        let out = g.silu(&x, "host alternative SiLU").unwrap();
        assert_eq!(out.layouts().len(), 2);
        let transposed = out.transpose(&[1, 0]).unwrap();
        let before = g.upper();
        let reshaped = g
            .reshape(&transposed, &[6], "host alternative reshape")
            .unwrap();
        // Exact activation's transposed full backing needs a copy; the scalar-broadcast
        // eager branch remains a view. Neither descriptor is erased by taking the maximum.
        assert!(g.upper() > before);
        assert_eq!(reshaped.layouts().len(), 2);
    }
    #[test]
    fn both_stream_families_pass_scope_validation_without_querying_a_device() {
        for cpu_stream in [false, true] {
            assert!(certified_execution_refusal(CertifiedExecution {
                load_order: HeavyLoadOrder::VaeBeforeQuantize,
                ios_quantum_scope: true,
                cpu_stream,
            })
            .is_ok());
        }
        assert!(certified_execution_refusal(CertifiedExecution {
            load_order: HeavyLoadOrder::VaeBeforeQuantize,
            ios_quantum_scope: false,
            cpu_stream: false,
        })
        .is_err());
    }
    #[test]
    fn reduction_all_passes_and_full_sdpa_layout_are_host_only() {
        let mut g = graph();
        let x = Tensor::row(&[8192], Dtype::Float16).unwrap();
        g.reduce(&x, &[0], "host reduce").unwrap();
        let mut expected = graph();
        expected.counter.charge("expected", "host", 4).unwrap();
        expected
            .allocation(&[128], Dtype::Float16, "expected", "host")
            .unwrap();
        assert_eq!(g.upper(), expected.upper());
        let q = Tensor::row(&[1, 4, 16, 64], Dtype::Float16).unwrap();
        let y = g.sdpa(&q, &q, &q, None, "host full SDPA").unwrap();
        assert_eq!(y.strides, [4096, 64, 256, 1]);
    }
}
