//! CPU-stream allocation rules for the pinned native096 source, not Metal kernel selectors.
//! Every charge is a requested backing passed through the existing allocator-cache law.
//! Data is NEVER retired here. Host containers and BLAS-internal storage are not this counter.
use super::super::pid_sources::ActiveCounterRequests;
use super::*;
use mlx_gen::adapters::facts::dtype_width;

fn fail(message: &str) -> Error {
    Error::Msg(format!("Z native CPU graph: {message}"))
}
pub(super) fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(|| fail("addition overflow"))
}
pub(super) fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(|| fail("product overflow"))
}
pub(super) fn count(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1, |n, &d| mul(n, d))
}

/// An actual source descriptor's layout. `data_size` is NOT replaced by numel: broadcasts,
/// transposes, splits and reshape views preserve/transform it using the native source rules.
#[derive(Clone, Debug)]
pub(super) struct Tensor {
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub strides: Vec<u64>,
    pub data_size: u64,
    pub contiguous: bool,
    // Metal runtime alternatives can have equal shape/dtype but different layouts. Keep every
    // descriptor for downstream lowering; CPU descriptors always leave this empty.
    pub alternatives: Vec<Tensor>,
}
impl Tensor {
    pub fn row(shape: &[u64], dtype: Dtype) -> Result<Self> {
        let mut strides = vec![0; shape.len()];
        let mut n = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = n;
            n = mul(n, shape[i])?;
        }
        Ok(Self {
            shape: shape.into(),
            dtype,
            strides,
            data_size: n,
            contiguous: true,
            alternatives: Vec::new(),
        })
    }
    pub fn layouts(&self) -> Vec<Self> {
        let mut first = self.clone();
        first.alternatives.clear();
        let mut layouts = vec![first];
        layouts.extend(self.alternatives.iter().cloned());
        layouts
    }
    pub fn same_layout(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype == other.dtype
            && self.strides == other.strides
            && self.data_size == other.data_size
            && self.contiguous == other.contiguous
    }
    pub fn merge_layouts(&mut self, other: Self) -> Result<()> {
        if self.shape != other.shape || self.dtype != other.dtype {
            return Err(Error::Unsupported(
                "Metal alternative outputs require the same frontend shape and dtype".into(),
            ));
        }
        for mut layout in other.layouts() {
            layout.alternatives.clear();
            if !self.same_layout(&layout)
                && !self.alternatives.iter().any(|x| x.same_layout(&layout))
            {
                self.alternatives.push(layout);
            }
        }
        Ok(())
    }
    pub fn elements(&self) -> Result<u64> {
        count(&self.shape)
    }
    pub fn nbytes(&self) -> Result<u64> {
        mul(self.elements()?, dtype_width(self.dtype))
    }
    pub fn row_contiguous(&self) -> bool {
        let mut s = 1u64;
        for (&d, &st) in self.shape.iter().zip(&self.strides).rev() {
            if d != 1 && st != s {
                return false;
            }
            let Some(next) = s.checked_mul(d) else {
                return false;
            };
            s = next;
        }
        true
    }
    fn col_contiguous(&self) -> bool {
        let mut s = 1u64;
        for (&d, &st) in self.shape.iter().zip(&self.strides) {
            if d != 1 && st != s {
                return false;
            }
            let Some(next) = s.checked_mul(d) else {
                return false;
            };
            s = next;
        }
        true
    }
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        if !self.alternatives.is_empty() {
            let mut layouts = self.layouts().into_iter();
            let mut out = layouts
                .next()
                .ok_or_else(|| fail("empty layout alternatives"))?
                .transpose(axes)?;
            for layout in layouts {
                out.merge_layouts(layout.transpose(axes)?)?;
            }
            return Ok(out);
        }
        if axes.len() != self.shape.len()
            || axes.iter().copied().collect::<BTreeSet<_>>() != (0..axes.len()).collect()
        {
            return Err(fail("invalid transpose permutation"));
        }
        Ok(Self {
            shape: axes.iter().map(|&i| self.shape[i]).collect(),
            strides: axes.iter().map(|&i| self.strides[i]).collect(),
            ..self.clone()
        })
    }
    pub fn broadcast(&self, shape: &[u64]) -> Result<Self> {
        if !self.alternatives.is_empty() {
            let mut layouts = self.layouts().into_iter();
            let mut out = layouts
                .next()
                .ok_or_else(|| fail("empty layout alternatives"))?
                .broadcast(shape)?;
            for layout in layouts {
                out.merge_layouts(layout.broadcast(shape)?)?;
            }
            return Ok(out);
        }
        if shape.len() < self.shape.len() {
            return Err(fail("broadcast rank"));
        }
        let skip = shape.len() - self.shape.len();
        let mut strides = vec![0; shape.len()];
        for i in 0..self.shape.len() {
            if self.shape[i] != shape[i + skip] && self.shape[i] != 1 {
                return Err(fail("broadcast shape"));
            }
            strides[i + skip] = if self.shape[i] == 1 && self.shape != shape {
                0
            } else {
                self.strides[i]
            };
        }
        Ok(Self {
            shape: shape.into(),
            strides,
            ..self.clone()
        })
    }
    /// Native common Split: no allocation; data_size excludes broadcasted dimensions.
    pub fn split(&self, axis: usize, size: u64) -> Result<Self> {
        if !self.alternatives.is_empty() {
            let mut layouts = self.layouts().into_iter();
            let mut out = layouts
                .next()
                .ok_or_else(|| fail("empty layout alternatives"))?
                .split(axis, size)?;
            for layout in layouts {
                out.merge_layouts(layout.split(axis, size)?)?;
            }
            return Ok(out);
        }
        if axis >= self.shape.len() || size > self.shape[axis] {
            return Err(fail("slice extent"));
        }
        let mut out = self.clone();
        out.shape[axis] = size;
        out.data_size = out
            .shape
            .iter()
            .zip(&out.strides)
            .try_fold(1, |n, (&d, &s)| if s == 0 { Ok(n) } else { mul(n, d) })?;
        out.contiguous = if out.data_size == 1 {
            true
        } else if out.data_size == self.data_size {
            self.contiguous
        } else {
            self.contiguous && (out.row_contiguous() || out.col_contiguous())
        };
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lowering {
    Cpu,
    Metal,
}

#[derive(Clone, Debug)]
pub(super) struct Graph {
    pub counter: ActiveCounterRequests,
    pub lowering: Lowering,
}
impl Graph {
    pub fn new(page: u64) -> Result<Self> {
        Ok(Self {
            counter: ActiveCounterRequests::new(page)?,
            lowering: Lowering::Cpu,
        })
    }
    pub fn with_lowering(page: u64, lowering: Lowering) -> Result<Self> {
        let mut graph = Self::new(page)?;
        graph.lowering = lowering;
        Ok(graph)
    }
    pub fn is_metal(&self) -> bool {
        self.lowering == Lowering::Metal
    }
    pub fn upper(&self) -> u64 {
        self.counter.upper()
    }
    pub fn allocation(
        &mut self,
        shape: &[u64],
        dtype: Dtype,
        rule: &'static str,
        site: &str,
    ) -> Result<Tensor> {
        let out = Tensor::row(shape, dtype)?;
        self.counter.charge(rule, site, out.nbytes()?)?;
        Ok(out)
    }
    pub fn immediate(&mut self, shape: &[u64], dtype: Dtype, site: &str) -> Result<Tensor> {
        self.allocation(
            shape,
            dtype,
            "array.cpp host constructor/allocator::malloc",
            site,
        )
    }
    pub fn scalar(&mut self, dtype: Dtype, site: &str) -> Result<Tensor> {
        self.immediate(&[], dtype, site)
    }
    pub fn rust_scalar(&mut self, dtype: Dtype, site: &str) -> Result<Tensor> {
        self.immediate(&[1], dtype, site)
    }
    pub fn load(&mut self, bytes: u64, site: &str) -> Result<()> {
        // IO wrappers default to CPU independently of the compute default. Under Metal this
        // also dominates a task-local GPU override: native Load::eval_gpu throws before malloc.
        self.counter
            .charge("backend/common/load.cpp: Load::eval_cpu", site, bytes)
    }
    /// Native common prepare_reshape; the selected backend owns a required copy.
    pub fn reshape(&mut self, x: &Tensor, shape: &[u64], site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_reshape(x, shape, site);
        }
        if count(shape)? != x.elements()? {
            return Err(fail("reshape element count"));
        }
        if x.row_contiguous() || x.elements()? == 0 {
            let mut out = Tensor::row(shape, x.dtype)?;
            out.data_size = x.data_size;
            return Ok(out);
        }
        if x.shape.is_empty() {
            return Ok(Tensor {
                shape: shape.into(),
                strides: vec![0; shape.len()],
                ..x.clone()
            });
        }
        // collapse_contiguous_dims: singleton axes can be ignored; adjacent axes collapse only
        // when their actual strides prove the same contiguous chunk (including zero strides).
        let mut chunks: Vec<(u64, u64)> = Vec::new();
        for (&d, &st) in x.shape.iter().zip(&x.strides).rev() {
            if d == 1 {
                continue;
            }
            if let Some((n, s)) = chunks.last_mut() {
                if st == mul(*n, *s)? {
                    *n = mul(*n, d)?;
                    continue;
                }
            }
            chunks.push((d, st));
        }
        chunks.reverse();
        let mut j = 0;
        let mut strides = Vec::new();
        let mut copy = false;
        for &d in shape {
            if d != 0 && j < chunks.len() && chunks[j].0.is_multiple_of(d) {
                chunks[j].0 /= d;
                strides.push(mul(chunks[j].0, chunks[j].1)?);
                if chunks[j].0 == 1 {
                    j += 1;
                }
            } else if d == 1 {
                strides.push(strides.last().copied().unwrap_or(0));
            } else {
                copy = true;
                break;
            }
        }
        if copy {
            return self.allocation(shape, x.dtype, "cpu/copy.cpp reshape General copy", site);
        }
        Ok(Tensor {
            shape: shape.into(),
            strides,
            ..x.clone()
        })
    }
    pub fn cast(&mut self, x: &Tensor, dtype: Dtype, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_cast(x, dtype, site);
        }
        if x.dtype == dtype {
            return Ok(x.clone());
        }
        // AsType's copy uses Vector only for contiguous full data; otherwise General.
        if x.contiguous {
            self.counter.charge(
                "cpu/primitives.cpp AsType / common/copy.h Vector",
                site,
                mul(x.data_size, dtype_width(dtype))?,
            )?;
            Ok(Tensor { dtype, ..x.clone() })
        } else {
            self.allocation(
                &x.shape,
                dtype,
                "cpu/primitives.cpp AsType / common/copy.h General",
                site,
            )
        }
    }
    pub fn unary(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_unary(x, site);
        }
        // common/unary.h set_unary_output_data. Donation is not a release or allocation credit.
        if x.contiguous {
            self.counter.charge(
                "common/unary.h non-donated contiguous output",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x.clone())
        } else {
            self.allocation(&x.shape, x.dtype, "common/unary.h General output", site)
        }
    }
    pub fn binary(&mut self, a: &Tensor, b: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_binary(a, b, site);
        }
        let shape = broadcast_shape(&a.shape, &b.shape)?;
        let dtype = promote(a.dtype, b.dtype)?;
        let a = self.cast(a, dtype, site)?.broadcast(&shape)?;
        let b = self.cast(b, dtype, site)?.broadcast(&shape)?;
        // Source-exact get_binary_op_type, common/binary.h:21-95. Do not assume nbytes ==
        // data_size*itemsize for broadcasts. All possible donation outcomes share this capacity.
        let model = if a.data_size == 1 && b.data_size == 1 {
            Some(&a)
        } else if a.data_size == 1 && b.contiguous {
            Some(&b)
        } else if (b.data_size == 1 && a.contiguous)
            || (a.row_contiguous() && b.row_contiguous())
            || (a.col_contiguous() && b.col_contiguous())
        {
            // VectorScalar and VectorVector: binary.h:60-79 both malloc a.data_size()*itemsize.
            Some(&a)
        } else {
            None
        };
        if let Some(out) = model {
            self.counter.charge(
                "common/binary.h non-donated output data_size",
                site,
                mul(out.data_size, dtype_width(dtype))?,
            )?;
            Ok(out.clone())
        } else {
            self.allocation(&shape, dtype, "common/binary.h General output nbytes", site)
        }
    }
    pub fn concat(&mut self, xs: &[Tensor], axis: usize, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_concat(xs, axis, site);
        }
        let first = xs.first().ok_or_else(|| fail("empty concatenate"))?;
        if xs.len() == 1 {
            return Ok(first.clone());
        }
        if axis >= first.shape.len() {
            return Err(fail("concat axis"));
        }
        let mut shape = first.shape.clone();
        shape[axis] = 0;
        let mut dtype = first.dtype;
        for x in xs {
            if x.shape.len() != shape.len()
                || x.shape
                    .iter()
                    .enumerate()
                    .any(|(i, d)| i != axis && *d != shape[i])
            {
                return Err(fail("concat shape"));
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
            "cpu/primitives.cpp Concatenate::eval_cpu",
            site,
        )
    }
    pub fn reduce(&mut self, x: &Tensor, axes: &[usize], site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_reduce(x, axes, site);
        }
        if axes.iter().all(|&axis| x.shape.get(axis) == Some(&1)) {
            return Ok(x.clone());
        }
        let mut shape = x.shape.clone();
        for &axis in axes {
            *shape.get_mut(axis).ok_or_else(|| fail("reduce axis"))? = 1;
        }
        self.allocation(&shape, x.dtype, "cpu/reduce.cpp Reduce::eval_cpu", site)
    }
    pub fn mean(&mut self, x: &Tensor, axes: &[usize], site: &str) -> Result<Tensor> {
        // ops.cpp mean_impl = sum + NumberOfElements(inverted) + multiply.
        let sum = self.reduce(x, axes, site)?;
        let inv = self.allocation(&[], x.dtype, "common/common.cpp NumberOfElements", site)?;
        self.binary(&sum, &inv, site)
    }
    pub fn softmax(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_softmax(x, site);
        }
        if x.contiguous && x.strides.last() == Some(&1) {
            self.counter.charge(
                "cpu/softmax.cpp non-donated data_size output",
                site,
                mul(x.data_size, dtype_width(x.dtype))?,
            )?;
            Ok(x.clone())
        } else {
            // The contiguous copy IS the output backing; there is no second output allocation.
            self.allocation(
                &x.shape,
                x.dtype,
                "cpu/softmax.cpp contiguous_copy_cpu shared output",
                site,
            )
        }
    }
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor, site: &str) -> Result<Tensor> {
        self.matrix_product(a, b, None, site)
    }
    pub fn matrix_product(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        bias: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_matrix_product(a, b, bias, site);
        }
        if a.shape.len() < 2
            || b.shape.len() < 2
            || a.shape[a.shape.len() - 1] != b.shape[b.shape.len() - 2]
        {
            return Err(fail("matmul geometry"));
        }
        let mut dtype = promote(a.dtype, b.dtype)?;
        if let Some(bias) = bias {
            dtype = promote(dtype, bias.dtype)?;
            self.cast(bias, dtype, site)?;
        }
        let original_a_shape = a.shape.clone();
        let mut a = self.cast(a, dtype, site)?;
        let b = self.cast(b, dtype, site)?;
        let flatten = a.shape.len() > 2 && b.shape.len() == 2;
        if flatten {
            a = self.reshape(
                &a,
                &[
                    count(&a.shape[..a.shape.len() - 1])?,
                    a.shape[a.shape.len() - 1],
                ],
                site,
            )?;
        }
        let mut shape =
            broadcast_shape(&a.shape[..a.shape.len() - 2], &b.shape[..b.shape.len() - 2])?;
        shape.push(a.shape[a.shape.len() - 2]);
        shape.push(b.shape[b.shape.len() - 1]);
        let out = self.allocation(
            &shape,
            dtype,
            if bias.is_some() {
                "cpu/matmul.cpp AddMM copy output"
            } else {
                "cpu/matmul.cpp Matmul output"
            },
            site,
        )?;
        for x in [&a, &b] {
            let n = x.shape.len();
            let (sx, sy) = (x.strides[n - 2], x.strides[n - 1]);
            if !((sx == x.shape[n - 1] && sy == 1) || (sx == 1 && sy == x.shape[n - 2])) {
                self.allocation(
                    &x.shape,
                    x.dtype,
                    "cpu/matmul.cpp check_transpose General copy",
                    site,
                )?;
            }
        }
        if flatten {
            let mut shape = original_a_shape;
            *shape.last_mut().unwrap() = b.shape[b.shape.len() - 1];
            self.reshape(&out, &shape, site)
        } else {
            Ok(out)
        }
    }
    pub fn qmm(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        scales: &Tensor,
        offsets: &Tensor,
        bias: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_qmm(x, weight, scales, offsets, bias, site);
        }
        if weight.shape.len() != 2 || x.shape.is_empty() {
            return Err(fail("Z qmm rank"));
        }
        let dtype = promote(x.dtype, promote(scales.dtype, offsets.dtype)?)?;
        let x = self.cast(x, dtype, site)?;
        let scales = self.cast(scales, dtype, site)?;
        let offsets = self.cast(offsets, dtype, site)?;
        for input in [&x, weight, &scales, &offsets] {
            if !input.row_contiguous() {
                self.allocation(
                    &input.shape,
                    input.dtype,
                    "cpu/quantized.cpp qmm ensure_row_contiguous",
                    site,
                )?;
            }
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = weight.shape[0];
        let out = self.allocation(
            &shape,
            dtype,
            "cpu/quantized.cpp QuantizedMatmul output",
            site,
        )?;
        if let Some(bias) = bias {
            self.binary(&out, bias, site)
        } else {
            Ok(out)
        }
    }
    pub fn rms(&mut self, x: &Tensor, weight: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_rms(x, weight, site);
        }
        // fast.cpp CPU fallback, not a guessed fused normalization kernel.
        let dtype = promote(x.dtype, weight.dtype)?;
        let x32 = self.cast(x, Dtype::Float32, site)?;
        let sq = self.unary(&x32, site)?;
        let mean = self.mean(&sq, &[sq.shape.len() - 1], site)?;
        let eps = self.scalar(Dtype::Float32, site)?;
        let variance = self.binary(&mean, &eps, site)?;
        let inv = self.unary(&variance, site)?;
        let norm = self.binary(&x32, &inv, site)?;
        let norm = self.cast(&norm, dtype, site)?;
        let weight = self.cast(weight, dtype, site)?;
        self.binary(&norm, &weight, site)
    }
    pub fn layer_norm(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_layer_norm(x, site);
        }
        // fast.cpp: affine-free CPU LN still constructs passed weight/bias scalar arrays.
        self.scalar(x.dtype, site)?;
        self.scalar(x.dtype, site)?;
        let a = self.cast(x, Dtype::Float32, site)?;
        let mu = self.mean(&a, &[a.shape.len() - 1], site)?;
        let xc = self.binary(&a, &mu, site)?;
        let square = self.unary(&xc, site)?;
        let v = self.mean(&square, &[square.shape.len() - 1], site)?;
        let eps = self.scalar(Dtype::Float32, site)?;
        let v = self.binary(&v, &eps, site)?;
        let inv = self.unary(&v, site)?;
        let out = self.binary(&xc, &inv, site)?;
        self.cast(&out, x.dtype, site)
    }
    pub fn silu(&mut self, x: &Tensor, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_silu(x, site);
        }
        // nn::silu eager x*sigmoid(x); compiled CPU allocation is bounded by these source outputs.
        let sigmoid = self.unary(x, site)?;
        self.binary(x, &sigmoid, site)
    }
    pub fn sdpa(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        site: &str,
    ) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_sdpa(q, k, v, mask, site);
        }
        if q.shape.len() != 4
            || k.shape.len() != 4
            || v.shape.len() != 4
            || q.shape[1] != k.shape[1]
            || k.shape[1] != v.shape[1]
        {
            return Err(fail("SDPA requires provider-expanded heads"));
        }
        // fast.cpp fallback: q*scale, qk^T, optional additive mask, softmax, probabilities*v.
        let dtype = promote(promote(q.dtype, k.dtype)?, v.dtype)?;
        let q = self.cast(q, dtype, site)?;
        let k = self.cast(k, dtype, site)?;
        let v = self.cast(v, dtype, site)?;
        let scale = self.scalar(dtype, site)?;
        let q = self.binary(&q, &scale, site)?;
        let kt = k.transpose(&[0, 1, 3, 2])?;
        let mut scores = self.matmul(&q, &kt, site)?;
        if let Some(mask) = mask {
            scores = self.binary(&scores, mask, site)?;
        }
        let p = self.softmax(&scores, site)?;
        self.matmul(&p, &v, site)
    }
    pub fn normal_f32(&mut self, shape: &[u64], site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_normal_f32(shape, site);
        }
        // random.cpp explicit key + uniform(-1,1) -> erfinv -> sqrt(2) multiply.
        self.immediate(&[2], Dtype::Uint32, site)?;
        for _ in 0..6 {
            self.scalar(Dtype::Float32, site)?;
        }
        self.allocation(
            shape,
            Dtype::Uint32,
            "cpu/primitives.cpp RandomBits output",
            site,
        )?;
        for _ in 0..7 {
            self.allocation(
                shape,
                Dtype::Float32,
                "random.cpp normal evaluated composed output",
                site,
            )?;
        }
        Tensor::row(shape, Dtype::Float32)
    }
    pub fn affine_quantize(&mut self, x: &Tensor, bits: u64, site: &str) -> Result<[Tensor; 3]> {
        if self.is_metal() {
            return self.metal_affine_quantize(x, bits, site);
        }
        let mut packed = x.shape.clone();
        let mut groups = packed.clone();
        let last = packed.last_mut().ok_or_else(|| fail("quantize rank"))?;
        if !matches!(bits, 4 | 8) || *last % 64 != 0 {
            return Err(fail("Z quantize alignment/bits"));
        }
        *groups.last_mut().unwrap() = *last / 64;
        *last = mul(*last, bits)? / 32;
        if !x.row_contiguous() {
            self.allocation(
                &x.shape,
                x.dtype,
                "cpu/quantized.cpp quantize ensure_row_contiguous",
                site,
            )?;
        }
        Ok([
            self.allocation(
                &packed,
                Dtype::Uint32,
                "cpu/quantized.cpp Quantize packed output",
                site,
            )?,
            self.allocation(
                &groups,
                x.dtype,
                "cpu/quantized.cpp Quantize scales output",
                site,
            )?,
            self.allocation(
                &groups,
                x.dtype,
                "cpu/quantized.cpp Quantize biases output",
                site,
            )?,
        ])
    }
    pub fn affine_dequantize(
        &mut self,
        packed: &Tensor,
        scales: &Tensor,
        offsets: &Tensor,
        bits: u64,
        site: &str,
    ) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_affine_dequantize(packed, scales, offsets, bits, site);
        }
        if !matches!(bits, 4 | 8) || packed.shape.is_empty() {
            return Err(fail("Z dequantize bits/rank"));
        }
        let mut shape = packed.shape.clone();
        *shape.last_mut().unwrap() = mul(*shape.last().unwrap(), 32)? / bits;
        let dtype = promote(scales.dtype, offsets.dtype)?;
        // ops.cpp affine_dequantize CPU power-of-two fallback. Shift outputs preserve compact
        // packed data_size; concatenate materializes codes once, then cast/multiply/add.
        for _ in 0..32 / bits {
            self.scalar(Dtype::Uint32, site)?;
            self.scalar(Dtype::Uint32, site)?;
            self.unary(packed, site)?;
            self.unary(packed, site)?;
        }
        self.allocation(
            &shape,
            Dtype::Uint32,
            "ops.cpp affine dequantize concatenate fields",
            site,
        )?;
        self.allocation(
            &shape,
            dtype,
            "ops.cpp affine dequantize integer-to-float cast",
            site,
        )?;
        self.allocation(
            &shape,
            dtype,
            "ops.cpp affine dequantize scale multiply",
            site,
        )?;
        self.allocation(&shape, dtype, "ops.cpp affine dequantize offsets add", site)
    }
    pub fn gather_rows(&mut self, x: &Tensor, rows: u64, site: &str) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_gather_rows(x, rows, site);
        }
        if x.shape.is_empty() {
            return Err(fail("gather source rank"));
        }
        let mut shape = x.shape.clone();
        shape[0] = rows;
        self.allocation(&shape, x.dtype, "cpu/indexing.cpp Gather output", site)
    }
    pub fn conv2d(
        &mut self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: u64,
        padding: u64,
        site: &str,
    ) -> Result<Tensor> {
        if self.is_metal() {
            return self.metal_conv2d(x, weight, bias, stride, padding, site);
        }
        if x.shape.len() != 4
            || weight.shape.len() != 4
            || x.shape[3] != weight.shape[3]
            || stride == 0
        {
            return Err(fail("Z VAE convolution geometry"));
        }
        // Provider fixes groups=1, dilation=1, input_dilation=1, flip=false. CPU routes to
        // explicit_gemm_conv_ND_cpu (conv.cpp:1002-1170,1220-1254), NOT Metal unfold/Winograd.
        let dtype = promote(x.dtype, weight.dtype)?;
        let x = self.cast(x, dtype, site)?;
        let w = self.cast(weight, dtype, site)?;
        let ph = add(x.shape[1], mul(2, padding)?)?;
        let pw = add(x.shape[2], mul(2, padding)?)?;
        if ph < w.shape[1] || pw < w.shape[2] {
            return Err(fail("convolution kernel exceeds padded input"));
        }
        let oh = (ph - w.shape[1]) / stride + 1;
        let ow = (pw - w.shape[2]) / stride + 1;
        let shape = [x.shape[0], oh, ow, w.shape[0]];
        let out = self.allocation(&shape, dtype, "cpu/conv.cpp Convolution output", site)?;
        self.scalar(Dtype::Float32, site)?;
        self.allocation(
            &[x.shape[0], ph, pw, x.shape[3]],
            Dtype::Float32,
            "cpu/conv.cpp in_padded Scalar copy",
            site,
        )?;
        self.allocation(
            &[
                mul(mul(x.shape[0], oh)?, ow)?,
                mul(mul(x.shape[3], w.shape[1])?, w.shape[2])?,
            ],
            Dtype::Float32,
            "cpu/conv.cpp in_strided General copy",
            site,
        )?;
        if w.dtype != Dtype::Float32 || !w.row_contiguous() {
            self.allocation(&w.shape, Dtype::Float32, "cpu/conv.cpp gemm_wt copy", site)?;
        }
        if dtype != Dtype::Float32 {
            self.allocation(
                &shape,
                Dtype::Float32,
                "cpu/conv.cpp gemm_out F32 backing",
                site,
            )?;
        }
        // BLAS-internal workspace is OUTSIDE this MLX counter, not an unknown zero-byte Array.
        if let Some(bias) = bias {
            self.binary(&out, bias, site)
        } else {
            Ok(out)
        }
    }
}
pub(super) fn broadcast_shape(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for i in 0..a.len().max(b.len()) {
        let x = a.len().checked_sub(i + 1).map_or(1, |j| a[j]);
        let y = b.len().checked_sub(i + 1).map_or(1, |j| b[j]);
        if x != y && x != 1 && y != 1 {
            return Err(fail("incompatible broadcast"));
        }
        out.push(if x == 1 { y } else { x });
    }
    out.reverse();
    Ok(out)
}
pub(super) fn promote(a: Dtype, b: Dtype) -> Result<Dtype> {
    if a == b {
        return Ok(a);
    }
    // Served Z numeric streams: MLX promotes mixed F16/BF16 and all F32 mixtures to F32.
    if matches!(a, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
        && matches!(b, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
    {
        return Ok(Dtype::Float32);
    }
    Err(fail("dtype promotion has no served Z graph contract"))
}
