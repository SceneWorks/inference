//! Candle executor for **SceneWorks fx-program** checkpoints (`sceneworks-fx-program/1`, epic 2123
//! sc-24831) — the MediaPipe FaceMesh-v2 landmark detector of the face-landmark loss; the twin of
//! `mlx-gen-face`'s `program`. The structure is parsed once, backend-neutrally, by
//! [`candle_gen::gen_core::fx_program`]; this interpreter runs it NCHW (candle's native layout, the
//! program's own semantics) and is **differentiable in the input** (parameters are plain tensors,
//! never `Var`s, so backprop only reaches the input).
//!
//! Backward safety: every conv pads explicitly and, when strided, crops its input to the extent its
//! windows read (candle's Conv2D backward derives the transposed conv's output padding from the
//! height alone — the same workaround as `candle-gen-depth`'s `conv2d_nhwc`); max-pool is the max of
//! `index_select`ed window views (candle's own max-pool backward requires `kernel == stride` and
//! integer upsampling).

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::fx_program::{BinaryOp, Op, Operand, ProgramSpec, FORMAT};
use candle_gen::gen_core::weightsmeta;
use candle_gen::{CandleError, Result};

fn err(m: impl Into<String>) -> CandleError {
    CandleError::Msg(format!("fx-program: {}", m.into()))
}

/// A loaded program: its structure plus frozen parameters (torch layout, f32).
pub struct Program {
    spec: ProgramSpec,
    params: HashMap<String, Tensor>,
    device: Device,
}

/// Read a safetensors file's `__metadata__` entry `key`.
fn metadata(path: &Path, key: &str) -> Result<Option<String>> {
    let meta = weightsmeta::safetensors_file_metadata(path)
        .map_err(|e| err(format!("{}: {e}", path.display())))?;
    Ok(meta.get(key).cloned())
}

impl Program {
    /// Load a `sceneworks-fx-program/1` checkpoint (program in `__metadata__`, params as tensors).
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        match metadata(path, "format")?.as_deref() {
            Some(FORMAT) => {}
            other => {
                return Err(err(format!(
                    "{}: not a {FORMAT} checkpoint (format = {other:?})",
                    path.display()
                )))
            }
        }
        let json = metadata(path, "program")?.ok_or_else(|| {
            err(format!(
                "{}: missing the 'program' metadata",
                path.display()
            ))
        })?;
        let tensors = candle_gen::candle_core::safetensors::load(path, device)?;
        Self::new(ProgramSpec::parse(&json)?, tensors, device)
    }

    /// Bind `spec` to `tensors` (torch layout).
    pub fn new(
        spec: ProgramSpec,
        mut tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let mut params = HashMap::new();
        for (key, conv) in spec.param_keys() {
            let t = tensors
                .remove(&key)
                .ok_or_else(|| err(format!("missing parameter '{key}'")))?
                .to_dtype(DType::F32)?;
            if conv && t.rank() != 4 {
                return Err(err(format!(
                    "conv weight '{key}' must be 4-D, got {:?}",
                    t.dims()
                )));
            }
            params.insert(key, t);
        }
        Ok(Self {
            spec,
            params,
            device: device.clone(),
        })
    }

    fn param(&self, key: &str) -> Result<&Tensor> {
        self.params
            .get(key)
            .ok_or_else(|| err(format!("unbound parameter '{key}'")))
    }

    fn operand(&self, env: &HashMap<&str, Tensor>, o: &Operand) -> Result<Tensor> {
        Ok(match o {
            Operand::Value(n) => env
                .get(n.as_str())
                .cloned()
                .ok_or_else(|| err(format!("undefined value '{n}'")))?,
            Operand::Param(k) => self.param(k)?.clone(),
            Operand::Scalar(s) => Tensor::new(*s, &self.device)?,
        })
    }

    /// Run the program on an NCHW input; returns every program output (torch layout).
    /// Differentiable in `x`.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut env: HashMap<&str, Tensor> = HashMap::new();
        env.insert(self.spec.input.as_str(), x.clone());
        for node in &self.spec.nodes {
            let ins = node
                .inputs
                .iter()
                .map(|o| self.operand(&env, o))
                .collect::<Result<Vec<_>>>()?;
            let first = || {
                ins.first()
                    .cloned()
                    .ok_or_else(|| err(format!("node '{}' has no input", node.out)))
            };
            let out = match &node.op {
                Op::Conv {
                    weight,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let b = bias.as_ref().map(|b| self.param(b)).transpose()?;
                    conv2d(
                        &first()?,
                        self.param(weight)?,
                        b,
                        *stride,
                        *padding,
                        *dilation,
                        *groups,
                    )?
                }
                Op::Prelu { slope } => {
                    let x = first()?;
                    let s = self.param(slope)?;
                    let s = s.reshape((1, s.elem_count(), 1, 1))?;
                    let pos = x.relu()?;
                    pos.broadcast_add(&(&x - &pos)?.broadcast_mul(&s)?)?
                }
                Op::Relu => first()?.relu()?,
                Op::Sigmoid => sigmoid(&first()?)?,
                Op::Binary(op) => {
                    if ins.len() != 2 {
                        return Err(err(format!("binary node '{}' needs 2 operands", node.out)));
                    }
                    let (a, b) = (&ins[0], &ins[1]);
                    match op {
                        BinaryOp::Add => a.broadcast_add(b)?,
                        BinaryOp::Sub => a.broadcast_sub(b)?,
                        BinaryOp::Mul => a.broadcast_mul(b)?,
                        BinaryOp::Div => a.broadcast_div(b)?,
                    }
                }
                Op::MaxPool {
                    kernel,
                    stride,
                    padding,
                } => max_pool(&first()?, *kernel, *stride, *padding)?,
                Op::Pad { pads, value } => {
                    let mut t = first()?;
                    let rank = t.rank();
                    for (axis, &(before, after)) in pads[4 - rank.min(4)..].iter().enumerate() {
                        t = pad_axis(&t, axis, before, after, *value)?;
                    }
                    t
                }
                Op::Reshape(shape) => {
                    let t = first()?.contiguous()?;
                    let dims = ProgramSpec::resolve_shape(shape, t.elem_count())?;
                    t.reshape(dims)?
                }
                Op::Concat(axis) => {
                    let rank = ins
                        .first()
                        .map(|t| t.rank() as i64)
                        .ok_or_else(|| err("empty concat"))?;
                    Tensor::cat(&ins, axis.rem_euclid(rank) as usize)?
                }
            };
            env.insert(node.out.as_str(), out);
        }
        self.spec
            .outputs
            .iter()
            .map(|o| self.operand(&env, o))
            .collect()
    }
}

/// `1 / (1 + e^{-x})` from differentiable primitives.
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok((x.neg()?.exp()? + 1.0)?.recip()?)
}

/// Constant-pad `axis` of `t` by `(before, after)` with `value`.
fn pad_axis(t: &Tensor, axis: usize, before: usize, after: usize, value: f32) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(t.clone());
    }
    if value == 0.0 {
        return Ok(t.pad_with_zeros(axis, before, after)?);
    }
    let mut parts = Vec::with_capacity(3);
    let mut dims = t.dims().to_vec();
    if before > 0 {
        dims[axis] = before;
        parts.push(Tensor::full(value, dims.clone(), t.device())?);
    }
    parts.push(t.clone());
    if after > 0 {
        dims[axis] = after;
        parts.push(Tensor::full(value, dims, t.device())?);
    }
    Ok(Tensor::cat(&parts, axis)?)
}

/// Backward-safe NCHW conv: explicit zero padding, and for a strided conv the input cropped to the
/// extent the windows read, so candle's Conv2D backward never sees unequal H/W remainders.
fn conv2d(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    if stride.0 != stride.1 || dilation.0 != dilation.1 {
        return Err(err(format!(
            "anisotropic conv stride {stride:?} / dilation {dilation:?} is not supported"
        )));
    }
    let x = x
        .pad_with_zeros(2, padding.0, padding.0)?
        .pad_with_zeros(3, padding.1, padding.1)?;
    let (_, _, h, wd) = x.dims4()?;
    let (s, d) = (stride.0, dilation.0);
    let ext = |k: usize| d * (k - 1) + 1;
    let (kh, kw) = (ext(w.dim(2)?), ext(w.dim(3)?));
    if h < kh || wd < kw {
        return Err(err(format!("conv input {h}×{wd} smaller than its kernel")));
    }
    let used = |n: usize, k: usize| ((n - k) / s) * s + k;
    let x = x
        .narrow(2, 0, used(h, kh))?
        .narrow(3, 0, used(wd, kw))?
        .contiguous()?;
    let y = x.conv2d(w, 0, s, d, groups)?;
    Ok(match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1, 1))?)?,
        None => y,
    })
}

/// Max-pool (floor mode, `-inf` padding) as the max of `index_select`ed window views.
fn max_pool(
    x: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    let x = pad_axis(x, 2, padding.0, padding.0, f32::NEG_INFINITY)?;
    let x = pad_axis(&x, 3, padding.1, padding.1, f32::NEG_INFINITY)?;
    let (_, _, h, w) = x.dims4()?;
    if h < kernel.0 || w < kernel.1 {
        return Err(err(format!(
            "max-pool kernel {kernel:?} exceeds the padded input {h}×{w}"
        )));
    }
    let oh = (h - kernel.0) / stride.0 + 1;
    let ow = (w - kernel.1) / stride.1 + 1;
    let idx = |off: usize, n: usize, s: usize| -> Result<Tensor> {
        let v: Vec<u32> = (0..n).map(|i| (off + i * s) as u32).collect();
        Ok(Tensor::from_vec(v, n, x.device())?)
    };
    let mut out: Option<Tensor> = None;
    for dy in 0..kernel.0 {
        let rows = x.index_select(&idx(dy, oh, stride.0)?, 2)?;
        for dx in 0..kernel.1 {
            let view = rows.index_select(&idx(dx, ow, stride.1)?, 3)?;
            out = Some(match out {
                Some(m) => m.maximum(&view)?,
                None => view,
            });
        }
    }
    out.ok_or_else(|| err("empty max-pool kernel"))
}
