//! The leaf modules of the Iris graph and the precision policy they share — the Candle twin of
//! `mlx-gen-iris`'s `nn` module, op for op.
//!
//! **Precision policy** (mirrors upstream's CUDA path, `torch.autocast(bfloat16)` over FP32
//! parameters, and degenerates to upstream's FP32 CPU path when the compute dtype is f32):
//!
//! * every `Linear` casts its input to the compute dtype and returns the compute dtype (autocast's
//!   matmul rule) — its weight and bias are stored in the compute dtype;
//! * every `RMSNorm` normalises in f32 and multiplies by an f32 gain, so it returns f32 (upstream's
//!   `RMSNorm` computes in fp32 and its fp32 gain promotes the product);
//! * attention runs in the compute dtype (`scaled_dot_product_attention` is on autocast's
//!   low-precision list); RoPE rotates in f32 and returns the dtype it was given;
//! * every other elementwise op follows ordinary **type promotion** — candle has none, so the
//!   binary helpers here ([`add`], [`mul`], [`cat`], …) promote to f32 whenever their operands
//!   disagree, exactly as torch (and MLX) do. The residual streams therefore stay f32 where
//!   upstream's do and the PiT pixel stream stays in the compute dtype where upstream's does.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::iris;
use candle_gen::{CandleError as Error, Result};

/// The dtype two operands meet at: themselves when they agree, f32 otherwise (the only two dtypes
/// this graph carries are f32 and the compute dtype).
fn common(a: DType, b: DType) -> DType {
    if a == b {
        a
    } else {
        DType::F32
    }
}

fn to(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(if x.dtype() == dtype {
        x.clone()
    } else {
        x.to_dtype(dtype)?
    })
}

/// Promoting broadcast `a + b`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_add(&to(b, dt)?)?)
}

/// Promoting broadcast `a − b`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_sub(&to(b, dt)?)?)
}

/// Promoting broadcast `a · b`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_mul(&to(b, dt)?)?)
}

/// `x · (1 + scale) + shift`, promoting like torch.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    add(&mul(x, &one_plus)?, shift)
}

/// Concatenate along `axis`, promoting every part to the common dtype first.
pub fn cat(parts: &[&Tensor], axis: usize) -> Result<Tensor> {
    let dt = parts
        .iter()
        .skip(1)
        .fold(parts[0].dtype(), |acc, p| common(acc, p.dtype()));
    let parts = parts
        .iter()
        .map(|p| to(p, dt))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&parts, axis)?)
}

/// Split the last axis into `n` equal chunks (`tensor.chunk(n, dim=-1)`).
pub fn chunk_last(x: &Tensor, n: usize) -> Result<Vec<Tensor>> {
    Ok(x.chunk(n, D::Minus1)?)
}

/// `x[.., :at, ..]` and `x[.., at:, ..]` along `axis`.
pub fn split_at(x: &Tensor, axis: usize, at: usize) -> Result<(Tensor, Tensor)> {
    let len = x.dim(axis)?;
    Ok((x.narrow(axis, 0, at)?, x.narrow(axis, at, len - at)?))
}

/// The checkpoint the backbone modules are built from: a `'static` mmap [`VarBuilder`] over the
/// single `model.safetensors` plus the key list from its header, so every key a module consumes is
/// recorded and a leftover (or missing) key is a load error naming it.
pub struct Checkpoint {
    vb: VarBuilder<'static>,
    /// Key prefix of this view (`""` for the whole file); `keys` are stored without it.
    prefix: String,
    keys: BTreeSet<String>,
    used: RefCell<BTreeSet<String>>,
    /// f32 adapter deltas folded into their weight as it is read (`W += δ`), keyed by checkpoint key.
    deltas: HashMap<String, Tensor>,
}

impl Checkpoint {
    /// mmap `file` onto `device` (tensors are read one at a time at f32, then cast by the caller).
    pub fn open(file: &Path, device: &Device) -> Result<Self> {
        Self::open_prefixed(file, device, "")
    }

    /// The view of `file` under `prefix` (e.g. the depth export's `pixel.` backbone): keys are
    /// taken and reported without the prefix, and keys outside it are not part of this view.
    pub fn open_prefixed(file: &Path, device: &Device, prefix: &str) -> Result<Self> {
        let keys = safetensors_keys(file)?
            .into_iter()
            .filter_map(|k| k.strip_prefix(prefix).map(str::to_owned))
            .collect();
        let vb = candle_gen::mmap_var_builder(&[file.to_path_buf()], DType::F32, device)?;
        Ok(Self {
            vb,
            prefix: prefix.to_owned(),
            keys,
            used: RefCell::new(BTreeSet::new()),
            deltas: HashMap::new(),
        })
    }

    /// Fold adapter `deltas` (f32 `[out, in]`, keyed by checkpoint key) into their weights as they
    /// are read. Every key must exist in the checkpoint.
    pub fn with_deltas(mut self, deltas: HashMap<String, Tensor>) -> Result<Self> {
        if let Some(key) = deltas.keys().find(|k| !self.keys.contains(*k)) {
            return Err(Error::Msg(format!(
                "iris: an adapter delta targets `{key}`, which the backbone checkpoint does not carry"
            )));
        }
        self.deltas = deltas;
        Ok(self)
    }

    /// Read one tensor (as f32, with any adapter delta folded in), recording the key as consumed.
    pub fn take(&self, key: &str) -> Result<Tensor> {
        if !self.keys.contains(key) {
            return Err(Error::Msg(format!(
                "iris: the backbone checkpoint is missing `{key}` (config.yaml and \
                 model.safetensors disagree)"
            )));
        }
        self.used.borrow_mut().insert(key.to_owned());
        let tensor = self.vb.get_unchecked(&format!("{}{key}", self.prefix))?;
        Ok(match self.deltas.get(key) {
            // The merge in f32, before the caller's cast to the compute dtype (`W += δ`).
            Some(delta) => (tensor + delta.to_device(self.vb.device())?)?,
            None => tensor,
        })
    }

    /// Keys the checkpoint carries that no module consumed, sorted.
    pub fn unused_keys(&self) -> Vec<String> {
        let used = self.used.borrow();
        self.keys.difference(&used).cloned().collect()
    }
}

/// The tensor names in a `.safetensors` header (`__metadata__` excluded), read without touching the
/// tensor bytes.
pub fn safetensors_keys(file: &Path) -> Result<BTreeSet<String>> {
    Ok(safetensors_shapes(file)?.into_keys().collect())
}

/// The tensor names and shapes in a `.safetensors` header (`__metadata__` excluded), read without
/// touching the tensor bytes.
pub fn safetensors_shapes(file: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    use std::io::Read;
    let read_err = |e: std::io::Error| Error::Msg(format!("iris: reading {}: {e}", file.display()));
    let mut f = std::fs::File::open(file).map_err(read_err)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(read_err)?;
    let len = u64::from_le_bytes(len);
    if len > 100 * 1024 * 1024 {
        return Err(Error::Msg(format!(
            "iris: {} has an implausible {len}-byte safetensors header",
            file.display()
        )));
    }
    let mut header = vec![0u8; len as usize];
    f.read_exact(&mut header).map_err(read_err)?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|e| {
        Error::Msg(format!(
            "iris: {} has a malformed safetensors header: {e}",
            file.display()
        ))
    })?;
    let map = value.as_object().ok_or_else(|| {
        Error::Msg(format!(
            "iris: {} safetensors header is not an object",
            file.display()
        ))
    })?;
    map.iter()
        .filter(|(k, _)| k.as_str() != "__metadata__")
        .map(|(k, v)| {
            let shape = v
                .get("shape")
                .and_then(|s| s.as_array())
                .and_then(|dims| {
                    dims.iter()
                        .map(|d| d.as_u64().map(|d| d as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "iris: {} header entry `{k}` has no valid shape",
                        file.display()
                    ))
                })?;
            Ok((k.clone(), shape))
        })
        .collect()
}

/// Builds modules from a checkpoint at one compute dtype.
pub struct Loader<'a> {
    pub weights: &'a Checkpoint,
    pub compute: DType,
}

impl Loader<'_> {
    /// A tensor kept in f32 (norm gains, adaLN biases, the text position table).
    pub fn f32(&self, key: &str) -> Result<Tensor> {
        to(&self.weights.take(key)?, DType::F32)
    }

    pub fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        let weight = to(
            &self.weights.take(&format!("{prefix}.weight"))?,
            self.compute,
        )?;
        let bias = if bias {
            Some(to(
                &self.weights.take(&format!("{prefix}.bias"))?,
                self.compute,
            )?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }

    pub fn norm(&self, prefix: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.f32(&format!("{prefix}.weight"))?,
            eps,
        })
    }
}

/// `nn.Linear` (`[out, in]` weight).
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.weight.dtype();
        let x = to(x, dtype)?;
        let dims = x.dims().to_vec();
        let (last, lead) = dims
            .split_last()
            .expect("linear input has at least one axis");
        let rows: usize = lead.iter().product();
        let out_dim = self.weight.dim(0)?;
        let flat = x.reshape((rows, *last))?;
        // Candle's CPU backend has no half-precision GEMM; there the bf16 policy is run as what a
        // bf16 GEMM is — bf16 operands, f32 accumulation, one rounding of the result.
        let gemm = gemm_dtype(&self.weight);
        let mut y = to(&flat, gemm)?.matmul(&to(&self.weight, gemm)?.t()?)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&to(b, gemm)?)?;
        }
        let y = to(&y, dtype)?;
        let mut shape = lead.to_vec();
        shape.push(out_dim);
        Ok(y.reshape(shape)?)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// A linear layer from an already-shaped `[out, in]` weight (and optional `[out]` bias), stored
    /// as given — e.g. the depth task's 1×1-conv reducer, whose `[1, C, 1, 1]` kernel is reshaped.
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

/// The dtype a GEMM over `t`'s dtype runs in on `t`'s device: itself, except a half-precision
/// operand on the Candle CPU backend (which has no bf16/f16 matmul kernel) accumulates in f32 and
/// is rounded back by the caller.
fn gemm_dtype(t: &Tensor) -> DType {
    kernel_dtype(t.device(), t.dtype())
}

/// See [`gemm_dtype`].
fn kernel_dtype(device: &Device, dtype: DType) -> DType {
    if device.is_cpu() && matches!(dtype, DType::BF16 | DType::F16) {
        DType::F32
    } else {
        dtype
    }
}

/// `iris3b.nn.norms.RMSNorm`: f32 normalisation, f32 gain.
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = to(x, DType::F32)?.contiguous()?;
        Ok(candle_gen::candle_nn::ops::rms_norm(
            &x,
            &self.weight,
            self.eps,
        )?)
    }
}

/// Rotation tables `(cos, sin)`, `[n, pairs]` f32, for [`Rope::apply`].
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    fn from_angles(angles: &[f32], n: usize, pairs: usize, device: &Device) -> Result<Self> {
        let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, pairs), device)?,
            sin: Tensor::from_vec(sin, (n, pairs), device)?,
        })
    }

    /// `rope_2d(head_dim, height, width, theta, scale, aspect="isotropic", frame_pairs=0)`:
    /// `head_dim/4` frequencies, interleaved (x, y) per pair along the last axis, row-major grid,
    /// both axes at one step `scale / (max(h, w) − 1)`.
    pub fn grid_2d(
        head_dim: usize,
        height: usize,
        width: usize,
        theta: f32,
        scale: f32,
        device: &Device,
    ) -> Result<Self> {
        let angles = iris::rope_2d_angles(head_dim, height, width, theta, scale);
        Self::from_angles(&angles, height * width, 2 * (head_dim / 4), device)
    }

    /// `rope_1d(head_dim, length, theta)`: standard 1-D RoPE over integer positions.
    pub fn line_1d(head_dim: usize, length: usize, theta: f32, device: &Device) -> Result<Self> {
        let angles = iris::rope_1d_angles(head_dim, length, theta);
        Self::from_angles(&angles, length, head_dim / 2, device)
    }

    /// Rotate `x` `[B, N, H, head_dim]` by complex multiplication over **adjacent** channel pairs
    /// (`view_as_complex`), in f32; returns `x`'s dtype.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, h, d) = x.dims4()?;
        let pairs = d / 2;
        let xf = to(x, DType::F32)?.reshape((b, n, h, pairs, 2))?;
        let re = xf.narrow(4, 0, 1)?.squeeze(4)?;
        let im = xf.narrow(4, 1, 1)?.squeeze(4)?;
        let cos = self.cos.reshape((1, n, 1, pairs))?;
        let sin = self.sin.reshape((1, n, 1, pairs))?;
        let out_re = re.broadcast_mul(&cos)?.sub(&im.broadcast_mul(&sin)?)?;
        let out_im = re.broadcast_mul(&sin)?.add(&im.broadcast_mul(&cos)?)?;
        let out = Tensor::stack(&[out_re, out_im], 4)?.reshape((b, n, h, d))?;
        to(&out, x.dtype())
    }
}

/// `[B, KV, N, hd]` → `[B, KV·rep, N, hd]`, head `h` reading kv head `h / rep` (torch's
/// `repeat_interleave` / SDPA `enable_gqa` layout).
fn repeat_kv(x: Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x);
    }
    let (b, kv, n, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .expand((b, kv, rep, n, d))?
        .reshape((b, kv * rep, n, d))?)
}

/// Grouped-query SDPA over `[B, N, heads, head_dim]` q and `[B, N, kv_heads, head_dim]` k/v, in the
/// compute dtype; returns `[B, N, heads·head_dim]`. `mask` is additive `[B|1, 1, N, N]` f32. The
/// scores run through the shared i32-overflow-safe query-row chunking (a 2048² render's trunk
/// attention is ~11e9 score elements under CFG).
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compute: DType,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, n, h, d) = q.dims4()?;
    let kv = k.dim(2)?;
    // Operands are rounded to the compute dtype; on the CPU backend (no half GEMM) a half compute
    // dtype then runs the kernel in f32 and rounds its output back, as in `Linear::forward`.
    let kernel = kernel_dtype(q.device(), compute);
    let t = |a: &Tensor| -> Result<Tensor> {
        Ok(to(&to(a, compute)?, kernel)?
            .transpose(1, 2)?
            .contiguous()?)
    };
    let q = t(q)?;
    let k = repeat_kv(t(k)?, h / kv)?;
    let v = repeat_kv(t(v)?, h / kv)?;
    let mask = mask.map(|m| to(m, kernel)).transpose()?;
    let scale = 1.0 / (d as f64).sqrt();
    let out = candle_gen::sdpa_budgeted_bhsd(
        &q,
        &k,
        &v,
        scale,
        mask.as_ref(),
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?;
    let out = to(&out, compute)?;
    Ok(out.transpose(1, 2)?.reshape((b, n, h * d))?)
}

/// `TransformerTextEmbedder`'s key mask: real keys OR the diagonal, additive `[B, 1, T, T]` f32.
pub fn key_padding_mask(mask: &[Vec<i32>], tokens: usize, device: &Device) -> Result<Tensor> {
    let data = iris::key_padding_additive(mask, tokens);
    Ok(Tensor::from_vec(
        data,
        (mask.len(), 1, tokens, tokens),
        device,
    )?)
}

/// Shape check with a named error (checkpoint/config disagreement surfaces at load, by key).
pub fn expect_shape(name: &str, a: &Tensor, want: &[usize]) -> Result<()> {
    if a.dims() != want {
        return Err(Error::Msg(format!(
            "iris: {name} has shape {:?}, the config implies {want:?}",
            a.dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_follows_torch() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1f32, 2.0], &dev).unwrap();
        let b = a.to_dtype(DType::BF16).unwrap();
        assert_eq!(add(&a, &b).unwrap().dtype(), DType::F32);
        assert_eq!(add(&b, &b).unwrap().dtype(), DType::BF16);
        assert_eq!(cat(&[&b, &a], 0).unwrap().dtype(), DType::F32);
    }

    #[test]
    fn gqa_repeat_is_interleaved_per_kv_head() {
        let dev = Device::Cpu;
        // kv heads hold 0 and 1; with rep 2 the four query heads read 0,0,1,1.
        let k = Tensor::new(&[0f32, 1.0], &dev)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let r = repeat_kv(k, 2).unwrap().flatten_all().unwrap();
        assert_eq!(r.to_vec1::<f32>().unwrap(), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let dev = Device::Cpu;
        let rope = Rope::line_1d(4, 1, 10_000.0, &dev).unwrap();
        let x = Tensor::new(&[1f32, 2.0, 3.0, 4.0], &dev)
            .unwrap()
            .reshape((1, 1, 1, 4))
            .unwrap();
        let y = rope.apply(&x).unwrap().flatten_all().unwrap();
        assert_eq!(y.to_vec1::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    }
}
