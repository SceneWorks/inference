//! SenseNova-Loop's two residual streams. Router weights and Sinkhorn
//! reductions stay FP32. Only the stream products use the activation dtype.
use super::*;
use candle_core::D;
use core_llm::sensenova_loop::SenseNovaLoop;

pub(super) struct Routers {
    attention: Router,
    ffn: Router,
    physical: usize,
}

struct Router {
    weight: Tensor,
    bias: Tensor,
}

struct Routed {
    branch: Tensor,
    post: Tensor,
    mix: Tensor,
}

impl Router {
    fn load(w: &Weights, prefix: &str, hidden: usize) -> Result<Self> {
        let read = |suffix: &str, shape: &[usize]| {
            let key = format!("{prefix}.{suffix}");
            let tensor = w.plain(&key).ok_or_else(|| {
                Error::Config(format!(
                    "SenseNova-Loop requires the published F32 router coefficient {key}"
                ))
            })?;
            if tensor.dtype() != DType::F32 || tensor.dims() != shape {
                return Err(Error::Config(format!(
                    "SenseNova-Loop requires {key} to be F32 {shape:?}, found {:?} {:?}",
                    tensor.dtype(),
                    tensor.dims()
                )));
            }
            Ok(tensor.clone())
        };
        Ok(Self {
            weight: read("weight", &[8, 2 * hidden])?,
            bias: read("bias", &[8])?,
        })
    }

    fn route(&self, streams: &Tensor, b: usize, s: usize, h: usize) -> Result<Routed> {
        let flat = streams.reshape((b * s, 2 * h))?.to_dtype(DType::F32)?;
        let inverse_rms = flat
            .sqr()?
            .mean_keepdim(1)?
            .affine(1., 1e-6)?
            .sqrt()?
            .recip()?;
        let logits = flat
            .broadcast_mul(&inverse_rms)?
            .matmul(&self.weight.t()?)?
            .broadcast_add(&self.bias)?;
        let pre = sigmoid(&logits.narrow(1, 0, 2)?)?
            .to_dtype(streams.dtype())?
            .reshape((b * s, 1, 2))?;
        let post = sigmoid(&logits.narrow(1, 2, 2)?)?.affine(2., 0.)?;
        let mut matrix = logits
            .narrow(1, 4, 4)?
            .contiguous()?
            .reshape((b * s, 2, 2))?;
        for _ in 0..SenseNovaLoop::ITERATIONS {
            for axis in [2, 1] {
                let partition = matrix.log_sum_exp(axis)?.unsqueeze(axis)?;
                matrix = matrix.broadcast_sub(&partition)?;
            }
        }
        Ok(Routed {
            branch: pre.matmul(streams)?.reshape((b, s, h))?,
            post,
            mix: matrix.exp()?,
        })
    }
}

impl Routed {
    fn merge(&self, streams: &Tensor, branch: &Tensor) -> Result<Tensor> {
        let (n, _, h) = streams.dims3()?;
        let mixed = self.mix.to_dtype(streams.dtype())?.matmul(streams)?;
        let update = self
            .post
            .to_dtype(branch.dtype())?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&branch.reshape((n, 1, h))?)?;
        Ok((mixed + update)?)
    }
}

impl Routers {
    pub(super) fn load(w: &Weights, layer: String, physical: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            attention: Router::load(w, &format!("{layer}.mhc_attn"), hidden)?,
            ffn: Router::load(w, &format!("{layer}.mhc_ffn"), hidden)?,
            physical,
        })
    }

    pub(super) fn tensors(&self) -> [&Tensor; 4] {
        [
            &self.attention.weight,
            &self.attention.bias,
            &self.ffn.weight,
            &self.ffn.bias,
        ]
    }

    pub(super) fn forward(
        &self,
        x: &Tensor,
        attention: impl FnOnce(&Tensor) -> Result<Tensor>,
        ffn: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let (b, s, width) = x.dims3()?;
        let streams = if self.physical == SenseNovaLoop::START {
            x.reshape((b * s, 1, width))?
                .broadcast_as((b * s, 2, width))?
                .contiguous()?
        } else {
            x.reshape((b * s, 2, width / 2))?
        };
        let h = streams.dim(2)?;
        let attention_streams = {
            let routed = self.attention.route(&streams, b, s, h)?;
            routed.merge(&streams, &attention(&routed.branch)?)?
        };
        let routed = self.ffn.route(&attention_streams, b, s, h)?;
        let output = routed.merge(&attention_streams, &ffn(&routed.branch)?)?;
        if self.physical + 1 == SenseNovaLoop::END {
            Ok(output
                .to_dtype(DType::F32)?
                .mean(1)?
                .to_dtype(x.dtype())?
                .reshape((b, s, h))?)
        } else {
            Ok(output.reshape((b, s, 2 * h))?)
        }
    }
}
