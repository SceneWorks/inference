//! The two residual streams carried by a SenseNova mHC block.
use super::{Error, Result, Weights};
use core_llm::sensenova_loop::SenseNovaLoop;
use mlx_rs::ops::{
    add, broadcast_to, exp, logsumexp_axis, mean_axis, multiply, rsqrt, sigmoid, split_sections,
    subtract,
};
use mlx_rs::{Array, Dtype};

#[derive(Debug)]
struct Router {
    weight: Array,
    bias: Array,
}

struct Routing {
    input: Array,
    post: Array,
    mixing: Array,
}

impl Router {
    fn load(weights: &Weights, prefix: &str, hidden: i32) -> Result<Self> {
        let tensor = |name: &str, shape: &[i32]| -> Result<Array> {
            let key = format!("{prefix}.{name}");
            let value = weights.require(&key)?;
            if value.dtype() != Dtype::Float32 || value.shape() != shape {
                return Err(Error::Config(format!(
                    "{key}: mHC requires original FP32 {shape:?}, found {:?} {:?}",
                    value.dtype(),
                    value.shape()
                )));
            }
            Ok(value.clone())
        };
        Ok(Self {
            weight: tensor("weight", &[8, 2 * hidden])?,
            bias: tensor("bias", &[8])?,
        })
    }

    fn route(&self, residual: &Array) -> Result<Routing> {
        let (batch, tokens, hidden) = (
            residual.shape()[0],
            residual.shape()[1],
            residual.shape()[3],
        );
        let flat = residual
            .reshape(&[batch, tokens, 2 * hidden])?
            .as_dtype(Dtype::Float32)?;
        let variance = mean_axis(&multiply(&flat, &flat)?, -1, true)?;
        let normalized = multiply(&flat, rsqrt(add(variance, Array::from_f32(1e-6))?)?)?;
        let logits = add(normalized.matmul(self.weight.t())?, &self.bias)?;
        let parts = split_sections(&logits, &[2, 4], -1)?;
        let pre = sigmoid(&parts[0])?
            .as_dtype(residual.dtype())?
            .reshape(&[batch, tokens, 1, 2])?;
        let input = pre.matmul(residual)?.reshape(&[batch, tokens, hidden])?;
        let post =
            multiply(sigmoid(&parts[1])?, Array::from_f32(2.))?.reshape(&[batch, tokens, 2, 1])?;
        let mut log_mixing = parts[2].reshape(&[batch, tokens, 2, 2])?;
        for _ in 0..SenseNovaLoop::ITERATIONS {
            log_mixing = subtract(&log_mixing, logsumexp_axis(&log_mixing, -1, true)?)?;
            log_mixing = subtract(&log_mixing, logsumexp_axis(&log_mixing, -2, true)?)?;
        }
        Ok(Routing {
            input,
            post,
            mixing: exp(&log_mixing)?,
        })
    }
}

impl Routing {
    fn merge(self, residual: &Array, branch: &Array) -> Result<Array> {
        let mixed = self.mixing.as_dtype(residual.dtype())?.matmul(residual)?;
        let update = multiply(self.post.as_dtype(branch.dtype())?, branch.expand_dims(-2)?)?;
        Ok(add(mixed, update)?)
    }
}

#[derive(Debug)]
pub(super) struct RoutedBlock {
    physical: usize,
    attention: Router,
    ffn: Router,
}

impl RoutedBlock {
    pub(super) fn load(
        weights: &Weights,
        prefix: &str,
        physical: usize,
        hidden: i32,
    ) -> Result<Self> {
        Ok(Self {
            physical,
            attention: Router::load(weights, &format!("{prefix}.mhc_attn"), hidden)?,
            ffn: Router::load(weights, &format!("{prefix}.mhc_ffn"), hidden)?,
        })
    }

    pub(super) fn arrays(&self, out: &mut Vec<Array>) {
        for router in [&self.attention, &self.ffn] {
            out.extend([router.weight.clone(), router.bias.clone()]);
        }
    }

    pub(super) fn forward(
        &self,
        x: &Array,
        attention: impl FnOnce(&Array) -> Result<Array>,
        ffn: impl FnOnce(&Array) -> Result<Array>,
    ) -> Result<Array> {
        let (b, s, width) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let residual = if self.physical == SenseNovaLoop::START {
            broadcast_to(x.expand_dims(-2)?, &[b, s, 2, width])?
        } else {
            x.reshape(&[b, s, 2, width / 2])?
        };
        let route = self.attention.route(&residual)?;
        let update = attention(&route.input)?;
        let residual = route.merge(&residual, &update)?;
        let route = self.ffn.route(&residual)?;
        let update = ffn(&route.input)?;
        let residual = route.merge(&residual, &update)?;
        // Each router's 20 row/column reductions otherwise stay in the lazy
        // graph through both passes. Bound the graph and scratch to one block.
        residual.eval()?;
        if self.physical + 1 == SenseNovaLoop::END {
            Ok(mean_axis(&residual, -2, false)?)
        } else {
            Ok(residual.reshape(&[b, s, -1])?)
        }
    }
}
