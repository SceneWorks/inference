//! Per-stream feed-forward: `mlp_out(gelu_approx(mlp_in(x)))` (both biased, 4× expansion).
//! Port of the fork's `QwenFeedForward`. Both Linears are [`AdaptableLinear`] so the transformer
//! can be quantized (Q8) without changing the forward.

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, power, tanh};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{compile_glue, join, linear_from};

pub struct FeedForward {
    mlp_in: AdaptableLinear,
    mlp_out: AdaptableLinear,
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file (diffusers) naming: `{img,txt}_mlp.net.0.proj` (in) / `.net.2` (out).
        match path {
            ["net", "0", "proj"] => Some(&mut self.mlp_in),
            ["net", "2"] => Some(&mut self.mlp_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["net.0.proj", "net.2"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp_in: linear_from(w, &join(prefix, "mlp_in"), true)?,
            mlp_out: linear_from(w, &join(prefix, "mlp_out"), true)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Dtype-preserving, golden-bit-exact tanh-GELU (sc-2779). `mlx_rs::nn::gelu_approximate`
        // uses an f32 `√(2/π)` (1 ULP off the fork's f64-host const) and promotes a bf16 input to
        // f32; `gelu_tanh` matches `nn.GELU(approx="tanh")` and preserves the input dtype.
        let h = gelu_ffn(&self.mlp_in.forward(x)?)?;
        self.mlp_out.forward(&h)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mlp_in.quantize(bits, None)?;
        self.mlp_out.quantize(bits, None)?;
        Ok(())
    }
}

/// The tanh-GELU FFN activation. Body mirrors [`mlx_gen::nn::gelu_tanh`] exactly (dtype-preserving,
/// f64-host `√(2/π)`); when the sc-2963 glue toggle is on, MLX fuses its ~8 elementwise ops into one
/// kernel — the single biggest per-step glue cost (the 4× FFN expansion). Off ⇒ defers to the core
/// `gelu_tanh`, so the eager path is byte-for-byte the previous behaviour.
fn gelu_ffn(x: &Array) -> Result<Array> {
    if !compile_glue() {
        return gelu_tanh(x);
    }
    let f = |x_: &Array| -> std::result::Result<Array, Exception> {
        let dt = x_.dtype();
        let s = |v: f32| -> std::result::Result<Array, Exception> { scalar(v).as_dtype(dt) };
        let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
        let x3 = power(x_, Array::from_int(3))?;
        let inner = multiply(&add(x_, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
        let gate = add(&tanh(&inner)?, &s(1.0)?)?;
        multiply(&multiply(x_, &s(0.5)?)?, &gate)
    };
    Ok(compile(f, true)(x)?)
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use crate::transformer::compile_test_util::rnd;
    use crate::transformer::set_compile_glue;
    use mlx_rs::Dtype::{Bfloat16, Float32};

    // sc-2963: the compiled tanh-GELU FFN activation is bit-identical to eager (`max|Δ|=0`); the
    // eager branch defers to core `gelu_tanh`, so this also proves the inline body matches it exactly.
    #[test]
    fn compiled_gelu_ffn_bit_identical_to_eager() {
        for dt in [Float32, Bfloat16] {
            let x = rnd(&[2, 16, 512], dt);
            set_compile_glue(false);
            let e = gelu_ffn(&x).unwrap();
            set_compile_glue(true);
            let c = gelu_ffn(&x).unwrap();
            set_compile_glue(false);
            assert_eq!(c.dtype(), e.dtype(), "gelu_ffn preserves dtype {dt:?}");
            // sc-12747: under MLX 0.32.0 the compiled tanh-GELU FFN rounds ~1 ULP-f32 differently
            // from eager (0-ULP on the prior 0.31.2 pin); bf16 stays bit-identical. f32 takes the
            // shared re-baselined tolerance; bf16 stays exact.
            let tol = if dt == Float32 {
                mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL
            } else {
                0.0
            };
            let rel = mlx_gen::nn::max_rel_diff(&c, &e);
            assert!(
                rel <= tol,
                "gelu_ffn compiled vs eager {dt:?}: rel|Δ|={rel:e} exceeds {tol:e}"
            );
        }
    }
}
