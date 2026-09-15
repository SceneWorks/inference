//! Per-stream feed-forward: `mlp_out(gelu_approx(mlp_in(x)))` (both biased, 4× expansion).
//! Port of the fork's `QwenFeedForward`. Both Linears are [`AdaptableLinear`] so the transformer
//! can be quantized (Q8) without changing the forward.

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, power, tanh};
use mlx_rs::transforms::compile::{compile, compile_retained};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{compile_glue, join, linear_from};

const SITE_GELU_FFN: &str = "qwen_image::feed_forward::gelu_ffn";

fn gelu_ffn_impl(x: &Array) -> std::result::Result<Array, Exception> {
    let dt = x.dtype();
    let s = |v: f32| -> std::result::Result<Array, Exception> { scalar(v).as_dtype(dt) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    multiply(&multiply(x, &s(0.5)?)?, &gate)
}

thread_local! {
    static RETAINED_GELU_FFN: std::cell::RefCell<Option<mlx_gen::nn::RetainedUnary>> =
        const { std::cell::RefCell::new(None) };
}

fn retained_gelu_ffn(x: &Array) -> std::result::Result<Array, Exception> {
    mlx_gen::nn::prepare_retained_compilation_thread();
    RETAINED_GELU_FFN.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| {
                mlx_gen::nn::RetainedUnary::new(compile_retained(gelu_ffn_impl, true))
            })
            .call(SITE_GELU_FFN, x)
    })
}

/// Exercise this module's production retained handle once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &Array) -> Result<()> {
    let output = retained_gelu_ffn(input)?;
    output.eval()?;
    drop(output);
    Ok(())
}

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
        mlx_gen::diagnostics::record_fallback(SITE_GELU_FFN, "compiled_glue_disabled");
        return gelu_tanh(x);
    }
    if mlx_gen::nn::retained_compilation_requested() {
        Ok(retained_gelu_ffn(x)?)
    } else {
        mlx_gen::diagnostics::record_compile(
            SITE_GELU_FFN,
            mlx_gen::diagnostics::CompileDisposition::OneShot,
        );
        Ok(compile(gelu_ffn_impl, true)(x)?)
    }
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

    #[test]
    fn retained_gelu_reports_miss_hit_applied_and_oneshot_baseline() {
        use mlx_gen::diagnostics::{
            self, CompileDisposition, DiagnosticCounter, ToggleDisposition, RETAINED_COMPILATION,
        };

        RETAINED_GELU_FFN.with(|slot| *slot.borrow_mut() = None);
        let x = rnd(&[1, 4, 16], Bfloat16);
        set_compile_glue(false);
        let eager = gelu_ffn(&x).unwrap();

        set_compile_glue(true);
        let scope = diagnostics::begin_request_with_toggles(
            "qwen-retained-gelu",
            "qwen_image",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        for _ in 0..2 {
            assert_eq!(
                mlx_gen::nn::max_rel_diff(&gelu_ffn(&x).unwrap(), &eager),
                0.0
            );
        }
        let report = scope.finish();
        for disposition in [
            CompileDisposition::RetainedMiss,
            CompileDisposition::RetainedHit,
        ] {
            assert!(report.counters.iter().any(|counter| matches!(
                counter,
                DiagnosticCounter::Compile {
                    site: SITE_GELU_FFN,
                    disposition: recorded,
                    count: 1,
                } if *recorded == disposition
            )));
        }
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                count: 2,
            }
        )));

        let baseline = diagnostics::begin_request("qwen-oneshot-gelu", "qwen_image").unwrap();
        let oneshot = gelu_ffn(&x).unwrap();
        let baseline = baseline.finish();
        assert_eq!(mlx_gen::nn::max_rel_diff(&oneshot, &eager), 0.0);
        assert!(baseline.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GELU_FFN,
                disposition: CompileDisposition::OneShot,
                count: 1,
            }
        )));

        set_compile_glue(false);
        RETAINED_GELU_FFN.with(|slot| *slot.borrow_mut() = None);
    }
}
