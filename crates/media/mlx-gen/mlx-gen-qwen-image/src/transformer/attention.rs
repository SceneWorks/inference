//! Joint (dual-stream) attention. Port of the fork's `QwenAttention`: separate q/k/v projections
//! for the image (`to_*`) and text (`add_*_proj`) streams, per-head q/k RMSNorm, **interleaved**
//! complex RoPE, then attention over the concatenated `[txt, img]` sequence, split back into the
//! two streams and projected (`attn_to_out.0` / `to_add_out`). All eight projections are
//! [`AdaptableLinear`] (Q8-quantizable); the q/k RMSNorm weights stay dense.

use mlx_rs::error::Exception;
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply, split, split_sections, subtract};
use mlx_rs::transforms::compile::{compile, compile_retained};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{sdpa_budgeted_bhsd, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{compile_glue, join, linear_from};

const RMS_EPS: f32 = 1e-6;
const SITE_ROPE_ROTATE: &str = "qwen_image::attention::rope_rotate";

fn rope_rotate_impl(inp: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (real, imag, cos, sin) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    let out_real = subtract(&multiply(real, cos)?, &multiply(imag, sin)?)?;
    let out_imag = add(&multiply(real, sin)?, &multiply(imag, cos)?)?;
    Ok(vec![out_real, out_imag])
}

thread_local! {
    static RETAINED_ROPE_ROTATE: std::cell::RefCell<Option<mlx_gen::nn::RetainedSlice>> =
        const { std::cell::RefCell::new(None) };
}

fn retained_rope_rotate(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    mlx_gen::nn::prepare_retained_compilation_thread();
    RETAINED_ROPE_ROTATE.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| {
                mlx_gen::nn::RetainedSlice::new(compile_retained(rope_rotate_impl, true))
            })
            .call(SITE_ROPE_ROTATE, args)
    })
}

/// Exercise this module's production retained handle once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &Array) -> Result<()> {
    let args = [input.clone(), input.clone(), input.clone(), input.clone()];
    let outputs = retained_rope_rotate(&args)?;
    mlx_rs::transforms::eval(outputs.iter())?;
    drop(outputs);
    drop(args);
    Ok(())
}

pub struct QwenJointAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    to_out: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl AdaptableHost for QwenJointAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file (diffusers) naming → fields: image stream `to_q/k/v` + `to_out.0`; text
        // stream `add_{q,k,v}_proj` → `add_{q,k,v}` and `to_add_out`.
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            ["add_q_proj"] => Some(&mut self.add_q),
            ["add_k_proj"] => Some(&mut self.add_k),
            ["add_v_proj"] => Some(&mut self.add_v),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

impl QwenJointAttention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        let g = |s: &str| w.require(&join(prefix, s)).cloned();
        Ok(Self {
            to_q: linear_from(w, &join(prefix, "to_q"), true)?,
            to_k: linear_from(w, &join(prefix, "to_k"), true)?,
            to_v: linear_from(w, &join(prefix, "to_v"), true)?,
            add_q: linear_from(w, &join(prefix, "add_q_proj"), true)?,
            add_k: linear_from(w, &join(prefix, "add_k_proj"), true)?,
            add_v: linear_from(w, &join(prefix, "add_v_proj"), true)?,
            to_out: linear_from(w, &join(prefix, "attn_to_out.0"), true)?,
            to_add_out: linear_from(w, &join(prefix, "to_add_out"), true)?,
            norm_q: g("norm_q.weight")?,
            norm_k: g("norm_k.weight")?,
            norm_added_q: g("norm_added_q.weight")?,
            norm_added_k: g("norm_added_k.weight")?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_q.quantize(bits, None)?;
        self.to_k.quantize(bits, None)?;
        self.to_v.quantize(bits, None)?;
        self.add_q.quantize(bits, None)?;
        self.add_k.quantize(bits, None)?;
        self.add_v.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        self.to_add_out.quantize(bits, None)?;
        Ok(())
    }

    /// `img`/`txt`: `[B, seq, dim]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, txt+img]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        self.forward_budgeted(
            img,
            txt,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            mask,
            AttentionPlan::UNBOUNDED,
        )
    }

    /// Joint attention with the shared request-scoped rung-3 query budget.
    ///
    /// The unbounded plan is the historical single fused-SDPA call. A constrained plan splits only
    /// query rows, so every row still attends over the complete text+image key/value sequence and
    /// preserves the same projections, RoPE, scale, mask, precision, and output split.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_budgeted(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
        plan: AttentionPlan<'_>,
    ) -> Result<(Array, Array)> {
        let (b, img_seq) = (img.shape()[0], img.shape()[1]);
        let txt_seq = txt.shape()[1];
        let (h, hd) = (self.num_heads, self.head_dim);
        let to_heads = |lin: &AdaptableLinear, x: &Array, seq: i32| -> Result<Array> {
            Ok(lin.forward(x)?.reshape(&[b, seq, h, hd])?)
        };

        let img_q = rms_norm(&to_heads(&self.to_q, img, img_seq)?, &self.norm_q, RMS_EPS)?;
        let img_k = rms_norm(&to_heads(&self.to_k, img, img_seq)?, &self.norm_k, RMS_EPS)?;
        let img_v = to_heads(&self.to_v, img, img_seq)?;
        let txt_q = rms_norm(
            &to_heads(&self.add_q, txt, txt_seq)?,
            &self.norm_added_q,
            RMS_EPS,
        )?;
        let txt_k = rms_norm(
            &to_heads(&self.add_k, txt, txt_seq)?,
            &self.norm_added_k,
            RMS_EPS,
        )?;
        let txt_v = to_heads(&self.add_v, txt, txt_seq)?;

        let img_q = apply_rope_qwen(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope_qwen(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope_qwen(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope_qwen(&txt_k, txt_cos, txt_sin)?;

        // joint [txt, img] over the sequence axis, then to [B, heads, seq, head_dim] for SDPA.
        let q = concatenate_axis(&[&txt_q, &img_q], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[&txt_k, &img_k], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = concatenate_axis(&[&txt_v, &img_v], 1)?.transpose_axes(&[0, 2, 1, 3])?;

        let o = sdpa_budgeted_bhsd(&q, &k, &v, self.scale, mask, plan)?;
        let joint = txt_seq + img_seq;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, joint, h * hd])?;

        // Split back along the sequence axis at the static text/image boundary: a zero-copy strided
        // split, vs the old pair of arange `take_axis` gathers run 60 blocks × 2 CFG / step (F-114).
        let parts = split_sections(&o, &[txt_seq], 1)?;
        let txt_attn = self.to_add_out.forward(&parts[0])?;
        let img_attn = self.to_out.forward(&parts[1])?;
        Ok((img_attn, txt_attn))
    }
}

/// Interleaved complex RoPE: pairs `(x_2i, x_2i+1)` rotated by `(cos_i, sin_i)`. `x`:
/// `[B, seq, heads, head_dim]`; `cos`/`sin`: `[seq, head_dim/2]`.
fn apply_rope_qwen(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x.reshape(&[b, seq, heads, half, 2])?;
    let parts = split(&x5, 2, 4)?; // even/odd lanes, each [B,seq,heads,half,1]
    let xr = parts[0].reshape(&[b, seq, heads, half])?;
    let xi = parts[1].reshape(&[b, seq, heads, half])?;
    let cos = cos.reshape(&[1, seq, 1, half])?;
    let sin = sin.reshape(&[1, seq, 1, half])?;
    let (out_r, out_i) = rope_rotate(&xr, &xi, &cos, &sin)?;
    let stacked = concatenate_axis(&[&out_r.expand_dims(4)?, &out_i.expand_dims(4)?], 4)?;
    Ok(stacked.reshape(&[b, seq, heads, hd])?)
}

/// The complex RoPE rotation `(xr + xi·i)·(cos + sin·i)` → `(out_r, out_i)`. Fused into one kernel
/// when the sc-2963 glue toggle is on (vs 6 eager ops, applied to img/txt q and k every block);
/// dtype-preserving, bit-identical to the eager form.
fn rope_rotate(xr: &Array, xi: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let args = [xr.clone(), xi.clone(), cos.clone(), sin.clone()];
    let mut out = if compile_glue() {
        if mlx_gen::nn::retained_compilation_requested() {
            retained_rope_rotate(&args)?
        } else {
            mlx_gen::diagnostics::record_compile(
                SITE_ROPE_ROTATE,
                mlx_gen::diagnostics::CompileDisposition::OneShot,
            );
            compile(rope_rotate_impl, true)(&args)?
        }
    } else {
        mlx_gen::diagnostics::record_fallback(SITE_ROPE_ROTATE, "compiled_glue_disabled");
        rope_rotate_impl(&args)?
    };
    let out_i = out.pop().unwrap();
    let out_r = out.pop().unwrap();
    Ok((out_r, out_i))
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use crate::transformer::compile_test_util::{max_abs, rnd};
    use crate::transformer::set_compile_glue;
    use mlx_rs::Dtype::Float32;

    // sc-2963: the compiled RoPE rotation is bit-identical to eager (`max|Δ|=0`).
    #[test]
    fn compiled_rope_rotate_bit_identical_to_eager() {
        let (b, seq, heads, half) = (2i32, 16i32, 2i32, 64i32);
        let xr = rnd(&[b, seq, heads, half], Float32);
        let xi = rnd(&[b, seq, heads, half], Float32);
        let cos = rnd(&[1, seq, 1, half], Float32);
        let sin = rnd(&[1, seq, 1, half], Float32);
        set_compile_glue(false);
        let (er, ei) = rope_rotate(&xr, &xi, &cos, &sin).unwrap();
        set_compile_glue(true);
        let (cr, ci) = rope_rotate(&xr, &xi, &cos, &sin).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&cr, &er), 0.0, "rope_rotate real");
        assert_eq!(max_abs(&ci, &ei), 0.0, "rope_rotate imag");
    }

    #[test]
    fn retained_rope_reports_miss_hit_and_applied() {
        use mlx_gen::diagnostics::{
            self, CompileDisposition, DiagnosticCounter, ToggleDisposition, RETAINED_COMPILATION,
        };

        RETAINED_ROPE_ROTATE.with(|slot| *slot.borrow_mut() = None);
        let xr = rnd(&[1, 4, 1, 8], Float32);
        let xi = rnd(&[1, 4, 1, 8], Float32);
        let cos = rnd(&[1, 4, 1, 8], Float32);
        let sin = rnd(&[1, 4, 1, 8], Float32);
        set_compile_glue(false);
        let eager = rope_rotate(&xr, &xi, &cos, &sin).unwrap();

        set_compile_glue(true);
        let scope = diagnostics::begin_request_with_toggles(
            "qwen-retained-rope",
            "qwen_image",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        for retained in [
            rope_rotate(&xr, &xi, &cos, &sin).unwrap(),
            rope_rotate(&xr, &xi, &cos, &sin).unwrap(),
        ] {
            assert_eq!(max_abs(&retained.0, &eager.0), 0.0);
            assert_eq!(max_abs(&retained.1, &eager.1), 0.0);
        }
        let report = scope.finish();
        for disposition in [
            CompileDisposition::RetainedMiss,
            CompileDisposition::RetainedHit,
        ] {
            assert!(report.counters.iter().any(|counter| matches!(
                counter,
                DiagnosticCounter::Compile {
                    site: SITE_ROPE_ROTATE,
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

        let baseline = diagnostics::begin_request("qwen-oneshot-rope", "qwen_image").unwrap();
        let _ = rope_rotate(&xr, &xi, &cos, &sin).unwrap();
        let baseline = baseline.finish();
        assert!(baseline.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_ROPE_ROTATE,
                disposition: CompileDisposition::OneShot,
                count: 1,
            }
        )));

        set_compile_glue(false);
        RETAINED_ROPE_ROTATE.with(|slot| *slot.borrow_mut() = None);
    }
}
