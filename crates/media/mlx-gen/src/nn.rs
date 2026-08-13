//! Model-agnostic neural-net primitives — the shared `nn` layer of `mlx-gen` core.
//!
//! These are the genuinely family-independent leaf ops: dense linear, SiLU, NHWC `conv2d`,
//! pytorch-compatible `group_norm`, and nearest `upsample`. Model-specific block assemblies
//! (attention / RoPE / SwiGLU layouts) intentionally stay in their family crates — see
//! `docs/MODEL_ARCHITECTURE.md` §3.2 ("each family crate owns its blocks"). A primitive
//! graduates here only once it is provably model-agnostic; we do not lift a block to a shared
//! abstraction off a single model.

use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
};

use mlx_rs::error::Exception;
use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{
    add, addmm, broadcast_to, conv2d as conv2d_op, conv3d as conv3d_op, dequantize, divide, erf,
    multiply, pad, power, quantize, sigmoid, subtract, tanh,
};
use mlx_rs::transforms::compile::{
    compile, compile_retained, RetainedBinary as MlxRetainedBinary,
    RetainedSlice as MlxRetainedSlice, RetainedTernary as MlxRetainedTernary,
    RetainedUnary as MlxRetainedUnary,
};
use mlx_rs::{Array, Dtype};

use crate::array::scalar;
use crate::{Error, Result};

/// A token-embedding table (`[vocab, hidden]`), dense or group-quantized (Q4/Q8) — the shared
/// implementation behind the FLUX.2 and Z-Image text encoders' `embed_tokens` (F-083). `forward` is a
/// row-gather (dequantizing the packed form on the fly), mirroring mlx's `QuantizedEmbedding`, and
/// returns **f32** (both encoders run f32 internally). FLUX.1's CLIP/T5 embedding intentionally
/// returns native bf16, so it keeps its own variant rather than using this.
pub enum TokenEmbedding {
    Dense(Array),
    Quantized {
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl TokenEmbedding {
    /// Gather rows for `ids` (`[b, s]`) → `[b, s, hidden]`, cast to f32.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let out = match self {
            Self::Dense(w) => w.take_axis(ids, 0)?,
            Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let pw = wq.take_axis(ids, 0)?;
                let sc = scales.take_axis(ids, 0)?;
                let bi = biases.take_axis(ids, 0)?;
                dequantize(&pw, &sc, &bi, *group_size, *bits)?
            }
        };
        Ok(out.as_dtype(Dtype::Float32)?)
    }

    /// Build a quantized table directly from **already-packed** parts read off disk — the
    /// consume-side counterpart to [`quantize`](Self::quantize), for a *pre-quantized* checkpoint
    /// (the FLUX.2-dev path, sc-5917). No re-quantization happens; the on-disk `scales`/`biases`
    /// are used as-is. Mirrors [`AdaptableLinear::from_quantized_parts`] for the embedding case.
    ///
    /// [`AdaptableLinear::from_quantized_parts`]: crate::adapters::AdaptableLinear::from_quantized_parts
    pub fn from_quantized_parts(
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::Quantized {
            wq,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Quantize a dense table in place (group_size 64), the mlx-rs equivalent of
    /// `nn.Embedding.to_quantized`. No-op if already quantized. `cast_to_bf16` packs the **bf16**
    /// weight (FLUX.2's path, to byte-match the fork's bf16 `nn.quantize`); when false the weight is
    /// packed as-is (Z-Image's path). The two are equivalent for a bf16-native checkpoint — the flag
    /// preserves each caller's exact prior behaviour.
    pub fn quantize(&mut self, bits: i32, cast_to_bf16: bool) -> Result<()> {
        if let Self::Dense(w) = self {
            let (wq, scales, biases) = if cast_to_bf16 {
                quantize(&w.as_dtype(Dtype::Bfloat16)?, 64, bits)?
            } else {
                quantize(w, 64, bits)?
            };
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size: 64,
                bits,
            };
        }
        Ok(())
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Evaluate the retained embedding representation without evaluating any other model weight.
    /// Used by bounded load-time quantization walks to keep the dense-to-packed transient local to
    /// this table.
    pub fn materialize_weights(&self) -> Result<()> {
        match self {
            Self::Dense(weight) => mlx_rs::transforms::eval([weight])?,
            Self::Quantized {
                wq, scales, biases, ..
            } => mlx_rs::transforms::eval([wq, scales, biases])?,
        }
        Ok(())
    }
}

struct RetainedSignatures {
    cache_generation: u64,
    values: HashSet<Vec<(Dtype, usize)>>,
}

impl RetainedSignatures {
    fn new() -> Self {
        Self {
            cache_generation: mlx_rs::transforms::compile::cache_generation(),
            values: HashSet::new(),
        }
    }

    fn disposition(
        &mut self,
        signature: Vec<(Dtype, usize)>,
    ) -> crate::diagnostics::CompileDisposition {
        let cache_generation = mlx_rs::transforms::compile::cache_generation();
        if self.cache_generation != cache_generation {
            self.values.clear();
            self.cache_generation = cache_generation;
        }
        if self.values.insert(signature) {
            crate::diagnostics::CompileDisposition::RetainedMiss
        } else {
            crate::diagnostics::CompileDisposition::RetainedHit
        }
    }
}

fn record_retained(site: &'static str, disposition: crate::diagnostics::CompileDisposition) {
    crate::diagnostics::record_compile(site, disposition);
    crate::diagnostics::record_toggle(
        crate::diagnostics::RETAINED_COMPILATION,
        crate::diagnostics::ToggleDisposition::Applied,
    );
}

/// A retained unary `mx.compile` handle, shape-polymorphic within each dtype/rank signature.
#[doc(hidden)]
pub struct RetainedUnary {
    compiled: Box<dyn MlxRetainedUnary>,
    signatures: RetainedSignatures,
}

impl RetainedUnary {
    #[doc(hidden)]
    pub fn new(compiled: Box<dyn MlxRetainedUnary>) -> Self {
        Self {
            compiled,
            signatures: RetainedSignatures::new(),
        }
    }

    #[doc(hidden)]
    pub fn call(&mut self, site: &'static str, x: &Array) -> std::result::Result<Array, Exception> {
        let signature = vec![(x.dtype(), x.ndim())];
        let out = self.compiled.call_mut(x)?;
        record_retained(site, self.signatures.disposition(signature));
        Ok(out)
    }
}

/// A retained two-input `mx.compile` handle, shape-polymorphic within each dtype/rank signature.
#[doc(hidden)]
pub struct RetainedBinary {
    compiled: Box<dyn MlxRetainedBinary>,
    signatures: RetainedSignatures,
}

impl RetainedBinary {
    #[doc(hidden)]
    pub fn new(compiled: Box<dyn MlxRetainedBinary>) -> Self {
        Self {
            compiled,
            signatures: RetainedSignatures::new(),
        }
    }

    #[doc(hidden)]
    pub fn call(
        &mut self,
        site: &'static str,
        args: (&Array, &Array),
    ) -> std::result::Result<Array, Exception> {
        let signature = vec![
            (args.0.dtype(), args.0.ndim()),
            (args.1.dtype(), args.1.ndim()),
        ];
        let out = self.compiled.call_mut(args)?;
        record_retained(site, self.signatures.disposition(signature));
        Ok(out)
    }
}

/// A retained three-input `mx.compile` handle, shape-polymorphic within each dtype/rank signature.
#[doc(hidden)]
pub struct RetainedTernary {
    compiled: Box<dyn MlxRetainedTernary>,
    signatures: RetainedSignatures,
}

impl RetainedTernary {
    #[doc(hidden)]
    pub fn new(compiled: Box<dyn MlxRetainedTernary>) -> Self {
        Self {
            compiled,
            signatures: RetainedSignatures::new(),
        }
    }

    #[doc(hidden)]
    pub fn call(
        &mut self,
        site: &'static str,
        args: (&Array, &Array, &Array),
    ) -> std::result::Result<Array, Exception> {
        let signature = vec![
            (args.0.dtype(), args.0.ndim()),
            (args.1.dtype(), args.1.ndim()),
            (args.2.dtype(), args.2.ndim()),
        ];
        let out = self.compiled.call_mut(args)?;
        record_retained(site, self.signatures.disposition(signature));
        Ok(out)
    }
}

/// A retained variadic `mx.compile` handle, shape-polymorphic within each dtype/rank signature.
#[doc(hidden)]
pub struct RetainedSlice {
    compiled: Box<dyn MlxRetainedSlice>,
    signatures: RetainedSignatures,
}

impl RetainedSlice {
    #[doc(hidden)]
    pub fn new(compiled: Box<dyn MlxRetainedSlice>) -> Self {
        Self {
            compiled,
            signatures: RetainedSignatures::new(),
        }
    }

    #[doc(hidden)]
    pub fn call(
        &mut self,
        site: &'static str,
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        let signature = args.iter().map(|arg| (arg.dtype(), arg.ndim())).collect();
        let out = self.compiled.call_mut(args)?;
        record_retained(site, self.signatures.disposition(signature));
        Ok(out)
    }
}

/// Whether the active diagnostic request selected retained compilation.
#[doc(hidden)]
pub fn retained_compilation_requested() -> bool {
    crate::diagnostics::toggle_requested(crate::diagnostics::RETAINED_COMPILATION)
}

/// Prime MLX's compiler-cache TLS and its native lifetime token before accessing Rust TLS that owns
/// retained handles. This makes main-thread and worker-thread teardown ordering safe.
#[doc(hidden)]
pub fn prepare_retained_compilation_thread() {
    mlx_rs::transforms::compile::prepare_retained_compilation_thread();
}

const TRACE_GATED: usize = 0;
const TRACE_ROPE_ROTATE: usize = 1;
const TRACE_MODULATE_MATCHING_ONE: usize = 2;
const TRACE_MODULATE_F32_ONE: usize = 3;
const TRACE_GELU_EXACT: usize = 4;
const TRACE_GELU_QUICK: usize = 5;

const SITE_GATED: &str = "mlx_gen::nn::gated";
const SITE_ROPE_ROTATE: &str = "mlx_gen::nn::rope_rotate";
const SITE_MODULATE: &str = "mlx_gen::nn::modulate";
const SITE_GELU_EXACT: &str = "mlx_gen::nn::gelu_exact";
const SITE_GELU_QUICK: &str = "mlx_gen::nn::gelu_quick";

#[cfg(test)]
static COMPILE_TRACE_COUNTS: [std::sync::atomic::AtomicUsize; 6] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 6];

#[inline(always)]
fn note_compile_trace(index: usize) {
    #[cfg(test)]
    COMPILE_TRACE_COUNTS[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    #[cfg(not(test))]
    let _ = index;
}

fn gated_impl((x, gate, y): (&Array, &Array, &Array)) -> std::result::Result<Array, Exception> {
    add(x, &multiply(gate, y)?)
}

fn compiled_gated(args: (&Array, &Array, &Array)) -> std::result::Result<Array, Exception> {
    note_compile_trace(TRACE_GATED);
    gated_impl(args)
}

fn rope_rotate_impl(inp: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    let (real, imag, cos, sin) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    let out_real = subtract(&multiply(real, cos)?, &multiply(imag, sin)?)?;
    let out_imag = add(&multiply(imag, cos)?, &multiply(real, sin)?)?;
    Ok(vec![out_real, out_imag])
}

fn compiled_rope_rotate(inp: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
    note_compile_trace(TRACE_ROPE_ROTATE);
    rope_rotate_impl(inp)
}

fn modulate_matching_one_impl(
    (norm, scale, shift): (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    let one = scalar(1.0).as_dtype(scale.dtype())?;
    add(&multiply(norm, &add(scale, &one)?)?, shift)
}

fn compiled_modulate_matching_one(
    args: (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    note_compile_trace(TRACE_MODULATE_MATCHING_ONE);
    modulate_matching_one_impl(args)
}

fn modulate_f32_one_impl(
    (norm, scale, shift): (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    add(&multiply(norm, &add(scale, scalar(1.0))?)?, shift)
}

fn compiled_modulate_f32_one(
    args: (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    note_compile_trace(TRACE_MODULATE_F32_ONE);
    modulate_f32_one_impl(args)
}

fn gelu_exact_impl(x: &Array) -> std::result::Result<Array, Exception> {
    let dt = x.dtype();
    let inv_sqrt2 = scalar(std::f64::consts::SQRT_2 as f32).as_dtype(dt)?;
    let one = scalar(1.0).as_dtype(dt)?;
    let two = scalar(2.0).as_dtype(dt)?;
    let inner = erf(&divide(x, &inv_sqrt2)?)?;
    let gate = add(&one, &inner)?;
    divide(&multiply(x, &gate)?, &two)
}

fn compiled_gelu_exact(x: &Array) -> std::result::Result<Array, Exception> {
    note_compile_trace(TRACE_GELU_EXACT);
    gelu_exact_impl(x)
}

fn gelu_quick_impl(x: &Array) -> std::result::Result<Array, Exception> {
    let c = scalar(1.702).as_dtype(x.dtype())?;
    multiply(x, &sigmoid(&multiply(x, &c)?)?)
}

fn compiled_gelu_quick(x: &Array) -> std::result::Result<Array, Exception> {
    note_compile_trace(TRACE_GELU_QUICK);
    gelu_quick_impl(x)
}

/// The six retained `mx.compile` transforms owned by the shared `nn` layer. The graph functions
/// capture no arrays, so this state retains compiler metadata/kernels but cannot pin request inputs
/// or outputs.
struct RetainedCompileGlue {
    gated: RetainedTernary,
    rope_rotate: RetainedSlice,
    modulate_matching_one: RetainedTernary,
    modulate_f32_one: RetainedTernary,
    gelu_exact: RetainedUnary,
    gelu_quick: RetainedUnary,
}

impl RetainedCompileGlue {
    fn new() -> Self {
        Self {
            gated: RetainedTernary::new(compile_retained(compiled_gated, true)),
            rope_rotate: RetainedSlice::new(compile_retained(compiled_rope_rotate, true)),
            // Keep the two dtype policies as distinct graph functions so diagnostics and parity
            // failures identify the exact literal-`1` policy even though retained handles now own
            // independent backend identities.
            modulate_matching_one: RetainedTernary::new(compile_retained(
                compiled_modulate_matching_one,
                true,
            )),
            modulate_f32_one: RetainedTernary::new(compile_retained(
                compiled_modulate_f32_one,
                true,
            )),
            gelu_exact: RetainedUnary::new(compile_retained(compiled_gelu_exact, true)),
            gelu_quick: RetainedUnary::new(compile_retained(compiled_gelu_quick, true)),
        }
    }
}

thread_local! {
    /// Request/thread-local enablement prevents one concurrent render from changing another's
    /// numeric path. The production guards wrap synchronous denoise loops, so their request scope
    /// does not migrate between threads.
    static COMPILE_GLUE: Cell<bool> = const { Cell::new(false) };

    /// Retained until the owning worker thread exits. This is the narrowest lifecycle available to
    /// shared leaf helpers without moving model-specific state into core or serializing all renders
    /// behind one process-global mutex.
    static RETAINED_COMPILE_GLUE: RefCell<Option<RetainedCompileGlue>> =
        const { RefCell::new(None) };
}

fn with_retained_compile_glue<T>(
    f: impl FnOnce(&mut RetainedCompileGlue) -> std::result::Result<T, Exception>,
) -> std::result::Result<T, Exception> {
    prepare_retained_compilation_thread();
    RETAINED_COMPILE_GLUE.with(|slot| {
        let mut slot = slot.borrow_mut();
        f(slot.get_or_insert_with(RetainedCompileGlue::new))
    })
}

/// Exercise every retained handle owned by the shared `nn` layer once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &Array) -> Result<()> {
    let output = with_retained_compile_glue(|compiled| {
        compiled.gated.call(SITE_GATED, (input, input, input))
    })?;
    output.eval()?;
    drop(output);

    let args = [input.clone(), input.clone(), input.clone(), input.clone()];
    let outputs =
        with_retained_compile_glue(|compiled| compiled.rope_rotate.call(SITE_ROPE_ROTATE, &args))?;
    mlx_rs::transforms::eval(outputs.iter())?;
    drop(outputs);
    drop(args);

    for one_matches_scale in [true, false] {
        let output = with_retained_compile_glue(|compiled| {
            let handle = if one_matches_scale {
                &mut compiled.modulate_matching_one
            } else {
                &mut compiled.modulate_f32_one
            };
            handle.call(SITE_MODULATE, (input, input, input))
        })?;
        output.eval()?;
        drop(output);
    }

    let output =
        with_retained_compile_glue(|compiled| compiled.gelu_exact.call(SITE_GELU_EXACT, input))?;
    output.eval()?;
    drop(output);

    let output =
        with_retained_compile_glue(|compiled| compiled.gelu_quick.call(SITE_GELU_QUICK, input))?;
    output.eval()?;
    drop(output);
    Ok(())
}

/// Enable/disable the compiled elementwise glue (sc-2963).
///
/// **Request/thread scope (sc-18316):** this changes the numeric path only for work on the current
/// thread. The production render loops that use this API are synchronous, so a guard cannot migrate
/// between worker threads and concurrent renders no longer race through one process-global atomic.
/// In production prefer the RAII [`CompileGlueGuard`] over a bare
/// `set_compile_glue(true)`: the guard restores the prior value on drop (even on an early `?`), so the
/// toggle is scoped to the render instead of leaking into later work on the same thread. The raw
/// setter is for the `compile_parity` / `perf` A/B gates.
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.set(on);
}

/// Whether the compiled elementwise glue is currently enabled.
pub fn compile_glue() -> bool {
    COMPILE_GLUE.get()
}

/// RAII guard that enables the compiled elementwise glue for its lifetime and **restores the current
/// thread's prior `COMPILE_GLUE` value on drop** — even on an early `?` return. This is the one shared
/// guard the per-family transformers were each hand-rolling (F-007); bind one at the top of a denoise
/// loop (`let _g = CompileGlueGuard::enable();`) so the setting is scoped to that render and code that
/// runs afterward — notably the reference-parity gates, which must run eager — sees the prior value.
#[must_use = "dropping the guard restores the prior compile-glue setting; bind it for the render's lifetime"]
pub struct CompileGlueGuard {
    prev: bool,
}

impl CompileGlueGuard {
    /// Turn compiled glue on, remembering the prior value to restore on drop.
    pub fn enable() -> Self {
        Self {
            prev: COMPILE_GLUE.replace(true),
        }
    }
}

impl Drop for CompileGlueGuard {
    fn drop(&mut self) {
        COMPILE_GLUE.set(self.prev);
    }
}

/// Shared f32 tolerance for the compiled-vs-eager elementwise-glue fast paths — the fused tanh-GELU
/// FFN (over [`gelu_tanh`]), [`silu`], and the family-crate SwiGLU/GELU-FFN wrappers that defer to
/// them.
///
/// These fused kernels are **bit-identical** to their eager core op at fp16/bf16, but under MLX
/// 0.32.0 the *f32* fusion rounds 1–2 ULP differently from the eager op (epic 12742 pin bump —
/// sc-12744 measured it, sc-12747 re-baselined it; the prior 0.31.2 pin was 0-ULP). Expressed in f32
/// ULPs (`f32::EPSILON` = 2⁻²³ ≈ 1.19e-7) and applied **relative** to the reference peak magnitude
/// (see [`max_rel_diff`]): the observed drift is ≤1 ULP relative (abs `max|Δ|` 2.4e-7…4.8e-7 on the
/// fixed-seed compiled-glue parity tests), so 4 ULP accepts the harmless rounding with margin while
/// staying ~1000× under every golden tolerance. A real regression (wrong fusion / scale / transpose)
/// is O(1e-1) — orders of magnitude above this floor, so it is still caught. fp16/bf16 stay asserted
/// **exact** (`0.0`) by their guards; only the f32 path uses this tolerance.
pub const COMPILED_GLUE_F32_ULP_TOL: f32 = 4.0 * f32::EPSILON;

/// Peak **relative** divergence `max|a−b| / max(max|b|, 1e-12)` between two arrays (cast to f32),
/// as f32 — `0.0` iff bit-identical. Companion to [`COMPILED_GLUE_F32_ULP_TOL`]: the compiled-glue
/// parity guards compare this against that tolerance at f32 and against `0.0` (exact) at fp16/bf16.
pub fn max_rel_diff(a: &Array, b: &Array) -> f32 {
    use mlx_rs::ops::{abs, max};
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let num = max(abs(subtract(&a, &b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let den = max(abs(&b).unwrap(), None).unwrap().item::<f32>();
    num / den.max(1e-12)
}

/// Gated residual `x + gate·y` (`gate` pre-broadcast). One fused kernel when [`compile_glue`] is on;
/// bit-identical to the eager `add(x, gate·y)`. Shared by the FLUX.1/FLUX.2 MMDiTs (F-101).
pub fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    if compile_glue() {
        if retained_compilation_requested() {
            Ok(with_retained_compile_glue(|compiled| {
                compiled.gated.call(SITE_GATED, (x, gate, y))
            })?)
        } else {
            crate::diagnostics::record_compile(
                SITE_GATED,
                crate::diagnostics::CompileDisposition::OneShot,
            );
            Ok(compile(compiled_gated, true)((x, gate, y))?)
        }
    } else {
        crate::diagnostics::record_fallback(SITE_GATED, "compiled_glue_disabled");
        Ok(gated_impl((x, gate, y))?)
    }
}

/// The complex RoPE rotation `(real + imag·i)·(cos + sin·i)` → `(out_real, out_imag)`, in f32. One
/// fused kernel when [`compile_glue`] is on (vs 6 eager ops). Shared by FLUX.1/FLUX.2 (F-101).
pub fn rope_rotate(real: &Array, imag: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let args = [real.clone(), imag.clone(), cos.clone(), sin.clone()];
    let mut out = if compile_glue() {
        if retained_compilation_requested() {
            with_retained_compile_glue(|compiled| {
                compiled.rope_rotate.call(SITE_ROPE_ROTATE, &args)
            })?
        } else {
            crate::diagnostics::record_compile(
                SITE_ROPE_ROTATE,
                crate::diagnostics::CompileDisposition::OneShot,
            );
            compile(compiled_rope_rotate, true)(&args)?
        }
    } else {
        crate::diagnostics::record_fallback(SITE_ROPE_ROTATE, "compiled_glue_disabled");
        rope_rotate_impl(&args)?
    };
    let out1 = out.pop().unwrap();
    let out0 = out.pop().unwrap();
    Ok((out0, out1))
}

/// Per-axis RoPE sinusoid table from position ids `[N, n_axes]` (diffusers `FluxPosEmbed` /
/// `EmbedND`): for each axis `i` with `dim = axes_dim[i]`, `omega = theta^-(2k/dim)`,
/// `out = ids[:,i]·omega`, then `(cos, sin)`; concatenated across axes on the last dim →
/// `([N, Σ dim/2], [N, Σ dim/2])`. Shared by FLUX.1 + Chroma (F-015) — both previously open-coded the
/// **identical** MLX op sequence, so this is bit-exact for both (the host `libm` trig the comment in
/// each caller warned against is avoided; all trig stays on-device).
pub fn rope_sincos_from_ids(ids: &Array, axes_dim: &[i32], theta: f32) -> Result<(Array, Array)> {
    let ids = ids.as_dtype(Dtype::Float32)?;
    let n = ids.shape()[0];
    let mut coss = Vec::with_capacity(axes_dim.len());
    let mut sins = Vec::with_capacity(axes_dim.len());
    for (a, &dim) in axes_dim.iter().enumerate() {
        let half = dim / 2;
        let pos = ids
            .take_axis(Array::from_int(a as i32), 1)?
            .reshape(&[n, 1])?; // [N,1]
        let scale: Vec<f32> = (0..half).map(|k| (2 * k) as f32 / dim as f32).collect();
        let omega = divide(
            scalar(1.0),
            &power(scalar(theta), Array::from_slice(&scale, &[1, half]))?,
        )?; // [1, half]
        let out = multiply(&pos, &omega)?; // [N, half]
        coss.push(out.cos()?);
        sins.push(out.sin()?);
    }
    let cref: Vec<&Array> = coss.iter().collect();
    let sref: Vec<&Array> = sins.iter().collect();
    Ok((
        mlx_rs::ops::concatenate_axis(&cref, 1)?,
        mlx_rs::ops::concatenate_axis(&sref, 1)?,
    ))
}

/// diffusers `get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift,
/// max_period)` in f32, returning `[N, dim]` ordered `[cos, sin]` (flip). The exponent is built with
/// **MLX ops** — `exp((arange·−ln(max_period)) / (half − downscale_freq_shift))` — deliberately *not*
/// a host `libm` loop: FLUX.1's `_time_proj` parity hinges on this on-device computation (the host
/// path differs by ~1e-7, which flips a bf16 conditioning ULP that the 57-block joint stack amplifies,
/// F-016). Shared by FLUX.1 (`downscale_freq_shift = 0`) + Chroma; bit-exact for FLUX.1's prior
/// open-coded form and within Chroma's parity tolerance for its prior host-f64 form.
pub fn timestep_sincos(
    timesteps: &Array,
    dim: usize,
    max_period: f64,
    downscale_freq_shift: f64,
) -> Result<Array> {
    let half = (dim / 2) as i32;
    let neg_log = -(max_period.ln()) as f32;
    let denom = (half as f64 - downscale_freq_shift) as f32;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let exponent = divide(
        multiply(Array::from_slice(&arange, &[1, half]), scalar(neg_log))?,
        scalar(denom),
    )?;
    let f = exponent.exp()?;
    let t = timesteps
        .reshape(&[timesteps.shape()[0], 1])?
        .as_dtype(Dtype::Float32)?;
    let emb = multiply(&t, &f)?;
    Ok(mlx_rs::ops::concatenate_axis(
        &[&emb.cos()?, &emb.sin()?],
        1,
    )?) // flip_sin_to_cos ⇒ [cos, sin]
}

/// adaLN affine `(1 + scale)·norm + shift` (`scale`/`shift` pre-broadcast). One fused kernel when
/// [`compile_glue`] is on; bit-identical to the eager affine. Shared by FLUX.1/FLUX.2 (F-101) with
/// **`one_matches_scale`** carrying each family's deliberate dtype policy for the literal `1`: `true`
/// casts it to `scale`'s dtype so `1+scale` rounds in bf16 (FLUX.1 — the fork's weak python `1`,
/// sc-2787); `false` keeps a strong f32 `1` (FLUX.2), promoting the sum to f32.
pub fn modulate(
    norm: &Array,
    scale: &Array,
    shift: &Array,
    one_matches_scale: bool,
) -> Result<Array> {
    if compile_glue() {
        if retained_compilation_requested() {
            if one_matches_scale {
                Ok(with_retained_compile_glue(|compiled| {
                    compiled
                        .modulate_matching_one
                        .call(SITE_MODULATE, (norm, scale, shift))
                })?)
            } else {
                Ok(with_retained_compile_glue(|compiled| {
                    compiled
                        .modulate_f32_one
                        .call(SITE_MODULATE, (norm, scale, shift))
                })?)
            }
        } else {
            crate::diagnostics::record_compile(
                SITE_MODULATE,
                crate::diagnostics::CompileDisposition::OneShot,
            );
            if one_matches_scale {
                Ok(compile(compiled_modulate_matching_one, true)((
                    norm, scale, shift,
                ))?)
            } else {
                Ok(compile(compiled_modulate_f32_one, true)((
                    norm, scale, shift,
                ))?)
            }
        }
    } else {
        crate::diagnostics::record_fallback(SITE_MODULATE, "compiled_glue_disabled");
        if one_matches_scale {
            Ok(modulate_matching_one_impl((norm, scale, shift))?)
        } else {
            Ok(modulate_f32_one_impl((norm, scale, shift))?)
        }
    }
}

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight + bias (PyTorch `nn.Linear` convention).
///
/// FUSED `addmm(bias, x, Wᵀ)` — matching MLX's own `nn.Linear` (and the core `AdaptableLinear`
/// dense path, sc-2779). A separate `matmul` then `add` double-rounds the bias add in bf16
/// (~1.4e-3/Linear, compounding over a deep net); `addmm` rounds once. f32-invisible (`addmm ==
/// matmul+add` bit-for-bit with f32 activations, even over bf16 weights), so this is safe for the
/// current f32-activation text-encoder consumers (LTX, Qwen) and correct once they go bf16.
pub fn linear(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    Ok(addmm(b, x, w.t(), 1.0, 1.0)?)
}

/// SiLU / swish activation: `x · sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// GELU tanh approximation — `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` — **dtype-preserving**
/// and **bit-exact** to MLX-Python's `nn.GELU(approx="tanh")` / `mx.nn.gelu_approx`.
///
/// Two traps this avoids, both latent in f32 and surfacing only on a bf16 FFN path (sc-2779):
///   1. **Scalar dtype.** MLX weak-casts its python-float constants to the *input* dtype, so a
///      bf16 input computes in bf16. `mlx_rs::nn::gelu_approximate` (and any `scalar(..)`-based
///      hand-roll) uses f32 constants, which promote a bf16 input to f32 — a ~2.4e-3 mismatch
///      vs the golden. We cast every constant to `x.dtype()` so bf16 stays bf16 and f32 stays f32.
///   2. **The `√(2/π)` constant.** `mlx_rs::nn::gelu_approximate` builds it with an f32 MLX `sqrt`
///      op (1 ULP off MLX-Python's f64-host `math.sqrt`); over a deep f32 path that 1-ULP seed
///      amplifies (Wan UMT5: ~5e-7 → ~1e-3 over 24 layers — see [[mlx-rs-gelu-approx-f64-constant]]).
///      We compute it in f64 on the host, then cast.
///
/// Result: the f32 path is bit-exact to the reference and a bf16 input is preserved in bf16.
/// Shared across the family crates' tanh-approx FFNs (Wan today; Qwen/FLUX/FLUX.2/Z-Image as their
/// forwards move f32→bf16, sc-2718–2721). `x³` via integer-exponent `power`, as MLX does.
pub fn gelu_tanh(x: &Array) -> Result<Array> {
    let dt = x.dtype();
    let s = |v: f32| -> Result<Array> { Ok(scalar(v).as_dtype(dt)?) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    Ok(multiply(&multiply(x, &s(0.5)?)?, &gate)?)
}

/// Exact GELU — `x · (1 + erf(x/√2)) / 2` — **dtype-preserving** and **byte-identical to MLX-Python's
/// `nn.gelu`** (the activation in the SDXL GEGLU FFN — `y_a · gelu(y_b)`).
///
/// Two things are required to match the reference bit-for-bit, both fp16-only traps (f32-invisible):
///   1. **Dtype-weak constants.** Each python-float constant is cast to `x.dtype()` so an f16 input
///      stays f16. `mlx_rs::nn::gelu` builds `1/√2` / `2` as f32 *arrays* (`array!(2f32.sqrt())`),
///      which promote an f16 input to f32 — the silent f16→f32 leak that made the SDXL fp16 U-Net run
///      (and return) f32 (sc-2721). `√2` is taken in f64 (python `math.sqrt(2)`) before the cast.
///   2. **`mx.compile`.** MLX-Python decorates `nn.gelu` with `@mx.compile`; the fused kernel rounds
///      fp16 differently from the same ops run unfused (a 1-ULP/elem gap — measured as the *sole*
///      SDXL fp16 U-Net divergence: replacing the reference's compiled gelu with an unfused literal
///      reproduces the 57.7%-byte-exact / 5.3e-4 gap exactly). Compiling the identical graph here
///      reproduces the fused rounding → bit-exact. f32-safe: an f32 input is unchanged either way.
pub fn gelu_exact(x: &Array) -> Result<Array> {
    if retained_compilation_requested() {
        Ok(with_retained_compile_glue(|compiled| {
            compiled.gelu_exact.call(SITE_GELU_EXACT, x)
        })?)
    } else {
        crate::diagnostics::record_compile(
            SITE_GELU_EXACT,
            crate::diagnostics::CompileDisposition::OneShot,
        );
        Ok(compile(compiled_gelu_exact, true)(x)?)
    }
}

/// Fast-approx GELU ("quick_gelu") — `x · sigmoid(1.702·x)` — **byte-identical to MLX-Python's
/// `nn.gelu_fast_approx`** (the CLIP-L / `quick_gelu` activation). Same two requirements as
/// [`gelu_exact`]: the `1.702` constant is weak-cast to `x.dtype()` (an f32 scalar would promote an
/// f16 input to f32), and the graph is `mx.compile`'d so the fused fp16 rounding matches the
/// reference. f32-safe (no-op cast, fusion numerically identical).
pub fn gelu_quick(x: &Array) -> Result<Array> {
    if retained_compilation_requested() {
        Ok(with_retained_compile_glue(|compiled| {
            compiled.gelu_quick.call(SITE_GELU_QUICK, x)
        })?)
    } else {
        crate::diagnostics::record_compile(
            SITE_GELU_QUICK,
            crate::diagnostics::CompileDisposition::OneShot,
        );
        Ok(compile(compiled_gelu_quick, true)(x)?)
    }
}

/// 2-D conv over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias).
pub fn conv2d(x: &Array, w: &Array, b: Option<&Array>, stride: i32, padding: i32) -> Result<Array> {
    let mut y = conv2d_op(x, w, (stride, stride), (padding, padding), (1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// 3-D conv over NDHWC `x` with an mlx `[out, kD, kH, kW, in]` weight (+ optional bias).
/// `stride`/`padding` are per-axis `(depth, height, width)`. Qwen's causal-Conv3d VAE applies
/// its asymmetric temporal padding manually and calls this with `padding (0, 0, 0)`; future
/// video families (Wan2.2 / LTX) reuse it directly — hence it lives in shared core `nn`.
pub fn conv3d(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: (i32, i32, i32),
    padding: (i32, i32, i32),
) -> Result<Array> {
    let mut y = conv3d_op(x, w, stride, padding, (1, 1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// PyTorch-compatible group normalization over NHWC `x` (`weight`/`bias` are per-channel).
/// Mirrors mlx-rs `GroupNorm::pytorch_group_norm` + affine: split channels into `num_groups`,
/// layer-norm each group, then scale/shift by `weight`/`bias`.
pub fn group_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    num_groups: i32,
    eps: f32,
) -> Result<Array> {
    let sh = x.shape();
    // Shared pub primitive: all current callers satisfy the rank/group contract, but a malformed
    // caller would panic on the `sh[1..sh.len()-1]` slice (rank<2) or divide-by-zero / mis-group on
    // `dims / num_groups`. Reject those with a typed error instead (F-041).
    if sh.len() < 2 {
        return Err(Error::Msg(format!(
            "group_norm: expected a rank>=2 tensor, got shape {sh:?}"
        )));
    }
    let batch = sh[0];
    let dims = sh[sh.len() - 1];
    let rest = &sh[1..sh.len() - 1];
    if num_groups <= 0 || dims % num_groups != 0 {
        return Err(Error::Msg(format!(
            "group_norm: channel count {dims} not divisible by num_groups {num_groups}"
        )));
    }
    let group_size = dims / num_groups;

    let g = x
        .reshape(&[batch, -1, num_groups, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, num_groups, -1])?;
    let g = layer_norm(&g, None, None, eps)?;
    let g = g
        .reshape(&[batch, num_groups, -1, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let mut shape = vec![batch];
    shape.extend_from_slice(rest);
    shape.push(dims);
    let normed = g.reshape(&shape)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Nearest-neighbor upsample of NHWC `x` by `scale` (broadcast + reshape).
pub fn upsample_nearest(x: &Array, scale: i32) -> Result<Array> {
    let sh = x.shape();
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "upsample_nearest: expected an NHWC (rank 4) tensor, got shape {sh:?}"
        )));
    }
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x6 = x.reshape(&[b, h, 1, w, 1, c])?;
    let bc = broadcast_to(&x6, &[b, h, scale, w, scale, c])?;
    Ok(bc.reshape(&[b, h * scale, w * scale, c])?)
}

/// Rotary position embedding for **text encoders** — the HF "half-split" convention (distinct from
/// the DiT's interleaved RoPE, which stays family-owned). Port of the fork's `RotaryEmbedding`:
/// `inv_freq = 1/θ^(arange(0,dim,2)/dim)`; `freqs = outer(arange(seq), inv_freq)`;
/// `emb = concat([freqs, freqs])`; `cos/sin = cos/sin(emb)[None]`. Shared by the Z-Image and
/// Qwen-Image text encoders, which use the identical layout (the second-family trigger for lifting
/// this to core — F-006).
pub struct TextRope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl TextRope {
    /// `dim` = head_dim, `theta` = rope base (1e6 for both Z-Image and Qwen-Image).
    pub fn new(dim: i32, theta: f32) -> Self {
        let half = (dim / 2) as usize;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq, dim }
    }

    /// Returns `(cos, sin)`, each `[1, seq_len, dim]`, for positions `0..seq_len`.
    pub fn forward(&self, seq_len: i32) -> Result<(Array, Array)> {
        self.forward_offset(seq_len, 0)
    }

    /// Returns `(cos, sin)`, each `[1, q_len, dim]`, for the absolute positions
    /// `offset..offset+q_len`. The autoregressive-decode companion to [`forward`](Self::forward)
    /// (`offset == 0` is exactly `forward`): a KV-cached generate step embeds `q_len` fresh queries
    /// sitting at positions `offset..` over the cached keys, so it needs the RoPE table at those
    /// absolute positions, not `0..q_len`. Mirrors the per-step `Llama3Rope::forward(seq_len, offset)`
    /// the caption decoders use.
    pub fn forward_offset(&self, q_len: i32, offset: i32) -> Result<(Array, Array)> {
        let half = self.inv_freq.len();
        // freqs[s, j] = (offset + s) * inv_freq[j]  → [q_len, half]
        let mut freqs = Vec::with_capacity(q_len as usize * half);
        for s in 0..q_len {
            let pos = (offset + s) as f32;
            for &f in &self.inv_freq {
                freqs.push(pos * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[q_len, half as i32]);
        // emb = concat([freqs, freqs], -1) → [q_len, dim]
        let emb = mlx_rs::ops::concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos = mlx_rs::ops::cos(&emb)?.reshape(&[1, q_len, self.dim])?;
        let sin = mlx_rs::ops::sin(&emb)?.reshape(&[1, q_len, self.dim])?;
        Ok((cos, sin))
    }
}

/// HF **half-split** RoPE apply: `x·cos + rotate_half(x)·sin`, where `rotate_half([x1, x2]) =
/// [-x2, x1]` splits `x` on the head dim. `x` is `[b, s, heads, head_dim]`; `cos`/`sin` are
/// `[1, s, head_dim]` (from [`TextRope::forward`]) and broadcast over the head axis (2). This is the
/// GQA text-encoder RoPE convention (distinct from the DiT's interleaved rotation); shared by the
/// Qwen-Image and Krea Qwen3-VL text-encoder attentions, which open-coded the identical op sequence
/// (F-078).
pub fn apply_text_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?; // [1,s,1,hd]
    let sin = sin.expand_dims(2)?;
    let parts = mlx_rs::ops::split(x, 2, 3)?; // halves along the head dim
    let rot = mlx_rs::ops::concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
}

/// Grouped-query-attention key/value expansion: `[b, s, hkv, hd]` → `[b, s, hkv·groups, hd]`,
/// repeating each kv head `groups` times consecutively (`repeat_interleave` over the head axis,
/// matching `mx.repeat(x, groups, axis=2)` / torch `enable_gqa`). `groups == 1` is the identity.
/// Shared by the Qwen-Image / Krea text-encoder attentions and the Krea DiT (F-078).
pub fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?; // [b,s,hkv,1,hd]
    let x = broadcast_to(&x, &[b, s, hkv, groups, hd])?;
    Ok(x.reshape(&[b, s, hkv * groups, hd])?)
}

/// Additive attention mask `[b, 1, s, s]` for a **causal** LM: `0` where a query may attend (the key
/// is at or before the query position **and** is not padding per `attention_mask` `[b, s]` of 0/1),
/// `-inf` otherwise. Built on the host from the i32 attention mask, mirroring the fork's
/// `causal & key_padding` mask assembly. Shared by the Qwen3-VL text encoders (boogu/ideogram).
pub fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = crate::array::host_i32(attention_mask)?;
    let (b, s) = (b as usize, s as usize);
    // F-061: `am[bi*s + j]` indexes this caller-supplied slice against the independent `b`/`s` params;
    // a mask shorter than `b*s` (a provider bug) would otherwise panic (process abort) in shared core.
    if am.len() != b * s {
        return Err(Error::Msg(format!(
            "build_mask: attention_mask has {} elements, expected b*s = {}*{} = {}",
            am.len(),
            b,
            s,
            b * s
        )));
    }
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}

/// Partition NHWC `x` into `window`×`window` windows along H/W, zero-padding to a multiple of
/// `window`. Returns the `[-1, window, window, c]` windows plus the padded `(hp, wp)`; the inverse is
/// [`window_unpartition`]. Shared by the SAM2/SAM3 vision trunks' windowed-attention blocks (6940).
pub fn window_partition(x: &Array, window: i32) -> Result<(Array, (i32, i32))> {
    let sh = x.shape();
    // F-062: mirror the F-041 sibling guards (`upsample_nearest`/`group_norm`). A non-NHWC input
    // panics on `sh[1..4]`; `window <= 0` divides by zero in the pad/reshape math below.
    if sh.len() != 4 {
        return Err(Error::Msg(format!(
            "window_partition: expected an NHWC (rank 4) tensor, got shape {sh:?}"
        )));
    }
    if window <= 0 {
        return Err(Error::Msg(format!(
            "window_partition: window must be > 0, got {window}"
        )));
    }
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let x = if pad_h > 0 || pad_w > 0 {
        pad(x, &[(0, 0), (0, pad_h), (0, pad_w), (0, 0)][..], None, None)?
    } else {
        x.clone()
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let windows = x
        .reshape(&[b, hp / window, window, wp / window, window, c])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[-1, window, window, c])?;
    Ok((windows, (hp, wp)))
}

/// Inverse of [`window_partition`]: stitch the `[-1, window, window, c]` windows back to
/// `[b, h, w, c]`, cropping the window padding `(hp, wp)` back to the unpadded `hw = (h, w)`.
pub fn window_unpartition(
    windows: &Array,
    window: i32,
    pad_hw: (i32, i32),
    hw: (i32, i32),
) -> Result<Array> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    // F-062: `window <= 0` divides by zero below, and a degenerate `pad_hw` (`hp*wp < window²`) makes
    // `num_per_image == 0` → a second divide-by-zero on `windows.shape()[0] / num_per_image`.
    if window <= 0 {
        return Err(Error::Msg(format!(
            "window_unpartition: window must be > 0, got {window}"
        )));
    }
    let num_per_image = (hp * wp) / window / window;
    if num_per_image <= 0 {
        return Err(Error::Msg(format!(
            "window_unpartition: degenerate pad_hw {pad_hw:?} for window {window} (no whole windows)"
        )));
    }
    let b = windows.shape()[0] / num_per_image;
    let x = windows
        .reshape(&[b, hp / window, wp / window, window, window, -1])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hp, wp, -1])?;
    if hp > h || wp > w {
        let rows = Array::from_slice(&(0..h).collect::<Vec<i32>>(), &[h]);
        let cols = Array::from_slice(&(0..w).collect::<Vec<i32>>(), &[w]);
        Ok(x.take_axis(&rows, 1)?.take_axis(&cols, 2)?)
    } else {
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, array_eq, max, subtract};
    use mlx_rs::Dtype;

    #[test]
    fn text_rope_offset_zero_equals_forward_and_shifts_positions() {
        // sc-6030: `forward_offset(q_len, offset)` is the autoregressive-decode companion. At
        // offset 0 it must reproduce `forward(seq_len)` exactly, and the row at (offset, s) must
        // equal `forward`'s row at absolute position `offset + s`.
        let rope = TextRope::new(8, 10_000.0);
        let (c0, s0) = rope.forward(4).unwrap();
        let (co, so) = rope.forward_offset(4, 0).unwrap();
        assert!(array_eq(&c0, &co, false).unwrap().item::<bool>());
        assert!(array_eq(&s0, &so, false).unwrap().item::<bool>());

        // forward_offset(1, 3) == row 3 of forward(8).
        let (cfull, sfull) = rope.forward(8).unwrap();
        let (c3, s3) = rope.forward_offset(1, 3).unwrap();
        let row = |a: &Array, r: i32| {
            let idx = Array::from_slice(&[r], &[1]);
            a.take_axis(&idx, 1).unwrap()
        };
        let close = |a: &Array, b: &Array| {
            max(abs(subtract(a, b).unwrap()).unwrap(), None)
                .unwrap()
                .item::<f32>()
                < 1e-6
        };
        assert!(close(&c3, &row(&cfull, 3)));
        assert!(close(&s3, &row(&sfull, 3)));
    }

    #[test]
    fn compile_glue_helpers_are_bit_exact_and_modulate_dtype_policy() {
        // F-101: the hoisted glue helpers must be bit-identical eager vs compiled (the sc-2963
        // invariant), now testable in core without weights.
        let bit_exact = |on: bool| {
            set_compile_glue(false);
            let x = Array::from_slice(&[1.0f32, -2.0, 3.0, 0.5], &[1, 1, 4]);
            let g = Array::from_slice(&[0.5f32, 0.25, -0.5, 2.0], &[1, 1, 4]);
            let y = Array::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[1, 1, 4]);
            let eager_gated = gated(&x, &g, &y).unwrap();
            let eager_mod = modulate(&x, &g, &y, false).unwrap();
            let (er, ei) = rope_rotate(&x, &g, &y, &x).unwrap();
            set_compile_glue(on);
            let c_gated = gated(&x, &g, &y).unwrap();
            let c_mod = modulate(&x, &g, &y, false).unwrap();
            let (cr, ci) = rope_rotate(&x, &g, &y, &x).unwrap();
            set_compile_glue(false);
            for (a, b) in [
                (eager_gated, c_gated),
                (eager_mod, c_mod),
                (er, cr),
                (ei, ci),
            ] {
                assert!(array_eq(&a, &b, None).unwrap().item::<bool>());
            }
        };
        bit_exact(false); // eager == eager (sanity)
        bit_exact(true); // compiled == eager (the invariant)

        // modulate dtype policy (F-101): with a bf16 `scale`, `one_matches_scale=true` keeps the
        // `1+scale` add in bf16 (coarse), while `false` promotes it to f32 — a real, different value.
        set_compile_glue(false);
        let norm = Array::from_slice(&[1.0f32], &[1]);
        // 0.003 is not representable in bf16; (1 + 0.003) rounds differently bf16 vs f32.
        let scale = Array::from_slice(&[0.003f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let shift = Array::from_slice(&[0.0f32], &[1]);
        let bf16_one = modulate(&norm, &scale, &shift, true).unwrap();
        let f32_one = modulate(&norm, &scale, &shift, false).unwrap();
        assert_ne!(
            bf16_one.as_dtype(Dtype::Float32).unwrap().item::<f32>(),
            f32_one.as_dtype(Dtype::Float32).unwrap().item::<f32>(),
            "the dtype policy for the literal 1 must actually change the result"
        );
    }

    #[test]
    fn compile_glue_guard_scopes_and_restores() {
        // F-007: the shared RAII guard enables compiled glue for its scope and restores the PRIOR
        // value on drop. The setting is isolated to this test's thread.
        set_compile_glue(false);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue(), "guard enables compiled glue for its scope");
        }
        assert!(!compile_glue(), "guard restores the prior (off) on drop");

        // Restores the *prior* value, not a hardcoded false: a prior `on` stays on after drop.
        set_compile_glue(true);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue());
        }
        assert!(compile_glue(), "guard restores the prior (on) on drop");

        set_compile_glue(false); // leave this thread eager, as the reference-parity gates expect
    }

    #[test]
    fn compile_glue_setting_is_thread_local() {
        use std::sync::{Arc, Barrier};

        let barrier = Arc::new(Barrier::new(2));
        let enabled_barrier = Arc::clone(&barrier);
        let enabled = std::thread::spawn(move || {
            set_compile_glue(true);
            enabled_barrier.wait();
            assert!(compile_glue());
            enabled_barrier.wait();
            set_compile_glue(false);
        });

        let eager_barrier = Arc::clone(&barrier);
        let eager = std::thread::spawn(move || {
            set_compile_glue(false);
            eager_barrier.wait();
            assert!(
                !compile_glue(),
                "a concurrent render must not inherit another thread's compiled path"
            );
            eager_barrier.wait();
        });

        enabled.join().unwrap();
        eager.join().unwrap();
    }

    #[test]
    fn retained_compile_glue_is_bit_exact_traces_once_and_reports_hits() {
        use crate::diagnostics::{
            self, CompileDisposition, DiagnosticCounter, ToggleDisposition, RETAINED_COMPILATION,
        };
        use std::sync::atomic::Ordering;

        RETAINED_COMPILE_GLUE.with(|slot| *slot.borrow_mut() = None);
        set_compile_glue(true);

        let x = Array::from_slice(&[1.0f32, -2.0, 3.0, 0.5], &[1, 1, 4]);
        let gate = Array::from_slice(&[0.5f32, 0.25, -0.5, 2.0], &[1, 1, 4]);
        let y = Array::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[1, 1, 4]);

        // Capture the exact one-shot outputs first. Reset the trace probes afterward so the
        // assertions below count only retained-handle traces, not these reference calls.
        let oneshot_gated = gated(&x, &gate, &y).unwrap();
        let oneshot_rope = rope_rotate(&x, &gate, &y, &x).unwrap();
        let oneshot_modulate_matching = modulate(&x, &gate, &y, true).unwrap();
        let oneshot_modulate_f32 = modulate(&x, &gate, &y, false).unwrap();
        let oneshot_gelu_exact = gelu_exact(&x).unwrap();
        let oneshot_gelu_quick = gelu_quick(&x).unwrap();
        for count in &COMPILE_TRACE_COUNTS {
            count.store(0, Ordering::Relaxed);
        }

        let scope = diagnostics::begin_request_with_toggles(
            "retained-shared-nn",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();

        let gated_first = gated(&x, &gate, &y).unwrap();
        let gated_second = gated(&x, &gate, &y).unwrap();
        for retained in [&gated_first, &gated_second] {
            assert!(array_eq(retained, &oneshot_gated, None)
                .unwrap()
                .item::<bool>());
        }

        let (rope_real_first, rope_imag_first) = rope_rotate(&x, &gate, &y, &x).unwrap();
        let (rope_real_second, rope_imag_second) = rope_rotate(&x, &gate, &y, &x).unwrap();
        for ((first, second), oneshot) in [
            (rope_real_first, rope_real_second),
            (rope_imag_first, rope_imag_second),
        ]
        .into_iter()
        .zip([&oneshot_rope.0, &oneshot_rope.1])
        {
            for retained in [&first, &second] {
                assert!(array_eq(retained, oneshot, None).unwrap().item::<bool>());
            }
        }

        for (matches_scale, oneshot) in [
            (true, &oneshot_modulate_matching),
            (false, &oneshot_modulate_f32),
        ] {
            let first = modulate(&x, &gate, &y, matches_scale).unwrap();
            let second = modulate(&x, &gate, &y, matches_scale).unwrap();
            for retained in [&first, &second] {
                assert!(array_eq(retained, oneshot, None).unwrap().item::<bool>());
            }
        }

        for (activation, oneshot) in [
            (
                gelu_exact as fn(&Array) -> Result<Array>,
                &oneshot_gelu_exact,
            ),
            (gelu_quick, &oneshot_gelu_quick),
        ] {
            let first = activation(&x).unwrap();
            let second = activation(&x).unwrap();
            for retained in [&first, &second] {
                assert!(array_eq(retained, oneshot, None).unwrap().item::<bool>());
            }
        }

        let report = scope.finish();
        let compile_count = |site: &'static str, disposition| {
            report
                .counters
                .iter()
                .find_map(|counter| match counter {
                    DiagnosticCounter::Compile {
                        site: recorded_site,
                        disposition: recorded_disposition,
                        count,
                    } if *recorded_site == site && *recorded_disposition == disposition => {
                        Some(*count)
                    }
                    _ => None,
                })
                .unwrap_or(0)
        };
        for (site, misses, hits) in [
            (SITE_GATED, 1, 1),
            (SITE_ROPE_ROTATE, 1, 1),
            (SITE_MODULATE, 2, 2),
            (SITE_GELU_EXACT, 1, 1),
            (SITE_GELU_QUICK, 1, 1),
        ] {
            assert_eq!(
                compile_count(site, CompileDisposition::RetainedMiss),
                misses
            );
            assert_eq!(compile_count(site, CompileDisposition::RetainedHit), hits);
        }
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                count: 12,
            }
        )));
        assert!(COMPILE_TRACE_COUNTS
            .iter()
            .all(|count| count.load(Ordering::Relaxed) == 1));

        // Finishing diagnostics must not drop the retained handle. The same dtype at a new shape is
        // a shapeless-cache hit in the next request; a genuinely new dtype signature is a miss.
        let next_scope = diagnostics::begin_request_with_toggles(
            "retained-shared-nn-next-request",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        let shape_x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]);
        let shape_gate = Array::from_slice(&[0.5f32, 0.25], &[1, 1, 2]);
        let shape_y = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 2]);
        gated(&shape_x, &shape_gate, &shape_y)
            .unwrap()
            .eval()
            .unwrap();
        let half_x = shape_x.as_dtype(Dtype::Float16).unwrap();
        let half_gate = shape_gate.as_dtype(Dtype::Float16).unwrap();
        let half_y = shape_y.as_dtype(Dtype::Float16).unwrap();
        gated(&half_x, &half_gate, &half_y).unwrap().eval().unwrap();
        let next_report = next_scope.finish();
        assert!(next_report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GATED,
                disposition: CompileDisposition::RetainedHit,
                count: 1,
            }
        )));
        assert!(next_report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GATED,
                disposition: CompileDisposition::RetainedMiss,
                count: 1,
            }
        )));
        assert!(next_report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                count: 2,
            }
        )));
        assert_eq!(
            COMPILE_TRACE_COUNTS[TRACE_GATED].load(Ordering::Relaxed),
            2,
            "only the new dtype signature should retrace"
        );

        let baseline = diagnostics::begin_request("oneshot-shared-nn", "test").unwrap();
        gated(&x, &gate, &y).unwrap().eval().unwrap();
        let baseline = baseline.finish();
        assert!(baseline.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GATED,
                disposition: CompileDisposition::OneShot,
                count: 1,
            }
        )));
        assert!(!baseline.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                ..
            }
        )));

        // A one-shot compile owns an independent MLX lease. It must neither touch this retained
        // handle nor make the diagnostic signature inventory claim that the retained graph missed.
        let after_oneshot = diagnostics::begin_request_with_toggles(
            "retained-shared-nn-after-oneshot",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        gated(&x, &gate, &y).unwrap().eval().unwrap();
        let after_oneshot = after_oneshot.finish();
        assert!(after_oneshot.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GATED,
                disposition: CompileDisposition::RetainedHit,
                count: 1,
            }
        )));

        // A real process cache clear is different: the upstream generation advances, so the next
        // retained call must fail closed as a miss even though the Rust handle remains alive.
        mlx_rs::transforms::compile::clear_cache();
        let after_cache_clear = diagnostics::begin_request_with_toggles(
            "retained-shared-nn-after-cache-clear",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        gated(&x, &gate, &y).unwrap().eval().unwrap();
        let after_cache_clear = after_cache_clear.finish();
        assert!(after_cache_clear.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: SITE_GATED,
                disposition: CompileDisposition::RetainedMiss,
                count: 1,
            }
        )));

        set_compile_glue(false);
        RETAINED_COMPILE_GLUE.with(|slot| *slot.borrow_mut() = None);
    }

    #[test]
    #[ignore = "serialized audit of process-global MLX active/cache counters"]
    fn retained_compile_handles_do_not_pin_request_arrays() {
        use crate::diagnostics::{self, RETAINED_COMPILATION};
        use mlx_rs::memory::{clear_cache, get_active_memory, get_cache_memory};
        use mlx_rs::ops::ones;

        set_compile_glue(false);
        RETAINED_COMPILE_GLUE.with(|slot| *slot.borrow_mut() = None);
        clear_cache();
        let baseline_active = get_active_memory();

        let input = ones::<f32>(&[1024, 1024]).unwrap();
        input.eval().unwrap();
        clear_cache();
        let input_active = get_active_memory();
        let request_bytes = input.nbytes();

        let scope = diagnostics::begin_request_with_toggles(
            "retained-memory-audit",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        set_compile_glue(true);
        let output = gelu_quick(&input).unwrap();
        output.eval().unwrap();
        let _ = gelu_quick(&input).unwrap();
        let _ = scope.finish();
        drop(output);
        drop(input);
        clear_cache();

        let retained_active = get_active_memory();
        let retained_cache = get_cache_memory();
        assert!(
            retained_active.saturating_sub(baseline_active) < request_bytes / 8,
            "a retained graph must not pin a request-sized input/output allocation: baseline={baseline_active}, retained={retained_active}, request={request_bytes}"
        );
        assert_eq!(
            retained_cache, 0,
            "clear_cache must release allocator cache"
        );

        RETAINED_COMPILE_GLUE.with(|slot| *slot.borrow_mut() = None);
        clear_cache();
        let released_active = get_active_memory();
        let released_cache = get_cache_memory();
        assert!(released_active <= retained_active);
        assert_eq!(released_cache, 0);
        eprintln!(
            "[retained-memory] baseline_active={baseline_active} input_active={input_active} retained_active={retained_active} released_active={released_active} retained_cache={retained_cache} released_cache={released_cache} request_bytes={request_bytes}"
        );
        set_compile_glue(false);
    }

    #[test]
    fn linear_is_fused_addmm() {
        // sc-2779: the shared biased-linear helper must be a FUSED addmm. In bf16, addmm differs
        // from matmul+add (single vs double rounding), so prove fusion on bf16 inputs.
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let b = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..4 * 64)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let got = linear(&x, &w, &b).unwrap();
        let want = addmm(&b, &x, w.t(), 1.0, 1.0).unwrap();
        assert!(array_eq(&got, &want, false).unwrap().item::<bool>());
    }

    #[test]
    fn gelu_tanh_preserves_dtype() {
        // sc-2779: a bf16 input must stay bf16 (MLX weak-casts the constants to the input dtype),
        // and an f32 input must stay f32.
        let x32 = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        assert_eq!(gelu_tanh(&x32).unwrap().dtype(), Dtype::Float32);
        let xbf = x32.as_dtype(Dtype::Bfloat16).unwrap();
        assert_eq!(gelu_tanh(&xbf).unwrap().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn gelu_tanh_f32_matches_closed_form() {
        // f32 path matches the closed form computed with the f64-host √(2/π) constant, bit-exact.
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        let c = scalar((2.0_f64 / std::f64::consts::PI).sqrt() as f32);
        let x3 = power(&x, Array::from_int(3)).unwrap();
        let inner = multiply(
            add(&x, multiply(&x3, scalar(0.044_715)).unwrap()).unwrap(),
            &c,
        )
        .unwrap();
        let gate = add(tanh(&inner).unwrap(), scalar(1.0)).unwrap();
        let want = multiply(multiply(&x, scalar(0.5)).unwrap(), &gate).unwrap();
        assert!(array_eq(gelu_tanh(&x).unwrap(), &want, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn gelu_tanh_bf16_is_close_to_f32_truth() {
        // Preserving bf16 must not mean *wrong*: the bf16 result tracks the f32 reference within
        // bf16 rounding.
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        let truth = gelu_tanh(&x).unwrap();
        let bf = gelu_tanh(&x.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let d = max(abs(subtract(&bf, &truth).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert!(d < 5e-2, "bf16 gelu_tanh diverged from f32 truth: {d}");
    }

    #[test]
    fn conv3d_1x1x1_sums_input_channels_with_bias() {
        // NDHWC: a single voxel with 2 input channels [1, 2].
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 1, 2]);
        // weight [out=1, kD=1, kH=1, kW=1, in=2] = ones -> sums over the input channels.
        let w = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 1, 1, 2]);
        let bias = Array::from_slice(&[10.0f32], &[1]);
        let y = conv3d(&x, &w, Some(&bias), (1, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert_eq!(y.item::<f32>(), 13.0); // 1 + 2 + bias 10
    }
}
