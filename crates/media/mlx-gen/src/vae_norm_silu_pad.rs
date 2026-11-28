//! SW-6142 (SW-F287 M4) — the MLX video VAEs' conv input `pad(silu(norm(x)))`, fused.
//!
//! The Wan z16 and LTX video VAEs feed every 3-D conv after a norm the same chain: a per-voxel
//! channel normalization (`ChannelL2` for Wan — `x / sqrt(max(Σx², eps)) · √C · γ`; `Pixel` for LTX
//! — `x / sqrt(Σx²/C + eps)`), SiLU, then the conv's padding (Wan: causal **zero** frames, with the
//! chunked encoder's cached frames in front of the chunk; LTX: **replicated** first/last frames) and
//! a symmetric spatial zero pad, and finally the NCTHW → NDHWC transpose the channels-last MLX conv
//! copies into a contiguous buffer. Eagerly that is ~6 full-resolution passes (divide, scale,
//! γ-multiply, SiLU, pad, the conv's layout copy), and the layout flips twice: the conv output is a
//! channels-last buffer viewed as NCTHW, the broadcast divide writes it channels-first, and the conv
//! copies it channels-last again.
//!
//! [`try_fused`] produces the conv-ready, contiguous NDHWC padded input directly, by one of three
//! [`NormSiluPadRoute`]s:
//!
//! - [`NormSiluPadRoute::Metal`] (default) — one custom Metal kernel (`vae_norm_silu_pad.metal`,
//!   through MLX's `mlx_fast_metal_kernel` wrapper [`MetalKernel`]) that reads `x` in whatever
//!   layout it is in and writes `silu(norm(x))`, the zero/replicated temporal frames, the cached
//!   prefix frames and the spatial zeros straight into the padded NDHWC buffer, plus (for the
//!   chunked encoder) the last activated frames the next chunk's cache needs.
//! - [`NormSiluPadRoute::Compiled`] — `mx.compile` of the elementwise tail on channels-last views
//!   (so the fused elementwise kernel already writes NDHWC), then MLX's own concat/pad.
//! - [`NormSiluPadRoute::Reference`] — `try_fused` returns `None` and the caller runs its own
//!   unchanged op chain (E1: the parity oracle and the path for a CPU stream, an unserved shape, or
//!   a failed kernel compile).
//!
//! **Exactness (E2).** Neither fused route re-implements the channel reduction: the per-voxel
//! denominator is computed by [`ChannelNorm::denominator`] — the very ops the reference
//! [`ChannelNorm::apply`] runs, on the same input — so it is bit-identical by construction. The
//! Metal kernel then performs only the elementwise tail in the reference order with FP contraction
//! off (the SW-6139 GDN lesson) and the metallib's precise `exp` / divide named explicitly; its f32
//! output targets bit identity with the op chain (reported by the parity tests and proofs), and
//! E2's `max rel err ≤ 1e-5` is the gate. Every route returns **f32** for any float input: the
//! reference promotes there too (its `eps`/`√C` operands are f32 scalars), so bf16/f16 inputs
//! compare against the same f32 values.
//!
//! **Telemetry and switch (E1).** The route is [`DEFAULT_ROUTE`], overridable per process by
//! [`ROUTE_ENV`] (`reference` | `compiled` | `metal`) and per thread by [`RouteGuard`]. Every call
//! records in diagnostics which route ran ([`METAL_TOGGLE`] / [`COMPILED_TOGGLE`] `Applied`, or a
//! [`SITE`] fallback with a stable [`NormSiluPadRefusal::label`]) and bumps this thread's
//! [`thread_counts`]; a kernel compile failure is probed once per variant, cached (E8), reported
//! once on stderr, and from then on routes to the reference with reason `kernel_compile_failed`.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use mlx_rs::error::Exception;
use mlx_rs::fast::MetalKernel;
use mlx_rs::ops::{add, concatenate_axis, divide, maximum, multiply, pad, sigmoid, sum_axes};
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype, Stream};

use crate::array::{scalar, slice_range};
use crate::{Error, Result};

/// Diagnostics toggle recorded each time the fused Metal kernel ran.
pub const METAL_TOGGLE: &str = "vae_norm_silu_pad_metal";
/// Diagnostics toggle recorded each time the compiled route ran.
pub const COMPILED_TOGGLE: &str = "vae_norm_silu_pad_compiled";
/// Diagnostics site of every fallback to the caller's reference chain (and of the compiled
/// route's compile receipts).
pub const SITE: &str = "mlx_gen::vae_norm_silu_pad";
/// Process-wide route override: `reference`, `compiled` or `metal`.
pub const ROUTE_ENV: &str = "MLX_GEN_VAE_NORM_SILU_PAD";

/// Which implementation produces the VAE conv input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormSiluPadRoute {
    /// The caller's eager op chain (the parity oracle).
    Reference,
    /// `mx.compile` of the elementwise tail on NDHWC views, then MLX concat/pad.
    Compiled,
    /// The custom Metal kernel.
    Metal,
}

impl NormSiluPadRoute {
    /// Stable name (also the [`ROUTE_ENV`] value).
    pub fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Compiled => "compiled",
            Self::Metal => "metal",
        }
    }

    /// Parse a [`label`](Self::label) (case-insensitive; `off` is an alias of `reference`).
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "reference" | "off" => Some(Self::Reference),
            "compiled" | "compile" => Some(Self::Compiled),
            "metal" | "kernel" | "on" => Some(Self::Metal),
            _ => None,
        }
    }
}

/// The route used when neither [`ROUTE_ENV`] nor a [`RouteGuard`] says otherwise.
///
/// The Metal kernel is exact (see the module docs), so it is on by default; the measured decision
/// and its numbers live with SW-6142 (`vae_norm_silu_pad_proof` in `mlx-gen-wan` / `mlx-gen-ltx`).
pub const DEFAULT_ROUTE: NormSiluPadRoute = NormSiluPadRoute::Metal;

thread_local! {
    static ROUTE_OVERRIDE: Cell<Option<NormSiluPadRoute>> = const { Cell::new(None) };
    static COUNTS: Cell<RouteCounts> = const { Cell::new(RouteCounts::ZERO) };
}

fn process_route() -> NormSiluPadRoute {
    static ROUTE: OnceLock<NormSiluPadRoute> = OnceLock::new();
    *ROUTE.get_or_init(|| match std::env::var(ROUTE_ENV) {
        Ok(value) => NormSiluPadRoute::parse(&value).unwrap_or_else(|| {
            eprintln!(
                "{SITE}: ignoring {ROUTE_ENV}={value:?} (expected reference|compiled|metal); \
                 using {}",
                DEFAULT_ROUTE.label()
            );
            DEFAULT_ROUTE
        }),
        Err(_) => DEFAULT_ROUTE,
    })
}

/// The route [`try_fused`] takes on this thread.
pub fn route() -> NormSiluPadRoute {
    ROUTE_OVERRIDE.get().unwrap_or_else(process_route)
}

/// Override (or with `None` clear the override of) the route on this thread.
pub fn set_route(route: Option<NormSiluPadRoute>) {
    ROUTE_OVERRIDE.set(route);
}

/// RAII guard setting this thread's route for its lifetime and restoring the prior override.
#[must_use = "dropping the guard restores the prior route"]
pub struct RouteGuard {
    prev: Option<NormSiluPadRoute>,
}

impl RouteGuard {
    pub fn set(route: NormSiluPadRoute) -> Self {
        Self {
            prev: ROUTE_OVERRIDE.replace(Some(route)),
        }
    }
}

impl Drop for RouteGuard {
    fn drop(&mut self) {
        ROUTE_OVERRIDE.set(self.prev);
    }
}

/// Per-thread route tally (tests and proofs read deltas of it; diagnostics carry the same facts
/// per request).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteCounts {
    /// Calls served by the Metal kernel.
    pub metal: u64,
    /// Calls served by the compiled route.
    pub compiled: u64,
    /// Calls that ran the caller's reference chain because the route is `Reference`.
    pub reference: u64,
    /// Calls that fell back to the reference chain because a fused route refused the input or
    /// its kernel failed to compile.
    pub refused: u64,
    /// The most recent refusal's [`NormSiluPadRefusal::label`].
    pub last_refusal: Option<&'static str>,
}

impl RouteCounts {
    const ZERO: Self = Self {
        metal: 0,
        compiled: 0,
        reference: 0,
        refused: 0,
        last_refusal: None,
    };

    /// Counts accumulated since `start` (an earlier [`thread_counts`] snapshot).
    pub fn since(&self, start: &RouteCounts) -> RouteCounts {
        RouteCounts {
            metal: self.metal - start.metal,
            compiled: self.compiled - start.compiled,
            reference: self.reference - start.reference,
            refused: self.refused - start.refused,
            last_refusal: self.last_refusal,
        }
    }
}

/// This thread's route tally.
pub fn thread_counts() -> RouteCounts {
    COUNTS.get()
}

fn bump(f: impl FnOnce(&mut RouteCounts)) {
    let mut counts = COUNTS.get();
    f(&mut counts);
    COUNTS.set(counts);
}

/// The per-voxel channel normalization in front of the SiLU.
#[derive(Clone, Copy, Debug)]
pub enum ChannelNorm<'a> {
    /// Wan z16 `RMS_norm`: `x / sqrt(max(Σ_C x², eps)) · √C · γ` (`gamma` carries `C` elements in
    /// any shape; `eps` is the clamp on the sum of squares).
    ChannelL2 { gamma: &'a Array, eps: f32 },
    /// LTX `pixel_norm`: `x / sqrt(Σ_C x² / C + eps)` (no `√C`, no γ).
    Pixel { eps: f32 },
}

impl ChannelNorm<'_> {
    /// The per-voxel denominator `[.., 1, ..]` (channel axis 1, kept) — the reference ops, so the
    /// fused routes divide by exactly the value [`apply`](Self::apply) divides by.
    pub fn denominator(&self, x: &Array) -> Result<Array> {
        let sum_sq = sum_axes(&multiply(x, x)?, &[1], true)?;
        Ok(match self {
            ChannelNorm::ChannelL2 { eps, .. } => maximum(&sum_sq, scalar(*eps))?.sqrt()?,
            ChannelNorm::Pixel { eps } => {
                let c = x.shape()[1] as f32;
                let mean = divide(&sum_sq, scalar(c))?;
                add(&mean, scalar(*eps))?.sqrt()?
            }
        })
    }

    /// The reference normalization (any rank, channel axis 1) — the op chain the Wan
    /// `rms_norm_channels` and LTX `pixel_norm` run.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        let denom = self.denominator(x)?;
        let normed = divide(x, &denom)?;
        match self {
            ChannelNorm::ChannelL2 { gamma, .. } => {
                let shape = x.shape();
                let nd = shape.len();
                let c = shape[1];
                let scaled = multiply(&normed, scalar((c as f32).sqrt()))?;
                let mut wshape = vec![1i32; nd];
                wshape[1] = c;
                Ok(multiply(&scaled, &gamma.reshape(&wshape)?)?)
            }
            ChannelNorm::Pixel { .. } => Ok(normed),
        }
    }

    fn gamma(&self) -> Option<&Array> {
        match self {
            ChannelNorm::ChannelL2 { gamma, .. } => Some(*gamma),
            ChannelNorm::Pixel { .. } => None,
        }
    }
}

/// The temporal padding of the conv that follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalPad {
    /// `front` zero frames, then the [`PadSpec::cache`] frames (if any), then `x`, then `back`
    /// zero frames (Wan `CausalConv3d`: causal left pad, cached left context).
    Zeros { front: i32, back: i32 },
    /// `front` copies of the first activated frame, `x`, then `back` copies of the last (LTX
    /// `CausalConv3d` frame replication).
    Replicate { front: i32, back: i32 },
}

impl TemporalPad {
    fn frames(self) -> (i32, i32, bool) {
        match self {
            TemporalPad::Zeros { front, back } => (front, back, false),
            TemporalPad::Replicate { front, back } => (front, back, true),
        }
    }
}

/// Everything about the conv input's padding.
#[derive(Clone, Copy, Debug)]
pub struct PadSpec<'a> {
    pub temporal: TemporalPad,
    /// Already-activated frames `[B, C, TC, H, W]` (f32, any strides) inserted between the front
    /// zero frames and `x` — the chunked encoder's carried context. Zeros mode only.
    pub cache: Option<&'a Array>,
    /// Symmetric spatial zero pad (H, W).
    pub ph: i32,
    pub pw: i32,
}

/// The conv-ready input: contiguous (fused routes) channels-last padded `[B, TP, HP, WP, C]`,
/// plus, when asked for, the last `tail` activated frames of `x` as `[B, C, tail, H, W]`.
pub struct NormSiluPadOut {
    pub ndhwc: Array,
    pub tail: Option<Array>,
}

/// Why a fused route did not serve an input (the caller then runs its reference chain).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormSiluPadRefusal {
    /// Not a GPU stream: a custom Metal kernel cannot run there.
    CpuStream,
    /// `x` is not rank 5 (`[B, C, T, H, W]`).
    Rank(usize),
    /// `x`, γ or the denominator is not a supported float dtype.
    Dtype(Dtype),
    /// An empty axis.
    Empty,
    /// γ does not carry exactly `C` elements.
    GammaShape,
    /// The cache is not `[B, C, TC ≥ 1, H, W]` f32.
    CacheShape,
    /// Cached frames only exist in front of zero padding (Wan); replicate padding has none.
    CacheWithReplicate,
    /// `tail` outside `0..=T`, or asked for under replicate padding.
    Tail,
    /// A negative pad.
    Pad,
    /// A tensor above `i32::MAX` elements (E6).
    IndexOverflow,
    /// The kernel variant failed to compile or failed its one-time probe (cached, E8).
    KernelCompile,
}

impl NormSiluPadRefusal {
    /// Stable diagnostics reason.
    pub fn label(&self) -> &'static str {
        match self {
            Self::CpuStream => "cpu_stream",
            Self::Rank(_) => "rank",
            Self::Dtype(_) => "dtype",
            Self::Empty => "empty",
            Self::GammaShape => "gamma_shape",
            Self::CacheShape => "cache_shape",
            Self::CacheWithReplicate => "cache_with_replicate",
            Self::Tail => "tail",
            Self::Pad => "pad",
            Self::IndexOverflow => "index_overflow",
            Self::KernelCompile => "kernel_compile_failed",
        }
    }
}

/// The admitted geometry of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormSiluPadPlan {
    /// `[B, C, T, H, W]` of `x`.
    pub x: [i32; 5],
    pub cache_frames: i32,
    pub front: i32,
    pub back: i32,
    pub ph: i32,
    pub pw: i32,
    pub tail: i32,
    pub replicate: bool,
    pub affine: bool,
    /// `[B, TP, HP, WP, C]` of the conv-ready output.
    pub out: [i32; 5],
}

impl NormSiluPadPlan {
    /// Output elements (≤ `i32::MAX` by construction).
    pub fn out_elements(&self) -> i32 {
        self.out.iter().product()
    }
}

fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
}

/// The pure admission check (CPU-testable): shapes, dtypes and pads are validated **before** any
/// launch. `gamma` is `(elements, dtype)` for [`ChannelNorm::ChannelL2`], `None` for `Pixel`;
/// `cache` is `(shape, dtype)`.
#[allow(clippy::too_many_arguments)]
pub fn plan_norm_silu_pad(
    x_shape: &[i32],
    x_dtype: Dtype,
    gamma: Option<(usize, Dtype)>,
    temporal: TemporalPad,
    cache: Option<(&[i32], Dtype)>,
    spatial: (i32, i32),
    tail: i32,
    on_gpu: bool,
) -> std::result::Result<NormSiluPadPlan, NormSiluPadRefusal> {
    if !on_gpu {
        return Err(NormSiluPadRefusal::CpuStream);
    }
    if x_shape.len() != 5 {
        return Err(NormSiluPadRefusal::Rank(x_shape.len()));
    }
    if !is_float(x_dtype) {
        return Err(NormSiluPadRefusal::Dtype(x_dtype));
    }
    let [b, c, t, h, w] = [x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]];
    if [b, c, t, h, w].iter().any(|&d| d < 1) {
        return Err(NormSiluPadRefusal::Empty);
    }
    if let Some((elements, dtype)) = gamma {
        if !is_float(dtype) {
            return Err(NormSiluPadRefusal::Dtype(dtype));
        }
        if elements != c as usize {
            return Err(NormSiluPadRefusal::GammaShape);
        }
    }
    let (front, back, replicate) = temporal.frames();
    let (ph, pw) = spatial;
    if front < 0 || back < 0 || ph < 0 || pw < 0 {
        return Err(NormSiluPadRefusal::Pad);
    }
    let cache_frames = match cache {
        None => 0,
        Some(_) if replicate => return Err(NormSiluPadRefusal::CacheWithReplicate),
        Some((shape, dtype)) => {
            if shape.len() != 5
                || dtype != Dtype::Float32
                || shape[0] != b
                || shape[1] != c
                || shape[2] < 1
                || shape[3] != h
                || shape[4] != w
            {
                return Err(NormSiluPadRefusal::CacheShape);
            }
            shape[2]
        }
    };
    if !(0..=t).contains(&tail) || (tail > 0 && replicate) {
        return Err(NormSiluPadRefusal::Tail);
    }
    let tp = i64::from(front) + i64::from(cache_frames) + i64::from(t) + i64::from(back);
    let (hp, wp) = (
        i64::from(h) + 2 * i64::from(ph),
        i64::from(w) + 2 * i64::from(pw),
    );
    let count = |dims: &[i64]| dims.iter().product::<i64>();
    let (b64, c64) = (i64::from(b), i64::from(c));
    let limit = i64::from(i32::MAX);
    let biggest = [
        count(&[b64, tp, hp, wp, c64]),
        count(&[b64, c64, i64::from(t), i64::from(h), i64::from(w)]),
        count(&[
            b64,
            c64,
            i64::from(cache_frames),
            i64::from(h),
            i64::from(w),
        ]),
    ];
    if biggest.iter().any(|&n| n > limit) || tp > limit || hp > limit || wp > limit {
        return Err(NormSiluPadRefusal::IndexOverflow);
    }
    Ok(NormSiluPadPlan {
        x: [b, c, t, h, w],
        cache_frames,
        front,
        back,
        ph,
        pw,
        tail,
        replicate,
        affine: gamma.is_some(),
        out: [b, tp as i32, hp as i32, wp as i32, c],
    })
}

fn stream_is_gpu(stream: &Stream) -> bool {
    matches!(
        stream.get_device().and_then(|device| device.get_type()),
        Ok(mlx_rs::DeviceType::Gpu)
    )
}

fn refuse(refusal: NormSiluPadRefusal) -> Result<Option<NormSiluPadOut>> {
    crate::diagnostics::record_fallback(SITE, refusal.label());
    bump(|c| {
        c.refused += 1;
        c.last_refusal = Some(refusal.label());
    });
    Ok(None)
}

/// **The shared entry point.** On the thread's [`route`], produce the conv-ready NDHWC input for
/// `pad(silu(norm(x)))` (+ the last `tail` activated frames), or return `None` — after recording
/// why — for the caller to run its own reference chain (`Reference` route, a refused input, a
/// failed kernel compile).
pub fn try_fused(
    x: &Array,
    norm: &ChannelNorm<'_>,
    pad_spec: &PadSpec<'_>,
    tail: i32,
) -> Result<Option<NormSiluPadOut>> {
    let route = route();
    if route == NormSiluPadRoute::Reference {
        crate::diagnostics::record_fallback(SITE, "disabled");
        bump(|c| c.reference += 1);
        return Ok(None);
    }
    let stream = Stream::task_local_or_default();
    let plan = match plan_for(x, norm, pad_spec, tail, stream_is_gpu(&stream)) {
        Ok(plan) => plan,
        Err(refusal) => return refuse(refusal),
    };
    match route {
        NormSiluPadRoute::Metal => {
            if !kernel_available(&stream, x.dtype(), norm, &plan) {
                return refuse(NormSiluPadRefusal::KernelCompile);
            }
            let denom = norm.denominator(x)?;
            if denom.dtype() != Dtype::Float32 {
                return refuse(NormSiluPadRefusal::Dtype(denom.dtype()));
            }
            let out = launch_metal(x, &denom, norm, pad_spec, &plan, &stream)?;
            crate::diagnostics::record_toggle(
                METAL_TOGGLE,
                crate::diagnostics::ToggleDisposition::Applied,
            );
            bump(|c| c.metal += 1);
            Ok(Some(out))
        }
        NormSiluPadRoute::Compiled => {
            let out = compiled_conv_input(x, norm, pad_spec, &plan)?;
            crate::diagnostics::record_compile(
                SITE,
                crate::diagnostics::CompileDisposition::OneShot,
            );
            crate::diagnostics::record_toggle(
                COMPILED_TOGGLE,
                crate::diagnostics::ToggleDisposition::Applied,
            );
            bump(|c| c.compiled += 1);
            Ok(Some(out))
        }
        NormSiluPadRoute::Reference => unreachable!("handled above"),
    }
}

fn plan_for(
    x: &Array,
    norm: &ChannelNorm<'_>,
    pad_spec: &PadSpec<'_>,
    tail: i32,
    on_gpu: bool,
) -> std::result::Result<NormSiluPadPlan, NormSiluPadRefusal> {
    plan_norm_silu_pad(
        x.shape(),
        x.dtype(),
        norm.gamma().map(|g| (g.size(), g.dtype())),
        pad_spec.temporal,
        pad_spec.cache.map(|c| (c.shape(), c.dtype())),
        (pad_spec.ph, pad_spec.pw),
        tail,
        on_gpu,
    )
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Reference (the parity oracle, shared with the tests)
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// The reference conv input, op for op what the Wan / LTX `CausalConv3d` callers build eagerly:
/// `silu(norm(x))`, the cache concat + zero pad (Wan) or frame replication + spatial zero pad
/// (LTX), and the NCTHW → NDHWC transpose (a **view**; the conv copies it). The tail is a slice of
/// the activated tensor.
pub fn reference_conv_input(
    x: &Array,
    norm: &ChannelNorm<'_>,
    pad_spec: &PadSpec<'_>,
    tail: i32,
) -> Result<NormSiluPadOut> {
    let act = crate::nn::silu(&norm.apply(x)?)?;
    let t = x.shape()[2];
    let tail = if tail > 0 {
        Some(slice_range(&act, 2, t - tail, t)?)
    } else {
        None
    };
    let (ph, pw) = (pad_spec.ph, pad_spec.pw);
    let mut y = act;
    match pad_spec.temporal {
        TemporalPad::Zeros { front, back } => {
            if let Some(cache) = pad_spec.cache {
                y = concatenate_axis(&[cache, &y], 2)?;
            }
            if front > 0 || back > 0 || ph > 0 || pw > 0 {
                y = pad(
                    &y,
                    &[(0, 0), (0, 0), (front, back), (ph, ph), (pw, pw)][..],
                    None,
                    None,
                )?;
            }
        }
        TemporalPad::Replicate { front, back } => {
            if front > 0 || back > 0 {
                y = replicate_frames(&y, 2, front, back)?;
            }
            if ph > 0 || pw > 0 {
                y = pad(
                    &y,
                    &[(0, 0), (0, 0), (0, 0), (ph, ph), (pw, pw)][..],
                    None,
                    None,
                )?;
            }
        }
    }
    Ok(NormSiluPadOut {
        ndhwc: y.transpose_axes(&[0, 2, 3, 4, 1])?,
        tail,
    })
}

/// `front` copies of the first frame, `x`, `back` copies of the last, along time axis `axis` —
/// the LTX `repeat_frame` concat.
fn replicate_frames(x: &Array, axis: i32, front: i32, back: i32) -> Result<Array> {
    let t = x.shape()[axis as usize];
    let first = slice_range(x, axis, 0, 1)?;
    let last = slice_range(x, axis, t - 1, t)?;
    let mut parts: Vec<&Array> = Vec::new();
    parts.extend((0..front).map(|_| &first));
    parts.push(x);
    parts.extend((0..back).map(|_| &last));
    Ok(concatenate_axis(&parts, axis)?)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Compiled route
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// `silu(x / d · √C · γ)` on channels-last views (C last). Traced per shape (`shapeless = false`):
/// `√C` is read off the traced shape exactly as the reference reads it off axis 1.
fn compiled_channel_l2(
    (x, d, g): (&Array, &Array, &Array),
) -> std::result::Result<Array, Exception> {
    let c = x.shape()[x.ndim() - 1];
    let n = divide(x, d)?;
    let n = multiply(&n, scalar((c as f32).sqrt()))?;
    let n = multiply(&n, g)?;
    multiply(&n, &sigmoid(&n)?)
}

/// `silu(x / d)` on channels-last views.
fn compiled_pixel((x, d): (&Array, &Array)) -> std::result::Result<Array, Exception> {
    let n = divide(x, d)?;
    multiply(&n, &sigmoid(&n)?)
}

/// Last `n` frames of a channels-last `[B, T, H, W, C]` activation as an independent `[B, C, n,
/// H, W]` (gathered, so a carried cache slot never pins the whole activation).
fn tail_from_ndhwc(act: &Array, n: i32) -> Result<Array> {
    let t = act.shape()[1];
    let idx: Vec<i32> = (t - n..t).collect();
    Ok(act
        .take_axis(Array::from_slice(&idx, &[n]), 1)?
        .transpose_axes(&[0, 4, 1, 2, 3])?)
}

fn compiled_conv_input(
    x: &Array,
    norm: &ChannelNorm<'_>,
    pad_spec: &PadSpec<'_>,
    plan: &NormSiluPadPlan,
) -> Result<NormSiluPadOut> {
    let denom = norm.denominator(x)?;
    let xt = x.transpose_axes(&[0, 2, 3, 4, 1])?;
    let dt = denom.transpose_axes(&[0, 2, 3, 4, 1])?;
    let act = match norm {
        ChannelNorm::ChannelL2 { gamma, .. } => {
            let g = gamma.reshape(&[plan.x[1]])?;
            compile(compiled_channel_l2, false)((&xt, &dt, &g))?
        }
        ChannelNorm::Pixel { .. } => compile(compiled_pixel, false)((&xt, &dt))?,
    };
    let tail = if plan.tail > 0 {
        Some(tail_from_ndhwc(&act, plan.tail)?)
    } else {
        None
    };
    let (ph, pw) = (plan.ph, plan.pw);
    let mut y = act;
    if plan.replicate {
        if plan.front > 0 || plan.back > 0 {
            y = replicate_frames(&y, 1, plan.front, plan.back)?;
        }
        if ph > 0 || pw > 0 {
            y = pad(
                &y,
                &[(0, 0), (0, 0), (ph, ph), (pw, pw), (0, 0)][..],
                None,
                None,
            )?;
        }
    } else {
        if let Some(cache) = pad_spec.cache {
            let cache_t = cache.transpose_axes(&[0, 2, 3, 4, 1])?;
            y = concatenate_axis(&[&cache_t, &y], 1)?;
        }
        if plan.front > 0 || plan.back > 0 || ph > 0 || pw > 0 {
            y = pad(
                &y,
                &[(0, 0), (plan.front, plan.back), (ph, ph), (pw, pw), (0, 0)][..],
                None,
                None,
            )?;
        }
    }
    Ok(NormSiluPadOut { ndhwc: y, tail })
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Metal route
// ─────────────────────────────────────────────────────────────────────────────────────────────

const KERNEL_NAME: &str = "sceneworks_vae_norm_silu_pad";
const KERNEL_FILE: &str = include_str!("vae_norm_silu_pad.metal");
const KERNEL_BODY_MARKER: &str = "// ---- sceneworks kernel body ----";
const THREADS_PER_GROUP: i32 = 256;
/// MLX passes an input smaller than this many elements in `constant` memory (and names the
/// kernel variant after it); dummies are sized to stay in `device` memory like real inputs.
const DUMMY_ELEMENTS: i32 = 8;

/// `(header, body)` of the kernel source.
fn kernel_source() -> (&'static str, &'static str) {
    KERNEL_FILE
        .split_once(KERNEL_BODY_MARKER)
        .expect("vae_norm_silu_pad.metal carries its body marker")
}

fn launch_metal(
    x: &Array,
    denom: &Array,
    norm: &ChannelNorm<'_>,
    pad_spec: &PadSpec<'_>,
    plan: &NormSiluPadPlan,
    stream: &Stream,
) -> Result<NormSiluPadOut> {
    let [b, c, t, h, w] = plan.x;
    let (gamma, scale_bits) = match norm {
        ChannelNorm::ChannelL2 { gamma, .. } => {
            (gamma.reshape(&[c])?, (c as f32).sqrt().to_bits() as i32)
        }
        ChannelNorm::Pixel { .. } => (mlx_rs::ops::zeros::<f32>(&[DUMMY_ELEMENTS])?, 0),
    };
    let cache = match pad_spec.cache {
        Some(cache) => cache.clone(),
        None => mlx_rs::ops::zeros::<f32>(&[DUMMY_ELEMENTS])?,
    };
    let [_, tp, hp, wp, _] = plan.out;
    let total = plan.out_elements();
    let dims: [i32; 16] = [
        b,
        c,
        t,
        h,
        w,
        plan.cache_frames,
        plan.front,
        plan.back,
        plan.ph,
        plan.pw,
        plan.tail,
        tp,
        hp,
        wp,
        scale_bits,
        total,
    ];
    let dims = Array::from_slice(&dims, &[16]);
    let tail_shape = if plan.tail > 0 {
        vec![b, c, plan.tail, h, w]
    } else {
        vec![1]
    };
    let (header, body) = kernel_source();
    let kernel = MetalKernel::with_options(
        KERNEL_NAME,
        &["inp", "denom", "gamma", "cache", "dims"],
        &["out", "tail"],
        body,
        header,
        false,
        false,
    )?;
    let mut outs = kernel
        .apply()
        .inputs([x, denom, &gamma, &cache, &dims])
        .output_shape(plan.out.to_vec(), Dtype::Float32)
        .output_shape(tail_shape, Dtype::Float32)
        .template_arg("AFFINE", plan.affine)
        .template_arg("HAS_CACHE", plan.cache_frames > 0)
        .template_arg("HAS_TAIL", plan.tail > 0)
        .template_arg("REPLICATE", plan.replicate)
        .grid(total, 1, 1)
        .thread_group(total.min(THREADS_PER_GROUP), 1, 1)
        .run_device(stream)?;
    let tail_out = outs
        .pop()
        .ok_or_else(|| Error::Msg("vae_norm_silu_pad: kernel returned no tail output".into()))?;
    let out = outs
        .pop()
        .ok_or_else(|| Error::Msg("vae_norm_silu_pad: kernel returned no output".into()))?;
    Ok(NormSiluPadOut {
        ndhwc: out,
        tail: (plan.tail > 0).then_some(tail_out),
    })
}

/// One compiled kernel variant: the template bools plus the dtypes that name MLX's library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VariantKey {
    x: Dtype,
    gamma: Option<Dtype>,
    cache: bool,
    tail: bool,
    replicate: bool,
}

/// Whether this variant compiles and reproduces the reference on a tiny probe — decided once per
/// variant per process (E8: a failure is cached, reported once, and routes to the reference).
fn kernel_available(
    stream: &Stream,
    x_dtype: Dtype,
    norm: &ChannelNorm<'_>,
    plan: &NormSiluPadPlan,
) -> bool {
    static PROBES: OnceLock<Mutex<HashMap<VariantKey, bool>>> = OnceLock::new();
    let key = VariantKey {
        x: x_dtype,
        gamma: norm.gamma().map(Array::dtype),
        cache: plan.cache_frames > 0,
        tail: plan.tail > 0,
        replicate: plan.replicate,
    };
    let probes = PROBES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut probes = probes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&ok) = probes.get(&key) {
        return ok;
    }
    let ok = match probe(stream, key, norm) {
        Ok(()) => true,
        Err(error) => {
            eprintln!(
                "{SITE}: the fused Metal kernel ({key:?}) is unavailable, using the reference op \
                 chain: {error}"
            );
            false
        }
    };
    probes.insert(key, ok);
    ok
}

/// Compile and run one variant on a `[1, 8, 2, 2, 2]` input and compare it with the reference
/// chain (E2 tolerance). Sizes keep every input in `device` memory, as in production.
fn probe(stream: &Stream, key: VariantKey, norm: &ChannelNorm<'_>) -> Result<()> {
    let c = DUMMY_ELEMENTS;
    let x = seeded(&[1, c, 2, 2, 2], 0.37, 0.11)?.as_dtype(key.x)?;
    let gamma = match key.gamma {
        Some(dtype) => Some(seeded(&[c], 0.53, 0.29)?.as_dtype(dtype)?),
        None => None,
    };
    let probe_norm = match (norm, &gamma) {
        (ChannelNorm::ChannelL2 { eps, .. }, Some(gamma)) => {
            ChannelNorm::ChannelL2 { gamma, eps: *eps }
        }
        (ChannelNorm::Pixel { eps }, None) => ChannelNorm::Pixel { eps: *eps },
        _ => {
            return Err(Error::Msg(
                "vae_norm_silu_pad: probe norm/gamma mismatch".into(),
            ))
        }
    };
    let cache = if key.cache {
        Some(crate::nn::silu(&seeded(&[1, c, 1, 2, 2], 0.71, -0.4)?)?)
    } else {
        None
    };
    let spec = PadSpec {
        temporal: if key.replicate {
            TemporalPad::Replicate { front: 1, back: 1 }
        } else {
            TemporalPad::Zeros { front: 1, back: 0 }
        },
        cache: cache.as_ref(),
        ph: 1,
        pw: 1,
    };
    let tail = i32::from(key.tail);
    let plan = plan_for(&x, &probe_norm, &spec, tail, true)
        .map_err(|r| Error::Msg(format!("probe refused: {}", r.label())))?;
    let denom = probe_norm.denominator(&x)?;
    let got = launch_metal(&x, &denom, &probe_norm, &spec, &plan, stream)?;
    let want = reference_conv_input(&x, &probe_norm, &spec, tail)?;
    let mut checks = vec![(got.ndhwc, want.ndhwc)];
    match (got.tail, want.tail) {
        (Some(g), Some(w)) => checks.push((g, w)),
        (None, None) => {}
        _ => return Err(Error::Msg("vae_norm_silu_pad: probe tail mismatch".into())),
    }
    // Evaluate through the fallible path first: a JIT compile failure surfaces here as an error,
    // not as a panic inside the comparison below.
    mlx_rs::transforms::eval(checks.iter().flat_map(|(g, w)| [g, w]))?;
    for (g, w) in &checks {
        let rel = crate::nn::max_rel_diff(g, w);
        if rel.is_nan() || rel > TOLERANCE_F32 {
            return Err(Error::Msg(format!(
                "probe output differs from the reference chain (max rel {rel:e})"
            )));
        }
    }
    Ok(())
}

/// E2's declared f32 tolerance for an exact kernel: max relative error ≤ 1e-5. The one gate of the
/// parity tests, the real-weight proofs and the probe (every route returns f32, so no other dtype
/// bound applies); bit identity is reported, not asserted.
pub const TOLERANCE_F32: f32 = 1e-5;

/// Deterministic smooth test data `sin(i·scale + offset)·2` (signed, no RNG).
fn seeded(shape: &[i32], scale: f32, offset: f32) -> Result<Array> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * scale + offset).sin() * 2.0)
        .collect();
    Ok(Array::from_slice(&data, shape))
}

#[doc(hidden)]
pub mod parity;

#[cfg(test)]
mod tests;
