//! Gated DeltaNet linear attention — the recurrence (story sc-7627).
//!
//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) interleaves 3 **Gated DeltaNet**
//! linear-attention layers with 1 gated full-attention layer. Unlike softmax attention over a
//! growing KV cache, a linear layer carries a **fixed-size recurrent state** `S ∈ [Dv, Dk]` per head
//! and updates it with the gated delta rule each step — so it costs O(1) memory in sequence length.
//!
//! This module ports the **ops path** of `mlx_lm.models.gated_delta` (`gated_delta_ops` /
//! `_gated_delta_step_ops`) — the sequential reference the MLX engine itself falls back to off-GPU —
//! into `mlx-rs`. The recurrence is validated bit-for-bit against that reference via an embedded
//! numeric fixture (see the tests). The per-step update, for head state `S` (decayed by the gate `g`,
//! `β` the delta strength):
//!
//! ```text
//!   S      = S · g                          # forget (per-head scalar decay)
//!   kv_mem = (S · kᵀ) summed over Dk         # what the current key already recalls  → [Dv]
//!   Δ      = (v − kv_mem) · β                # the correction to write               → [Dv]
//!   S      = S + Δ ⊗ k                        # delta-rule outer-product write         → [Dv, Dk]
//!   y      = (S · qᵀ) summed over Dk          # read out with the query                → [Dv]
//! ```
//!
//! The gate and delta strength come from the layer's learned projections via [`compute_g`] (`g =
//! exp(−exp(A_log) · softplus(a + dt_bias))`) and `β = sigmoid(b)`; the surrounding short-conv,
//! normalisation, and in/out projections (the full layer) build on this in the layer story (sc-7628).
//! GQA is handled by repeating each of the `Hk` key/query heads to the `Hv` value heads.

use mlx_rs::fast::MetalKernel;
use mlx_rs::nn::{silu, softplus};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, exp, multiply, subtract, sum_axis};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype, Stream};

use crate::error::Result;

/// The per-step gate `g = exp(−exp(A_log) · softplus(a + dt_bias))` (a faithful port of
/// `mlx_lm.models.gated_delta.compute_g`). `a` is `[B, T, Hv]` (the gating projection), `A_log` and
/// `dt_bias` are per-value-head `[Hv]`. The inner exponentials are evaluated in f32 (matching the
/// reference's `.astype(float32)`) and the result is cast back to `a`'s dtype.
pub fn compute_g(a: &Array, a_log: &Array, dt_bias: &Array) -> Result<Array> {
    let a32 = a.as_dtype(Dtype::Float32)?;
    let dt32 = dt_bias.as_dtype(Dtype::Float32)?;
    let al32 = a_log.as_dtype(Dtype::Float32)?;
    let sp = softplus(&add(&a32, &dt32)?)?; // softplus(a + dt_bias)            [B,T,Hv]
    let coeff = multiply(&exp(&al32)?, &sp)?; // exp(A_log) · softplus(...)      [B,T,Hv]
    let g = exp(&coeff.negative()?)?; // exp(−coeff)
    Ok(g.as_dtype(a.dtype())?)
}

/// Run the gated delta recurrence over a `[B, T, ·]` chunk — the entry point every Qwen35-family
/// linear layer reaches (sc-24443).
///
/// Shapes: `q`, `k` are `[B, T, Hk, Dk]`; `v` is `[B, T, Hv, Dv]`; `g` (the per-step gate from
/// [`compute_g`]) and `beta` are `[B, T, Hv]`; `state` (the carried recurrent state, or `None` to
/// start from zeros) is `[B, Hv, Dv, Dk]`. Returns the per-step output `y` `[B, T, Hv, Dv]` (in
/// `q`'s dtype) and the final `state` `[B, Hv, Dv, Dk]` (in the carried state's dtype, or `q`'s
/// when starting from zeros) — feed `state` back in for the next chunk / decode step.
///
/// GQA: when `Hv > Hk` each key/query head pairs with `Hv / Hk` value heads (`Hv` must be a
/// multiple of `Hk`).
///
/// Dispatch (every route matches [`gated_delta_recurrence_ops`] — see the tests' error budgets;
/// the kernel and chunked routes accumulate in f32 whatever the input dtype, the op route in the
/// inputs' dtype, which Qwen35 makes f32):
/// - **GPU, any `T`** — decode (`T = 1`), speculative verify (`T = M`) and prefill: the fused
///   Metal recurrence ([`gated_delta_kernel`]), one dispatch per [`KERNEL_MAX_STEPS`] tokens with
///   the state in registers. It is measured faster than the chunkwise form on Metal at every
///   length (release, Qwen3.6-27B linear-layer dims `Hk = 16, Hv = 48, Dk = Dv = 128`: 1.7 ms vs
///   2.7 ms at 512 tokens, 7.3 ms vs 11.3 ms at 2048, against 40 / 161 ms for the op loop) and it
///   is exact f32, whereas MLX runs f32 GEMMs as TF32 on NAX GPUs (M5+), which puts the
///   chunkwise form ~2e-3 off the reference there.
/// - **CPU stream, `T ≥ `[`CHUNKED_PREFILL_MIN_TOKENS`]** — the chunkwise-parallel form
///   ([`gated_delta_chunked`]; a custom Metal kernel cannot run off the GPU, and the CPU GEMMs are
///   exact f32).
/// - **CPU stream, shorter `T`** — the op-by-op reference.
///
/// The recurrent state layout is unchanged by every path: `[B, Hv, Dv, Dk]`, the value-major form
/// `y = S · q` reads, so a caller that checkpoints states (the planned DeltaNet ring, sc-24435)
/// stores exactly what this returns.
pub fn gated_delta_recurrence(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    #[cfg(test)]
    if FORCE_OPS_REFERENCE.with(|f| f.get()) {
        return gated_delta_recurrence_ops(q, k, v, g, beta, state);
    }
    let t = q.shape()[1];
    // Resolve the stream once: the device check and the kernel dispatch must see the same one
    // (the task-local stream when set, which need not be on the process default device).
    let stream = Stream::task_local_or_default();
    let (y, next) = if stream_is_gpu(&stream) {
        kernel_segmented(q, k, v, g, beta, state, &stream)?
    } else if t >= CHUNKED_PREFILL_MIN_TOKENS {
        gated_delta_chunked(q, k, v, g, beta, state)?
    } else {
        return gated_delta_recurrence_ops(q, k, v, g, beta, state);
    };
    let state_dtype = state.map_or(q.dtype(), Array::dtype);
    Ok((cast_to(y, q.dtype())?, cast_to(next, state_dtype)?))
}

/// The most tokens one [`gated_delta_kernel`] dispatch runs: a longer sequence is split into
/// dispatches carrying the state, so no single command stays on the GPU long enough to approach
/// the Metal watchdog (a dispatch is ~15 ms at Qwen3.6-27B dims) however long the prompt is.
pub const KERNEL_MAX_STEPS: i32 = 4096;

/// [`gated_delta_kernel`] over `T` in dispatches of at most [`KERNEL_MAX_STEPS`] tokens.
fn kernel_segmented(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let t = q.shape()[1];
    if t <= KERNEL_MAX_STEPS {
        return kernel_on(q, k, v, g, beta, state, stream);
    }
    let mut state = state.cloned();
    let mut ys = Vec::new();
    let mut start = 0;
    while start < t {
        let end = (start + KERNEL_MAX_STEPS).min(t);
        let part = |x: &Array| slice_axis(x, 1, start, end);
        let (y, next) = kernel_on(
            &part(q)?,
            &part(k)?,
            &part(v)?,
            &part(g)?,
            &part(beta)?,
            state.as_ref(),
            stream,
        )?;
        ys.push(y);
        state = Some(next);
        start = end;
    }
    let refs: Vec<&Array> = ys.iter().collect();
    Ok((
        concatenate_axis(&refs, 1)?,
        state.expect("t > 0 ran a segment"),
    ))
}

/// The shortest sequence [`gated_delta_recurrence`] runs chunkwise on a CPU stream (one full
/// chunk); shorter ones take the op-by-op reference.
pub const CHUNKED_PREFILL_MIN_TOKENS: i32 = GDN_CHUNK;

/// Tokens per chunk of the chunkwise-parallel form (the flash-linear-attention default).
const GDN_CHUNK: i32 = 64;

/// Chunks whose intra-chunk work is batched into one graph before the carried state and outputs
/// are evaluated: bounds the prefill's transient to `GDN_SEGMENT_CHUNKS · GDN_CHUNK` tokens'
/// `[C, C]` matrices however long the prompt is (the role `EVAL_CHUNK` plays in the ops path).
const GDN_SEGMENT_CHUNKS: i32 = 8;

#[cfg(test)]
thread_local! {
    /// Test-only switch routing [`gated_delta_recurrence`] to the op-by-op reference on this thread,
    /// so a model-level test can compare the production dispatch against the reference path.
    static FORCE_OPS_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with [`gated_delta_recurrence`] forced onto the op-by-op reference (this thread only).
#[cfg(test)]
pub(crate) fn with_ops_reference<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_OPS_REFERENCE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FORCE_OPS_REFERENCE.with(|c| c.replace(true)));
    f()
}

/// Which recurrence implementation ran — recorded (test builds only) at each implementation's
/// entry, so a test can assert the route [`gated_delta_recurrence`] took, not just its numbers.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    /// One fused Metal kernel dispatch (a long run records one per segment).
    Kernel,
    /// One call of the chunkwise-parallel form.
    Chunked,
    /// One call of the op-by-op reference.
    Ops,
}

#[cfg(test)]
thread_local! {
    static ROUTES: std::cell::RefCell<Option<Vec<Route>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn record(route: Route) {
    ROUTES.with(|r| {
        if let Some(routes) = r.borrow_mut().as_mut() {
            routes.push(route);
        }
    });
}

/// Run `f`, returning every [`Route`] a recurrence implementation recorded on this thread.
#[cfg(test)]
pub(crate) fn recording_routes<R>(f: impl FnOnce() -> R) -> (R, Vec<Route>) {
    let previous = ROUTES.with(|r| r.replace(Some(Vec::new())));
    let out = f();
    let routes = ROUTES.with(|r| r.replace(previous)).unwrap_or_default();
    (out, routes)
}

/// Whether `stream` is a GPU stream — the only place a custom Metal kernel can run.
fn stream_is_gpu(stream: &Stream) -> bool {
    // SAFETY: `dev` is created and freed here; `stream` outlives both calls.
    unsafe {
        let mut dev = mlx_sys::mlx_device_new();
        let mut ty: mlx_sys::mlx_device_type = mlx_sys::mlx_device_type__MLX_CPU;
        let ok = mlx_sys::mlx_stream_get_device(&mut dev, stream.as_ptr()) == 0
            && mlx_sys::mlx_device_get_type(&mut ty, dev) == 0;
        mlx_sys::mlx_device_free(dev);
        ok && ty == mlx_sys::mlx_device_type__MLX_GPU
    }
}

fn cast_to(x: Array, dtype: Dtype) -> Result<Array> {
    Ok(if x.dtype() == dtype {
        x
    } else {
        x.as_dtype(dtype)?
    })
}

/// The fused recurrence body. One SIMD group (32 lanes) owns one `(batch, value head, value row)`
/// state row `S[dv, :]`, `ceil(Dk/32)` elements per lane kept in registers across all `T` steps;
/// each step decays it, reads `S·k` / `S·q` with a `simd_sum`, and writes the delta. GQA maps a
/// value head to its key head by index (no repeated q/k). Inputs of any float dtype are read as
/// f32; `y` and `state_out` are f32. Same per-step operation order as
/// [`gated_delta_recurrence_ops`] (the reductions differ only in summation order).
const GDN_KERNEL_SOURCE: &str = r#"
    const uint n = thread_position_in_grid.z;
    const uint b_idx = n / Hv;
    const uint hv_idx = n % Hv;
    const uint hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = (Dk + 31) / 32;
    const int lane = thread_position_in_threadgroup.x;
    const uint dv_idx = thread_position_in_grid.y;
    const int steps = t_len;

    auto q_ = q + (b_idx * steps * Hk + hk_idx) * Dk;
    auto k_ = k + (b_idx * steps * Hk + hk_idx) * Dk;
    auto v_ = v + (b_idx * steps * Hv + hv_idx) * Dv;
    auto y_ = y + (b_idx * steps * Hv + hv_idx) * Dv;
    auto g_ = g + b_idx * steps * Hv + hv_idx;
    auto beta_ = beta + b_idx * steps * Hv + hv_idx;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float st[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        const int s = n_per_t * lane + i;
        st[i] = s < Dk ? static_cast<float>(i_state[s]) : 0.0f;
    }
    for (int t = 0; t < steps; ++t) {
        const float decay = static_cast<float>(g_[0]);
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            const int s = n_per_t * lane + i;
            if (s < Dk) {
                st[i] = st[i] * decay;
                kv_mem += st[i] * static_cast<float>(k_[s]);
            }
        }
        kv_mem = simd_sum(kv_mem);
        const float delta =
            (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[0]);
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            const int s = n_per_t * lane + i;
            if (s < Dk) {
                st[i] = st[i] + static_cast<float>(k_[s]) * delta;
                out += st[i] * static_cast<float>(q_[s]);
            }
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
            y_[dv_idx] = out;
        }
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y_ += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
        const int s = n_per_t * lane + i;
        if (s < Dk) {
            o_state[s] = st[i];
        }
    }
"#;

/// The gated delta recurrence as one fused Metal kernel (decode `T = 1`, verify `T = M`, or any
/// `T`): the whole `T`-step loop runs on the GPU with the state in registers. Shapes as
/// [`gated_delta_recurrence`]; inputs may be f32/bf16/f16 and are accumulated in f32; returns
/// `(y [B,T,Hv,Dv], state [B,Hv,Dv,Dk])`, **both f32**. Dispatches on the task-local default
/// stream (else the default device's), which must be a GPU stream.
pub fn gated_delta_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    kernel_on(q, k, v, g, beta, state, &Stream::task_local_or_default())
}

/// [`gated_delta_kernel`] dispatched on `stream` (a GPU stream).
fn kernel_on(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    #[cfg(test)]
    record(Route::Kernel);
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);
    let state = match state {
        Some(s) => s.clone(),
        None => zeros_state(b, hv, dv, dk, Dtype::Float32)?,
    };
    let steps = Array::from_int(t);
    let kernel = MetalKernel::new(
        "sceneworks_gated_delta_step",
        &["q", "k", "v", "g", "beta", "state_in", "t_len"],
        &["y", "state_out"],
        GDN_KERNEL_SOURCE,
    )?;
    // Four value rows per threadgroup when they divide `Dv` (the grid is exact either way).
    let rows = if dv % 4 == 0 {
        4
    } else if dv % 2 == 0 {
        2
    } else {
        1
    };
    let mut out = kernel
        .apply()
        .inputs([q, k, v, g, beta, &state, &steps])
        .output_shape([b, t, hv, dv], Dtype::Float32)
        .output_shape([b, hv, dv, dk], Dtype::Float32)
        .template_arg("Dk", dk)
        .template_arg("Dv", dv)
        .template_arg("Hk", hk)
        .template_arg("Hv", hv)
        .grid(32, dv, b * hv)
        .thread_group(32, rows, 1)
        .run_device(stream)?;
    let next = out.pop().expect("two kernel outputs");
    let y = out.pop().expect("two kernel outputs");
    Ok((y, next))
}

/// The gated delta recurrence in its **chunkwise-parallel** form (the WY / flash-linear-attention
/// `chunk_gated_delta_rule` formulation), for prefill. Shapes as [`gated_delta_recurrence`];
/// inputs of any float dtype are computed in f32 and `(y, state)` are returned **f32**.
///
/// Within a chunk of `C` tokens starting from state `S₀`, with `Γᵢ = ∏_{j≤i} gⱼ` (a cumulative
/// sum of `ln g`), the per-token deltas `δᵢ = βᵢ(vᵢ − gᵢSᵢ₋₁kᵢ)` solve the unit-lower-triangular
/// system `(I + A) Δ = V_β − (K_β Γ) S₀ᵀ` with `Aᵢⱼ = βᵢ (Γᵢ/Γⱼ) kᵢ·kⱼ` (`i > j`), so with
/// `T = (I + A)⁻¹`, `U = T V_β` and `W = T (K_β Γ)` (all chunks at once):
///
/// ```text
///   Δ    = U − W S₀ᵀ
///   Y    = (Q Γ) S₀ᵀ + ((Q Kᵀ) ⊙ Γᵢ/Γⱼ, j ≤ i) Δ
///   S_C  = Γ_C S₀ + Δᵀ (K Γ_C/Γ)
/// ```
///
/// Only the `S₀ → S_C` hand-off is sequential (one step per chunk, not per token). `T` is built
/// by block doubling — `T₂ₛ = Tₛ − Tₛ Eₛ Tₛ`, `Eₛ` the part of `A` inside the `2s`-diagonal
/// blocks but outside the `s`-diagonal ones — which is exact block forward substitution. Every
/// exponent is `≤ 0` (`g ∈ (0, 1]`), so nothing overflows; a gate that underflowed to 0 is
/// clamped to the smallest normal f32 before its log.
pub fn gated_delta_chunked(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    #[cfg(test)]
    record(Route::Chunked);
    let f32 = Dtype::Float32;
    let c = GDN_CHUNK;
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);
    let (q, k) = if hv != hk {
        let r = hv / hk;
        (repeat_heads(q, r)?, repeat_heads(k, r)?)
    } else {
        (q.clone(), k.clone())
    };

    // Pad T to whole chunks with inert steps (g = 1, β = 0, k = q = v = 0 leave the state as is),
    // then lay every tensor out per head and chunk: [B, Hv, nC, C, ·].
    let n_chunks = (t + c - 1) / c;
    let pad = n_chunks * c - t;
    let heads_first = |x: &Array, fill: f32, d: i32| -> Result<Array> {
        let x = x.as_dtype(f32)?;
        let x = if pad > 0 {
            let mut shape = x.shape().to_vec();
            shape[1] = pad;
            let filler = mlx_rs::ops::full::<f32>(&shape, Array::from_f32(fill))?;
            concatenate_axis(&[&x, &filler], 1)?
        } else {
            x
        };
        if d == 0 {
            // [B, Tp, Hv] → [B, Hv, nC, C]
            Ok(x.transpose_axes(&[0, 2, 1])?
                .reshape(&[b, hv, n_chunks, c])?)
        } else {
            // [B, Tp, Hv, D] → [B, Hv, nC, C, D]
            Ok(x.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, hv, n_chunks, c, d])?)
        }
    };
    let q = heads_first(&q, 0.0, dk)?;
    let k = heads_first(&k, 0.0, dk)?;
    let v = heads_first(v, 0.0, dv)?;
    let g = heads_first(g, 1.0, 0)?;
    let beta = heads_first(beta, 0.0, 0)?;

    let masks = ChunkMasks::new(c)?;
    let mut state = match state {
        Some(s) => s.as_dtype(f32)?,
        None => zeros_state(b, hv, dv, dk, f32)?,
    };
    let mut ys: Vec<Array> = Vec::with_capacity(n_chunks as usize);
    let mut seg_start = 0;
    while seg_start < n_chunks {
        let n = GDN_SEGMENT_CHUNKS.min(n_chunks - seg_start);
        let seg = |x: &Array| -> Result<Array> { slice_axis(x, 2, seg_start, seg_start + n) };
        let (qc, kc, vc, gc, bc) = (seg(&q)?, seg(&k)?, seg(&v)?, seg(&g)?, seg(&beta)?);

        // Intra-chunk work, batched over the segment's chunks: [B, Hv, n, C, ·].
        let log_g = mlx_rs::ops::log(&mlx_rs::ops::maximum(
            &gc,
            Array::from_f32(f32::MIN_POSITIVE),
        )?)?;
        let cum = log_g.cumsum(-1, None, None)?; // ln Γᵢ           [B,Hv,n,C]
        let gamma = exp(&cum)?; // Γᵢ                              [B,Hv,n,C]
        let diff = subtract(&cum.expand_dims(-1)?, &cum.expand_dims(-2)?)?; // ln Γᵢ − ln Γⱼ
        let decay = multiply(
            &exp(&mlx_rs::ops::minimum(&diff, Array::from_f32(0.0))?)?,
            &masks.lower,
        )?; // Γᵢ/Γⱼ for j ≤ i, else 0                            [B,Hv,n,C,C]
        let k_t = kc.swap_axes(-1, -2)?;
        let kb = multiply(&kc, &bc.expand_dims(-1)?)?; // β k
        let a = multiply(
            &multiply(&mlx_rs::ops::matmul(&kb, &k_t)?, &decay)?,
            &masks.strict,
        )?; // Aᵢⱼ, i > j
        let tinv = unit_lower_inverse(&a, &masks)?; // (I + A)⁻¹
        let w = mlx_rs::ops::matmul(&tinv, &multiply(&kb, &gamma.expand_dims(-1)?)?)?; // [..,C,Dk]
        let u = mlx_rs::ops::matmul(&tinv, &multiply(&vc, &bc.expand_dims(-1)?)?)?; // [..,C,Dv]
        let q_gamma = multiply(&qc, &gamma.expand_dims(-1)?)?; // Q Γ
        let qk = multiply(&mlx_rs::ops::matmul(&qc, &k_t)?, &decay)?; // causal (Q Kᵀ) ⊙ Γᵢ/Γⱼ
        let last = slice_axis(&cum, -1, c - 1, c)?; // ln Γ_C          [B,Hv,n,1]
        let k_tail = multiply(&kc, &exp(&subtract(&last, &cum)?)?.expand_dims(-1)?)?; // K Γ_C/Γ
        let chunk_decay = exp(&last)?; // Γ_C                          [B,Hv,n,1]

        // The sequential hand-off, one step per chunk.
        for i in 0..n {
            let pick = |x: &Array| -> Result<Array> {
                Ok(slice_axis(x, 2, i, i + 1)?.squeeze_axes(&[2])?)
            };
            let s_t = state.swap_axes(-1, -2)?; // S₀ᵀ [B,Hv,Dk,Dv]
            let delta = subtract(&pick(&u)?, &mlx_rs::ops::matmul(&pick(&w)?, &s_t)?)?; // [B,Hv,C,Dv]
            let y = add(
                &mlx_rs::ops::matmul(&pick(&q_gamma)?, &s_t)?,
                &mlx_rs::ops::matmul(&pick(&qk)?, &delta)?,
            )?; // [B,Hv,C,Dv]
            let carried = multiply(&state, &pick(&chunk_decay)?.expand_dims(-1)?)?; // Γ_C S₀
            state = add(
                &carried,
                &mlx_rs::ops::matmul(&delta.swap_axes(-1, -2)?, &pick(&k_tail)?)?,
            )?; // + Δᵀ (K Γ_C/Γ)
            ys.push(y);
        }
        seg_start += n;
        if seg_start < n_chunks {
            // Materialize this segment's outputs and the carried state so its [C, C] transients
            // free before the next segment is built.
            eval(ys.iter().chain(std::iter::once(&state)))?;
        }
    }
    let refs: Vec<&Array> = ys.iter().collect();
    let y = concatenate_axis(&refs, 2)?; // [B,Hv,Tp,Dv]
    let y = slice_axis(&y, 2, 0, t)?.transpose_axes(&[0, 2, 1, 3])?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// Constant `[C, C]` masks of the chunkwise form, plus the block-doubling bands.
struct ChunkMasks {
    /// `1` where `j ≤ i`.
    lower: Array,
    /// `1` where `j < i`.
    strict: Array,
    /// The identity.
    eye: Array,
    /// For each doubling level `s` (1, 2, 4, …, C/2): `1` where `i, j` share a `2s` block but
    /// not an `s` block.
    bands: Vec<Array>,
}

impl ChunkMasks {
    fn new(c: i32) -> Result<Self> {
        let n = c as usize;
        let build = |f: &dyn Fn(usize, usize) -> bool| -> Array {
            let data: Vec<f32> = (0..n * n)
                .map(|x| if f(x / n, x % n) { 1.0 } else { 0.0 })
                .collect();
            Array::from_slice(&data, &[c, c])
        };
        let mut bands = Vec::new();
        let mut s = 1usize;
        while s < n {
            bands.push(build(&|i, j| i / (2 * s) == j / (2 * s) && i / s != j / s));
            s *= 2;
        }
        Ok(Self {
            lower: build(&|i, j| j <= i),
            strict: build(&|i, j| j < i),
            eye: build(&|i, j| i == j),
            bands,
        })
    }
}

/// `(I + A)⁻¹` for strictly-lower-triangular `A` `[…, C, C]` by block doubling:
/// `T₁ = I`, `T₂ₛ = Tₛ − Tₛ (A ⊙ bandₛ) Tₛ` — exact, since `bandₛ` holds exactly the one
/// off-diagonal block each `2s` block adds and `Eₛ Tₛ Eₛ = 0`.
fn unit_lower_inverse(a: &Array, masks: &ChunkMasks) -> Result<Array> {
    let mut inv = broadcast_to(&masks.eye, a.shape())?;
    for band in &masks.bands {
        let e = multiply(a, band)?;
        let correction = mlx_rs::ops::matmul(&mlx_rs::ops::matmul(&inv, &e)?, &inv)?;
        inv = subtract(&inv, &correction)?;
    }
    Ok(inv)
}

/// `x[.., start..end, ..]` along `axis` (negative counts from the end).
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    let arr = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&arr, axis)?)
}

/// The op-by-op gated delta recurrence — a faithful port of the `mlx_lm.models.gated_delta` ops
/// path (`gated_delta_ops` / `_gated_delta_step_ops`), one graph node group per token. It is the
/// **numeric reference** the fused kernel and the chunked form are tested against, and the path
/// [`gated_delta_recurrence`] takes off the GPU. Shapes as [`gated_delta_recurrence`]; the math runs
/// in the inputs' dtype.
pub fn gated_delta_recurrence_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    #[cfg(test)]
    record(Route::Ops);
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);

    // GQA: repeat each of the Hk key/query heads to the Hv value heads (contiguous, matching
    // `mx.repeat(q, Hv/Hk, axis=-2)`): expand → broadcast → reshape, the engine's GQA idiom.
    let (q, k) = if hv != hk {
        let r = hv / hk;
        (repeat_heads(q, r)?, repeat_heads(k, r)?)
    } else {
        (q.clone(), k.clone())
    };

    let mut state = match state {
        Some(s) => s.clone(),
        None => zeros_state(b, hv, dv, dk, q.dtype())?,
    };

    // The recurrence is sequential, so the whole `t`-step graph is built before the caller's single
    // `eval` at the end of the forward. Every step eagerly allocates index arrays (one Metal buffer
    // each, via `slice_time`'s gather) that stay live until that eval — so on a long sequence the live
    // buffer count grows without bound and trips the Metal allocator's resource limit, surfacing as a
    // spurious "expected a non-empty mlx_array" from the next gather (a high-res image expands to
    // thousands of vision tokens × the decoder's many linear layers). Force the outputs + carried
    // state every `EVAL_CHUNK` steps so MLX materializes and frees the chunk's transients, bounding
    // peak buffers to one chunk. Decode (`t == 1`) and short prefills never reach the cadence, so their
    // graphs — and the per-step sync cost `sc-7469` warned against — are unchanged.
    const EVAL_CHUNK: i32 = 256;
    let mut ys: Vec<Array> = Vec::with_capacity(t as usize);
    let mut flushed = 0usize;
    for ti in 0..t {
        let qt = slice_time(&q, ti)?; // [B,Hv,Dk]
        let kt = slice_time(&k, ti)?; // [B,Hv,Dk]
        let vt = slice_time(v, ti)?; // [B,Hv,Dv]
        let gt = slice_time(g, ti)?; // [B,Hv]
        let bt = slice_time(beta, ti)?; // [B,Hv]
        let (y, next) = delta_step(&qt, &kt, &vt, &gt, &bt, &state, b, hv, dk, dv)?;
        state = next;
        ys.push(y.expand_dims(1)?); // [B,1,Hv,Dv]
        if (ti + 1) % EVAL_CHUNK == 0 {
            // Force this chunk's outputs + the carried state (which transitively pins every step's
            // index/intermediate arrays) so they free before the next chunk.
            eval(ys[flushed..].iter().chain(std::iter::once(&state)))?;
            flushed = ys.len();
        }
    }
    let refs: Vec<&Array> = ys.iter().collect();
    let y = concatenate_axis(&refs, 1)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// Causal depthwise short convolution over `[B, S, C]` with per-channel kernel `weight` `[C, K]`
/// (the HF/MLX depthwise `Conv1d`, no bias), left-seeded by `conv_state` `[B, K-1, C]` (the previous
/// step's tail). Returns `(silu(conv) [B,S,C], new_conv_state [B,K-1,C])` — a port of the Qwen3-Next
/// short-conv path: `out[b,s,c] = silu(Σ_j weight[c,j] · concat(conv_state, x)[b, s+j, c])`. Mixing
/// q/k/v through this 1-D conv before the recurrence is what gives Gated DeltaNet its local context.
pub fn causal_depthwise_conv(
    x: &Array,
    weight: &Array,
    conv_state: &Array,
) -> Result<(Array, Array)> {
    let xs = x.shape();
    let (s, c) = (xs[1], xs[2]);
    let kk = weight.shape()[1]; // kernel size K (weight is [C, K])
    let cat = concatenate_axis(&[conv_state, x], 1)?; // [B, S+K-1, C]
    let mut acc: Option<Array> = None;
    for j in 0..kk {
        let window = slice_seq(&cat, j, j + s)?; // cat[:, j:j+S, :] → [B,S,C]
        let wj = weight
            .take_axis(Array::from_slice(&[j], &[1]), 1)? // weight[:, j] → [C,1]
            .reshape(&[1, 1, c])?; // → [1,1,C] (broadcast over B,S)
        let term = multiply(&window, &wj)?;
        acc = Some(match acc {
            None => term,
            Some(a) => add(&a, &term)?,
        });
    }
    let out = silu(acc.expect("conv kernel size must be >= 1"))?; // [B,S,C]
    let new_state = slice_seq(&cat, s, s + kk - 1)?; // last K-1 of conv_in → [B,K-1,C]
    Ok((out, new_state))
}

/// Gated RMSNorm (`Qwen3NextRMSNormGated`): `rms_norm(x, weight, eps) · silu(gate)`. Applied to the
/// delta-net output before the out-projection (`x`, `gate`, and `weight` share the head-value dim).
pub fn rms_norm_gated(x: &Array, weight: &Array, gate: &Array, eps: f32) -> Result<Array> {
    let normed = mlx_rs::fast::rms_norm(x, weight, eps)?;
    Ok(multiply(&normed, &silu(gate)?)?)
}

/// The recurrent state of one Gated DeltaNet layer — the linear-attention analog of a KV-cache slot
/// (the Mamba/SSM cache). It holds the short-conv tail `conv_state` `[B, K-1, conv_dim]` and the
/// delta-rule `ssm_state` `[B, Hv, Dv, Dk]`, both **fixed size** in sequence length (unlike the
/// growing KV cache). A hybrid decoder keeps one of these per linear layer alongside a
/// [`KvCache`](super::KvCache) per full-attention layer (the decoder assembles the mixed list).
#[derive(Clone, Debug, Default)]
pub struct DeltaNetCache {
    /// The short-conv history (previous `K-1` tokens), or `None` before the first step.
    pub conv_state: Option<Array>,
    /// The delta-rule recurrent state, or `None` before the first step.
    pub ssm_state: Option<Array>,
    offset: i32,
}

impl DeltaNetCache {
    /// An empty cache (no conv history, zero recurrent state).
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions consumed so far (the linear-layer analog of [`KvCache::offset`](super::KvCache::offset)).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Store the post-step `(conv_state, ssm_state)` and advance the position by `step` tokens.
    pub fn update(&mut self, conv_state: Array, ssm_state: Array, step: i32) {
        self.conv_state = Some(conv_state);
        self.ssm_state = Some(ssm_state);
        self.offset += step;
    }

    /// Drop all state, returning the cache to its freshly-constructed condition.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
        self.offset = 0;
    }
}

/// Slice `x` `[B, L, ...]` to `x[:, start:end, ...]` (a contiguous range along the sequence axis).
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    let arr = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&arr, 1)?)
}

/// One recurrent step (`_gated_delta_step_ops`). `q`,`k` `[B,Hv,Dk]`; `v` `[B,Hv,Dv]`; `g`,`beta`
/// `[B,Hv]`; `state` `[B,Hv,Dv,Dk]`. Returns `(y [B,Hv,Dv], new_state [B,Hv,Dv,Dk])`.
#[allow(clippy::too_many_arguments)]
fn delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    b: i32,
    hv: i32,
    dk: i32,
    dv: i32,
) -> Result<(Array, Array)> {
    let decay = g.reshape(&[b, hv, 1, 1])?; // [B,Hv,1,1]
    let state = multiply(state, &decay)?; // S · g
    let k_r = k.reshape(&[b, hv, 1, dk])?; // [B,Hv,1,Dk]
    let kv_mem = sum_axis(&multiply(&state, &k_r)?, -1, false)?; // (S·k).sum(Dk) → [B,Hv,Dv]
    let delta = multiply(&subtract(v, &kv_mem)?, &beta.reshape(&[b, hv, 1])?)?; // (v−kv)·β → [B,Hv,Dv]
    let state = add(&state, &multiply(&k_r, &delta.reshape(&[b, hv, dv, 1])?)?)?; // S + Δ⊗k
    let q_r = q.reshape(&[b, hv, 1, dk])?;
    let y = sum_axis(&multiply(&state, &q_r)?, -1, false)?; // (S·q).sum(Dk) → [B,Hv,Dv]
    Ok((y, state))
}

/// Repeat each head of `x` `[B,T,H,D]` `r` times along the head axis (contiguous), giving
/// `[B,T,H·r,D]` — the GQA expansion (`mx.repeat(x, r, axis=-2)`).
fn repeat_heads(x: &Array, r: i32) -> Result<Array> {
    let s = x.shape();
    let (b, t, h, d) = (s[0], s[1], s[2], s[3]);
    let expanded = x.expand_dims(3)?; // [B,T,H,1,D]
    let broad = broadcast_to(&expanded, &[b, t, h, r, d])?; // [B,T,H,r,D]
    Ok(broad.reshape(&[b, t, h * r, d])?) // [B,T,H·r,D]
}

/// Take time index `t` along axis 1 of `[B, T, ...]`, dropping that axis.
fn slice_time(x: &Array, t: i32) -> Result<Array> {
    let idx = Array::from_slice(&[t], &[1]);
    let picked = x.take_axis(&idx, 1)?; // [B,1,...]
    let mut shape: Vec<i32> = picked.shape().to_vec();
    shape.remove(1);
    Ok(picked.reshape(&shape)?)
}

/// A zero recurrent state `[B, Hv, Dv, Dk]` in `dtype`.
fn zeros_state(b: i32, hv: i32, dv: i32, dk: i32, dtype: Dtype) -> Result<Array> {
    let n = (b * hv * dv * dk) as usize;
    Ok(Array::from_slice(&vec![0.0f32; n], &[b, hv, dv, dk]).as_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::sigmoid;

    // Numeric oracle generated from `mlx_lm.models.gated_delta.gated_delta_update(..., use_kernel=
    // False)` (the ops reference) on seeded inputs — B=1, T=3, Hk=2, Hv=4 (GQA ×2), Dk=2, Dv=2.
    // Regenerate with the venv MLX: see story sc-7627.
    const Q: &[f32] = &[
        -0.8805123, -0.2520431, 0.6974789, -0.8069373, 0.5367976, -0.5163696, 0.4404979, -0.721925,
        0.7900037, -0.5997525, 0.0088437, -0.2685511,
    ];
    const K: &[f32] = &[
        -0.5530653, -0.0336156, 0.34984, -0.9507952, 0.3385786, 0.7130073, 0.4130355, 0.7005113,
        -0.9424242, -0.8679754, 0.0911343, -0.1635016,
    ];
    const V: &[f32] = &[
        0.9033704, 0.9242766, -0.5594231, 0.5815381, -0.9665546, 0.8363392, 0.8484675, -0.1261052,
        -0.8719618, 0.0562458, 0.8790255, 0.7971791, -0.8360307, -0.2071904, -0.6701862,
        -0.7691332, -0.2697922, -0.7519733, 0.0098643, -0.2587476, 0.5248392, 0.9719371,
        -0.0304079, -0.2898675,
    ];
    const A: &[f32] = &[
        -0.6942793, -1.9834223, 1.6954608, 1.8882596, 1.5180013, 1.9647338, -0.7497661, -1.9337821,
        -1.3342639, 1.7648504, -0.3198055, -1.4922521,
    ];
    const B: &[f32] = &[
        -1.6042936, 1.7927692, 0.1977825, 0.2890182, 0.152185, -0.4371433, 0.8649859, 0.2619474,
        -1.2190499, -1.3681375, -1.4745429, 1.3650055,
    ];
    const A_LOG: &[f32] = &[2.0919125, 1.5201275, 2.7469416, 0.104467];
    const DT_BIAS: &[f32] = &[-0.2865368, -0.2390987, 0.5489618, 0.1692053];
    const EXP_Y: &[f32] = &[
        0.0749167, 0.0766504, -0.2376069, 0.2469999, -0.5368805, 0.4645513, 0.490568, -0.0729116,
        0.0874512, -0.0056413, -0.0642828, -0.0583461, 0.1904479, 0.0472361, 0.4258981, 0.0956538,
        0.0364119, 0.0369535, -0.0004748, 0.0117344, 0.0043713, 0.0080946, 0.1139456, 0.041226,
    ];
    const EXP_STATE: &[f32] = &[
        0.0431024, -0.0039364, 0.1626127, 0.1525815, -0.0018656, -0.0016657, 0.0495011, 0.0456383,
        0.0089079, -0.015984, 0.0164975, -0.0295984, 0.0197504, -0.4236471, -0.1824342, -0.1595202,
    ];

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn recurrence_matches_python_ops_reference() {
        let q = Array::from_slice(Q, &[1, 3, 2, 2]);
        let k = Array::from_slice(K, &[1, 3, 2, 2]);
        let v = Array::from_slice(V, &[1, 3, 4, 2]);
        let a = Array::from_slice(A, &[1, 3, 4]);
        let b_raw = Array::from_slice(B, &[1, 3, 4]);
        let a_log = Array::from_slice(A_LOG, &[4]);
        let dt_bias = Array::from_slice(DT_BIAS, &[4]);

        // beta = sigmoid(b); g = compute_g(a, A_log, dt_bias) — exactly what gated_delta_update does.
        let beta = sigmoid(&b_raw).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        let (y, state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        assert_eq!(y.shape(), &[1, 3, 4, 2]);
        assert_eq!(state.shape(), &[1, 4, 2, 2]);

        let yh = y
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let sh = state
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!(
            max_abs_diff(&yh, EXP_Y) < 1e-4,
            "y diff {}",
            max_abs_diff(&yh, EXP_Y)
        );
        assert!(
            max_abs_diff(&sh, EXP_STATE) < 1e-4,
            "state diff {}",
            max_abs_diff(&sh, EXP_STATE)
        );
    }

    #[test]
    fn decode_step_matches_chunked_prefill() {
        // Feeding the sequence one token at a time (carrying state) must equal one T-step call —
        // the prefill/decode equivalence the hybrid cache relies on.
        let q = Array::from_slice(Q, &[1, 3, 2, 2]);
        let k = Array::from_slice(K, &[1, 3, 2, 2]);
        let v = Array::from_slice(V, &[1, 3, 4, 2]);
        let g = Array::from_slice(
            &[
                0.6f32, 0.7, 0.8, 0.9, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.4, 0.45,
            ],
            &[1, 3, 4],
        );
        let beta = Array::from_slice(
            &[
                0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15, 0.25, 0.35,
            ],
            &[1, 3, 4],
        );

        let (y_full, s_full) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        // Step one token at a time, carrying the state.
        let pick =
            |x: &Array, t: i32| -> Array { slice_time(x, t).unwrap().expand_dims(1).unwrap() };
        let mut state: Option<Array> = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let (y, s) = gated_delta_recurrence(
                &pick(&q, t),
                &pick(&k, t),
                &pick(&v, t),
                &pick(&g, t),
                &pick(&beta, t),
                state.as_ref(),
            )
            .unwrap();
            ys.push(y);
            state = Some(s);
        }
        let refs: Vec<&Array> = ys.iter().collect();
        let y_step = concatenate_axis(&refs, 1).unwrap();

        let yf = y_full.as_slice::<f32>().to_vec();
        let ys = y_step.as_slice::<f32>().to_vec();
        let sf = s_full.as_slice::<f32>().to_vec();
        let ss = state.unwrap().as_slice::<f32>().to_vec();
        assert!(
            max_abs_diff(&yf, &ys) < 1e-5,
            "prefill vs step y: {}",
            max_abs_diff(&yf, &ys)
        );
        assert!(max_abs_diff(&sf, &ss) < 1e-5, "prefill vs step state");
    }

    #[test]
    fn compute_g_is_in_unit_interval_and_shaped() {
        // g = exp(−positive) ∈ (0, 1]: a per-head forget gate.
        let a = Array::from_slice(A, &[1, 3, 4]);
        let a_log = Array::from_slice(A_LOG, &[4]);
        let dt_bias = Array::from_slice(DT_BIAS, &[4]);
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[1, 3, 4]);
        for x in g.as_slice::<f32>() {
            assert!(*x > 0.0 && *x <= 1.0 + 1e-6, "gate out of (0,1]: {x}");
        }
    }

    // Conv oracle from MLX's depthwise `nn.Conv1d` + silu — C=3, K=4, S=2, conv_state K-1=3.
    const CW: &[f32] = &[
        0.2422637, 0.3207079, 0.3157361, 0.493552, 0.6888506, 0.2758986, -0.2604986, 0.5719915,
        -0.677193, -0.1786878, -0.1721832, 0.024104,
    ];
    const CX: &[f32] = &[
        -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096, 0.7106992,
    ];
    const CSTATE: &[f32] = &[
        0.2526282, 0.8012137, 0.0075967, -0.7815045, -0.8113322, -0.8019857, 0.0879031, 0.8514872,
        0.00599,
    ];
    const CEXP_OUT: &[f32] = &[
        -0.1465172, 0.0095216, 0.0727914, -0.1432332, -0.2082713, 0.3602719,
    ];
    const CEXP_STATE: &[f32] = &[
        0.0879031, 0.8514872, 0.00599, -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096,
        0.7106992,
    ];

    #[test]
    fn causal_conv_matches_mlx_conv1d() {
        // weight stored [C,K,1] in the checkpoint; the helper takes [C,K] (squeezed).
        let weight = Array::from_slice(CW, &[3, 4]);
        let x = Array::from_slice(CX, &[1, 2, 3]);
        let state = Array::from_slice(CSTATE, &[1, 3, 3]);
        let (out, new_state) = causal_depthwise_conv(&x, &weight, &state).unwrap();
        assert_eq!(out.shape(), &[1, 2, 3]);
        assert_eq!(new_state.shape(), &[1, 3, 3]);
        assert!(max_abs_diff(out.as_slice::<f32>(), CEXP_OUT) < 1e-5);
        // new conv_state is the last K-1 tokens of [conv_state ++ x] — exact (a slice, no arithmetic).
        assert_eq!(new_state.as_slice::<f32>().to_vec(), CEXP_STATE.to_vec());
    }

    #[test]
    fn delta_cache_tracks_state_and_offset() {
        let mut cache = DeltaNetCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none() && cache.ssm_state.is_none());
        let conv = Array::from_slice(&[0.0f32; 9], &[1, 3, 3]);
        let ssm = Array::from_slice(&[0.0f32; 16], &[1, 4, 2, 2]);
        cache.update(conv, ssm, 5);
        assert_eq!(cache.offset(), 5);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none());
    }

    /// Deterministic uniform `[-1, 1)` values (a 64-bit LCG), so the parity fixtures need no RNG state.
    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Model-shaped recurrence inputs: L2-normalized q (scaled by `1/√Dk`) and k, gates from
    /// [`compute_g`], `β = sigmoid(b)`, plus a non-zero carried state — all rounded through `dtype`
    /// so the f32 reference sees exactly the values the path under test reads.
    struct Inputs {
        q: Array,
        k: Array,
        v: Array,
        g: Array,
        beta: Array,
        state: Array,
    }

    fn inputs(dims: [i32; 6], dtype: Dtype, seed: u64) -> Inputs {
        let [b, t, hk, hv, dk, dv] = dims;
        let n = |shape: &[i32]| shape.iter().product::<i32>() as usize;
        let arr = |shape: &[i32], s: u64| Array::from_slice(&lcg(n(shape), seed ^ s), shape);
        let l2 = |x: Array| -> Array {
            let norm =
                mlx_rs::ops::sqrt(sum_axis(multiply(&x, &x).unwrap(), -1, true).unwrap() + 1e-6f32)
                    .unwrap();
            mlx_rs::ops::divide(&x, &norm).unwrap()
        };
        let round = |x: Array| x.as_dtype(dtype).unwrap();
        let q = l2(arr(&[b, t, hk, dk], 1)) * (dk as f32).powf(-0.5);
        let k = l2(arr(&[b, t, hk, dk], 2));
        let a = arr(&[b, t, hv], 4) * 3.0f32;
        let a_log = arr(&[hv], 5);
        let dt_bias = arr(&[hv], 6);
        Inputs {
            q: round(q),
            k: round(k),
            v: round(arr(&[b, t, hv, dv], 3)),
            g: round(compute_g(&a, &a_log, &dt_bias).unwrap()),
            beta: round(sigmoid(&(arr(&[b, t, hv], 7) * 2.0f32)).unwrap()),
            state: arr(&[b, hv, dv, dk], 8) * 0.5f32,
        }
    }

    /// Host copy in logical (row-major) order. `as_slice` reads raw memory, and an elementwise op
    /// keeps a permuted input's layout (the chunked `y` is a transposed view), so materialize with
    /// one (no view offset) and walk the strides.
    fn host32(x: &Array) -> Vec<f32> {
        let x = add(x.as_dtype(Dtype::Float32).unwrap(), Array::from_f32(0.0)).unwrap();
        x.eval().unwrap();
        let (shape, strides) = (x.shape().to_vec(), x.strides().to_vec());
        let data = x.as_slice::<f32>();
        let mut out = Vec::with_capacity(x.size());
        let mut idx = vec![0i32; shape.len()];
        for _ in 0..x.size() {
            let at: usize = idx
                .iter()
                .zip(&strides)
                .map(|(&i, &s)| i as usize * s)
                .sum();
            out.push(data[at]);
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    /// `(max |a − r| / max |r|, max |a − r|)` — scale-relative and absolute error against the
    /// reference `r` (never a cosine: that is scale-invariant).
    fn errors(a: &Array, r: &Array) -> (f32, f32) {
        let (a, r) = (host32(a), host32(r));
        assert_eq!(a.len(), r.len());
        assert!(a.iter().all(|x| x.is_finite()), "non-finite output");
        let abs = max_abs_diff(&a, &r);
        let scale = r.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        (abs / scale, abs)
    }

    /// The f32 op-by-op reference on the (already dtype-rounded) inputs.
    fn reference(x: &Inputs, carry: bool) -> (Array, Array) {
        let f = |a: &Array| a.as_dtype(Dtype::Float32).unwrap();
        let state = carry.then(|| f(&x.state));
        gated_delta_recurrence_ops(
            &f(&x.q),
            &f(&x.k),
            &f(&x.v),
            &f(&x.g),
            &f(&x.beta),
            state.as_ref(),
        )
        .unwrap()
    }

    type Path =
        fn(&Array, &Array, &Array, &Array, &Array, Option<&Array>) -> Result<(Array, Array)>;

    /// Run `path` against the f32 reference over `(label, dims)` × {f32, bf16, f16} × {zero, carried
    /// state}, asserting `y` and the final state within `rel` (scale-relative) and `abs`.
    fn assert_matches_reference(
        path: Path,
        name: &str,
        cases: &[(&str, [i32; 6])],
        rel: f32,
        abs: f32,
    ) {
        for &(label, dims) in cases {
            for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
                for carry in [false, true] {
                    let x = inputs(dims, dtype, dims.iter().product::<i32>() as u64);
                    let state = carry.then_some(&x.state);
                    let (y, s) = path(&x.q, &x.k, &x.v, &x.g, &x.beta, state).unwrap();
                    let (y_ref, s_ref) = reference(&x, carry);
                    assert_eq!(y.shape(), y_ref.shape());
                    assert_eq!(s.shape(), s_ref.shape());
                    let (y_rel, y_abs) = errors(&y, &y_ref);
                    let (s_rel, s_abs) = errors(&s, &s_ref);
                    eprintln!(
                        "{name} {label} {dims:?} {dtype:?} carry={carry}: y rel {y_rel:.2e} abs \
                         {y_abs:.2e} | state rel {s_rel:.2e} abs {s_abs:.2e}"
                    );
                    assert!(
                        y_rel <= rel && y_abs <= abs && s_rel <= rel && s_abs <= abs,
                        "{name} {label} {dtype:?} carry={carry}: y rel {y_rel} abs {y_abs}, \
                         state rel {s_rel} abs {s_abs} (budget rel {rel} abs {abs})"
                    );
                }
            }
        }
    }

    /// AC (sc-24443): the fused Metal recurrence matches the op-by-op reference within 1e-5
    /// scale-relative at decode (`T = 1`), verify width (`T = 4`) and prefill (`T = 512`) shapes —
    /// GQA ×2 at the Qwen3.6 head dims, a batch of two, and tiny non-multiple-of-32 head dims —
    /// on f32, bf16 and f16 inputs (accumulated in f32), from a zero and from a carried state.
    #[test]
    fn metal_kernel_matches_ops_reference() {
        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            assert_matches_reference(
                gated_delta_kernel,
                "kernel",
                &[
                    ("decode", [1, 1, 2, 4, 128, 128]),
                    ("verify", [1, 4, 2, 4, 128, 128]),
                    ("prefill", [1, 512, 2, 4, 128, 128]),
                    ("prefill", [1, 2048, 2, 4, 128, 128]),
                    ("batch2", [2, 5, 1, 2, 64, 32]),
                    ("tiny-dims", [1, 7, 2, 4, 4, 6]),
                ],
                1e-5,
                1e-5,
            )
        });
    }

    /// AC (sc-24443): the chunkwise prefill matches the per-token reference's outputs and final
    /// state at 512 and 2048 tokens (plus a batch of two at a length that is not a whole number of
    /// chunks), on f32, bf16 and f16 inputs, from a zero and from a carried state. Run on the CPU
    /// stream — the route [`gated_delta_recurrence`] gives it, where GEMMs are exact f32 (MLX's
    /// GPU GEMMs are TF32 on NAX hardware, a ~2e-3 floor for any matmul formulation).
    #[test]
    fn chunked_prefill_matches_ops_reference() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            assert_matches_reference(
                gated_delta_chunked,
                "chunked",
                &[
                    ("512", [1, 512, 2, 4, 128, 128]),
                    ("2048", [1, 2048, 2, 4, 128, 128]),
                    ("ragged", [2, 300, 1, 2, 64, 32]),
                ],
                2e-5,
                2e-5,
            )
        });
    }

    /// The dispatcher's contract: every route agrees with the reference — the GPU kernel at any
    /// length (split into dispatches past [`KERNEL_MAX_STEPS`]), and on a CPU stream (where a Metal
    /// kernel cannot run) the chunked form from [`CHUNKED_PREFILL_MIN_TOKENS`] and the op loop
    /// below it — each length takes exactly that route, and output dtypes follow the inputs (y in
    /// `q`'s dtype, state in the carried state's).
    #[test]
    fn dispatch_routes_by_device_and_length_and_keeps_dtypes() {
        let check = |t: i32, dims: [i32; 4], route: &[Route]| {
            let [hk, hv, dk, dv] = dims;
            let x = inputs([1, t, hk, hv, dk, dv], Dtype::Float32, t as u64);
            let ((y, s), routes) = recording_routes(|| {
                gated_delta_recurrence(&x.q, &x.k, &x.v, &x.g, &x.beta, Some(&x.state)).unwrap()
            });
            assert_eq!(routes, route, "t={t}");
            let (y_ref, s_ref) = reference(&x, true);
            let (ey, es) = (errors(&y, &y_ref), errors(&s, &s_ref));
            assert!(ey.0 < 2e-5 && es.0 < 2e-5, "t={t}: y {ey:?} state {es:?}");
        };
        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            for t in [1, 4, 200] {
                check(t, [2, 4, 32, 16], &[Route::Kernel]);
            }
            check(
                KERNEL_MAX_STEPS + 37,
                [1, 2, 32, 8],
                &[Route::Kernel, Route::Kernel],
            );
        });
        mlx_rs::with_new_default_stream(Stream::cpu(), || {
            for t in [3, CHUNKED_PREFILL_MIN_TOKENS - 1] {
                check(t, [2, 4, 32, 16], &[Route::Ops]);
            }
            for t in [CHUNKED_PREFILL_MIN_TOKENS, 150] {
                check(t, [2, 4, 32, 16], &[Route::Chunked]);
            }
        });

        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            let x = inputs([1, 3, 2, 4, 32, 16], Dtype::Bfloat16, 9);
            let (y, s) = gated_delta_recurrence(&x.q, &x.k, &x.v, &x.g, &x.beta, None).unwrap();
            assert_eq!((y.dtype(), s.dtype()), (Dtype::Bfloat16, Dtype::Bfloat16));
            let (_, s) =
                gated_delta_recurrence(&x.q, &x.k, &x.v, &x.g, &x.beta, Some(&x.state)).unwrap();
            assert_eq!(s.dtype(), Dtype::Float32);
        });
    }

    /// The route follows the stream ops are issued on: a task-local GPU stream runs the kernel on
    /// that stream even when the process default device is the CPU (another test may have set it —
    /// the default device is process-global).
    #[test]
    fn kernel_runs_on_the_task_local_gpu_stream_under_a_cpu_default_device() {
        let _cpu_default = crate::primitives::kv_cache::testing::CpuStream::enter();
        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            let x = inputs([1, 4, 2, 4, 32, 16], Dtype::Float32, 21);
            let ((y, s), routes) = recording_routes(|| {
                gated_delta_recurrence(&x.q, &x.k, &x.v, &x.g, &x.beta, Some(&x.state)).unwrap()
            });
            assert_eq!(routes, [Route::Kernel]);
            let (y_ref, s_ref) = reference(&x, true);
            let (ey, es) = (errors(&y, &y_ref), errors(&s, &s_ref));
            assert!(ey.0 < 1e-5 && es.0 < 1e-5, "y {ey:?} state {es:?}");
            // The public kernel entry point dispatches on the same task-local stream.
            let (y, s) =
                gated_delta_kernel(&x.q, &x.k, &x.v, &x.g, &x.beta, Some(&x.state)).unwrap();
            let (ey, es) = (errors(&y, &y_ref), errors(&s, &s_ref));
            assert!(ey.0 < 1e-5 && es.0 < 1e-5, "kernel y {ey:?} state {es:?}");
        });
    }

    /// A long prefill must not exhaust the Metal allocator. The op-by-op path builds the whole
    /// `t`-step graph before a single eval, eagerly allocating per-step index buffers; without
    /// periodic flushing a sequence past ~100k steps trips the allocator's resource limit and the
    /// next gather fails with a spurious "expected a non-empty mlx_array". `t = 130_000` clears that
    /// limit (~5 buffers/step > 499_000) and must complete because the op path flushes every
    /// `EVAL_CHUNK` steps; the production dispatch (the fused kernel, split every
    /// [`KERNEL_MAX_STEPS`] tokens) must complete too and agree with it. Marked `ignore` only for
    /// runtime (the loop is long), not flakiness — it is the direct regression guard for the
    /// high-res-image vision crash.
    #[test]
    #[ignore = "slow: 130k-step recurrence; guards the long-prefill Metal buffer-limit regression"]
    fn long_prefill_does_not_exhaust_buffer_limit() {
        let t = 130_000i32;
        let q = Array::from_slice(&vec![0.01f32; t as usize], &[1, t, 1, 1]);
        let k = Array::from_slice(&vec![0.02f32; t as usize], &[1, t, 1, 1]);
        let v = Array::from_slice(&vec![0.03f32; t as usize], &[1, t, 1, 1]);
        let g = Array::from_slice(&vec![0.9f32; t as usize], &[1, t, 1]);
        let beta = Array::from_slice(&vec![0.5f32; t as usize], &[1, t, 1]);
        let (y, state) = gated_delta_recurrence_ops(&q, &k, &v, &g, &beta, None)
            .expect("long recurrence must not exhaust the Metal allocator");
        assert_eq!(y.shape(), &[1, t, 1, 1]);
        assert_eq!(state.shape(), &[1, 1, 1, 1]);
        // Force a final materialization so a deferred allocator failure can't hide behind laziness.
        eval([&y, &state]).unwrap();
        let (y_fast, state_fast) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();
        assert!(errors(&y_fast, &y).0 < 1e-5 && errors(&state_fast, &state).0 < 1e-5);
    }
}
