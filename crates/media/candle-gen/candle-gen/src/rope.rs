//! Shared rotary / sinusoidal position helpers (sc-27286): the byte-identical bodies the Lens,
//! Qwen-Image, FLUX.2 and Wan transformers each carried. Hoisting them changes no numerics — every
//! caller keeps its own axis layout, table build and per-render cache.
//!
//! The shared RoPE apply (SW-6128): [`apply_rope`], [`apply_rope_in_place`], [`rms_norm_rope`] and
//! [`rms_norm_rope_in_place`] cover every convention in the tree through one [`RopeSpec`]
//! (interleaved or split-half pairs, BSHD or BHSD, partial rotary, f32 or native-dtype math,
//! cos/sin or complex `freqs_cis` tables via [`freqs_cis_cos_sin`]) and run a fused, strided,
//! optionally in-place CUDA kernel with the op chain ([`apply_rope_reference`],
//! [`rms_norm_rope_reference`]) as oracle and fallback.

use candle_core::{DType, Device, Result, Tensor, D};

/// The convention of [`apply_interleaved_rope`]: interleaved pairs, `[B, H, S, head_dim]`, f32 math.
pub const INTERLEAVED_BHSD_F32: RopeSpec = RopeSpec::interleaved(RopeLayout::Bhsd);

/// Apply **interleaved** RoPE (lanes `2i`/`2i+1` are the real/imag pair) to `x` `[B, H, S, head_dim]`
/// with `cos`/`sin` `[S, head_dim/2]` (or `[B, S, head_dim/2]`). The rotation runs in f32 (the tables
/// are f32) and the result is cast back to `x`'s dtype.
///
/// SW-6129: this is [`apply_rope`] with [`INTERLEAVED_BHSD_F32`] — the fused strided kernel on CUDA
/// (no f32 / contiguous copy of `x`), the shared reference chain everywhere else. It used to be
/// candle's `rope_i` on an f32 contiguous copy: on CPU the reference chain is **bit-identical** to
/// `rope_i` (both round each product, then the sum; `cpu_rope_i_equals_the_shared_interleaved_chain`
/// pins it). On CUDA, nvcc contracts `rope_i`'s `a·c − b·s` into an fma while the fused kernel
/// reproduces the chain's separately rounded products, so the two differ by at most one rounding of
/// a product — the declared tolerance of `interleaved_rope_gpu`: in f32 `|Δ| ≤ 2⁻²²·(|a·c| + |b·s|)`;
/// in bf16 / f16 that plus one output ulp (sm_120, Qwen-Image 1024² q: 0.002% of bf16 elements move
/// by one ulp).
pub fn apply_interleaved_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    apply_rope(x, cos, sin, &INTERLEAVED_BHSD_F32)
}

// ── The shared RoPE apply (SW-6128): one API for every pairing / layout / precision ─────────────
//
// `apply_rope` / `apply_rope_in_place` / `rms_norm_rope` / `rms_norm_rope_in_place` take any rank-4
// `x` (`[B, S, H, D]` or `[B, H, S, D]`, any strides with a unit-stride lane axis — a permuted view
// or q/k narrowed out of a fused QKV projection), `cos` / `sin` tables and a `RopeSpec` (pairing,
// layout, partial rotary dim, f32 or native-dtype math). On CUDA they run the fused nvrtc kernels in
// `candle_quant_kernels::media_rope`; everywhere else, and for any input those kernels do not serve,
// the reference op chains below (epic E1). The path that ran is counted on `ROPE_PATH` /
// `RMS_NORM_ROPE_PATH` with a typed reason (`candle_gen::fused_path`).

use candle_quant_kernels::fused_path::Attempt;
/// The kernel module (plan, refusals, the cuda-only kernel-level launches).
pub use candle_quant_kernels::media_rope;
pub use candle_quant_kernels::media_rope::{
    RopeLayout, RopeMath, RopePairing, RopeRefusal, RopeSpec, RMS_NORM_ROPE_PATH, ROPE_PATH,
};

/// The tables as `[Bt, S, W]` (`Bt` 1 or batch; `W` = rotary/2, one entry per pair, or rotary, one
/// per lane). Accepted: `[S, W]`, `[Bt, S, W]`, and a rank-4 table already broadcast over heads in
/// `x`'s layout (`[Bt, S, 1, W]` for BSHD, `[Bt, 1, S, W]` for BHSD). Views stay views.
fn canonical_table(t: &Tensor, spec: &RopeSpec) -> Result<Tensor> {
    match t.rank() {
        2 => t.unsqueeze(0),
        3 => Ok(t.clone()),
        4 if t.dim(spec.layout.head_axis())? == 1 => t.squeeze(spec.layout.head_axis()),
        _ => candle_core::bail!(
            "rope table {:?}: expected [S, W], [B, S, W] or a rank-4 table broadcast over heads",
            t.dims()
        ),
    }
}

/// The reference RoPE chain (the parity oracle and the path for every input the fused kernel does
/// not serve) over `x` rank 4 and canonical `[Bt, S, W]` tables: Krea's narrow / mul / stack chain
/// for interleaved pairs, the HF `x·cos + rotate_half(x)·sin` chain for split-half pairs, in f32
/// ([`RopeMath::F32`]: `x` and the tables upcast, one rounding to `x`'s dtype at the end) or in
/// `x`'s dtype ([`RopeMath::Native`]: the tables cast down). Lanes past the rotary dim pass through.
fn rope_chain(x: &Tensor, cos: &Tensor, sin: &Tensor, spec: &RopeSpec) -> Result<Tensor> {
    let dt = x.dtype();
    let (_, _, _, d) = x.dims4()?;
    let rd = spec.rotary(d);
    if rd == 0 || !rd.is_multiple_of(2) || rd > d {
        candle_core::bail!("rope: rotary dim {rd} must be even and in 2..={d}");
    }
    let half = rd / 2;
    let math = match spec.math {
        RopeMath::F32 => DType::F32,
        RopeMath::Native => dt,
    };
    // [Bt, S, W] -> broadcast over heads in x's layout.
    let table = |t: &Tensor| t.to_dtype(math)?.unsqueeze(spec.layout.head_axis());
    let (cos, sin) = (table(cos)?, table(sin)?);
    let w = cos.dim(3)?;
    if w != half && w != rd {
        candle_core::bail!("rope: table width {w} is neither rotary/2 ({half}) nor rotary ({rd})");
    }
    let xm = x.to_dtype(math)?;
    let (xr, pass) = if rd < d {
        (xm.narrow(3, 0, rd)?, Some(xm.narrow(3, rd, d - rd)?))
    } else {
        (xm, None)
    };
    let rotated = match spec.pairing {
        RopePairing::Interleaved => {
            let (b, a1, a2, _) = xr.dims4()?;
            // [.., rd] -> [.., half, 2]; the last axis holds the (even, odd) pair.
            let pairs = xr.contiguous()?.reshape((b, a1, a2, half, 2))?;
            let lane = |t: &Tensor, i: usize| -> Result<Tensor> {
                let (tb, ta1, ta2, _) = t.dims4()?;
                t.contiguous()?
                    .reshape((tb, ta1, ta2, half, 2))?
                    .narrow(4, i, 1)?
                    .squeeze(4)
            };
            let xe = pairs.narrow(4, 0, 1)?.squeeze(4)?;
            let xo = pairs.narrow(4, 1, 1)?.squeeze(4)?;
            let (ce, se, co, so) = if w == half {
                (cos.clone(), sin.clone(), cos, sin)
            } else {
                (
                    lane(&cos, 0)?,
                    lane(&sin, 0)?,
                    lane(&cos, 1)?,
                    lane(&sin, 1)?,
                )
            };
            let out_e = (xe.broadcast_mul(&ce)? - xo.broadcast_mul(&se)?)?;
            let out_o = (xe.broadcast_mul(&so)? + xo.broadcast_mul(&co)?)?;
            Tensor::stack(&[&out_e, &out_o], D::Minus1)?.reshape((b, a1, a2, rd))?
        }
        RopePairing::SplitHalf => {
            let x1 = xr.narrow(3, 0, half)?;
            let x2 = xr.narrow(3, half, half)?;
            let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
            let (cf, sf) = if w == rd {
                (cos, sin)
            } else {
                (
                    Tensor::cat(&[&cos, &cos], 3)?,
                    Tensor::cat(&[&sin, &sin], 3)?,
                )
            };
            (xr.broadcast_mul(&cf)? + rot.broadcast_mul(&sf)?)?
        }
    };
    let out = match pass {
        Some(pass) => Tensor::cat(&[&rotated, &pass], 3)?,
        None => rotated,
    };
    out.to_dtype(dt)
}

/// The reference per-head RMSNorm over `x`'s last axis: candle's `rms_norm` on the f32 upcast
/// (weight in f32) cast back ([`RopeMath::F32`], Krea's `rms_scale`), or in `x`'s dtype with the
/// weight cast down ([`RopeMath::Native`]).
fn rms_norm_chain(x: &Tensor, weight: &Tensor, eps: f64, math: RopeMath) -> Result<Tensor> {
    let dt = x.dtype();
    // candle's fused `rms_norm` records no backward op: when the activation or the norm weight is
    // on an autograd tape (the same operands the fused plan refuses as `Autograd`), take the
    // composable `rms_norm_slow` so gradients reach them (E9). Untracked calls keep the fused op,
    // so inference numerics are unchanged.
    let tracked = x.track_op() || weight.track_op();
    let norm = |x: &Tensor, w: &Tensor| {
        if tracked {
            candle_nn::ops::rms_norm_slow(x, w, eps as f32)
        } else {
            candle_nn::ops::rms_norm(x, w, eps as f32)
        }
    };
    match math {
        RopeMath::F32 => norm(
            &x.to_dtype(DType::F32)?.contiguous()?,
            &weight.to_dtype(DType::F32)?.contiguous()?,
        )?
        .to_dtype(dt),
        RopeMath::Native => norm(&x.contiguous()?, &weight.to_dtype(dt)?.contiguous()?),
    }
}

/// The reference op chain of [`apply_rope`] (always runs it; the parity oracle).
pub fn apply_rope_reference(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
) -> Result<Tensor> {
    rope_chain(
        x,
        &canonical_table(cos, spec)?,
        &canonical_table(sin, spec)?,
        spec,
    )
}

/// The reference op chain of [`rms_norm_rope`]: `rope(rms_norm(x))` (always runs it).
pub fn rms_norm_rope_reference(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
) -> Result<Tensor> {
    let normed = rms_norm_chain(x, weight, eps, spec.math)?;
    apply_rope_reference(&normed, cos, sin, spec)
}

#[cfg(feature = "cuda")]
fn rope_attempt(
    x: &Tensor,
    weight: Option<(&Tensor, f64)>,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
    in_place: bool,
) -> Result<Attempt<Tensor>> {
    use candle_quant_kernels::media_rope::{fused_rms_norm_rope, fused_rope};
    match weight {
        Some((w, eps)) => fused_rms_norm_rope(x, w, eps, cos, sin, spec, in_place),
        None => fused_rope(x, cos, sin, spec, in_place),
    }
}

#[cfg(not(feature = "cuda"))]
fn rope_attempt(
    _: &Tensor,
    _: Option<(&Tensor, f64)>,
    _: &Tensor,
    _: &Tensor,
    _: &RopeSpec,
    _: bool,
) -> Result<Attempt<Tensor>> {
    // `dispatch` never calls the attempt without `cuda` (it records `cuda_feature_off`).
    Ok(Attempt::Reference(
        candle_quant_kernels::fused_path::PathReason::CudaFeatureOff,
    ))
}

fn rope_dispatch(
    x: &Tensor,
    weight: Option<(&Tensor, f64)>,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
    in_place: bool,
) -> Result<Tensor> {
    let (cos, sin) = (canonical_table(cos, spec)?, canonical_table(sin, spec)?);
    let path = if weight.is_some() {
        &RMS_NORM_ROPE_PATH
    } else {
        &ROPE_PATH
    };
    path.dispatch(
        || rope_attempt(x, weight, &cos, &sin, spec, in_place),
        || match weight {
            Some((w, eps)) => rope_chain(&rms_norm_chain(x, w, eps, spec.math)?, &cos, &sin, spec),
            None => rope_chain(x, &cos, &sin, spec),
        },
    )
}

/// Apply RoPE to `x` (rank 4 in `spec.layout`; any strides with a unit-stride last axis) with
/// `cos` / `sin` `[S, W]`, `[1|B, S, W]` or rank 4 broadcast over heads, where `W` is
/// `rotary/2` (one angle per pair) or `rotary` (one per lane). Lanes past `spec.rotary_dim` pass
/// through. A new tensor in `x`'s shape and dtype; fused on CUDA, the reference chain elsewhere
/// (counted on [`ROPE_PATH`]).
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, spec: &RopeSpec) -> Result<Tensor> {
    rope_dispatch(x, None, cos, sin, spec, false)
}

/// [`apply_rope`] that may write the result into `x`'s own buffer: on the fused path it does (no
/// q/k-sized allocation) and returns `x`; on the reference path (CPU, an unserved input, a view
/// that overlaps itself, a gradient-tracking operand, the switch off) it returns a new tensor and
/// leaves `x` as it was. Either way the returned tensor holds the result. The caller must own every
/// element of `x`'s view (no other live tensor may read it afterwards expecting the old values).
pub fn apply_rope_in_place(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
) -> Result<Tensor> {
    rope_dispatch(x, None, cos, sin, spec, true)
}

/// Per-head RMSNorm over `x`'s last axis (`weight` `[head_dim]`, `eps`) followed by
/// [`apply_rope`] — the QK-norm + RoPE of a DiT attention, one launch on CUDA (counted on
/// [`RMS_NORM_ROPE_PATH`]). The norm covers the whole head; the rotation `spec.rotary_dim`.
pub fn rms_norm_rope(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
) -> Result<Tensor> {
    rope_dispatch(x, Some((weight, eps)), cos, sin, spec, false)
}

/// [`rms_norm_rope`] with [`apply_rope_in_place`]'s in-place contract.
pub fn rms_norm_rope_in_place(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
    cos: &Tensor,
    sin: &Tensor,
    spec: &RopeSpec,
) -> Result<Tensor> {
    rope_dispatch(x, Some((weight, eps)), cos, sin, spec, true)
}

/// How a complex `freqs_cis` table stores each pair's rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreqsCisLayout {
    /// `[.., W, 2]` — `(cos, sin)` per pair (`torch.view_as_real(freqs_cis)`).
    Pairs,
    /// `[.., W, 2, 2]` — FLUX's rotation matrix `[[cos, −sin], [sin, cos]]` per pair.
    Matrix,
}

/// `(cos, sin)` views of a complex `freqs_cis` table for [`apply_rope`] (no copy: strided views
/// the fused kernel reads in place). The leading axes are kept (`[.., S, W]`), so FLUX's
/// `[B, 1, S, W, 2, 2]` becomes `[B, 1, S, W]` for a BHSD `x`. FLUX's apply (`out = fr0·x0 +
/// fr1·x1` in `x`'s dtype) is `RopeSpec::interleaved(RopeLayout::Bhsd).with_math(RopeMath::Native)`
/// on these.
pub fn freqs_cis_cos_sin(freqs_cis: &Tensor, layout: FreqsCisLayout) -> Result<(Tensor, Tensor)> {
    let dims = freqs_cis.dims();
    let n = dims.len();
    match layout {
        FreqsCisLayout::Matrix if n >= 3 && dims[n - 1] == 2 && dims[n - 2] == 2 => {
            // m[.., 0, 0] = cos, m[.., 1, 0] = sin.
            let col = freqs_cis.narrow(n - 1, 0, 1)?.squeeze(n - 1)?;
            let cos = col.narrow(n - 2, 0, 1)?.squeeze(n - 2)?;
            let sin = col.narrow(n - 2, 1, 1)?.squeeze(n - 2)?;
            Ok((cos, sin))
        }
        FreqsCisLayout::Pairs if n >= 2 && dims[n - 1] == 2 => Ok((
            freqs_cis.narrow(n - 1, 0, 1)?.squeeze(n - 1)?,
            freqs_cis.narrow(n - 1, 1, 1)?.squeeze(n - 1)?,
        )),
        _ => candle_core::bail!("freqs_cis {dims:?} is not a {layout:?} table"),
    }
}

/// [`apply_rope`] with the angles as a complex `freqs_cis` table (see [`freqs_cis_cos_sin`]).
pub fn apply_rope_freqs_cis(
    x: &Tensor,
    freqs_cis: &Tensor,
    layout: FreqsCisLayout,
    spec: &RopeSpec,
) -> Result<Tensor> {
    let (cos, sin) = freqs_cis_cos_sin(freqs_cis, layout)?;
    apply_rope(x, &cos, &sin, spec)
}

/// Sinusoidal timestep embedding `[1, dim]`, `[cos(args) | sin(args)]` (diffusers
/// `flip_sin_to_cos = True`, `downscale_freq_shift = 0`), base 10000:
/// `args[k] = t · scale · 10000^{-k/half}`, computed in f32 in exactly that order. `scale` is `1000.0`
/// for the providers that feed a raw sigma (diffusers `Timesteps(scale=1000)`) and `1.0` for a caller
/// that pre-scales its timestep (multiplying by `1.0` is exact, so that case is bit-identical to
/// `t · freq`).
pub fn sinusoidal_timestep_embedding(
    t: f32,
    dim: usize,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let ln = 10000f32.ln();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for k in 0..half {
        let freq = (-ln * k as f32 / half as f32).exp();
        let arg = t * scale * freq;
        cos[k] = arg.cos();
        sin[k] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}

/// Three-axis (frame / height / width) RoPE table builder shared by the Qwen-Image and Lens DiTs
/// (sc-27286): the two differ only in `theta` / `axes_dim`. Frequencies are `ω_d[k] = θ^{-(2k)/d}` per
/// axis, concatenated `[ω_frame | ω_h | ω_w]` (`Σ axes_dim / 2` pairs); angles are computed host-side in
/// f32 and applied **interleaved** via [`apply_interleaved_rope`].
///
/// - **Image tokens** of grid `i` (`(lat_h, lat_w)`): the frame band uses position `i`, the height /
///   width bands the **centred** positions `h − (lat_h − lat_h/2)` / `w − (lat_w − lat_w/2)`; grids are
///   concatenated in order, each row-major.
/// - **Text tokens** at index `t`: one scalar position `txt_base + t` across every pair, with
///   `txt_base = max_i(max(lat_h_i/2, lat_w_i/2))`.
#[derive(Clone, Debug)]
pub struct AxialRope3 {
    theta: f32,
    axes_dim: [usize; 3],
    half: usize,
}

impl AxialRope3 {
    pub fn new(theta: f32, axes_dim: [usize; 3]) -> Self {
        Self {
            theta,
            axes_dim,
            half: axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The concatenated frequency vector `[ω_frame, ω_h, ω_w]` (`Σ axes_dim / 2` wide).
    fn omega(&self) -> Vec<f32> {
        let mut all = Vec::with_capacity(self.half);
        for &dim in &self.axes_dim {
            for k in 0..dim / 2 {
                all.push(1.0f32 / self.theta.powf((2 * k) as f32 / dim as f32));
            }
        }
        all
    }

    /// Image-token `(cos, sin)` `[Σ lat_h_i·lat_w_i, Σ axes_dim / 2]` over `grids` in order.
    pub fn img_cos_sin_multi(
        &self,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let (n_f, n_h) = (self.axes_dim[0] / 2, self.axes_dim[1] / 2);
        let total: usize = grids.iter().map(|(h, w)| h * w).sum();
        let mut cos = vec![0f32; total * self.half];
        let mut sin = vec![0f32; total * self.half];
        let mut off = 0usize;
        for (idx, &(lat_h, lat_w)) in grids.iter().enumerate() {
            let h_center = (lat_h - lat_h / 2) as i64;
            let w_center = (lat_w - lat_w / 2) as i64;
            for h in 0..lat_h {
                for w in 0..lat_w {
                    let row = off + h * lat_w + w;
                    let hp = h as i64 - h_center;
                    let wp = w as i64 - w_center;
                    for (j, &om) in omega.iter().enumerate() {
                        // frame band → grid index, height band → hp, width band → wp.
                        let pos = if j < n_f {
                            idx as i64
                        } else if j < n_f + n_h {
                            hp
                        } else {
                            wp
                        } as f32;
                        let a = pos * om;
                        cos[row * self.half + j] = a.cos();
                        sin[row * self.half + j] = a.sin();
                    }
                }
            }
            off += lat_h * lat_w;
        }
        Ok((
            Tensor::from_vec(cos, (total, self.half), device)?,
            Tensor::from_vec(sin, (total, self.half), device)?,
        ))
    }

    /// Text-token `(cos, sin)` `[txt_seq, Σ axes_dim / 2]` for `grids` (see the type docs for
    /// `txt_base`).
    pub fn txt_cos_sin_multi(
        &self,
        txt_seq: usize,
        grids: &[(usize, usize)],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let omega = self.omega();
        let txt_base = grids
            .iter()
            .map(|(h, w)| (h / 2).max(w / 2))
            .max()
            .unwrap_or(0) as i64;
        let mut cos = vec![0f32; txt_seq * self.half];
        let mut sin = vec![0f32; txt_seq * self.half];
        for t in 0..txt_seq {
            let pos = (txt_base + t as i64) as f32;
            for (j, &om) in omega.iter().enumerate() {
                let a = pos * om;
                cos[t * self.half + j] = a.cos();
                sin[t * self.half + j] = a.sin();
            }
        }
        Ok((
            Tensor::from_vec(cos, (txt_seq, self.half), device)?,
            Tensor::from_vec(sin, (txt_seq, self.half), device)?,
        ))
    }
}

/// A small bounded, geometry-keyed cache for per-render RoPE tables (sc-8992, sc-11201;
/// shared by sc-27286). The tables depend only on the render's fixed token geometry — never on
/// σ or the current latent — so they are identical across every denoise step. On a miss the value is
/// built and inserted; past `cap` entries the oldest is evicted (FIFO). `cap == 1` is the single-slot
/// "rebuild when the geometry changes" cache; a larger `cap` keeps alternating true-CFG geometries
/// resident. Hits clone the (Arc-backed) `V`, so a hit is byte-identical to rebuilding. `Mutex` (not
/// `RefCell`): the transformers are shared as `Arc<…>` and must stay `Send + Sync`.
pub struct GeometryCache<K, V> {
    entries: std::sync::Mutex<Vec<(K, V)>>,
    cap: usize,
}

impl<K: PartialEq, V: Clone> GeometryCache<K, V> {
    /// An empty cache holding at most `cap` (≥ 1) geometries.
    pub fn new(cap: usize) -> Self {
        assert!(cap >= 1, "a geometry cache must hold at least one entry");
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
            cap,
        }
    }

    /// Return the cached value for `key`, building + inserting it on a miss. A failed build caches
    /// nothing.
    pub fn get_or_build<E>(
        &self,
        key: K,
        build: impl FnOnce() -> std::result::Result<V, E>,
    ) -> std::result::Result<V, E> {
        let mut guard = crate::lock_recover(&self.entries);
        if let Some((_, v)) = guard.iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let v = build()?;
        self.insert(&mut guard, key, v.clone());
        Ok(v)
    }

    /// [`Self::get_or_build`] for a key that is costly to own (e.g. a per-token id list): `is_key`
    /// probes the cached entries without allocating, and `key` materializes the owned key only on a
    /// miss.
    pub fn get_or_build_by<E>(
        &self,
        is_key: impl Fn(&K) -> bool,
        key: impl FnOnce() -> K,
        build: impl FnOnce() -> std::result::Result<V, E>,
    ) -> std::result::Result<V, E> {
        let mut guard = crate::lock_recover(&self.entries);
        if let Some((_, v)) = guard.iter().find(|(k, _)| is_key(k)) {
            return Ok(v.clone());
        }
        let v = build()?;
        self.insert(&mut guard, key(), v.clone());
        Ok(v)
    }

    fn insert(&self, entries: &mut Vec<(K, V)>, key: K, value: V) {
        if entries.len() >= self.cap {
            entries.remove(0);
        }
        entries.push((key, value));
    }

    /// Drop every cached entry (e.g. after the owner moved to another device).
    pub fn clear(&self) {
        crate::lock_recover(&self.entries).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SW-6128 shared RoPE: the reference chains equal the copies they replace, bit for bit ──

    fn randn(dims: &[usize], seed: u64) -> Tensor {
        // Deterministic xorshift values in [-2, 2) (no device RNG).
        let n: usize = dims.iter().product();
        let mut s = seed | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32
            })
            .collect();
        Tensor::from_vec(v, dims, &Device::Cpu).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// `(cos, sin)` `[1, s, half]` of real angles.
    fn tables(s: usize, half: usize) -> (Tensor, Tensor) {
        let a = (randn(&[1, s, half], 99) * 3.0).unwrap();
        (a.cos().unwrap(), a.sin().unwrap())
    }

    /// Krea's pre-SW-6128 `transformer::rope::apply_interleaved_rope`, verbatim.
    fn krea_chain(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let dt = x.dtype();
        let (b, s, h, hd) = x.dims4().unwrap();
        let half = hd / 2;
        let cos = cos.contiguous().unwrap().reshape((1, s, 1, half)).unwrap();
        let sin = sin.contiguous().unwrap().reshape((1, s, 1, half)).unwrap();
        let xr = x
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((b, s, h, half, 2))
            .unwrap();
        let xe = xr.narrow(4, 0, 1).unwrap().contiguous().unwrap();
        let xe = xe.reshape((b, s, h, half)).unwrap();
        let xo = xr.narrow(4, 1, 1).unwrap().contiguous().unwrap();
        let xo = xo.reshape((b, s, h, half)).unwrap();
        let out_e = (xe.broadcast_mul(&cos).unwrap() - xo.broadcast_mul(&sin).unwrap()).unwrap();
        let out_o = (xe.broadcast_mul(&sin).unwrap() + xo.broadcast_mul(&cos).unwrap()).unwrap();
        Tensor::stack(&[&out_e, &out_o], D::Minus1)
            .unwrap()
            .reshape((b, s, h, hd))
            .unwrap()
            .to_dtype(dt)
            .unwrap()
    }

    #[test]
    fn interleaved_reference_is_krea_chain_bit_for_bit() {
        let (s, h, d) = (7, 3, 16);
        let (cos, sin) = tables(s, d / 2);
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let x = randn(&[2, s, h, d], 1).to_dtype(dt).unwrap();
            let spec = RopeSpec::interleaved(RopeLayout::Bshd);
            let got = apply_rope_reference(&x, &cos, &sin, &spec).unwrap();
            assert_eq!(got.dtype(), dt);
            assert_eq!(bits(&got), bits(&krea_chain(&x, &cos, &sin)), "{dt:?}");
            // The public entry point on CPU is the same chain.
            assert_eq!(
                bits(&apply_rope(&x, &cos, &sin, &spec).unwrap()),
                bits(&got)
            );
        }
    }

    /// The HF half-split copy (Anima / Ideogram: tables cast to x's dtype, math in x's dtype) and
    /// the FLUX.2 Pixtral copy (all f32) with full-width `[s, hd]` tables.
    #[test]
    fn split_half_reference_is_the_hf_chains_bit_for_bit() {
        let (s, h, d) = (5, 2, 12);
        let (c, sn) = tables(s, d / 2);
        let cos = Tensor::cat(&[&c, &c], 2).unwrap();
        let sin = Tensor::cat(&[&sn, &sn], 2).unwrap();
        let hf = |x: &Tensor, cos: &Tensor, sin: &Tensor| -> Tensor {
            let cos = cos.to_dtype(x.dtype()).unwrap().unsqueeze(2).unwrap();
            let sin = sin.to_dtype(x.dtype()).unwrap().unsqueeze(2).unwrap();
            let x1 = x.narrow(3, 0, d / 2).unwrap();
            let x2 = x.narrow(3, d / 2, d / 2).unwrap();
            let rot = Tensor::cat(&[&x2.neg().unwrap(), &x1], 3).unwrap();
            (x.broadcast_mul(&cos).unwrap() + rot.broadcast_mul(&sin).unwrap()).unwrap()
        };
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let x = randn(&[1, s, h, d], 2).to_dtype(dt).unwrap();
            let native = RopeSpec::split_half(RopeLayout::Bshd).with_math(RopeMath::Native);
            let got = apply_rope_reference(&x, &cos, &sin, &native).unwrap();
            assert_eq!(bits(&got), bits(&hf(&x, &cos, &sin)), "native {dt:?}");
            // f32 math == the Pixtral chain (upcast, f32, cast back).
            let f32_spec = RopeSpec::split_half(RopeLayout::Bshd);
            let got = apply_rope_reference(&x, &cos, &sin, &f32_spec).unwrap();
            let want = hf(&x.to_dtype(DType::F32).unwrap(), &cos, &sin)
                .to_dtype(dt)
                .unwrap();
            assert_eq!(bits(&got), bits(&want), "f32 {dt:?}");
            // A half-width table is the same rotation.
            let got_half = apply_rope_reference(&x, &c, &sn, &f32_spec).unwrap();
            assert_eq!(bits(&got_half), bits(&want), "half-width {dt:?}");
        }
    }

    /// FLUX.1's `apply_rope(x, freq_cis)` (rotation matrices, math in x's dtype) through the
    /// complex-table entry point.
    #[test]
    fn freqs_cis_matrix_entry_point_is_flux_apply_bit_for_bit() {
        let (b, h, s, d) = (1, 2, 6, 8);
        let (cos, sin) = tables(s, d / 2);
        // [1, s, half, 2, 2] = [[cos, -sin], [sin, cos]], then FLUX's unsqueeze(1).
        let fc = Tensor::stack(&[&cos, &sin.neg().unwrap(), &sin, &cos], 3)
            .unwrap()
            .reshape((1, s, d / 2, 2, 2))
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        for dt in [DType::F32, DType::BF16] {
            let x = randn(&[b, h, s, d], 3).to_dtype(dt).unwrap();
            let fc = fc.to_dtype(dt).unwrap();
            let flux = {
                let xr = x.reshape((b, h, s, d / 2, 2)).unwrap();
                let x0 = xr.narrow(D::Minus1, 0, 1).unwrap();
                let x1 = xr.narrow(D::Minus1, 1, 1).unwrap();
                let fr0 = fc.get_on_dim(D::Minus1, 0).unwrap();
                let fr1 = fc.get_on_dim(D::Minus1, 1).unwrap();
                (fr0.broadcast_mul(&x0).unwrap() + fr1.broadcast_mul(&x1).unwrap())
                    .unwrap()
                    .reshape((b, h, s, d))
                    .unwrap()
            };
            let spec = RopeSpec::interleaved(RopeLayout::Bhsd).with_math(RopeMath::Native);
            let got = apply_rope_freqs_cis(&x, &fc, FreqsCisLayout::Matrix, &spec).unwrap();
            assert_eq!(bits(&got), bits(&flux), "{dt:?}");
            // The pairs layout of the same angles.
            let pairs = Tensor::stack(&[&cos, &sin], 3)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let got = apply_rope_freqs_cis(&x, &pairs, FreqsCisLayout::Pairs, &spec).unwrap();
            assert_eq!(bits(&got), bits(&flux), "pairs {dt:?}");
        }
        assert!(freqs_cis_cos_sin(&cos, FreqsCisLayout::Matrix).is_err());
    }

    #[test]
    fn partial_rotary_passes_the_tail_through_and_layouts_agree() {
        let (s, h, d, rd) = (4, 3, 10, 6);
        let (cos, sin) = tables(s, rd / 2);
        let x = randn(&[2, s, h, d], 4).to_dtype(DType::BF16).unwrap();
        for pairing in [RopePairing::Interleaved, RopePairing::SplitHalf] {
            let spec = RopeSpec::new(pairing, RopeLayout::Bshd).with_rotary_dim(rd);
            let got = apply_rope_reference(&x, &cos, &sin, &spec).unwrap();
            let tail = |t: &Tensor| bits(&t.narrow(3, rd, d - rd).unwrap());
            assert_eq!(
                tail(&got),
                tail(&x),
                "{pairing:?}: lanes past rd pass through"
            );
            let full = RopeSpec::new(pairing, RopeLayout::Bshd);
            let head =
                apply_rope_reference(&x.narrow(3, 0, rd).unwrap(), &cos, &sin, &full).unwrap();
            assert_eq!(
                bits(&got.narrow(3, 0, rd).unwrap()),
                bits(&head),
                "{pairing:?}"
            );
            // The same rotation on the permuted BHSD view, then permuted back.
            let bhsd = RopeSpec::new(pairing, RopeLayout::Bhsd).with_rotary_dim(rd);
            let xt = x.transpose(1, 2).unwrap();
            let gt = apply_rope_reference(&xt, &cos, &sin, &bhsd).unwrap();
            assert_eq!(
                bits(&gt.transpose(1, 2).unwrap()),
                bits(&got),
                "{pairing:?} bhsd"
            );
        }
        // A rank-4 table broadcast over heads, and a per-batch table, are accepted.
        let spec = RopeSpec::interleaved(RopeLayout::Bshd).with_rotary_dim(rd);
        let want = apply_rope_reference(&x, &cos, &sin, &spec).unwrap();
        let c4 = cos.unsqueeze(2).unwrap();
        let s4 = sin.unsqueeze(2).unwrap();
        assert_eq!(
            bits(&apply_rope_reference(&x, &c4, &s4, &spec).unwrap()),
            bits(&want)
        );
        let cb = Tensor::cat(&[&cos, &cos], 0).unwrap();
        let sb = Tensor::cat(&[&sin, &sin], 0).unwrap();
        assert_eq!(
            bits(&apply_rope_reference(&x, &cb, &sb, &spec).unwrap()),
            bits(&want)
        );
        let bad = RopeSpec::interleaved(RopeLayout::Bshd).with_rotary_dim(7);
        assert!(apply_rope_reference(&x, &cos, &sin, &bad).is_err());
    }

    /// `rms_norm_rope_reference` is Krea's `rms_scale` (f32 rms_norm, cast back) then its RoPE.
    #[test]
    fn rms_norm_rope_reference_is_krea_norm_then_rope() {
        let (s, h, d) = (5, 4, 32);
        let (cos, sin) = tables(s, d / 2);
        let w = (randn(&[d], 5) + 1.0).unwrap();
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let x = randn(&[1, s, h, d], 6).to_dtype(dt).unwrap();
            let normed = candle_nn::ops::rms_norm(
                &x.to_dtype(DType::F32).unwrap().contiguous().unwrap(),
                &w,
                1e-5,
            )
            .unwrap()
            .to_dtype(dt)
            .unwrap();
            let want = krea_chain(&normed, &cos, &sin);
            let spec = RopeSpec::interleaved(RopeLayout::Bshd);
            let got = rms_norm_rope_reference(&x, &w, 1e-5, &cos, &sin, &spec).unwrap();
            assert_eq!(bits(&got), bits(&want), "{dt:?}");
            assert_eq!(
                bits(&rms_norm_rope(&x, &w, 1e-5, &cos, &sin, &spec).unwrap()),
                bits(&want)
            );
        }
    }

    /// On CPU every entry point runs the reference, counted with its reason, and the in-place form
    /// leaves `x` untouched (it returns the reference's new tensor).
    #[test]
    fn cpu_calls_take_the_counted_reference_and_never_write_in_place() {
        let _rope = ROPE_PATH.guard(Some(true));
        let _rms = RMS_NORM_ROPE_PATH.guard(Some(true));
        let _media = crate::fused_path::media_fused_policy_guard(Some(true));
        let expected = if cfg!(feature = "cuda") {
            "not_cuda"
        } else {
            "cuda_feature_off"
        };
        let (s, h, d) = (3, 2, 8);
        let (cos, sin) = tables(s, d / 2);
        let w = randn(&[d], 7);
        let x = randn(&[1, s, h, d], 8);
        let before = bits(&x);
        let spec = RopeSpec::interleaved(RopeLayout::Bshd);

        let start = ROPE_PATH.thread_counts();
        let out = apply_rope_in_place(&x, &cos, &sin, &spec).unwrap();
        let d1 = ROPE_PATH.thread_counts().since(&start);
        assert_eq!((d1.fused, d1.reference_for(expected)), (0, 1), "{d1}");
        assert_eq!(bits(&x), before, "x is untouched on the reference path");
        assert_eq!(
            bits(&out),
            bits(&apply_rope_reference(&x, &cos, &sin, &spec).unwrap())
        );

        let start = RMS_NORM_ROPE_PATH.thread_counts();
        rms_norm_rope_in_place(&x, &w, 1e-6, &cos, &sin, &spec).unwrap();
        let d2 = RMS_NORM_ROPE_PATH.thread_counts().since(&start);
        assert_eq!((d2.fused, d2.reference_for(expected)), (0, 1), "{d2}");
        assert_eq!(bits(&x), before);

        // The kernel switch off is recorded as `disabled` before anything else.
        ROPE_PATH.set_enabled(Some(false));
        let start = ROPE_PATH.thread_counts();
        apply_rope(&x, &cos, &sin, &spec).unwrap();
        let d3 = ROPE_PATH.thread_counts().since(&start);
        let off = if cfg!(feature = "cuda") {
            "disabled"
        } else {
            "cuda_feature_off"
        };
        assert_eq!(d3.reference_for(off), 1, "{d3}");
    }

    /// A tracked activation or norm weight takes the reference with a differentiable norm (E9):
    /// both `rms_norm_rope` entry points, in both precision branches, return the untracked value
    /// and backpropagate to `x` and `weight` exactly like a chain built from composable ops only
    /// (`rms_norm_slow` then the reference RoPE).
    #[test]
    fn tracked_rms_norm_rope_backpropagates_to_x_and_the_norm_weight() {
        let (s, h, d) = (3, 2, 8);
        let (cos, sin) = tables(s, d / 2);
        let x = candle_core::Var::from_tensor(&randn(&[1, s, h, d], 11)).unwrap();
        let w = candle_core::Var::from_tensor(&randn(&[d], 12).affine(0.25, 1.0).unwrap()).unwrap();
        let eps = 1e-6;
        let grads_of = |out: &Tensor| {
            let g = out.sqr().unwrap().sum_all().unwrap().backward().unwrap();
            [x.as_tensor(), w.as_tensor()].map(|v| {
                g.get(v)
                    .expect("a gradient reached the operand")
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
        };
        let max_diff = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max)
        };
        for pairing in [RopePairing::Interleaved, RopePairing::SplitHalf] {
            for math in [RopeMath::F32, RopeMath::Native] {
                let spec = RopeSpec {
                    math,
                    ..RopeSpec::new(pairing, RopeLayout::Bshd)
                };
                let slow = apply_rope_reference(
                    &candle_nn::ops::rms_norm_slow(x.as_tensor(), w.as_tensor(), eps as f32)
                        .unwrap(),
                    &cos,
                    &sin,
                    &spec,
                )
                .unwrap();
                let want = grads_of(&slow);
                let plain = rms_norm_rope(
                    &x.as_tensor().detach(),
                    &w.as_tensor().detach(),
                    eps,
                    &cos,
                    &sin,
                    &spec,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                for in_place in [false, true] {
                    let entry = if in_place {
                        rms_norm_rope_in_place
                    } else {
                        rms_norm_rope
                    };
                    let out = entry(x.as_tensor(), w.as_tensor(), eps, &cos, &sin, &spec).unwrap();
                    let tag = format!("{pairing:?}/{math:?}/in_place={in_place}");
                    let value = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let drift = max_diff(&value, &plain);
                    assert!(drift < 1e-5, "{tag}: slow norm drifted {drift}");
                    let got = grads_of(&out);
                    for (name, (g, r)) in ["x", "weight"].iter().zip(got.iter().zip(&want)) {
                        assert!(
                            g.iter().any(|v| *v != 0.0),
                            "{tag}: zero gradient at {name}"
                        );
                        let diff = max_diff(g, r);
                        assert!(diff < 1e-5, "{tag}: {name} gradient off by {diff}");
                    }
                }
            }
        }
    }

    #[test]
    fn sinusoidal_embedding_matches_the_inline_scaled_formula() {
        let dim = 16;
        let half = dim / 2;
        let t = 0.6180339f32;
        let got = sinusoidal_timestep_embedding(t, dim, 1000.0, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let ln = 10000f32.ln();
        for k in 0..half {
            let freq = (-ln * k as f32 / half as f32).exp();
            // The providers' original inline expression, bit for bit.
            let arg = t * 1000.0 * freq;
            assert_eq!(got[k].to_bits(), arg.cos().to_bits(), "cos lane {k}");
            assert_eq!(got[half + k].to_bits(), arg.sin().to_bits(), "sin lane {k}");
        }
        // scale = 1.0 is exactly the unscaled `t · freq` of the FLUX.2 copy.
        let unscaled = sinusoidal_timestep_embedding(t, dim, 1.0, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (k, got) in unscaled.iter().take(half).enumerate() {
            let freq = (-ln * k as f32 / half as f32).exp();
            assert_eq!(got.to_bits(), (t * freq).cos().to_bits());
        }
    }

    #[test]
    fn axial_omega_bands_start_at_theta_zero() {
        for (axes, starts) in [([16, 56, 56], [0, 8, 36]), ([8, 28, 28], [0, 4, 18])] {
            let rope = AxialRope3::new(10_000.0, axes);
            let om = rope.omega();
            assert_eq!(om.len(), axes.iter().sum::<usize>() / 2);
            for start in starts {
                assert_eq!(om[start], 1.0, "{axes:?} band at {start}");
            }
        }
    }

    #[test]
    fn axial_multi_grid_over_frame_copies_is_bitwise_the_frame_layout() {
        // The Lens builder walked `f in 0..frame` over ONE grid with frame position `f`; the shared
        // builder walks a grid list with frame position = grid index. Over `frame` copies of the grid
        // they must agree bit for bit — recompute the Lens loop inline as the reference.
        let (theta, axes, frame, lat_h, lat_w) = (10_000.0f32, [8usize, 28, 28], 3, 5, 6);
        let rope = AxialRope3::new(theta, axes);
        let (cos, sin) = rope
            .img_cos_sin_multi(&vec![(lat_h, lat_w); frame], &Device::Cpu)
            .unwrap();
        let (cos, sin) = (
            cos.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            sin.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let half = 32;
        let omega = rope.omega();
        let (h_center, w_center) = ((lat_h - lat_h / 2) as i64, (lat_w - lat_w / 2) as i64);
        for f in 0..frame {
            for h in 0..lat_h {
                for w in 0..lat_w {
                    let row = (f * lat_h * lat_w + h * lat_w + w) * half;
                    for (j, &om) in omega.iter().enumerate() {
                        let pos = if j < 4 {
                            f as i64
                        } else if j < 18 {
                            h as i64 - h_center
                        } else {
                            w as i64 - w_center
                        } as f32;
                        let a = pos * om;
                        assert_eq!(cos[row + j].to_bits(), a.cos().to_bits());
                        assert_eq!(sin[row + j].to_bits(), a.sin().to_bits());
                    }
                }
            }
        }
        let (txt_cos, _) = rope
            .txt_cos_sin_multi(2, &vec![(lat_h, lat_w); frame], &Device::Cpu)
            .unwrap();
        let txt_cos = txt_cos.to_vec2::<f32>().unwrap();
        let base = (lat_h / 2).max(lat_w / 2) as f32;
        assert_eq!(
            txt_cos[1][0].to_bits(),
            ((base + 1.0) * omega[0]).cos().to_bits()
        );
    }

    #[test]
    fn geometry_cache_hits_by_key_and_evicts_oldest_beyond_cap() {
        let cache: GeometryCache<usize, usize> = GeometryCache::new(2);
        let builds = std::cell::Cell::new(0);
        let get = |k: usize| {
            cache
                .get_or_build(k, || {
                    builds.set(builds.get() + 1);
                    Ok::<_, ()>(k * 10)
                })
                .unwrap()
        };
        assert_eq!(get(1), 10);
        assert_eq!(get(2), 20);
        assert_eq!(get(1), 10);
        assert_eq!(builds.get(), 2);
        assert_eq!(get(3), 30); // evicts 1
        assert_eq!(get(2), 20);
        assert_eq!(builds.get(), 3);
        assert_eq!(get(1), 10);
        assert_eq!(builds.get(), 4);
        let failed: GeometryCache<usize, usize> = GeometryCache::new(1);
        assert!(failed.get_or_build(7, || Err::<usize, _>("boom")).is_err());
        assert_eq!(failed.get_or_build(7, || Ok::<_, ()>(70)), Ok(70));
        failed.clear();
        assert_eq!(failed.get_or_build(7, || Ok::<_, ()>(71)), Ok(71));
    }

    /// SW-6129: [`apply_interleaved_rope`] now runs the shared chain off CUDA; on CPU that chain is
    /// bit-identical to the candle `rope_i` it replaced (f32 contiguous copy, rotate, cast back), for
    /// shared `[S, W]` and per-batch `[B, S, W]` tables, f32 / bf16 / f16 and a non-contiguous `x`.
    #[test]
    fn cpu_rope_i_equals_the_shared_interleaved_chain() {
        use candle_nn::rotary_emb::rope_i;
        let (b, h, s, d) = (2, 3, 7, 10);
        let angles = (randn(&[b, s, d / 2], 5) * 3.0).unwrap();
        let (cos3, sin3) = (angles.cos().unwrap(), angles.sin().unwrap());
        let cos2 = cos3.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let sin2 = sin3.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        // A permuted view: `[B, S, H, D]` storage read as `[B, H, S, D]`.
        let bshd = randn(&[b, s, h, d], 6);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = bshd.to_dtype(dtype).unwrap().transpose(1, 2).unwrap();
            for (cos, sin) in [(&cos2, &sin2), (&cos3, &sin3)] {
                let xf = x.to_dtype(DType::F32).unwrap().contiguous().unwrap();
                let want = rope_i(&xf, cos, sin).unwrap().to_dtype(dtype).unwrap();
                let got = apply_interleaved_rope(&x, cos, sin).unwrap();
                assert_eq!(got.dims(), want.dims());
                assert_eq!(got.dtype(), dtype);
                assert_eq!(bits(&got), bits(&want), "{dtype:?} table {:?}", cos.dims());
            }
        }
    }
}
