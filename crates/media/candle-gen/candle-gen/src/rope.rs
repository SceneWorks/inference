//! Shared rotary / sinusoidal position helpers (sc-27286): the byte-identical bodies the Lens,
//! Qwen-Image, FLUX.2 and Wan transformers each carried. Hoisting them changes no numerics — every
//! caller keeps its own axis layout, table build and per-render cache.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::rotary_emb::rope_i;

/// Apply **interleaved** RoPE (lanes `2i`/`2i+1` are the real/imag pair) to `x` `[B, H, S, head_dim]`
/// with `cos`/`sin` `[S, head_dim/2]`, via candle's fused `rope_i`. The rotation runs in f32 (the
/// tables are f32) and the result is cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
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
}
