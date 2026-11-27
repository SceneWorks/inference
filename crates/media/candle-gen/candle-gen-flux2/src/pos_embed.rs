//! FLUX.2's **4-axis RoPE** over `(t, h, w, layer)` position ids. Port of `mlx-gen-flux2`'s
//! `pos_embed.rs`. Each of the 4 axes (`axes_dim = [32,32,32,32]`) contributes `dim/2 = 16`
//! frequencies; the four 16-wide blocks concatenate into a `head_dim/2 = 64`-wide `(cos, sin)`
//! table. Frequencies `ω[j] = theta^{-(2j)/dim}` (θ = 2000) are computed **host-side** in f32 (no
//! MLX-op parity contract for FLUX.2), then the table is materialized on-device.
//!
//! Application is **interleaved** (GPT-J style): lanes `2i` / `2i+1` of each head are the real /
//! imaginary parts rotated together by `(cos[i], sin[i])` — `out0 = r·cos − i·sin`,
//! `out1 = i·cos + r·cos`. The apply is the shared [`candle_gen::rope::apply_interleaved_rope`]
//! (SW-6129): the fused strided RoPE kernel on CUDA, bit-identical on CPU to the candle `rope_i`
//! (f32 copy, rotate, cast back) this module used to call.

use candle_gen::candle_core::{Device, Result, Tensor};

use crate::config::Flux2Config;

/// Builds the FLUX.2 4-axis RoPE `(cos, sin)` tables and applies them to q/k.
pub struct Flux2PosEmbed {
    theta: f32,
    axes_dim: [usize; 4],
    /// `sum(axes_dim) / 2` — the cos/sin table width (= `head_dim / 2`, 64 for klein).
    half: usize,
}

impl Flux2PosEmbed {
    pub fn new(cfg: &Flux2Config) -> Self {
        Self {
            theta: cfg.rope_theta,
            axes_dim: cfg.axes_dim,
            half: cfg.axes_dim.iter().sum::<usize>() / 2,
        }
    }

    /// The cos/sin table width (`head_dim / 2`).
    pub fn half(&self) -> usize {
        self.half
    }

    /// Build the `(cos, sin)` tables `[seq, half]` (f32) for the ordered list of 4-axis position
    /// `ids` (each `[t, h, w, layer]`). Computed host-side, then moved to `device`.
    pub fn cos_sin(&self, ids: &[[i64; 4]], device: &Device) -> Result<(Tensor, Tensor)> {
        let seq = ids.len();
        let mut cos = vec![0f32; seq * self.half];
        let mut sin = vec![0f32; seq * self.half];
        for (s, id) in ids.iter().enumerate() {
            let mut col = 0usize;
            for (axis, &dim) in self.axes_dim.iter().enumerate() {
                let pos = id[axis] as f32;
                let half_axis = dim / 2;
                for j in 0..half_axis {
                    let omega = 1.0f32 / self.theta.powf((2 * j) as f32 / dim as f32);
                    let angle = pos * omega;
                    cos[s * self.half + col] = angle.cos();
                    sin[s * self.half + col] = angle.sin();
                    col += 1;
                }
            }
        }
        let cos = Tensor::from_vec(cos, (seq, self.half), device)?;
        let sin = Tensor::from_vec(sin, (seq, self.half), device)?;
        Ok((cos, sin))
    }

    /// Apply the interleaved RoPE to `x` `[B, H, S, head_dim]` with `cos`/`sin` `[S, head_dim/2]`
    /// (f32, contiguous). Returns the rotated tensor in `x`'s dtype.
    pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        candle_gen::rope::apply_interleaved_rope(x, cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_sin_shapes_and_known_values() {
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        assert_eq!(pe.half(), 64);
        // A position-0 id: every angle is 0 → cos 1, sin 0 across all 64 lanes.
        let (cos, sin) = pe.cos_sin(&[[0, 0, 0, 0]], &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[1, 64]);
        let cos_v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sin_v = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for c in &cos_v {
            assert!((c - 1.0).abs() < 1e-6);
        }
        for s in &sin_v {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn first_lane_of_each_axis_uses_base_frequency() {
        // For axis position p and j=0, ω=1, so angle == p; the first lane of each 16-wide axis block
        // (cols 0,16,32,48) reflects that axis's position directly.
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        let (cos, _) = pe.cos_sin(&[[1, 2, 3, 4]], &Device::Cpu).unwrap();
        let v = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.0f32.cos()).abs() < 1e-5, "t axis @ pos 1");
        assert!((v[16] - 2.0f32.cos()).abs() < 1e-5, "h axis @ pos 2");
        assert!((v[32] - 3.0f32.cos()).abs() < 1e-5, "w axis @ pos 3");
        assert!((v[48] - 4.0f32.cos()).abs() < 1e-5, "layer axis @ pos 4");
    }

    /// Interleaved RoPE at position 0 is the identity (cos 1, sin 0).
    #[test]
    fn apply_at_pos_zero_is_identity() {
        let cfg = Flux2Config::klein_9b();
        let pe = Flux2PosEmbed::new(&cfg);
        let (cos, sin) = pe.cos_sin(&[[0, 0, 0, 0]], &Device::Cpu).unwrap();
        // x: [B=1, H=1, S=1, D=128]
        let x = Tensor::arange(0f32, 128f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, 128))
            .unwrap();
        let y = Flux2PosEmbed::apply(&x, &cos, &sin).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

/// SW-6129: the RoPE apply is the shared `candle_gen::rope::apply_interleaved_rope`.
#[cfg(test)]
mod sw6129_rope_route_tests {
    use super::*;
    use candle_gen::candle_core::DType;
    use candle_gen::candle_nn::rotary_emb::rope_i;

    /// Deterministic xorshift values in [-2, 2) (no device RNG).
    fn values(dims: &[usize], seed: u64, device: &Device) -> Tensor {
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
        Tensor::from_vec(v, dims, device).unwrap()
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

    /// The former private apply, verbatim.
    fn former_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        rope_i(&xf, cos, sin).unwrap().to_dtype(dtype).unwrap()
    }

    fn ids(n: usize) -> Vec<[i64; 4]> {
        (0..n as i64)
            .map(|i| [0, i / 5, i % 5, (i % 3) - 1])
            .collect()
    }

    /// The shared route equals the former apply bit for bit on CPU, f32 / bf16 / f16, at FLUX.2's
    /// real 4-axis table (klein: 4 x 32 axes, head_dim 128).
    #[test]
    fn shared_rope_route_matches_the_former_private_chain() {
        let pe = Flux2PosEmbed::new(&Flux2Config::klein_9b());
        let (cos, sin) = pe.cos_sin(&ids(23), &Device::Cpu).unwrap();
        let x32 = values(&[2, 3, 23, 2 * pe.half()], 7, &Device::Cpu);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = x32.to_dtype(dtype).unwrap();
            let got = Flux2PosEmbed::apply(&x, &cos, &sin).unwrap();
            assert_eq!(got.dtype(), dtype);
            assert_eq!(bits(&got), bits(&former_apply(&x, &cos, &sin)), "{dtype:?}");
        }
    }

    /// SW-6129 CUDA smoke: flux2 Flux2PosEmbed::apply (f32, [1,32,4608,128]) on CUDA device 0 takes the fused strided RoPE (every call counted
    /// `fused`) and is bit-identical to the same call with the kernel switch off (the shared
    /// reference chain) — the bf16 / production-dtype run the CPU goldens cannot see.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_rope_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::ROPE_PATH;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ROPE_PATH.guard(Some(true));
        let pe = Flux2PosEmbed::new(&Flux2Config::klein_9b());
        let (cos, sin) = pe.cos_sin(&ids(512 + 4096), &dev).unwrap();
        // FLUX.2 computes in f32 (bf16 conversion is SW-F381); joint q and k at 1024^2.
        let q = values(&[1, 32, 512 + 4096, 2 * pe.half()], 3, &dev);
        let k = values(&[1, 32, 512 + 4096, 2 * pe.half()], 4, &dev);
        let run = || {
            Tensor::cat(
                &[
                    Flux2PosEmbed::apply(&q, &cos, &sin).unwrap(),
                    Flux2PosEmbed::apply(&k, &cos, &sin).unwrap(),
                ],
                0,
            )
            .unwrap()
        };
        ROPE_PATH.set_enabled(Some(true));
        let start = ROPE_PATH.thread_counts();
        let fused = run();
        let on = ROPE_PATH.thread_counts().since(&start);
        assert_eq!((on.fused, on.reference_total()), (2, 0), "{on}");
        ROPE_PATH.set_enabled(Some(false));
        let start = ROPE_PATH.thread_counts();
        let reference = run();
        let off = ROPE_PATH.thread_counts().since(&start);
        assert_eq!(off.reference_for("disabled"), 2, "{off}");
        ROPE_PATH.set_enabled(Some(true));
        let (f, r) = (bits(&fused), bits(&reference));
        let mismatches = f.iter().zip(&r).filter(|(a, b)| a != b).count();
        println!("flux2 Flux2PosEmbed::apply (f32, [1,32,4608,128]): {mismatches} of {} elements differ; on {on}, off {off}", f.len());
        assert_eq!(mismatches, 0);
    }
}
