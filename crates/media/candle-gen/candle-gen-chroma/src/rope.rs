//! Chroma's **FluxPosEmbed** 3-axis RoPE over `(t, h, w)` position ids — the candle port of
//! `mlx-gen-chroma`'s `build_rope` / `apply_rope_one`.
//!
//! `axes_dims_rope = [16, 56, 56]` (summing to `head_dim = 128`); each axis contributes `dim/2`
//! frequencies, concatenated into a `head_dim/2 = 64`-wide `(cos, sin)` table. Frequencies
//! `ω[j] = theta^{-(2j)/dim}` (θ = 10000) are computed host-side in f32, then the table is
//! materialized on-device.
//!
//! Application is **interleaved** (adjacent-pair): lanes `2i` / `2i+1` of each head are the real /
//! imaginary parts rotated by `(cos[i], sin[i])` — `out0 = r·cos − i·sin`, `out1 = i·cos + r·sin`.
//! The apply is the shared [`candle_gen::rope::apply_interleaved_rope`] (SW-6129; the fused strided RoPE
//! kernel on CUDA, bit-identical on CPU to the candle `rope_i` this module used to call — the same
//! route `candle-gen-flux2`'s `pos_embed` takes). The text tokens sit at the RoPE origin
//! `(0,0,0)`; the packed image tokens carry `(0, row, col)` over the `(height/16, width/16)` grid.

use candle_gen::candle_core::{Device, Result, Tensor};

use crate::config::ChromaTransformerConfig;

const ROPE_THETA: f32 = 10000.0;

/// The precomputed `(cos, sin)` RoPE tables `[seq, head_dim/2]` for one ordered position-id list.
pub struct RopeTable {
    pub cos: Tensor,
    pub sin: Tensor,
}

/// Build the `(cos, sin)` tables for the ordered `[t, h, w]` position `ids` (text ids first, then
/// image ids — the same `cat(txt_ids, img_ids)` order the DiT attends). Host-side f32, then moved to
/// `device`.
pub fn build_rope(ids: &[[i64; 3]], axes: [usize; 3], device: &Device) -> Result<RopeTable> {
    let half: usize = axes.iter().map(|d| d / 2).sum();
    let seq = ids.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (s, id) in ids.iter().enumerate() {
        let mut col = 0usize;
        for (axis, &dim) in axes.iter().enumerate() {
            let pos = id[axis] as f32;
            let half_axis = dim / 2;
            for j in 0..half_axis {
                let omega = 1.0f32 / ROPE_THETA.powf((2 * j) as f32 / dim as f32);
                let angle = pos * omega;
                cos[s * half + col] = angle.cos();
                sin[s * half + col] = angle.sin();
                col += 1;
            }
        }
    }
    Ok(RopeTable {
        cos: Tensor::from_vec(cos, (seq, half), device)?,
        sin: Tensor::from_vec(sin, (seq, half), device)?,
    })
}

impl RopeTable {
    /// Apply the interleaved RoPE to `x` `[B, H, S, head_dim]` (the rotation runs in f32, casting back
    /// to `x`'s dtype). `cos`/`sin` are `[S, head_dim/2]`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        candle_gen::rope::apply_interleaved_rope(x, &self.cos, &self.sin)
    }
}

/// FluxPosEmbed image position ids for the packed `(height/16, width/16)` grid: `(0, row, col)`,
/// row-major — diffusers `_prepare_latent_image_ids`. Matches the pack order in
/// [`crate::pipeline`] (which mirrors candle FLUX's `State::new`).
pub fn latent_image_ids(h2: usize, w2: usize) -> Vec<[i64; 3]> {
    let mut ids = Vec::with_capacity(h2 * w2);
    for i in 0..h2 {
        for j in 0..w2 {
            ids.push([0, i as i64, j as i64]);
        }
    }
    ids
}

/// Text position ids `[L]` — all `(0,0,0)` (FluxPosEmbed places every text token at the origin).
pub fn zero_text_ids(l: usize) -> Vec<[i64; 3]> {
    vec![[0, 0, 0]; l]
}

/// The full RoPE id list the DiT attends: `cat(txt_ids, img_ids)` (text first).
pub fn joint_ids(text_len: usize, h2: usize, w2: usize) -> Vec<[i64; 3]> {
    let mut ids = zero_text_ids(text_len);
    ids.extend(latent_image_ids(h2, w2));
    ids
}

/// Build the RoPE table for a `text_len`-token prompt over a `(h2, w2)` packed image grid, against
/// `cfg.axes_dims_rope`.
pub fn build_for(
    cfg: &ChromaTransformerConfig,
    text_len: usize,
    h2: usize,
    w2: usize,
    device: &Device,
) -> Result<RopeTable> {
    build_rope(&joint_ids(text_len, h2, w2), cfg.axes_dims_rope, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_width_is_head_dim_half() {
        let cfg = ChromaTransformerConfig::default();
        let t = build_for(&cfg, 2, 2, 2, &Device::Cpu).unwrap();
        // 16/2 + 56/2 + 56/2 = 8 + 28 + 28 = 64 = head_dim/2.
        assert_eq!(t.cos.dims(), &[2 + 4, 64]);
        assert_eq!(t.sin.dims(), &[2 + 4, 64]);
    }

    #[test]
    fn origin_position_is_identity() {
        // Text tokens (and image token (0,0,0)) sit at the origin → cos 1, sin 0 everywhere.
        let t = build_rope(&[[0, 0, 0]], [16, 56, 56], &Device::Cpu).unwrap();
        let c = t.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = t.sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &c {
            assert!((v - 1.0).abs() < 1e-6);
        }
        for v in &s {
            assert!(v.abs() < 1e-6);
        }
        // rope_i at the origin is the identity.
        let x = Tensor::arange(0f32, 128f32, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 1, 128))
            .unwrap();
        let y = t.apply(&x).unwrap();
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn image_ids_are_row_major() {
        let ids = latent_image_ids(2, 3);
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[0], [0, 0, 0]);
        assert_eq!(ids[1], [0, 0, 1]);
        assert_eq!(ids[3], [0, 1, 0]);
        assert_eq!(ids[5], [0, 1, 2]);
    }

    #[test]
    fn first_lane_of_each_axis_uses_base_frequency() {
        // For axis position p and j=0, ω=1, so the first lane of each axis block reflects p directly.
        // Cols: t at 0, h at 8, w at 36.
        let t = build_rope(&[[1, 2, 3]], [16, 56, 56], &Device::Cpu).unwrap();
        let c = t.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((c[0] - 1.0f32.cos()).abs() < 1e-5, "t axis @ pos 1");
        assert!((c[8] - 2.0f32.cos()).abs() < 1e-5, "h axis @ pos 2");
        assert!((c[36] - 3.0f32.cos()).abs() < 1e-5, "w axis @ pos 3");
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

    /// The former private apply, verbatim (candle `rope_i` on an f32 contiguous copy, cast back).
    fn former_apply(t: &RopeTable, x: &Tensor) -> Tensor {
        let dtype = x.dtype();
        let xf = x.to_dtype(DType::F32).unwrap().contiguous().unwrap();
        rope_i(&xf, &t.cos, &t.sin)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    /// The shared route equals the former apply bit for bit on CPU, f32 / bf16 / f16, at Chroma's
    /// real `[16, 56, 56]` table (text at the origin, a 4x5 image grid) and head_dim 128.
    #[test]
    fn shared_rope_route_matches_the_former_private_chain() {
        let cfg = ChromaTransformerConfig::default();
        let t = build_for(&cfg, 3, 4, 5, &Device::Cpu).unwrap();
        let x32 = values(&[2, 3, 3 + 20, 128], 7, &Device::Cpu);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = x32.to_dtype(dtype).unwrap();
            let got = t.apply(&x).unwrap();
            assert_eq!(got.dtype(), dtype);
            assert_eq!(bits(&got), bits(&former_apply(&t, &x)), "{dtype:?}");
        }
    }

    /// SW-6129 CUDA smoke: chroma RopeTable::apply (f32, [1,24,1536,128]) on CUDA device 0 takes the fused strided RoPE (every call counted
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
        let cfg = ChromaTransformerConfig::default();
        let t = build_for(&cfg, 512, 32, 32, &dev).unwrap();
        // Chroma computes in f32 (bf16 conversion is SW-F381); q and k at 1024^2.
        let q = values(&[1, 24, 512 + 1024, 128], 3, &dev);
        let k = values(&[1, 24, 512 + 1024, 128], 4, &dev);
        let run = || Tensor::cat(&[t.apply(&q).unwrap(), t.apply(&k).unwrap()], 0).unwrap();
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
        println!("chroma RopeTable::apply (f32, [1,24,1536,128]): {mismatches} of {} elements differ; on {on}, off {off}", f.len());
        assert_eq!(mismatches, 0);
    }
}
