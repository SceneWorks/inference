//! One native Veda pointer route, preserving frozen Sage INT8/FP16 arithmetic.
//! Q/K: [heads, slots, dim], slots padded to 128. Mask rows are head-major query tiles.
//! Q uses 128-row scales with scale*LOG2E folded in, K uses 64-row scales;
//! softmax is FP32 exp2, P/V and each block's PV result are FP16.
//! One launch covers every head of the mask; the attention kernel stages each retained K/V block
//! in shared memory once per 32 query rows (rows of a query tile share their retained blocks).
use crate::nvrtc::KernelSource;
use candle_core::{DType, Device, Result, Tensor};

pub const SOURCE: KernelSource = KernelSource {
    name: "cuda-veda-sage-int8-tiled-v2",
    src: include_str!("veda_int8.cu"),
    cc_floor: (8, 0),
};
pub const KERNEL: &str = SOURCE.name;
pub const LOG2E: f32 = std::f32::consts::LOG2_E;

/// CSR retained 64-row blocks, expanded from frozen 128-token selected tiles.
#[derive(Debug, Clone)]
pub struct Mask {
    pub offsets: Vec<u32>,
    pub indices: Vec<u32>,
    pub valid: Vec<u32>,
    pub heads: usize,
}
impl Mask {
    pub fn new(rows: &[Vec<usize>], valid: &[usize], heads: usize) -> Result<Self> {
        if heads == 0
            || heads > 65535
            || valid.is_empty()
            || valid.len() > i32::MAX as usize / 128
            || rows.len() != heads * valid.len()
            || valid.iter().any(|&n| n > 128)
        {
            candle_core::bail!("Veda invalid mask dimensions/valid prefix")
        }
        let mut offsets = vec![0];
        let mut indices = Vec::new();
        for (r, keys) in rows.iter().enumerate() {
            if keys.iter().any(|&k| k >= valid.len() || valid[k] == 0)
                || keys.windows(2).any(|a| a[0] >= a[1])
                || (valid[r % valid.len()] > 0 && keys.is_empty())
            {
                candle_core::bail!("Veda invalid retained key list")
            }
            for &k in keys {
                indices.extend([2 * k as u32, 2 * k as u32 + 1]);
            }
            offsets.push(u32::try_from(indices.len()).map_err(candle_core::Error::wrap)?);
        }
        Ok(Self {
            offsets,
            indices,
            valid: valid
                .iter()
                .flat_map(|&n| [n.min(64) as u32, n.saturating_sub(64) as u32])
                .collect(),
            heads,
        })
    }
    /// Validate host-provided metadata before any raw device access.
    pub fn validate(&self) -> Result<()> {
        let tiles = self.valid.len() / 2;
        if self.heads == 0
            || self.heads > 65535
            || tiles == 0
            || !self.valid.len().is_multiple_of(2)
            || self.slots() > i32::MAX as usize
            || self.offsets.len() != self.heads * tiles + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.last().copied().map(|n| n as usize) != Some(self.indices.len())
            || self.offsets.windows(2).any(|a| a[0] > a[1])
            || self.valid.iter().any(|&v| v > 64)
            || self.valid.chunks_exact(2).any(|v| v[0] < 64 && v[1] != 0)
            || self.indices.iter().any(|&k| k as usize >= self.valid.len())
        {
            candle_core::bail!("Veda malformed device mask")
        }
        for (r, at) in self.offsets.windows(2).enumerate() {
            let keys = &self.indices[at[0] as usize..at[1] as usize];
            if keys.windows(2).any(|a| a[0] >= a[1])
                || (self.valid[2 * (r % tiles)] + self.valid[2 * (r % tiles) + 1] > 0
                    && !keys.iter().any(|&k| self.valid[k as usize] > 0))
            {
                candle_core::bail!("Veda malformed retained block row")
            }
        }
        Ok(())
    }
    pub fn slots(&self) -> usize {
        self.valid.len() * 64
    }
    /// Peak kernel-only bytes: float input copies/output, INT8 Q/K, scales, CSR arrays.
    /// Original tiled QKV remains owned by the caller and is counted separately there.
    pub fn workspace_bytes(&self, dim: usize) -> u64 {
        let elements = self.heads as u64 * self.slots() as u64 * dim as u64;
        20 * elements
            + 4 * (self.heads * (self.slots() / 128 + self.slots() / 64)
                + self.offsets.len()
                + self.indices.len()
                + self.valid.len()) as u64
    }
}
fn quantize(
    x: &[f32],
    heads: usize,
    slots: usize,
    dim: usize,
    block: usize,
    pre: f32,
) -> (Vec<i8>, Vec<f32>) {
    let mut values = vec![0; x.len()];
    let mut scales = Vec::new();
    for h in 0..heads {
        for b in 0..slots / block {
            let base = (h * slots + b * block) * dim;
            let chunk = &x[base..base + block * dim];
            let peak = chunk.iter().map(|v| (v * pre).abs()).fold(0f32, f32::max);
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales.push(scale);
            for (i, &v) in chunk.iter().enumerate() {
                let z = v * pre / scale;
                values[base + i] = (z + if z >= 0.0 { 0.5 } else { -0.5 }) as i8;
            }
        }
    }
    (values, scales)
}
fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}
/// CPU frozen-arithmetic oracle; independent of CUDA launch and reduction implementation.
pub fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dim: usize,
    scale: f32,
    mask: &Mask,
) -> Result<Vec<f32>> {
    mask.validate()?;
    let slots = mask.slots();
    if dim == 0 || q.len() != mask.heads * slots * dim || k.len() != q.len() || v.len() != q.len() {
        candle_core::bail!("Veda oracle QKV dimensions")
    }
    let (q, qs) = quantize(q, mask.heads, slots, dim, 128, scale * LOG2E);
    let (k, ks) = quantize(k, mask.heads, slots, dim, 64, 1.0);
    let mut out = vec![0.0; v.len()];
    for h in 0..mask.heads {
        for row in 0..slots {
            let tile = row / 128;
            if row % 128 >= (mask.valid[2 * tile] + mask.valid[2 * tile + 1]) as usize {
                continue;
            }
            let query = (h * slots + row) * dim;
            let ri = h * (slots / 128) + tile;
            let mut m = -1e30f32;
            let mut l = 0f32;
            let mut acc = vec![0f32; dim];
            for &block in &mask.indices[mask.offsets[ri] as usize..mask.offsets[ri + 1] as usize] {
                let block = block as usize;
                let n = mask.valid[block] as usize;
                let mut logits = vec![0f32; n];
                for (i, z) in logits.iter_mut().enumerate() {
                    let key = (h * slots + block * 64 + i) * dim;
                    let dot: i32 = (0..dim)
                        .map(|d| q[query + d] as i32 * k[key + d] as i32)
                        .sum();
                    *z = dot as f32 * qs[h * (slots / 128) + tile] * ks[h * (slots / 64) + block];
                }
                let next = logits.iter().copied().fold(m, f32::max);
                let alpha = (m - next).exp2();
                let p: Vec<f32> = logits.iter().map(|&z| (z - next).exp2()).collect();
                l = l * alpha + p.iter().sum::<f32>();
                for d in 0..dim {
                    let mut pv = 0f32;
                    for (i, &p) in p.iter().enumerate() {
                        pv = half(p).mul_add(half(v[(h * slots + block * 64 + i) * dim + d]), pv);
                    }
                    acc[d] = acc[d] * alpha + half(pv);
                }
                m = next;
            }
            for d in 0..dim {
                out[query + d] = acc[d] / l.max(f32::MIN_POSITIVE)
            }
        }
    }
    Ok(out)
}
/// No host oracle or alternate sparse implementation is selected by this API.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Mask, scale: f32) -> Result<Tensor> {
    mask.validate()?;
    let (heads, slots, dim) = q.dims3()?;
    if heads != mask.heads
        || slots != mask.slots()
        || k.dims() != q.dims()
        || v.dims() != q.dims()
        || dim == 0
        || dim > 128
        || !dim.is_multiple_of(4)
        || !scale.is_finite()
        || scale <= 0.0
        || !matches!(q.dtype(), DType::BF16 | DType::F16)
        || k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
        || !q.device().same_device(k.device())
        || !q.device().same_device(v.device())
    {
        candle_core::bail!("Veda unsupported QKV device/dtype/layout")
    }
    if !matches!(q.device(), Device::Cuda(_)) {
        candle_core::bail!("Veda requires CUDA")
    }
    #[cfg(feature = "cuda")]
    {
        cuda::launch(q, k, v, mask, scale)
    }
    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("Veda CUDA kernel was not compiled")
    }
}
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::{cuda_backend::cudarc, op::BackpropOp, CudaStorage, Shape, Storage};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    /// Query rows per attention CTA (`VEDA_ROWS` in the kernel); divides the 128-row query tile.
    const ATTEND_ROWS: usize = 32;
    pub(super) fn launch(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Mask,
        scale: f32,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = q.device() else {
            unreachable!()
        };
        let (heads, slots, dim) = q.dims3()?;
        let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
        let quant = module
            .function("veda_quant")
            .map_err(candle_core::Error::wrap)?;
        let attention = module
            .function("veda_attend")
            .map_err(candle_core::Error::wrap)?;
        let stream = dev.cuda_stream();
        let mut qi = unsafe { dev.alloc::<u8>(q.elem_count()) }?;
        let mut ki = unsafe { dev.alloc::<u8>(q.elem_count()) }?;
        let mut qs = unsafe { dev.alloc::<f32>(heads * slots / 128) }?;
        let mut ks = unsafe { dev.alloc::<f32>(heads * slots / 64) }?;
        let mut out = unsafe { dev.alloc::<f32>(q.elem_count()) }?;
        let offsets = stream
            .clone_htod(&mask.offsets)
            .map_err(candle_core::Error::wrap)?;
        let indices = stream
            .clone_htod(&mask.indices)
            .map_err(candle_core::Error::wrap)?;
        let valid = stream
            .clone_htod(&mask.valid)
            .map_err(candle_core::Error::wrap)?;
        let qf = q.to_dtype(DType::F32)?.contiguous()?;
        let kf = k.to_dtype(DType::F32)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.contiguous()?;
        let (qst, ql) = qf.storage_and_layout();
        let (kst, kl) = kf.storage_and_layout();
        let (vst, vl) = vf.storage_and_layout();
        fn slice(s: &Storage) -> Result<&CudaStorage> {
            match s {
                Storage::Cuda(s) => Ok(s),
                _ => candle_core::bail!("Veda requires CUDA storage"),
            }
        }
        let qv = slice(&qst)?
            .as_cuda_slice::<f32>()?
            .slice(ql.start_offset()..);
        let kv = slice(&kst)?
            .as_cuda_slice::<f32>()?
            .slice(kl.start_offset()..);
        let vv = slice(&vst)?
            .as_cuda_slice::<f32>()?
            .slice(vl.start_offset()..);
        let n = slots as i32;
        let d = dim as i32;
        let qb = 128i32;
        let kb = 64i32;
        let pre = scale * LOG2E;
        let one = 1f32;
        let cfg = |rows: usize| LaunchConfig {
            grid_dim: (rows as u32, heads as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&quant)
                .arg(&qv)
                .arg(&mut qi)
                .arg(&mut qs)
                .arg(&n)
                .arg(&d)
                .arg(&qb)
                .arg(&pre)
                .launch(cfg(slots / 128))
                .map_err(candle_core::Error::wrap)?;
            stream
                .launch_builder(&quant)
                .arg(&kv)
                .arg(&mut ki)
                .arg(&mut ks)
                .arg(&n)
                .arg(&d)
                .arg(&kb)
                .arg(&one)
                .launch(cfg(slots / 64))
                .map_err(candle_core::Error::wrap)?;
            stream
                .launch_builder(&attention)
                .arg(&qi)
                .arg(&ki)
                .arg(&vv)
                .arg(&qs)
                .arg(&ks)
                .arg(&offsets)
                .arg(&indices)
                .arg(&valid)
                .arg(&mut out)
                .arg(&n)
                .arg(&d)
                .launch(cfg(slots / ATTEND_ROWS))
                .map_err(candle_core::Error::wrap)?;
        }
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from(q.dims()),
            BackpropOp::none(),
            false,
        )
        .to_dtype(q.dtype())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_rounding_block_scales_and_empty_padding() {
        let (q, s) = quantize(&[127., -127., 0.5, -0.5, 0., 0., 0., 0.], 1, 2, 4, 1, 1.);
        assert_eq!(&q[..4], &[127, -127, 1, -1]);
        assert_eq!(s, vec![1., 1.]);
    }
    #[test]
    fn expansion_and_partial_prefix_are_frozen() {
        let m = Mask::new(&[vec![0, 2], vec![1], vec![0, 1, 2]], &[128, 17, 3], 1).unwrap();
        assert_eq!(m.indices, vec![0, 1, 4, 5, 2, 3, 0, 1, 2, 3, 4, 5]);
        assert_eq!(m.valid, vec![64, 64, 17, 0, 3, 0]);
        assert!(Mask::new(&[vec![]], &[1], 1).is_err());
        assert!(Mask::new(&[vec![0, 0]], &[1], 1).is_err());
    }
    #[test]
    fn malformed_metadata_is_refused_before_device_access() {
        let good = Mask::new(&[vec![0]], &[17], 1).unwrap();
        good.validate().unwrap();
        for bad in 0..5 {
            let mut m = good.clone();
            match bad {
                0 => m.indices[0] = 99,
                1 => m.offsets[1] = 99,
                2 => m.valid[0] = 65,
                3 => m.valid[1] = 1,
                _ => m.offsets[0] = 1,
            }
            assert!(m.validate().is_err());
        }
    }
    #[test]
    fn oracle_ignores_unselected_and_padded_values() {
        let dim = 4;
        let n = 256 * dim;
        let m = Mask::new(&[vec![0], vec![1]], &[3, 7], 1).unwrap();
        let q = vec![0f32; n];
        let k = vec![0f32; n];
        let mut v = vec![999f32; n];
        for row in 0..3 {
            v[row * dim..row * dim + dim].fill(2.);
        }
        for row in 128..135 {
            v[row * dim..row * dim + dim].fill(-4.);
        }
        let got = reference(&q, &k, &v, dim, 0.5, &m).unwrap();
        assert_eq!(&got[..12], &[2.; 12]);
        assert_eq!(&got[128 * dim..135 * dim], &[-4.; 28]);
        assert!(got[12..128 * dim].iter().all(|&x| x == 0.));
    }
}
