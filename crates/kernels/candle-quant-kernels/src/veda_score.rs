//! The frozen Veda tile scorer on CUDA: TripPool over each video tile, the residual FP32
//! projections and the scaled pooled-QK logits, equal bit for bit to gen-core's host
//! `contract::pool_video_tiles` + `contract::tile_logits` on identical inputs.
//!
//! Every output is computed by one thread in the host's exact operation order with
//! non-contracting round-to-nearest intrinsics, so the logits (and therefore the frozen top-k
//! masks selected from them) do not depend on how many heads one launch scores. A cuBLAS GEMM
//! would pick its algorithm and reduction order from the batch shape and could flip near-ties.
use crate::nvrtc::KernelSource;
use candle_core::{Device, Result, Tensor};

pub const SOURCE: KernelSource = KernelSource {
    name: "cuda-veda-frozen-scorer-v1",
    src: include_str!("veda_score.cu"),
    cc_floor: (8, 0),
};

/// Scores `heads` heads of one tile shape with a single host readback: `q`/`k` are the tiled
/// `[heads, tiles, 128, dim]` inputs (zero padding), `valid` the real rows of each tile, `video`
/// the leading video tiles that are scored, and `pq`/`pk` the decoded `[heads, 3*dim, dim]`
/// projections. Returns `[heads, video, video]` logits. Any non-finite logit is refused (the
/// host scorer refuses non-finite pooled inputs, which always yield one).
pub fn tile_logits(
    q: &Tensor,
    k: &Tensor,
    valid: &[usize],
    video: usize,
    pq: &[f32],
    pk: &[f32],
) -> Result<Vec<f32>> {
    let (heads, tiles, rows, dim) = q.dims4()?;
    if k.dims() != q.dims()
        || rows != 128
        || dim == 0
        || heads == 0
        || video == 0
        || video > tiles
        || valid.len() != tiles
        || valid.iter().any(|&n| n > 128)
        || pq.len() != heads * 3 * dim * dim
        || pk.len() != pq.len()
        || !q.device().same_device(k.device())
    {
        candle_core::bail!("Veda scorer shape mismatch")
    }
    if !matches!(q.device(), Device::Cuda(_)) {
        candle_core::bail!("Veda scorer requires CUDA")
    }
    #[cfg(feature = "cuda")]
    {
        let scores = cuda::launch(q, k, valid, video, pq, pk)?;
        if scores.iter().any(|s| !s.is_finite()) {
            candle_core::bail!("Veda scorer produced non-finite logits")
        }
        Ok(scores)
    }
    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("Veda CUDA scorer was not compiled")
    }
}
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::{cuda_backend::cudarc, DType, Storage};
    use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
    const THREADS: u32 = 128;
    fn grid(n: usize) -> Result<LaunchConfig> {
        let blocks =
            u32::try_from(n.div_ceil(THREADS as usize)).map_err(candle_core::Error::wrap)?;
        Ok(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    pub(super) fn launch(
        q: &Tensor,
        k: &Tensor,
        valid: &[usize],
        video: usize,
        pq: &[f32],
        pk: &[f32],
    ) -> Result<Vec<f32>> {
        let Device::Cuda(dev) = q.device() else {
            unreachable!()
        };
        let (heads, _, _, dim) = q.dims4()?;
        let module = SOURCE.compiled(dev).map_err(candle_core::Error::wrap)?;
        let function = |name: &str| module.function(name).map_err(candle_core::Error::wrap);
        let (pool, project, logits) = (
            function("veda_pool")?,
            function("veda_project")?,
            function("veda_logits")?,
        );
        let stream = dev.cuda_stream();
        let upload = |x: &[f32]| stream.clone_htod(x).map_err(candle_core::Error::wrap);
        let valid = stream
            .clone_htod(&valid.iter().map(|&n| n as u32).collect::<Vec<_>>())
            .map_err(candle_core::Error::wrap)?;
        let (pq, pk) = (upload(pq)?, upload(pk)?);
        let (h, v, d) = (heads as i32, video as i32, dim as i32);
        let features = heads * video * dim;
        let mut projected = Vec::with_capacity(2);
        for (x, p) in [(q, &pq), (k, &pk)] {
            // Only the scored video tiles are converted; the kernel reads them as `[heads, video,
            // 128, dim]`.
            let x = x.narrow(1, 0, video)?.to_dtype(DType::F32)?.contiguous()?;
            let (storage, layout) = x.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                candle_core::bail!("Veda scorer requires CUDA storage")
            };
            let x = storage
                .as_cuda_slice::<f32>()?
                .slice(layout.start_offset()..);
            let mut f = unsafe { dev.alloc::<f32>(3 * features) }?;
            let mut out: CudaSlice<f32> = unsafe { dev.alloc::<f32>(features) }?;
            unsafe {
                stream
                    .launch_builder(&pool)
                    .arg(&x)
                    .arg(&valid)
                    .arg(&mut f)
                    .arg(&h)
                    .arg(&v)
                    .arg(&v)
                    .arg(&d)
                    .launch(grid(features)?)
                    .map_err(candle_core::Error::wrap)?;
                stream
                    .launch_builder(&project)
                    .arg(&f)
                    .arg(p)
                    .arg(&mut out)
                    .arg(&h)
                    .arg(&v)
                    .arg(&d)
                    .launch(grid(features)?)
                    .map_err(candle_core::Error::wrap)?;
            }
            projected.push(out);
        }
        let n = heads * video * video;
        let mut scores = unsafe { dev.alloc::<f32>(n) }?;
        let divisor = (dim as f32).sqrt();
        unsafe {
            stream
                .launch_builder(&logits)
                .arg(&projected[0])
                .arg(&projected[1])
                .arg(&mut scores)
                .arg(&h)
                .arg(&v)
                .arg(&d)
                .arg(&divisor)
                .launch(grid(n)?)
                .map_err(candle_core::Error::wrap)?;
        }
        // The batch's one host readback.
        stream.clone_dtoh(&scores).map_err(candle_core::Error::wrap)
    }
}
