# H3 CUDA memory envelope proof

SW-6925 uses the existing capped-pool precedent (`candle-gen/tests/rung4_support`)
for allocation enforcement on a larger CUDA card. An admission-budget override,
memory balloon, or sampled cancellation does not enforce a device limit. The
Windows/WDDM balloon precedent completed with a working set larger than available
VRAM; it cannot establish the 24 GiB target.

`SCENEWORKS_CUDA_PROOF_POOL_BYTES` is a separate, explicit **integer byte count**
read by both Candle CUDA device constructors. It creates a private device-local
stream-ordered pool with `CUmemPoolProps.maxSize` before Candle creates streams
and library handles. cudarc's async allocator uses the current pool; allocation
support, a clean previous pool, and continued current-pool identity are checked.
The variable's absence leaves normal construction unchanged. Run each anchor in
a fresh owned worker process; the pool and retained primary context intentionally
remain alive until process exit. Changing or removing the variable requires a
restart. Do not set it for unrelated jobs/tests sharing the same process.

NVIDIA documents the enforced maximum in [cuMemPoolCreate](https://docs.nvidia.com/cuda/archive/12.3.0/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
and [CUmemPoolProps.maxSize](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-driver-api/structCUmemPoolProps__v1.html).

## Primitive proof

Dispatch `cuda-memory-envelope-proof.yml` at the exact candidate source. It binds
physical GPU1 by UUID, runs one exact ignored raw-allocation test, and retains
source, device, output, and continuous pool high-water counters. The test holds
64 MiB Candle tensors across both device constructors under a 512 MiB cap until
an allocation fails. It rejects zero execution, failure well below the cap,
successful allocation beyond it, a pool watermark below the retained tensors,
or driver reservation above the cap. Cleanup returns live pool use to baseline.
This is allocator evidence only; it does not execute H3 or establish model fit.

The workflow is manual-only and independent of PR/main CI. Estimated execution
after compilation is under one minute; fresh CUDA compilation cost is unknown
and can dominate. It adds no weights/download/history requirement. One job has
a 90-minute bound; the test holds at most 512 MiB plus context/library overhead.

## 24 GiB anchor protocol

Use total `24 * 1024^3` bytes. Deduct an explicitly recorded conservative reserve
for all non-pool allocations and other GPU processes to choose the pool cap.
Keep the matching admission budget and allocator cap distinct in the evidence.
For example, a 22 GiB pool leaves 2 GiB for reserve/non-pool use; those numbers
are a configuration, not prior proof that 2 GiB is sufficient for H3.

Collect `candle_core::cuda_backend::proof_pool::snapshot(0)` during the owned
worker epoch and after all outputs complete. These read the **current capped
pool**, never the default pool: `used_high_bytes` and `reserved_high_bytes` are
driver-continuous peaks. Do not reset them during an anchor, hide loading by a
late observer start, or relabel sampled whole-device values as continuous peaks.

Also collect whole-device used memory with physical UUID, timestamps/cadence,
other-process attribution, and separate host RSS through loading, conditioning,
denoising, and decode. A valid fit needs pool RESERVED high-water at or below
the pool cap, and the observed whole-device footprint plus any unobserved
reserve within 24 GiB. Missing counters and blind spots remain explicit. If
context/library/foreign use exceeds the declared external reserve, the anchor
fails even if its pool stays capped. A sampled whole-device counter alone is
insufficient proof of transient allocation bounds.

## Allocation coverage and limits

- Candle tensors and packed quantized storage use `CudaDevice`/cudarc allocation.
  On a context without async allocation the proof is refused rather than falling
  back to legacy `cuMemAlloc`.
- Explicit cuBLASLt workspace (32 MiB per retained shared workspace), Veda
  scorers/gather/output, FlashAttention output/LSE, and cuDNN convolution
  workspaces use the same Candle/cudarc stream-ordered route.
- CUDA context, module images, cuBLAS/cuRAND/cuDNN internal state, a legacy or
  managed allocation, or another process are **outside this cap**. Source-route
  inspection and an explicit conservative reserve plus whole-device observations
  remain mandatory for the actual workload; the primitive test does not bound
  these. The cap does not propagate to another process or other CUDA device.
- No claim about RTX 3090 support/performance follows from an enforced Blackwell
  envelope. Keep actual hardware and any uncapped allocation limitations visible.

The final H3 campaign still needs both sparse-to-dense and fused-unavailable
materializing routes identified separately; Veda-to-dense may still use fused
attention. This allocator proof does not replace numerical, lifecycle, output,
audio, duration, or long-clip product acceptance.
