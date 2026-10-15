# Veda problem-bound residency (sc-25776)

The tensor-neutral `gen-core::veda::workspace` contract prices each enabled request
without opening the predictor, creating a tensor, or consulting free memory.
Its identity includes the backend, chosen frozen plan, actual execution grid and
trained grid. Disabled/absent controls return `None` before plan parsing and leave
existing dense estimates untouched.

`bound` consumes the runtime's actual padded layouts and plan head groups;
`for_video_request` is a conservative pre-encoding envelope of the same contract.
It includes byte-tokenized prompt length and presentation overhead, the native
vision processor's snapped merged-image geometry and every temporal clip vision block, the requested reference-image
short edge with stride rounding, each reference clip's own aspect at the generated
duration, soundtracks and standalone reference audio. It does not substitute a
reference count or nearest trained size for the requested workload. Invalid or
unrepresentable shapes fail closed. Every grid it prices (target, reference image,
keyframe and clip spans) is in DiT patch units — the 16x VAE latent divided by the
2x2 spatial patch, i.e. pixels / 32 — the same grid the runtimes pack and upstream
`h3_layout.py` tiles (1344x768 at 124 frames prices `[37, 24, 42]`).

The verified, decoded predictor is retained once per process (`Bundle::load_cached`),
keyed on path, size, modification time and inode; a changed file is re-read and
re-hashed, a failed load is never cached. The MLX first-use self-test outcome is
cached per (device, kernel id). The per-request bound still charges the predictor.

The total sums these owned lifetimes conservatively (it may exceed the temporal
peak because initialization, scoring, self-test and fallback do not all overlap):

- BF16 host predictor projections, frozen plan/JSON trees, the exact checkpoint
  buffer, one FP8-to-FP32/BF16 decode staging projection and FP32 device projections.
  Load reads at most the exact checkpoint size plus one sentinel byte, with size
  checked before allocating, so a corrupt oversized file cannot evade the bound.
- Every cached layout's permutation, inverse and valid prefixes, and native
  packed position-row owners retained by layout/model clones, including vector growth capacity.
- Head-order and padded QKV gathers, masked tile copies, pool casts and mean/max/min
  operands, residual/projected features, host/device logits, downloaded scorer
  values, selector candidates and retained row/index vectors.
- CUDA: gathered K/V, query and output chunks, host/device indices and masks, conservative
  query/key score scratch. MLX: the kernel's tiled output, global query/K/V/output
  copies, key-tile lists and counts. Both: retained tile/head/batch outputs and final
  concatenate and inverse-permutation outputs. Caller QKV is explicitly included.
- CUDA FP32 conversion/contiguous copies, INT8 Q/K, scales, output, and expanded
  64-key CSR masks on host/device, plus each backend's fixed first-use self-test
  with retained masked reference/oracle outputs and numerical comparison buffers.
- Dense fallback with resident predictor/layout state. MLX uses its declared fused
  streaming SDPA; Candle uses its production 1e9 score-element query budget, with
  the minimum one-query batch explicitly priced when that exceeds the budget.

The copy/scorer and attention limits are allocation *allowances*, not whole-layer
estimates. A minimum head exceeding either allowance is still charged and reports
refusal before any scorer or attention allocation. CUDA batches heads under a 4 GiB
copy allowance (512 MiB attention). MLX (sc-26214) batches same-shape heads under a
4 GiB scorer allowance and a 4 GiB attention allowance: its Metal route scores the
whole batch with one readback (pooled/projected features, device logits and their
host copy, key-tile lists), runs one block-sparse kernel launch over every video
query tile reading Q/K/V in place through the tile permutation, and one dense call
over the batch's global query rows (with their head-selected K/V and output), then
assembles the native-order output. No tile-order or gathered Q/K/V copy exists on
the MLX Metal route, so none is charged; sparse execution refuses a non-Metal MLX
device (the CPU gather route is a test-only reference). Every layer's BF16 Q/K
projections are device-resident and charged once with one layer's FP32 upload
transients (`predictor_device_bytes`); FP32 Q/K/V are narrowed to BF16 once per layer
for the 16-bit NAX products, charged in `input_qkv_bytes`. Each batch's output is evaluated before the
next batch's scorer graph, so lazy graphs cannot retain all batches concurrently.
Globals stay separate from sparse video batches.

Native H3 memory-strategy methods expose the explicit contract. SceneWorks adds
it to the conservative dense baseline at both Candle pre-load and common queued
admission. Common admission uses H3's existing fixed phase releases, re-reads the
current post-load memory snapshot for each job, credits only that provider's
committed bytes once, and fails closed on missing budgets. Dense-only curves are
not labeled as sparse evidence; sparse cannot pass solely on a dense measurement.
The problem bound never grows when the budget increases.

CPU tests verify owned categories and their total, off behavior, tiny allowances,
nearest/long/reference-heavy geometry, dense fallback and successive queued
budgets. The ignored MLX allocator test measures the complete first-use lifetime
against its self-test and total bounds, only with coordinator GPU admission.
CUDA first-use numerical/device fixtures remain mandatory on admitted CUDA.
CPU compilation/tests are not an allocator/device pass. Real-weight peak and
quality reconciliation belongs to the single S9 terminal campaign.
