# Native MOSS-Image on MLX (sc-26323)

Base and SFT use the exact frozen asset/control contracts in `gen_core::moss_image`.
`MossImage::load` returns the concrete advanced generator for supplied embeddings,
masks, FP32 noise and latent output. Registry loads return ordinary `Generator`s.
Acceleration is explicit per request and defaults OFF. Optional component presence
never selects a mode. The provider keeps one immutable renderer keyed by resolved
policy, replacing it on mode changes without changing ordinary weights in place.

The implemented memory surface is resident, eager, official BF16 plus explicit
published mixed/full-FP32 DMD policies. Arbitrary generic adapters, quantization,
staging, decoder tiling, block windows and alternate execution policies are refused.
Footprints and request workspace are structural estimates; no native calibration is
claimed. The generic constructor prepares/materializes shared Qwen/VAE components only,
with no DiT renderer until the validated request chooses its exact source closure.
Its memory contract describes those common components; a full-FP32 prepared
spec never needs unused ordinary DiT pins. The selected published config is
authenticated before renderer construction.
Before loading, callers selecting acceleration use
`memory_strategy::selected_asset_facts(variant, &spec, resolved.acceleration)` plus
`request_workspace_bytes(width, height, expanded_count, max_sequence_length, branches)`;
the concrete generator's `estimated_request_peak` uses those same selected facts.
For a warm committed-memory snapshot, call
`estimated_incremental_peak_bytes(&request, inputs, budget: MemoryBudget)` before
`budget.fits(...)`. It credits only this provider's materialized retained buffers,
bounded by `budget.committed_bytes`. Foreign models receive no credit. Admission
and generation must be serialized for one generator. Mode switches release the old
renderer before conditioning.
`base_bytes + overlay_bytes` counts one selected renderer. Optional cached full-FP32
files never inflate OFF residency; full-FP32 substitutes the ordinary DiT and excludes
LoRA/time replacement storage. Workspace separately includes one largest-shard owned
source buffer, so this admission estimate can exceed steady-state residency.

FP32 transformer projections, adapter residuals and dataspace mixing use float-accumulating
batched GEMV, so M5's process-wide TF32 setting cannot narrow them. FP32 attention keeps
all keys and runs at most eight query rows per precise vector-kernel call, with cancellation
between calls. This prioritizes the published FP32 contract over NAX throughput; BF16
backbone projections and attention retain their normal kernels. No global precision setting
is changed.

Each sample's text mask is compacted before text/joint attention. Text
encoding runs one unpadded fresh-state prompt at a time, with materialized block
boundaries for cooperative cancellation. The admitted default MLX device is retained.

Numerical unit tests use independently generated CPU author fixtures (FP32/BF16,
mask compaction, skipped layers, exact low-rank arithmetic and data-space DMD).
They initialize MLX arrays and require an admitted local Metal lane or normal hosted
macOS CI. The `cpu_tests` and `hidden_block_hook_cpu_tests` filters are tensor-free.

After coordinator admission, one terminal native smoke invocation is:

```sh
MOSS_NATIVE_TEST_ADMITTED=1 \
MOSS_NATIVE_TEST_VARIANT=sft MOSS_NATIVE_TEST_MODE=off \
MOSS_NATIVE_TEST_SNAPSHOT=/explicit/frozen/sft/snapshot \
MOSS_NATIVE_TEST_OUTPUT_DIR=/explicit/owned/acceptance/output \
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target \
cargo test --locked -p mlx-gen-moss-image --test integration \
  native_smoke::native_generation_or_cancel_with_actual_receipt -- --ignored --exact --nocapture
```

For Base set `VARIANT=base`, supply the Base snapshot and
`MOSS_NATIVE_TEST_SHARED_SNAPSHOT` pointing to frozen SFT. For either published
SFT DMD route select `MODE=unmerged_bf16` or `MODE=full_fp32` and supply
`MOSS_NATIVE_TEST_DMD_SNAPSHOT`. `MOSS_NATIVE_TEST_STEPS` can select a trained prefix
1–8 for DMD; ordinary defaults remain 50. Width/height default 512 and can be
explicitly overridden with `MOSS_NATIVE_TEST_WIDTH`/`HEIGHT`. Exact prompt/seed facts
are retained through the actual execution receipt; default seed is 173. The test
writes a PNG and validated `execution.json`. It never downloads assets.

Set `MOSS_NATIVE_TEST_CANCEL_AFTER_STEP=1` for a cancellation run: it requires the
typed canceled result and absent completed receipt, writing only `cancelled.json`.
Real weights require roughly 27 GB resident ordinary model storage, more for the
full-FP32 renderer, plus request activations and at most one 5.4 GB source staging
buffer. Use the actual provider estimator/admission rather than treating this
rough storage context as measured peak evidence.

Set `MOSS_NATIVE_TEST_TRANSITIONS=1` with either accelerated MODE to execute
OFF→ON→OFF through one retained generator and compare the two ordinary image bytes,
writing three PNGs and three actual receipts. This explicit terminal test is never
run by default. For cancellation, optional `MOSS_NATIVE_TEST_WARM_REENTRY=1` resets
the cancellation token and executes one native step to latent output through the
same generator, publishing `warm-reentry.json` only on successful reentry.
