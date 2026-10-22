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

## Native editing (sc-26753)

Catalog-loaded `Box<dyn Generator>` exposes `generate_moss_image_edit`,
`moss_image_edit_request_peak` and `moss_image_edit_execution_report`. Provision the
exact `chinoll/Moss-Image-Edit@e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11`
package through `LoadSpec.components[gen_core::moss_image_edit::EDIT_COMPONENT]` as
a directory containing `adapter_config.json` and `adapter_model.safetensors`.
Prepared sources need finalized pins for both files. Both complete file SHA256s,
all 252 rank64 projection configurations, and 504 BF16 A/B tensors are validated.
The weights license is unspecified; the upstream code's GPL license does not license weights.

`MossImageEditInputs` carries one ordered RGB reference, reference maximum edge
(default 512), a cache toggle (default true), and the existing advanced embedding,
mask, negative-batch, FP32 noise and latent-output inputs. Keep generic conditioning
empty to avoid supplying a reference twice. Edit defaults are 30 steps, CFG 4,
shift 5 and full-layer unconditional CFG. Width/height remain explicit request fields
(the product default is 512x512). Nonempty skip layers, SLG and edit+DMD are rejected.
Ordinary generation and both existing DMD policies retain their prior behavior.

Reference pixels use the frozen Pillow Lanczos/no-crop/aligned resize, deterministic
VAE posterior mode, channel-major 2x2 packing and FP32 BN normalization. Reference queries
see only references with timestep 0 and their own reference-index coordinates; text
and target queries see `[text,reference,target]`. Per-layer reference K/V is request-local,
shared across CFG branches and steps, and discarded on completion/error/cancellation.
A cached/uncached primitive comparison also covers four mixed-aspect latent references;
the public one-reference bound widens in sc-26755.

Admission includes adapter bytes, reference encoding/resampling, reference K/V and
attention workspace. Call the typed edit peak method before dispatch and serialize
admission/generation/receipt retrieval. Successful edit receipts include exact adapter
identity, original/resized/latent reference hashes, effective controls and actual noise
seeds/digests. Supplied noise records no generated seed. Ordinary receipts are absent
after an edit, and both receipt kinds clear before any new attempt.

Frozen transformer and VAE numerical fixtures require explicit coordinator-admitted Metal
execution because the exact fused activation is Metal-only. They are synthetic; no real
weights are loaded. CPU projection, reference-coordinate, package and byte-exact Pillow
resize checks run independently. Real-weight quality and product acceptance belong to the
integrated epic-end campaign.
