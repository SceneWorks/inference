# Native MOSS-Image

`moss_image_base` and `moss_image_sft` use the exact shared frozen manifest and
caller-prepared file pins. Base needs the explicit `moss-shared-snapshot` SFT
component. Optional DMD files make modes available; only request
`moss_image.acceleration = "unmerged_bf16"` or `"full_fp32"` enables them.
Default is `"off"`, including when `steps = 8` or DMD assets are present.

`MossImageGenerator::load_on_device` accepts the allocator's admitted Candle
device. The registered loader uses process logical device zero. The allocator
must verify the physical GPU UUID/mapping before starting native execution.
No runtime fetches assets or invokes Python.

The ordinary API and `MossImageAdvancedGenerator` share one implementation.
Advanced calls accept FP32 embeddings/binary masks, batched negative text or
embeddings, exact FP32 NCHW noise and latent output. CFG (including zero/one),
negative-branch skips and positive SLG follow the frozen source. Mixed DMD keeps
unmerged FP32 residuals and all six replacement tensors; full FP32 loads the
published distilled transformer and applies no second LoRA. Every successful
call publishes an actual validated execution receipt. Serialize generation and
receipt retrieval together.

Worker admission for a concrete loaded provider calls
`estimated_incremental_peak_bytes(request, advanced_inputs, budget)` with its
fresh committed-memory snapshot, writes that demand into the normal
`MemoryRunContext` (including the actual expanded batch), and calls
`Generator::memory_strategy_safety_check`. Generate then independently rechecks
the same selected-mode estimate against that admitted context. The separate
`estimated_peak_bytes` returns the absolute structural bound. Warm credit is
limited to actual retained provider weights and the committed snapshot; foreign
allocations receive no credit. A mode change releases the old renderer before
conditioning. These provider-specific estimators are public concrete methods;
the generic `Generator` trait still owns admission and generation.

Resident reuse and synchronized sequential phase release are implemented.
Structural estimates include expanded batches, context, branch work, FP32
islands and one authenticated owned shard buffer. No native memory calibration
is claimed, and optimized-fit strategies remain Missing. Tiling, block
streaming, alternate quantization, reference/edit/control tensors and previews
are refused by the advertised surface.

CPU tests use only deterministic tiny author fixtures and explicit `Device::Cpu`.
The BF16 CPU GEMM oracle accumulates already-rounded BF16 operands in FP32 and
rounds its output to BF16; accelerator GEMMs remain native BF16. CPU fixture
tolerances do not constitute CUDA numerical or image-quality acceptance.

## Terminal native campaign

The ignored CUDA integration test is exclusively for sc-26325 after fresh
physical GPU0 PID/UUID/runner ownership checks and root admission. Compilation
does not execute it. Set absolute local paths explicitly:

```powershell
$env:MOSS_BASE_SNAPSHOT = 'E:\models\MOSS-Image-Base\snapshots\1ceede80badddd6a7812069619edaa5633587192'
$env:MOSS_SHARED_SNAPSHOT = 'E:\models\MOSS-Image-SFT\snapshots\fc5de0d56fc910cbca6f6625a70f5f582b26deb1'
$env:MOSS_DMD_SNAPSHOT = 'E:\models\MOSS-Image-SFT-8-steps-dmd\snapshots\5ff314b4c1470876b114b810146085db8a181a6e'
$env:MOSS_NATIVE_OUTPUT_DIR = 'E:\sceneworks-evaluation\moss\native-sc26325-candle'
$env:MOSS_NATIVE_TERMINAL_ADMITTED = 'sc-26325-physical-gpu0'
$env:MOSS_PHYSICAL_GPU_INDEX = '0'
$env:MOSS_EXPECTED_CUDA_UUID = '<fresh verified physical GPU0 UUID>'
$env:CUDA_VISIBLE_DEVICES = $env:MOSS_EXPECTED_CUDA_UUID
cargo test --locked -p candle-gen-moss-image --features cuda --release --test integration real_weights::native_complete_surface_campaign -- --ignored --exact --nocapture
```

The DMD snapshot must contain both split adapter/replacement files and all four
published FP32 transformer shards. The test emits PNGs and actual JSON receipts,
checks ordinary→mixed/full→ordinary identity, advanced latent conditioning/batch
transport, typed cancellation with no stale receipt, and warm reentry. Environment
variables are a guard against accidental invocation, not an ownership lease or
independent physical-device proof; the external terminal allocator supplies those.


## Static consumer admission (sc-26325)

Before the generic boxed provider loader, call public
`memory_strategy::selected_preload_estimate(variant, &spec, &request,
MossImageAdvancedInputs::default())`. Its `peak_bytes` reuses the runtime formula;
`asset_facts` is selected resident accounting, `workspace_bytes` is scratch plus
512MiB fixed overhead, and `owned_shard_transient_bytes` is the largest selected
owned source shard. **The shard is added once in `peak_bytes`; workspace excludes
it.** This differs from MLX's older workspace helper, which already includes its
fixed source staging envelope. `selected_asset_facts` and `request_workspace_bytes`
are also public, but consumers should use the combined call to authenticate once.
No device or tensor is created by static admission. Authentication itself reads
one owned shard on the host; callers must provide that staging headroom before
calling it. OFF does not authenticate optional DMD availability; mixed does not
read full-FP32 shards; full replaces the ordinary BF16 DiT closure for static facts and runtime weights.
The ordinary SFT core remains a catalog installation prerequisite. Both normal
constructors prepare only shared sources and defer DiT selection to the validated
request; no ordinary DiT config/index/shard pin is required for the selected full
FP32 closure. Its published config is authenticated before constructing that
renderer. Constructor facts describe the common components; selected static and
runtime estimates describe the complete requested policy. Warm admission uses
`Generator::moss_image_owned_weight_bytes()` inside the serialized transaction,
bounded by the current committed snapshot. It reports actual materialized retained
weights without tensor evaluation or new allocations.

The dedicated inference `moss-native-terminal-windows.yml` workflow is exclusively
for the stationary final inference-main pin in the frozen product epic feature. Its strict JSON control
and physical-GPU0 lease are described in
`docs/moss-native-terminal-windows.md`. The CPU-only `native_inventory` example
verifies the shared 41-file frozen manifest; it does not initialize CUDA.
