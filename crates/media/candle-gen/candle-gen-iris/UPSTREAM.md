# candle-gen-iris — upstream coverage

The coverage table, pinned revisions, fixtures and tolerances for Iris-3B live in the MLX twin's
[`UPSTREAM.md`](../../mlx-gen/mlx-gen-iris/UPSTREAM.md), which carries a Candle column; this crate's
parity tests read the same committed fixtures. This file records only what is specific to the
Candle crate.

## Memory-strategy contract (sc-25690)

Upstream ships no memory policy; this is the repo's shared ladder (`gen_core::MemoryProviderContract`)
declared for what each Candle route actually runs. `src/memory_strategy.rs` owns all three contracts,
and `register_providers` registers each with its weights-free fixture and behavior seam. `iris_3b` is a
registry generator; depth (a provider-specific API) and restoration (a transform) are registered as
composed routes.

| | `iris_3b` | `iris_3b_depth` | `iris_3b_restore` |
|---|---|---|---|
| Memory mode key | `text_to_image` | `depth` | `restoration` |
| Resident | implemented | implemented | implemented (one backbone forward per export tile, always) |
| StagedResidency | implemented (text encoder → encode → drop → backbone) | not applicable (no text encoder) | not applicable (no text encoder) |
| BoundedDecode | not applicable (pixel space) | not applicable | not applicable |
| BoundedAttention | implemented (64 Mi score elements) | missing | missing |
| BoundedTransformerResidency | missing | missing | missing |
| Witnessed load surfaces | bf16 × {resident, sequential} × eager | same | same |
| Resident bytes (materialized snapshot) | text tower (`model.language_model.*` + untied `lm_head`, at the production device's tower dtype) + backbone (projections at the compute dtype, norm gains / modulation tables / text position table at f32) + adapter overlays | `empty_prompt.safetensors` (f32) + backbone and reducer | `empty_prompt.safetensors` (f32) + backbone |

Differences from the MLX twin, both deliberate:

- **BoundedAttention is implemented here and missing on MLX.** Candle's attention materializes the
  `[rows, heads, queries, keys]` scores tensor, so bounding it is a real lever; MLX's fused kernel
  never builds it. The budget is the shared 64 Mi operating point, a derived candidate the epic's
  terminal campaign has yet to measure.
- **No deferred load surface.** No Iris Candle loader has a block-streaming shape, so a
  `DeferredMaterialization` spec has no contract and is refused at load.

Staged residency and bounded attention are per-request decisions on generation: the request scope
writes `GenerationMemory`, and `generate` honours it. A `Sequential` load stages every request.
Depth and restoration requests carry no memory block, so those routes are resident only; that is
also why their bounded-attention rung is missing rather than not applicable.

Every byte figure is read from safetensors headers, including the adapter stack
(`adapters::adapter_footprint`), which is declared as up to two typed components:
`iris_3b.adapters.forward_residuals` (LoRA factors, structured LoKr factors, low-rank LoHa) and
`iris_3b.adapters.materialized_deltas` (a LoHa at or above its break-even rank, a LoKr whose
factors are not plain matrices). Each loader prices its contract before reading a weight and
propagates the error, so a spec with no contract fails `load`.

Two request-side rules follow from the contract:

- `validate` refuses the memory controls this route has no mechanism for
  (`GenerationMemory::tile_vae_decode`, `stream_transformer_blocks`, `chunk_attention` without a
  size).
- The request scope refuses a request whose `prompt_batch.len() × count` exceeds the admitted
  `MemoryGeometry::batch`: a prompt batch denoises its rows together, so each prompt is a batch row.

### What bounded attention does to the image

Chunking computes the same attention to GEMM-order f32 rounding. On the FP32 path a bounded render
equals the unbounded one to within one RGB8 level. Under the bf16 policy — the only tier witnessed —
every projection re-rounds to bf16, so the render is not bit-identical. Declared bound, held on the
miniature snapshot on Candle CPU over three seeds and three binding budgets: no channel moves by
more than **8 RGB8 levels** and the mean absolute change is at most **1.0 level** (observed on arm64
macOS: 2 levels and 0.20). This bounds the miniature; the release weights on CUDA are unmeasured.

Verification: `memory_strategy` unit tests (declaration, admission gates, header pricing held to the
loader's recorded dtypes and installed residuals, registry conformance, load refusals), `nn` unit
tests (no kernel launch exceeds its budget), and `tests/memory_strategy.rs` (a staged selection on a
warm generator loads the two components in staged order and renders the identical image; an admitted
budget reaches the kernels; the bf16 bound above).
