# Native memory admission

Both native providers estimate load and request memory before allocating checkpoint tensors or
vision/model workspaces. These are conservative admission estimates, not measured peaks or a
claim that the architectural context window fits on every device. Architectural context limits
remain unchanged.

`SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES` optionally supplies an operational byte budget. It caps
fresh available capacity; it cannot enlarge it. Invalid or non-Unicode values fail closed. Missing
capacity also fails closed, even when a budget was supplied. CPU capacity comes from current host
availability (Linux MemAvailable, macOS reclaimable/free vm_stat pages, Windows FreePhysicalMemory).
Qwen3.8 and Prism/Bonsai are accelerator-only and Candle rejects those snapshots on CPU before
checkpoint inventory or tensor allocation; this host-capacity behavior remains available to other
model families.
MLX uses that unified-memory capacity. Candle CUDA applies the operational cap to VRAM and reads
current free memory from the loaded CUDA device's own context; host RAM is never substituted for
VRAM, and a launch-time GPU snapshot never substitutes for current post-load free VRAM. CUDA load
admission checks host staging independently against current host capacity. Request estimates apply
to *additional* request memory against current free capacity, so loaded weights are not charged
twice.

Load upper bounds use checkpoint files without evaluating tensor payloads. Dense Candle CUDA's
pinned loader reads one safetensors shard into a host buffer, copies its
tensors directly to CUDA at their stored dtype, then drops that buffer before reading the next
shard. Its host bound is therefore the largest shard rather than two complete checkpoint copies.
Qwen3.8 is BF16 and same-dtype model construction shares the loaded storage. CUDA device admission
reserves the complete payload plus 25 percent temporary space, covering construction-time casts
such as the F32 vision tower. Candle external projectors reserve four times their stored payload.

MLX safetensors follow a different allocation path. In the pinned mlx-rs dependency, upstream
`mlx/io/safetensors.cpp` creates lazy Load arrays from headers. `mlx/backend/common/load.cpp`
allocates the output buffer and reads directly into it; the Metal allocator uses
`ResourceStorageModeShared`. `mlx/ops.cpp::astype` returns the same array when its dtype already
matches. Rust `Weights` and model clones share array handles. Thus BF16 Qwen language weights and
vision weights do not need separate full host and GPU copies.

`mlx-llm/src/load_memory.rs` counts the stored safetensors payload once, plus additional BF16 casts
for non-BF16 language tensors, vector/norm intermediates and a possible vision patch transpose.
Other architectures and explicit quantization retain the prior conservative two-copy bound.
For GGUF it counts the source mapping, retained affine words and both intermediate/final scales,
dense conversion arrays, and the largest per-tensor host conversion/reordering buffers. Projectors
are priced separately from their headers because MLX decodes them to F32. This remains conservative
when mapped source pages are reclaimed; it does not price compact language weights as dense 27B.

The pinned parent safetensors bound is 55,572,906,808 bytes; Bonsai safetensors is 9,514,891,750;
the baseline is 17,542,650,296. GGUF language bounds are 17,061,529,952 (PQ2) and 15,802,009,952
(PTQ1), plus 2,868,438,080 for BF16 projector or 2,566,539,200 for Q8 projector. These are computed
upper bounds, not measured residency. Terminal admission adds its separately named operational
reserve to these bounds. The ignored `pinned_header_only_load_admission_bounds` audit can compare
native estimates against a JSON list of pinned paths and expected bounds without loading a model.

Request bounds include expanded visual tokens, eager attention scores/mask/softmax, full-length
K/V storage, recurrent state, projection/MLP/logit buffers, MTP cache/rollback and draft-width
buffers, RGB preprocessing, unmerged vision-patch attention and vision MLP workspaces. Candle prices
every workspace element at four bytes. Since sc-20671, MLX prices K/V, activations and logits at the
decoder's compute width (BF16) and eager attention scores at F32. Prism/Bonsai stays at four bytes
until its frozen allocator peaks are re-taken on the BF16 path; see
`docs/architecture/SC_20671_DENSE_BASELINE_MODEL_CONTRACT.md`. Checked
arithmetic rejects overflow. Visual geometry is computed before visual encoding. A low budget
therefore rejects a request inside the architectural token window without invoking the decoder.

The native composition fixtures exercise text, image and video routes for generic Qwen3-VL and
Qwen35 decoders, including reasoning, JSON, MTP and stops. Isolated subprocess tests verify that a
one-byte budget rejects both load and within-window visual requests, and malformed budgets fail.
The terminal campaign must still record actual accepted/rejected lengths and measured residency;
these deterministic guards do not replace that hardware evidence.

## Compressed KV cache (sc-20682)

A request opts into the compressed KV cache with `TextLlmRequest::kv_compression`
(`KvCompressionPolicy::Off` by default). MLX decides the request's cache before admission
(`plan_kv_cache` against `core_llm::KV_COMPRESSION_QUALIFICATIONS`), so the estimate prices the
cache the request actually runs on: the dense K/V term for every dense plan, and for a qualified
plan `core_llm::compressed_kv_cache_bytes` in its place. That size is derived from the
`KvCompressionFormat` (key/value code bits, group size, an f16 scale and zero per group), not
from a fixed ratio, at the cache's block-rounded capacity of prompt plus `max_new_tokens`:

* resident: per layer and KV head, packed K codes `⌈C/G⌉·G·D·bits_k/8`, K scale/zero
  `⌈C/G⌉·D·4`, V codes `C·⌈D·bits_v/8⌉` and V scale/zero `C·⌈D/G⌉·4`, plus one group of dense K
  and V residual rows at the compute width (`2·G·D·element_bytes`). `C` is the token count
  rounded up to the 256-position growth block and `G = 32`.
* transient: `min(layers, 11)` layers' packed arrays at full capacity plus one block (a block
  growth's pre-growth copy, or a group flush's output, beside the live arrays: one per in-flight
  MLX evaluator buffer), every layer's residual rows again (a step's rollback point), and every
  layer's fused-reader split-KV scratch (`Hq · 128 splits · (D + 2) · 4` bytes for the one-token
  decode dispatches a product generation issues; its prompt step attends through dense SDPA),
  plus one layer's dense prompt K/V (`P·Hkv·D·element_bytes·2`). The prompt step evaluates each
  layer's packed store and attention output before building the next layer; left as one lazy
  graph, every layer's dense prompt K/V stayed resident beside the packed store (measured 1.88x
  the estimate on a 28-layer, 8×128-KV-head decoder, above the same request run dense).

Every dense MLX estimate (a dense plan, an un-opted request, a dense re-admission) prices the
dense cache the same way: its block-rounded buffers plus `min(layers, 11)` layers' pre-growth
buffers, which a block growth holds beside their successors until the evaluator releases them
(a dense decode across growth blocks peaked 1.30x the unrounded K/V term without this).
`admission_estimates_cover_the_measured_peak_compressed_and_dense` holds the MLX allocator's
measured peak at or below the estimate for both cache kinds through prefill and a decode across
growth blocks.

For K8V8 the resident term is `2.25·D` bytes per layer, head and token against `4·D` for BF16
dense. A shallow decoder can price above dense once the transients are added; a qualified request
is still priced compressed, because that is the cache it runs on.

A compressed plan whose cache selection then refuses the fused reader runs dense from the start,
so the provider admits the request again at the dense estimate before any K/V exists. A compressed
generation that later transitions to dense (a reader fault over resident history) admits that
transition against fresh capacity before reconstructing: the larger of the reconstruction (every
layer's dense K/V at block capacity plus one layer's Float32 dequantization) and the dense cache
the rest of the generation grows to (block-rounded, with its window of growth copies). It is
admitted even when no history is resident to rebuild, because the rest of the generation still
runs dense. A refusal fails the generation with the same typed
`RequestResourceExhausted` rather than oversubscribing memory.

Every generation reports its cache on `TextLlmOutput::kv_cache` (`KvCacheReport`):
`format_version` (`KV_CACHE_FORMAT_VERSION`), `format` (`group-affine-k8v8`, or none when it ran
dense throughout), `fallback` (a stable `KvCacheFallbackReason::id`; none exactly when it ran
wholly compressed), `detail` (the backend's operation and reason words, never prompt content) and
`counters` (fused attention calls, dense fallback events, full-cache dequantizations, retained
compressed bytes). SceneWorks records this report on its prompt-refine job result and telemetry.
