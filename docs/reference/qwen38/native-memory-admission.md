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
buffers, RGB preprocessing, unmerged vision-patch attention and vision MLP workspaces. Four-byte
workspace elements conservatively cover FP32 intermediates even with BF16 weights. Checked
arithmetic rejects overflow. Visual geometry is computed before visual encoding. A low budget
therefore rejects a request inside the architectural token window without invoking the decoder.

The native composition fixtures exercise text, image and video routes for generic Qwen3-VL and
Qwen35 decoders, including reasoning, JSON, MTP and stops. Isolated subprocess tests verify that a
one-byte budget rejects both load and within-window visual requests, and malformed budgets fail.
The terminal campaign must still record actual accepted/rejected lengths and measured residency;
these deterministic guards do not replace that hardware evidence.
