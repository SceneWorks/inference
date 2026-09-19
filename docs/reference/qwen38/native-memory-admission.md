# Native memory admission

Both native providers estimate load and request memory before allocating checkpoint tensors or
vision/model workspaces. These are conservative admission estimates, not measured peaks or a
claim that the architectural context window fits on every device. Architectural context limits
remain unchanged.

`SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES` optionally supplies an operational byte budget. It caps
fresh available capacity; it cannot enlarge it. Invalid or non-Unicode values fail closed. Missing
capacity also fails closed, even when a budget was supplied. CPU capacity comes from current host
availability (Linux MemAvailable, macOS reclaimable/free vm_stat pages, Windows FreePhysicalMemory).
MLX uses that unified-memory capacity. Candle CUDA reads free memory from the loaded CUDA device's
own context; host RAM is never substituted for VRAM. Load admission checks host staging as well as
CUDA device capacity. Request estimates apply to *additional* request memory against current free
capacity, so loaded weights are not charged twice.

Load upper bounds use stored checkpoint payload sizes without reading weights. Dense Candle CPU
reserves three times the payload for source tensors plus conversion; packed Candle and MLX reserve
two copies for staging/conversion. CUDA device load reserves the payload plus 25 percent temporary
space. External projectors reserve four times their payload for dense conversion. These conservative
bounds can reject a load that a more specialized streaming loader could fit; they do not silently
assume that such a loader exists. Packed language weights are not priced as a dense 27B tensor.

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
