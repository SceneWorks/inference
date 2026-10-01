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

MLX safetensors follow a different allocation path. In the pinned mlx-rs dependency (MLX
0.32.0), upstream `mlx/io/safetensors.cpp` creates lazy Load arrays from headers.
`mlx/backend/common/load.cpp` allocates the output buffer and `pread`s directly into it — owned
Metal shared buffers, not a mapping, so source pages are never reclaimable by the OS while an array
holds them. `mlx/ops.cpp::astype` returns the same array when its dtype already matches. Rust
`Weights` and model clones share array handles. Thus BF16 language weights and vision weights do
not need separate full host and GPU copies.

### MLX safetensors allocation order (sc-24446)

Read from `mlx-llm` (`Weights::from_dir`, `CausalLm::build` / `LayerPlan::load`,
`Qwen35Model::from_weights_layout`, `build_ffn`, `Gemma4Mm::from_weights`, `Projection::load`,
`SwitchLinear::{load,stack}`) and MLX 0.32.0 (`affine_quantize`, `fast::Quantize::eval_gpu`, the
Metal allocator and its buffer cache):

1. Construction is lazy. Every projection, cast, split and quantization is an unevaluated graph
   node over the loaded source arrays; nothing is allocated yet.
2. Each constructor ends with `Weights::verify_accessed_gpu_view` (sc-22414), which evaluates every
   accessed **source** tensor in 512 MiB batches and checks the GPU view. After it, the whole
   accessed payload is resident, and the `Weights` map keeps it so until `load_admitted` returns.
3. Derived arrays — load-time quantized projections, BF16 casts, `1 + w` norms, gathered splits,
   stacked expert banks — are evaluated by the **first forward** (the first request's prefill),
   not at load. Load-time quantization is therefore *not* tensor-by-tensor: the full source payload
   is resident when quantization starts. MLX affine `quantize` on the GPU stream is one kernel per
   tensor that writes the packed `uint32` words plus scales and biases in the input dtype (BF16);
   it has no float intermediates. After a node is evaluated MLX detaches its inputs, so each
   consumed source returns its buffer to MLX's freed-buffer cache.
4. That cache is reused only for an allocation of (page-rounded) equal size, and is released only by
   `clear_cache` — which the decode loop calls after the first step — or when active + cached +
   request crosses MLX's GC limit (0.95 × the device's recommended working set). A consumed
   transient therefore still occupies memory through the first forward.

So the honest upper bound is additive: the payload once, plus every derived allocation, each
rounded to a 16 KiB page. It is not "final payload + the largest single transient".

### The bound (`mlx-llm/src/load_memory.rs`)

`required = payload + derived`, where `derived` sums, per stored tensor of `n` elements (header
only; "language" = not under `model.visual.` / `vision_tower.`):

| Term | Charge | Why |
| --- | --- | --- |
| page rounding | 16 KiB per stored tensor and per derived array | Metal buffers round to 16 KiB pages |
| BF16 cast | `2n` for a language F16/F32/F64 tensor | `as_dtype(BF16)` allocates; BF16 is shared |
| vector intermediates | `4n` for a language tensor of rank ≤ 1 | `1 + w` norms, `exp(A_log)`, signs |
| vision patch transpose | `4n` for `*patch_embed.proj.weight` | contiguous channels-last copy |
| checksum widening | `4·bytes` when `bytes % 4 ≠ 0` | the load-boundary checksum widens a byte view to `u32` |
| gathered split | `2n` for `*.mlp.experts.gate_up_proj` and a rank-3 `*.pos_embedding` | `take_axis` copies (Qwen3.6 gate/up halves, Gemma 4 row/column tables) |
| load-time Q4/Q8 | `n·bits/8 + 4·n/group` for every language float matrix whose input width divides the group, except embeddings and the LM head | packed words + BF16 scales + biases |
| expert stacking | one more copy of each `*experts.{e}.*` tensor in its loaded form (quantized, BF16, or the stored packed part) | `SwitchLinear::stack` (sc-24440) concatenates the bank |

Beside the arrays, a load builds host heap MLX does not see: the parsed tokenizer (measured at
12–17× the `tokenizer.json` bytes for Llama 3.2, Qwen3 and Gemma 2), the chat template, the lazy
graph's nodes and the Metal pipeline states its checksum and first forward compile. Every
safetensors load is charged `24 × tokenizer.json bytes + 64 MiB` for it. The Metal device and MLX's
kernel library (~170 MB) are process-wide one-time costs and are not charged per load.

The quantized term over-charges a few small dense matrices the decoders keep dense (a router, a
one-row gate, `in_proj_a/b`); it never misses one the decoders quantize. Load-time quantization
keeps the BF16 payload resident (step 2), so a Q4 load holds ≈ `1.28×` and a Q8 load ≈ `1.53×`
the BF16 payload of the quantized matrices, not `2×` the checkpoint.

A verified (architecture × conversion) is charged exactly `required` plus the host heap. Anything
unverified keeps the two-copy bound **raised to `required` when that is larger**, plus the host
heap — the derived total of a Qwen3.6 fused expert bank quantized to Q8 is `2.12×` its payload,
which the old flat `2×` under-charged.

| Architecture (loader dispatch) | Unquantized (dense / stored) | Load-time Q4 / Q8 |
| --- | --- | --- |
| Qwen3.5/3.6/3.8 dense, Prism (`qwen3_5*`, `prism_hadamard_qwen35`) | verified (bbddc165b; stored-quantized prepared snapshots included) | verified (sc-24446 probes) |
| Qwen3-VL | verified (bbddc165b) | unverified → `max(2×, required)` |
| Qwen3 | verified, dense only (sc-24446 probes) | verified (sc-24446 probes) |
| Llama (also Mistral, dense Qwen2 — one decoder) | verified, dense only (sc-24446 probes) | verified (sc-24446 probes) |
| Gemma 2 | verified, dense only (sc-24446 probes) | verified (sc-24446 probes) |
| Gemma 4 unified (LTX-2.5 enhancer) | verified, dense only (sc-24446 probes) | verified (sc-24446 probes) |
| any MoE checkpoint (Qwen3.6-35B-A3B `qwen3_5_moe`, Qwen2-MoE, DeepSeek-V2) | unverified → `max(2×, required)` | unverified → `max(2×, required)` |
| stored-quantized snapshot outside Qwen3.5 | unverified → `max(2×, required)` | — |
| Phi-3, GLM-4, DeepSeek-V2 (MLA), plain Gemma 4 | unverified → `max(2×, required)` | unverified → `max(2×, required)` |

MoE stays unverified: its derived bound is dominated by the gathered `gate_up` split copies
(Qwen3.6) or the expert-stacking copies, which MLX's equal-size cache reuse may absorb in practice
but nothing here proves, and the derived totals for Qwen3.6-35B-A3B (BF16 ≈ 116 GB, Q4 ≈ 135 GB,
Q8 152.6 GB) exceed both a 128 GiB host and the 80 GiB probe cap, so no guarded probe can verify
them. Phi-3's packed `qkv_proj`/`gate_up_proj` splits are row slices (views) but unmeasured.

For GGUF it counts the source mapping, retained affine words and both intermediate/final scales,
dense conversion arrays, and the largest per-tensor host conversion/reordering buffers. Projectors
are priced separately from their headers because MLX decodes them to F32. This remains conservative
when mapped source pages are reclaimed; it does not price compact language weights as dense 27B.

A load the bound refuses fails with `load admission: loading this model requires an estimated …
bytes of resident weights and load-time conversions but only … bytes are available` (both
backends, `core_llm::admit_load_memory`) — never the request-side "reduce prompt/media length or
max_new_tokens" remedy, which cannot help a load.

The pinned parent safetensors bound is 55,597,679,416 bytes of arrays (sc-24446 page rounding
included; 55,572,906,808 before) plus its 374,532,544-byte host heap
(`24 × 12,809,320` tokenizer bytes `+ 64 MiB`). GGUF language bounds are 17,061,529,952 (PQ2)
and 15,802,009,952 (PTQ1), plus 2,868,438,080 for BF16 projector or 2,566,539,200 for Q8 projector. Terminal admission adds
its separately named operational reserve to these bounds. The ignored
`pinned_header_only_load_admission_bounds` audit compares native estimates (optionally with a
`quantize` of `q4`/`q8`) against a JSON list of pinned paths and expected bounds without loading a
model.

### Measured probes (sc-24446)

2026-10-01, Apple M5 Max (Mac17,6), 128 GiB, one model at a time under the memory guard
(`HARD_GB=80`, 0.5 s guard plus an in-process 100 ms `phys_footprint` sampler —
`mlx-llm/tests/load_admission_probe.rs`). The baseline is taken after the Metal device and MLX's
kernel library initialize. "Measured load peak" is the load's share: for BF16, the footprint
growth through the end of the load (nothing is derived later); for load-time Q4/Q8, whose
quantized arrays are evaluated by the first forward, the first one-token request's peak growth
minus the request's own MLX working set, measured by a second identical request on the
materialized model. The raw first-request growth includes that request working set, which
request admission prices, not load admission.

| Snapshot | Conversion | Estimate (GB) | Measured load peak (GB) | Margin | Raw first-request growth / request MLX working set (GB) |
| --- | --- | --- | --- | --- | --- |
| Gemma 2 2B-it | BF16 | 5.724 | 5.519 | 3.72% | 9.052 / 3.528 |
| Gemma 2 2B-it | Q4 | 6.865 | 6.587 | 4.23% | 9.013 / 2.426 |
| Llama 3.2 1B Instruct | BF16 | 2.760 | 2.622 | 5.24% | 2.680 / 0.035 |
| Llama 3.2 1B Instruct | Q4 | 3.309 | 3.023 | 9.46% | 3.065 / 0.041 |
| Llama 3.2 1B Instruct | Q8 | 3.796 | 3.685 | 2.99% | 3.723 / 0.038 |
| Qwen3-1.7B | BF16 | 3.790 | 3.574 | 6.05% | 3.658 / 0.046 |
| Qwen3-1.7B | Q4 | 4.586 | 4.418 | 3.79% | 4.456 / 0.038 |
| Qwen3-8B | Q4 | 20.644 | 19.875 | 3.87% | 19.962 / 0.087 |
| Qwen3-8B | Q8 | 24.117 | 23.906 | 0.88% | 23.988 / 0.082 |
| Qwen3-8B | BF16 | 16.733 | 16.513 | 1.33% | 16.610 / 0.053 |
| Gemma 4 unified (LTX-2.5 enhancer) | BF16 | 24.797 | 24.304 | 2.03% | 30.482 / 5.416 |
| Gemma 4 unified (LTX-2.5 enhancer) | Q4 | 30.963 | 29.925 | 3.47% | 34.391 / 4.466 |
| Qwen3.8-27B | Q4 | 69.917 | 69.477 | 0.63% | 69.713 / 0.236 |
| Qwen3.8-27B | Q8 | 82.304 | 80.640 | 2.06% | 80.870 / 0.230 |

The `every_recorded_probe_peak_is_covered_by_its_estimate_with_margin` test pins these relations
(estimate ≥ measured + 0.5%); the ignored `pinned_real_weight_estimates_match_the_recorded_table`
audit recomputes every estimate from the local headers. Load-time Q8 was measured on Llama, Qwen3
and Qwen3.8; Gemma 2 and Gemma 4 Q8 run the identical quantize path at a different `bits` and are
covered by the same derivation. The raw Qwen3.8 Q8 peak footprint was 81.0 GB (75.3 GiB) — under
the 80 GiB guard cap, which its 0.5 s `footprint` sampling under-read at 70 GiB.

On a 128 GiB host (~105 GB available when idle) this admits Qwen3.8-27B at load-time Q4
(69.9 GB, previously 111.1 GB → refused) and Q8 (82.3 GB), the Gemma 4 enhancer at BF16 (24.8 GB,
previously 47.8 GB) and Q4 (31.0 GB), and Qwen3-8B Q4 with a Qwen3-1.7B Q4 draft
(20.6 + 4.6 GB, previously 32.8 + 6.9 GB). Qwen3.6-35B-A3B (MoE) stays refused: ≈ 144.2 GB at
BF16/Q4 (the retained two-copy floor) and ≈ 153 GB at Q8 (its derived bound), host heap included.

Observed in passing, outside load admission: a one-token request on Gemma 2 and on the Gemma 4
enhancer has an MLX working set of 2.4–3.5 GB and 4.5–5.4 GB respectively (vocabulary-wide
buffers at 256K/262K tokens); whether request admission prices that is a request-side question.

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
