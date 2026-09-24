# sc-24140 — terminal measurements at the release candidate: evidence

Epic sc-24128, terminal measurements at release candidate
R = `e9faeabdbf8ded296ec8083828e717cd951240ae` (tag `runtime-2026.09.1-rc.0`). This note covers
two items: the stale AT3 memory assertion, and E6 memory honesty for quantizing and GGUF loads.
The fixes go into `runtime-2026.09.1-rc.1`.

**RTX Pro 6000 / sm_120.** Host: Windows 11. Everything ran on **GPU 1** only
(`CUDA_VISIBLE_DEVICES=1`, RTX PRO 6000 Blackwell Max-Q, 97,887 MiB), one run at a time. Before
each load, GPU 1 used about 879–889 MiB (a co-tenant). Build: CUDA 12.9, MSVC 14.44 vcvars64,
`CUDA_COMPUTE_CAP=120`, `--release --features cuda`. The AT2 campaign was **not** re-run. The
Qwen3.8-27B rows marked *campaign* are read from the sealed campaign at
`C:/evidence/sc-24128/at2-campaign` (GPU 0).

## Item 1 — AT3: the stale `ac3` memory assertion

**Cause (confirmed).** `ac3_static_kv_device_pointers_are_stable_across_the_fixture_and_a_rollback`
asserted `cache.memory().live_bytes == model.static_kv_bytes(capacity)` for a fresh
`new_static_cache(capacity, 4)`. Since sc-24131, `DeltaNetCache::memory_bytes` prices the per-token
checkpoint ring as soon as the ring is specified, which is at creation. One slot counts as live
and the other `max_checkpoints` count as checkpoints. So the live bytes also hold one recurrent
state per linear layer:

* SSM: 48 value heads × 128 × 128 × 4 B = 3,145,728 B.
* Conv tail: 3 × 10,240 × 2 B = 61,440 B.
* One state: 3,207,168 B. Qwen3.8-27B has 48 linear layers: 48 × 3,207,168 = **153,944,064 B**.

This is exactly R's `177,078,272 − 23,134,208`. The test panicked at line 163, so its
pointer-stability half never ran.

**New assertions** (`tests/static_kv_parity.rs`). Each is priced by the function admission uses:

* `cache.recurrent_bytes() == model.recurrent_state_bytes(1 + max_checkpoints)` — the ring. This is
  the provider's `memory_geometry_with_checkpoints` term.
* `memory.total_bytes() − cache.recurrent_bytes() == model.static_kv_bytes(capacity)` — the KV
  component. `StaticKvCache::bytes` reads the real buffers.
* `memory.total_bytes() == static_kv_bytes(capacity) + recurrent_state_bytes(1 + max_checkpoints)`.
* `memory == CacheMemory { live: kv + recurrent_state_bytes(1), checkpoint: ring − recurrent_state_bytes(1) }`
  — the exact split.
* After the 256-token run, `cache.memory() == preallocated`: nothing grows. This replaces the old
  "past the preallocation only the live recurrent state" line. That line was stale for the same
  reason.
* After the rollback to 349, `cache.memory() == preallocated` again: a rollback keeps every buffer
  and ring slot it was priced for (added in the review).

A CPU twin carries the same assertions on the tiny config, and also checks the rollback:
`models::qwen35::tests::ringed_static_cache_holds_exactly_what_admission_prices`.

**Real-weight run** (`static-kv-parity-qwen38-27b.log`):

```text
REQUIRE_SM120=1 BONSAI_QWEN38_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
cargo test --release --locked -p candle-llm --features cuda --test static_kv_parity -- --ignored --nocapture --test-threads=1
```

All four ignored tests pass: ac1, **ac3**, the provider off-path parity and the teacher-forced
report. ac3 prints `16 attention buffers pinned across 256 steps and a rollback to 349;
preallocated 792854528 bytes (23134208 KV + 769720320 checkpoint ring)`. The ring is
5 × 153,944,064 B. The 16 `(K, V)` device pointers were identical at every 32nd step, at the end
of the run, after the rollback and after one more forward, and `memory()` still equalled the
preallocation after the rollback.

**Mutations** (`mutations.log`):

* **M1a** — `recurrent_state_bytes` prices one slot too many. RED on the CPU twin.
* **M1b** — `static_kv_bytes` prices `capacity − 1` positions. RED on the CPU twin.
* **M1c** — the ring is counted only once it is allocated (`memory_bytes` gated on `self.ring`).
  This is the accounting before sc-24131, and the stale assertion held under it. RED on the CPU
  twin.
* **R6a** — a rollback releases the ring's accounting (`DeltaNetCache::rollback_to` clears the
  ring spec). RED on the **real-weight** ac3, at the new post-rollback assertion
  (`checkpoint_bytes: 0` against `615776256`), and on the CPU twin.

## Item 2 — E6: load residency against admission

`tests/load_residency.rs` is an `#[ignore]`d harness. It loads a snapshot directory or a single
`*.gguf` file through `LlamaProvider::load` with `cuda_graphs: Some(false)`, generates 256 greedy
tokens, and samples after a device-wide synchronize. Each sample records:

* **device used** — `cuMemGetInfo` total − free.
* **pool reserved** and **pool used** — the device's current CUDA memory pool, which cudarc
  serves every candle allocation from. `used` is the live allocations.
* the pool's high watermarks.

It also polls `cuMemGetInfo` in the background through the load. Device figures below are
**deltas over the pre-load sample**: 1.79 GB, which is this process's CUDA context plus the
co-tenant. **Admission** is `LlamaProvider::load_memory_estimate`, the function `load` itself
admits with. The harness also prints R's bound and the bound before the review (`bd66280ed`).
Since the review it **fails** when the polled load peak exceeds the admitted device bound, so a
pricing regression fails the measurement (mutation R6b). All figures are GB (10⁹ bytes). The JSON
documents are `load-residency-*.json`.

### The measurements

| load | payload | admission at R | before review | **admission (this PR)** | source / bf16 cast / quantized copy | load peak: device Δ / pool live | margin |
|---|---|---|---|---|---|---|---|
| Qwen3-8B bf16 | 16.38 | 20.48 | 20.48 | **20.48** | 16.38 / 0 / 0 | 16.42 / 16.38 | 4.06 |
| Qwen3-8B Q8 | 16.38 | 20.48 | 29.18 | **27.86** | 16.38 / 0 / 7.38 | 26.82 / 24.02 | 1.04 |
| Qwen3-8B NVFP4 | 16.38 | 25.08 | 25.08 | **24.73** | 16.38 / 0 / 4.26 | 21.18 / 20.89 | 3.55 |
| Qwen3-8B Q4_K GGUF, dense | 4.61 | 5.76 | 5.76 | **57.34** | 32.76 / 16.38 / 0 | 49.47 / 49.14 | 7.87 |
| Qwen3-8B Q4_K GGUF → Q8 | 4.61 | 5.76 | 5.76 | **50.82** | 32.76 / 2.49 / 7.38 | 48.26 / 42.99 | 2.56 |
| Qwen3.8-27B Q8 (provider) | 55.56 | 69.45 | 98.97 | **97.10** | 55.56 / 0 / 27.65 | 89.27 / 85.06 | 7.83 |
| Qwen3.8-27B NVFP4 (provider) | 55.56 | 85.08 | 85.08 | **84.09** | 55.56 / 0 / 14.64 | 72.69 / 72.08 | 11.40 |

The margin is the new admission minus the device-Δ peak. Every load was admitted by the new bound
on GPU 1 with the co-tenant present, and none peaked above it. At R, both GGUF loads and both Q8
loads peaked above their admission (the GGUF by **8.6×**).

The **priced quantized copy equals the copy the loader built**, byte for byte, on every row: the
census's GGML bytes plus candle's 544 B of CUDA row padding per Q8_0 tensor (8B: 7,379,877,888 +
252 × 544 = 7,380,014,976; 27B: 27,649,474,560 + 409 × 544 = 27,649,697,056), and the census's
NVFP4 bytes exactly (8B 4,257,054,720; 27B 14,637,957,120).

After the load and through decode:

| load | census | after load: device Δ / pool live / idle | at token 256: device Δ / pool live | decode peak pool live |
|---|---|---|---|---|
| Qwen3-8B bf16 | 16.38 | 16.42 / 16.38 / 0.03 | 16.58 / 16.43 | 16.43 |
| Qwen3-8B Q8 | 9.87 | 15.64 / 9.87 / 5.77 | 15.77 / 9.93 | 9.94 |
| Qwen3-8B NVFP4 | 5.50 | 8.70 / 5.54 / 3.15 | 8.83 / 5.58 | 5.59 |
| Qwen3-8B Q4_K GGUF, dense | 16.38 | 19.17 / 16.38 / 2.78 | 19.30 / 16.43 | 16.43 |
| Qwen3-8B Q4_K GGUF → Q8 | 9.87 | 14.44 / 9.87 / 4.56 | 14.56 / 9.93 | 9.94 |
| Qwen3.8-27B Q8 (provider) | 30.24 | 42.09 / 32.09 / 9.99 | 42.21 / 32.44 | 32.47 |
| Qwen3.8-27B NVFP4 (provider) | 17.23 | 25.64 / 19.11 / 6.53 | 25.77 / 19.44 | 19.48 |

The campaign rows for Qwen3.8-27B (bench path: `Weights::from_dir` + model + MTP; absolute device
used, context included) are unchanged: bf16 57.08 after load / 58.08 decode, Q8 42.96 / 43.42,
NVFP4 25.14 / 26.28. The provider path also loads the f32 ViT tower: 460,730,096 params × 4 B =
1,842,920,384 B. That tower is not in the decoder census.

### Why a quantizing load sits above its census after load

It is **idle, reusable CUDA-pool reservation, not live memory.**

* **Live memory equals the census, to the byte.**
  * 8B Q8: pool live after load − census = 137,088 B. That is 252 GGML tensors × 544 B of
    candle's CUDA `MATRIX_ROW_PADDING` (512 × 34/32).
  * 27B Q8: the gap is the ViT tower's 1,842,920,384 B plus 409 × 544 B.
  * 8B NVFP4: the gap is 33,554,432 B, the cuBLASLt handle's 32 MiB workspace.
  * GGML scales and blocks are counted correctly: 8.5 bits per parameter, the census's
    `storage_size_in_bytes`.
  * Dense embeddings and the head are in the census: `other` holds 1.25 GB and dense `lm_head`
    1.24 GB.
* **The gap is the pool keeping freed load memory.**
  * `Weights::from_dir` loads every bf16 source tensor onto the device (a GGUF load: its dense f32
    map, below).
  * `Projection::load_as` quantizes each projection while that whole map is still alive. The
    load's live peak is sources + copy: 16.38 + 7.38 + transients = 24.02 GB on the 8B.
  * When the map drops, the sources return to the pool.
  * The pool does release idle memory at a synchronize. Every sample here synchronizes first, and
    the sample after the provider drops shows the reservation already gone, with no explicit trim.
    Yet 5.77 GB (8B Q8) and 9.99 GB (27B Q8) stay reserved after the load. Our reading is that the
    pool gives memory back only in units that are wholly free. Still-live tensors are interleaved
    with the freed sources: the dense embeddings, norms and head (which share storage with the
    source map), and the new copies. So those units cannot be returned.
  * NVFP4 and the GGUF loads show the same mechanism. bf16 does not (0.03 GB), because its sources
    *are* its weights.
* **The idle reservation is reusable, not leaked.**
  * Every decode allocation over 256 tokens was served from it: pool reserved did not grow.
  * Request admission already counts it as available. `cuda_usable_memory_bytes` adds
    reserved − used to the driver's free bytes.
  * Dropping the provider returns it. Reserved falls to 0.13 GB on the 8B. What remains is
    candle's grow-only MMQ/MMVQ workspaces.
* **No per-matmul dequantized temporaries.** Decode raised the 8B Q8 pool's live peak by only
  73 MB over the post-load live bytes (KV, activations and the MMVQ Q8_1 workspace). Candle's fast
  MMVQ/MMQ kernels quantize activations to Q8_1 instead of dequantizing weights. A dequantized
  f32 MLP weight alone would be 201 MB. The cuBLAS workspace is inside that 73 MB.

### The E6 defect at R

The steady state is under admission for every format, even at R: 8B Q8 is 15.64 < 20.48. The
**load peak** is not. R priced a CUDA load at payload + 25 % headroom. It added the growing
quantized copy only for NVFP4 (`payload · 9/32`). A GGML Q4/Q8 load builds its copy on the device
the same way, beside the resident sources, but that copy was unpriced:

| load | load peak (device Δ) | admission at R | shortfall |
|---|---|---|---|
| Qwen3-8B Q8 | 26.82 | 20.48 | **6.34** |
| Qwen3.8-27B Q8 | 89.27 | 69.45 | **19.82** |
| Qwen3-8B Q4_K GGUF, dense | 49.47 | 5.76 | **43.71** |
| Qwen3-8B Q4_K GGUF → Q8 | 48.26 | 5.76 | **42.50** |

So a Q8 or GGUF load could be admitted and then fail with an OOM part-way through.

### The fix (`crates/llm/candle-llm/src/provider.rs`)

The device bound is the load's **working set**: the source the loader holds resident while it
builds the decoder, the unchanged **25 % headroom over that source**, and the copies it builds
beside the source. `LlamaProvider::load_memory_estimate` returns it with its parts
(`source_bytes`, `cast_copy_bytes`, `quantized_copy_bytes`; `LoadMemoryEstimate` is
`#[non_exhaustive]`).

* **The quantized copy covers exactly the tensors the loader quantizes, at the bytes the quantizer
  allocates for them.** The first version of this PR priced it over the whole payload (Q8_0 at
  17/32), which charged the dense embeddings, a dense head and the ViT tower. Now admission reads
  the snapshot's safetensors headers (never tensor data) and walks them with the loader's own
  rules:
  * `models::llama::quantizes_layer_tensor` under `models::llama::decoder_root` for the llama
    family (q/k/v/o, Phi-3's fused `qkv_proj` and `gate_up_proj` split as the loader splits them,
    MLA's low-rank projections, the MLP, every MoE expert and shared expert; the head only under
    NVFP4).
  * `models::qwen35::quantizes_tensor` for the qwen3_5 hybrid (the Gated DeltaNet in/out
    projections, attention, MLP, every stacked MoE expert slice, the shared expert, the MTP head
    and `mtp.fc`; the head under every format). The per-head `in_proj_a` / `in_proj_b`, the
    router, norms and embeddings are never priced.
  * The loader debug-asserts every tensor it quantizes against the same rule, so the two cannot
    drift apart unnoticed (mutations R5b / R5e).
  * Each tensor is priced by `ggml_device_bytes` (the GGML blocks plus candle's CUDA row padding)
    or `nvfp4_device_bytes` (the E2M1 nibbles over the `K`-padded columns plus the UE4M3 scales in
    cuBLASLt's 128 × 4 atoms). An NVFP4 shape that `load_eligible` keeps dense (it uses the same
    `nvfp4_shape_refusal`) costs nothing.
* **A persisted `quantization` block is priced only where the loader honours it.** A
  llama-family snapshot re-quantizes on load; the qwen3_5 loader reads only `spec.quantize`, so
  its persisted block is not priced (the snapshot loads dense).
* **An MLX-affine triple is priced as a Q8_0 repack only where the loader repacks it**: an 8-bit
  format over an 8-bit triple, tested with the geometry check `QuantizedLinear::from_mlx_affine_q8`
  itself uses (`mlx_affine_q8_in_dim`). A 4-bit triple falls through to the loader's own typed
  refusal (`MLX affine projection requires Q8`, or `invalid MLX affine Q8 triple` under an explicit
  Q8 request) and costs nothing.
* **A llama-family GGUF is priced by what its loader really holds.** `GgufCheckpoint::open`
  dequantizes every tensor it maps into a dense **f32** tensor on the device and keeps the whole
  map until the provider is built. `CausalLm::from_weights_with` then casts every tensor it keeps
  dense to bf16 beside it (a second copy), and under a Q4 / Q8 request quantizes the layer
  projections. So the source is the f32 map (4 B per element — 32.76 GB for the 8B, 7.1× its
  4.61 GB Q4_K file), with the 25 % headroom over it, plus the bf16 casts and the copy. It is read
  from the GGUF header with candle's own reader. On a host device the f32 map is host memory, so a
  host-device GGUF load prices its host domain by the same working set. A Prism GGUF (packed
  blocks the loader wraps) keeps its pricing.
* **A snapshot not stored in bf16 is charged the bf16 cast beside its source** (coordinator
  follow-up). The decoder takes every tensor it keeps dense through `to_dtype(bf16)`, which
  shares a bf16 tensor's storage but copies any other float. So a CUDA load of an f16 or f32
  snapshot holds a second, 2-byte copy of each such tensor beside its source until the decoder is
  built. Before, only the 25 % headroom covered it: a Llama-2-7B-shaped f16 snapshot (13.5 GB)
  peaks near 27 GB against a 16.9 GB bound. `snapshot_cast_copy_bytes` now charges 2 B per element
  for every float tensor not stored in bf16, reported as `cast_copy_bytes`. It skips the projections the loader
  consumes whole into its quantized copy (they are cast one at a time, inside the headroom), and an
  MLX-affine triple's sidecars, which the loader converts on the host. An NVFP4 shape the loader
  keeps dense is charged. Every measured snapshot is all-bf16, so no measured load's pricing
  changed.

The 25 % headroom is needed on the GGUF path too. The GGUF → Q8 load's pool live peak was 42.99 GB,
but the pool reserved 48.25 GB: 5.26 GB of fragmentation around the per-projection bf16 → f32 →
Q8_0 transients. A headroom of 25 % of the bf16-equivalent size (4.1 GB) would have under-priced
it; 25 % of the f32 map (8.19 GB) covers it with a 2.56 GB margin.

**What this changes for the 27B Q8 load.** The card holds 102.64 GB and this process's CUDA context
takes 0.86 GB. Admission now refuses the 27B Q8 load once other tenants hold more than about
**4.7 GB** (before the review: about 2.8 GB). The load itself fits until they hold about 12.5 GB,
so a co-tenant between about 4.7 and 12.5 GB is still refused up front although the load would fit.
That band is the 25 % headroom (13.9 GB on the 27B), which this PR leaves as it was: the 8B Q8 load
consumed 3.06 GB of its 4.10 GB headroom, and the GGUF → Q8 load 5.63 GB of 8.19 GB. At R, a
co-tenant between about 12.5 and 32.3 GB was admitted into a failure.

### Quantizing a view: candle's quantizer reads from the storage's start (coordinator follow-up)

Candle's GGML `QTensor::quantize` hands its source's **whole storage** to the block quantizer. It
ignores the view's offset and extent. Two loaders quantize views:

* the qwen3_5 MoE loader: every expert is narrowed out of the stacked `experts.gate_up_proj` /
  `experts.down_proj`;
* the llama loader: Phi-3's fused `qkv_proj` / `gate_up_proj` are narrowed into q / k / v and
  gate / up.

On CUDA the bf16 → f32 cast before the quantizer builds a fresh tensor, so the defect is masked.
On a host device the compute dtype is already f32. There the cast is a no-op and the view itself
reaches candle. In a debug build that trips candle's size check. In a release build it silently
quantizes every expert from the first expert's rows, and k / v / up from q's / gate's rows.

The fix is at the one GGML call site in candle-llm: `QuantizedLinear::quantize` compacts an f32
source with `force_contiguous` before quantizing it. Candle itself is not patched. The only other
`QTensor::quantize` call site in candle-llm, the `prepare` writer, is handed whole tensors read
from safetensors.

The same idiom was broken in two NVFP4 sites (`candle-quant-kernels`), which I fixed the same way:

* **The fused NVFP4 quantizer** (`quantize_nvfp4_activation_fused`) meant to materialize an
  offset view with `Tensor::copy`. `copy` keeps the view's storage and offset, so the materialization did nothing. Its test,
  `a_row_narrowed_f32_view_quantizes_its_own_rows`, compared the view against `view.copy()` and so
  passed either way. The test now compares against a `force_contiguous` tensor. It failed against
  the old code (different nibbles) and passes with `force_contiguous`.
* **The NVFP4 decode GEMV** realigned an activation view off the 16-byte boundary with `copy` too.
  A new test, `gemv_realigns_a_contiguous_view_off_the_vector_boundary`, hit
  `CUDA_ERROR_MISALIGNED_ADDRESS` before the fix and passes with `force_contiguous`.

The shipped paths hand both NVFP4 sites a fresh bf16 → f32 cast or an aligned activation, so neither
was reached in practice.

A follow-up could shrink both the quantizing load peak and the idle reservation. The loader would
quantize shard by shard and drop each source as it is consumed, instead of holding the whole map.
That is a loader change and is not made here.

### Unit tests (synthetic geometry, CPU unless noted)

* `admission_prices_the_working_set_beside_the_resident_source` — the bound's arithmetic for
  every load kind and both devices.
* `quantized_load_estimates_price_exactly_the_copy_the_loader_builds` — a bf16 tiny-llama snapshot
  loaded Q8 and (at a 256-wide geometry) Q4 through the provider: the priced copy equals the
  census's GGML bytes plus padding; a wider vocabulary leaves it unchanged; NVFP4 prices the
  eligible projections and not the 40-row head.
* `a_persisted_block_is_priced_only_where_the_loader_honours_it` — item 2 of the review.
* `an_affine_triple_is_priced_only_where_the_loader_repacks_it` — item 3: the 8-bit triple priced
  and loaded; the 4-bit triple unpriced and refused, under both formats.
* `qwen35_q8_estimate_prices_exactly_the_copy_the_loader_builds` — the dense + MTP hybrid against
  the census; the MoE hybrid against its expert slices (and, on the CUDA build, the census).
* `gguf_admission_prices_the_dequantized_map_and_the_copies_beside_it` — item 1 on a tiny `qwen3`
  GGUF: the priced f32 map equals the map `GgufCheckpoint::open` holds, the priced copy equals the
  loader's census, the file-plus-25 % rule is several times below.
* `nvfp4_load_estimate_prices_exactly_the_copy_the_loader_builds` (CUDA) — the NVFP4 copy equals
  the census.
* `a_non_bf16_snapshot_is_charged_the_bf16_cast_beside_its_source` — f16 and f32 snapshots of the
  tiny llama: every tensor's bf16 cast dense, all but the consumed projections under Q8 and
  NVFP4 (the ineligible 40-row NVFP4 head charged), none off CUDA, only the norms of a bf16
  snapshot with f32 norms, none for a bf16 snapshot.
* `models::qwen35::tests::q8_moe_experts_quantize_from_their_own_slices` — every Q8_0 expert
  gate / up / down of a tiny qwen3_5 MoE (f32, host) tracks its dense twin (relative error
  < 2 %), and the logits track the dense model's (cosine > 0.999).
* `a_qwen35_moe_snapshot_loads_q8_and_decodes` — the MoE snapshot loads `Quantize::Q8` through the
  provider on the host and decodes 4 tokens; `qwen35_q8_estimate_prices_exactly_the_copy_the_loader_builds`
  now checks the MoE copy against the census on the host too.
* `models::llama::tests::ggml_quantizes_each_fused_part_from_its_own_rows` — Phi-3's fused q / k /
  v / gate / up, each Q8_0 and Q4_K part against its dense twin.

**Mutations** (`mutations.log`, the review block). Each was re-run against the final code:

* R1a–R1d (the GGUF working set), R2 (the persisted block), R3 (the affine triple), R5a–R5h (the
  per-projection copy), R0a–R0b (the restructured bound), R8a–R8c (the bf16 cast) — all RED on the
  named unit tests.
* R6b, the device bound without the quantized copy — RED on the **real-weight** Qwen3-8B Q8
  `load_residency` run (`the load peaked at 26818379776 B over its baseline, above the
  20476895970 B it was admitted against`).
* R7, the quantizer fix reverted:
  * **debug:** the four view tests RED, each on candle's size check (`size mismatch 6144 32 32`).
  * **release:** the two model tests RED on their numeric assertions — the silent-corruption case.
* R9a / R9b, the NVFP4 `copy` idiom restored — RED on GPU 1.

### The GGUF measurement

This host has no llama-family GGUF snapshot (only the Prism/Bonsai GGUFs, which keep their
pricing). The Q4_K GGUF above was converted locally from the Qwen3-8B snapshot with candle's GGUF
writer (llama.cpp tensor names and `qwen3` metadata, every matrix Q4_K, norms f32, 4,608,372,128
B) by a throwaway test that is not committed. Both loads generated the 256-token fixture.

## Release notes (runtime-2026.09.1-rc.1)

**Load admission prices what a load really holds (E6).** `LlamaProvider::load` now admits a
quantizing load against the resident source plus the quantized copy of exactly the projections
it quantizes, and a llama-family GGUF load against the dense f32 map it dequantizes the file into
plus the bf16 and quantized copies built beside it. `LlamaProvider::load_memory_estimate` returns
the figure with its parts (`LoadMemoryEstimate`, now `#[non_exhaustive]`).

User-visible effect:

* A **Q8 / Q4 load** that rc.0 admitted and that then ran out of device memory part-way through
  is now refused up front with the typed memory error. On an otherwise idle 96 GB card the
  Qwen3.8-27B Q8 load needs 97.10 GB free (rc.0 charged 69.45 GB; its real peak is 89.27 GB), so
  it is refused once other tenants hold more than about 4.7 GB. The Qwen3-8B Q8 load needs
  27.86 GB (rc.0: 20.48 GB; peak 26.82 GB).
* A **GGUF load** is charged many times its file size: a 4.61 GB Qwen3-8B Q4_K GGUF needs
  57.34 GB of device memory dense (peak 49.47 GB) and 50.82 GB re-quantized to Q8 (peak
  48.26 GB), where rc.0 charged 5.76 GB. On a host device the same working set is charged against
  host memory. Loads that fit are unchanged; loads that could not fit now fail before reading a
  weight instead of part-way through.
* **NVFP4** loads are charged slightly less than rc.0 (27B: 84.09 GB, rc.0 85.08 GB; peak
  72.69 GB), because only the projections NVFP4 quantizes are charged. **bf16** loads and
  Prism/Bonsai checkpoints are unchanged.
* A prepared **qwen3_5** snapshot's persisted `quantization` block is no longer charged (its loader
  ignores the block and loads dense), and a **4-bit MLX-affine** snapshot gets the loader's typed
  refusal instead of a memory refusal.
* An **f16 or f32 safetensors** snapshot is charged the bf16 copy the loader builds beside it
  (2 bytes per dense-kept element), so its loads no longer run out of device memory part-way
  through. bf16 snapshots are unaffected.
* **Fixed:** a qwen3_5 MoE snapshot, or a Phi-3 one, loaded with Q4 / Q8 on a CPU device quantized
  the wrong rows. MoE experts were built from the first expert's weights, and Phi-3's k / v / up
  from q's / gate's. Debug builds hit an assertion instead. It now quantizes each expert and each
  fused part from its own rows. CUDA loads were not affected.

## Commands

```text
# measurement (per load, sequential; GPU 1 checked < 30000 MiB before each load)
CUDA_VISIBLE_DEVICES=1 LOAD_RESIDENCY_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218 \
  LOAD_RESIDENCY_FORMAT=q8|bf16|nvfp4 LOAD_RESIDENCY_OUTPUT=<json> \
  cargo test --release --locked -p candle-llm --features cuda --test load_residency -- --ignored --nocapture --test-threads=1
# the 27B rows: LOAD_RESIDENCY_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# the GGUF rows: LOAD_RESIDENCY_SNAPSHOT=<dir>\Qwen3-8B-Q4_K.gguf (tokenizer.json and
#   tokenizer_config.json beside it), LOAD_RESIDENCY_FORMAT=bf16 (no re-quantization) or q8
```

## Gates

| shell | gate | result |
|---|---|---|
| Git Bash | `cargo fmt -p candle-llm -p core-llm -p candle-quant-kernels -- --check` | ok |
| Git Bash | `cargo test --locked -p candle-llm -p core-llm -p candle-quant-kernels --lib` | ok: 385 + 204 + 40 passed, 0 failed |
| Git Bash | `cargo clippy --locked -p candle-llm -p core-llm -p candle-quant-kernels --all-targets -- -D warnings` | ok |
| Git Bash | `RUSTDOCFLAGS="-D warnings" cargo doc --locked -p candle-llm -p core-llm -p candle-quant-kernels --no-deps` | ok |
| Git Bash | `python scripts/check-workspace.py`; `python scripts/check_docs.py` | ok |
| PowerShell, MSVC 14.44 vcvars64, `CUDA_COMPUTE_CAP=120`, `CUDA_VISIBLE_DEVICES=1` | `cargo clippy --locked -p candle-llm -p candle-quant-kernels --all-targets --features cuda -- -D warnings` | ok |
| same | `cargo test --locked --lib --tests -p candle-llm -p candle-quant-kernels --features cuda` | ok: candle-llm lib 413 passed / 13 ignored, candle-quant-kernels lib 63 passed; all 42 test binaries green (648 passed, 0 failed) |
| same, `--release`, real weights | `static_kv_parity` (4 ignored tests) and `load_residency` (8B bf16 / Q8 / NVFP4, 8B Q4_K GGUF dense / Q8, 27B Q8 / NVFP4) | ok |
