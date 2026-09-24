# sc-24140 — terminal measurements at the release candidate: evidence

Epic sc-24128, terminal measurements at release candidate
R = `e9faeabdbf8ded296ec8083828e717cd951240ae` (tag `runtime-2026.09.1-rc.0`). This note covers
two items: the stale AT3 memory assertion, and E6 memory honesty for Q8 loads.

**RTX Pro 6000 / sm_120.** Host: Windows 11. Everything ran on **GPU 1** only
(`CUDA_VISIBLE_DEVICES=1`, RTX PRO 6000 Blackwell Max-Q, 97,887 MiB), one run at a time. Before
each load, GPU 1 used about 889 MiB (a co-tenant). Build: CUDA 12.9, MSVC 14.44 vcvars64,
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
of the run, after the rollback and after one more forward.

**Mutations** (`mutations.log`, all RED on the CPU twin):

* **M1a** — `recurrent_state_bytes` prices one slot too many.
* **M1b** — `static_kv_bytes` prices `capacity − 1` positions.
* **M1c** — the ring is counted only once it is allocated (`memory_bytes` gated on `self.ring`).
  This is the accounting before sc-24131, and the stale assertion held under it.

## Item 2 — E6: Q8 residency against admission

`tests/load_residency.rs` is a new `#[ignore]`d harness. It loads through `LlamaProvider::load`
with `cuda_graphs: Some(false)`, generates 256 greedy tokens, and samples after a device-wide
synchronize. Each sample records:

* **device used** — `cuMemGetInfo` total − free.
* **pool reserved** and **pool used** — the device's current CUDA memory pool, which cudarc
  serves every candle allocation from. `used` is the live allocations.
* the pool's high watermarks.

It also polls `cuMemGetInfo` in the background through the load. Device figures below are
**deltas over the pre-load sample**: 1.79 GB, which is this process's CUDA context plus the
0.93 GB co-tenant. **Admission** is `LlamaProvider::load_memory_estimate`, the function `load`
itself admits with. The harness also prints R's figure. All figures are GB (10⁹ bytes). The JSON
documents are `load-residency-*.json`.

### Qwen3-8B (measured here), payload 16.38 GB

| format | admission at R | admission (this PR) | census | load peak: device Δ / pool live | after load: device Δ / pool live / idle | after 1st decode step: device Δ / pool live | at token 256: device Δ / pool live | decode peak pool live |
|---|---|---|---|---|---|---|---|---|
| bf16 | 20.48 | 20.48 | 16.38 | 16.42 / 16.38 | 16.42 / 16.38 / 0.03 | 16.58 / 16.43 | 16.58 / 16.43 | 16.43 |
| **Q8** | **20.48** | **29.18** | 9.87 | **26.82 / 24.02** | 15.64 / 9.87 / 5.77 | 15.77 / 9.93 | 15.77 / 9.93 | 9.94 |
| NVFP4 | 25.08 | 25.08 | 5.50 | 21.18 / 20.89 | 8.46 / 5.54 / 2.92 | 8.59 / 5.58 | 8.59 / 5.58 | 5.59 |

### Qwen3.8-27B, payload 55.56 GB

| format | admission at R | admission (this PR) | census | load peak: device Δ / pool live | after load | decode peak |
|---|---|---|---|---|---|---|
| bf16 (campaign) | 69.45 | 69.45 | 54.64 | not sampled | 57.08 device used | 58.08 device used |
| **Q8 (campaign)** | **69.45** | **98.97** | 30.24 | not sampled | 42.96 device used | 43.42 device used |
| **Q8 (measured here, provider)** | **69.45** | **98.97** | 30.24 | **89.31 / 85.06** | 41.85 Δ / 32.09 live / 9.75 idle | 32.47 pool live (41.98 Δ) |
| NVFP4 (campaign) | 85.08 | 85.08 | 17.23 | not sampled | 25.14 device used | 26.28 device used |

The campaign rows are the bench path (`Weights::from_dir` + model + MTP). They use absolute device
used, which includes the CUDA context. The measured 27B row is the provider path. The provider
also loads the f32 ViT tower: 460,730,096 params × 4 B = 1,842,920,384 B. That tower is not in
the decoder census.

### Why Q8 sits above its census after load

It is **idle, reusable CUDA-pool reservation, not live memory.**

* **Live memory equals the census, to the byte.**
  * 8B Q8: pool live after load − census = 137,088 B. That is 252 GGML tensors × 544 B of
    candle's CUDA `MATRIX_ROW_PADDING` (512 × 34/32).
  * 27B Q8: the gap is 1,843,142,880 B. That is the ViT tower's 1,842,920,384 B plus 409 × 544 B.
  * 8B NVFP4: the gap is 33,554,432 B, the cuBLASLt handle's 32 MiB workspace.
  * GGML scales and blocks are counted correctly: 8.5 bits per parameter, the census's
    `storage_size_in_bytes`.
  * Dense embeddings and the head are in the census: `other` holds 1.25 GB and dense `lm_head`
    1.24 GB.
* **The gap is the pool keeping freed load memory.**
  * `Weights::from_dir` loads every bf16 source tensor onto the device.
  * `Projection::load_as` quantizes each projection while that whole map is still alive. The
    load's live peak is sources + copy: 16.38 + 7.38 + transients = 24.02 GB on the 8B.
  * When the map drops, about 15 GB of sources return to the pool.
  * The pool does release idle memory at a synchronize. Every sample here synchronizes first, and
    the sample after the provider drops shows the reservation already gone, with no explicit trim.
    Yet 5.77 GB (8B) and 9.75 GB (27B provider) stay reserved after the load. Our reading is that
    the pool gives memory back only in units that are wholly free. Still-live tensors are
    interleaved with the freed sources: the dense embeddings, norms and head (which share storage
    with the source map), and the new copies. So those units cannot be returned.
  * NVFP4 shows the same mechanism (2.92 GB idle). bf16 does not (0.03 GB), because its sources
    *are* its weights.
* **The idle reservation is reusable, not leaked.**
  * Every decode allocation over 256 tokens was served from it: pool reserved did not grow.
  * Request admission already counts it as available. `cuda_usable_memory_bytes` adds
    reserved − used to the driver's free bytes.
  * Dropping the provider returns it. Reserved falls to 0.10 GB on the 8B. What remains is
    candle's grow-only MMQ/MMVQ workspaces.
* **No per-matmul dequantized temporaries.** Decode raised the 8B Q8 pool's live peak by only
  73 MB over the post-load live bytes (KV, activations and the MMVQ Q8_1 workspace). Candle's fast
  MMVQ/MMQ kernels quantize activations to Q8_1 instead of dequantizing weights. A dequantized
  f32 MLP weight alone would be 201 MB. The cuBLAS workspace is inside that 73 MB.

### The E6 defect, and the fix

The steady state is under admission for every format, even at R: 8B Q8 is 15.64 < 20.48. The
**load peak** is not. R priced a CUDA load at payload + 25 % headroom. It added the growing
quantized copy only for NVFP4 (`payload · 9/32`). A GGML Q4/Q8 load builds its copy on the device
the same way, beside the resident sources, but that copy was unpriced:

| load | load peak (device Δ) | admission at R | shortfall |
|---|---|---|---|
| Qwen3-8B Q8 | 26.82 | 20.48 | **6.34** |
| Qwen3.8-27B Q8 | 89.31 | 69.45 | **19.86** |

Both shortfalls are beyond the documented 25 % headroom, which covers constructor casts and the
quantizer's transient f32 copy. So a Q8 load could be admitted and then fail with an OOM part-way
through. **Fix** (`crates/llm/candle-llm/src/provider.rs`):

* `load_memory_requirements` takes the `QuantizedCopy` the load builds:
  * Q8_0: `payload · 17/32` (8.5 of 16 bits).
  * Q4_K: `payload · 9/32`.
  * NVFP4: `payload · 9/32` (unchanged).
  * A Q8_0 repack of a packed MLX-affine 8-bit source: `payload · 17/16`.
* The copy is chosen from `spec.quantize`, or else from the snapshot's persisted `quantization`
  block (a prepared Q4/Q8 snapshot re-quantizes on load). A packed GGUF or Prism checkpoint builds
  none.
* `LlamaProvider::load_memory_estimate` (with `LoadMemoryEstimate`) exposes the figure that `load`
  admits with.

Re-measured with the fix:

| load | admission (this PR) | load peak (device Δ) | margin |
|---|---|---|---|
| Qwen3-8B Q8 | 29.18 | 26.82 | 2.36 |
| Qwen3.8-27B Q8 | 98.97 | 89.31 | 9.66 |
| Qwen3-8B bf16 (unchanged) | 20.48 | 16.42 | 4.06 |
| Qwen3-8B NVFP4 (unchanged) | 25.08 | 21.18 | 3.90 |

The 27B Q8 row above was loaded through the fixed admission on GPU 1, with the 0.93 GB co-tenant
present, so the flagship Q8 load still admits on an idle 96 GB card. The headroom is tight,
though. The card holds 102.64 GB and this process's CUDA context takes 0.85 GB, so admission now
refuses the 27B Q8 load once other tenants hold more than about 2.8 GB. At R the same load was
admitted with up to about 32.3 GB held elsewhere, but it OOMs part-way through beyond about
12.5 GB. So at R, co-tenants between about 12.5 and 32.3 GB were admitted into a failure. The
new bound is conservative in the other direction: a load with a co-tenant between about 2.8 and
12.5 GB would fit, but is now refused up front. That 9.7 GB of margin comes from the 25 %
headroom, which this PR leaves as it was.

A follow-up could shrink the Q8 load peak and the idle reservation together. The loader would
quantize shard by shard and drop each source as it is consumed, instead of holding the whole map.
That is a loader change and is not made here.

**Unit tests** (synthetic geometry, CPU):

* `ggml_admission_prices_the_quantized_copy_beside_the_source`
* `load_quantized_copy_follows_the_request_then_the_persisted_block`
* `q8_load_estimate_prices_the_copy_the_loader_builds` — a bf16 tiny-llama snapshot loaded Q8
  through the provider. The census's Q8_0 bytes are exactly 8.5 bits per parameter and fit the
  priced copy. Sources + copy fit the new bound and exceed R's.

**Mutations** (`mutations.log`, all RED):

* **M2a** — the Q8 copy unpriced again (R's pricing).
* **M2b** — Q8_0 priced at 8/16 instead of 8.5/16.
* **M2c** — the persisted block ignored.
* **M2d** — the device bound drops the copy.

## Commands

```text
# measurement (per format, sequential; GPU 1 checked < 30000 MiB before each load)
CUDA_VISIBLE_DEVICES=1 LOAD_RESIDENCY_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218 \
  LOAD_RESIDENCY_FORMAT=q8|bf16|nvfp4 LOAD_RESIDENCY_OUTPUT=<json> \
  cargo test --release --locked -p candle-llm --features cuda --test load_residency -- --ignored --nocapture --test-threads=1
# the 27B Q8 row: LOAD_RESIDENCY_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

## Gates

| shell | gate | result |
|---|---|---|
| Git Bash | `cargo fmt -p candle-llm -p core-llm -- --check` | ok |
| Git Bash | `cargo test --locked -p candle-llm -p core-llm --lib` | ok: 388 + 207 run, 0 failed |
| Git Bash | `cargo clippy --locked -p candle-llm -p core-llm --all-targets -- -D warnings` | ok |
| Git Bash | `RUSTDOCFLAGS="-D warnings" cargo doc --locked -p candle-llm -p core-llm --no-deps` | ok |
| Git Bash | `python scripts/check-workspace.py`; `python scripts/check_docs.py` | ok |
| PowerShell, MSVC 14.44 vcvars64, `CUDA_COMPUTE_CAP=120`, `CUDA_VISIBLE_DEVICES=1` | `cargo clippy --locked -p candle-llm --all-targets --features cuda -- -D warnings` | ok |
| same | `cargo test --locked --lib --tests -p candle-llm --features cuda` | ok: lib 406 passed / 13 ignored; every integration binary green |
| same, `--release`, real weights | `static_kv_parity` (4 ignored tests) and `load_residency` (8B bf16 / Q8 / NVFP4, 27B Q8) | ok |
