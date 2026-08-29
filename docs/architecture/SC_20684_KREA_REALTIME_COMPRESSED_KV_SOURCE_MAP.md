# SC-20684: Krea Realtime compressed-KV source map

**Status:** experimental source POC, off by default; no Metal, model, performance, memory, or quality receipt has been produced.

The sealed companion is [`sc-20684-krea-realtime-compressed-kv-contract.json`](sc-20684-krea-realtime-compressed-kv-contract.json). It binds the POC to the checked-out Krea seams and requires an evidence receipt before any product claim.

## Proven current route

Krea's `krea_realtime_14b` T2V, I2V, and V2V modes share one persistent self-attention cache; prompt cross-K/V is separate and out of scope. `PackedKv::pack` stores post-RoPE K and raw V from `[B,H,S,D]` along **D**, as uint32 words plus bf16 scale/bias `[B,H,S,D/group]`. `window_prev` currently gathers then dequantizes a full dense cache window before Wan's fused SDPA. `denoise_chunk_inner` supplies a frame-aligned absolute RoPE offset and optional block-causal mask; `run_ar_loop_conditioned` has readonly denoise followed by exactly one clean-context/final append with cancellation checks.

The Krea 14B backbone is normally B=1, H=40, D=128 and uses group 64. `S_q` is request-derived (the canonical three-frame chunk is 4680); `S_kv` is the retained history plus current chunk. The POC keeps that shape identity and supports only none or analytic nonzero block-causal masking, never an allocated mask tensor.

## Existing evidence boundary

Existing Q8 storage evidence applies to the **old dequantize-then-SDPA** route only, including its recorded quality cost. Q4 is unmeasured. This POC contains no device output and makes no local performance, resident-memory, image/video-quality, or backend-dispatch claim.

## Frozen upstream comparison and decision

VeloxQuant-MLX is frozen at `v0.65.0`, `54989ee223611627592f7f9bd925e924658f1f22`. Its scalar group-affine path groups K on tokens, whereas Krea groups K and V on D. Its RaBitQ prefill is a useful large-`S_q` online-softmax shape (32 query rows, D <= 128) but is a different signed/codebook format and unmasked.

Accordingly the implementation is Krea-owned: `compressed_kv.rs` packs K/V in Krea's D-axis affine physical layout (Q8 default, Q4 only with an explicit quality-arm acknowledgement), reads codes and bf16 metadata directly, and streams tiles through online softmax. It does not call the upstream kernels or claim their dispatch. The device-facing `RetainedKernelHandle` is opaque and lifetime-owned so a real simdgroup pipeline can be bound without a dangling string-only identity.

## Experimental source POC

`ExperimentalCompressedKvConfig::default()` is disabled. `CompressedKvCache::decide` checks the feature flag, Q4 acknowledgement, compiled handle, B=1/D=128/group=64, and exact mask descriptor **before append/trim changes packed state**. Unsupported conditions return a named dense fallback reason for the existing route.

`append_after_decision` quantizes a full K/V pair before replacement; no half-written layer is observable. `trim_prefix` copies only retained packed words and metadata, checks cancellation before state replacement, and does not dequantize a historical window. `tiled_online_attention` uses 32-row tiles, direct packed-code decode, online max/sum/value accumulation, and a bounded scratch accounting field. It explicitly records zero dense-window and score-matrix bytes. The CPU function is a structural/reference oracle, not a claim that the production MLX route has changed.

Absolute RoPE remains the Wan/Krea producer's responsibility: keys passed to append are post-RoPE and `query_start`/`key_start` make block-mask comparison global. The current transformer is intentionally not switched to this path until a retained compiled handle and a device-produced receipt prove the actual MLX integration. That preserves current mask, head mapping, clean-context recompute, and cancellation behavior without a speculative partial route.

## Receipt and migration contract

A future same-run device receipt must name the model/snapshot, exact B/H/Sq/Skv/D and mask, representation/version, compiled-handle identity, packed persistent bytes, retained-handle bytes, bounded scratch bytes, zero/nonzero dense-window and score-matrix bytes, timing label, fallback reason, parity, quality, and cancellation outcome. Q4 cannot be promoted using Q8 evidence. Dense full-cache dequantize-then-SDPA remains a non-goal for compressed-domain execution.

The source tests cover Q8 tiled/tail/block-mask/outlier parity against an independent dense oracle, arbitrary append boundaries, packed trim, disabled pre-mutation fallback, and cancellation scratch cleanup. They are weightless but are not executed in this resource-held lane; the Python checker rejects missing source mappings, fallback axes, receipt axes, checksum drift, or dense-window/score-route needles.

## Source-only validation

```sh
python3 scripts/check_sc20684_krea_realtime_contract.py
python3 -m unittest scripts/tests/test_sc20684_krea_realtime_contract.py -v
python3 -m py_compile scripts/check_sc20684_krea_realtime_contract.py scripts/tests/test_sc20684_krea_realtime_contract.py
python3 scripts/check_docs.py
python3 scripts/check-workspace.py
python3 scripts/check_clock_assertions.py --check-baseline .
git diff --check origin/feature/sc-20669-fused-compressed-kv-cache...HEAD
```
