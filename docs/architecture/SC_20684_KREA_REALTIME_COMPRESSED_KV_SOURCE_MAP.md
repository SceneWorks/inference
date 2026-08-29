# SC-20684: Krea Realtime compressed-KV source map

**Status:** source-only audit. No Krea compressed-domain kernel is implemented or claimed here.

The sealed companion is [`sc-20684-krea-realtime-compressed-kv-contract.json`](sc-20684-krea-realtime-compressed-kv-contract.json). Its deliberately blocking POC contract cannot become a result receipt until an exact representation, attention route, and device-lane evidence exist.

## Proven current route

Krea has one registered model family, `krea_realtime_14b`, a Wan-2.1-T2V-14B-compatible causal DiT. T2V, I2V, and V2V share its persistent **self-attention** cache. Text cross-K/V is a separate once-per-prompt cache and out of this POC.

| Concern | Current seam | Established behavior |
| --- | --- | --- |
| Creation | `causal.rs::CausalKreaTransformer::new_cache` | One `CausalKvCache` slot per 40-layer DiT, carrying `kv_cache_quant`. |
| Stored layout | `PackedKv::pack`, `StoredKv::store` | K and V are both MLX affine packed from `[B,H,S,D]` along **D**: uint32 payload plus bf16 scale/bias `[B,H,S,D/group]`; Q8/Q4 use group 64. |
| Append/lifetime | `CausalKvCache::append`, `window_prev` | Post-RoPE K/raw V commits once per chunk. Logical tokens only grow; lazy eviction preserves sink plus rolling tail. |
| Read | `StoredKv::dense`, `window_prev` | Packed window is gathered then fully dequantized K/V per layer. `eval_retained` evaluates stored packed arrays, not the read transient. |
| Attention | Wan `SelfAttention::forward_causal` | Concatenates dense previous/current K/V and calls MLX fused `scaled_dot_product_attention`, with additive block-causal mask when supplied. |
| Geometry/RoPE | `denoise_chunk_inner`, `resolve_request_config` | Start must equal committed tokens and be frame aligned; `start_frame = start_token/frame_seq_length`; token count derives from latent H/W. |
| Commit/cancel | `run_ar_loop_conditioned` | Read-only denoise until the one chunk commit (final pass or clean-context recompute); cancellation is checked before every chunk and denoise step. |

The standard backbone is B=1, H=40, D=128, bf16 Q/K/V. At canonical 832x480 a three-frame block has `S_q=4680`; `S_kv` is request/history/sink/final-chunk dependent. The source permits an additive block-causal mask: an implementation must not infer maskless support merely because normal aligned blocks are commonly all-allowed.

## Existing evidence boundary

`KvCacheQuant::Q8` is an opt-in resident-storage trade with recorded bounded real-weight drift; Q4 is explicitly unmeasured. Existing source comments and tests concern the current dequantize-then-SDPA route only; they authorize neither a fused performance claim nor quality parity for a new representation.

## Frozen upstream comparison and decision

The inspected upstream is VeloxQuant-MLX `v0.65.0`, commit `54989ee223611627592f7f9bd925e924658f1f22`. Its group-affine scalar kernel requires K token-axis groups and V channel-axis groups with uint8 codes and fp32 scale/zero. Krea Q8/Q4 groups both K and V along D with MLX uint32 payload and bf16 scale/bias, so the formats are not interchangeable.

The frozen RaBitQ tiled prefill kernel is the only examined large-`S_q` strategy: it tiles 32 query rows and requires D divisible by 8 and `D <= 128`. Krea's D=128 meets that bound, but the kernel requires one-bit sign K plus nibble-packed codebook V and is explicitly unmasked cross-attention. It cannot consume Krea Q8/Q4, cannot preserve a nontrivial additive block-causal mask, and has no Rust/MLX dispatch owner here.

The correct source-only result is therefore **no implementation route yet**. A future POC may add a new representation only after it proves, before cache mutation:

1. Exact post-RoPE K/raw-V conversion with no dense historical K/V retained.
2. Direct support or named pre-mutation dense fallback for all-allowed/masked, sink/window, short-final-chunk, RoPE-offset, readonly/recompute, append, eviction, and cancellation cases.
3. Tiled dispatch only on compatible-format, `D=128`, unmasked geometry, never by materializing an `S_q x S_kv` score matrix.

This is source structure, not local performance, resident-byte, or image/video-quality evidence.

## First implementation blocker

The kernel story must choose and implement one Krea-owned physical format and lifecycle atomically. Current Q8/Q4 cannot feed either frozen direct kernel; silently dequantizing is the existing dense fallback. The future receipt producer must record request-derived `S_q`, `S_kv`, B/H/D, mask capability, representation identity, separate persistent/transient bytes, parity, quality, cancellation, and any fallback reason.

## Source-only validation

```sh
python3 scripts/check_sc20684_krea_realtime_contract.py
python3 -m unittest scripts/tests/test_sc20684_krea_realtime_contract.py -v
python3 scripts/check_docs.py
python3 scripts/check-workspace.py
python3 scripts/check_clock_assertions.py --check-baseline .
git diff --check origin/feature/sc-20669-fused-compressed-kv-cache...HEAD
```
