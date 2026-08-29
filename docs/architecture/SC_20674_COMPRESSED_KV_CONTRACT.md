# SC-20674: compressed KV lifecycle contract

**Status:** proposed CPU-only contract. No compressed attention backend is implemented or selected; SC-20673 qualifies candidates.

This ADR is sealed by `sc-20674-compressed-kv-contract.json` and its SHA-256 sidecar. The accompanying checker rejects an incomplete operation matrix, stale source seam, or altered receipt.

## Decision and current seams

`ContiguousKvCache::update` and `PagedKvCache::gather` return dense `[B,Hkv,S,D]` K/V to `sdpa`; Krea `CausalKvCache::window_prev` dequantizes packed storage before dense SDPA. Those dense routes remain authoritative. The planner chooses a route before any append, page allocation, trim, eviction, command encoding, or cancellation registration. An unsupported request records a stable reason then invokes the exact current dense operation with no partial compressed mutation.

```rust
// Proposal only; add after SC-20673 qualifies a representation.
trait AttentionCache {
    fn representation(&self) -> CacheRepresentation; // identity/version/layout/resident bytes
    fn support(&self, request: &AttentionRequest) -> CacheSupport;
    fn route_before_mutation(&self, op: CacheOperation, request: &AttentionRequest) -> CacheRoute;
    fn logical_len(&self) -> usize;
    fn allocated_len(&self) -> usize;
    fn absolute_rope_offset(&self) -> usize;
}
enum CacheRoute { Compressed, DenseFallback { reason: FallbackReason } }
```

Representation metadata includes algorithm identity, wire version, layout, logical/allocated lengths, absolute RoPE origin, and stored resident bytes (never a dense transient). Capability answers decode and large-`S_q` independently and is model/shape/GQA/scale/mask/sink/softcap specific.

## State diagrams

```text
Empty -> preflight -> Compressed staging -> atomic all-layer commit -> ResidentCompressed
                  -> DenseFallback(reason) -> current dense operation -> ResidentDense
                  -> reject -> unchanged + diagnostic
Resident* -> read/support -> compressed dispatch | dense fallback before mutation
Resident* -> trim/rollback/clear/cancel -> shortened/empty/unchanged
Resident* -> clone/split/COW/import/export -> independent or shared immutable snapshot

logical_len = absolute token end; allocated_len = pages/tail capacity; allocated >= resident range
RoPE position = absolute_rope_offset + local chunk position, never a retained-buffer index
Krea retains sink [0,sink) plus tail [tail_base,logical_len); an evicted gap is never reread
```

## Compatibility and migration

| Owner | Current contract | Migration rule |
| --- | --- | --- |
| `ContiguousKvCache` | batch concat, offset, retain rows, truncate/reset, seeded/exported dense prefix | Wrapper first; append, batch split/merge, clone/import/export dense-fallback until qualified. |
| `PagedKvCache` | batch-1 page table/tail, shared immutable prefix, reserved tokens, gather, truncate/reset/drop | Preserve page ownership/COW; matching identity/version only may import pages, else dense materialization before mutation. |
| Krea `CausalKvCache` | post-RoPE K/raw V, committed vs retained sink/tail, packed read dequantizes | Preserve absolute positions/window/reread rejection; current Q8/Q4 is storage-only dense fallback. |
| `sdpa`/`sdpa_capped` | native GQA, causal/additive/sliding masks, `q_len > 8` chunking, eager softcap | Compressed route must advertise every requested semantic or call this dense route first. |

Migration is additive: introduce capability wrappers, route one qualified representation, then versioned import/export/serialization envelopes. Unknown identity/version, cross-backend payload, or missing dense material rejects or falls back deterministically. Candle/CUDA receive no MLX/Metal claim.

## Capability, error, and fallback taxonomy

| Taxonomy | Required action |
| --- | --- |
| `UnsupportedCapability` | Decode/large-`S_q`, GQA/MQA, shape, scale, mask, sink, or softcap unsupported: pre-mutation dense fallback. |
| `RepresentationMismatch` / `VersionMismatch` | Reject import; dense fallback only from verified dense material. |
| `LifecycleViolation` | Partial-layer append, invalid trim/rollback, Krea evicted reread, stale checkpoint: unchanged. |
| `OwnershipViolation` | Wrong thread/JIT/command buffer/cancellation owner, clone/COW/page refcount breach: unchanged. |
| `SerializationFailure` | Missing checksum, unknown version, incomplete payload: no partial install. |
| `DenseFallback` | Diagnostic contains operation, candidate, model/shape, requested capability, lengths/bytes, route and reason. |

Fallback is not compressed success evidence. A compressed failure after mutation is a bug, not fallback.

## Exhaustive lifecycle-to-dense-equivalent plan

| Group | Required test |
| --- | --- |
| append/chunk/decode | Empty/one/many chunks, split-vs-one-shot, decode and large-`S_q` queries; unsupported route leaves lengths/bytes unchanged. |
| lengths/RoPE/bytes | Logical vs allocated length, tail slack, resident bytes, prefix/trim/rollback RoPE, Krea committed/sink/tail positions. |
| attention | MHA/MQA/GQA, scale, causal/additive/sliding masks, sinks, softcap, `q_len > 8` chunking, no compressed claim with materialized `S_q×S_kv`. |
| representation | Identity/version/layout/bytes round trip; corrupted/unknown payload fails closed. |
| lifecycle | trim, speculative rollback, clear, cancellation before/during/after prepare, no partial-layer commit. |
| ownership | clone, batch split/merge/reorder, prefix COW refcounts, page import/export, serialization, thread/JIT/command-buffer/cancellation transfer. |
| Krea | Global/bounded windows, sink, eviction, short-reread rejection, packed-storage fallback, absolute mask positions. |
| fallback | Each taxonomy reason leaves pre-state unchanged and dense output equals direct owner. |

## Candidate boundary and first-push gate

RaBitQ, group-affine/KIVI, and TurboQuant RVQ remain capability inputs. TurboQuant is ineligible while it reconstructs K and retains fp16 V. No candidate promotion precedes SC-20673 model/shape/parity, resident-byte, and long-context evidence. Full-cache dequantize-then-SDPA never satisfies `Compressed`.

```sh
python3.12 scripts/check_compressed_kv_contract.py
python3.12 -m unittest discover -s scripts/tests -p 'test_compressed_kv_contract.py' -v
python3.12 scripts/check_docs.py
python3.12 scripts/check-workspace.py
python3.12 scripts/check_clock_assertions.py --check-baseline .
```
