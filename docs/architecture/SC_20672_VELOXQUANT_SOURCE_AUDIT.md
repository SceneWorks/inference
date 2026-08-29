# SC-20672: VeloxQuant-MLX source freeze and candidate audit

**Status:** source-frozen, CPU-only audit; no local Metal, MLX, benchmark, or
real-weight result is claimed here.

The receipt-ready companion is
[`sc-20672-veloxquant-provenance.json`](sc-20672-veloxquant-provenance.json).
Its exact bytes are sealed by
[`sc-20672-veloxquant-provenance.json.sha256`](sc-20672-veloxquant-provenance.json.sha256)
and validated by
[`scripts/check_veloxquant_source_audit.py`](../../scripts/check_veloxquant_source_audit.py).

## Immutable source and local provenance

The frozen upstream is the annotated **VeloxQuant-MLX v0.65.0** tag object
`94b7cf8be96331127ef9fd7e04b436232b7882b0`, resolving to commit
`54989ee223611627592f7f9bd925e924658f1f22` (2026-08-28). It is MIT licensed.
The planning baseline was v0.53.0 commit
`92909d441cfe1cad6693d9eec5cbf6f57a1d8ff4`; v0.65.0 is 50 commits later.

The audit was prepared against inference `f32fce06804e21eec86d488c8ba320eaf4cbfe11`
and the product-consumed inference revision
`3775a5f80a07a38071c7859f6ac565bcab5d1c7b`. The contemporary SceneWorks
checkout was `367044f3553bbb89af4a6690aac3db5d0f40263b`, which pins that same
inference revision. The manifest records the pmetal/mlx-rs pin and the exact
macOS, Xcode, Rust, and hardware observation. These host facts identify a
future receipt environment; they are not performance evidence.

`productInferenceRevision` remains immutable provenance for that product pin;
it is not a required local Git object. The audit checker binds its local source
mappings to the checked-out inference files under validation and fails on a
missing path or source needle. That keeps fetch-depth-1 PR checks meaningful
without misrepresenting the historical pin as the current checkout.

The v0.53.0-to-v0.65.0 material change is fused TurboQuant RVQ
quantize-and-pack: it removes temporary index buffers on the *write* path. It
also adds RocketKV/pool/profiling infrastructure. It does not turn the live
TurboQuant path into compressed-domain attention: its current
`update_and_fetch` reconstructs a dense K history and retains fp16 V. No
performance, quality, memory, or portability claim from upstream is promoted
to a SceneWorks result.

## Exact local seams

The MLX LLM cache contract in
`crates/llm/mlx-llm/src/primitives/kv_cache.rs::KvCache::update` returns a
dense `[batch, n_kv_heads, seq, head_dim]` K/V pair. Decoder call sites then
use `primitives/attention.rs::sdpa`; this is the seam that must change for a
direct compressed-domain backend. `paged_kv_cache.rs::PagedKvCache::gather`
also reconstructs a contiguous K/V pair before stock SDPA, so it cannot be
treated as direct paged attention.

Persistent media caches are deliberately not folded into that LLM trait:

| Path | Existing behavior | SC-20672 classification |
| --- | --- | --- |
| `mlx-gen-krea-realtime/src/causal.rs::CausalKvCache` | Retains dense or group-affine packed post-RoPE K/raw V; packed reads dequantize the window before dense SDPA. | Real resident storage and lifecycle baseline; **not** compressed-domain attention. |
| `mlx-gen-flux2/src/kv_cache.rs::Flux2KvCache::apply` | Extracts and splices reference-image K/V per layer. | Persistent reference-cache seam, separate adapter required. |
| `mlx-gen-wan/src/pipeline.rs::build_cache` | Evaluates per-block text cross-K/V once per generate. | Persistent text-cache seam, separate adapter required. |

The current pmetal MLX SDPA wrapper also has a correctness mitigation for
multi-head power-of-two head dimensions when `q_len > 8`: it chunks prefill
queries. A replacement dispatch must preserve its causal/additive/sliding mask
semantics and that correctness boundary until independently superseded.

## Upstream classification

| Mechanism | Immutable mapping | Classification | Audit result |
| --- | --- | --- | --- |
| TurboQuant RVQ cache | `cache/turboquant_rvq_cache.py::TurboQuantRVQKVCache.update_and_fetch` | Live cache storage, unsafe for promotion | Keys are genuinely two-stream packed; every fetch rebuilds full dense K and V is fp16. Excluded as a direct backend without a new fused read path. |
| RVQ quantize-and-pack | `metal/_rvq_quant_pack.py::rvq_quant_pack` | Live write-path helper on supported dimensions | New since planning; removes write intermediates only. It does not satisfy E3 by itself. |
| MLX-LM cache hook | `integration/mlx_lm_patch.py::patch_model_kv_cache` | Live cache-factory hook | It replaces `model.make_cache`; it does not replace attention dispatch. |
| VecInfer fused SDPA | `metal/fused_sdpa.py::metal_fused_sdpa` | Standalone callable kernel | Its dispatcher patch is explicitly a no-op in the live generation loop because the cache still returns fp16 K/V. |
| Group-affine decode | `metal/_scalar_attend.py::scalar_fused_decode_attend` | Standalone callable kernel | Candidate mechanism only; no current cache wrapper routes decoder attention to it. |
| RaBitQ decode | `metal/_rabitq_attend.py::rabitq_fused_attend` | Standalone callable kernel | Direct packed-key / online-softmax mechanism, limited to supported shapes (`D <= 256`). |
| RaBitQ tiled prefill | `metal/_rabitq_prefill.py::rabitq_prefill_attend` | Standalone callable kernel | Cross-attention mechanism, limited to `D <= 128`, nibble-packed values, and unmasked non-causal semantics. |
| KIVI cache | `cache/kivi_cache.py::KIVIKVCache` | Live storage/cache wrapper | Candidate source material, but not evidence of a direct fused SceneWorks read path. |
| VLM hook | `integration/mlx_vlm_patch.py::patch_vlm_kv_cache` | Live cache-factory hook | Demonstrates wrapper construction only; VLM cache lifecycle still needs an inference-owned adapter. |
| RocketKV | `cache/rocketkv_cache.py::RocketKVKVCache` | True-resident eviction with dense fetch | Excluded: token selection/eviction is not a compressed-domain attention representation. |
| Pool-backed cache and profiler | `memory/pool_backed_cache.py`, `profiling/kv_profiler.py` | Accounting/measurement utilities | Useful supporting instrumentation only; neither changes attention dispatch. |
| Metal tests and benchmark scripts | `tests/metal/*`, `benchmark_scripts/*`, `scripts/metal_*` | Test/benchmark-only evidence | Upstream synthetic results must be reproduced by SC-20673 before they influence candidate promotion. |

The grid-sensitive kernels explicitly encode their dispatch geometry. In
particular, RVQ pack uses one `D`-thread group per vector; RaBitQ decode uses
one multi-SIMD-group team per query; RaBitQ prefill uses tiled query blocks.
SC-20673 must inspect each actual MSL grid/threadgroup pair and tail shape
before a Rust translation. No source finding here overrides the historical
Metal dispatch, RoPE-offset, cache-restore, chunking, or quantized-SDPA fixes:
each has to be preserved in the target contract and independently exercised.

## Candidate matrix and decision boundary

| Candidate | Resident representation / intended route | Eligibility boundary | Provisional disposition |
| --- | --- | --- | --- |
| Group-affine / KIVI-style | Packed K and V plus group metadata; register-dequantized online-softmax decode. | Exact Krea lifecycle and direct decode semantics must be implemented without a full dense window; model-specific group/head/mask support is mandatory. | **Fallback candidate.** It has the closest existing SceneWorks storage/lifecycle analogue, but that analogue currently dequantizes on read. |
| Packed TurboQuant RVQ | Two packed key-code streams plus norm; values require a compressed representation before promotion. | A fused read must avoid `_dequantize_range` and fp16 value residency; preservation of restore, trim, masks, GQA, and chunked prefill is required. | **Not eligible as-is.** Retain only as a write-path/storage reference. |
| Asymmetric RaBitQ | One-bit packed keys and nibble-packed values; direct online-softmax decode and tiled cross-attention. | Decode and prefill shape limits, masks, GQA, quality, and physical resident-byte accounting must pass per model/shape. | **Provisional POC candidate.** It is the only frozen mechanism with direct compressed-domain kernels for both decode and large-query cross-attention; this is an eligibility decision, not a local speed claim. |

SC-20673 may reject every candidate. Stop rather than port if a path retains or
reconstructs full dense historical K/V, materializes an `S_q × S_kv` score
matrix, cannot preserve cache lifecycle/offset/cancellation semantics, lacks
an explicit dense fallback reason, or fails the later independent parity,
quality, resident-memory, and long-context measurements. Full-cache
dequantize-then-SDPA is therefore a recorded non-goal, including the current
Krea packed-read behavior.

All other VeloxQuant registry methods are outside this epic because they are
eviction, token-selection, cross-layer reuse, accounting, or alternate
quantization algorithms rather than the smallest credible set of direct
compressed-domain candidates. They may not be added merely because a registry
entry, cache wrapper, benchmark, or Metal kernel exists.

## Backend and media boundaries

Metal kernels are MLX-only candidates. Candle/CUDA retain correct dense paths;
shared policy does not imply a portable Metal implementation. Media evaluation
is restricted to persistent/reused K/V: Krea Realtime first, then FLUX2
reference K/V and Wan text cross-K/V. Ordinary per-denoise self-attention is
out of scope unless reuse is separately demonstrated.

## First-push gate

Run the CPU-only checks below before push. They verify manifest bytes and
schema-like invariants, local pinned source mappings, Markdown links, and the
repository's fast workspace/tooling guards without compiling or dispatching
MLX/Metal work:

```sh
python3.12 scripts/check_veloxquant_source_audit.py
python3.12 -m unittest discover -s scripts/tests -p 'test_veloxquant_source_audit.py' -v
python3.12 scripts/check_docs.py
python3.12 scripts/check-workspace.py
python3.12 scripts/check_clock_assertions.py --check-baseline .
git diff --check origin/feature/sc-20669-fused-compressed-kv-cache...HEAD
```
