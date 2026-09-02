# SC-20684: Krea Realtime compressed-KV source map

**Status:** experimental retained Metal kernel and terminal receipt producer implemented, off by default. Focused Metal JIT/oracle dispatch was previously verified, but no full real-weight generation, performance, memory, or quality receipt has been produced. The complete launcher remains unrun until the coordinator owns the Metal lane.

The sealed companion is [`sc-20684-krea-realtime-compressed-kv-contract.json`](sc-20684-krea-realtime-compressed-kv-contract.json). It binds the POC to the checked-out Krea seams and requires an evidence receipt before any product claim.

## Proven current route

Krea's `krea_realtime_14b` T2V, I2V, and V2V modes share one persistent self-attention cache; prompt cross-K/V is separate and out of scope. `PackedKv::pack` stores post-RoPE K and raw V from `[B,H,S,D]` along **D**, as uint32 words plus bf16 scale/bias `[B,H,S,D/group]`. `window_prev` currently gathers then dequantizes a full dense cache window before Wan's fused SDPA. `denoise_chunk_inner` supplies a frame-aligned absolute RoPE offset and optional block-causal mask; `run_ar_loop_conditioned` has readonly denoise followed by exactly one clean-context/final append with cancellation checks.

The Krea 14B backbone is normally B=1, H=40, D=128 and uses group 64. `S_q` is request-derived (the canonical three-frame chunk is 4680); `S_kv` is the retained history plus current chunk. The POC keeps that shape identity and implements the product's nonzero block-causal rule analytically from O(S) query/key position vectors, never by passing an allocated mask tensor to the packed kernel.

## Existing evidence boundary

Existing Q8 storage evidence applies to the **old dequantize-then-SDPA** route only, including its recorded quality cost. Q4 is unmeasured. The POC's focused Metal JIT/oracle test proves only that Q8/Q4 kernels compile, dispatch, retain their handle, and match the packed oracle on its test geometries; it does not establish real-model performance, resident-memory reduction, or image/video quality.

## Frozen upstream comparison and decision

VeloxQuant-MLX is frozen at `v0.65.0`, `54989ee223611627592f7f9bd925e924658f1f22`. Its scalar group-affine path groups K on tokens, whereas Krea groups K and V on D. Its RaBitQ prefill is a useful large-`S_q` online-softmax shape (32 query rows, D <= 128) but is a different signed/codebook format and unmasked.

Accordingly the implementation is Krea-owned: `compressed_kv.rs` packs K/V in Krea's D-axis affine physical layout (Q8 default, Q4 only with an explicit quality-arm acknowledgement), reads codes and bf16 metadata directly, and streams tiles through online softmax. It does not call the upstream kernels. `KreaPackedMetalKernel` owns MLX's `MetalKernel` object, supplies a simdgroup-matrix MSL body, and is retained by the cache rather than recreated per layer.

## Experimental source POC

`ExperimentalCompressedKvConfig::default()` is disabled. `CompressedKvCache::decide` checks the feature flag, Q4 acknowledgement, compiled handle, B=1/D=128/group=64, and exact mask descriptor **before append/trim changes packed state**. Device dispatch additionally validates every packed/current/metadata dtype and dimension, exact history/current position-vector lengths, and a nonzero product block size before lazy JIT. Unsupported conditions return a named dense fallback reason for the existing route.

`append_after_decision` quantizes a full K/V pair before replacement; no half-written layer is observable. `trim_prefix` copies only retained packed words and metadata, checks cancellation before state replacement, and does not dequantize a historical window. `tiled_online_attention` is the CPU parity oracle. The retained Metal path uses `KreaPackedMetalKernel::dispatch` over actual `PackedKv` uint32/bf16 arrays, current K/V, and O(S) global position vectors; it allocates neither a dense historical K/V window nor an `Sq × Sk` score/mask array.

The Metal POC accepts only an already-physical global packed window. It rejects sliding/sink gathers before dispatch, so MLX lazy-JIT or command-buffer failure cannot strand a partially gathered cache; the caller can retry the unchanged dense path from the same cache state. Its bounded MSL tile is eight queries by eight keys, with 256 threads (eight simdgroups), real `simdgroup_matrix` MMA for the score tile, lane-owned four-channel value accumulators, and online max rescaling on every key tile.

Absolute RoPE remains the Wan/Krea producer's responsibility: keys passed to append are post-RoPE and `query_start`/`key_start` make block-mask comparison global. The opt-in `CausalPackedAttention` seam now wires the retained Krea kernel into Wan's per-layer causal forward. It only dispatches the exact B=1/H=40/D=128 product scale and represents the product block-causal rule from absolute position vectors; missing packed history, Q4 without acknowledgement, unsupported geometry/scale, or kernel setup failure stays on the pre-existing dense route. Every lazy custom-kernel result is forced before the caller can commit new cache state, so a JIT or command-buffer failure retries from the unchanged dense cache.

## Receipt and migration contract

A same-run device receipt must name the model/snapshot, deterministic request input, complete source-owned five-step schedule, exact B/H/Sq/Skv/D and mask, representation/version, compiled-handle identity, packed persistent bytes, retained-handle bytes, bounded scratch bytes, zero/nonzero dense-window and score-matrix bytes, independent load/conditioning/generation/decode/first-frame/cold-forward/steady-forward/append timings, paired allocator/process-memory samples, fallback reason, latent parity, decoded spatial/temporal quality, generated review artifacts, and cancellation/release outcome. Q4 cannot be promoted using Q8 evidence. Dense full-cache dequantize-then-SDPA remains a non-goal for compressed-domain execution.

The source tests cover Q8 tiled/tail/block-mask/outlier parity against an independent dense oracle, arbitrary append boundaries, packed trim, disabled pre-mutation fallback, and cancellation scratch cleanup. The focused Rust suite and strict package clippy pass locally; the Python checker rejects missing source mappings, fallback axes, receipt axes, checksum drift, or dense-window/score-route needles.

## Real-weight campaign handoff

The parent command is `scripts/sc20684_krea_realtime_campaign.py`. For each T2V/I2V/V2V × Q8/Q4 cell it launches the ignored `generate_smoke::sc20684_packed_campaign_observer` test in two fresh processes: a paired packed-plus-dense correctness run and an independent dense-read-window baseline for uncontaminated whole-generation memory/timing comparison. It passes only a run nonce, measurement role, selected arm, pinned snapshot path, and new external artifact directory. T2V uses a fixed prompt; I2V VAE-encodes a deterministic gradient still; V2V VAE-encodes a deterministic 25-frame motion clip with fixed strength and seed. Every arm runs all five source-owned Self-Forcing steps at 832×480×25: T2V/I2V use the configured `[1000, 937, 833, 625, 0]` list, while V2V records and validates the actual strength-0.6 warped schedule `[882.3529052734375, 803.5714111328125, 681.8181762695312, 468.75, 0]`. The geometry distinguishes seven total output latents from generated latents: T2V/V2V generate all seven, while I2V follows the product route exactly by warming one reference latent and generating the remaining six (two full AR chunks and ten denoise progress steps).

The test, not the parent, reads the checked-out source hashes, snapshot metadata, toolchain, Mac hardware model, actual Metal chipset, live cache receipt, the accepted-forward count for every distinct per-dispatch `(Sq, Skv)` pair, actual packed-vs-dense latent parity, decoded RGB/temporal quality, fallback counters, pre-cancelled route result, Darwin physical footprint, MLX allocator samples, phase timings, and generated review-frame identities. Timing reports both whole-generation mean output rate (including the non-streaming VAE decode) and a separately named steady-denoise-equivalent rate derived from the post-warmup full AR chunk; it never labels denoise-only throughput as delivered output FPS. The paired quality gate freezes an additional packed-kernel mean RGB budget of 1.0/255 for Q8 and 3.0/255 for the separately acknowledged Q4 arm; Q8 is intentionally tighter than the previously recorded 1.20–2.79/255 Q8-versus-bf16 cache-tier drift because this POC is compared against the same tier's dequantize-then-attend route. It emits one `SC20684_KREA_PROVIDER_OBSERVATION` JSON line only after those values and the external artifacts are terminal.

The reducer rejects malformed or non-terminal evidence: missing/duplicate cells, identity disagreement, synthetic or mode-substituted input, incomplete schedules, conflated timing, inadequate memory-sampling coverage, contradictory status/measurement pairs, missing or hash-drifted review artifacts, or Q8/Q4 substitution. It does **not** erase a terminal negative result. A failed Q4 quality gate, a named dense fallback, zero accepted packed dispatches, nonzero dense-window/score bytes, failed cancellation/release evidence, or a source-owned post-observation assertion is sealed as an arm-level No-go with its raw process exit code and transcripts. Q4 is selected only when the launcher's explicit `KREA_SC20684_Q4_QUALITY_ARM=acknowledged` selector is present; this acknowledges the distinct arm but cannot turn a failed measured quality row into a passing one.

The decision policy is source-frozen before the first product process starts and source identity is rechecked before publication. Material memory reduction requires at least 256 MiB **and** 5% in both Darwin physical-footprint peak and MLX sampled-footprint peak. Throughput-neutral requires at least 0.95× the dense baseline's delivered mean output FPS and steady-denoise-equivalent FPS, with request first-frame latency no more than 5% slower. Every arm additionally requires a compiled/used packed handle, zero dense-window and score-matrix bytes, parity and quality passes, no fallback, clean cancellation, and verified candidate/baseline release. The sealed receipt publishes each arm decision, separate Q8/Q4 decisions, the overall decision, and the complete exact geometry plus hardware identity for every eligible arm.

This schedule varies request mode, Q8/Q4 cache tier, and the request-derived `Sq`/`Skv` exercised by every accepted packed dispatch in the complete T2V/I2V/V2V routes; repeated denoise forwards at one coordinate are counted rather than collapsed into an inferred summary shape. It intentionally does not generalize one run across alternate head counts, head dimensions, group sizes, tile shapes, window policies, or GPU families. Those fixed and unswept axes are named in `decisionPolicy.coverage`; eligibility is limited to an exact measured arm geometry, its per-dispatch coordinate inventory, and Metal device. A future wider sweep must add real rows for those axes rather than relabel this six-cell result.

The publisher preserves hash-identified raw stdout/stderr and all six verified artifact sets, writes both a receipt checksum and a whole-tree checksum manifest, and publishes the evidence directory outside the repository only by an atomic final rename.

```sh
python3 scripts/sc20684_krea_realtime_campaign.py \
  --snapshot /Volumes/Models/huggingface/hub/models--SceneWorks--krea-realtime-14b-mlx/snapshots/e68e9a3d98187fdf6936838ffcf6df5aa48d6626/q4 \
  --output /Users/michael/.codex/worktrees/epic20669/evidence/sc20684/campaign-$(date +%Y%m%dT%H%M%S) \
  --product-command 'cargo test -p mlx-gen-krea-realtime --test integration generate_smoke::sc20684_packed_campaign_observer -- --ignored --nocapture'
```

## Pre-campaign validation

```sh
python3 scripts/check_sc20684_krea_realtime_contract.py
python3 -m unittest scripts/tests/test_sc20684_krea_realtime_contract.py -v
python3 -m py_compile scripts/check_sc20684_krea_realtime_contract.py scripts/tests/test_sc20684_krea_realtime_contract.py
python3 scripts/check_docs.py
python3 scripts/check-workspace.py
python3 scripts/check_clock_assertions.py --check-baseline .
git diff --check origin/feature/sc-20669-fused-compressed-kv-cache...HEAD
```
