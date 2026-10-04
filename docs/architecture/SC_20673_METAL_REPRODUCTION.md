# SC-20673: VeloxQuant Metal reproduction receipt

## Outcome

The exact VeloxQuant-MLX v0.65.0 commit `54989ee223611627592f7f9bd925e924658f1f22`
was executed from an external bounded checkout on the development Apple M5 Max
(`applegpu_g17s`, 128 GiB unified memory, macOS 26.5.2, Python 3.13.13,
MLX 0.32.2, mlx-lm 0.31.3, mlx-metal 0.32.2). The upstream Metal parity
suite passed **301/301 tests** in 6.75 s. The complete four-command campaign
finished without failures and downloaded no weights.

The raw sealed transcript is
[`receipts/sc-20673-metal-reproduction.json`](receipts/sc-20673-metal-reproduction.json).
Its SHA-256 is recorded in `sc-20673-metal-reproduction.json.sha256`.
The required-axis/result matrix is
[`receipts/sc-20673-coverage.json`](receipts/sc-20673-coverage.json), checked
by the fail-closed `scripts/check_sc20673_receipt.py` validator.

## Upstream benchmark medians (not probe metrics)

The group-affine scalar fused decode path produced the following upstream
medians at `B=1,H=32,D=128,b=2,g=32,nsg=8` against MLX dequantize plus SDPA:

| `S_kv` | baseline (ms) | fused (ms) | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 0.665 | 0.262 | 2.54x |
| 2,048 | 1.869 | 0.531 | 3.52x |
| 8,192 | 6.617 | 1.936 | 3.42x |
| 16,384 | 13.068 | 3.718 | 3.51x |

The same receipt's direct RaBitQ fused-attend benchmark did **not** beat
dequantize plus MLX SDPA on this host. At `S_q=1,H=8,D=128`, fused latency was
0.323/0.435/1.212 ms versus 0.192/0.222/0.428 ms at
`S_kv=512/2,048/8,192`, or 0.60x/0.51x/0.35x. RaBitQ prefill was also slower:
1.898 ms versus 0.446 ms at `(S_q,S_kv)=(256,2,048)`, 7.113 ms versus
1.761 ms at `(256,8,192)`, and 14.438 ms versus 1.830 ms at
`(1,024,8,192)`, or 0.23x/0.25x/0.13x. These are benchmark-only upstream
callables, not SceneWorks live-path measurements.

RaBitQ encode and RVQ quantize+pack did show isolated write-path wins, but
those do not establish compressed-domain attention. The evidence therefore
supports a measured no-go for porting the RaBitQ attention kernels as a default
decode/prefill path on this development GPU, while preserving group-affine as
the candidate requiring independent SceneWorks integration evaluation.

## Fresh-process probe metrics

The campaign invoked `scripts/sc20673_frozen_probe.py` in a fresh child against
the frozen checkout. All values below come from that child; enqueue, first
synchronized evaluation, explicit synchronization, warm synchronized median,
MLX peak delta, and physical bytes are separately labeled in the sealed JSON.

| kernel and geometry | first eval (ms) | warm median (ms) | explicit sync (ms) | first-run estimate (ms) | MLX peak delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| group-affine decode, `B1 H8 Sq1 Skv2048 D128 g32` | 5.715 | 1.551 | 3.235 | 4.164 | 6,152 B |
| RaBitQ decode, `B1 H8 Sq1 Skv2048 D128` | 0.882 | 0.528 | 0.476 | 0.355 | 2,048 B |
| RaBitQ prefill, `B1 H8 Sq256 Skv2048 D128` | 2.933 | 1.922 | 1.928 | 1.011 | 1,568,768 B |
| RVQ quantize+pack, `N2048 D128 b2` | 0.618 | 0.163 | 0.143 | 0.455 | 0 B |

The group-affine probe stores 5,242,880 persistent bytes versus 8,388,608
dense K/V bytes, a **37.5% reduction**. This misses the epic's 40% product
gate at the representative geometry. RaBitQ stores 1,441,856 bytes, an 82.8%
reduction, but its measured decode and prefill attention paths lose to the
dense baseline. RVQ emits 131,072 packed bytes from 524,288 dense input bytes
and avoids 524,288 bytes of `uint8` intermediates, but it is a write-path
helper rather than compressed-domain attention.

## Decision boundary

- **RaBitQ attention: measured no-go** for a default decode or prefill path on
  this development GPU because its latency regresses despite clearing the
  storage threshold.
- **Group-affine attention: candidate for an isolated SceneWorks POC only.**
  The upstream speedups justify testing integration, but this probe misses the
  storage gate and does not cover cache lifecycle, masks, GQA, real weights, or
  end-to-end generation.
- **Product eligibility: pending independent SceneWorks integration.** Neither
  the upstream benchmarks nor these isolated probes constitute product
  evidence or authorize enabling a production path.

## Scope and limitations

- Upstream parity covered scalar group-affine, RaBitQ encode/values/attend/
  prefill, KIVI, TurboQuant kernels, and RVQ quantize+pack, including tails and
  validation cases exposed by the upstream tests.
- Physical formats were inspected in the frozen MSL/source: RVQ uses packed
  `uint32` streams; RaBitQ keys are bit-packed and values are nibble-packed;
  `uint8` index tensors are not counted as sub-byte storage.
- MSL dispatch inspection confirms explicit grid/threadgroup sizing: one
  threadgroup per RVQ vector, SIMD-group teams for RaBitQ decode, tiled
  prefill, and bounded `D` support (`D<=256` decode, `D<=128` prefill).
- GQA ratios greater than one; causal, additive, sliding-window, sink, and
  softcap semantics; RaBitQ prefill at `D=256`; and masked/causal RaBitQ
  prefill remain explicit dense-fallback cases in this evidence boundary.
- The fresh-process probe separates first evaluation, async submission,
  explicit synchronization, warm synchronized dispatch, and MLX allocator
  peak delta. It does not claim a resident SceneWorks cache or real-weight
  generation.
- The older Apple GPU family was unavailable on this host. This is an explicit
  portability blocker/unknown, so no cross-generation claim is made.
- No SceneWorks/inference production implementation was started.

## Reproduction

```sh
git clone --depth 1 --branch v0.65.0 https://github.com/rajveer43/VeloxQuant-MLX.git /private/tmp/sc20673-veloxquant-v0650
git -C /private/tmp/sc20673-veloxquant-v0650 rev-parse HEAD
/private/tmp/sc20673-veloxquant-v0650/.venv/bin/python scripts/sc20673_metal_campaign.py \
  --source /private/tmp/sc20673-veloxquant-v0650 \
  --output docs/architecture/receipts/sc-20673-metal-reproduction.json \
  --timeout 300
```
