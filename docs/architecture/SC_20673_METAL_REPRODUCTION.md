# SC-20673: VeloxQuant Metal reproduction receipt

## Outcome

The exact VeloxQuant-MLX v0.65.0 commit `54989ee223611627592f7f9bd925e924658f1f22`
was executed from an external bounded checkout on the development Apple M5 Max
(128 GB unified memory, macOS 26.5.2, MLX 0.32.2). The upstream Metal parity
suite passed **301 tests** in 7.75 s. No weights were downloaded.

The raw sealed transcript is
[`receipts/sc-20673-metal-reproduction.json`](receipts/sc-20673-metal-reproduction.json).
Its SHA-256 is recorded in `sc-20673-metal-reproduction.json.sha256`.
The required-axis/result matrix is
[`receipts/sc-20673-coverage.json`](receipts/sc-20673-coverage.json), checked
by the fail-closed `scripts/check_sc20673_receipt.py` validator.

## Measured result

The group-affine scalar fused decode path showed the upstream result at
`B=1,H=32,D=128,b=2,g=32,nsg=8`: 2.12x at `S_kv=512`, 3.60x at 2048,
3.42x at 8192, and 3.49x at 16384 versus the upstream MLX baseline.
The same receipt's direct RaBitQ fused-attend benchmark did **not** beat
dequantize plus MLX SDPA on this host: at `S_q=1,H=8,D=128`, fused/packed-V
speedups were 0.71x/0.73x (512), 0.57x/0.63x (2048), and 0.35x/0.37x (8192).
Large-query RaBitQ prefill was also slower than dense SDPA (0.24x, 0.25x,
and 0.13x at `(S_q,S_kv)=(256,2048),(256,8192),(1024,8192)`). These are
benchmark-only upstream callables, not SceneWorks live-path measurements.

RaBitQ encode and RVQ quantize+pack did show isolated write-path wins, but
those do not establish compressed-domain attention. The evidence therefore
supports a measured no-go for porting the RaBitQ attention kernels as a default
decode/prefill path on this development GPU, while preserving group-affine as
the candidate requiring independent SceneWorks integration evaluation.

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
- Compilation, first-dispatch, transient-peak, and command-buffer
  synchronization were not measured separately. Steady-state values are only
  the named upstream benchmark medians; no resident SceneWorks cache or
  real-weight generation was claimed.
- The older Apple GPU family was unavailable on this host. This is an explicit
  portability blocker/unknown, so no cross-generation claim is made.
- No SceneWorks/inference production implementation was started.

## Reproduction

```sh
git clone --depth 1 --branch v0.65.0 https://github.com/rajveer43/VeloxQuant-MLX.git /private/tmp/sc20673-veloxquant-v0650
git -C /private/tmp/sc20673-veloxquant-v0650 rev-parse HEAD
uv run --project /private/tmp/sc20673-veloxquant-v0650 --with scipy --with pytest python scripts/sc20673_metal_campaign.py \
  --source /private/tmp/sc20673-veloxquant-v0650 \
  --output docs/architecture/receipts/sc-20673-metal-reproduction.json
```
