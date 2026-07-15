# SC-12111 Candle depthwise-convolution checkpoint

Date: 2026-07-15

## Change

The inference workspace advances its shared Candle pin from
`c1e6756a89faefa888ea57b056394a0619925b87` to
`1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d`, the immutable head commit of
[huggingface/candle#3531](https://github.com/huggingface/candle/pull/3531).

That upstream patch replaces the per-group convolution plus `cat` with a tensor-level weighted-slice
fast path when all of these conditions hold:

- depthwise weights (`c_in_k == 1`);
- no channel multiplier (`c_out == c_in == groups`);
- unit stride and dilation.

SANA-1.6B's Mix-FFN `conv_depth` is exactly this case: f32 input shaped
`[1, 11200, 32, 32]`, weights shaped `[11200, 1, 3, 3]`, padding 1, stride 1, dilation 1.

General grouped convolution and depthwise convolution with non-unit stride or dilation still use
Candle's existing per-group decomposition. This checkpoint does not claim those cases are fixed.

The workspace pin guard in `scripts/check-workspace.py` advances deliberately with the manifests and
lockfile. Candle's newer manifest also resolves `zip` 7.2 for `candle-core` and `fancy-regex` 0.17;
those lockfile changes are part of the reviewed pin transition.

## Exclusive-GPU measurements

Device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (`sm_120`), CUDA 12.9, release build.

| Measurement | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Isolated realistic `conv_depth` | 982 ms/call | 4.042 ms/call | 243x |
| Dense f32 denoise step, real SANA-1.6B at 1024 px | 21.05 s/step | 315.1 ms/step | 66.8x |

Post-fix real-trunk timings, six measured steps after warmup:

| Regime | Time |
| --- | ---: |
| Dense f32 | 315.1 ms/step |
| NVFP4 W4A16 | 316.7 ms/step |
| NVFP4 blanket W4A4 | 8418.3 ms/step |
| NVFP4 mixed W4A4/W4A16 | 4335.2 ms/step |

The removed convolution bottleneck exposes the separate unfused activation-quantization cost tracked
by sc-12078. These SANA numbers do not settle NVFP4 SC#1/SC#2 because SANA has neither the required
bf16 baseline nor a Q4 comparison tier.

## Regression gate

`candle-gen-sana/tests/depthwise_conv_gpu.rs` runs the exact SANA shape in release mode after warmup
and requires a mean below 100 ms/call. The ceiling is intentionally much looser than the measured
4.042 ms so slower supported GPUs and allocator noise do not make the ignored exclusive-GPU gate
flaky, while the pre-fix 982 ms path still fails decisively.

## Validation

```text
python scripts/check-workspace.py --offline
cargo test --locked -p candle-gen-sana
CUDA_COMPUTE_CAP=120 cargo test --locked -j 1 -p candle-gen-sana \
  --test depthwise_conv_gpu --features cuda --release -- --ignored --nocapture
CUDA_COMPUTE_CAP=120 cargo test --locked -j 1 -p candle-gen-sana \
  --test nvfp4_sana_dit_gpu --features cuda --release \
  nvfp4_sana_dit_real_throughput_dense_vs_w4a16_vs_w4a4 -- --ignored --nocapture
```

Results: workspace gate passed; 34 SANA unit tests and focused integration suites passed; the CUDA
regression passed at 4.042 ms/call; the real-weight throughput benchmark passed with the timings above.

## SceneWorks consumption

SceneWorks currently consumes inference tag `runtime-2026.07.3`. Publish this inference change under a
new runtime tag, then advance SceneWorks' `runtime-cuda`/`runtime-macos` pins together in its clean
SC-12111 worktree. No SceneWorks pin has been changed before that release boundary exists.
