# sc-24137 — fused RMSNorm+residual / SwiGLU / QK-norm+RoPE on Qwen3.8-27B (RTX Pro 6000 / sm_120)

Write-once evidence for story sc-24137 (epic sc-24128, S9). Hardware: **RTX Pro 6000 / sm_120**
(NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, driver 596.36, CUDA 12.9), GPU 1 via
`CUDA_VISIBLE_DEVICES=1`. Snapshot: `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (bf16, `bonsai-qwen38-parent`). Build:
`--release --features cuda`, `CUDA_COMPUTE_CAP=120`, MSVC 14.44, source `297e0845c` (clean tree,
verified by the harness).

**Later changes, not in the sealed runs.** The Prism packed-operator refactor onto the nvrtc seam
(`aec2d2264`) landed after these runs and is covered by
`cuda_packed_operator_oracles_compile_and_execute_nvrtc`. The review fix pass after it (per-device
resolved function table for the fused kernels, per-key `OnceLock` slots and name-to-source binding
in the seam, the NVFP4 quantizer moved onto the seam, the fused-policy test lock) is covered by the
in-process parity tests rather than a new sealed run:
`fused_primitives::cuda::fused_on_and_off_are_bit_identical_and_both_visible` and the
`fused_decode::cuda_tests` reference parity tests, all green in
`cargo test --locked --lib --tests -p candle-llm -p candle-quant-kernels --features cuda` on GPU 1
(RTX PRO 6000 / sm_120, CUDA 12.9), with the `fused_primitives` binary looped 200× and
`primitives::fused::` 1000× at the default thread count, 0 failures.

Files:

- [`decode-bench/comparison.md`](decode-bench/comparison.md) — the decode bench (sc-24129 harness,
  `scripts/release/decode_bench.py`) with the fused primitives **off** (`CANDLE_LLM_FUSED_KERNELS=0`,
  run `head-297e0845c-fused-off`) and **on** (default, run `head-297e0845c-fused-on`, which also
  carries the in-process `reference_unfused` row: the reference loop with the switch forced off
  for that row only). Sealed run directories beside it (`decode_bench.json`, `run.json`,
  `SEAL.json`, logs).
- [`ac2_fused_parity.log`](ac2_fused_parity.log) — the `#[ignore]`d
  `fused_primitives::cuda::qwen38_27b_greedy_decode_is_token_identical_fused_on_vs_off` run: the
  same 256-token fixture decoded twice in one process, fused off then on.

## AC2 — token identity, fused on vs off

Every MTP-off row is token-identical across the switch: the fused-on `reference` and `step_model`
rows and the in-process `reference_unfused` row all match the fused-off run's reference row
(`match baseline ref = yes`), and the in-process parity test passes with the fused run reporting
**69,888 fused / 0 reference** leaves and the off run **0 fused / 69,888 reference (disabled)**.
The MTP rows diverge from the reference at token 107 in **both** runs, exactly as on the sc-24129
baseline (`docs/migration/evidence/sc-24129`): that is the speculative path's own behaviour, not
the fused primitives (the MTP rows are themselves identical between the off and on runs).

69,888 leaves = 256 forwards (prefill + 255 decode steps) × 273 leaves per forward: 64 layers ×
(input norm + residual-add+post norm + SwiGLU) + 48 DeltaNet layers × 1 gated norm
(`rms_norm_gated` builds on `rms_norm`) + 16 attention layers × 2 QK-norm+RoPE + the final norm.
Every leaf of every forward went through the fused kernels; no refusal and no fallback occurred.

## Throughput (tok/s, 256 greedy tokens after a 97-token prompt)

| row | fused off | fused on | delta |
|---|---:|---:|---:|
| MTP off (reference) — separate runs | 11.38 | **14.17** | **+24.5 %** |
| MTP off (reference) — same process (`reference_unfused` vs `reference`) | 10.79 | **14.17** | **+31.3 %** |
| MTP off (StepModel) | 11.07 | **13.76** | **+24.3 %** |
| MTP K=1 | 12.15 | 15.89 | +30.8 % |
| MTP K=2 | 14.00 | 18.95 | +35.4 % |
| MTP K=3 | 12.84 | 16.86 | +31.3 % |

Reference for the numbers: the sc-24129 head measured 12.17 tok/s (reference) and 11.85 tok/s
(StepModel) on the same fixture; the fused-off rows here are the same binary with the kernels
switched off, so the delta isolates the three fused leaves. Host syncs per token, forwards per
token, acceptance and device memory are unchanged between the two runs (see the table).

## Reproduce

```sh
# Git Bash; the binary is built in PowerShell with vcvars64 14.44 + CUDA_COMPUTE_CAP=120.
cargo test --locked --release -p candle-llm --features cuda --test decode_bench --test fused_primitives --no-run
SNAP=<Qwen3.8-27B snapshot dir>
python scripts/release/decode_bench.py run --binary <decode_bench exe> --run-name head-<sha>-fused-on \
  --runtime-sha <sha> --snapshot "$SNAP" --output docs/migration/evidence/sc-24137/decode-bench/head-<sha>-fused-on \
  --rows reference,reference_unfused,step_model,mtp --drafts 1,2,3 --gpu-index 1
CANDLE_LLM_FUSED_KERNELS=0 python scripts/release/decode_bench.py run --binary <decode_bench exe> \
  --run-name head-<sha>-fused-off --runtime-sha <sha> --snapshot "$SNAP" \
  --output <a dir outside the checkout> --rows reference,step_model,mtp --drafts 1,2,3 --gpu-index 1
python scripts/release/decode_bench.py table <off dir> <on dir> --output comparison.md
CUDA_VISIBLE_DEVICES=1 FUSED_PARITY_SNAPSHOT="$SNAP" <fused_primitives exe> --ignored --exact \
  cuda::qwen38_27b_greedy_decode_is_token_identical_fused_on_vs_off --nocapture
```

(The harness refuses a dirty checkout, so the second run's output goes outside the tree and is
moved in afterwards.)
