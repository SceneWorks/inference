# sc-24136 — fused NVFP4 decode GEMV on Qwen3.8-27B (RTX Pro 6000 / sm_120)

Write-once evidence for story sc-24136 (epic sc-24128, S8). Hardware: **RTX Pro 6000 / sm_120**
(NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, driver 596.36, CUDA 12.9), GPU 1 via
`CUDA_VISIBLE_DEVICES=1`. Snapshot: `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (bf16 checkpoint, `bonsai-qwen38-parent`), loaded with
**NVFP4 projections** (quantized at load, sc-24135). Build: `--release --features cuda`,
`CUDA_COMPUTE_CAP=120`, MSVC 14.44, source `43f5c6f18` (clean tree, verified by the harness).

Files:

- [`decode-bench/comparison.md`](decode-bench/comparison.md) — the decode bench
  (`scripts/release/decode_bench.py run --format nvfp4`) with the GEMV **off**
  (`CANDLE_LLM_NVFP4_GEMV=0`, run `head-43f5c6f18-nvfp4-gemv-off`) and **on** (default, run
  `head-43f5c6f18-nvfp4-gemv-on`, which also carries the in-process `reference_cublaslt` row: the
  reference loop with the GEMV forced off for that row only). Sealed run directories beside it.
- [`parity.json`](parity.json) — AC1: the GEMV against the dequantize-then-matmul reference on every
  Qwen3.8-27B NVFP4 projection shape and edge shapes, rows 1..=8, plus the cuBLASLt W4A4 delta.
- [`kernel_microbench.json`](kernel_microbench.json) — µs per call and effective GB/s, GEMV vs
  cuBLASLt W4A4, every 27B shape × rows 1..=8.
- [`nvfp4_gemv_tests.log`](nvfp4_gemv_tests.log) — the console of the release
  `tests/nvfp4_gemv.rs` binary that wrote both JSON files (all four tests, `--include-ignored`).

## The kernel

`candle_quant_kernels::nvfp4_gemv` (`nvfp4_gemv.cu`, compiled through the nvrtc compile-once seam,
floor sm_80): one launch per projection for 1..=8 **unquantized** bf16 token rows against the
resident packed weight. A block of 8 warps owns 16 output rows and splits `K`; each lane loads 8
packed bytes of two weight rows (one 16-column block, so one UE4M3 scale per row) and 16 bf16 of
one activation row per 64-column unit, dequantizes **exactly** to bf16 (E2M1 × E4M3 fits bf16's 8
significant bits) and issues `mma.m16n8k16` (bf16 in, f32 accumulate) with the activation rows in
the MMA's 8-wide `N`; the per-warp tiles are summed in shared memory, scaled by the per-tensor
scale and rounded once to bf16. The cost is flat in the row count (below).

A first, SIMT variant (f32 FMAs, 1/2/4 rows per warp) was measured and dropped: it reached ~1.1–1.3
TB/s at one row but scaled linearly with the row count (it lost to cuBLASLt on the `lm_head` from
2 rows and on the MLP from 8). Only the tensor-core kernel ships.

**Dispatch** (`candle-llm::primitives::nvfp4_path`, called from `Projection::Nvfp4::forward`): the
GEMV when the switch is on (default) and the input passes `check_nvfp4_gemv` (CUDA, same device,
bf16, 1..=8 rows, matching `K`); otherwise cuBLASLt W4A4, recorded with the reason (`rows`,
`dtype`, `shape`, `device`, `not_cuda`, `disabled`, or the seam's compile error label).
`CANDLE_LLM_NVFP4_GEMV=0` / `set_nvfp4_gemv(Some(false))` selects cuBLASLt everywhere.

## AC2 — decode tok/s, GEMV vs cuBLASLt W4A4 (256 greedy tokens after a 97-token prompt)

| row | cuBLASLt W4A4 | fused GEMV | delta |
|---|---:|---:|---:|
| MTP off (reference) — separate runs | 9.80 | **19.85** | **+103 %** |
| MTP off (reference) — same process (`reference_cublaslt` vs `reference`) | 9.55 | **19.85** | **+108 %** |
| MTP off (StepModel) | 10.13 | **20.44** | **+102 %** |
| MTP K=2 | 14.07 | **20.88** | **+48 %** |

The GEMV is faster, so it is the default. For reference, the bf16 checkpoint (no NVFP4) measured
14.17 tok/s on the same fixture with the S9 fused primitives on (`docs/migration/evidence/sc-24137`).

**Telemetry.** GEMV-on MTP-off rows report `102256 gemv / 400 cuBLASLt (rows)`: the 400 are the
prefill's 97-row projections (64 layers' q/k/v/o or DeltaNet qkv/z/out plus the MLP), which run
cuBLASLt by the row rule; the `lm_head` sees only the last prefill position and every decode step
(255 × 401 + 1) runs the GEMV. The GEMV-off rows report `0 gemv / 102656 cuBLASLt (disabled)`.
MTP K=2 with the GEMV: `80245 gemv / 408 cuBLASLt (rows)` (the target prefill plus the MTP head's
prefill; verify passes are 3 rows, served by the GEMV).

**Greedy tokens** (lossy either way; not an AC): the GEMV and the cuBLASLt path first diverge at
token **48**; against the bf16 checkpoint's reference the GEMV first diverges at **52**, cuBLASLt at
**48** (sc-24135 recorded 48 for cuBLASLt). The cuBLASLt path is deterministic across processes (the
in-process `reference_cublaslt` row matches the separate run token for token). A side effect worth
recording: cuBLASLt quantizes the activation with one per-tensor scale across all rows, so an MTP
verify (3 rows) computes different numbers for the same position than a 1-row step — its MTP K=2 row
diverges from its own reference at token 7 with acceptance 0.718; the GEMV is row-independent
(beyond f32 accumulation order), and its MTP K=2 row stays with its reference to token 63 at
acceptance 0.550.

Host syncs per token (candle-llm's own counter), forwards per token and device memory are
unchanged; the cuBLASLt path's own per-projection amax readback is inside `candle-quant-kernels`
and not counted there.

## AC1 — parity against the dequantize-then-matmul reference

Reference: f64 `x · dequant(W)ᵀ`, the weight read back through the codec's canonical layout
(`Nvfp4Weight::to_host`), sharing none of the kernel's layout arithmetic. Weights Normal(0, 0.02)
bf16 quantized at load, activations Normal(0, 1) bf16, rows 1..=8 for every shape. Every row of
every projection is checked except the 248320-row `lm_head` (first and last 128-row scale atoms plus
every 16th row: 15.8 k rows across all 1940 row atoms).

**Declared tolerance** (`GEMV_REL_RMS_TOL`, `gemv_abs_bound`): relative RMS ≤ 2⁻⁷, and per element
`|y − ref| ≤ 2⁻⁷·|ref| + K·2⁻²⁴·Σₖ|x·W|` — twice bf16's worst-case output rounding (half an ulp is
up to 2⁻⁸ relative) plus the worst-case f32 dot-product accumulation error. Measured worst case over
all 104 (shape, rows) cases: **rel-RMS 1.91e-3** (≈ 4× inside), **max error 0.49 of the per-element
bound**, max |err| 0.031.

| projection | [N, K] | GEMV rel-RMS (max over rows 1..=8) | max err / bound | cuBLASLt W4A4 rel-RMS vs same reference | GEMV vs cuBLASLt rel-RMS |
|---|---|---:|---:|---:|---:|
| self_attn.q_proj (+gate) | [12288, 5120] | 1.67e-3 | 0.307 | 9.69e-2 | 9.69e-2 |
| self_attn.k_proj / v_proj | [1024, 5120] | 1.71e-3 | 0.301 | 9.92e-2 | 9.89e-2 |
| self_attn.o_proj / linear_attn.out_proj | [5120, 6144] | 1.66e-3 | 0.260 | 9.73e-2 | 9.72e-2 |
| linear_attn.in_proj_qkv | [10240, 5120] | 1.67e-3 | 0.307 | 9.61e-2 | 9.63e-2 |
| linear_attn.in_proj_z | [6144, 5120] | 1.71e-3 | 0.303 | 9.62e-2 | 9.63e-2 |
| mlp.gate_proj / up_proj | [17408, 5120] | 1.67e-3 | 0.306 | 9.65e-2 | 9.63e-2 |
| mlp.down_proj | [5120, 17408] | 1.68e-3 | 0.107 | 9.78e-2 | 9.79e-2 |
| mtp.fc | [5120, 10240] | 1.68e-3 | 0.206 | 9.84e-2 | 9.85e-2 |
| lm_head | [248320, 5120] | 1.66e-3 | 0.306 | 9.62e-2 | 9.59e-2 |
| edge K=17 | [16, 17] | 1.91e-3 | 0.481 | 1.21e-1 | 1.17e-1 |
| edge K=80 (pads to 96) | [96, 80] | 1.88e-3 | 0.486 | 1.09e-1 | 1.12e-1 |
| edge K=1000 | [32, 1000] | 1.79e-3 | 0.435 | 1.25e-1 | 1.25e-1 |
| edge K=5130 | [128, 5130] | 1.76e-3 | 0.241 | 9.93e-2 | 9.96e-2 |

The cuBLASLt column is the W4A4 path's distance from the *unquantized-activation* reference — its
activation quantization error, which the GEMV does not have. That is also the size of the
GEMV-vs-cuBLASLt delta (~0.1 rel-RMS on these random activations).

## Kernel microbench (µs per call, GEMV / cuBLASLt W4A4)

Wall clock over back-to-back calls between two device synchronizes (200 calls; 50 for the head),
after 5 warm-ups. Effective GB/s = (packed weight + scales + bf16 activation + bf16 output) / time.
All 8 row counts are in `kernel_microbench.json`.

| projection | [N, K] | 1 row | GB/s @1 | 3 rows | 8 rows | GB/s @8 | GEMV speedup, rows 1..=8 |
|---|---|---:|---:|---:|---:|---:|---:|
| self_attn.q_proj (+gate) | [12288, 5120] | 23.2 / 127.1 | 1524 | 22.6 / 118.4 | 25.0 / 115.2 | 1429 | 4.61×–5.70× |
| self_attn.k_proj / v_proj | [1024, 5120] | 10.0 / 131.6 | 298 | 10.3 / 135.8 | 11.2 / 139.2 | 273 | 10.73×–13.66× |
| self_attn.o_proj / out_proj | [5120, 6144] | 12.6 / 129.6 | 1410 | 14.4 / 128.9 | 14.5 / 108.1 | 1232 | 7.45×–10.32× |
| linear_attn.in_proj_qkv | [10240, 5120] | 18.6 / 110.2 | 1588 | 24.4 / 262.6 | 24.3 / 257.3 | 1224 | 5.93×–10.78× |
| linear_attn.in_proj_z | [6144, 5120] | 17.3 / 115.3 | 1025 | 14.9 / 115.2 | 14.4 / 142.7 | 1238 | 6.51×–9.88× |
| mlp.gate_proj / up_proj | [17408, 5120] | 30.2 / 130.5 | 1663 | 28.7 / 125.7 | 30.8 / 126.3 | 1642 | 3.82×–4.48× |
| mlp.down_proj | [5120, 17408] | 29.8 / 140.8 | 1682 | 30.8 / 138.2 | 31.0 / 142.9 | 1631 | 4.30×–4.72× |
| mtp.fc | [5120, 10240] | 18.6 / 125.0 | 1588 | 20.1 / 118.2 | 20.6 / 122.6 | 1443 | 5.27×–6.73× |
| lm_head | [248320, 5120] | 522.0 / 733.9 | 1371 | 559.6 / 772.9 | 606.2 / 774.0 | 1186 | 1.12×–1.41× |

The GEMV runs at 1.2–1.7 TB/s on every shape but the 1024-row k/v projections (2.6 MB, launch
bound at ~10 µs) and is flat in the row count; cuBLASLt's ~110–140 µs floor per projection is its
activation quantizer (two launches and one host sync for the per-tensor amax) plus the M-padded GEMM.

## Reproduce

```sh
# Git Bash; binaries built in PowerShell with vcvars64 14.44 + CUDA_COMPUTE_CAP=120.
cargo test --locked --release -p candle-llm --features cuda --test decode_bench --test nvfp4_gemv --no-run
SNAP=<Qwen3.8-27B snapshot dir>
python scripts/release/decode_bench.py run --binary <decode_bench exe> --format nvfp4 \
  --run-name head-<sha>-nvfp4-gemv-on --runtime-sha <sha> --snapshot "$SNAP" --gpu-index 1 \
  --output <dir> --rows reference,reference_cublaslt,step_model,mtp --drafts 2
CANDLE_LLM_NVFP4_GEMV=0 python scripts/release/decode_bench.py run --binary <decode_bench exe> \
  --format nvfp4 --run-name head-<sha>-nvfp4-gemv-off --runtime-sha <sha> --snapshot "$SNAP" \
  --gpu-index 1 --output <dir> --rows reference,step_model,mtp --drafts 2
python scripts/release/decode_bench.py table <off dir> <on dir> --output comparison.md
CUDA_VISIBLE_DEVICES=1 NVFP4_GEMV_PARITY_OUTPUT=parity.json NVFP4_GEMV_BENCH_OUTPUT=kernel_microbench.json \
  <nvfp4_gemv exe> --include-ignored --test-threads=1 --nocapture
```

(The harness refuses a dirty checkout, so both runs write outside the tree and are moved in.)
