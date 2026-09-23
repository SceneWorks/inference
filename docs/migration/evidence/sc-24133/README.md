# sc-24133 — shared on-device sampler: evidence

Hardware: **RTX Pro 6000 / sm_120** (GPU 0, `CUDA_VISIBLE_DEVICES=0`), Windows 11, CUDA lane with
MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`. Epic sc-24128.

## AC2 — Qwen3.8-27B, temperature 0.7 + top-p 0.9, 128 tokens (`ac2-qwen38-27b.json`)

Snapshot `models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, driven
through `generate_step` (the `StepModel` seam), seed 20240923, 81 prompt tokens, one run each after
an 8-token warm-up of both paths.

| sampler path | syncs/token | logits rows to host | device / host draws | decode tok/s |
|---|---|---|---|---|
| `device` | 1.00 (128) | **0** | 128 / 0 | 10.84 |
| `host:reference` (forced, pre-sc-24133 behaviour) | 1.00 (128) | 128 | 0 / 128 | 10.92 |

The device path copies no logits: the one sync per token is the sampled id, the same count
greedy has. End-to-end decode speed is unchanged within run-to-run noise — this checkpoint's decode
is ~92 ms/token, dominated by the weights, and the sampler's cost on either path is a few percent of
that at most (see below). The win this story buys is structural: no vocabulary-wide device→host
copy per token, which the later CUDA-graph and speculation stories need.

```text
BONSAI_QWEN38_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
SAMPLER_EVIDENCE_OUTPUT=docs\migration\evidence\sc-24133\ac2-qwen38-27b.json
cargo test --release --locked -p candle-llm --features cuda --test device_sampler -- --ignored --nocapture --exact ac2_qwen38_temperature_top_p_decodes_without_logits_copies
```

Kernel revision note: the 27B row was measured before two performance-only kernel changes landed on
this branch (warp-aggregated radix histograms; top-p pruning of tokens lighter than
`(1 - top_p) / vocab`). Neither touches the transfer structure (the counters are properties of the
code path, not of the kernel's speed) and both are covered by the AC1 distribution suite, re-run
green after them. The story's lane allowed a single real-weight run, so it was not repeated.

## Sampler cost per token at the Qwen3.8 vocabulary width (`sampler-cost-qwen38-vocab.json`)

One full `sample` call (including the id sync) on a 248 320-wide bf16 row, no model. `flat` puts
every token within ~8 logits of the max (worst case for the thresholds); `peaked` is LLM-shaped.

| logits | sampling | device µs | host µs |
|---|---|---|---|
| flat | temperature | 321 | 1 491 |
| flat | temperature + top-p 0.9 | 2 501 | 15 243 |
| flat | temperature + top-k 20 | 929 | 2 469 |
| flat | temperature + top-k 20 + top-p 0.8 | 1 971 | 2 118 |
| peaked | temperature | 348 | 1 695 |
| peaked | temperature + top-p 0.9 | 853 | 2 975 |
| peaked | temperature + top-k 20 | 690 | 1 355 |
| peaked | temperature + top-k 20 + top-p 0.8 | 1 202 | 2 077 |
| both | greedy argmax (unchanged) | ~190–200 | — |

The kernel is one 1024-thread block per row; its passes are bound by one SM's load latency.

## AC1 — distribution (from the CUDA test run)

`tests/device_sampler.rs`, 100 000 seeded draws per row, chi-square against the host reference's
own shaped distribution (`shaped_candidates`), all `p > 0.01` and no draw outside the reference's
support: vocab 64 and 5 000 × {temperature, low temperature, top-k, top-p, temperature+top-p,
temperature+top-k+top-p} via the batched kernel, and vocab 248 320 × {temperature,
temperature+top-k+top-p} via the per-token `sample` route. The host reference passes the same test
(calibration) and a mismatched distribution is rejected (`p < 1e-6`).
