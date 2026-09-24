# sc-24133 — shared on-device sampler: evidence

Hardware: **RTX Pro 6000 / sm_120** (GPU 0, `CUDA_VISIBLE_DEVICES=0`), Windows 11, CUDA lane with
MSVC 14.44 vcvars, `CUDA_COMPUTE_CAP=120`. Epic sc-24128.

## AC2 — Qwen3.8-27B, temperature 0.7 + top-p 0.9, 128 tokens (`ac2-qwen38-27b.json`)

Snapshot `models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, driven
through `generate_step` (the `StepModel` seam), seed 20240923, 81 prompt tokens, one run each after
an 8-token warm-up of both paths. Measured at commit `31fe9b074cdf8182e5dc3094cafa3df098589b4b`,
the PR's final code commit, on a clean tree. The JSON records `commit` and `worktree_dirty`.

| sampler path | syncs/token | logits rows to host | device / host draws | decode tok/s |
|---|---|---|---|---|
| `device` | 1.00 (128) | **0** | 128 / 0 | 11.36 |
| `host:reference` (forced, pre-sc-24133 behaviour) | 1.00 (128) | 128 | 0 / 128 | 11.46 |

The device path copies no logits: the one sync per token is the sampled id, the same count
greedy has. End-to-end decode speed is unchanged within run-to-run noise — this checkpoint's decode
is ~88 ms/token, dominated by the weights, and the sampler's cost on either path is a few percent of
that at most (see below). The win this story buys is structural: no vocabulary-wide device→host
copy per token, which the later CUDA-graph and speculation stories need.

```text
BONSAI_QWEN38_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
SAMPLER_EVIDENCE_OUTPUT=<worktree>\docs\migration\evidence\sc-24133\ac2-qwen38-27b.json
cargo test --release --locked -p candle-llm --features cuda --test device_sampler -- --ignored --nocapture --exact ac2_qwen38_temperature_top_p_decodes_without_logits_copies
```

The test runs with the package directory as its working directory, so give
`SAMPLER_EVIDENCE_OUTPUT` as an absolute path.

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

Tie handling: vocab 64 and 5 000 also run the top-k tie cut (top-k 10 and 16 split the exact tie
between tokens 3 and 11 at the kernel's index cutoff) and a run of eight equal weights. Top-p 0.7
and top-k 5 each put their threshold inside that run, so 4 of the 8 are kept. p-values for these
cases range from 0.24 to 0.39. `tie_cases_split_their_ties_in_the_reference` checks that the
reference splits them this way.

Top-p boundary tolerance (`nucleus_boundary_divergence_is_at_most_one_token`): on exact boundaries
(equal weights of 1.0 with an integer `top_p * count`), the device and host kept sets are the same
size, with no divergence. With top-p placed at each prefix's exact share of the mass and one f32 ulp
either side, the kept sets matched in 15 of 15 cases. The documented tolerance is ±1 token.

## MLX nucleus mass in f64 — macOS-lane CI evidence (CI-only)

The same review commit (`31fe9b074`) moved `mlx-llm`'s `nucleus_select` from an f32 to an f64
running sum, matching `candle-llm`. That changes MLX top-p sampling only where an f32 sum's
rounding had moved the nucleus boundary: a knife-edge boundary may keep one token more or fewer
than the f32 mlx-gen references. MLX cannot be built or run on the Windows dev box, so the only
evidence for the change is the hosted macOS lane of CI. It is **CI-only evidence**. No local MLX
run and no real-weight MLX sampling comparison backs it (feature-end review, sc-24140 item 8).

| where | head | job | `mlx_llm` unit tests |
|---|---|---|---|
| PR #1023 (S5, sc-24133, the change itself) | `4c227be36` (contains `31fe9b074`) | [MLX + Candle Metal packages (macOS)](https://github.com/SceneWorks/inference/actions/runs/35897208137/job/107304179290): success | 291 passed, 0 failed, 7 ignored |
| PR #1031 (S10; its merge `e3248a3a8` is the feature head this round started from; the MLX sampler is unchanged since `31fe9b074`) | `49dece6bb` | [MLX + Candle Metal packages (macOS)](https://github.com/SceneWorks/inference/actions/runs/35926318443/job/107402310102): success | 291 passed, 0 failed, 7 ignored |

Both jobs ran `cargo test --locked --lib --tests -p mlx-llm -p mlx-llm-server -p mlx-gen -p
'mlx-gen-*' -p runtime-macos` on the hosted macOS runner (Metal 320, not the NAX lane). The sampler
tests that exercise the nucleus pass there, including
`primitives::sampler::tests::nucleus_matches_full_sort_for_distinct_weights` and
`primitives::sampler::tests::top_p_restricts_to_nucleus`. These tests check the heap nucleus
against a full sort and check nucleus membership. None of them pins a knife-edge boundary against the f32
references, so the one-token boundary difference itself is documented, not measured.
