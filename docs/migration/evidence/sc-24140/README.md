# sc-24140 — feature-end review, round 1 (decode routing): evidence

The round's measurement and llama-family findings (A: llama-family NVFP4, B: the decode-perf
matrix and campaign, C: stochastic rows, D: the llama knife-edge gate, E: `REQUIRE_SM120`) have
their own evidence in [`measure/README.md`](measure/README.md).

Epic sc-24128, feature-end fix story. Host: Windows 11, **RTX Pro 6000 / sm_120** (GPU 0,
`CUDA_VISIBLE_DEVICES=0`), CUDA 12.9, MSVC 14.44, `CUDA_COMPUTE_CAP=120`.

## Item 2 — Qwen3.8-27B: the provider's Off path against the reference loop

`provider-off-path-parity-qwen38-27b.json`. A Qwen3.8-27B request with speculation off is now
decoded by the engine with no proposer, on the static KV cache (`step_model` / `static` / `gqa`,
`proposer=none`). This row compares it with the reference `Decode` loop, selected on the **same
provider** with `set_decode_path(DecodePath::Reference)` (`reference` / `growing` / `gqa`).

Setup: `LoadSpec::dense` BF16, the sc-24132 256-token fixture prompt through the chat template,
greedy, `MtpMode::Off`. The comparison covers every streamed event (token id, index, channel,
text), the final text, the reasoning text, the usage and the finish reason.

Both paths attend through un-expanded GQA and both decode one token per step (M = 1), so the
knife-edge exception for verify-shaped forwards does not apply. The requirement is exact
equality.

| | engine (default) | reference (selected) |
|---|---|---|
| path / KV / attention | `step_model` / `static` / `gqa` | `reference` / `growing` / `gqa` |
| generated tokens | 256 | 256 |
| streamed events | 255 | 255 |
| target forwards | 256 (1 prefill + 255 verify steps) | 256 |
| host syncs | 256 | 256 |
| sampler | `device`, 0 logits rows to host | `device` |
| **identical** | **yes** (no divergent event) | |

The 256 tokens produce 255 events because the `<think>` marker token is stripped by the
reasoning segmenter on both paths.

Measured at commit `007c3a8181b2e704e04c50d4ad0e527ebe8bdd6e` on a clean tracked tree. That
commit is this branch merged with the feature head that carries S11 (#1035). The JSON records
`commit` and `worktree_dirty`. Load took 22 s, and each 256-token decode took 17–18 s on both
paths. An earlier run at the pre-merge commit `4180cec58` was identical as well. The command:

```text
BONSAI_QWEN38_SNAPSHOT=E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
SC24140_EVIDENCE_OUTPUT=<worktree>\docs\migration\evidence\sc-24140\provider-off-path-parity-qwen38-27b.json
cargo test --release --locked -p candle-llm --features cuda --test static_kv_parity -- --ignored --nocapture --exact provider_off_path_is_token_identical_to_the_reference_loop
```

The sc-24132 AC1 test was re-run over the new `generate_step`, which is now the engine with no
proposer. Its tokens are identical across reference / static / growing (`gqa`), with the engine's
record convention (`ac1-generate-step-engine-qwen38-27b.log`).

## Mutations

`review-fix-mutations.log` lists one mutation for each added or changed assertion, with its RED
result. The CPU rows are recorded as CPU and the CUDA row as CUDA.

## Item 8 — MLX nucleus in f64

This evidence is CI-only, from the hosted macOS lane. It is recorded in
`docs/migration/evidence/sc-24133/README.md`, in the section "MLX nucleus mass in f64 — macOS-lane
CI evidence (CI-only)".
