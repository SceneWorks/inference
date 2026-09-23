# sc-24135 — NVFP4 projections on Qwen3.8-27B (RTX Pro 6000 / sm_120)

Write-once evidence for story sc-24135 (epic sc-24128). Hardware: **RTX Pro 6000 / sm_120**
(NVIDIA RTX PRO 6000 Blackwell Max-Q, driver 596.36, CUDA 12.9), GPU 1 via
`CUDA_VISIBLE_DEVICES=1`. Snapshot: `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (bf16). Build: `--release --features cuda`,
`CUDA_COMPUTE_CAP=120`, MSVC 14.44.

Files:

- [`nvfp4_evidence.json`](nvfp4_evidence.json) — the full run document (census by projection kind,
  device memory, greedy tokens and text, perplexity), written by
  `crates/llm/candle-llm/tests/nvfp4_evidence.rs::nvfp4_real_weight_evidence`.
- [`nvfp4_evidence.stderr.log`](nvfp4_evidence.stderr.log) — that run's console.
- [`ac2_cpu_refusal.log`](ac2_cpu_refusal.log) — AC2 on the same snapshot with
  `CANDLE_LLM_DEVICE=cpu` (`nvfp4_refused_on_cpu_for_the_real_snapshot`).
- [`ppl_slice_source.txt`](ppl_slice_source.txt) — the perplexity text (a frozen copy of
  `docs/architecture/inference-rearchitecture.md`, sha256
  `f238f90bf67cd78f023cf7a5430f161b2bea306e996201315f7af37b9f0629b1`).

## Results

| | bf16 | NVFP4 |
|---|---:|---:|
| NVFP4 projections (count / params) | 0 | 401 / 25,598,361,600 |
| NVFP4 projection bits/param | — | **4.500** |
| all projections bits/param (incl. 96 dense `in_proj_a/b`) | 16.0 | **4.511** |
| resident weight bytes, decoder (census) | 53,791,996,928 | **16,994,352,128** |
| whole-decoder bits/param (bf16 embedding + norms included) | 16.0 | 5.055 |
| device memory in use after load (`cuMemGetInfo`, incl. vision tower + allocator pool) | 56,378,327,040 | 22,953,918,464 |
| load time (page-cache warm) | 17.4 s | 17.0 s |
| greedy 256-token fixture coherent | yes | **yes** |
| first greedy divergence vs bf16 (token index) | — | **48** |
| perplexity, slice below | **7.8366** | **8.4140** (+7.37 %) |

Provider path (`LlamaProvider::load` with `Quantize::Nvfp4`, target + native MTP predictor):
`LoadRecord.requested = Some(Nvfp4)`, census 409 NVFP4 projections at 4.50 bits/param, 96 dense,
17,233,283,072 resident weight bytes (5.046 bits/param over 27.32 B params), load 18.4 s.

The NVFP4 greedy text follows the same plan as bf16 (identical for the first 48 tokens, then a
differently worded but coherent outline: tokenization → attention → KV cache → decoding
strategies); see `rows[1].greedy.text` in the JSON.

## Fixture and slice identity

- **Greedy fixture:** the sc-24129 decode-bench prompt through the snapshot's chat template
  (97 prompt tokens), `temperature = 0`, no stop tokens, 256 new tokens, reference decode loop.
- **Perplexity slice:** the first 2,048 tokens of the plain tokenizer encoding of
  `ppl_slice_source.txt` (3,422 tokens total; no template, no special tokens); 2,047 scored
  tokens; teacher-forced in 512-token chunks on one cache. Token-id sha256 (little-endian i32):
  `2d6ab0257e49941d0cdb71985059e6c84438c5f4b7aa871b8aab6459f88f09c5`.

The 2 % perplexity gate is the epic terminal story's to judge; this story records the numbers.

## AC2

`CANDLE_LLM_DEVICE=cpu`, same snapshot, `Quantize::Nvfp4`: the load fails in 0.000 s, before the
accelerator gate, admission or any weight read, with
`Unsupported("nvfp4: NVFP4 projections need a CUDA device with compute capability >= sm_120
(cuBLASLt block-scaled FP4 GEMM); the load device is Cpu")`. No sub-sm_120 GPU exists on this
host; that refusal is covered by the mocked-capability unit tests
(`nvfp4_weight::tests::sub_sm120_capabilities_are_refused_by_name`,
`provider::tests::a_sub_sm120_refusal_reaches_the_contract_as_unsupported`).

## Reproduce

```sh
# Git Bash env equivalent; the run used a cmd wrapper around vcvars64 14.44.
CUDA_VISIBLE_DEVICES=1 CUDA_COMPUTE_CAP=120 \
NVFP4_EVIDENCE_SNAPSHOT=<Qwen3.8-27B snapshot dir> \
NVFP4_EVIDENCE_OUTPUT=<new json path> \
NVFP4_EVIDENCE_PPL_TEXT=docs/migration/evidence/sc-24135/ppl_slice_source.txt \
cargo test --release --locked -p candle-llm --features cuda --test nvfp4_evidence -- \
  --ignored --exact nvfp4_real_weight_evidence --nocapture
```
