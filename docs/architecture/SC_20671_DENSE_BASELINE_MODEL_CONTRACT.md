# SC-20671 dense-baseline model contract

The dense-baseline parent and every worker select only the four identities below. A path supplied
at launch is a local storage location, never a model selector: before `LlamaProvider` loads it,
the harness verifies the required files' byte lengths and SHA-256 digests plus `config.json` family,
architecture, exact native context, and candidate/reference precision role. The source contract is
`mlx_llm::campaign::{LLAMA_CANDIDATE, LLAMA_REFERENCE, QWEN_CANDIDATE, QWEN_REFERENCE}`.

| family | role | immutable Hugging Face repository and revision | native context |
| --- | --- | --- | --- |
| Llama | 4-bit candidate | `mlx-community/Llama-3.2-3B-Instruct-4bit@7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` | 131,072 |
| Llama | bf16 reference | `mlx-community/Llama-3.2-3B-Instruct-bf16@6d88ba43024fef71b10e52e101c7cd4598322601` | 131,072 |
| Qwen | 4-bit candidate | `mlx-community/Qwen3-1.7B-4bit@3b1b1768f8f8cf8351c712464f906e86c2b8269e` | 40,960 |
| Qwen | bf16 reference | `mlx-community/Qwen3-1.7B-bf16@9cd6692855d3e06772228e9a962b2606359b2d24` | 40,960 |

The required inventory includes every published weights shard, `config.json`, and the published
model/index and tokenizer configuration files. The contract records every required file's
SHA-256 and byte length in source; a mismatch fails the parent before it starts workers and fails
the worker again before product loading. The fixed 64-coordinate schedule remains source-owned by
`required_schedule`; no CLI flag supplies families, model IDs, revisions, coordinates, or receipt
identity.

The benchmark is not runnable by merely materializing a directory. The coordinator must separately
own the Metal/real-weight lane and run the exact product-loader campaign; this document is the
pre-results provenance gate, not a substitute for a sealed measurement receipt.

## Operator stop between rows

A campaign parent (`sc20671_kv_baseline parent`, `sc20676_packed_evidence parent`, and the
SC-20684 / SC-20686 media launchers) can be halted safely: create `<resume-dir>/STOP` (always
honoured) or the path given as `--stop-file <path>` (honoured too); only a missing path counts as
absent, any other stat failure is an error. The parent never signals a running worker — killing an MLX render
mid command buffer can wedge the GPU — so the row in flight finishes and is accepted normally.
Before spawning the next row it writes a sealed, never-overwritten
`<resume-dir>/logs/operator-stop.attempt-<n>.json` (`"status": "stopped-by-operator"`,
`beforeRow`, `beforeRowSlug`, `rowsAccepted`, `stopFiles`) with its `.sha256` seal and exits with status **75** (sysexits
`EX_TEMPFAIL`), distinct from success and from every refusal/failure status. The stop file is not
part of the resume identity: remove it and rerun the identical command with the same resume
directory, and the accepted rows resume while the campaign continues at the stopped row.

## SC-20677 real K/V capture

`sc20677_capture_kv` loads one campaign snapshot through the same `LlamaProvider` campaign load
and causal decoder, prefills `N-1` tokens of the prompt (repeated/truncated to `--tokens N` with
the snapshot's tokenizer), runs one real decode step, and writes per-layer `q` `[B,Hq,1,D]`
(post-RoPE at position `N-1`) and dense-cache `k`/`v` `[B,Hkv,N,D]` safetensors with their scale,
mask, geometry, dtype, snapshot/prompt digests, and inference revision, plus
`capture-manifest.json` and `SHA256SUMS`. The worker runs under the campaign supervisor policy
(child footprint cap, host reserve, deadline). One command captures and compares:

```sh
cargo run --locked --release -p mlx-llm --bin sc20677_capture_kv -- parent \
  --snapshot <llama-4bit-snapshot> --prompt-file <prompt.txt> --tokens 8192 \
  --layers 0,mid,last --safety-policy <policy.json> --out /abs/sc20677-kv-llama && \
cargo run --locked --release -p mlx-llm --bin sc20677_kv_candidates -- \
  $(for f in /abs/sc20677-kv-llama/*.safetensors; do printf -- '--kv %s ' "$f"; done) \
  --out /abs/sc20677-comparison-llama.json
```
