# SC-20671 dense-baseline model contract

The dense-baseline parent and every worker select only the eight identities below (four families,
each a 4-bit candidate and its bf16 reference). A path supplied
at launch is a local storage location, never a model selector: before `LlamaProvider` loads it,
the harness verifies the required files' byte lengths and SHA-256 digests plus `config.json` family,
architecture, exact native context, and candidate/reference precision role. The source contract is
`mlx_llm::campaign::{LLAMA_CANDIDATE, LLAMA_REFERENCE, QWEN_CANDIDATE, QWEN_REFERENCE,
LLAMA8B_CANDIDATE, LLAMA8B_REFERENCE, QWEN8B_CANDIDATE, QWEN8B_REFERENCE}`.

| family | role | immutable Hugging Face repository and revision | native context |
| --- | --- | --- | --- |
| Llama | 4-bit candidate | `mlx-community/Llama-3.2-3B-Instruct-4bit@7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` | 131,072 |
| Llama | bf16 reference | `mlx-community/Llama-3.2-3B-Instruct-bf16@6d88ba43024fef71b10e52e101c7cd4598322601` | 131,072 |
| Qwen | 4-bit candidate | `mlx-community/Qwen3-1.7B-4bit@3b1b1768f8f8cf8351c712464f906e86c2b8269e` | 40,960 |
| Qwen | bf16 reference | `mlx-community/Qwen3-1.7B-bf16@9cd6692855d3e06772228e9a962b2606359b2d24` | 40,960 |
| Llama 8B (`llama8b`) | 4-bit candidate | `mlx-community/Llama-3.1-8B-Instruct-4bit@90215b22ec18e72f623dde2ea7af4097025160e2` | 131,072 |
| Llama 8B (`llama8b`) | bf16 reference | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16@f8311090f9ee47782b6f094984a20c856eb841d6` | 131,072 |
| Qwen 8B (`qwen8b`) | 4-bit candidate | `mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192` | 40,960 |
| Qwen 8B (`qwen8b`) | bf16 reference | `mlx-community/Qwen3-8B-bf16@85dd0f16bfe491befbc9cf0b4e966664236e5050` | 40,960 |

Schedule v3 (sc-20688) runs every family on the same four rows (sixteen in all). Each band's
target derives from the native window (short 32, medium 1,024, memory-material window/4, fit-boundary
window−512), except `llama8b`'s fit-boundary row. A compressed row is priced on the candidate it
loads, scaled by the measured A2 v5 peak ratio (`COMPRESSED_MEASURED_PEAK_SCALE_BPS`, 1.27). At
130,560 tokens that is 75.2 GiB, over the 68 GiB child footprint cap of
`.github/kv-poc/policies/llm.json`, so `llama8b` is measured at 106,496 tokens
(`LLAMA8B_FIT_WINDOW_TOKENS` 107,008): 63.5 GiB, 4.5 GiB under the cap. A dense (A1) row also
prices its bf16 reference, 71.5 GiB for that row, so A1 refuses `llama8b-fit-boundary` before
spawn and an A1 campaign cannot complete at schedule v3; A1 is therefore opt-in and A3 (which binds
a complete A1 at the same source closure) is refused by the workflow's config (sc-20688).

The required inventory includes every published weights shard, `config.json`, and the published
model/index and tokenizer configuration files. The contract records every required file's
SHA-256 and byte length in source; a mismatch fails the parent before it starts workers and fails
the worker again before product loading. The fixed 64-coordinate schedule remains source-owned by
`required_schedule`; no CLI flag supplies families, model IDs, revisions, coordinates, or receipt
identity.

The benchmark is not runnable by merely materializing a directory. The coordinator must separately
own the Metal/real-weight lane and run the exact product-loader campaign; this document is the
pre-results provenance gate, not a substitute for a sealed measurement receipt.

## Dense K/V element width

Every role's dense K/V is the causal loader's compute dtype, BF16, at 2 bytes per element.
`DENSE_KV_COMPUTE_DTYPE` is `CausalLm::COMPUTE_DTYPE`, and `DENSE_KV_COMPUTE_ELEMENT_BYTES` is
derived from it. MLX affine `quantized_matmul` returns `promote_types(activations, scales)`, and
BF16 with F16 promotes to F32. Before the sc-20671 dtype fix, the loader kept stored scales in
their stored dtype. Any snapshot with F16 or F32 scales therefore ran its whole decoder in F32,
including the K/V it cached, and the harness encoded that promoted width as the expected one. The
loader now holds stored scales and biases in the compute dtype at load. The product observer and
receipt validation refuse any other observed width, naming the observed and expected dtype,
instead of recording a widened dense baseline.

### Affected surfaces

| Surface | Stored scale dtype | Path before the fix |
| --- | --- | --- |
| SC-20671 Llama candidate `mlx-community/Llama-3.2-3B-Instruct-4bit` | F16 (header-verified) | F32 decoder and K/V |
| SC-20671 Qwen candidate `mlx-community/Qwen3-1.7B-4bit` | BF16 (header-verified) | Unchanged, BF16 |
| Prism/Bonsai `prism_hadamard_qwen35`, e.g. `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (production Qwen3.8 Bonsai route) | F16, required by `validate_affine_parts` (header-verified) | F32 decoder and full-attention K/V |
| Any other stored-quantized mlx-community Llama-family or `qwen3_5` snapshot | Its converter's dtype | F32 when F16/F32, unchanged when BF16 |
| Engine-prepared (`write_snapshot`) and GGUF MLX-requant snapshots | BF16 | Unchanged |

The Prism change reaches production. The Bonsai route's numerics, K/V footprint and speed all move
from the F32-promoted path to BF16. The Prism linear scales are cast F16→BF16. The Prism embedding
keeps its stored F16 scales because its rows are cast to BF16 and feed no matmul. MLX request
admission prices K/V, activations and logits at the compute width and eager scores at F32. The
exception is Prism, which stays priced at the F32 width. Its frozen 27-token allocator peak
(773,229,760 bytes) was measured on the F32 path, and the compute-width estimate no longer covers
it.

### Evidence to re-take at epic end

The evidence below came from the F32-promoted path. It is not rewritten, relabelled, or assumed
to transfer:

- **Frozen Prism allocator peaks.** The floors in
  `provider::tests::qwen35_prism_estimate_covers_frozen_allocator_peaks_and_rejects_remote_long_context`
  (local probe `02ed7e958`, campaign `35461924246`). Re-take them on the BF16 path. Only then can
  the Prism admission width in `priced_compute_element_bytes` move to the compute width.
- **Prism/Bonsai coherence and quality.** Any MLX Prism coherence, quality, or decode-speed
  observation taken before this fix.
- **SC-20671 Llama-candidate rows.** Every row, short rows included, including memory
  denominators, timings, logits, and quality observations.
- **SC-20677 Llama-candidate K/V captures.** Their manifests record F32 K/V, so they are not the
  BF16 cache the candidates compress.

The F16→BF16 scale cast is a real numeric change: BF16 keeps 8 mantissa bits, F16 keeps 11. Quality
has to be re-measured rather than assumed unchanged.

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
