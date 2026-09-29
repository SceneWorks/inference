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

## Dense K/V element width

Every role's dense K/V is the loader's BF16 compute dtype: 2 bytes per element. The pinned Llama
4-bit candidate stores its quantized `scales`/`biases` as F16 (the Qwen candidate stores BF16).
MLX affine `quantized_matmul` returns `promote_types(activations, scales)`, and BF16 with F16
promotes to F32. Before the sc-20671 dtype fix, the loader kept stored scales in their stored
dtype. The whole Llama candidate decoder, including the K/V it cached, therefore ran in F32
(4 bytes). The harness encoded that promoted width as the expected one. The loader now holds
stored scales and biases in the compute dtype at load. `DENSE_KV_COMPUTE_ELEMENT_BYTES` is the
only accepted width. The product observer and receipt validation refuse any other observed width,
naming the reason, instead of recording a widened dense baseline.

Any Llama-candidate row, including short rows, captured before that fix came from the F32 path.
Its memory denominator, timings, logits, and quality observations do not describe the shipped
BF16 path, and they are not rewritten or relabelled. Re-take them on the fixed loader. The
F16→BF16 scale cast is a real numeric change (BF16 keeps 8 mantissa bits against F16's 11), so
quality has to be re-measured rather than assumed unchanged.
