# SC-20671 dense-baseline model contract

The dense-baseline parent and every worker select only the four identities below. A path supplied
at launch is a local storage location, never a model selector: before `LlamaProvider` loads it,
the harness verifies the required files' byte lengths and SHA-256 digests plus `config.json` family,
architecture, exact native context, and candidate/reference precision role. The source contract is
`mlx_llm::campaign::{LLAMA_CANDIDATE, LLAMA_REFERENCE, QWEN_CANDIDATE, QWEN_REFERENCE}`.

| family | role | immutable Hugging Face repository and revision | native context |
| --- | --- | --- | --- |
| Llama | 4-bit candidate | `mlx-community/Llama-3.2-1B-Instruct-4bit@08231374eeacb049a0eade7922910865b8fce912` | 131,072 |
| Llama | bf16 reference | `mlx-community/Llama-3.2-1B-Instruct-bf16@863c846a9ac6fad4e49e1743d52984dff262e953` | 131,072 |
| Qwen | 4-bit candidate | `mlx-community/Qwen3-1.7B-4bit@3b1b1768f8f8cf8351c712464f906e86c2b8269e` | 40,960 |
| Qwen | bf16 reference | `mlx-community/Qwen3-1.7B-bf16@9cd6692855d3e06772228e9a962b2606359b2d24` | 40,960 |

The required inventory includes the complete single-file weights payload, `config.json`, and the
published model/index and tokenizer configuration files. The contract records every required file's
SHA-256 and byte length in source; a mismatch fails the parent before it starts workers and fails
the worker again before product loading. The fixed 64-coordinate schedule remains source-owned by
`required_schedule`; no CLI flag supplies families, model IDs, revisions, coordinates, or receipt
identity.

The benchmark is not runnable by merely materializing a directory. The coordinator must separately
own the Metal/real-weight lane and run the exact product-loader campaign; this document is the
pre-results provenance gate, not a substitute for a sealed measurement receipt.
