# candle-llm

An **on-device LLM serving engine** — the [Candle](https://github.com/huggingface/candle) backend of
a backend-neutral serving contract (cross-platform, including Windows/CUDA). It implements the
[`core-llm`](https://github.com/SceneWorks/core-llm) contract and, by passing that contract's
conformance suite as a *second, independent* backend, **de-provisionalizes** it. A sibling
[`mlx-llm`](https://github.com/SceneWorks/mlx-llm) crate provides the Apple MLX backend.

## Architecture

Built bottom-up, mirroring `mlx-llm`'s structure on Candle tensors:

1. **`primitives`** — backend-owned tensor leaves: a batch-capable `KvCache`, a pluggable sampler,
   the `Rope` family (standard / Llama-3 scaled), GQA attention helpers, group-wise quantization
   (via Candle's `QTensor`/`QMatMul`), the `nn` leaves, and a safetensors `Weights` loader.
2. **`config` + `models`** — `LlamaConfig` parsed from `config.json` and the generic causal decoder
   (`LlamaModel`), `&self` forward + `from_weights`, with architecture dispatch (Llama / Mistral /
   Qwen3).
3. **`decode`** — the streaming, cancellable decode loop driving any `Decode` model.
4. **`provider`** — implements `core_llm::TextLlm` over the engine and registers it (`candle-llama`).

## Backends / features

- default → CPU (builds anywhere)
- `--features cuda` → NVIDIA CUDA (the Windows target)
- `--features metal` → Apple Metal

Compute runs in `bf16` on the GPU backends and `f32` on CPU.

## Testing

Unit tests + the synthetic conformance run need no models and no GPU:

```sh
cargo test                       # CPU, no weights
cargo clippy --all-targets -- -D warnings
```

The real-weights checks are `#[ignore]`d and gated on environment variables pointing at on-disk
snapshots; run them with `-- --ignored` (add `--features cuda` for the GPU path). They cover the
full `core-llm-testkit` conformance suite across architectures and load formats, plus per-feature
parity tests:

| env var | points at | exercised by |
|---|---|---|
| `CANDLE_LLM_TEST_MODEL` | a Llama-family HF snapshot dir (e.g. SmolLM2-135M-Instruct) | conformance (dense + **Q8** quantize-on-load), batch decode, **prefix-cache** reuse |
| `CANDLE_LLM_QWEN3_MODEL` | a Qwen3 HF snapshot dir | conformance (dense + **Q4** quantize-on-load; q/k RMSNorm, head_dim 128), **prefix-cache** reuse |
| `CANDLE_LLM_GGUF` | a single `*.gguf` file | conformance + GGUF parity vs the HF load |
| `CANDLE_LLM_{PHI3,QWEN2MOE,GEMMA2,GLM4,DEEPSEEK}_MODEL` | a snapshot for that architecture family | `breadth` — coherent-text streaming per family |

The `breadth` test streams a prompt through each non-Llama architecture: **Phi-3** (packed qkv/gate_up),
**Qwen2-MoE** (router + experts + shared, q/k/v bias), **Gemma-2** (sandwich norms + soft-caps + GeGLU),
**GLM-4** (sandwich + partial/interleaved RoPE), and **DeepSeek-V2** (Multi-head Latent Attention +
fine-grained MoE; verified on `deepseek-ai/DeepSeek-V2-Lite-Chat`, which fits in 96GB).

The `prefix` test covers **shared-prefix KV reuse** (`generate_cached` over a `PrefixCache`): a request
sharing a leading run of tokens with a stored one (a system prompt, a few-shot preamble, a growing
chat) reuses that span's keys/values instead of recomputing prefill. A tiny synthetic CPU model proves
it is **bit-exact** — `generate_cached` is token-for-token identical to a cold `generate` — and the
`#[ignore]`d real-weights variants confirm the mechanic on a GPU snapshot (first-token logits match
within a small bf16 tolerance; reuse accounting is exact) and report the reused-token count.

> Q4_K's block size is 256, so Q4 quantize-on-load needs projection `in`-dims that are multiples of
> 256 (true of Qwen3's hidden 1024, not of SmolLM2's 576); Q8_0's block is 32 and applies broadly.

```sh
# Whole real-weights suite on CUDA:
CANDLE_LLM_TEST_MODEL=/path/SmolLM2-135M-Instruct \
CANDLE_LLM_QWEN3_MODEL=/path/Qwen3-0.6B \
CANDLE_LLM_GGUF=/path/Model-Q4_K_M.gguf \
  cargo test --features cuda -- --ignored --nocapture
```

On Windows/CUDA the build needs the VS dev environment (`vcvars64` + `CUDA_COMPUTE_CAP`); see the
helper `.bat` scripts under the workspace root.

## Status

Work in progress (epic 7153, story 7237). Not yet published.

## License

Apache-2.0.
