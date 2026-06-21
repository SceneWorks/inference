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

## Status

Work in progress (epic 7153, story 7237). Not yet published.

## License

Apache-2.0.
