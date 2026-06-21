# core-llm

Reserved crate name for the **backend-neutral core** of an on-device LLM serving library: the contract (streaming, multimodal text + vision), host-side policy (chat templates, scheduling, sampling), and the provider registry — with **no tensor dependencies**.

Backend engines implement this contract: [`mlx-llm`](https://github.com/SceneWorks/mlx-llm) (Apple MLX) and [`candle-llm`](https://github.com/SceneWorks/candle-llm) (Candle, cross-platform). This `0.0.0` release is a placeholder — the implementation is in progress.

## Status

Work in progress. Not yet usable.

## License

Apache-2.0.
