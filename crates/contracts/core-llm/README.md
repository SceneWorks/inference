# core-llm

The **backend-neutral core** of an on-device LLM serving library: the provider contract (streaming,
cancellable, multimodal text + vision), host-side policy (chat templates, sampling policy, tokenizer
text ↔ ids, constrained decoding), and the provider registry — with **no tensor dependencies**.

`core-llm` builds standalone on Linux, Windows, and macOS and depends on nothing from any tensor
backend or image-generation stack. Backends implement the [`TextLlm`] contract and expose ordinary
registration values. Platform bundles compose exactly the providers they ship into a deterministic
registry; consumers select one and stream a generation entirely through this contract.

The contract is **extracted from the working mlx-llm engine**, not designed in a vacuum, and is
provisional until `candle-llm` validates it.

## Surface

- **`TextLlm`** — streaming, cancellable, multimodal provider trait (`descriptor` / `validate` /
  `generate`).
- **`TextLlmRequest` / `Message` / `Content`** — multi-turn, multimodal (text + image) request model.
- **`StreamEvent` / `TextLlmOutput` / `Usage` / `FinishReason`** — streaming + result types.
- **`Sampling`** — backend-neutral sampling policy (temperature / top-p / top-k / repetition penalty).
- **`Constraint` + `JsonState`** — constrained-decoding policy (a pure, incremental JSON grammar).
- **`Tokenizer` + `ChatTemplate`** — host-side text policy (HF tokenizers; typed Llama-3 / ChatML
  templates, with a Jinja renderer to follow).
- **`TextLlmRegistry`** — explicit provider composition, id-based routing, and model-first loading.

```rust
use core_llm::{LoadSpec, Message, StreamEvent, TextLlmRegistry, TextLlmRequest};

fn generate(registry: &TextLlmRegistry) -> core_llm::Result<()> {
let provider = registry.load_textllm("mlx-llama", &LoadSpec::dense("/path/to/snapshot"))?;
let req = TextLlmRequest {
    messages: vec![Message::user("Hello!")],
    max_new_tokens: 64,
    ..Default::default()
};
provider.generate(&req, &mut |ev| {
    if let StreamEvent::Token { text, .. } = ev { print!("{text}"); }
})?;
Ok(())
}
```

[`TextLlm`]: https://docs.rs/core-llm

## License

Apache-2.0.
