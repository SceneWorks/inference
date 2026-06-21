# core-llm-testkit

Conformance suite for the [`core-llm`](https://github.com/SceneWorks/core-llm) `TextLlm` contract.

A generic, **tensor-free** harness that drives any provider through the contract's guarantees:

- **validation** is capability-honest (rejects what the descriptor doesn't declare),
- **streaming** emits a token event per token whose deltas reconstruct the final output,
- **cancellation** — typed `Canceled` before inference, prompt partial `Cancelled` mid-stream,
- **seed determinism** — same seed reproduces, different seed diverges (anti-cheat),
- **capabilities / registry** — descriptor is well-formed and discoverable,
- **multimodal** — a vision provider generates from an image; a text-only one rejects it as
  `Unsupported`.

Backends dev-depend on it and run their real model through the checks:

```rust
core_llm_testkit::textllm_conformance(
    || my_backend::Provider::load(&spec).map(|p| Box::new(p) as _).unwrap(),
    &core_llm_testkit::TextLlmProfile::cheap(),
);
```

`textllm_conformance` runs every check, aggregates failures, and panics once with a combined
message. Each `check_*` is also public and returns `Result<(), String>` to target one guarantee at a
time.

## License

Apache-2.0.
