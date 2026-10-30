# gen-core-tokenizer

> Package: `sceneworks-gen-core-tokenizer` · library: `gen_core_tokenizer`

The **text tokenization** layer: `TextTokenizer`, a thin wrapper over the HF `tokenizers` crate
(the same Rust core `transformers` wraps) that adds chat-template rendering, max-length
padding/truncation and the attention mask, so a model's `tokenizer.json` reproduces the reference
token ids exactly. Every text-encoder provider that tokenizes host-side uses it; mlx-gen re-exports
it as `mlx_gen::tokenizer`.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. It holds gen-core's former
`tokenizer` module, so gen-core no longer compiles `tokenizers` (and its Oniguruma build).

## License

Apache-2.0.
