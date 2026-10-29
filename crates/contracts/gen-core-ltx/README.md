# gen-core-ltx

> Package: `sceneworks-gen-core-ltx` · library: `gen_core_ltx`

The tensor-free **LTX** helpers: the split-checkpoint component resolver (layout selection keyed on
`model_version`, per-component config isolation, the `gemma_source_checkpoint` ⇄ text-encoder
`gemma_version` assertion), the Gemma encoder assets, the DFR keyframe policy and the duration
head. The auto-duration range a request carries stays in `gen_core::duration_head` and is
re-exported here.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. Its tests read the Gemma tokenizer
fixtures in `gen-core/tests/fixtures`.

## License

Apache-2.0.
