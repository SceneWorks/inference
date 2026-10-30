# gen-core-h3

> Package: `sceneworks-gen-core-h3` · library: `gen_core_h3`

The tensor-free **MiniMax-H3** planners and policy: request/component native allocation pricing
(`NativeAllocationFactsExt`), adapter and conditioning memory pricing, the Veda sparse-attention
predictor, selection and workspace pricing, reference normalization and the Ref2VA reference set,
audio config, keyframes, the denoise schedule, chunking and pruned adaLN. The data shapes the
contract carries (`MemoryPhaseFacts::native_quant`, the Veda request/report, the workspace bound,
the reference resize mode) stay in gen-core and are re-exported from the matching modules here.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. Its tests read the Veda frozen
fixture in `gen-core/tests/fixtures/veda`.

## License

Apache-2.0.
