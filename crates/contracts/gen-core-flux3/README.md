# gen-core-flux3

> Package: `sceneworks-gen-core-flux3` · library: `gen_core_flux3`

The tensor-free **FLUX 3 [klein]** contract: frozen upstream identity, the release package layout
and `config.json`, the Qwen3-VL text window, the flow-map schedule, packed-token ids and RoPE
tables, checkpoint conversion and quantization oracles, the prompt upsampler (PUP), reference
planning and preparation, layout compilation (`LayoutCompile`), and the training controller. The
provider ids, the PUP token cap, the layout document/bindings and the reference output size a
`GenerationRequest` carries stay in `gen_core::flux3` and are re-exported here.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. It holds gen-core's former `flux3`
module and its integration tests and fixtures.

## License

Apache-2.0.
