# gen-core-iris

> Package: `sceneworks-gen-core-iris` · library: `gen_core_iris`

The tensor-free **Iris-3B** contract: task identity, resource layout, the release `config.yaml`,
the Qwen3-VL conditioning window, the FlowDPM-Solver++ plan, restoration/depth/downstream tasks,
and the trainer controllers (generation, depth, restoration, the distributed contract the
[`gen-core-dist`](../gen-core-dist/README.md) transport implements). The MLX (`mlx-gen-iris`) and
Candle (`candle-gen-iris`) providers read it.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. It holds gen-core's former `iris`
module unchanged in content.

## License

Apache-2.0.
