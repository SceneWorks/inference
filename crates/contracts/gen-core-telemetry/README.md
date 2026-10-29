# gen-core-telemetry

> Package: `sceneworks-gen-core-telemetry` · library: `gen_core_telemetry`

The **opt-in MiniMax-H3 terminal telemetry** collector: per-step wall time plus host process RSS
sampled on a 40 ms background thread (`proc_pidinfo` on macOS, `/proc/self/statm` on Linux,
`GetProcessMemoryInfo` on Windows), reported as one JSON object by `Collector::finish`. The MLX
and Candle MiniMax-H3 providers create a `Collector` only when the request sets
`h3_terminal_telemetry`; that request flag stays on gen-core's `GenerationRequest`.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316 so the contract crate carries no
host-process FFI. It depends only on `serde_json` and the platform FFI crates.

## License

Apache-2.0.
