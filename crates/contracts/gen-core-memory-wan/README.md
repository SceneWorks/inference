# gen-core-memory-wan

> Package: `sceneworks-gen-core-memory-wan` · library: `gen_core_memory_wan`

The tensor-free **Wan2.2 I2V memory planner** shared by the native Wan providers: canonical
repository/revision, exact direct-file inventory, header-derived resident facts, request geometry,
routes, ordered adapter receipts and calibration fingerprints. The Wan, Qwen-Image 2.1 and SCAIL2
providers read it.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. It holds gen-core's former
`wan_i2v_memory` module, so a change to a Wan memory table no longer rebuilds gen-core.

## License

Apache-2.0.
