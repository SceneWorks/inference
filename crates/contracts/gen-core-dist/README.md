# gen-core-dist

> Package: `sceneworks-gen-core-dist` · library: `gen_core_dist`

The **host TCP collective transport** of the Iris distributed trainers: `TcpCollective`, a
host-staged all-reduce / all-gather / reduce-scatter / broadcast / barrier over `std::net` with
bounded waits, a token-authenticated rendezvous, TCP keepalive and typed failures. The Candle
trainer (`candle-gen-iris`, CPU / CUDA, including Windows where NCCL does not exist) uses it as
its transport; the MLX trainer (`mlx-gen-iris`) uses it as the bounded side channel around MLX's
own collectives.

The backend-neutral contract it implements — the `Collective` trait, `World`, `DistPlan`, mesh
resolution, shard layout and the typed `DistError` — stays in
[`gen-core`](../gen-core/README.md) at `gen_core::iris::train::dist`. This crate was split out of
gen-core in sc-27316 so the contract crate carries no network transport or socket FFI
(`libc` / `windows-sys` keepalive options). Like gen-core it has no tensor dependency, so it
builds and tests on the Linux contracts lane.

## License

Apache-2.0.
