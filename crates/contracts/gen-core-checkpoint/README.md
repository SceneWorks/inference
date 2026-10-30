# gen-core-checkpoint

> Package: `sceneworks-gen-core-checkpoint` · library: `gen_core_checkpoint`

Tensor-free **checkpoint readers and converters**: the torch `.pth`/`.pt` zip + pickle reader
(`pth`, STORED entries only, no inflater) and the Supra2 checkpoint converter and pinned-bundle
contract (`supra2`; CLI: `cargo run --release -p sceneworks-gen-core-checkpoint --example
supra2_convert`). mlx-gen and the Supra2 providers read it.

Split out of [`gen-core`](../gen-core/README.md) in sc-27316. It depends on gen-core (gen-core
never depends on it) and, like gen-core, on no tensor library. With it goes `zip`; its tests read
the Supra2 and latent-perceptual fixtures in `gen-core/tests/fixtures`.

## License

Apache-2.0.
