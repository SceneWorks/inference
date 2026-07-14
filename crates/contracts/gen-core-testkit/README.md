# gen-core-testkit

> Package: `sceneworks-gen-core-testkit` · library: `gen_core_testkit`

A **contract conformance suite** for [`gen-core`](../gen-core/README.md) providers. Given any
boxed provider — an MLX family from `mlx-gen`, a Candle family from `candle-gen` — it
exercises the behavioral guarantees the contract *promises but cannot express in the type
system*:

- **typed cancellation** — a tripped `CancelFlag` actually stops the work;
- **progress monotonicity** — `Progress` events advance and never regress;
- **seed determinism** — the same seed reproduces output; a fresh seed does not;
- **capability honesty** — a provider serves exactly what its `Capabilities` advertise, and
  rejects the rest in `validate`.

Like `gen-core`, the testkit has **zero tensor dependencies** — it drives the provider purely
through the public contract, so it runs on the Linux `gen-core` lane against an in-crate stub
exactly as it does on a backend lane against a real family. Both backends run it in CI, so a
provider that silently ignores `CancelFlag` or reports no progress becomes a CI failure
instead of a field report.

## Usage

Family crates dev-depend on it and run their real model through the suite:

```rust
// generator, trainer, and captioner conformance:
gen_core_testkit::conformance(
    || registry.load("z_image_turbo", &spec).unwrap(),
    &gen_core_testkit::Profile::cheap(),
);
gen_core_testkit::trainer_conformance(
    || registry.load_trainer("z_image_turbo", &spec).unwrap(),
    &gen_core_testkit::TrainerProfile::cheap(items, out_dir),
);
gen_core_testkit::captioner_conformance(
    || registry.load_captioner("<captioner-id>", &spec).unwrap(),
    &gen_core_testkit::CaptionerProfile::cheap(),
);
```

The `*_conformance` entry points run every check and panic with the aggregated failures; the
individual `check_*` functions are public so a provider's own test can target one guarantee at
a time.

## License

Apache-2.0.
