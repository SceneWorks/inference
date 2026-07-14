# candle-gen-catalog

The **composition root** for the SceneWorks Candle media platform (shared by the CUDA and CPU
runtimes). Provider crates own their registrations; this crate owns only **selection and
stable ordering** — which families ship on Candle, and in what order they appear in the
registry. It does not own model implementations, duplicate descriptors, or infer membership by
scanning the workspace.

## Surface

```rust
use candle_gen_catalog::{provider_registry, register_providers, ProviderRegistry, ProviderRegistryBuilder};

// The complete, validated Candle media catalog:
let registry: ProviderRegistry = provider_registry()?;
let generator = registry.load("sdxl", &spec)?;

// Or add the whole platform to an existing builder:
let registry = register_providers(ProviderRegistryBuilder::new()).build()?;
```

- **`provider_registry()`** — build the complete explicit Candle catalog (43 generators, 6
  trainers, the JoyCaption captioner, and CLIP image/text embedders). The full list and the
  MLX/Candle deltas are in the [model catalog reference](../../../../docs/reference/model-catalog.md).
- **`register_providers(builder)`** — add every shipped Candle family to a builder, in stable
  order. This is the single reviewable source of truth for what ships: depending on a provider
  crate does **not** add it to the catalog; a family must be listed here explicitly.
- **`providers`** — re-exports every backend crate the platform owns (as `flux`, `krea`, …).
- **`BESPOKE_UTILITY_CRATES`** — the crates consumed through provider-specific APIs rather than
  the registry: `depth`, `face`, `instantid`, `pid`, `pulid`, `sam3`. (Note `pulid` is a
  bespoke utility here, whereas MLX ships it as the registered `pulid_flux` generator.)

`candle-gen` is re-exported as `media`, and `gen_core`'s `ProviderRegistry` /
`ProviderRegistryBuilder` are re-exported for convenience.

## Executable catalog contract

The crate's test pins the **complete, ordered id surface** for every provider kind, asserts
that every descriptor is on the `candle` backend, and runs the weights-free descriptor
conformance sweep. Changing the shipped surface requires an intentional update to that test —
that edit is the review point where platform inclusion becomes visible.

Consumers reach this catalog through the [`runtime-cuda`](../../../bundles/runtime-cuda/README.md)
and [`runtime-cpu`](../../../bundles/runtime-cpu/README.md) bundles, which validate it against
the Candle backend and pair it with the Candle LLM and snapshot-preparer catalogs.

## License

Apache-2.0.
