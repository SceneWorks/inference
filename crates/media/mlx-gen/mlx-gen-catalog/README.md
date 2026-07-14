# mlx-gen-catalog

The **composition root** for the SceneWorks MLX media platform. Provider crates own their
registrations; this crate owns only **selection and stable ordering** — which families ship
on MLX, and in what order they appear in the registry. It does not own model implementations,
duplicate descriptors, or infer membership by scanning the workspace.

## Surface

```rust
use mlx_gen_catalog::{provider_registry, register_providers, ProviderRegistry, ProviderRegistryBuilder};

// The complete, validated MLX media catalog:
let registry: ProviderRegistry = provider_registry()?;
let generator = registry.load("z_image_turbo", &spec)?;

// Or add the whole platform to an existing builder:
let registry = register_providers(ProviderRegistryBuilder::new()).build()?;
```

- **`provider_registry()`** — build the complete explicit MLX catalog (57 generators, 14
  trainers, the JoyCaption captioner, and CLIP image/text embedders). The full list is in the
  [model catalog reference](../../../../docs/reference/model-catalog.md).
- **`register_providers(builder)`** — add every shipped MLX family to a builder, in stable
  order. This is the single reviewable source of truth for what ships: depending on a provider
  crate does **not** add it to the catalog; a family must be listed here explicitly.
- **`providers`** — re-exports every backend crate the platform owns (as `flux`, `krea`, …).
- **`BESPOKE_UTILITY_CRATES`** — the crates consumed through provider-specific APIs rather than
  the registry: `depth`, `face`, `instantid`, `pid`, `sam2`, `sam3`.

`mlx-gen` is re-exported as `media`, and `gen_core`'s `ProviderRegistry` /
`ProviderRegistryBuilder` are re-exported for convenience.

## Executable catalog contract

The crate's test pins the **complete, ordered id surface** for every provider kind, asserts
that every descriptor is on the `mlx` backend, and runs the weights-free descriptor
conformance sweep. Changing the shipped surface requires an intentional update to that test —
that edit is the review point where platform inclusion becomes visible.

Consumers reach this catalog through the [`runtime-macos`](../../../bundles/runtime-macos/README.md)
bundle, which validates it against the MLX backend and pairs it with the MLX LLM and
snapshot-preparer catalogs.

## License

Apache-2.0.
