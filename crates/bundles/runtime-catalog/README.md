# runtime-catalog

The **validated, machine-readable composition** shared by every named runtime bundle
(`runtime-macos`, `runtime-cuda`, `runtime-cpu`). This crate is **tensor-neutral**: it
owns no providers and no model code. A platform bundle supplies its explicit media, LLM,
and snapshot-preparer registries; `runtime-catalog` checks that the composition is
internally consistent and exposes a stable snapshot.

## What it validates

`RuntimeCatalog::try_new(platform, backend, media, text, preparers)` builds a catalog only
if the composition is coherent, and returns a `RuntimeCatalogError` otherwise:

- every media descriptor's `backend` matches the bundle's declared backend;
- every text-LLM descriptor's `backend` matches, and its identity fields are non-empty;
- the media registry passes the weights-free **descriptor conformance sweep**
  (`ProviderRegistry::descriptor_conformance_errors`);
- the bundle declares at least one snapshot preparer, all on the declared backend;
- `platform` and `backend` are non-empty.

This is the seam that makes "an MLX generator ended up in the Candle bundle" a construction
error rather than a runtime surprise. No weights are loaded.

## Surface

```rust
let catalog = runtime_catalog::RuntimeCatalog::try_new(
    "macos", "mlx",
    media_registry(),      // gen_core::Result<ProviderRegistry>
    text_registry(),       // core_llm::Result<TextLlmRegistry>
    preparer_registry(),   // core_llm::Result<SnapshotPreparerRegistry>
)?;

catalog.platform();   // &'static str
catalog.backend();    // &'static str
catalog.media();      // &ProviderRegistry        — generators, trainers, captioners, embedders
catalog.text();       // &TextLlmRegistry
catalog.preparers();  // &SnapshotPreparerRegistry
catalog.snapshot();   // RuntimeCatalogSnapshot   — stable, JSON-serializable, weights-free
```

`RuntimeCatalogSnapshot` carries the id lists for every provider kind plus the snapshot
preparer backends, and `to_json()` renders the form consumed by release tooling and
external smoke projects.

The crate re-exports `core_llm` and `gen_core` so a bundle (and its consumers) reach the
contract types through this one path.

## Where this sits

```text
gen-core / core-llm (contracts)
        │
mlx-gen-catalog / candle-gen-catalog + mlx-llm / candle-llm  (explicit registries)
        │
   runtime-catalog        ← you are here: validate + snapshot
        │
runtime-macos / runtime-cuda / runtime-cpu   (the product boundary)
```

Consumers depend on a **bundle**, not on this crate directly. See the
[Getting Started guide](../../../docs/guide/getting-started.md) and the
[architecture rationale](../../../docs/architecture/inference-rearchitecture.md).

## License

Apache-2.0.
