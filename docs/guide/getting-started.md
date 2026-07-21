# Getting Started

> **Audience:** engineers embedding SceneWorks inference in a product (SceneWorks,
> ChatWorks, or an external smoke project).
> **Scope:** how to depend on a named runtime bundle, build its validated provider
> registries, and load and run media generators, LLMs, and snapshot preparers.

This guide is the *how*. The [architecture rationale](../architecture/inference-rearchitecture.md)
is the *why* — read it if you want the reasoning behind explicit composition, or the
[model catalog reference](../reference/model-catalog.md) for the full list of shipped
providers.

## The mental model

A product never assembles backend crates or scans for providers. It depends on **one
named runtime bundle**, builds the bundle's **validated catalog**, and loads models
**by id** through ordinary registry values:

```text
runtime-macos │ runtime-cuda │ runtime-cpu      ← one bundle per product target
        │
        ▼   catalog()  →  RuntimeCatalog         ← validated: every descriptor's backend
        │                                            matches the bundle, ids are unique,
        │                                            weights-free conformance passes
        ├── .media()      → &ProviderRegistry     ← generators, trainers, captioners, embedders
        ├── .text()       → &TextLlmRegistry      ← text LLMs (streaming, multimodal)
        ├── .preparers()  → &SnapshotPreparerRegistry
        └── .snapshot()   → RuntimeCatalogSnapshot ← stable, weights-free, JSON-serializable
```

Everything below flows from that: pick a bundle, call `catalog()`, then load by id.

## 1. Choose a bundle

| Bundle          | Backend  | `PLATFORM` | Target triples                                                      | Native prerequisites                       |
| --------------- | -------- | ---------- | ------------------------------------------------------------------ | ------------------------------------------ |
| `runtime-macos` | MLX      | `macos`    | `aarch64-apple-darwin`                                             | macOS 26.2+, Xcode Metal toolchain         |
| `runtime-cuda`  | Candle   | `cuda`     | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`              | NVIDIA CUDA toolkit, supported NVIDIA driver |
| `runtime-cpu`   | Candle   | `cpu`      | `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc` | none                                       |

A bundle is a **composition profile of one platform release**, not a separately pinned
set of backend crates. MLX, CUDA, and CPU are mutually exclusive targets — there is no
`--all-features` build that spans them (see the ADR's *Named runtime bundles*).

Each bundle crate also exposes these facts as constants for build gating and diagnostics:
`PLATFORM`, `BACKEND`, `SUPPORTED_TARGET_TRIPLES`, and `NATIVE_PREREQUISITES`.

## 2. Add the dependency

Releases are immutable, calendar-versioned git tags on this repository
(`runtime-YYYY.MM.patch`; see [`release/README.md`](../../release/README.md)). A consumer
pins the tag — there is no crates.io publication:

```toml
# Cargo.toml — a macOS/MLX product
[dependencies]
runtime-macos = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

The `media` feature is **on by default**, so the bundle compiles the full media-provider
graph. An **LLM-only product** (e.g. ChatWorks) turns it off and receives the same
explicitly composed LLM and snapshot-preparer catalog without compiling an unrelated
media graph:

```toml
[dependencies]
runtime-cpu = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0", default-features = false }
```

`runtime-cuda` additionally offers a `flash-attn` feature that forwards to
`candle-llm/flash-attn`.

> This repository is private; a consumer needs credentialed git access to resolve the
> dependency.

## 3. Build and validate the catalog

`catalog()` composes the bundle's media, LLM, and snapshot-preparer registries and
**validates** them: every media/LLM descriptor's `backend` must match the bundle, the
registry must contain no duplicate ids, the weights-free descriptor conformance sweep
must pass, and the bundle must carry a snapshot preparer. A cross-backend or malformed
composition fails here — before any weights are touched.

```rust
use runtime_macos as runtime; // swap for runtime_cuda / runtime_cpu

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = runtime::catalog()?; // RuntimeCatalog — validated
    assert_eq!(catalog.platform(), runtime::PLATFORM); // "macos"
    assert_eq!(catalog.backend(), runtime::BACKEND);   // "mlx"
    Ok(())
}
```

The bundle re-exports everything you need through one path — `runtime::gen_core`,
`runtime::core_llm`, the backend engine as `runtime::llm`, and (with `media`)
`runtime::media` and `runtime::providers` — so a consumer that pins the bundle does not
separately pin `gen-core` or `core-llm`.

## 4. Inspect the surface without loading weights

`snapshot()` returns a stable, serializable inventory. This is the mechanism release
tooling and product compatibility checks use, and it never loads a model:

```rust
let snapshot = catalog.snapshot();
println!("{} generators, {} LLMs", snapshot.generator_ids.len(), snapshot.text_llm_ids.len());
for id in &snapshot.generator_ids {
    println!("generator: {id}");
}
// Machine-readable form for release manifests / external smoke projects:
let json = snapshot.to_json();
```

A `RuntimeCatalogSnapshot` carries `generator_ids`, `transform_ids`, `trainer_ids`,
`captioner_ids`, `image_embedder_ids`, `text_embedder_ids`, `text_llm_ids`, and
`snapshot_preparer_backends`. The full set of ids per platform is enumerated in the
[model catalog reference](../reference/model-catalog.md).

## 5. Generate media

`catalog.media()` is a `&ProviderRegistry`. Load a generator by id with a `LoadSpec`,
`validate` the request, then `generate`. `generate` is **synchronous and blocking** (the
worker runs each job on its own thread); it streams progress through the callback and
honors the request's cancel flag.

```rust
use runtime::gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};

let media = catalog.media();

// LoadSpec::new is a dense load from a weights directory; builder methods layer on
// quantization, ControlNet/IP-Adapter/PiD overlays, LoRA adapters, and residency policy.
let spec = LoadSpec::new(WeightsSource::Dir("/models/flux1-dev".into()))
    .with_quant(runtime::gen_core::Quant::Q8);

let generator = media.load("flux1_dev", &spec)?; // Box<dyn Generator>

let request = GenerationRequest {
    prompt: "a red fox in snow, cinematic".to_string(),
    width: 1024,
    height: 1024,
    count: 1,
    ..Default::default()
};

generator.validate(&request)?; // reject unsupported conditioning/size/guidance early
let output = generator.generate(&request, &mut |p: Progress| {
    // step/decode progress events — surface a progress bar, check for cancellation, etc.
    let _ = p;
})?;

match output {
    GenerationOutput::Images(images) => { /* encode/save each Image */ }
    GenerationOutput::Video { frames, fps, audio } => { let _ = (frames, fps, audio); }
}
```

The registry exposes one `load_*` method per provider kind, each resolving by id:
`load` (generators), `load_trainer`, `load_captioner`, `load_image_embedder`,
`load_text_embedder`, and `load_transform`. `descriptor()` on a loaded provider — or
iterating `media.generators()` and calling `(r.descriptor)()` — gives capabilities,
modality, and supported samplers/schedulers/guidance methods without a load.

## 6. Serve an LLM

`catalog.text()` is a `&TextLlmRegistry`. Text generation is streaming, cancellable, and
multimodal (text + image). Load by id, or let the registry resolve the right provider for
a snapshot with `load_for_model`:

```rust
use runtime::core_llm::{LoadSpec, Message, StreamEvent, TextLlmRequest};

let text = catalog.text();
let provider = text.load_textllm("mlx-llama", &LoadSpec::dense("/models/llama-3.1-8b"))?;
// Alternatively: let provider = text.load_for_model(&LoadSpec::dense("/models/..."))?;

let request = TextLlmRequest::new(vec![Message::user("Explain diffusion in one sentence.")], 128);
provider.generate(&request, &mut |event| {
    if let StreamEvent::Token { text, .. } = event {
        print!("{text}");
    }
})?;
```

Sampling (`Sampling`), constrained decoding (`Constraint` / JSON grammar), tools, and
reasoning mode are set on `TextLlmRequest`; a provider rejects requests it can't serve in
`validate`. The full LLM contract lives in [`core-llm`](../../crates/contracts/core-llm/README.md).

## 7. Prepare a snapshot

`catalog.preparers()` is a `&SnapshotPreparerRegistry` — the backend-scoped step that
converts/normalizes on-disk weights into a loadable snapshot before serving:

```rust
use runtime::core_llm::PrepareSpec;

let report = catalog.preparers()
    .prepare_snapshot(&PrepareSpec::dense("/downloads/model", "/snapshots/model"))?;
let _ = report;
```

The backend engines (`runtime::llm`, i.e. `mlx-llm` / `candle-llm`) also expose convenience
free functions — `text_registry()`, `snapshot_preparer_registry()`, `load_textllm`,
`load_for_model`, `prepare_snapshot` — that build the registry internally. Prefer the
`catalog()`-derived registries in product code so the validated, backend-checked
composition is the single entry point.

## 8. Compose a narrower registry (tests, tools)

A test or tool that needs only one family should **not** inherit the whole platform
catalog from its link graph. Build exactly the registry you want from the family's
`register_providers` builder (or its `provider_registry()` convenience):

```rust
use runtime::gen_core::ProviderRegistryBuilder;

// Just the Z-Image family:
let registry = mlx_gen_z_image::register_providers(ProviderRegistryBuilder::new()).build()?;
let generator = registry.load("z_image_turbo", &spec)?;
```

`ProviderRegistryBuilder::build()` rejects duplicate ids per kind and yields an immutable
registry, so the set under test is exactly what you registered — never widened by an
unrelated dependency. This is the same composition model production uses, one family
instead of the whole platform.

## 9. Where weights come from

Loaders read local `.safetensors` (`WeightsSource::Dir` for a sharded directory,
`WeightsSource::File` for a single file). Inference never self-fetches weights and never derives a
download-cache location: the consumer resolves, fetches, and stages every path before calling
`load`, so a user-supplied model at an arbitrary path works and a missing component is a load-time
error rather than a mid-render fetch. There is deliberately no hub-fetch `WeightsSource` variant
(the sc-2340 direction is permanently rejected; see
[the architecture invariants](../architecture/inference-rearchitecture.md#invariants-for-future-changes)).
The revisions used for real-weight release validation are pinned in
[`release/real-weight-models.toml`](../../release/real-weight-models.toml). Descriptor
introspection and the conformance sweep are entirely weights-free — you can enumerate and
validate the whole catalog in CI without any model present.

## 10. Extend: add a provider

Adding a model is additive and does not edit core or other providers: implement the
contract (`Generator` / `Trainer` / `Captioner` / `Transform`), publish a named
registration constant, expose `register_providers`, and — the reviewable decision — add
that builder to the platform catalog and update its exact-surface test. The full recipe
and workspace conventions are in the backend guides
([`mlx-gen`](../../crates/media/mlx-gen/README.md),
[`mlx-gen/docs/MODEL_ARCHITECTURE.md`](../../crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md))
and enforced by [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## See also

- [Architecture rationale](../architecture/inference-rearchitecture.md) — why composition
  is explicit, alternatives rejected, invariants for future changes.
- [Model catalog reference](../reference/model-catalog.md) — every shipped provider id, per
  platform, with the MLX/Candle deltas.
- [`core-llm` README](../../crates/contracts/core-llm/README.md) and
  [`gen-core` README](../../crates/contracts/gen-core/README.md) — the two backend-neutral
  contracts.
- [Runtime bundle READMEs](../../crates/bundles/) — per-platform composition details.
