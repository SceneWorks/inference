# gen-core

> Package: `sceneworks-gen-core` · library: `gen_core`

The **backend-neutral contract layer** for SceneWorks generative-media inference. `gen-core`
has **zero tensor dependencies** — no `mlx_rs::Array`, no candle tensor. The tensor backends
([`mlx-gen`](../../media/mlx-gen/README.md) for Apple MLX,
[`candle-gen`](../../media/candle-gen/README.md) for CUDA/CPU) implement these contracts and
re-export this crate at their own paths, so a change here is reviewed against every backend
at one revision.

Numeric types on the contract are restricted to `f32` / `f64` / `Vec<f32>` / `Vec<i32>` /
`&[u8]`. `gen-core` builds and tests standalone on Linux — that lane is the proof the
contract is backend-independent.

## What it owns

- **Provider contracts** — `Generator` (text → image/video/both), `Trainer` (LoRA/LoKr
  fine-tuning), `Captioner`, `Transform`, plus the `ImageEmbedder` / `TextEmbedder` /
  `FaceEmbedder` contracts.
- **Request / output types** — `GenerationRequest`, `GenerationOutput`, `Conditioning`,
  `Capabilities`, `ModelDescriptor`, `Progress`, `CancelFlag`, and the training/caption
  analogues.
- **Load types** — `LoadSpec`, `WeightsSource`, `Quant`, `Precision`, `OffloadPolicy`,
  `AdapterSpec`, and the ControlNet / IP-Adapter / PiD / identity / external-text-encoder
  overlays layered onto a base model at load time.
- **The explicit provider registry** — `ProviderRegistryBuilder` → `build()` (rejects
  duplicate ids per kind) → an immutable `ProviderRegistry` with resolve-by-id `load_*`
  methods and a weights-free `descriptor_conformance_errors()` sweep.
- **Pure host-side policy** — tokenizer text↔ids, PIL-compatible image resize (`imageops`),
  VAE tiling, guidance/sampling policy, and the LR schedule — the math that must match the
  reference exactly and has no reason to live in a tensor backend.

## Registry model

Providers publish registration *values*; there is no `inventory`, no global mutable state,
and no linker discovery. A family adds its constants to a builder; a platform catalog selects
the families it ships:

```rust
use gen_core::ProviderRegistryBuilder;

let registry = ProviderRegistryBuilder::new()
    .register_generator(SOME_MODEL)      // fn() -> ModelDescriptor + fn(&LoadSpec) -> Box<dyn Generator>
    .build()?;                            // immutable; duplicate ids rejected here
let generator = registry.load("some_id", &spec)?;
```

See the [architecture rationale](../../../docs/architecture/inference-rearchitecture.md) for
why discovery is explicit, and the [Getting Started guide](../../../docs/guide/getting-started.md)
for the consumer path.

## The LLM contract

The independent LLM-serving library [`core-llm`](../core-llm/README.md) is re-exported at
`gen_core::core_llm`. The dependency is **inverted**: `gen-core` consumes `core-llm` (itself
tensor-free), so a consumer that already pins `gen-core` reaches the unified
`core_llm::TextLlm` engine through one path, with no separate `core-llm` pin. The legacy
in-crate `gen_core::TextLlm` trait was removed once every provider migrated.

## License

Apache-2.0.
