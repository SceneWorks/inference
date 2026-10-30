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
  methods and weights-free conformance sweeps. Memory-strategy registrations publish both their
  contract constructor and real admission `safety_check`; PiD-capable contracts declare exact
  native/PiD decode domains so catalog tests can exercise route safety without loading weights.
- **Pure host-side policy** — PIL-compatible image resize (`imageops`), VAE tiling,
  guidance/sampling policy, and the LR schedule — the math that must match the reference
  exactly and has no reason to live in a tensor backend.
- **The request-surface halves of family contracts** — the types a `GenerationRequest`,
  `MemoryPhaseFacts` or the `Generator` trait carries for one family (the MOSS-Image options and
  execution receipts, the FLUX 3 layout/reference output size, the Veda request/report and
  workspace bound, the H3 native allocation data, the HDR colour space, the LTX auto-duration
  range) stay here because the contract names them.

## What lives beside it

Family- and feature-specific host code lives in sibling crates under `crates/contracts/`, each
depending on gen-core and never the reverse, so a family's change no longer rebuilds gen-core and
everything above it (sc-27316). Each is tensor-free like gen-core.

| Crate | Holds | Consumers |
| --- | --- | --- |
| [`gen-core-train`](../gen-core-train/README.md) | trainer request validation, aspect-ratio assignment, subject-mask decode, aux-loss scheduling, resume identity, body/face loss geometry | every family trainer |
| [`gen-core-tokenizer`](../gen-core-tokenizer/README.md) | the HF `tokenizers` wrapper (`TextTokenizer`, chat templating, padding) | text-encoder providers |
| [`gen-core-iris`](../gen-core-iris/README.md) | the Iris inference and training contracts (incl. the distributed contract) | Iris providers |
| [`gen-core-flux3`](../gen-core-flux3/README.md) | FLUX 3 resources, conversion, prompt upsampler, references, training | FLUX 3 providers |
| [`gen-core-moss-image`](../gen-core-moss-image/README.md) | MOSS-Image schedule/DMD/package helpers, edit helpers, training recovery | MOSS providers |
| [`gen-core-memory-wan`](../gen-core-memory-wan/README.md) | the Wan I2V memory planner | Wan, Qwen-Image 2.1, SCAIL2 providers |
| [`gen-core-h3`](../gen-core-h3/README.md) | MiniMax-H3 memory planners, Veda predictor/workspace pricing, reference/schedule policy | MiniMax-H3 providers |
| [`gen-core-ltx`](../gen-core-ltx/README.md) | LTX split-checkpoint resolution, Gemma assets, DFR keyframes, duration head | LTX providers |
| [`gen-core-checkpoint`](../gen-core-checkpoint/README.md) | the torch `.pth` zip/pickle reader and the Supra2 converter | mlx-gen, Supra2 providers |
| [`gen-core-image`](../gen-core-image/README.md) | OpenEXR read/write and the HDR conversions | HDR lane |
| [`gen-core-family`](../gen-core-family/README.md) | small single-family policy (schedulers, resolution, keypoints, Qwen-Image 2.1, SD3) | those families' providers |
| [`gen-core-dist`](../gen-core-dist/README.md) | the Iris TCP collective transport | Iris trainers |
| [`gen-core-telemetry`](../gen-core-telemetry/README.md) | the opt-in MiniMax-H3 host telemetry collector (the request flag stays here) | MiniMax-H3 providers |

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
