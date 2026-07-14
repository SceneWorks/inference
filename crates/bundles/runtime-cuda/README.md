# runtime-cuda

The **supported NVIDIA runtime bundle**: the explicit Candle **CUDA** media, LLM, and
snapshot-preparer composition, validated through
[`runtime-catalog`](../runtime-catalog/README.md). This is the crate an NVIDIA product
depends on — the whole product boundary, not a loose collection of backend crates.

| | |
| --- | --- |
| `PLATFORM` | `cuda` |
| `BACKEND` | `candle` |
| `SUPPORTED_TARGET_TRIPLES` | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `NATIVE_PREREQUISITES` | NVIDIA CUDA toolkit, supported NVIDIA driver |
| Media surface | 43 generators, 6 trainers, JoyCaption captioner, CLIP image/text embedders |
| Text LLMs | `candle-llama`, `candle-llava` |
| Snapshot preparer | `candle` |

`runtime-cuda` and `runtime-cpu` ship the **same Candle media/LLM provider surface**; they
differ in target triples and native prerequisites. The full provider surface is enumerated
in the [model catalog reference](../../../docs/reference/model-catalog.md).

## Usage

```rust
use runtime_cuda as runtime;

let catalog = runtime::catalog()?;            // RuntimeCatalog — validated at construction
let generator = catalog.media().load("flux1_dev", &spec)?;
let llm = catalog.text().load_textllm("candle-llama", &llm_spec)?;
```

`catalog()` composes `candle-gen-catalog` (media), `candle-llm` (LLM), and the Candle
snapshot preparer, then validates that every descriptor is on the `candle` backend. The
bundle re-exports `gen_core`, `core_llm`, `candle-llm` (as `llm`), and — under the default
`media` feature — `candle-gen-catalog` (as `media`) and the provider crates (as `providers`).

## Features

- **`media`** (default) — compile the full Candle media-provider graph. An LLM-only
  product sets `default-features = false` to keep the explicitly composed CUDA LLM catalog
  without the media graph.
- **`flash-attn`** — forwards to `candle-llm/flash-attn`.

## Depend on it

Pin an immutable release tag (this repository is not published to crates.io):

```toml
[dependencies]
runtime-cuda = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

See the [Getting Started guide](../../../docs/guide/getting-started.md).

## License

Apache-2.0.
