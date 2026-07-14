# runtime-cpu

The **supported CPU runtime bundle**: the explicit Candle media, LLM, and
snapshot-preparer composition, validated through
[`runtime-catalog`](../runtime-catalog/README.md). This is the portable, no-accelerator
product boundary — the crate a CPU product depends on, not a loose collection of backend
crates.

| | |
| --- | --- |
| `PLATFORM` | `cpu` |
| `BACKEND` | `candle` |
| `SUPPORTED_TARGET_TRIPLES` | `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc` |
| `NATIVE_PREREQUISITES` | none |
| Media surface | 43 generators, 6 trainers, JoyCaption captioner, CLIP image/text embedders |
| Text LLMs | `candle-llama`, `candle-llava` |
| Snapshot preparer | `candle` |

`runtime-cpu` and `runtime-cuda` ship the **same Candle media/LLM provider surface**; they
differ in target triples and native prerequisites. The full provider surface is enumerated
in the [model catalog reference](../../../docs/reference/model-catalog.md).

## Usage

```rust
use runtime_cpu as runtime;

let catalog = runtime::catalog()?;            // RuntimeCatalog — validated at construction
let generator = catalog.media().load("sdxl", &spec)?;
let llm = catalog.text().load_textllm("candle-llama", &llm_spec)?;
```

`catalog()` composes `candle-gen-catalog` (media), `candle-llm` (LLM), and the Candle
snapshot preparer, then validates that every descriptor is on the `candle` backend. The
bundle re-exports `gen_core`, `core_llm`, `candle-llm` (as `llm`), and — under the default
`media` feature — `candle-gen-catalog` (as `media`) and the provider crates (as `providers`).

## Features

- **`media`** (default) — compile the full Candle media-provider graph. An LLM-only
  product sets `default-features = false` to receive the same explicit LLM and
  snapshot-preparer catalog without compiling the media graph.

## Depend on it

Pin an immutable release tag (this repository is not published to crates.io):

```toml
[dependencies]
runtime-cpu = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

See the [Getting Started guide](../../../docs/guide/getting-started.md).

## License

Apache-2.0.
