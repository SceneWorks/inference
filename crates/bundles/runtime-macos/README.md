# runtime-macos

The **supported Apple-silicon runtime bundle**: the explicit MLX media, LLM, and
snapshot-preparer composition, validated through [`runtime-catalog`](../runtime-catalog/README.md).
This is the crate an Apple-silicon product depends on — it is the whole product boundary,
not a loose collection of backend crates.

| | |
| --- | --- |
| `PLATFORM` | `macos` |
| `BACKEND` | `mlx` |
| `SUPPORTED_TARGET_TRIPLES` | `aarch64-apple-darwin` |
| `NATIVE_PREREQUISITES` | macOS 26.2+, Xcode Metal toolchain |
| Media surface | 57 generators, 14 trainers, JoyCaption captioner, CLIP image/text embedders |
| Text LLMs | `mlx-llama`, `mlx-joycaption` |
| Snapshot preparer | `mlx` |

The full provider surface is enumerated in the
[model catalog reference](../../../docs/reference/model-catalog.md).

## Usage

```rust
use runtime_macos as runtime;

let catalog = runtime::catalog()?;            // RuntimeCatalog — validated at construction
let generator = catalog.media().load("flux1_dev", &spec)?;
let llm = catalog.text().load_textllm("mlx-llama", &llm_spec)?;
```

`catalog()` composes `mlx-gen-catalog` (media), `mlx-llm` (LLM), and the MLX snapshot
preparer, then validates that every descriptor is on the `mlx` backend. The bundle
re-exports `gen_core`, `core_llm`, `mlx-llm` (as `llm`), and — under the default `media`
feature — `mlx-gen-catalog` (as `media`) and the provider crates (as `providers`).

## Features

- **`media`** (default) — compile the full MLX media-provider graph. An LLM-only product
  sets `default-features = false` to receive the same explicit LLM and snapshot-preparer
  catalog without the media graph. `catalog()` then returns an empty media registry.

## Depend on it

Pin an immutable release tag (this repository is not published to crates.io):

```toml
[dependencies]
runtime-macos = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

See the [Getting Started guide](../../../docs/guide/getting-started.md).

## License

Apache-2.0.
