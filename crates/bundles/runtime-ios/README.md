# runtime-ios

The **supported iOS runtime bundle**: the explicit MLX LLM and snapshot-preparer composition,
validated through [`runtime-catalog`](../runtime-catalog/README.md). This is the crate an
iPhone/iPad product depends on — the whole product boundary, not a loose collection of backend
crates.

| | |
| --- | --- |
| `PLATFORM` | `ios` |
| `BACKEND` | `mlx` |
| `SUPPORTED_TARGET_TRIPLES` | `aarch64-apple-ios`, `aarch64-apple-ios-sim` |
| `NATIVE_PREREQUISITES` | iOS 18.0+, Xcode 16+ with the Metal toolchain |
| Media surface | none — see below |
| Text LLMs | `mlx-llama`, `mlx-joycaption` |
| Snapshot preparer | `mlx` |

The LLM surface is **identical to [`runtime-macos`](../runtime-macos/README.md)'s**: MLX runs on
iOS, so the same `mlx-llm` engine and the same provider catalog serve both. The bundles differ in
target triples, prerequisites, and media composition.

## Why there is no media or audio registry

Not an omission — a composition decision, enforced by this crate's surface test.

`mlx-gen-catalog` composes 32 provider crates, including the video families (`wan`, `ltx`,
`mochi`, `svd`). That is the wrong shape for a device with a hard per-app memory cap, and none of
it has been validated on iOS. Depending on the whole catalog to obtain one small image generator
would compile a graph the platform cannot run.

On-device image generation instead composes a **narrow, purpose-built registry** — `mlx-gen-sana`
(with its `mlx-gen-pid` dependency) and `mlx-gen-sensenova` — as its own composition root. See
[the iOS epic breakdown](../../../docs/ios-epics.md) (E5 for SANA, E6 for the unified model).

The Candle `audio` lane is likewise out of scope until an iOS audio story exists: that lane is the
one sanctioned cross-backend seam (`docs/architecture/audio-backend-strategy.md`) and should not
be extended to a new platform incidentally.

Because the bundle is LLM-only by construction, it needs no feature gate — a consumer gets the
same surface with or without `--no-default-features`.

## Usage

```rust
use runtime_ios as runtime;

let catalog = runtime::catalog()?;            // RuntimeCatalog — validated at construction
let llm = catalog.text().load_textllm("mlx-llama", &llm_spec)?;
```

Weights arrive as caller-provisioned local paths (`WeightsSource::Dir`); this workspace never
fetches. On iOS that means the app is responsible for getting model files into its container.

## Packaging: the metallib is required

An iOS app **must** carry MLX's Metal kernel library or it fails at first Metal use, with no
build-time warning. Inside the app sandbox only two links of MLX's resolver chain are reachable:

1. `$PMETAL_METALLIB_PATH`, set before first use; or
2. `mlx.metallib` next to the executable.

The host-side links do not apply — `~/.cache/pmetal/lib` is unreadable in the sandbox, and the
compiled-in `METAL_PATH` points into the cargo target directory, which is not shipped.

Add [`scripts/ios/bundle_metallib.py`](../../../scripts/ios/bundle_metallib.py) as an Xcode "Run
Script" build phase:

```sh
python3 scripts/ios/bundle_metallib.py \
  --target-dir target --triple aarch64-apple-ios --profile release \
  --dest "$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH" \
  --expect-platform ios \
  --codesign-identity "$EXPANDED_CODE_SIGN_IDENTITY"
```

`--expect-platform ios` is worth keeping: a macOS metallib loads as far as
`newLibraryWithURL:` and then fails at the first kernel dispatch, so the guard turns a runtime
crash into a build failure.

## Threading: one provider, one thread

A loaded provider holds MLX `Array`s, and MLX's default Metal device is **not thread-safe**. Drive
one provider from one thread, or put it behind a mutex.

The type system enforces this: `Box<dyn TextLlm>` is **not `Send`**, so moving a provider between
threads does not compile (pinned by a test in this crate). That covers Rust callers completely.

**It does not cover a Swift host**, which reaches the runtime through a C ABI where Rust's marker
traits are invisible. On that side the rule is a convention you have to keep:

- Own the provider on one thread — a dedicated serial `DispatchQueue` is the simplest correct
  choice. Do **not** call in from arbitrary `Task` contexts, whose executor may hop threads.
- Never call in from the main thread for anything long: model load and generation are seconds of
  blocking work, and a blocked main thread is a watchdog kill.
- Stream callbacks fire on the calling thread. Marshal to the main thread before touching UI.

On macOS this reads as a test detail — `.cargo/config.toml` forces `RUST_TEST_THREADS=1`, so
violations stay hidden. On iOS it is app correctness, and the failure mode is intermittent.

## Building

No environment variables are needed — `.cargo/config.toml` pins the iOS deployment target:

```sh
cargo build --locked --target aarch64-apple-ios -p runtime-ios
```

The simulator triple (`aarch64-apple-ios-sim`) is a supported **build** target, exercised by CI.
It is not a supported runtime: it has no Apple Neural Engine and its Metal implementation differs
from a device's, so performance numbers and kernel-correctness claims belong on real hardware.

## Depend on it

Pin an immutable release tag (this repository is not published to crates.io):

```toml
[dependencies]
runtime-ios = { git = "https://github.com/SceneWorks/inference", tag = "runtime-2026.07.0" }
```

See the [Getting Started guide](../../../docs/guide/getting-started.md) and the
[iOS strategy](../../../docs/architecture/ios-strategy.md).

## License

Apache-2.0.
