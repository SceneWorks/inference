# runtime-ios

The **supported iOS runtime bundle**: the narrow MLX image catalog plus the MLX LLM and
snapshot-preparer composition, validated through [`runtime-catalog`](../runtime-catalog/README.md).
This is the crate an iPhone/iPad product depends on — the whole product boundary, not a loose
collection of backend crates.

| | |
| --- | --- |
| `PLATFORM` | `ios` |
| `BACKEND` | `mlx` |
| `SUPPORTED_TARGET_TRIPLES` | `aarch64-apple-ios`, `aarch64-apple-ios-sim` |
| `NATIVE_PREREQUISITES` | iOS 18.0+, Xcode 16+ with the Metal toolchain |
| Media surface (`media`, default) | `sana_1600m`, `sana_sprint_1600m` |
| Text LLMs | `mlx-llama`, `mlx-joycaption`, `mlx-starvector-1b`, `mlx-starvector-8b` |
| Snapshot preparer | `mlx` |

The LLM surface is **identical to [`runtime-macos`](../runtime-macos/README.md)'s**: MLX runs on
iOS, so the same `mlx-llm` engine and the same evidence-gated provider catalog serve both. The
bundles differ in target triples, prerequisites, and media composition.

## Why the media registry is narrow and there is no audio registry

Not an omission — a composition decision, enforced by this crate's surface test.

`mlx-gen-catalog` composes every MLX provider crate, including the video families (`wan`, `ltx`,
`mochi`, `svd`). That is the wrong shape for a device with a hard per-app memory cap, and none of
it has been validated on iOS. The image lane instead composes
[`mlx-gen-ios-catalog`](../../media/mlx-gen/mlx-gen-ios-catalog), a reviewable subset: today SANA,
whose memory ladder is the provider's own contract and whose iOS defaults (the `Sequential` load
and the DC-AE decode tile chosen from recorded device footprints against a 4 GiB budget) live in
that crate.

The Candle `audio` lane is out of scope until an iOS audio story exists: that lane is the one
sanctioned cross-backend seam (`docs/architecture/audio-backend-strategy.md`) and should not be
extended to a new platform incidentally.

## Usage

```rust
use runtime_ios as runtime;

let catalog = runtime::catalog()?;            // RuntimeCatalog — validated at construction
let limits = runtime::bound_mlx_to_platform_limits(); // once, at startup, before any load
let spec = runtime::media::sana_load_spec(weights);   // iOS load defaults
let memory = runtime::media::sana_generation_memory()?; // iOS decode geometry, on the request
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
`pmetal-mlx-sys` publishes the built metallib's path as `DEP_MLX_METALLIB`; copy that file into
the app bundle. A macOS metallib loads as far as `newLibraryWithURL:` and then fails at the first
kernel dispatch, so check its platform before shipping it.

## Threading: one provider, one thread

A loaded provider holds MLX `Array`s, and MLX's default Metal device is **not thread-safe**. Drive
one provider from one thread, or put it behind a mutex.

The type system enforces this for Rust callers: `Box<dyn TextLlm>` is **not `Send`** (pinned by a
test in this crate). **It does not cover a Swift host**, which reaches the runtime through a C ABI
where Rust's marker traits are invisible:

- Own the provider on one thread — a dedicated serial `DispatchQueue` is the simplest correct
  choice. Do **not** call in from arbitrary `Task` contexts, whose executor may hop threads.
- Never call in from the main thread for anything long: model load and generation are seconds of
  blocking work, and a blocked main thread is a watchdog kill.
- Stream callbacks fire on the calling thread. Marshal to the main thread before touching UI.

## Building

No environment variables are needed — `.cargo/config.toml` pins the iOS deployment target (18.0):

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --locked --target aarch64-apple-ios -p runtime-ios
cargo build --locked --target aarch64-apple-ios-sim -p runtime-ios
```

The simulator triple is a supported **build** target. It is not a supported runtime: it has no
Apple Neural Engine and its Metal implementation differs from a device's, so performance numbers
and kernel-correctness claims belong on real hardware.

## License

Apache-2.0.
