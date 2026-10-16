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
| Media surface (`media`, default) | `sana_1600m`, `sana_sprint_1600m`, `anima_turbo`, `sdxl` |
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
Anima Turbo and the complete SDXL family. Each memory ladder is the provider's own contract; the iOS defaults live in that
crate — for SANA the `Sequential` load and the DC-AE decode tile chosen from recorded device
footprints against a 4 GiB budget, for Anima Turbo (built without its trainers) the deferred load and
the staged + tiled selection admitted through the provider's safety check against host-measured
bounds.

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

### SDXL family registration and admission

The one `sdxl` engine serves SDXL, Illustrious-XL v1/v2, RealVisXL and RealVisXL Lightning
snapshots with the original txt2img, img2img, masked inpaint/outpaint, tile ControlNet,
IP-Adapter, LoRA/LoKr and sampler surfaces. Its generator-only registration never registers
a trainer, including when a desktop dependency unifies the optional `training` feature.
The iOS dependency itself has default features disabled.

`media::sdxl` supplies load/execution defaults, artifact-layout inspection and caller-ordered
candidate selection through the provider's registered shared memory safety check.
`media::load_sdxl` first requires bound platform limits and artifact/device/route/execution-bound
footprint evidence, then applies the runtime cap and live headroom before loading.
No validated iPhone SDXL capture exists yet, so product loads fail closed. Mac allocator peaks
and synthetic budget tests do not authorize a snapshot or constitute an iPhone fit claim.

The SDXL product loader returns an admission-bound generator, never the raw provider. Each candidate
includes its actual `GenerationRequest`; geometry, conditioning, full prompt/negative prompt (and
therefore token windows), sampling and complete execution settings are bound. A changed request
requires new admission. Cancellation and seed remain usable, and every generator/lifecycle method
is forwarded under the binding (scope ladder configuration preserves independent sequential CFG).
The source receipt retains the standard content-sealed file tokens plus the exact directory
inventory. Directory phase loaders and transformer block materialization verify those identities
before/after deferred consumption; replacing a shard/component or adding a new file invalidates it.
Initial seals may hash payloads once; later guards perform identity checks, not weight rehashing.
This has the existing `PinnedWeightsFile` persistent-replacement guarantee, not filesystem
transaction isolation against fully restored-path attacks. Candidate selection considers live
headroom before preference, and a missing/malformed preferred artifact does not hide a valid
fallback. A final capacity check runs again before load and actual generation.
