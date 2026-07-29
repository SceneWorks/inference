# Project spec — on-device generative inference for iPhone

**Status: draft for review.** The plan. The decision it implements, the options considered, and
the evidence are in the [iOS strategy](ios-strategy.md).

**Lane: MLX on iOS.** Not gated on a comparison spike — see the strategy doc §6/§7. Phase 0 below
is a *feasibility* check on the one assumption this plan rests on, with a defined fallback.

Date: 2026-07-29. Target: `SceneWorks/inference` @ `main`.

---

## 0. Scope

| Decision | Choice |
|---|---|
| **Lane** | **A — MLX on iOS.** CoreML/ANE considered and not recommended ([strategy §3, §6, §7](ios-strategy.md)) |
| **First architecture** | `Architecture::Qwen3` / `Architecture::Llama` |
| **LLM surface** | text + streaming + cancel, **tool calling**, **thinking/reasoning**, **JSON constraint** |
| **Image generation** | **launch requirement** — `mlx-gen-sana` (small, image-only) + `mlx-gen-sensenova` (unified AR LLM + image) |
| **Target device** | iPhone 17 Pro (the only device available). Older devices are a *later* priority if the work gains traction — see §0.1 |
| **Shipping** | Text-only milestone ships independently at ~wk 9–11; full v1 with image at ~wk 18–20 (§8.3) |
| **Out of scope** | video, audio, training/LoRA on device, the remaining ~47 media providers |

### 0.1 Device target, and the guardrail that goes with it

v1 develops and validates on an **iPhone 17 Pro** (12 GB RAM, ~6 GB app cap) because that is the
device we have. Broader support is explicitly wanted later if the work gains traction.

**Therefore: do not bake a 12 GB device into the architecture.** The difference between "we
tuned for 6 GB" and "we assumed 6 GB" is the difference between a later tuning exercise and a
later rearchitecture. Concretely:

- **Keep the staged load/unload seam even where v1 does not need it.** On a 17 Pro the text model
  and the image stack can be co-resident (~3.1 GB + ~2.0 GB ≈ 5.1 GB, fits). On an 8 GB device
  (~4 GB cap) they cannot. Build the load/unload boundary in T2 and simply leave it disabled —
  retrofitting it into a pipeline that assumed co-residency is the expensive version.
- **Record peak RSS against two lines, not one:** the ~6 GB cap we ship against, and a ~4 GB
  reference line for the 8 GB class. The second number costs nothing to log and tells us, at any
  moment, how far a broader-device release actually is.
- **Treat any *hard* dependency on >4 GB as a design decision requiring a note**, not an
  incidental outcome of tuning.

This is the one place where the narrow device availability could quietly become an architectural
commitment, so it is called out here rather than left to T2.

### What this is structurally

An **iOS-capable `mlx-sys` build plus one new bundle**, inside this workspace:

```
crates/bundles/runtime-ios/         new — the composition root
crates/bundles/runtime-ios-ffi/     new — the Swift/host boundary (§5.5)
pmetal-mlx-sys fork                 modified — iOS target support (§4.1)
```

`mlx-llm`, `mlx-gen`, `mlx-gen-sana`, `mlx-gen-pid`, and `mlx-gen-sensenova` are consumed
**unchanged**. No new engine crate, and **no contract changes** — Lane A registers
`mlx_llm::text_registry()` and `mlx_llm::snapshot_preparer_registry()` exactly as `runtime-macos`
does.

---

## 1. Goals and non-goals

**Goals**

- G1. A `TextLlm` provider passing `core_llm_testkit::textllm_conformance` on a real Qwen3/Llama
  snapshot, on a physical iPhone.
- G2. Declared capabilities: `supports_tools`, `supports_thinking`,
  `supported_constraints = [ConstraintKind::Json]`.
- G3. A `runtime-ios` bundle validated by `runtime-catalog`, consumable from Swift.
- G4. Weights arrive as caller-provisioned local paths (`WeightsSource::Dir`) — the epic-13657
  self-fetch boundary preserved without exception.
- G5. Published numbers: TTFT, tok/s steady, peak RSS, energy per 100 tokens.
- G6. `mlx-gen-sana` generating on-device within the memory cap.
- G7. `mlx-gen-sensenova` producing both text and image output on-device.

**Non-goals**

- N1. Video (`wan`, `ltx`, `mochi`, `svd`), audio, training/LoRA on device.
- N2. Replacing `mlx-llm` on macOS. `runtime-macos` untouched.
- N3. The ANE. Deferred, not rejected — see [strategy §7.3](ios-strategy.md) for the
  disaggregated prefill/decode form it would take post-v1.
- N4. Apple Foundation Models as a registry provider (strategy §4).
- N5. The remaining ~47 media provider crates.

---

## 2. Phase 0 — feasibility spike (2 weeks)

**This is not a lane gate.** The lane is chosen. This tests the single assumption Lane A rests
on — that MLX can be built for iOS at all — before three months of work is committed to it.

The pinned `pmetal-mlx-sys` @ `932beb4` has no iOS story: `build.rs` is
`#[cfg(target_os = "macos")]`-gated in three places, drives `cmake::Config` with no iOS toolchain
args, caches `mlx.metallib` to `~/.cache/pmetal/lib/` (meaningless in an app sandbox), and applies
three `required = true` patches to MLX core.

**Work:** fork the fork; add `CMAKE_SYSTEM_NAME=iOS` + `IPHONEOS_DEPLOYMENT_TARGET`; relocate the
metallib into the `.app` bundle via the patched resolver's `set_metallib_path()` /
`$PMETAL_METALLIB_PATH`; confirm all three patches still apply; link a Rust staticlib into a bare
Xcode app.

**Exit criteria — every row answered with a number or a hard no:**

| Question | Pass bar |
|---|---|
| Does `cargo build --target aarch64-apple-ios -p mlx-llm` link? | yes/no — **the gate** |
| Does a Qwen3-4B Q4 snapshot load and generate on-device? | yes/no |
| **Does `mlx-gen` core + `-pid` + `-sana` also build for iOS?** | yes/no — **the Track 2 gate** |
| Steady-state decode | tok/s |
| Peak RSS under sustained decode | MB, vs. the per-app cap |
| Energy per 100 tokens | mWh (Instruments Energy Log) |
| Sustained thermals over 5 min | tok/s at t=0 vs t=5min |
| Fork delta size | LoC diff vs upstream `build.rs` |

The media row costs about a day and is worth every hour: it converts Track 2 from an assumption
into a measured fact. The Rust is known to be platform-neutral (zero `cfg(target_os)` across
`mlx-gen/src`, `-sana`, `-pid`, `-sensenova`), so this is really a second check on the same
`build.rs`.

**Energy and thermals are measured here even though no comparison is pending.** They are the v1
regression baselines (G5), and they are the evidence that would reopen the ANE question later
([strategy §7.2](ios-strategy.md)).

### 2.1 First results — 2026-07-29

`cargo build --target aarch64-apple-ios -p mlx-llm` was run on the development machine (§8.4).
**It gets much further than the static analysis predicted.**

What worked:

- The entire Rust dependency tree — `core-llm`, `tokenizers` 0.21, `minijinja`, `compact_str` and
  the rest — compiles cleanly for `aarch64-apple-ios`. No source changes, no feature juggling.
- `pmetal-mlx-sys`'s cmake build of MLX's C++ **ran to completion**, and produced a metallib.

Two failures, and the distinction between them matters:

**(1) iOS-specific, small, well-understood.**

```
error: failed to add native library …/lib/clang/21.0.0/lib/darwin/libclang_rt.osx.a:
       Unsupported archive identifier
```

`build.rs`'s `find_clang_runtime_lib` resolves the clang runtime via `xcrun` and unconditionally
takes the **`osx`** variant; linking a macOS archive into an iOS binary is rejected. The fix is
target-aware selection (`libclang_rt.ios.a`) inside that one function.

Alongside it, an observed — not inferred — confirmation of the sandbox problem:

```
warning: pmetal-mlx-sys@0.2.4: Cached mlx.metallib to /Users/…/.cache/pmetal/lib/mlx.metallib
```

The metallib was cached to `$HOME` **for the iOS target**, exactly as predicted.

This also corrects a claim in an earlier draft: build scripts compile for the *host*, so the
`#[cfg(target_os = "macos")]` branches **run** during an iOS cross-build. They do not skip iOS
setup, they mis-configure it. The remedy is branching on `CARGO_CFG_TARGET_OS` / `TARGET`, not
removing the gates.

**(2) A latent bug in the fork's non-macOS path — diagnosed, and trivial.**

```
error[E0061]: this function takes 10 arguments but 8 arguments were supplied
  mlx-rs/src/ops/quantization.rs:280   mlx_sys::mlx_qqmm(…)
```

Diagnosed rather than guessed at, in three steps:

1. A clean `--target aarch64-apple-darwin` build in a fresh target directory **succeeds**
   (2m59s), so the pinned rev is not broken in general and is not merely green from cache.
2. The generated `mlx_qqmm` bindings are **byte-identical** between the darwin and iOS target
   directories — ten parameters in both. So this is not a bindgen or staging difference.
3. The call site sits inside `qqmm_device`, gated **`#[cfg(not(target_os = "macos"))]`**
   (`quantization.rs:253`). Its own doc comment says: *"only supported on GPU with the CUDA
   backend (Linux with NVIDIA GPU). It is not available on macOS."*

So the function is **dead code on macOS and has silently rotted** — its call site was never
updated when `mlx_qqmm` gained `global_scale_x` / `global_scale_w`. iOS is also
`not(target_os = "macos")`, so the gate lets it compile on a platform that has no CUDA backend at
all.

**The correct fix is to narrow the cfg, not to add two arguments.** `not(macos)` was written to
mean "Linux/CUDA"; it should say so (`target_os = "linux"`, or a CUDA feature gate). On iOS this
function is meaningless and should not be compiled. Small, obviously correct, and **the first
thing to send upstream to `pmetal-mlx-rs`** — it is a latent defect for any non-macOS,
non-Linux target, not something iOS introduced.

### 2.2 Phase 0 build result — **complete and green**

**It builds. All of it.** Worked through in one session on 2026-07-29:

```
$ IPHONEOS_DEPLOYMENT_TARGET=18.0 \
    cargo build --target aarch64-apple-ios -p mlx-llm-server -p mlx-gen-sana
    Finished `dev` profile in 11.60s

$ file target/aarch64-apple-ios/debug/mlx-llm-server
    Mach-O 64-bit executable arm64
    LC_BUILD_VERSION: platform 2 (iOS), minos 18.0, sdk 26.5
    links Metal.framework + Accelerate.framework
```

That artifact is the **OpenAI-compatible LLM server, fully linked for iPhone** — not an rlib, a
real executable. `mlx-gen` core, `mlx-gen-pid` and `mlx-gen-sana` build alongside it, so the
Track 2 gate is green too.

| # | Blocker | Status |
|---|---|---|
| 1 | `qqmm_device`'s `cfg(not(target_os = "macos"))` catches iOS for a CUDA-only function, whose call site had also gone stale (8 of 10 args) | **Fixed** — [SceneWorks/mlx-rs#23](https://github.com/SceneWorks/mlx-rs/pull/23) |
| 2 | `build.rs` links `libclang_rt.osx.a` unconditionally | **Fixed** — same PR. Skip the link off macOS entirely; it is a macOS-26+ `___isPlatformVersionAtLeast` workaround, and the `ios`/`iossim` archives are rejected identically *and* unneeded. |
| 3 | **`IPHONEOS_DEPLOYMENT_TARGET` unset** — `rustc`'s default for `aarch64-apple-ios` is **10.0**, which drops `___chkstk_darwin` (libSystem, iOS 12+) at link time | **Diagnosed, needs a home** — see below |
| 4 | `mlx.metallib` cached to `~/.cache/pmetal/lib` instead of bundled into the `.app` | **Worse than "open" — it poisons the macOS cache.** See below. |
| 5 | The iOS build's Metal kernels were compiled for **`macosx`**, not iOS | **Fixed** — `patches/ios-metal-sdk.patch`. Metallib now `air64-apple-ios18.0.0`. |

**Blocker 4 is a live hazard for macOS developers, not just an iOS gap.** Verified on this
machine: an iOS cross-build **overwrote `~/.cache/pmetal/lib/mlx.metallib`** — the shared,
platform-agnostic cache path — with its own artifact.

That cache is documented in the repo's own `CLAUDE.md` as load-bearing: local `cargo test` / `cargo run`
binaries have **no compiled-in metallib**, so the user-cache copy is their *sole* working
resolution. The existing sc-7889 hazard is a stale-but-valid macOS metallib shadowing a newer one;
this is a *different platform's* metallib landing in the same slot, and the resolver has no way to
tell them apart. The fix is to gate the cache write to `CARGO_CFG_TARGET_OS = "macos"`, which is
correct regardless of iOS: an `.app` bundles its metallib and cannot read `$HOME` usefully anyway.

**Blocker 5 was found by following blocker 4, and it is the one that matters.** `strings` on the
iOS build's metallib reports `macosx`. Adding `CMAKE_SYSTEM_NAME=iOS` +
`CMAKE_OSX_ARCHITECTURES=arm64` fixed the **C++** objects (`minos` 26.2 → 16.0, verified) but
**not** the Metal kernels, because the metallib is compiled by rules that hardcode the SDK:

```cmake
# mlx/backend/metal/kernels/CMakeLists.txt — upstream MLX, not the fork
"-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
COMMAND xcrun -sdk macosx metal ${METAL_FLAGS} -c ${SRCFILE}
...
set(METAL_LINK_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
COMMAND xcrun -sdk macosx metal ${METAL_LINK_FLAGS} ${KERNEL_AIR} -o
```

`xcrun -sdk macosx metal` compiles Metal against the **macOS** SDK unconditionally, and
`-mmacosx-version-min` is meaningless for an iOS target. MLX's root `CMakeLists.txt` *does* guard
its SDK probe on `CMAKE_SYSTEM_NAME STREQUAL "Darwin"` and sets `MLX_METAL_VERSION 0` otherwise —
so upstream anticipated non-macOS configuration — but the kernel rules were never given the
matching iOS path.

**So a binary that links cleanly still carries macOS Metal kernels and would fail at
runtime on a device.** This is exactly the failure §2.2's earlier draft could not have caught: a
green build is not evidence of a working device artifact. The `otool`/`strings` platform checks
above should be a **CI assertion in Tier 1**, not a manual step.

**Resolved — `patches/ios-metal-sdk.patch`, the fourth required `mlx-sys` patch.** Taken as a
fork patch rather than an upstream-first PR so E1 does not block on review latency; the upstream
PR follows.

The patch selects the SDK and version-min prefix by platform at all three sites (per-kernel
compile, metallib link, root probe), and adds an iOS arm to the root `MLX_METAL_VERSION` probe —
falling through to the existing `MLX_METAL_VERSION 0` would have silently dropped every
version-gated kernel.

**The deployment floor moved 16.0 → 18.0, and the reason is not recency.** Metal versions map to
OS versions:

| iOS | Metal | |
|---|---|---|
| 16 | 300 | **below MLX's own baseline** |
| 17 | 310 | = MLX's macOS 14.0 floor |
| 18 | 320 | chosen |

MLX's macOS floor of 14.0 *is* Metal 310, so an iOS 16 floor would have compiled its kernels
below the baseline they assume. 18.0 also keeps `fence` coherent: the kernel is built only at
`MLX_METAL_VERSION >= 320`, while `fence.cpp`'s runtime guard is
`__builtin_available(macOS 15, iOS 18, *)` — so a 17.0 floor would satisfy the runtime check on
an iOS 18 device with the kernel never compiled in. Latent today (`MLX_METAL_FAST_SYNCH` defaults
off) but a real trap. The NAX gate fails safe on iOS: it needs Metal ≥ 400 **and**
`MACOS_SDK_VERSION >= 26.2`, and the latter is unset off macOS.

**Verified on a clean build**, all three layers agreeing:

| Artifact | Before | After |
|---|---|---|
| `mlx.metallib` (15,660 kernels) | `apple-macos` | **`air64-apple-ios18.0.0`** |
| MLX C++ objects | platform iOS, `minos` 26.2 | platform iOS, **`minos` 18.0** |
| `mlx-llm-server` | linked, macOS kernels inside | **Mach-O arm64, platform 2, minos 18.0, sdk 26.5** |
| `~/.cache/pmetal/lib` | overwritten by cross-build | **hash unchanged** |

**Actual cost: under a day**, against the 3–5 day estimate — the work was diagnosis, not
kernel-level porting, and the `strings`/`otool` checks made each step falsifiable. The estimate
assumed a NAX-style correctness investigation that the fail-safe gate made unnecessary.

**Still unproven: that these kernels are *correct*, not merely iOS-targeted.** The sc-2772 NAX
miscompile is precedent for a metallib that builds and emits garbage. That needs the device, and
it is E3's first real test.

**Blocker 3 is the interesting one**, because it is configuration rather than code and it has a
second face. The MLX C++ objects came out with `minos 26.2` — `MACOSX_DEPLOYMENT_TARGET` from
`.cargo/config.toml` silently leaking in as the *iOS* floor. It happened not to error (iOS 26.2
exists), but it would have excluded every device below iOS 26.2 from a shipped app. Meanwhile the
Rust side linked at 10.0. Both halves need an **intentional** iOS deployment target; A1 must set
it in `.cargo/config.toml` (or in the build script's deployment-target logic) rather than relying
on an env var at the command line.

**Nothing architectural was found**, in the sense that matters: no missing Metal support on iOS,
no unbuildable C++, no absent capability. But blocker 5 is a genuine porting task in upstream
MLX, not a configuration fix — so the early read that "the blocker list is short and mechanical"
was **half right**. Blockers 1–3 were mechanical; blocker 5 is not.

**R1 is downgraded, not retired.** The build reaches a linked binary and the C++ cross-compiles
correctly, so the lane is sound. But "it builds" turned out not to mean "it runs", and A1's three
weeks should stay budgeted: ~1 day of it is spent, blocker 5 needs 3–5 days, and metallib
bundling plus the deployment-target homing follow. The device remains unproven.

### 2.3 Fork strategy — decided

**Fork `pmetal-mlx-rs` now, open upstream PRs in parallel.** A1 is not blocked on someone else's
review latency; the fork is dropped if and when the PRs merge.

Mechanics, checked against the gate:

- `scripts/check-workspace.py`'s `PINNED_WORKSPACE_DEPENDENCIES` (line 35) asserts the **`rev`**
  and the **`package` alias** — it does **not** assert the git URL. So a fork that keeps the
  package names `pmetal-mlx-rs` / `pmetal-mlx-sys` needs only the two revision strings updated
  there, plus the matching `git`/`rev` in the root `Cargo.toml`. `DEFAULT_GRAPH_PINNED_PACKAGES`
  derives from the same dict, so one edit covers both the manifest check and the resolved-graph
  check.
- Keep the fork's delta **additive and minimal** — three fixes, nothing else — so the upstream
  PRs stay reviewable and the eventual rebase is cheap.
- Record the fork's lineage in the root `Cargo.toml` comment block, matching the existing
  convention there for SHA history.

Three PRs, in ascending order of how much discussion they will need:

| PR | Change | Note |
|---|---|---|
| 1 | Narrow `qqmm_device`'s `cfg(not(target_os = "macos"))` to the CUDA/Linux target it documents | Fixes a latent defect on **any** non-macOS/non-Linux target, not just iOS. Strongest standalone case. |
| 2 | Target-aware clang runtime selection in `find_clang_runtime_lib` | `libclang_rt.ios.a` when `CARGO_CFG_TARGET_OS = "ios"`. |
| 3 | iOS metallib resolution — bundle rather than `$HOME` cache | Touches the existing patched resolver; most likely to need discussion upstream. Carry longest in the fork. |

### If Phase 0 fails

If `mlx-sys` proves genuinely un-buildable for `aarch64-apple-ios`, this plan does not survive
contact and **the fallback is §3, not a retry**. Escalate rather than absorbing it: the fallback
costs more, ships less, and strands the image-generation requirement.

---

## 3. Contingency — `coreml-llm` (only if Phase 0 fails)

Not the plan. Recorded so the fallback is not designed under time pressure.

A `crates/llm/coreml-llm` engine implementing `TextLlm` over CoreML: ~4–6k lines of Rust
(`objc2` FFI, prefill + stateful `MLState` step, CPU sampler over the logits buffer) plus a
Python `coremltools` conversion pipeline outside the workspace. **~12 weeks for one architecture
family, +2–4 weeks per family thereafter.** It also requires three contract-level changes Lane A
does not ([strategy §5](ios-strategy.md)): an additive `ModelFormat::CoreML`, a widened
`TextLlmDescriptor.backend` doc, and a second `runtime-catalog` backend carve-out.

**It does not satisfy G6/G7.** Image generation would need re-planning from scratch — see
[strategy §6](ios-strategy.md) for why the CoreML saving that makes an LLM port cheap does not
apply to a diffusion graph.

---

## 4. Lane A — the build (7 weeks)

| Phase | Wks | Content | Exit |
|---|---|---|---|
| A1 | 3 | Productionize the Phase 0 fork: iOS cmake toolchain, metallib bundling, patch verification, upstreaming attempt | reproducible `aarch64-apple-ios` build in CI |
| A2 | 2 | `runtime-ios` bundle; memory tuning under the per-app cap; **documented threading contract** | Qwen3-4B Q4 sustained decode without jetsam |
| A3 | 2 | CI lanes + on-device harness + docs | `textllm_conformance` green on-device |

**What this gets:** all ten `Architecture` variants, vision, GGUF ingest, speculative decode,
paged KV — `mlx-llm` unchanged — plus the foundation Track 2 builds on.

**What this owns:** an iOS-capable fork of a fork of `mlx-rs`. Keep the delta additive and attempt
upstreaming to `pmetal-mlx-rs` during A1; a merged iOS target erases most of the maintenance tax.

### 4.1 Threading contract (A2 exit criterion — do not skip)

`mlx-llm`'s own docs state engine instances hold MLX `Array`s and are **neither `Send` nor
`Sync`**, and `.cargo/config.toml` forces `RUST_TEST_THREADS=1` because MLX's shared default Metal
device is not thread-safe (it SIGSEGVs under a parallel harness). On macOS that reads as a test
detail. **On iOS it is a host-app correctness requirement**: a Swift host calling in from a
concurrency context will produce intermittent crashes unless the contract is explicit.

A2 must deliver: one engine per thread (or behind a mutex), a documented rule for which
thread/queue owns the engine, a stated lifetime for the stream callback, and a hostile-threading
test that fails if the rule is violated. This is the class of bug that surfaces weeks after the
lane is declared done.

### 4.2 Declared capabilities

`mlx-llm`'s existing `LlamaProvider` descriptor is reused as-is. For the record, the v1 surface:

```rust
TextLlmCapabilities {
    max_context_tokens: <model config>,
    max_new_tokens: <cap>,
    supports_system_prompt: true,
    supports_vision: <per-snapshot, via weightless_vision>,
    supports_video: false,
    supports_thinking: true,
    supports_tools: true,
    supported_constraints: vec![ConstraintKind::Json],
}
```

All four requested surfaces (text, tools, thinking, JSON constraint) are **already implemented and
conformance-tested** on `mlx-llm`. On this lane they are inherited, not built.

---

## 5. Repo-side changes

### 5.1 Checklist

- [ ] `scripts/check-workspace.py`: bump `EXPECTED_MEMBER_COUNT` (currently **90**) → 92
      (`runtime-ios` + `runtime-ios-ffi`)
- [ ] `scripts/check-workspace.py`: confirm the pinned-backend-revision assertion accommodates the
      iOS-capable `mlx-sys` rev, and that no new env side channel appears
- [ ] `scripts/ci/select_lanes.py`: add `"ios_device"` to the `LANES` tuple; add a
      `crates/bundles/runtime-ios` path rule (mirroring the `runtime-macos` rule); verify
      unclassified paths still fail safe to all lanes
- [ ] `crates/bundles/runtime-ios/`: `PLATFORM = "ios"`, `BACKEND = "mlx"`,
      `SUPPORTED_TARGET_TRIPLES = ["aarch64-apple-ios"]`,
      `NATIVE_PREREQUISITES = ["iOS 18+", "Xcode 16+"]`, `catalog()` via `RuntimeCatalog::try_new`,
      `default = ["media"]`
- [ ] Ordered catalog surface test for the new bundle, under both `--no-default-features` and
      `--features media`
- [ ] `deny.toml`: confirm any new iOS-side deps pass `cargo deny check licenses`
- [ ] `docs/architecture/ios-strategy.md`: record the Phase 0 outcome
- [ ] `.github/workflows/real-weights.yml`: gated on-device conformance job

**No contract-crate changes.** `core-llm` and `gen-core` are untouched — that is a property of
this lane, and a regression if it stops being true.

### 5.2 The Swift/host FFI layer and host app — building the first one

**Confirmed: no Swift host app exists.** This project builds it, and it is **required
infrastructure, not optional product work** — `xcodebuild test` needs an Xcode test host, so
Tier 3 device CI (§5.3) cannot exist without at least a minimal app target. Treat the host app
as a deliverable of A2/A3, not as something a consumer supplies later.

Two pieces:

**(a) `crates/bundles/runtime-ios-ffi`** — the C boundary. `TextLlm::generate` takes
`&mut dyn FnMut(StreamEvent)`, a Rust closure that **does not cross FFI**. Scope covers:

- a C ABI or `swift-bridge`/UniFFI surface for load / generate / cancel, and for image generation
  (`gen_core::Generator` has the same callback problem via `on_progress`)
- token-stream delivery: a C callback with a documented calling thread, or a pull-based iterator
- ownership and lifetime rules for the callback and the engine handle (see §4.1)
- error mapping across the boundary
- `.xcframework` packaging

**(b) A minimal SwiftUI host app** (`ios-host/`, outside the Cargo workspace) — a chat view, an
image-generation view, a model-picker over `WeightsSource::Dir` paths, and an XCTest target that
drives the conformance harness on-device. Deliberately thin: it exists to exercise and measure
the runtime, not to be a product.

**Budget 2–3 weeks for (a), 1–2 for (b).** Design (a) during Phase 0 — it needs no working build.

### 5.2.1 The headless option already exists — and it changes the critical path

The consumer for the week 9–11 milestone is undecided: either a test-harness app, or a headless
engine driven remotely. **The headless shape is already built.**

`crates/llm/mlx-llm/server` (`mlx-llm-server`) is an OpenAI-compatible HTTP server —
`POST /v1/chat/completions` with SSE streaming, `GET /v1/models`, a health check — written
directly on `std::net::TcpListener` with **no async runtime and no HTTP framework**, speaking only
the `TextLlm` contract. It is deliberately minimal: one model, one request at a time (matching
MLX's single-threaded Metal device), `Connection: close`, **no auth**.

**Recommendation: make the week 9–11 milestone the headless server.** It is the cheapest path to
a shippable milestone *and* the one that best fits an undecided consumer:

- **It takes §5.2(a) off the critical path — roughly 2–3 weeks.** The app target becomes a shell
  that starts the server on a background thread. No callback marshalling across FFI, no
  `.xcframework` API design, no stable-ABI commitment.
- **It resolves the §8.5 tension rather than deferring it.** With no named consumer, designing a
  frozen C ABI would be guessing. An HTTP surface is already a stable, versioned, well-understood
  contract, and it is trivially drivable for evaluation and regression work.
- **The test-harness app is then nearly free** — the same shell satisfies the `xcodebuild test`
  host requirement (§5.3), so the two candidate consumers stop being alternatives.

Two caveats that must be handled, not assumed away:

1. **iOS has no background daemons.** "Headless" here means *a foregrounded app exposing a local
   port*. It stops serving when backgrounded, and using audio/location background modes to dodge
   that is App Store abuse. Fine for development and evaluation; **not** a product architecture.
2. **The server has no authentication.** Binding it to a LAN interface exposes an unauthenticated
   LLM to the network. Bind to loopback and reach it over USB port forwarding, or add a bearer
   token before it leaves the desk. Its own docs already call it a reference, not a production
   gateway — respect that boundary.

The full FFI layer (§5.2a) remains necessary for a genuine native consumer. This defers it to
when one is named, which is the right time to design it.

### 5.3 CI — the part with no precedent

All three existing bundles are desktop targets that `cargo test` natively. iOS is not. **Budget
separately; this is infrastructure, not a line item.**

- **Tier 1 (every PR, hosted):** `cargo build --target aarch64-apple-ios` + `clippy -D warnings`.
  Build regressions only, no test execution.
- **Tier 2 (nightly, self-hosted Mac):** conformance on the **simulator**
  (`aarch64-apple-ios-sim`).
- **Tier 3 (pre-release, self-hosted Mac + tethered iPhone):** `textllm_conformance` and the
  media conformance suite on a physical device via `xcodebuild test`, plus the Phase 0 numbers as
  regression baselines.

**Decided: a self-hosted runner on the primary development machine (§8.4), with one dedicated
iPhone 17 Pro tethered by USB.** Chosen over a cloud device farm specifically because farms
typically restrict Instruments-level profiling, and the energy and thermal numbers in G5 are the
whole point of Tier 3 — a farm that cannot produce them cannot enforce the baselines.

Setup work, and its consequences:

- Register the machine as a GitHub Actions self-hosted runner, labelled for the iOS lane.
- Dedicate one iPhone 17 Pro to it. It cannot double as a daily-driver phone; unlock state and
  storage pressure both break unattended runs.
- The machine must stay online for nightly Tier 2 and pre-release Tier 3. If it sleeps, the lane
  silently stops reporting — add a heartbeat check rather than trusting green-by-absence.
- **Single point of failure, accepted.** One device, one runner, one machine. Tier 1 stays on
  hosted runners so an outage never blocks PRs.

### 5.4 Conformance wiring

`mlx-llm`'s existing gated conformance tests carry over unchanged; the on-device harness runs them
against `MLX_LLM_TEST_MODEL`. Passed-in-path env vars are explicitly allowed by the self-fetch
lint; do **not** derive a cache location.

---

## 6. Track 2 — image generation (9 weeks)

A launch requirement, not an add-on. Both capabilities already exist as crates; the work is
building and tuning them, not writing them.

| Need | Crate | Size | Notes |
|---|---|---|---|
| Unified AR LLM + image | `mlx-gen-sensenova` | 9.6k lines | SenseNova-U1 (NEO-Unify). Under active work (`e13aae06`). |
| Small image-only | `mlx-gen-sana` | 6.4k lines | SANA (NVlabs) + DC-AE decoder. Text encoder **is Gemma-2-2B-it**, reused from `mlx-gen-pid`. |
| Alternative | `mlx-gen-z-image` | 15.1k lines | Z-Image-Turbo — few-step, larger |

Rough budget at 4-bit: Gemma-2-2B encoder ~1.4 GB + SANA DiT ~0.35 GB + DC-AE decoder ≈ **~2 GB**.
The crate docs also reference a **2-bit** SANA drop, which would roughly halve the encoder.

### 6.1 Phasing

| Phase | Wks | Content | Exit |
|---|---|---|---|
| T1 | 2 | Build `mlx-gen` core + `-pid` + `-sana` for `aarch64-apple-ios`; resolve fixture/Metal-kernel issues | SANA produces a correct image on-device |
| T2 | 2 | Memory tuning: encoder/DiT/decoder residency, staged load-unload, DC-AE tiling; 2-bit encoder if needed | 1024px generation without jetsam, peak RSS recorded |
| T3 | 3 | `mlx-gen-sensenova` on-device: dual-path AR + flow-matching, sharing `mlx-llm`'s KV cache with the text lane | unified text+image generation on-device |
| T4 | 2 | `runtime-ios` `media` feature; ordered catalog surface test; `gen-core-testkit` conformance | media registry validated in the bundle |

Bundle behind a `media` feature exactly as `runtime-macos` does, so an LLM-only host builds
`--no-default-features`.

### 6.2 Why this track is cheap on this lane

`grep -rn 'target_os|cfg(unix)|cfg(windows)'` across `mlx-gen/src`, `-sana`, `-pid`, and
`-sensenova` returns **zero** platform gates. The media Rust is platform-neutral; the only iOS
blocker is `mlx-sys`'s `build.rs` — already A1's work. Track 2's build risk collapses into Lane
A's, which is the practical form of [strategy §6](ios-strategy.md)'s argument.

`mlx-gen-sensenova` also depends on `mlx-llm` directly, consuming `ContiguousKvCache`, `sample`,
and `Rope` (sc-7159). On this lane that coupling is free — both are the same MLX build. That is
precisely what a CoreML text lane would have broken.

---

## 7. Risks

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | **`mlx-sys` cannot be built for iOS** | **VL** | **Critical** | **Effectively retired** (§2.2). Metallib, C++ objects and the linked binary all target iOS 18. Fallback §3 stays formally live only until kernels are proven correct on device. |
| R8 | ~~Metal kernels must be cross-compiled for iOS in upstream MLX~~ | — | — | **Fixed** — `patches/ios-metal-sdk.patch` (blocker 5). |
| R11 | The iOS kernels compile but are **numerically wrong** | L | H | The residual of R8, and the one thing a build cannot prove. Precedent: sc-2772, where a metallib built cleanly and emitted garbage. First real test in E3, on device. |
| R9 | A green build is mistaken for a working device artifact | M | H | **Already happened once.** Add `otool -l` / `strings` platform assertions to Tier 1 CI so the metallib and objects are checked for `platform 2` / `iphoneos`, not just for linking. |
| R10 | Cross-builds poison the shared `~/.cache/pmetal/lib` metallib | — | H | **Fixed** — cache write gated to `CARGO_CFG_TARGET_OS = "macos"`. Verified: an iOS build now leaves the hash unchanged. Was silently breaking local macOS `cargo test`. |
| R2 | Per-app memory cap tighter than the reported ~6 GB | M | H | Measured in Phase 0. Conclusion holds at 4 GB: ~3–4B at 4-bit. |
| R3 | mlx-sys iOS fork drifts from upstream | H | M | Keep the delta additive; attempt upstreaming in A1. |
| R4 | Threading contract violated by the Swift host | M | H | Documented ownership rule + hostile-threading test in A2 (§4.1). |
| R5 | Host FFI layer underestimated | M | M | Designed during Phase 0, implemented in a named phase (§5.2). |
| R6 | Tier 3 device CI slips | M | M | Ship on Tier 1+2; Tier 3 is a pre-release gate, not per-PR. |
| R7 | GPU-only decode costs unacceptable battery/thermals | M | M | Measured in Phase 0 as a baseline. If it fails the product bar, that is the [strategy §7.2](ios-strategy.md) reopen trigger — and §7.3's disaggregated ANE prefill is the remedy, post-v1. |
| T-R1 | Peak RSS during generation exceeds the cap (encoder + DiT + decoder co-resident) | H | H | Staged load/unload between stages; 2-bit encoder; DC-AE tiling. Measure in T2 before committing to T3. |
| T-R2 | `mlx-gen` relies on macOS-only Metal paths / metallib cache (sc-7889) | L | H | Zero `cfg(target_os)` in the media Rust — verified. Residual is the metallib bundling, same fix as A1. |
| T-R3 | Generation latency unacceptable (thermal throttle mid-generation) | M | M | Few-step models only (SANA, Z-Image-Turbo); measure sustained, not cold-start. |

---

## 8. Staffing and timeline

**Team: one human engineer plus Claude.** That is not "two engineers", and modelling it that way
would give a wrong schedule. The work splits by *what requires physical hardware and an Apple
Developer account* — not by headcount.

### 8.1 Division of labour

| Work | Owner | Why |
|---|---|---|
| `mlx-sys` fork: cmake iOS toolchain, metallib bundling, patch verification | Claude | Pure build engineering, verified by `cargo build --target aarch64-apple-ios` |
| `runtime-ios` + `runtime-ios-ffi` crates, catalog surface tests, conformance wiring | Claude | Rust, verified by the repo's existing gates |
| CI config (`select_lanes.py`, workflows, Tier 1/2) | Claude | Config plus the repo's own Python gates |
| SwiftUI host app + XCTest target (§5.2b) | Claude drafts, you build and run | Claude can write Swift; only you can drive Xcode's GUI and a device |
| Signing, provisioning, `com.apple.developer.kernel.increased-memory-limit` entitlement | **You only** | Apple Developer account. The entitlement needs a request to Apple — **start it in Phase 0**, its lead time is not under our control |
| Device measurement: Instruments energy/thermals, jetsam behaviour, the real per-app cap | **You only** | Physical iPhone |
| Self-hosted runner + tethered device (Tier 3) | **You only** | Hardware and network access |

### 8.2 The actual bottleneck

**Code throughput is not the constraint; device time and Apple provisioning are.** Every phase
with a measurement exit criterion — Phase 0's energy/thermal rows, A2's memory tuning, T2's
residency work, all of §9's published numbers — is gated on your hands-on time, not on how fast
the Rust lands. Plan the schedule around your availability for device sessions.

Two consequences worth acting on now:

1. **Request the increased-memory-limit entitlement during Phase 0**, before it is needed. It
   gates A2 and T2, and its turnaround is Apple's, not ours.
2. **Batch device work.** A2 and T2 are both "load it, watch RSS, tune, repeat" — running them as
   scheduled sessions rather than ad-hoc interrupts is materially faster.

### 8.3 Timeline

```
        wk  0    2         5      7      9        11     13      16     18
Phase 0 [====]                                                            2
Lane A       [========][====][====]                                       7
              A1(3)    A2(2) A3(2)
FFI+host                     [~~~~~~~]                                    3-5
Track 2                            [====][====][=======][====]            9
                                    T1(2) T2(2)  T3(3)   T4(2)
```

- **~18–20 weeks** end to end, with the code-side phases compressed against the §7 estimates and
  the device-dependent ones unchanged.
- A1 is the serialization point: nothing else starts until `aarch64-apple-ios` builds.

### 8.5 The text-only milestone ships — what that commits us to

**Decided: the ~week 9–11 text-only build ships independently**, behind
`--no-default-features`, ahead of the image tracks.

This is the right call — it puts real usage against the runtime ~10 weeks before full v1, and T2's
memory tuning stops being guesswork — but it is not free, and the costs land early:

- **The FFI surface becomes a compatibility boundary at week 9**, not at week 18. Once a consumer
  builds against the `.xcframework`, changing its shape is a breaking change. §5.2(a) must be
  designed as a stable API, not as a test harness, and reviewed with the same care as a public
  crate surface (`CONTRIBUTING.md`'s rename rules apply in spirit).
- **The image tracks must not require changing it.** Design the C boundary for `Generator` /
  `on_progress` at the same time as the `TextLlm` one, even though image generation lands nine
  weeks later. Retrofitting a second callback shape into a shipped ABI is the expensive path.
- **Tier 3 CI must exist by week 9**, not by T4 — a shipped artifact needs enforced regression
  baselines. Pull the runner setup (§5.3) forward into A3.
- **The consumer is undecided** — not ChatWorks. The candidates are a test-harness app or a
  headless engine driven remotely. **§5.2.1 recommends the headless server**, which resolves this
  bullet rather than deferring it: an HTTP surface is already a stable contract, so the FFI ABI
  commitment moves to whenever a genuine native consumer is named. The three constraints above
  apply to the *server's* HTTP surface instead, where they are much cheaper to satisfy.

### 8.4 Development environment — verified ready

Checked on the primary development machine, 2026-07-29:

| Component | Status |
|---|---|
| macOS | 26.5.2 (arm64) — **above the 26.2 NAX floor**, so this host can exercise the fast path |
| Xcode | 26.6 (17F113) |
| iOS SDK | 26.5, device and simulator |
| `aarch64-apple-ios` | installed |
| `aarch64-apple-ios-sim` | installed (added for Tier 2) |
| rustc | 1.96.0 — matches `rust-toolchain.toml` |
| cmake | 4.3.4 |

**Nothing needs installing to begin Phase 0.** This machine is also a viable Tier 2/Tier 3
self-hosted runner host, which removes one dependency from §5.3 — it needs only a tethered device.

---

## 9. Definition of done (v1)

1. `textllm_conformance` passes on a physical iPhone with a real Qwen3/Llama snapshot — all eight
   always-on checks, with `check_tools` and `check_thinking` on their **generate** paths and
   `supported_constraints = [Json]` honored.
2. **G6:** `mlx-gen-sana` generates a correct 1024px image on-device within the memory cap, with
   `gen-core-testkit` conformance green.
3. **G7:** `mlx-gen-sensenova` produces both text and image output on-device.
4. `runtime-ios` builds and validates through `runtime-catalog`; its ordered surface test is green
   under both `--no-default-features` and `--features media`.
5. `./scripts/check-workspace.py` passes with the updated member count and no self-fetch violation.
6. `cargo deny --locked check advisories bans licenses sources` clean.
7. `core-llm` and `gen-core` are **unmodified** by this project.
8. A documented threading contract (§4.1) and a hostile-threading test.
9. Published numbers: TTFT, steady tok/s, peak RSS, energy per 100 tokens, image-generation
   latency — with the Phase 0 baselines as regression thresholds.
10. [ios-strategy.md](ios-strategy.md) updated with the Phase 0 outcome.

---

## 10. Decisions and remaining questions

**Resolved 2026-07-29:**

| Question | Answer | Where |
|---|---|---|
| Existing Swift host app? | None — this project builds it | §5.2b |
| Team | One engineer plus Claude; device time is the bottleneck | §8.1–8.2 |
| Development environment | Verified ready, nothing to install | §8.4 |
| Target device | iPhone 17 Pro only for now; broaden later if it gains traction — with a portability guardrail so that stays a tuning job | §0.1 |
| Device CI | Self-hosted runner on the dev machine, one tethered 17 Pro; chosen for Instruments access | §5.3 |
| Text-only milestone | Ships independently at ~wk 9–11 | §8.5 |
| mlx-rs fixes | Fork now, upstream three PRs in parallel | §2.3 |

| Entitlement request | Being requested by the engineer | §8.2 |
| Fork host | `zakkeown/mlx-rs` (pre-existing fork of `michaeltrefry/mlx-rs`; contains the pinned rev), branch `ios-support` | §2.3 |

**Still open — one decision:**

1. **The week 9–11 consumer: headless server or test-harness app?** Not ChatWorks. §5.2.1
   recommends the **headless server** — it already exists (`mlx-llm-server`), takes the FFI layer
   off the critical path for ~2–3 weeks, and defers the ABI commitment to when a real native
   consumer is named. The test-harness app comes nearly free alongside it, since the same shell
   satisfies the `xcodebuild test` host requirement.
