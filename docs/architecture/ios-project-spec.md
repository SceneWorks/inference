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
| **Out of scope** | video, audio, training/LoRA on device, the remaining ~47 media providers |

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

**(2) Possibly not iOS-specific at all — under investigation.**

```
error[E0061]: this function takes 10 arguments but 8 arguments were supplied
  mlx-rs/src/ops/quantization.rs:280   mlx_sys::mlx_qqmm(…)
```

The generated bindings declare `mlx_qqmm` with ten parameters (including `global_scale_x` /
`global_scale_w`); `mlx-rs`'s call site passes eight. **All ten cached host binding sets carry the
identical ten-parameter signature**, so this is not an artefact of the iOS staging. The open
question is whether a *clean* host build hits the same error — i.e. whether the pinned
`pmetal-mlx-rs` rev is currently green only from cache. A clean
`--target aarch64-apple-darwin` build is running to settle it.

- If the clean host build **fails identically**, this is a pre-existing defect in the pinned rev,
  it blocks fresh clones and CI as much as iOS, and it is upstream work — not iOS scope.
- If the clean host build **succeeds**, something in the iOS path regenerates different bindings,
  and it belongs to A1.

**Read-through:** the headline is positive. The iOS blocker list is shorter and more mechanical
than the static read suggested — no missing Metal support, no architectural obstacle, no
unbuildable C++. R1's likelihood drops accordingly.

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

Tier 3 needs a device attached to a runner — self-hosted Mac mini with a tethered phone, or a
cloud device farm. Decide early; it gates the A3 and T4 exits.

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
| R1 | **`mlx-sys` cannot be built for iOS** | L | **Critical** | The whole point of Phase 0. Fallback §3 — escalate, do not absorb. |
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
- **Text-only milestone at ~week 9–11**, shippable independently behind `--no-default-features`.
- A1 is the serialization point: nothing else starts until `aarch64-apple-ios` builds.

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

## 10. Open questions

**Resolved 2026-07-29:** no existing Swift host app — this project builds it (§5.2b). Team is one
engineer plus Claude (§8.1). Development environment verified ready (§8.4).

Still open:

1. **Which iPhone(s) are the test devices?** The per-app memory cap and thermal envelope differ
   enough across generations to change the model-size target. At minimum: the oldest device v1
   must support, plus a current Pro.
2. **Device CI:** this machine can host Tier 2/Tier 3 (§8.4) — is tethering a dedicated device to
   it acceptable, or is a cloud device farm preferred?
3. **Is the text-only milestone at ~week 9–11 worth shipping independently**, or does v1 wait for
   the image tracks?
4. **Has the `com.apple.developer.kernel.increased-memory-limit` entitlement been requested?**
   §8.2 — it gates A2 and T2 and its lead time is Apple's.
