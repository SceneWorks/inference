# iOS initiative — epic breakdown

**Status: draft for review.** A work breakdown for Shortcut, derived from the
[iOS project spec](architecture/ios-project-spec.md). The decision behind it is in the
[iOS strategy](architecture/ios-strategy.md).

Date: 2026-07-29. Lane: MLX on iOS.

---

## How this is cut

**Five epics**, drawn on **distinct failure modes** rather than on task order. Each owns one
thing that can go wrong, has its own exit criteria, and is independently demoable.

Two constraints set the grain:

1. **The text-only milestone ships at ~week 9–11**, ahead of image generation
   ([spec §8.5](architecture/ios-project-spec.md)). A single epic cannot have two ship dates, so
   at least two are required regardless. E1–E4 map onto that milestone; E5 onto full v1.
2. **This repo's epics are chunky.** 7153 was a whole engine, 3720 a whole contract layer, 13657
   a boundary policy. Ten epics here would produce several sub-week items — story-sized, not
   epic-sized — and would read inconsistently against that history.

| # | Epic | Failure mode it owns | Exit | ~wks |
|---|---|---|---|---|
| **E1** | iOS toolchain | Toolchain / upstream — *largely retired* | Green `aarch64-apple-ios` CI build, no local env vars | ~2 |
| **E2** | `runtime-ios` composition | Composition — low, well-trodden here | `RuntimeCatalog` validates; surface tests green both profiles | ~2 |
| **E3** | On-device proof | **Device runtime** — metallib in sandbox, provisioning | `textllm_conformance` green on a physical iPhone | ~3–4 |
| **E4** | Memory & performance | **Memory, thermals, threading** | G5 numbers published and enforced as thresholds | ~3 |
| **E5** | On-device image generation | **Model portability** | G6 + G7 | ~9 |

```
E1 ──> E2 ──> E3 ──> E4 ──┐
  toolchain  bundle  device  perf │
                                  ├──> ship: text-only (~wk 9–11)
                                  │
                                  └──> E5 ──> ship: full v1 (~wk 18–20)
                                       image
```

E5's *build* half is already de-risked — `mlx-gen`, `-pid` and `-sana` compile for
`aarch64-apple-ios` today ([spec §2.2](architecture/ios-project-spec.md)) — but its device and
memory work depends on E3 and E4.

---

## E1 — iOS toolchain

**Goal:** the workspace builds for iOS reproducibly, in CI, with no local environment fiddling.

**Why separate:** upstream-facing and fork-maintenance work. Different skills and a different
review path from anything else here — and it is the one epic with an external dependency
(upstream review latency) that we do not control.

**Status: S1.1–S1.2 and S1.4 done; S1.3, S1.5, S1.6 remain.** `cargo build --locked --target
aarch64-apple-ios -p mlx-llm-server` succeeds from a clean clone with **no environment variables
set**, producing a Mach-O arm64 binary (`platform 2`, `minos 18.0`) whose metallib reports
`apple-ios18.0.0` across all 15,660 kernels. The macOS lane is unaffected (`minos 26.2`, NAX floor
intact).

| Story | Notes |
|---|---|
| S1.1 Land the mlx-rs iOS fixes upstream | [SceneWorks/mlx-rs#23](https://github.com/SceneWorks/mlx-rs/pull/23) — **open**, three commits: `qqmm_device` cfg, target-aware clang runtime + cmake cross-compile + cache gating, and `ios-metal-sdk.patch`. |
| S1.2 Home the iOS deployment target in `.cargo/config.toml` | **Done** — `IPHONEOS_DEPLOYMENT_TARGET = "18.0"`, unforced so CI can override. Both halves now covered: the fork's `build.rs` carries it to cmake/Metal, and this entry carries it to rustc's link step (which `env::set_var` cannot reach). Verified with a clean env-free build; macOS `minos 26.2` unchanged. |
| S1.3 Bundle `mlx.metallib` into the `.app` | Today it is cached to `~/.cache/pmetal/lib`, meaningless in a sandbox. The `$PMETAL_METALLIB_PATH` / `set_metallib_path()` seam already exists. **The cross-build no longer poisons the macOS cache** (fixed), but bundling itself is outstanding. |
| S1.4 Repoint the workspace at the fork | **Done** — pinned at `zakkeown/mlx-rs` @ `b3c0e27e`. The gate now asserts the **git URL** too (it previously did not, so a same-rev pin from another remote passed silently). Touched four files beyond the manifests: `bump_pins.py` hardcodes the URL and regex-parses gate entries, plus its tests. Revert the URL when #23 merges. |
| S1.5 Tier 1 CI | `cargo build --target aarch64-apple-ios` + `clippy -D warnings` on hosted runners. Build regressions only. |
| S1.6 Simulator target builds | `aarch64-apple-ios-sim`, required by E3's Tier 2. |

**Exit:** CI builds `aarch64-apple-ios` and `-sim` green, from a clean clone, with no environment
variables set at the command line.

**Risk:** upstream review latency on S1.1. Mitigated by carrying the fork (spec §2.3) — S1.1
merging is not on the critical path.

---

## E2 — `runtime-ios` composition

**Goal:** a validated platform bundle that products consume, matching the `runtime-macos` pattern.

**Why separate:** internal architecture work governed by this repo's own invariants
(`RuntimeCatalog`, explicit registries, ordered surface tests). Genuinely low risk — the pattern
is established three times over — but it must be done deliberately, and it is the reviewable
source of truth for what ships.

| Story | Notes |
|---|---|
| S2.1 `crates/bundles/runtime-ios` | `PLATFORM = "ios"`, `BACKEND = "mlx"`, `SUPPORTED_TARGET_TRIPLES = ["aarch64-apple-ios"]`, `NATIVE_PREREQUISITES = ["iOS 18+", "Xcode 16+"]`, `catalog()` via `RuntimeCatalog::try_new`. Registers `mlx_llm::text_registry()` + `snapshot_preparer_registry()`. |
| S2.2 Feature profiles | `default = ["media"]`; `--no-default-features` for the LLM-only profile. |
| S2.3 Ordered catalog surface test | Both profiles. Mirrors the existing bundles'. |
| S2.4 Repo gates | `EXPECTED_MEMBER_COUNT` 90 → 91/92; `select_lanes.py` `ios_device` lane + path rules; verify unclassified paths still fail safe. |
| S2.5 Supply chain | `cargo deny check licenses` for any new deps. |
| S2.6 Bundle README | Matches the other bundles'. |

**Exit:** `RuntimeCatalog` validates the bundle; ordered surface tests green under both profiles;
`check-workspace.py` and `cargo deny` pass.

**Invariant to protect:** this lane touches **no contract crate**. `core-llm` and `gen-core` stay
unmodified — that is a property of choosing MLX over CoreML, and a regression if it stops being
true.

---

## E3 — On-device proof

**Goal:** the runtime actually executes on a physical iPhone, under test, in CI.

**Why separate:** this is the **first real unknown**. Everything before it is verifiable on a
Mac; nothing here is. It owns metallib resolution inside the app sandbox, model provisioning into
the app container, and the entire device-CI apparatus — which has no precedent in this repo.

| Story | Notes |
|---|---|
| S3.1 iOS app target (`ios-host/`) | Thin SwiftUI shell that starts `mlx-llm-server` on a background thread. Also serves as the `xcodebuild test` host, which Tier 3 requires. |
| S3.2 Bind to loopback + USB forwarding | The server has **no auth**. It must not reach a LAN interface. Bearer token if remote access is ever needed. |
| S3.3 Metallib resolution on device | **The first genuine unknown.** Verify the bundled metallib resolves inside the sandbox with no `$HOME` cache available. |
| S3.4 Model provisioning into the app container | How weights reach `WeightsSource::Dir` on a phone — Files app, iTunes file sharing, or a dev-time copy. Unspecified today; needed before anything runs. |
| S3.5 XCTest target running `textllm_conformance` | All eight always-on checks, on device. |
| S3.6 Self-hosted runner + tethered device | Register the dev machine (macOS 26.5.2 / Xcode 26.6 qualifies); dedicate one iPhone 17 Pro. Tier 2 (simulator) nightly, Tier 3 (device) pre-release. |
| S3.7 Runner heartbeat | A sleeping runner must fail loudly, not report green-by-absence. |

**Exit:** `textllm_conformance` passes on a physical iPhone 17 Pro, driven by CI.

**Risks:** metallib sandbox resolution (S3.3); single-device, single-runner point of failure
(accepted — Tier 1 stays hosted so an outage never blocks PRs).

---

## E4 — Memory & performance

**Goal:** good enough to ship, not merely working.

**Why separate from E3:** E3 answers *does it run*; E4 answers *is it shippable*. Different exit
criteria. Folding them invites declaring victory at first token — which is exactly the failure
this split is designed to prevent.

| Story | Notes |
|---|---|
| S4.1 Threading contract + hostile-threading test | `mlx-llm` engines are neither `Send` nor `Sync`, and MLX's Metal device is not thread-safe. On macOS a test detail; on iOS host-app correctness. |
| S4.2 Peak-RSS instrumentation against **two** lines | The ~6 GB cap we ship against, and a ~4 GB reference for the 8 GB device class. The second costs nothing and tells us how far a broader release is (spec §0.1). |
| S4.3 KV cache and buffer sizing under the cap | Sustained decode without jetsam. |
| S4.4 Energy + sustained thermal baselines | Instruments Energy Log; tok/s at t=0 vs t=5min. These are also the evidence that would reopen the ANE question (strategy §7.2). |
| S4.5 Staged load/unload **seam** | Built even though a 17 Pro does not need it, and left disabled. Retrofitting it into a pipeline that assumed co-residency is the expensive version. |
| S4.6 Regression thresholds in Tier 3 | Baselines enforced, not merely recorded. |
| S4.7 Integrate the increased-memory-limit entitlement | Once Apple grants it. Requested separately; lead time is not ours. |

**Exit:** G5 numbers published — TTFT, steady tok/s, peak RSS, energy per 100 tokens — and
enforced as regression thresholds.

---

## E5 — On-device image generation

**Goal:** G6 (small image-only) and G7 (unified AR LLM + image) on device.

**Why separate:** a distinct failure mode — model portability and memory residency — and it is
the only epic gated on a launch requirement that could slip independently of the text runtime.
It is also the largest.

| Story | Notes |
|---|---|
| S5.1 SANA on device | `mlx-gen` + `-pid` + `-sana`. Already **builds** for iOS; this is the device half. |
| S5.2 Memory residency | Encoder / DiT / DC-AE decoder. 2-bit Gemma-2 encoder if needed; DC-AE tiling. Depends on E4's seam. |
| S5.3 `gen-core-testkit` conformance on device | The media contract's equivalent of E3's S3.5. |
| S5.4 sensenova on device | Dual-path AR + flow-matching, sharing `mlx-llm`'s KV cache with the text lane. **The riskiest story in the initiative.** |
| S5.5 `media` feature in `runtime-ios` | Plus the ordered surface test for that profile. |
| S5.6 Image-generation latency baselines | Sustained, not cold-start — few-step models only. |

**Exit:** SANA generates a correct 1024px image within the memory cap; sensenova produces both
text and image output; media registry validated in the bundle.

**Risk:** S5.4. SANA is a known-shape diffusion port; the unified model with its dual-path
runtime is a different animal. See below.

---

## What would change the count

Recorded so a re-cut is a decision rather than drift:

- **E5 splits into two** if sensenova resists. SANA (S5.1–S5.3, S5.5) is well understood; the
  unified AR-plus-image model is not. Plan E5 as one epic and split at S5.4 if it stalls.
- **A sixth epic appears** if the week 9–11 consumer turns out to be a **native app** rather than
  the headless server. The FFI layer, a real host app, and API-stability review then become their
  own workstream. Today [spec §5.2.1](architecture/ios-project-spec.md) folds that into E3 as a
  thin shell — true only while the consumer is HTTP.
- **E1 and E2 merge** if S1.1 merges upstream quickly and the fork disappears, leaving E1 too
  small to stand alone. Do not pre-empt this; it is only worth doing after the fact.
- **E3 and E4 do *not* merge.** Called out explicitly because it is the most tempting
  consolidation and the most costly one.

---

## Sequencing

```
        wk  0    2      4        8        11              20
E1 toolchain [====]
E2 bundle         [====]
E3 device              [======]
E4 perf                       [=====]
                                    ^ ship: text-only
E5 image                             [==================]
                                                        ^ ship: v1
```

~18–20 weeks for one engineer plus Claude. Device time and Apple provisioning are the bottleneck,
not code throughput ([spec §8.2](architecture/ios-project-spec.md)) — E3 and E4 are the
device-bound epics and should be planned around scheduled hardware sessions rather than ad-hoc
interruptions.
