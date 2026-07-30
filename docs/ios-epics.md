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
| **E1** | iOS toolchain | Toolchain / upstream — **retired** | ~~Green `aarch64-apple-ios` CI build, no local env vars~~ **met** | ~~2~~ **done** |
| **E2** | `runtime-ios` composition | Composition — **retired** | ~~`RuntimeCatalog` validates; surface test green~~ **met** | ~~2~~ **done** |
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

**Status: COMPLETE** (S1.1's upstream PR is open but off the critical path — the fork carries it).

`cargo build --locked --target aarch64-apple-ios -p mlx-llm-server` succeeds from a clean clone
with **no environment variables set**, producing a Mach-O arm64 binary (`platform 2`,
`minos 18.0`) whose metallib reports `apple-ios18.0.0` across all 15,660 kernels; the simulator
triple builds; a packaging script places the metallib where the sandbox can find it; and CI
rebuilds and re-asserts all of it on every relevant change. The macOS lane is unaffected
(`minos 26.2`, NAX floor intact).

**Everything here is build-time evidence.** No iOS artifact has executed. The metallib is
correctly *targeted* and correctly *placed*, but whether those kernels are numerically right, and
whether resolution actually succeeds inside a real sandbox, are E3's questions (R9, R11). E1 has
made those questions *answerable*; it has not answered them.

| Story | Notes |
|---|---|
| S1.1 Land the mlx-rs iOS fixes upstream | [SceneWorks/mlx-rs#23](https://github.com/SceneWorks/mlx-rs/pull/23) — **open**, three commits: `qqmm_device` cfg, target-aware clang runtime + cmake cross-compile + cache gating, and `ios-metal-sdk.patch`. |
| S1.2 Home the iOS deployment target in `.cargo/config.toml` | **Done** — `IPHONEOS_DEPLOYMENT_TARGET = "18.0"`, unforced so CI can override. Both halves now covered: the fork's `build.rs` carries it to cmake/Metal, and this entry carries it to rustc's link step (which `env::set_var` cannot reach). Verified with a clean env-free build; macOS `minos 26.2` unchanged. |
| S1.3 Bundle `mlx.metallib` into the `.app` | **Done.** Fork emits `DEP_MLX_METALLIB` (via `links = "mlx"`); `scripts/ios/bundle_metallib.py` copies it next to the executable as an Xcode Run Script phase, with `--expect-platform` refusing a macOS metallib in an iOS bundle and `--codesign-identity` re-signing the copy. **Not yet exercised on device** — that is E3/S3.3. |
| S1.4 Repoint the workspace at the fork | **Done** — pinned at `zakkeown/mlx-rs` (now @ `c0e5c4a4` after S1.3/S1.6). The gate now asserts the **git URL** too (it previously did not, so a same-rev pin from another remote passed silently). Touched four files beyond the manifests: `bump_pins.py` hardcodes the URL and regex-parses gate entries, plus its tests. Revert the URL when #23 merges. |
| S1.5 Tier 1 CI | **Done** — `ios-build` job on `macos-15`, gated by the new `ios_build` lane. Builds both triples and **asserts the artifacts target iOS** (`otool` `platform 2`; metallib must not carry `apple-macos`), then exercises `bundle_metallib.py`. No env overrides — it proves a clean clone builds unaided. |
| S1.6 Simulator target builds | **Done**, and not a formality: the simulator triple did not build at all. mlx-c's example `.app` targets default ON and reference `_MTLIOErrorDomain`, absent from the simulator's Metal framework, failing the whole cmake build. Fixed in the fork with `MLX_C_BUILD_EXAMPLES=OFF`. |

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
| S2.1 `crates/bundles/runtime-ios` | **Done** — `PLATFORM = "ios"`, `BACKEND = "mlx"`, both iOS triples, `catalog()` via `RuntimeCatalog::try_new` over `mlx_llm::text_registry()` + `snapshot_preparer_registry()`. |
| S2.2 Feature profiles | **Done by removal.** No `media` feature: `mlx-gen-catalog` is 32 providers incl. video, wrong for a memory-capped device and unvalidated on iOS. The bundle is LLM-only *by construction*, so a feature gate would be decoration — `--no-default-features` yields the same surface. E5 adds a narrow media root instead. |
| S2.3 Ordered catalog surface test | **Done** — asserts the ordered LLM ids match `runtime-macos` (shared engine; divergence = silent drift) **and** that every media/audio registry is empty, so an incidental `mlx-gen-catalog` dep fails here rather than shipping. |
| S2.4 Repo gates | **Done** — `EXPECTED_MEMBER_COUNT` 90 → 91; `select_lanes.py` routes the bundle to `ios_build` + `release`; lane test pins that it does *not* wake `macos_metal`, `candle_cpu`, or `windows_cuda`. |
| S2.5 Supply chain | **Done** — no new third-party deps (`runtime-catalog` + `mlx-llm`, both already in the graph), so `cargo deny` is unchanged. |
| S2.6 Bundle README | **Done** — including the packaging step, which is not optional: an app without the bundled metallib fails at first Metal use with no build-time warning. |

**Exit: MET.** `RuntimeCatalog` validates the bundle, the ordered surface test is green, and
`check-workspace.py` (91 members), `check_docs.py`, clippy `-D warnings`, `fmt`, and 98 tooling
tests all pass. The bundle cross-compiles for `aarch64-apple-ios`.

**Invariant held:** this epic touched **no contract crate**. `core-llm` and `gen-core` are
unmodified — a property of choosing MLX over CoreML, and a regression if it stops being true.

**Note on S2.2.** The planned `default = ["media"]` profile was dropped, not deferred: on this
platform a media feature would have to gate a registry that does not and should not exist yet.
The surface test now enforces its absence, which is the stronger guarantee.

---

## E3 — On-device proof

**Goal:** the runtime actually executes on a physical iPhone, under test, in CI.

**Why separate:** this is the **first real unknown**. Everything before it is verifiable on a
Mac; nothing here is. It owns metallib resolution inside the app sandbox, model provisioning into
the app container, and the entire device-CI apparatus — which has no precedent in this repo.

**Status: the two hard questions are answered.** On an iPhone 17 Pro Max (iOS 26.5.2),
`scripts/ios/run_smoke.sh` reports:

```
SMOKE: PASS
  [ok] metallib resolves + elementwise kernel -- sum(ones[4,4]) = 16
  [ok] f32 GEMM (steel)  -- sum(64x64 matmul) = 262144 (expected 262144)
  [ok] bf16 GEMM (steel) -- sum(64x64 matmul) = 262144 (expected 262144)
  [ok] softmax reduction kernel -- sum(softmax(ones[4,8])) = 4
```

S3.3 (sandbox metallib resolution) and **R11** (are the cross-compiled kernels numerically
correct, or merely iOS-targeted?) are both closed. Expected values are arithmetic rather than MLX
references, so agreement is evidence, not tautology; bf16 is the one that matters most, given
sc-2772's precedent of 16-bit kernels compiling at the wrong deployment target and emitting
garbage.

**A 4B LLM now generates on the device, through the `runtime-ios` bundle.** Same
`scripts/ios/run_smoke.sh`, with the model pushed into the app container:

```
[ok] runtime-ios generation -- id=mlx-llama tools=true | load 0.1s, first answer 1 tok in 1.7s
     | steady 64 tok in 3.1s (20.4 tok/s) | RSS after load 215 MiB, peak 2903 MiB | "Paris"
```

Read carefully, because two of those numbers are easy to misread:

- **20.4 tok/s steady** on Qwen3-4B Q4. The *first* request reports ~0.6 tok/s, which is not
  throughput: it stops at EOS after one word, and MLX faults weights in lazily so the real load
  cost lands on the first forward pass rather than on `load` (which is why "load 0.1s" looks
  impossibly fast). Correctness and throughput are therefore measured by two separate requests.
- **Peak RSS 2903 MiB against a 2.63 GB model.** Only ~275 MiB over the weights, so there is no
  large transient spike — but this is a *single short* generation. Sustained decode with a growing
  KV cache is E4's question, not answered here.
- `tools=true` confirms the `chat_template.jinja` fix survives the trip to the device.
- The check runs through `runtime_ios::llm::load_for_model`, so it exercises E2's bundle —
  registry, provider selection, capability descriptor — not just the engine.

Correctness is greedy with a fixed seed and asserts the answer contains "Paris", so a wrong result
means wrong kernels rather than unlucky sampling.

**What remains is plumbing, not risk:** getting weights into the app container (S3.4), hosting
`mlx-llm-server` instead of the smoke test (S3.1), running the real conformance suite (S3.5), and
the runner (S3.6/S3.7).

| Story | Notes |
|---|---|
| S3.1 iOS app target (`ios-host/`) | **Scaffolded** — `ios-host/` builds, signs, installs and launches on device via `scripts/ios/run_smoke.sh` (XcodeGen spec, SwiftUI shell, workspace-excluded Rust staticlib). Still to do: host `mlx-llm-server` rather than the smoke test. |
| S3.2 Bind to loopback + USB forwarding | The server has **no auth**. It must not reach a LAN interface. Bearer token if remote access is ever needed. |
| S3.3 Metallib resolution on device | **DONE — verified on an iPhone 17 Pro Max (iOS 26.5.2).** The 124 MB bundled metallib resolves inside the sandbox via `load_colocated_library`. Run it with `scripts/ios/run_smoke.sh`. |
| S3.4 Model provisioning into the app container | **DONE.** `xcrun devicectl device copy to --domain-type appDataContainer --domain-identifier <bundle-id> --source <snapshot> --destination Documents/` pushes the 2.63 GB Q4 snapshot in ~80 s. Needs `UIFileSharingEnabled`. **Note it FLATTENS**: files land in `Documents/`, not `Documents/<dir>/`, so the loader accepts both layouts. |

**Snapshot format constraint — found the hard way, worth knowing before picking a model.**
`mlx-llm` cannot load the common `*-MLX-4bit` community snapshots. Those quantize the **embedding
table** (`model.embed_tokens.weight` as packed `U32` with `scales`/`biases`), while the engine
loads `embed_tokens` densely via `req_bf16` and has no quantized-embedding path — by design, since
its documented quant invariant is that only attention/MLP *projections* are quantized and
embeddings, the LM head, and norms stay dense.

The failure is not obvious from the error: MLX reports
`[rms_norm] (*weight) must have the same size as the last dimension of x but has 2560 elements`,
because the packed `[151936, 320]` table yields a 320-wide embedding where 2560 is expected, and
the first norm downstream is what actually complains. **It fails identically on macOS**, so it is
a snapshot-compatibility issue, not an iOS one.

Use a **dense bf16** snapshot (e.g. `mlx-community/Qwen3-4B-Instruct-2507-bf16`, 7.5 GB) and let
the engine quantize projections on ingest via `write_snapshot` / the `SnapshotPreparer`. For
on-device work that ingest step is required anyway — 7.5 GB dense will not fit the per-app memory
cap, so E3/E4 need a Q4-prepared snapshot, which is exactly what the preparer produces.

**Prepared, and it works.** `cargo run --release -p mlx-llm --example prepare_snapshot -- <src>
<out> q4` turns the 7.50 GiB dense source into **2.64 GiB** (35.2%, 902 tensors, 5.1 s), and the
result generates correctly on macOS — *"The capital of France is Paris, and the capital of Germany
is Berlin…"*, so Q4 preserved the model rather than merely the file format. Staged at
`~/models/ios-eval/Qwen3-4B-Instruct-2507-q4`.

**Separate bug, found here and FIXED — the chat template was silently dropped.**
`load_chat_template` read the model's Jinja template only from `tokenizer_config.json`, falling
back to the typed `Llama3Template` with `supports_tools = supports_thinking = false` when absent.
Newer HF exports — including this Qwen3 — ship the template as a **`chat_template.jinja` sidecar**
instead, which nothing read. The sidecar here renders a `tools` section and `<tool_call>` blocks,
so a tool-capable model reported `tools=false`. Not iOS-specific: it applied on macOS equally, and
put v1's `supports_tools` goal (G2) at risk.

Two halves, both fixed:
- **Discovery** — `JinjaChatTemplate::from_snapshot_dir` (core-llm) reads both conventions, inline
  first. Adopted by `mlx-llm` and by `candle-llm`'s three call sites, which had the same bug.
- **Preservation** — `SnapshotTokenizer` carries `chat_template_jinja` so the preparer copies the
  sidecar through. Note `write_snapshot` has **two** write paths (staged and direct); the first
  patch only fixed one, and the prepared snapshot still reported `tools=false` until both were.

Verified: source and prepared snapshot now both report `tools=true`, with three regression tests
in `core-llm` (sidecar read, inline precedence, error when neither exists).
| S3.5 XCTest target running `textllm_conformance` | All eight always-on checks, on device. **Generation itself is already proven** (below); this is about running the full conformance suite rather than a bespoke check. |
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
