# iOS initiative — epic breakdown

**Status: draft for review.** A work breakdown for Shortcut, derived from the
[iOS project spec](architecture/ios-project-spec.md). The decision behind it is in the
[iOS strategy](architecture/ios-strategy.md).

Date: 2026-07-29. Lane: MLX on iOS.

---

## How this is cut

**Six epics** (originally five — E5 was split at its riskiest story; see below), drawn on
**distinct failure modes** rather than on task order. Each owns one
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
| **E3** | On-device proof | Device runtime — **retired** | ~~`textllm_conformance` green on a physical iPhone~~ **met** | ~~3–4~~ **done** |
| **E4** | Memory & performance | Memory, thermals, threading — **mostly retired** | G5 numbers published + enforced — **all but energy** | ~3 → **1 story left** |
| **E5** | Small image generation (SANA) | Model portability | G6 | ~2 |
| **E6** | Unified AR LLM + image (sensenova) | **Dual-path runtime** — highest remaining risk | G7 | ~3 |

```
E1 ──> E2 ──> E3 ──> E4 ──┬──> ship: text-only            [E1-E3 done, E4 6/7]
  toolchain bundle device perf │
                              ├──> E5 ─> ship: + image     (SANA, ~2 wks)
                              │    sana
                              └──> E6 ─> ship: + unified   (sensenova, ~3 wks)
                                   unified
```

E5 and E6 are independent of each other, not sequential: both need E4's memory seam, neither needs
the other. E5 shipping is a real milestone on its own — which is why the riskiest story now lives
in E6 rather than inside it.

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

**Full conformance now passes on device**, so "conformant on iOS" means exactly what it means on
every other platform:

```
[ok] core-llm conformance suite -- all always-on checks passed in 6.6s
```

**Getting there exposed a real testkit bug, fixed rather than worked around.**
`check_seed_determinism` failed with *"a different seed produced identical output — the provider
appears to ignore the seed"*. It reproduced **identically on macOS**, so it was never an iOS
problem — and the provider was innocent. `TextLlmProfile::cheap()` prompted with `"Hello"`, which
a well-tuned instruct model answers with a near-deterministic canned reply: verified on
Qwen3-4B-Instruct, four different seeds all produced *"Hello! How can I assist you today? 😊"*,
while an open-ended prompt produced four distinct outputs. The fixture had too little entropy to
sample from, so a correct sampler looked broken.

Two existing tests (`mlx-llm/tests/conformance.rs`, `candle-llm/tests/qwen35.rs`) already carried
hand-tuned profiles working around exactly this, with comments describing the same false positive
— so the flaw was known per-test but never fixed at the source. `cheap()` now uses an open-ended
prompt, and the check's error message names the fixture as a candidate cause so the next person
gets a diagnosis instead of a mystery.

**E3 is COMPLETE** — app target and packaging (S3.1/S3.2), sandbox metallib resolution (S3.3),
model provisioning (S3.4), full conformance on device (S3.5), and the runner plus heartbeat
(S3.6/S3.7).

One deliberate deferral: S3.1 hosts the smoke test rather than `mlx-llm-server`. Swapping it is a
small change to the same app target, and the server's value is its HTTP surface — which needs a
named consumer to be worth proving. The runtime underneath is proven either way.

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
| S3.5 `textllm_conformance` on device | **DONE** — the identical `core_llm_testkit` suite the macOS lane runs, all always-on checks green in 6.6 s on an iPhone 17 Pro Max. Runs inside `catch_unwind` (the suite signals failure by panicking, which must not cross FFI) so a failure becomes a report line. |
| S3.6 Self-hosted runner + tethered device | **DONE** — `ios-device` job in `ci.yml`, `runs-on: [self-hosted, macOS, ARM64, ios-device]`. Manual dispatch like `macos-nax`: one runner with one phone must not gate every merge; `ios-build` is the per-PR guard. Uploads the device report as an artifact. Setup in [guide/ios-device-runner.md](guide/ios-device-runner.md). |
| S3.7 Runner heartbeat | **DONE** — `ios-device-heartbeat.yml`, every 6 h. A lane that never runs looks exactly like a lane that always passes, so this turns that silence into a failure. Checks only what the device lane needs (device paired, **unlocked**, Developer Mode on, signing identity **and** Xcode account) in seconds, no build. |

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
| S4.1 Threading contract | **DONE** — and better than planned: the **type system already enforces it**. `Box<dyn TextLlm>` is not `Send`, so cross-thread use does not compile; a test in `runtime-ios` pins that (with a Send control, so a false negative is caught). A hostile-threading *runtime* test is therefore unnecessary for Rust callers. The Swift side is a C ABI where marker traits are invisible, so the convention is documented in the bundle README instead. |
| S4.2 Peak-RSS instrumentation | **DONE** — `getrusage(RUSAGE_SELF)` peak RSS reported by every on-device run. Measured on a 2.64 GiB Q4 Qwen3-4B: **215 MiB after load, 2903 MiB peak, ~2980 MiB under sustained work**. Against the ~6 GB cap that is roughly half; against the ~4 GB line of an 8 GB device it is comfortable but not spacious. |
| S4.3 Sustained decode without jetsam | **DONE for repeated generations** — 512 tok over 4 segments, **RSS growth 0 MiB** (2980 → 2980), no jetsam. Note the scope: `generate` allocates a fresh KV cache per call, so this proves no leak across calls, *not* KV growth within one long context. That needs a prefix-cached or multi-turn path and remains open. |
| S4.4 Energy + sustained thermal baselines | **Thermal DONE, energy partial.** A **5-minute soak — 4992 tokens — shows no throttling**: 16.6 → 16.2 → 15.9 → 16.5 → 16.8 tok/s per minute, retention 101%, peak RSS 2973 MiB (`scripts/ios/soak.sh`). Energy: Instruments' *Energy Log* template is **GUI-only**, so the script captures the headless **Power Profiler** instrument to a `.trace` instead. That gives CPU/GPU power counters and thermal state, **not** the mWh-per-100-tokens figure this story names — open the trace in Instruments for that. |
**Measured on an iPhone 17 Pro Max (iOS 26.5.2), Qwen3-4B Q4:**

```
[ok] sustained decode -- 512 tok over 4 segments
     128tok@16.7t/s/2980MiB  128tok@20.7t/s  128tok@20.3t/s  128tok@18.8t/s
     retention 112% | RSS 2980 -> 2980 MiB (growth 0, baseline 2963)
```

Two readings worth stating carefully, because both invite over-claiming:

- **"Growth 0" is real but narrower than it sounds.** `generate` allocates a fresh KV cache per
  call, so this shows the runtime does not leak across repeated generations. A single long
  context growing its cache is a *different* measurement and is not covered.
- **"Retention 112%" is not a speed-up.** The first segment is slowest because MLX faults weights
  in lazily, so the real load cost lands on the first forward pass. The honest reading is
  *throughput is flat*, with no thermal decay over ~30 s.

The thresholds derived from these (S4.6) assert against the **last** segment, not the first: the
first is depressed by lazy weight faulting, so using it as the throughput figure would mask a real
slowdown. The RSS ceiling is set at 4 GiB — the cap of an *8 GB* device, not this 12 GB one — so
the lane fails **before** a broader-device release would, rather than after
([spec §0.1](architecture/ios-project-spec.md)).

| S4.5 Staged load/unload **seam** | **DONE and measured** — `idle 0 → loaded 2693 → dropped 0 → cleared 0 MiB, 100% reclaimed`. Built while the 17 Pro Max does not need it, because retrofitting it into a pipeline that assumed co-residency is the expensive version. **Finding: `drop` alone returns everything** — the buffer cache is not holding weights, so `clear_cache` is a guard, not the mechanism. Measured via MLX's `get_active_memory`, not RSS: `ru_maxrss` is a high-water mark that never falls, so it would have reported "nothing freed". |
| S4.6 Regression thresholds | **DONE** — `run_smoke.sh` asserts throughput ≥ 12 tok/s, peak RSS ≤ 4096 MiB, and RSS growth ≤ 256 MiB, overridable by env var. Deliberately loose: these catch a lost fast path, a leak, or thermal collapse — not a warm phone. A check that fails on a slow afternoon teaches people to ignore it. Verified by a negative test (`THRESHOLD_MIN_TPS=999` → exit 1, naming the metric). |
| S4.7 Integrate the increased-memory-limit entitlement | **DONE — and it never needed Apple.** `com.apple.developer.kernel.increased-memory-limit` is a **self-serve** capability: automatic signing regenerates the profile to include it, unlike the approval-gated ones (CarPlay, HLS, …). This story was written assuming a request-and-wait, and that assumption sat unexamined long enough to shape E5's device planning. Claimed in `ios-host/App/SceneWorksSmoke.entitlements` alongside `extended-virtual-addressing` — needed independently, since MLX memory-maps weights and SANA's snapshot is 4.73 GB, which can exhaust a 4 GB address space even when residency is fine. |

**Exit: substantially met.** Published and enforced: steady tok/s, peak RSS, sustained-decode
memory growth, and the unload seam — all asserted by `run_smoke.sh`, all verified to fail when
violated. **Not met: energy per 100 tokens** (S4.4), which needs an Instruments Energy Log capture
and a 5-minute soak. That is the one G5 number still missing, and it is also the evidence that
would reopen the ANE question ([strategy §7.2](architecture/ios-strategy.md)) — so it should not
be quietly dropped.

**Baselines** (iPhone 17 Pro Max / iOS 26.5.2, Qwen3-4B Q4, 2.64 GiB snapshot):

| Metric | Measured | Threshold |
|---|---|---|
| Steady throughput | ~20.6 tok/s (short), ~18.3 tok/s (last of 4 segments) | ≥ 12 tok/s |
| Peak RSS | 2892–2903 MiB | ≤ 4096 MiB |
| RSS growth over 512 tok | 0 MiB | ≤ 256 MiB |
| Unload reclaim | 100% (2693 MiB) | ≥ 90% |
| Load → first token | ~1.7 s | not asserted |
| Sustained (5 min) | 4992 tok, 16.6 → 16.8 tok/s, **no throttling** | — |
| Energy per 100 tok | **not measured** — needs Instruments GUI | — |

The RSS ceiling is deliberately the ~4 GB cap of an *8 GB* device rather than this one's ~6 GB, so
the lane fails before a broader-device release would rather than after.

---

## Working without a dedicated device

The iPhone is a daily driver, not lab hardware. That is a real constraint, and it shapes the
remaining work more than the estimates do.

**Most of what is left does not need the phone.** Sorting the remaining questions by where they
can actually be answered:

| Question | Mac? | Why |
|---|---|---|
| Does it build for iOS? | **yes** | CI cross-compiles both triples already |
| Is the composition right? | **yes** | Surface tests are target-independent |
| Are the Metal kernels correct? | **yes** | Same metallib; E3 proved the iOS build matches |
| Does the model generate correctly? | **yes** | Same code, same weights |
| **Does it fit the memory cap?** | **mostly** | `examples/memory_budget` — see below |
| **Does it thermally throttle?** | no | Passive cooling in a phone chassis |
| **Energy per 100 tokens** | no | Instruments, on device |

`cargo run --release -p mlx-llm --example memory_budget -- <snapshot> [--budget-mib N]` runs a
model under a simulated iOS per-app cap **on macOS**, reporting MLX's active/peak/cache footprint
and a fits / tight / over-budget verdict. Validated against the device: it reports 2719 MiB peak
where the iPhone measured 2693 MiB, ~1% apart. Allocator behaviour carries over because it is the
same code, weights, and Metal allocator.

It does **not** simulate jetsam (`set_memory_limit` is backpressure — MLX blocks rather than
failing) or the host app's own footprint. A pass is necessary, not sufficient. But it moves the
*search* for a memory configuration off the phone, leaving the device to *confirm* one number.

**Two batched device sessions, not continuous access:**

| Session | When | What | ~Time |
|---|---|---|---|
| **A** | after E5's Mac-side work | SANA generation, memory at the cap, latency baselines, **plus the Instruments energy capture that closes E4/S4.4** | ~2 h |
| **B** | after E6's Mac-side work | Two models under one cap — the one thing the Mac cannot simulate faithfully | ~2 h |

Between sessions the phone is free. During one: **plugged in, auto-lock off**. Three runs died on
auto-lock during E3/E4 bring-up, each costing a full rebuild cycle — that is the difference
between a 2-hour session and a 4-hour one.

---

## E5 — Small image generation (SANA)

**Goal:** G6 — a small image-only generator on device.

**Why separate:** a distinct failure mode — model portability and memory residency — and the only
epic gated on a launch requirement that could slip independently of the text runtime.

**Estimate: ~2 weeks.** This was planned at 9 weeks for E5-as-one-epic, before any of the runtime
worked. Six of that estimate's assumptions have since been measured and retired:

| Assumed at planning time | Now known |
|---|---|
| `mlx-gen` might not cross-compile | Zero `cfg(target_os)` gates; same `mlx-sys` build already fixed in E1 |
| Metal kernels unproven on device | Proven — f32/bf16 GEMM and softmax correct (E3) |
| Memory headroom unknown | ~2.9 GiB of ~6 GB used by the LLM; SANA + Gemma-2 is ~2 GB |
| No way to make room for a second model | Unload reclaims 100% (E4/S4.5) |
| No packaging or provisioning path | metallib bundling + `devicectl` push, both working |
| No device harness | `run_smoke.sh` builds, signs, installs, launches, asserts thresholds |

What is left is genuinely SANA-specific: getting it resident and correct, and measuring latency.

| Story | Notes |
|---|---|
| S5.1 SANA on device | **DONE.** A 1024px image generated on an iPhone 17 Pro Max in 36.5 s at 2839 MiB, through `provider_registry()` and the `Generator` contract. See the device section below. |
| S5.2 Memory residency | **DONE on a 12 GB device; 8 GB is now doubtful.** Confirmed on hardware at 2839 MiB against a measured 6136 MiB cap. But the host's 4773 MiB untiled configuration was jetsam-killed, so host numbers understate device demand and the earlier "fits 8 GB at 80–85%" claim cannot be trusted without 8 GB hardware. Mac-side detail follows. **Mac-side: fits 8 GB, tightly.** `Residency::run_staged` sheds the Q4 trunk before decode (−1905 MiB) and the now-working tiled decode bounds the DC-AE transient: 1024px at 3294 MiB, 512px at 3453, both inside a 4096 MiB budget at 80–85% (56–78% of a 12 GB device's). Peak verified count-independent; Resident/Sequential byte-parity and repeat-job bounding re-verified on real weights. Device confirmation is Session A, and at 8 GB it is the *deciding* measurement, not a formality. |
| S5.3 `gen-core-testkit` conformance on device | The media contract's equivalent of E3's S3.5. |
| S5.7 Device harness for image generation | **DONE.** `ios-smoke` gains a `media` feature carrying a SANA check (two configurations, loaded through `provider_registry()` rather than a direct loader), `scripts/ios/push_model.sh` provisions a snapshot into the app container, and `run_smoke.sh --media` drives both. Validated on the host first, where MLX peaks reproduce `image_budget`'s numbers exactly. |
| S5.5 `media` feature in `runtime-ios` | **DONE** — `mlx-gen-ios-catalog` (a new, narrow composition root: SANA only, **not** `mlx-gen-catalog`'s 32 providers) behind an off-by-default `media` feature. Both profiles have ordered surface tests; the media one asserts exactly `["sana_1600m", "sana_sprint_1600m"]` and the LLM-only one asserts the registry is empty. Cross-compiles for `aarch64-apple-ios`. |
| S5.6 Image-generation latency baselines | Sustained, not cold-start — few-step models only. Enforced like E4's thresholds. |

### Measured: SANA does not fit an 8 GB device (2026-07-30)

`cargo run --release -p mlx-gen-ios-catalog --example image_budget -- <q4-snapshot>` against a
4096 MiB budget, on macOS:

| Resolution | Resident | Sequential | Verdict |
|---|---|---|---|
| 1024px | 10553 MiB (258%) | 8340 MiB (204%) | over |
| 512px | 6199 MiB (151%) | 5065 MiB (124%) | over |
| 256px | 5131 MiB (125%) | **4363 MiB (107%)** | over |

**Resolution is not the lever.** Dropping from 1024px to 256px — a 16× reduction in pixels —
saves only ~4 GB of a ~8.3 GB peak, because most of the footprint is *weights*, not activations.
Even at 256px, sequential residency lands 7% over the cap. The floor is the Q4 tier itself:
Gemma-2 encoder 2.3 GB + DiT 2.0 GB + DC-AE 1.25 GB.

Two things this corrects:

- **The "~2 GB" figure quoted earlier in this document (and in `mlx-gen-ios-catalog`'s docs) was
  wrong** — it came from crate prose, not measurement. The real Q4 tier is ~5.6 GB on disk and
  peaks at 8.3 GB working set.
- **`OffloadPolicy::Sequential` is not just a memory win, it is also FASTER** (3.4 s vs 5.2 s at
  1024px). That is counter-intuitive — the expectation was that reloading components costs time —
  and worth understanding before relying on it. Likely less allocator pressure and better cache
  behaviour, but that is a hypothesis, not a measurement.

### Scoping the fix (2026-07-30)

**Correction first.** I earlier said 512px would fit a 17 Pro Max's ~6 GB cap sequentially. That
was wrong: I compared a number measured under a **4 GB** budget (5065 MiB) against a 6 GB cap. The
budget changes the measurement — MLX's backpressure limit alters allocator behaviour — so re-run
at 6144 MiB, 512px is **6678 MiB (109%)**, still over. Never compare across budgets.

Measured against a 12 GB device (~6144 MiB cap):

| Resolution | Resident | Sequential | Verdict |
|---|---|---|---|
| 1024px | — | 8340 MiB (136%) | over |
| 768px | 8287 MiB (135%) | 8146 MiB (133%) | over |
| 512px | 6899 MiB (112%) | 6678 MiB (109%) | over |
| **256px** | 6320 MiB (103%) | **5049 MiB (82%)** | **fits, tight** |

So there *is* a shipping configuration today — 256px sequential on a 12 GB device — but it is one
device class at a thumbnail resolution, and only 18% clear of the cap.

**The 2-bit quant is the wrong lever.** `mlx-gen-sana`'s docs mention an unported 2-bit Clark Labs
quant, and I assumed that was the fix. Reading the code: that quant applies to the **transformer
trunk**, and the trunk is already 2.0 GB in Q4.

**The real lever is the text encoder's embedding table**, found by breaking the encoder's 2.32 GB
down by tensor:

| Component | Size | Quantized? |
|---|---|---|
| `embed_tokens` | **1.18 GB** | **no — dense BF16** |
| packed projections | 1.01 GB | yes (Q4) |
| scales/biases | 0.13 GB | — |

`gemma2.rs:65` states it outright: *"`embed_tokens` is NOT routed here — it stays dense in every
tier."* That is a reasonable default for a decoder, where the embedding is also the LM head. But
**this is a caption encoder** — it takes last-hidden states, never produces logits
(`gemma2.rs:3`), and touches the table exactly once via `take_axis`, a pure gather of ≤300 rows
from 256,000 (`gemma2.rs:298`).

Quantizing it costs one gather-then-dequantize of the ~300 selected rows — MLX already exposes
`dequantize` — and saves:

| Embedding tier | Table | Encoder total |
|---|---|---|
| BF16 (today) | 1.18 GB | 2.32 GB |
| Q8 | 0.63 GB | 1.77 GB |
| Q4 | 0.33 GB | **1.47 GB** |

**~0.85 GB off the encoder**, which is the phase that dominates the low-resolution floor.

**Scope:** small and well-bounded — quantize `embed_tokens` when writing the tier, and gather-then
-dequantize in `Gemma2::forward`. It touches `mlx-gen-pid`'s Gemma-2 (shared with SANA), so PiD's
dense path must stay bit-identical, and it needs a new tier on the HF snapshot. Estimate **2–4
days**, against an unknown-size trunk port.

**What it does not do:** ~0.85 GB does not bring 1024px (8.3 GB) under a 6 GB cap. It would move
512px from 6678 → ~5.8 GB, i.e. 512px on a 12 GB device — a real improvement over 256px, and
still not an 8 GB device.

### Implemented the embedding fix — it works, and it barely helps (2026-07-30)

Quantizing `embed_tokens` to Q4 does exactly what the arithmetic said on disk: **1.18 GB → 0.33 GB**,
and `mlx-gen-pid`'s Gemma-2 now packed-detects it (`mlx_gen::quant::embedding`, the same shared
`TokenEmbedding` FLUX.2 and Z-Image already use — worth noting I started writing a duplicate before
finding it). All 40 PiD/SANA parity tests pass, so the dense path is unchanged.

**The peak barely moved:**

| Config (6144 MiB budget) | Before | After | Saved |
|---|---|---|---|
| 512px sequential | 6678 MiB | 6679 MiB | **0** |
| 256px sequential | 5049 MiB | 4890 MiB | 159 MiB |

Predicted ~850 MiB. Got 0–159. **The disk saving is real; the peak saving is not**, because peak is
set by whichever phase allocates most, and the encoder is *already dropped* before denoise under
`Sequential`. Shrinking a component that was not the peak does not lower the peak — I reasoned
about total weight size when I should have reasoned about per-phase maxima, and the measurement
caught it.

**Where the peak actually is.** `load_components` builds **both** DC-AE halves:

```rust
let encoder = DcAeEncoder::from_weights(&vae_w, dcfg.clone())?;   // 0.61 GB
let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone())?;   // 0.64 GB
```

Text-to-image never encodes an image — the `DcAeEncoder` is dead weight in the heavy bundle, held
through the phase that *is* the peak. That is ~0.61 GB against a 512px gap of ~535 MiB, i.e. it
alone could be the difference.

**Revised order of levers**, now that the phase structure is measured rather than assumed:

1. **Skip `DcAeEncoder` for text-to-image** (~0.61 GB off the peak phase). Small, targeted, and it
   removes weight that is never used rather than making used weight smaller.
2. **DC-AE decoder tiling** (~0.64 GB, the other half of the peak phase).
3. **Trunk quantization** — 2.0 GB, the largest single item in the heavy bundle, but the 2-bit
   Clark-Labs port is genuinely unported work of unknown size.
4. ~~Embedding quantization~~ — **done, keep it** (it is free and real on disk), but it is not the
   lever.

### Skipped the DC-AE encoder too — and found why none of it works (2026-07-30)

Implemented the lever I recommended: `SanaHeavy::encoder` is now `Option`, `load_heavy` builds the
`DcAeEncoder` only when the request carries a `Conditioning::Reference`, and the seam's spare
`use_pid` flag carries that decision (the same mechanism F-177 uses to skip a wasted PiD load).
All 46 SANA/PiD tests pass, img2img included.

**Peak at 512px: 6679 MiB. Identical to before.** As with the embedding, the weights went away and
the peak did not.

Measuring resident peak across resolutions with both fixes applied explains why:

| Resolution | Peak | Weights in the render phase |
|---|---|---|
| 1024px | 10257 MiB | ~2.6 GB |
| 512px | 7042 MiB | ~2.6 GB |
| 256px | 6239 MiB | ~2.6 GB |

**~3.6 GB of the peak is not weights at all**, and it is present even at 256px where activations
should be negligible. Weight reduction was never going to close a gap that weights do not
account for.

That is the real finding, and it retires the whole line of attack:

- Embedding quantization: −0.85 GB of weights → **0 MiB** off peak.
- DC-AE encoder skip: −0.61 GB of weights → **0 MiB** off peak.
- Both are correct, both are worth keeping (smaller downloads, less I/O), and **neither is the
  memory story**.

**What the ~3.6 GB actually is remains unidentified.** Candidates, in the order worth testing:
MLX's buffer cache retaining freed blocks (`set_cache_limit(0)` would show this immediately), the
DC-AE decoder's intermediate feature maps at f32c32 (its latent is 32-channel, so decode
activations are large even from a small latent), or a one-off allocator reservation. **Measure
before optimizing** — that is the lesson from two failed predictions in a row.

**Recommendation: stop optimizing SANA and re-scope.** Two failed levers with sound reasoning
behind each is evidence the model is simply too large for the target, not that the next lever will
land. `256px sequential @ 4890 MiB (80% of a 12 GB device's cap)` is the shipping configuration
that exists today. Anything better needs either the unidentified 3.6 GB explained, or a smaller
model — and the second is a product conversation, not an engineering one.

### Localized the memory: per-phase tracing (2026-07-30)

Stopped guessing and instrumented `image_budget` to sample MLX's allocator at every `Progress`
callback, plus a per-stage trace inside the DC-AE decoder. Two prior levers failed because they
targeted weights; the trace says weights were never the problem.

**Buffer cache: ruled out.** `--no-cache` (`set_cache_limit(0)`) gives 6678 MiB vs 6679. The peak
is live allocation, not retained-free blocks.

**Sequential @ 512px, phase by phase:**

```
Loading(Renderer)     active     3 MiB   peak  2064 MiB
Step (denoise)        active  1905 MiB   peak  3048 MiB
Decoding (start)      active  2045 MiB   peak  3048 MiB
final                 active     0 MiB   peak  6678 MiB    <- +3630 after the last callback
```

Denoise never exceeds ~3 GB. **The entire overage happens in the VAE decode**, after the last
progress event — which is why every earlier measurement saw one opaque number.

**Inside the decoder** (`MLX_GEN_DCAE_TRACE=1`, resident, 512px):

| Stage output | Size | Active | Peak |
|---|---|---|---|
| `[1,16,16,1024]` | 1 MiB | 3582 | 4519 |
| `[1,32,32,1024]` | 4 MiB | 3827 | 4537 |
| `[1,64,64,512]` | 8 MiB | 3841 | 5277 |
| `[1,128,128,512]` | 32 MiB | 3992 | 6116 |
| `[1,256,256,256]` | 64 MiB | 4042 | 6519 |
| `[1,512,512,128]` | 128 MiB | 4175 | 6865 |

That separates into **two independent problems**, which is why single levers kept failing:

1. **~3.5 GB already resident before the decoder runs one stage.** The first stage's output is
   1 MiB, yet active is 3582 MiB. This is denoise-phase memory that was never released — under
   `Resident` the trunk is still held, and the latents/graph from denoise are still alive.
2. **+2.3 GB of transient decode work** (peak 4519 → 6865 across stages), against stage outputs
   totalling only ~237 MiB. Each stage's *internal* convolutions allocate far more than the tensor
   they hand on. Stage-wise `eval` does **not** fix this — tested, 6680 MiB — because the
   transients are inside a stage, not between stages.

**Leverage, now that both parts are separated:**

- Problem 1 is the promising one: ~3.5 GB of *denoise-phase* memory held during decode. If the
  trunk and denoise graph were released before decode — the same load→use→drop the text encoder
  already gets — decode would start near zero rather than near 3.5 GB. That is a third residency
  phase, and the seam (`Residency`) already has the shape for it.
- Problem 2 needs DC-AE tiling: decode the latent in spatial tiles so a stage's internals are
  bounded by tile size rather than full resolution. Real work, and standard practice for VAEs on
  constrained memory.

**Neither is a weight problem, which is why −1.46 GB of weight savings moved the peak 0 MiB.**

### Released the denoise graph — problem 1 solved, problem 2 is the wall (2026-07-30)

Added `release_denoise_graph`: evaluate the latents and `clear_cache()` at the denoise→decode
boundary, so the denoise graph (and through it the trunk) is dropped before the DC-AE decode
allocates. Applied at both render tails (base and Sprint). 36 SANA tests pass.

**It does what it was meant to.** Decode's starting `active` at 512px sequential:

| | Before | After |
|---|---|---|
| active entering decode stage 1 | 3582 MiB | **2173 MiB** |

~1.4 GB released, confirming the diagnosis: MLX's laziness was keeping the entire denoise history
live through decode, because an un-evaluated array references the graph that produced it.

**The reported peak did not move: 6678 MiB**, verified in isolation with `--sequential-only` (the
two runs share a process and MLX's peak is a high-water mark, so a resident pass measured first
was masking the sequential number — worth knowing for any future comparison).

The stage trace says why. Post-release, sequential:

```
[1,16,16,1024]   active 2173   peak 3048
[1,32,32,1024]   active 2422   peak 3132
...
final                          peak 6678
```

The last two stages — `[1,256,256,256]` and `[1,512,512,128]` — add **~3.5 GB of transients**
while their outputs are 64 MiB and 128 MiB. At the final stage that is roughly **18× the tensor it
produces**. This is problem 2 from the previous entry, and it is now the entire remaining gap.

**Why it is a wall for the current approach:** the transients are *inside* one stage's
convolutions at full spatial resolution. Nothing outside the decoder can release them, which is
why stage-wise `eval` (6680 MiB) and every weight reduction (0 MiB) failed. The only lever that
reaches inside is **spatial tiling** — decode the latent in tiles so a stage's internals are
bounded by tile size rather than image size. That is standard practice for VAEs under memory
pressure and is real, non-trivial work: overlapping tiles, seam blending, and a parity test
proving tiled output matches whole-image output.

**Keep `release_denoise_graph` regardless.** It does not lower this peak, but it lowers the floor
decode starts from, which is what makes tiling viable rather than merely helpful — a tiled decode
starting at 2.2 GB has room to work in; one starting at 3.6 GB does not.

### Merged main — the memory work already exists, done properly (2026-07-30)

Pulled 181 commits from `main`. A large `memory-strategy` epic landed there while this branch was
attacking the same problem by hand, and it supersedes most of what I was about to build.

`gen_core::memory_strategy` is a tensor-neutral memory-planning contract with a five-rung ladder:

```
Resident → StagedResidency → BoundedDecode → BoundedAttention → BoundedTransformerResidency
```

Two of those rungs are precisely the levers this branch derived from measurement:

- **`StagedResidency`** — "requires Conditioning, Denoise, and Decode hooks with synchronized phase
  release". That is `release_denoise_graph` (the 1.4 GB floor drop), generalized into a lifecycle.
- **`BoundedDecode`** — "decode owns tile edge and overlap". That is the tiling I had started,
  with a calibrated budget model and a conformance suite around it.

**`mlx-gen-z-image` is the reference adopter** (`memory_strategy.rs`, 1866 lines): capability
declaration, parameter ranges per rung, evidence eligibility, and request-scoped lifecycle.
**SANA has not adopted it** — which is why our number is unchanged at 6678 MiB post-merge.

**This changes the plan for E5.** The hand-rolled tiling in `pipeline.rs` should be replaced by a
proper `BoundedDecode` adoption following z-image, not finished. Reasons, in order:

1. The contract carries a *budget model* — it tiles only when the predicted peak exceeds the
   budget, so tiling costs nothing when memory is plentiful. My env-var knob has no such gate.
2. Selection is worker-owned and evidence-gated: an optimized rung is eligible only when
   conformance is Verified and the calibration fingerprint matches. A hand-rolled tile size has no
   such guardrail and would silently drift from the numbers it was tuned against.
3. `gen_core_testkit::memory_strategy_conformance` exists. A bespoke path is untested by it.

**What this branch keeps regardless**, because it is measurement the contract does not supply:
the per-phase and per-stage tracing in `image_budget`, the finding that the peak is decode
transients rather than weights, and `release_denoise_graph` (which is the StagedResidency Decode
hook in all but name — worth porting into the adoption rather than deleting).

**Note the merge is not a free win.** The epic gives SANA the *machinery* to fix this, not the fix:
adopting it is real work.

*(Scoping correction, same day: the 1866-line figure overstates SANA's share. 874 of those lines are
pre-test code and much of that is module doc; SANA needs a strict subset of it — no PiD decode
routes, no rung 3, no rung 4 — declaring rungs 0/1/2 Implemented and 3/4 Missing. And the scoping
pass turned up a larger, cheaper lever first: see below.)*

### SANA fits — the lever was a seam we had never called (2026-07-30)

Scoping the `BoundedDecode` adoption turned up something better first. SANA drove the shared
residency seam through **`Residency::run`**, which holds the entire heavy bundle across denoise *and*
decode. The same seam also offers **`run_staged`**, which frees the DiT between the two phases:

```
run:         encode → drop encoder → [ denoise → decode ]     ← trunk resident through decode
run_staged:  encode → drop encoder →  denoise → drop DiT → decode
```

All four `z_image` variants use `run_staged`. SANA never adopted it. Its trunk is ~2.0 GB in Q4 and
was live underneath the ~3.5 GB decode transient — which is the whole story of the 6678 MiB peak.

`run_staged`'s `materialize_mid` hook is *literally* `release_denoise_graph`: this branch rebuilt one
piece of a seam whose other half was already there.

**Both levers, measured** (Q4 + Q4 embedding, sequential, 4 steps, count 1):

| | before | after `run_staged` | + tiling | vs 4 GB cap |
|---|---:|---:|---:|---|
| 512px | 6678 MiB | **4773 MiB** | **3453 MiB** (256px tiles) | 84% |
| 1024px | — | 9177 MiB | **3465 MiB** (256px tiles) | 85% |
| 1024px | — | — | **3294 MiB** (128px tiles) | 80% |

The DiT drop is worth **1905 MiB**, almost exactly the trunk. `Decoding` now begins at **0 MiB
active** where it began at 2173.

**Read those as TIGHT, not comfortable.** 80–85% of an 8 GB device's cap is what the harness itself
tags `TIGHT`, and `set_memory_limit` is backpressure while jetsam is a kill — so fitting the budget
proves the working set fits, not that iOS lets the app live. On a 12 GB device (~6 GB cap) the same
configurations sit at 56–78%, which is the comfortable case.

**Peak is essentially count-independent, which had to be checked rather than assumed.** The staged
change reordered the count loop to denoise every seed before anything decodes, so phase C now runs N
decodes back-to-back in one scope where the old code interleaved them — and a decode transient is
the largest allocation in the request. Measured with `--count`:

| config | count 1 | count 3 |
|---|---:|---:|
| 1024px, 128px tiles | 3294 | 3295 |
| 512px, 256px tiles | 3453 | 3538 |
| 512px, untiled | 4773 | 5129 |

The tiled paths are flat because `tiled_decode`'s per-tile `eval` bounds the graph; untiled 512px
grows +356 MiB across three images and still fits. Nothing stacks.

**1024px at 128px tiles lands on 3294 MiB, which is the denoise peak.** SANA is now denoise-bound,
not decode-bound: further decode tiling buys nothing. This inverts the §"two independent problems"
diagnosis above — that was correct *before* the trunk was shed.

**The tiled decode was broken and had never been run.** `vae_tiling::tiled_decode` is 5-D and slices
the latent and shapes the decoded tile through one `[t, h, w]` axis triple, but SANA's latent is NCHW
while DC-AE emits NHWC. Bridged through a channels-last NTHWC lift.

**Tiling is not parity, and cannot be.** DC-AE's decoder is `EfficientViTBlock` =
`SanaMultiscaleLinearAttention → GLUMBConv`, whose `1/(Σ + eps)` normalizer sums over every spatial
position it is given. A tile sees only its own, so tiled output is a *different render* by
construction. The sweep shows exactly that signature — doubling overlap at 512px moved the mean
2.41 → 1.89 while *halving* the tile made it worse (4.29), the opposite of how a boundary artifact
behaves. Overlap cannot repair a global operation.

**The renders were looked at, not just measured.** At every tile size the image is seam-free and
equally valid; the tone shifts smoothly because the trapezoidal blend spreads the per-tile
difference out. Max |Δ| reaches 226/255 on a perfectly good render, which is why
`decode_tiling_parity` prints it and does not assert it. What that test *does* gate is the layout
bridge, where an error is silent: one tile covering the whole image must reproduce the whole-image
decode exactly, and does (max |Δ| = 0).

**§0.1's guardrail survives after all.** Every configuration above fits a 4096 MiB budget — an 8 GB
device — so SANA does not need to be restricted to 12 GB hardware. The earlier "12 GB only"
recommendation is withdrawn; it was written against a peak that assumed the trunk stayed resident.
The 8 GB case is tight enough that the device session decides it, not the host harness.

> **Superseded the same day by the device run below — read that first.** The device session did
> decide it, and not in this claim's favour. A configuration this section calls comfortable
> (512px untiled, 4773 MiB against a 6136 MiB cap) was **jetsam-killed** on hardware, because
> `set_memory_limit` applies backpressure where the device applies a kill. Host numbers therefore
> *understate* device demand, and every "fits 8 GB at N%" figure above is an underestimate of
> unknown size. The 8 GB claim is not withdrawn — it is **unverified**, and cannot be verified
> without 8 GB hardware. What *is* confirmed on a 12 GB device is 1024px tiled at 2839 MiB.

**Found in passing — a progress-contract violation shared by every mlx-gen image provider.**
`gen_core_testkit`'s progress contract requires `Progress::Decoding` **exactly once** per generation
and names once-per-output as a failure ("the restarting-bar class", F-136/F-162). SANA emitted it
inside the per-image loop, so it was wrong for any `count > 1`. The testkit never caught it because
its request defaults to `count: 1`, where the two are indistinguishable.

Fixed in SANA, where the staged reordering makes once-per-batch the natural shape — all denoise, then
all decode, so a single `Decoding` marks the phase transition exactly.

**`mlx-gen-z-image`'s `decode_batch` has the identical bug** and is not fixed here: it is a shipping
provider on a different epic's critical path, and the change belongs with a testkit case that would
actually catch it (a `count > 1` progress assertion) rather than as a drive-by on an iOS branch. The
same `count > 1` blind spot also hides a **`Step` restart** — both providers run a fresh `1..=steps`
counter per image where the contract wants a folded `total = N × steps` bar. That one is a real
change to shared denoise-batch structure and is not attempted here.

### Device harness for image generation — three bugs the scripts had been hiding (2026-07-30)

Building the device path for SANA turned up three defects, none of which could be found by reading
the scripts. All were found by running them.

**1. `run_smoke.sh` picked the device by column position.** `awk '{print $(NF-3)}'` counts back from
the end of the `devicectl list devices` line, which lands inside the *Model* column the moment that
column has a different word count. "iPhone 17 Pro Max (iPhone18,2)" gives `NF=11`, and `$(NF-3)` is
the literal string `17` — after which every devicectl call fails with *"The specified device was not
found. (Name: 17)"*. It had never fired because every previous device run passed `--device`
explicitly. Now matched by UUID shape.

**2. The first push script reported success after failing.** Its copy loop ended in `|| true` and its
verification could not distinguish "the listing is empty" from "the listing command failed". Three
copies errored with the device-not-found above and it still exited 0. Copies now fail on the spot,
and the verification demands devicectl's `N files:` header before believing an empty result.

**3. `devicectl copy to` copies a directory's CONTENTS, not the directory.** Confirmed against the
device: pushing `inner/` (holding `probe.txt`) to `Documents/dirprobe` yields
`Documents/dirprobe/probe.txt`, not `Documents/dirprobe/inner/probe.txt`. That is not a cosmetic
difference here — SANA's `transformer/` and `vae/` each hold a file named
`diffusion_pytorch_model.safetensors`, so pushing both to one destination silently **overwrote the
1.99 GB trunk with the 1.25 GB decoder**. The push exits 0 and leaves a corrupt snapshot that fails
at load with an error pointing at the code. Each component now names itself in the destination, and
verification descends into every component comparing names *and* sizes — a top-level listing showed
the right names while the bytes underneath were wrong.

**A related trap avoided rather than hit:** `run_smoke.sh`'s threshold extraction scanned the whole
report and took the last match, so it silently depended on which checks existed and in what order.
The image check's detail carries `MLX peak N MiB` *and* `process RSS peak N MiB`, which would have
repointed the LLM's RSS ceiling at SANA's number. Extraction is now anchored to the named check.

**Jetsam-proofing.** Written when the app had no `increased-memory-limit` entitlement (since
claimed — see S4.7, it was self-serve all along). The mitigations stay: an entitlement raises the
cap, it does not remove it, and a kill is still a kill. So
SANA's 4773 MiB configuration may be killed outright — and a jetsam kill takes the whole report with
it, including the LLM checks that already passed, leaving "no report was produced", which is
indistinguishable from a launch failure. Two mitigations: configurations run in **ascending measured
peak** (1024-tiled at 3294 before 512-untiled at 4773), so a kill still leaves the cheaper one
proven; and each completed configuration appends a breadcrumb to `Documents/sana-progress.txt`,
which `run_smoke.sh` pulls when no report appears. The configuration *after* the last breadcrumb is
the one that exceeded the cap.

**Also note the RSS/MLX divergence.** On the host, `getrusage`'s peak RSS came back *below* MLX's own
peak (2961 vs 4773 MiB) — it is not seeing Metal buffer allocations. On iOS those do count toward the
footprint jetsam reads, so the divergence should not appear on device. The check prints both so that
can be checked rather than assumed, and the image threshold is read off MLX's number, since a ceiling
on RSS would be vacuous.

### On device: SANA generates 1024px on an iPhone, and tiling is why (2026-07-30)

First image-generation run on the iPhone 17 Pro Max, iOS 26.5.2, with the memory entitlements
claimed.

| config | host (`image_budget`) | device | outcome |
|---|---:|---:|---|
| 1024px, 128px tiles | 3294 MiB | **2839 MiB**, 36.5 s | **works**, 4263 MiB still available |
| 512px, untiled | 4773 MiB | — | **process died mid-run** |

**The per-app cap is 6136 MiB (5.99 GiB)**, read from `os_proc_available_memory()` rather than
assumed. The branch's "~6 GB on a 12 GB device" folklore was right; it is now a measurement that
prints on every run.

**The host said untiled fits. The device killed it.** The reason is precise and worth keeping:
`image_budget` measures under `set_memory_limit`, which is *backpressure* — MLX evicts rather than
grows. The device has no limit, it has jetsam, which kills. An untiled decode is therefore free to
allocate past whatever the host recorded. That is exactly the gap between "the working set fits" and
"iOS lets the app live" that `image_budget`'s own module docs warn about, now measured instead of
warned about.

**So tiling is not an optimization for large images — it is what makes SANA viable on a phone.**
That inverts what this doc said a few hours earlier, when the 512-untiled configuration was reported
as the comfortable one. The device harness now runs only tiled configurations.

**What is established, and what is inferred.** Established: `1024 tile128` completed and left a
breadcrumb, the configuration after it did not, `devicectl info processes` showed the app gone, and
no report was written. So the process died during the untiled decode. **Inferred, not confirmed:**
that the killer was jetsam specifically. An MLX allocation abort would look identical from here —
both are process death with no Rust panic, since `ios_smoke_run` catches unwinds and neither a
`SIGKILL` nor a C++ `abort` unwinds. Telling them apart needs the device's JetsamEvent report,
which lives behind Xcode's Devices GUI or a `sysdiagnose`, and is worth pulling before this is
written up as a memory-limit result rather than an "untiled decode does not survive" result.

The practical conclusion does not depend on which it was: the untiled path does not work on this
device, and the tiled path does.

Ordering configurations by ascending measured peak meant the survivable one was proven before the
fatal one ran — had they been ordered by resolution, the run would have died first and taught
nothing.

**MLX's accounting and process RSS agree on device** (2839 vs 2955 MiB) where they diverged sharply
on macOS (4773 vs 2961). So `getrusage` missing Metal allocations is a host artifact, and iOS does
count them toward the footprint jetsam reads — which is what makes the cap number meaningful.

**Still open:** the `BoundedDecode` contract adoption. The mechanism now exists and is measured, but
tiling is still selected by the `MLX_GEN_SANA_DECODE_TILE` env knob rather than by the contract's
budget model. That work is now an optimization rather than the critical path, and it should be sized
*after* on-device confirmation — the calibration fingerprint must not be minted before execution
structure settles, and this change moved it.

**Superseded by the above**, kept because the reasoning is still sound where it applies:

1. ~~**A smaller text encoder.**~~ At 2.3 GB the Gemma-2 CHI encoder is the largest single component,
   but it is dropped after the encode phase and the peak now lives elsewhere. Still the right lever
   if the *conditioning* phase ever sets the peak; it does not today.
2. ~~**DC-AE tiling**~~ — done, above.
3. ~~**A 12 GB device only.**~~ Withdrawn: 8 GB fits.

**Exit:** SANA generates a correct 1024px image within the memory cap, `gen-core-testkit`
conformance green on device, media registry validated in the bundle.

**Risk:** low-moderate. SANA is a known-shape diffusion port on a build that already works. The
open question is memory residency under the cap (S5.2), which E4's seam exists to answer.

---

## E6 — Unified AR LLM + image (sensenova)

**Goal:** G7 — `mlx-gen-sensenova` producing both text and image output on device.

**Why separate from E5:** this was S5.4, and the epic doc already flagged splitting here if it
resisted. Splitting it *before* starting is the better call: SANA shipping is a real milestone
that should not be held hostage to the riskiest story in the initiative, and the two have
different shapes. SANA is a diffusion port of known shape; sensenova is a dual-path AR +
flow-matching runtime that shares `mlx-llm`'s `ContiguousKvCache`, `sample`, and `Rope`
(sc-7159) — the coupling that made Lane A the right choice in the first place
([strategy §6.3](architecture/ios-strategy.md)).

**Estimate: ~3 weeks, low confidence.** Unlike E5's stories, nothing here has been de-risked by
the work so far. That coupling to `mlx-llm` is a benefit on this lane, but it also means the
unified model exercises paths the text lane does not.

| Story | Notes |
|---|---|
| S6.1 sensenova on device | Dual-path AR + flow-matching; shares the KV cache with the text lane. |
| S6.2 Co-residency or staged handoff | Two models under one cap. Where E4/S4.5's seam earns its keep — or where it turns out not to be enough. |
| S6.3 Unified conformance + latency | Both modalities, sustained. |

**Exit:** sensenova produces both text and image output on device, within the memory cap.

**Risk:** the highest remaining in the initiative. If it stalls, E5 has already shipped G6 and the
text lane is unaffected — which is the point of the split.

---

## What would change the count

Recorded so a re-cut is a decision rather than drift:

- ~~**E5 splits into two** if sensenova resists.~~ **Done, and pre-emptively rather than
  reactively** (2026-07-29): E5 is now SANA-only and E6 is sensenova. Waiting for it to stall
  would have meant discovering the split mid-epic, with G6 already entangled in G7's risk.
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
                     DONE ──────────────────┐   remaining
E1 toolchain  [done]                        │
E2 bundle     [done]                        │
E3 device     [done]                        │
E4 perf       [6/7 — energy open]           │
                                            ├─> ship: text-only  (ready now)
E5 sana                                     [====]        ~2 wks
E6 sensenova                                [======]      ~3 wks
```

**E1–E3 are complete and E4 is six of seven** — the text lane is shippable today. What remains is
~5 weeks: SANA (~2) and sensenova (~3), plus E4's energy measurement.

The original plan said ~18–20 weeks. That estimate was written before anything ran; most of it was
risk that has since been measured away rather than work that has been done faster. The remaining
5 weeks are the parts nothing so far has de-risked.

Device time and Apple provisioning remain the bottleneck, not code throughput
([spec §8.2](architecture/ios-project-spec.md)) — plan E5/E6 around scheduled hardware sessions.
