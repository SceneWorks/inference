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
| S4.4 Energy + sustained thermal baselines | **Partly done.** Throughput holds across ~30 s of continuous GPU work — 16.7 → 20.7 → 20.3 → 18.8 tok/s, **no thermal decay** (the first segment is slowest because weights fault in lazily, so >100% retention is expected, not a speed-up). Still open: the Instruments **energy** number and a 5-minute soak, which are the evidence that would reopen the ANE question (strategy §7.2). |
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
| S4.7 Integrate the increased-memory-limit entitlement | Once Apple grants it. Requested separately; lead time is not ours. |

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
| Energy per 100 tok | **not measured** | — |

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
| S5.1 SANA on device | `mlx-gen` + `-pid` + `-sana`. Already builds for iOS; this is the device half. |
| S5.2 Memory residency | Encoder / DiT / DC-AE decoder. 2-bit Gemma-2 encoder if needed; DC-AE tiling. Uses E4's unload seam. |
| S5.3 `gen-core-testkit` conformance on device | The media contract's equivalent of E3's S3.5. |
| S5.5 `media` feature in `runtime-ios` | Plus the ordered surface test for that profile — the bundle's current test asserts the media registry is *empty*, so this is a deliberate edit to both. |
| S5.6 Image-generation latency baselines | Sustained, not cold-start — few-step models only. Enforced like E4's thresholds. |

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
