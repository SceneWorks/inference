# Project spec — on-device generative inference for iPhone

**Status: draft for review.** The plan; the options and evidence behind it are in the
[iOS strategy](ios-strategy.md). Nothing here is committed until the Phase 0 gate (§2) reports.

Date: 2026-07-29. Target: `SceneWorks/inference` @ `main`.

---

## 0. Decisions locked

| Decision | Choice |
|---|---|
| **Path** | Phase 0 de-risking spike, then a data-driven gate between Lane A (MLX-on-iOS) and Lane B (CoreML/ANE) |
| **First architecture** | `Architecture::Qwen3` / `Architecture::Llama` — standard full-attention GQA, all stock ops |
| **v1 LLM surface** | text + streaming + cancel, **tool calling**, **thinking/reasoning**, **JSON constraint** |
| **Track 2 (added)** | small image generation + unified AR-LLM-plus-image models |
| **Out of scope** | video, audio, training/LoRA, speculative decode, paged KV, continuous batching |

### What this is structurally

An **additional provider crate plus one bundle, inside this workspace.** Not a fork, not a
parallel library. It composes through the existing registry exactly as `candle-llm` does today.

```
crates/llm/coreml-llm/          (Lane B only)   or   an iOS-capable mlx-llm (Lane A)
crates/bundles/runtime-ios/     (both lanes)
```

> **The image-generation addition materially shifts the recommendation toward Lane A.** See §6.
> Read that before running the Phase 0 gate — it changes what the gate is weighing.

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
- **G6.** One small image generator (`mlx-gen-sana`) generating on-device within the memory cap.
- **G7.** One unified AR-LLM-plus-image provider (`mlx-gen-sensenova`) producing both modalities.

**Non-goals**

- N1. Video (`wan`, `ltx`, `mochi`, `svd`), audio, training/LoRA on device.
- N2. Replacing `mlx-llm` on macOS. `runtime-macos` untouched.
- N3. All ten `Architecture` variants at v1. One family, proven end to end.
- N4. Apple Foundation Models as a registry provider — it cannot load caller-provisioned weights,
  contradicting G4.
- N5. The remaining ~47 media provider crates. Two image providers, deliberately chosen.

---

## 2. Phase 0 — the de-risking spike (3 weeks, hard gate)

Two spikes run **in parallel**. Neither produces shippable code; both produce numbers.

### Spike A — can we build MLX for iOS at all?

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
| Does `cargo build --target aarch64-apple-ios -p mlx-llm` link? | yes/no |
| Does a Qwen3-4B Q4 snapshot load and generate on-device? | yes/no |
| Steady-state decode | tok/s |
| Peak RSS under sustained decode | MB, vs. the per-app cap |
| Energy per 100 tokens | mWh (Instruments Energy Log) |
| Sustained thermals over 5 min | tok/s at t=0 vs t=5min |
| Fork delta size | LoC diff vs upstream `build.rs` |
| **Does `mlx-gen` core + `mlx-gen-sana` also build for iOS?** | yes/no — **cheap to check here, and it is the Track 2 gate** |

That last row costs a day and determines whether §6 is nearly free or a second project.

### Spike B — is the ANE actually worth it?

**Work:** convert **one** Qwen3-4B to `.mlmodelc` via `coremltools` with
`minimum_deployment_target=ct.target.iOS18`, a stateful `slice_update` KV cache as `MLState`, and
4-bit palettization. Drive it from a throwaway Swift harness — **no Rust yet.**

**Critical constraint to validate here:** the converted graph **must emit full float logits**.
Do not fuse argmax or sampling into the graph. `check_seed_determinism`, JSON-constraint masking,
and any future speculative decode all need host access to the distribution. Qwen3's ~151.9k vocab
at fp16 is ~300 KB crossing ANE→CPU *every token* — measure that readback explicitly; it is a
plausible project-killer.

**Exit criteria:**

| Question | Pass bar |
|---|---|
| ANE residency (Instruments Core ML template) | % ops on ANE, not GPU/CPU fallback |
| Logits readback cost per token | ms — isolated |
| TTFT (512-token prompt) | ms |
| Steady-state decode | tok/s |
| Parity vs the MLX reference | max abs logit delta; top-1 agreement over 500 tokens |
| Energy per 100 tokens | mWh — **the number that justifies Lane B at all** |
| Conversion wall-clock + peak host RAM | min / GB |

### The gate

Lane B exists **only** to buy an energy/thermal win; it is otherwise strictly more expensive.
Suggested bar: **≥30% lower energy per 100 tokens, or measurably better sustained tok/s under
thermal load**, at ≤20% worse TTFT.

**Now weighted by Track 2 (§6):** even a Lane B win on the LLM leaves image generation stranded,
because there is no CoreML media backend and `gen-core`'s contract makes one expensive. If G6/G7
are firm product requirements, Lane B must clear a **higher** bar to be worth splitting the stack —
or the project accepts MLX for media and CoreML for text in one bundle, which `runtime-catalog`
does **not** currently permit for the main registries (§5.3).

If ANE residency is low, or logits readback dominates the step, **take Lane A and stop.** That is
a successful Phase 0.

---

## 3. Lane B — `coreml-llm` (12 weeks post-gate)

### 3.1 Crate layout

```
crates/llm/coreml-llm/
  src/lib.rs              register_text_providers / text_registry (mirrors mlx-llm/src/lib.rs:81)
  src/provider.rs         the TextLlm impl                                   ~900
  src/ffi/mod.rs          objc2: MLModel, MLModelConfiguration, MLMultiArray,
                          MLState, MLFeatureProvider, MLComputeUnits        ~1000
  src/decode/mod.rs       prefill + stateful single-token step               ~700
  src/sampler.rs          CPU sampler over the logits buffer                 ~500
  src/config.rs           .mlmodelc metadata + sidecar manifest              ~400
  src/prepare.rs          passthrough SnapshotPreparer (see §5.1)            ~200
  src/error.rs            NSError -> core_llm::Error                         ~150
  tests/conformance.rs    gated real-model conformance                       ~250
```

~4–6k lines. `models/*`, `primitives/{rope,attention}`, `gguf/*` have no counterpart — the
compiled graph absorbs them.

### 3.2 Public surface (mirrors the existing backends exactly)

```rust
pub const PROVIDER_ID: &str = "coreml-qwen3";
pub fn register_text_providers(b: TextLlmRegistryBuilder) -> TextLlmRegistryBuilder;
pub fn text_registry() -> core_llm::Result<core_llm::TextLlmRegistry>;
pub fn register_snapshot_preparers(b: SnapshotPreparerRegistryBuilder) -> SnapshotPreparerRegistryBuilder;
pub fn snapshot_preparer_registry() -> core_llm::Result<core_llm::SnapshotPreparerRegistry>;
```

`TextLlmRegistration` fields:

| Field | Implementation |
|---|---|
| `descriptor` | static, weightless |
| `load` | open `.mlmodelc`, load `tokenizer.json` + `tokenizer_config.json`, build `MLState` |
| `can_load` | **weightless** probe: sidecar manifest names a CoreML graph + supported family. Must not read graph weights. |
| `weightless_vision` | `None` — v1 text-only, no per-snapshot vision distinction |

**A snapshot preparer is mandatory.** `runtime-catalog/src/lib.rs:358` fails validation with
`"runtime has no snapshot preparer"` on an empty registry — verified, not assumed. See §5.1 for
what it must do.

### 3.3 Declared capabilities

```rust
TextLlmCapabilities {
    max_context_tokens: <fixed at conversion time — the graph's shape>,
    max_new_tokens: <cap>,
    supports_system_prompt: true,
    supports_vision: false,       // check_multimodal takes the reject path
    supports_video: false,        // check_video takes the reject path
    supports_thinking: true,
    supports_tools: true,
    supported_constraints: vec![ConstraintKind::Json],
}
```

`max_context_tokens` is **not** a soft limit as it is on MLX — the ANE requires fixed input
shapes, so context length is baked into the converted graph. A longer context is a re-conversion,
not a config change. Document this for consumers.

### 3.4 Reused verbatim from `core-llm` — write none of this

`JinjaChatTemplate` (tools branch + `enable_thinking` kwarg) · `Tokenizer` · `IncrementalDetok` ·
`StopMatcher` · `ThinkingSegmenter` · `ToolCallSegmenter` · `JsonState` + `ConstraintDecodeTable` ·
`TextLlmCapabilities::validate_request`. The three chosen extras (tools, thinking, JSON) are
therefore **wiring, not implementation** — the only genuinely new work among them is masking the
logits buffer each step before sampling.

### 3.5 Conversion pipeline — the real cost centre

Lives **outside the Rust workspace** (§5.1):

```
tools/coreml-convert/          # Python, not a workspace member
  export.py     torch.export -> ct.convert(minimum_deployment_target=iOS18)
  kvcache.py    stateful slice_update KV cache as MLState
  palettize.py  4-bit palettization via coremltools.optimize.coreml
  chunk.py      model chunking for large graphs
  parity.py     logit-delta + top-1 agreement vs the MLX reference
  manifest.py   emit the sidecar manifest can_load probes
```

**4 weeks** for the first family, **2–4 weeks per family thereafter**. This tail does not exist on
MLX, where a new architecture is a Rust file every backend loads.

### 3.6 Phasing

| Phase | Wks | Content | Exit |
|---|---|---|---|
| B1 | 3 | `objc2` FFI + provider skeleton + prefill/step over a hand-converted graph | greedy decode correct on-device |
| B2 | 4 | Conversion pipeline productionized; parity harness; palettization | `parity.py` green on Qwen3-4B and Llama-3-8B |
| B3 | 2 | Seeded CPU sampler; JSON masking; tools + thinking wiring | `textllm_conformance` green on-device, all 8 checks |
| B4 | 3 | `runtime-ios` bundle; CI lane; on-device harness; docs | CI runs conformance on device/simulator |

---

## 4. Lane A — MLX on iOS (7 weeks post-gate)

| Phase | Wks | Content | Exit |
|---|---|---|---|
| A1 | 3 | Productionize the Spike A fork: iOS cmake toolchain, metallib bundling, patch verification, upstreaming attempt | reproducible `aarch64-apple-ios` build in CI |
| A2 | 2 | `runtime-ios` bundle; memory tuning under the per-app cap; **documented threading contract** | Qwen3-4B Q4 sustained decode without jetsam |
| A3 | 2 | CI lane + on-device harness + docs | conformance green on-device |

**What you get:** all ten architectures, vision, GGUF ingest, speculative decode, paged KV —
`mlx-llm` unchanged — **plus a viable path to Track 2 (§6).**
**What you own:** an iOS-capable fork of a fork of `mlx-rs`, permanently. Keep the delta additive
and attempt upstreaming to `pmetal-mlx-rs` in A1; a merged iOS target erases most of that tax.

### 4.1 Threading contract (A2 exit criterion — do not skip)

`mlx-llm`'s own docs state engine instances hold MLX `Array`s and are **neither `Send` nor
`Sync`**, and `.cargo/config.toml` forces `RUST_TEST_THREADS=1` because MLX's shared default Metal
device is not thread-safe (it SIGSEGVs under a parallel harness). On macOS that reads as a test
detail. **On iOS it is a host-app correctness requirement**: a Swift host calling in from a
concurrency context will produce intermittent crashes unless the contract is explicit.

A2 must deliver: one engine per thread (or behind a mutex), a documented rule for which
thread/queue owns the engine, and a stated lifetime for the stream callback. This is the class of
bug that surfaces weeks after the lane is declared done.

---

## 5. Repo-side changes (both lanes)

### 5.1 The `ModelFormat` decision — **resolved, and it needs one contract edit**

Checked rather than deferred, and the answer is more nuanced than "keep it all out of the repo":

1. **A preparer is mandatory.** `runtime-catalog/src/lib.rs:358` rejects an empty preparer registry.
2. **`can_prepare` is a free `fn(&PrepareSpec) -> bool`** — its doc says *"typically a
   `detect_format` call"*, not necessarily. So the probe can sniff a `.mlmodelc` directly and
   never touch the closed enum.
3. **But `PrepareReport.input_format: ModelFormat` is a required field**, and `ModelFormat` is
   closed over `{Gguf, Safetensors}`. A CoreML preparer must return *something*.

**Recommendation: add one additive variant, `ModelFormat::CoreML`.** The alternative is reporting
a `.mlmodelc` as `Safetensors`, which is exactly the kind of silent dishonesty this codebase
rejects everywhere else (`passthrough` is documented as "never silently true" for the same
reason). It is a small, additive, non-breaking change to a tensor-free crate.

**Conversion itself still stays out of the repo.** The preparer is a *passthrough*: validate that
the directory holds a compiled `.mlmodelc` plus a sidecar manifest, report
`passthrough: true`, and write nothing. G4 and the epic-13657 boundary hold. A preparer that
shelled out to Python would put a network-free contract crate in the business of invoking a
toolchain — recommend against.

*(This supersedes [ios-strategy.md](ios-strategy.md)'s §5.1, which assumed the enum could be
avoided entirely.)*

### 5.2 The `backend` tag — mechanically fine, needs a doc note

`runtime-catalog`'s validation is **plain string equality** against the bundle's declared backend
(`lib.rs:313` for media, `lib.rs:350` for LLM) — there is no allowlist. So
`BACKEND = "coreml"` works without a contract change.

But `TextLlmDescriptor.backend` is documented in `capabilities.rs:137` as
`Tensor backend tag ("mlx" | "candle")`, and CoreML is not a tensor backend in that sense — it's
an opaque compiled-graph runtime. **Update that doc comment** so the tag's meaning stays honest.
Cheap, but it belongs in the checklist rather than being discovered in B4.

### 5.3 The mixed-backend problem (Lane B + Track 2 only)

`runtime-catalog` enforces one tensor backend across the media, LLM, and preparer registries. The
**audio lane is the single sanctioned exception** (sc-12901), and it is a deliberate, documented
carve-out with its own backend field and its own preparer registry.

A Lane B bundle that also ships MLX image generation would need a **second** such carve-out. That
is an architecture decision at the level of `audio-backend-strategy.md`, not an implementation
detail — it needs its own ADR and sign-off. **This is a strong argument for Lane A if Track 2 is
firm.**

### 5.4 Checklist

- [ ] `scripts/check-workspace.py`: bump `EXPECTED_MEMBER_COUNT` (currently **90**) → 91 (Lane A)
      or 92 (Lane B: `coreml-llm` + `runtime-ios`)
- [ ] `scripts/check-workspace.py`: confirm the self-fetch lint does not trip on `.mlmodelc`
      handling and that no new env side channel appears
- [ ] `core-llm/src/prepare.rs`: add `ModelFormat::CoreML` (Lane B only) — additive
- [ ] `core-llm/src/capabilities.rs:137`: widen the `backend` doc comment beyond `"mlx" | "candle"`
- [ ] `scripts/ci/select_lanes.py`: add `"ios_device"` to the `LANES` tuple; add path rules for
      `crates/bundles/runtime-ios` (mirroring the `runtime-macos` rule) and `crates/llm/coreml-llm`;
      verify unclassified paths still fail safe to all lanes
- [ ] `crates/bundles/runtime-ios/`: `PLATFORM = "ios"`, `BACKEND = "mlx"` (Lane A) or `"coreml"`
      (Lane B), `SUPPORTED_TARGET_TRIPLES = ["aarch64-apple-ios"]`,
      `NATIVE_PREREQUISITES = ["iOS 18+", "Xcode 16+"]`, `catalog()` via `RuntimeCatalog::try_new`
- [ ] Ordered catalog surface test for the new bundle
- [ ] `deny.toml`: confirm `objc2`/`block2` pass `cargo deny check licenses`
- [ ] `docs/architecture/ios-strategy.md` recording the gate decision and its evidence
- [ ] `.github/workflows/real-weights.yml`: gated on-device conformance job

### 5.5 The Swift/host FFI layer — **unspecced work, scope it explicitly**

There is no host app in this repo and no C ABI surface anywhere. `TextLlm::generate` takes
`&mut dyn FnMut(StreamEvent)` — a Rust closure that **does not cross an FFI boundary**. Something
must exist to bridge it, and it appears in neither lane's phase table.

Scope a small `crates/bundles/runtime-ios-ffi` (or a Swift-bridge crate) covering:

- a C ABI or `swift-bridge`/UniFFI surface for load / generate / cancel
- token-stream delivery: a C callback with a documented calling thread, or a pull-based iterator
- ownership and lifetime rules for the callback and the engine handle (see §4.1)
- error mapping across the boundary
- `.xcframework` packaging

**Budget 2–3 weeks, in both lanes.** It is genuinely shared work — which is what makes host-app
integration parallelizable against a stub, but only *after* this surface is designed. Designing it
is a Phase 0 side-task; implementing it is not free.

### 5.6 CI — the part with no precedent

All three existing bundles are desktop targets that `cargo test` natively. iOS is not. **Budget
separately; this is infrastructure, not a line item.**

- **Tier 1 (every PR, hosted):** `cargo build --target aarch64-apple-ios` + `clippy -D warnings`.
  Build regressions only, no test execution.
- **Tier 2 (nightly, self-hosted Mac):** conformance on the **simulator**
  (`aarch64-apple-ios-sim`). The simulator has no ANE — for Lane B this validates correctness
  only, never performance.
- **Tier 3 (pre-release, self-hosted Mac + tethered iPhone):** `textllm_conformance` on a physical
  device via `xcodebuild test`, plus the Phase 0 numbers as regression baselines.

Tier 3 needs a device attached to a runner — self-hosted Mac mini with a tethered phone, or a
cloud device farm. Decide early; it gates the B4/A3 exits.

### 5.7 Conformance wiring (copy the existing pattern)

```rust
// crates/llm/coreml-llm/tests/conformance.rs
#[test]
#[ignore = "needs a converted CoreML model via COREML_LLM_TEST_MODEL"]
fn real_model_passes_core_llm_conformance() {
    let dir = std::env::var("COREML_LLM_TEST_MODEL").expect("set COREML_LLM_TEST_MODEL");
    textllm_conformance(&|| load(&dir), &TextLlmProfile::cheap());
}
```

Matches `candle-llm/tests/conformance.rs`. Passed-in-path env vars are explicitly allowed by the
self-fetch lint; do **not** derive a cache location.

---

## 6. Track 2 — image generation (the addition)

### 6.1 Both requested capabilities already exist as crates

| Need | Crate | Size | Notes |
|---|---|---|---|
| **Unified AR LLM + image** | `mlx-gen-sensenova` | 9.6k lines | SenseNova-U1 (NEO-Unify) — *"a unified AR LLM + flow-matching image generator"*. Exactly the Gemma-class pattern you described. Under active work (`e13aae06`). |
| **Small image-only** | `mlx-gen-sana` | 6.4k lines | SANA (NVlabs) 0.6B/1.6B + DC-AE deep-compression decoder. The most iPhone-viable generator in the repo. |
| (alternative) | `mlx-gen-z-image` | 15.1k lines | Z-Image-Turbo — few-step, but larger |

Nothing needs to be written. The question is purely whether they **run** on iPhone.

### 6.2 The Gemma connection is real and load-bearing

SANA's text conditioning **is Gemma-2-2B-it**, reused from `mlx-gen-pid`'s `CaptionEncoder`
(epic 8485 / sc-8488 — SANA and PiD share the Gemma-2 last-hidden CHI lineage). So "a Gemma that
generates images" is already the shipping architecture here, not a new direction.

Rough memory budget at 4-bit: Gemma-2-2B encoder ~1.4 GB + SANA DiT 0.6B ~0.35 GB + DC-AE decoder
(small) ≈ **~2 GB**. Comfortable under the per-app cap, with headroom to co-resident an LLM. The
crate docs also reference a **2-bit** SANA drop, which would roughly halve the encoder.

### 6.3 Why this tilts hard toward Lane A

Three structural reasons, each verified:

1. **`mlx-gen-sensenova` depends on `mlx-llm` directly** — its dual-path AR runtime and denoise
   loop consume `ContiguousKvCache`, `sample`, and `Rope` from the LLM engine (sc-7159), rather
   than hand-rolled copies. Those are precisely the primitives Lane B **deletes**. A CoreML LLM
   lane does not merely fail to help the unified model; it removes what the unified model is
   built on.

2. **`gen-core` has 19 public traits to `core-llm`'s one.** More importantly, several are
   *tensor-manipulation* traits — `LatentOps`, `Sampler<L: LatentOps>`, `SamplerPolicy`,
   `GuidanceOps`, `ModelSampling`. Diffusion does real latent arithmetic **between** every model
   call. `TextLlm` gets away with handing back an opaque logits buffer the host samples on CPU;
   a media backend cannot. On CoreML you would either read back full latents every step (slow) or
   compile the sampler into the graph (destroying the pluggable-sampler architecture). This is
   why image generation is **categorically** harder to port than text, not merely bigger.

3. **The provider reuse map is dense.** `mlx-gen-sana` → `mlx-gen-pid` → `mlx-gen` core; `-pid`
   is a near-universal dependency across the family. Porting one provider means porting the core
   and its reuse chain — but on Lane A, all of it comes along once `mlx-sys` builds for iOS.

**On Lane A, Track 2 is largely a build-and-tune exercise.** On Lane B it is a second engine, a
second conversion pipeline, and a `runtime-catalog` carve-out (§5.3).

### 6.4 Phasing (Lane A assumed; on Lane B, treat as a separate project)

| Phase | Wks | Content | Exit |
|---|---|---|---|
| T1 | 2 | Build `mlx-gen` core + `-pid` + `-sana` for `aarch64-apple-ios`; resolve `_vendor`/fixture and Metal-kernel issues | SANA produces a correct image on-device |
| T2 | 2 | Memory tuning: encoder/DiT/decoder residency, staged load-unload, DC-AE tiling; 2-bit encoder if needed | 1024px generation without jetsam, peak RSS recorded |
| T3 | 3 | `mlx-gen-sensenova` on-device: dual-path AR + flow-matching, shared `mlx-llm` KV cache with the text lane | unified text+image generation on-device |
| T4 | 2 | `runtime-ios` `media` feature; ordered catalog surface test; `gen-core-testkit` conformance | media registry validated in the bundle |

**+9 weeks** after Lane A's A3. Bundle it behind a `media` feature exactly as `runtime-macos`
does, so an LLM-only host builds `--no-default-features`.

**Prerequisite:** the extra Spike A row in §2 (does `mlx-gen` build for iOS?). If that comes back
no, T1 grows unpredictably and Track 2 needs re-planning before commitment.

### 6.5 Track 2 risks

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| T-R1 | Peak RSS during generation exceeds the cap (encoder + DiT + decoder co-resident) | H | H | Staged load/unload between stages; 2-bit encoder; DC-AE tiling. Measure in T2 before committing to T3. |
| T-R2 | `mlx-gen` core relies on macOS-only Metal kernel paths / metallib cache (sc-7889) | **L** | H | **Downgraded on evidence:** `grep target_os` across `mlx-gen/src`, `-sana`, `-pid`, `-sensenova` returns **zero** platform gates — no `cfg(target_os)`, no `cfg(unix)`. The media Rust is platform-neutral; the only iOS blocker is `mlx-sys`'s `build.rs`, which is *already* Lane A's A1 work. Track 2's build risk collapses into Spike A's. Residual: the metallib must be bundled into the `.app`, not resolved from `~/.cache/pmetal/lib` — same fix as Lane A. |
| T-R3 | SenseNova's shared-`mlx-llm` coupling breaks if the LLM lane is Lane B | H | H | **This is the §6.3 argument.** Decide Track 2 firmness *before* the Phase 0 gate. |
| T-R4 | Generation latency unacceptable on-device (thermal throttle mid-generation) | M | M | Few-step models only (SANA, Z-Image-Turbo); measure sustained, not cold-start. |

---

## 7. Risks (LLM track)

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | ANE residency low; graph falls back to GPU | M | H | **Exactly what Spike B exists to find.** Gate to Lane A. |
| R2 | Per-token logits readback (~300 KB) dominates the step | M | H | Measured in Spike B. Fallbacks: in-graph top-k (breaks exact seed-determinism — needs a testkit conversation) or ANE prefill + GPU decode. |
| R3 | Fixed-shape graphs make context length a re-conversion | H | M | Accept and document. Ship 2–3 context variants (4k/8k/32k) as separate artifacts. |
| R4 | Per-app memory cap tighter than the reported ~6 GB | M | H | Measure in **both** spikes. Conclusion holds at 4 GB: ~3–4B at 4-bit. |
| R5 | mlx-sys iOS fork drifts from upstream | H | M | Keep the delta additive; attempt upstreaming in A1. |
| R6 | Conversion pipeline rots per model release | H | M | `parity.py` in CI; each family is a versioned owned artifact. |
| R7 | Tier 3 device CI slips | M | M | Ship on Tier 1+2; Tier 3 is a pre-release gate, not per-PR. |
| R8 | ~~`RuntimeCatalog` rejects an empty preparer registry~~ | — | — | **Confirmed true** (`lib.rs:358`). Resolved by the passthrough preparer + `ModelFormat::CoreML` (§5.1). |
| R9 | Host FFI layer underestimated (§5.5) | M | M | Design it during Phase 0; implement in a named phase in both lanes. |
| R10 | Threading contract violated by the Swift host (§4.1) | M | H | Documented ownership rule + a hostile-threading test in A2/B4. |

---

## 8. Timeline and staffing

**One engineer**, iOS/Rust, plus ML-conversion support during Lane B / B2.

```
        wk  0    3         7        10        15         19        24
Phase 0 [======]                                                        3 wks
  GATE        ^
Lane A        [========][====][====][~FFI~]                             10-13 wks
                A1(3)   A2(2) A3(2)  (2-3)
  + Track 2                          [====][====][=======][====]        +9 = 19-22 wks
                                      T1(2) T2(2)  T3(3)   T4(2)
Lane B        [========][==========][====][========][~FFI~]             15-18 wks
                B1(3)     B2(4)     B3(2)  B4(3)     (2-3)
  + Track 2                          -> separate project (§6.3)
```

- **Lane A, LLM only: ~10–13 weeks** (incl. the FFI layer) — full model coverage.
- **Lane A + Track 2: ~19–22 weeks** — text, tools, thinking, JSON, small image gen, and unified
  AR+image.
- **Lane B, LLM only: ~15–18 weeks** — one architecture family, +2–4 weeks each thereafter, and
  Track 2 becomes a separate program.

Host-app integration can begin once §5.5's FFI surface is **designed** (a Phase 0 side-task), not
before.

---

## 9. Definition of done (v1)

1. `textllm_conformance` passes on a physical iPhone with a real Qwen3/Llama snapshot — all eight
   always-on checks, with `check_tools` and `check_thinking` on their **generate** paths (not
   reject paths) and `supported_constraints = [Json]` honored.
2. **G6:** `mlx-gen-sana` generates a correct 1024px image on-device within the memory cap, with
   `gen-core-testkit` conformance green.
3. **G7:** `mlx-gen-sensenova` produces both text and image output on-device.
4. `runtime-ios` builds and validates through `runtime-catalog`; its ordered surface test is green,
   under both `--no-default-features` and the `media` feature.
5. `./scripts/check-workspace.py` passes with the updated member count and no self-fetch violation.
6. `cargo deny --locked check advisories bans licenses sources` clean.
7. A documented threading contract (§4.1) and a hostile-threading test.
8. Published numbers: TTFT, steady tok/s, peak RSS, energy per 100 tokens, image-generation
   latency — with the Phase 0 baselines as regression thresholds.
9. `docs/architecture/ios-strategy.md` records the gate decision and its evidence.

---

## 10. Open questions for you

1. **How firm are G6/G7?** If image generation is a launch requirement rather than a follow-on,
   the Phase 0 gate should arguably be skipped and Lane A taken directly — §6.3 makes Lane B look
   like a poor fit regardless of what Spike B measures.
2. **Is there an existing Swift host app** this plugs into, or is §5.5 building the first one?
3. **Device CI:** self-hosted Mac + tethered iPhone, or a cloud device farm?
