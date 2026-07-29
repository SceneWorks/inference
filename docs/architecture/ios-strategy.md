# iOS strategy — CoreML vs MLX, feasibility and LoE

**Status: proposed, not ratified.** This records the options, the evidence behind each, and a
recommendation. The decision itself is gated on the Phase 0 spike defined in the
[iOS project spec](ios-project-spec.md); this document must be updated with the gate outcome
before any lane is committed to.

Date: 2026-07-29. Basis: read of `core-llm`, `mlx-llm`, `candle-llm`, the bundles, the
conformance testkit, and the pinned `mlx-sys` fork's `build.rs`.

> **Superseded in part.** §5.1's "keep conversion out of the repo, no contract edit needed"
> conclusion was refined after further reading — a snapshot preparer is mandatory
> (`runtime-catalog/src/lib.rs:358`) and `PrepareReport.input_format` requires a `ModelFormat`
> value, so Lane B needs one additive `ModelFormat::CoreML` variant. See the project spec's §5.1
> for the resolved position. This document is otherwise current.

---

## 0. The fork in the question

"CoreML-compatible" collapses two different asks, and they differ by roughly an order of
magnitude in cost:

| Ask | Door | Cost |
|---|---|---|
| **Run on iPhone at all** | MLX already runs on iOS (Metal). Reuse `mlx-llm` nearly whole. | weeks |
| **Run on the Neural Engine** (battery, thermals, leave the GPU free) | CoreML is the only door. New backend + an ahead-of-time conversion pipeline. | months |

Both are answered below. The good news is that they are **not** exclusive — the composition
architecture makes them two providers behind one contract, so Path A can ship first and Path B
land later without rework of anything above the engine.

**Not a total reimplementation, and not a fork.** In both cases this is an **additional provider
crate inside this same workspace** — a `coreml-llm` (or iOS-capable `mlx-llm`) engine plus a
`runtime-ios` bundle, composed through the existing registry exactly as `candle-llm` and
`mlx-llm` are today. No parallel library, no divergent history, one lockfile. The contract
boundary is thin and genuinely tensor-free, which is the thing that usually kills ports like
this. It holds here.

---

## 1. What the architecture buys you

`core_llm::TextLlm` is **three methods**:

```rust
fn descriptor(&self) -> &TextLlmDescriptor;
fn validate(&self, req: &TextLlmRequest) -> Result<()>;
fn generate(&self, req: &TextLlmRequest, on_event: &mut dyn FnMut(StreamEvent)) -> Result<TextLlmOutput>;
```

That is the entire surface a CoreML engine must satisfy. `candle-llm` already proves "add a
second backend behind this contract" is a repeated, solved exercise here.

### Reused verbatim — ~6.6k lines, zero port cost

All of `crates/contracts/core-llm/` is tensor-free by design and comes along free:

- `template.rs` (1058) — Jinja chat templates
- `constraint.rs` (712) — the `JsonState` incremental-validity machine
- `tool.rs` (516) — tool-call segmentation/parsing
- `prepare.rs` (477), `schedule.rs` (339), `prefix.rs` (315), `thinking.rs` (312),
  `message.rs` (302), `speculative.rs` (283), `stop.rs` (250), `paging.rs` (208),
  `detok.rs` (205), `tokenizer.rs` (200), `capabilities.rs`, `registry.rs`
- The `core-llm-testkit` conformance suite — your definition of done, already written

Tokenization, chat templating, stop matching, incremental detokenization, thinking/tool
segmentation, capability gating, scheduling and prefix policy: **none of it is rewritten.**

### Reimplemented — new code, but small

These are MLX-`Array`-typed and do not cross the seam:

- `decode/` (~1.7k) — the streaming loop. Note `Decode::step(&self, input_ids: &Array,
  cache: &mut dyn KvCache, offset) -> Result<Array>` is *shaped* like a stateful CoreML
  prediction call, which is encouraging, but it is an mlx-llm-internal trait taking MLX arrays.
- `primitives/sampler.rs` (455) — becomes plain CPU work over a logits buffer. Easy, but new.
- `primitives/kv_cache.rs` + `paged_kv_cache.rs` (~1.2k) — on CoreML a KV cache is an `MLState`
  handle, not owned tensors. Much of the paging sophistication becomes unavailable, not ported.

### Deleted — but the work *moves*, it doesn't vanish

The compiled `.mlpackage` absorbs these:

- `models/*` (~5k: qwen35, llama, qwen35_vision, siglip, deepstack)
- `primitives/{rope, attention, gated_delta}` (~1.9k)
- `gguf/{dequant, iq_grids, convert}` (~2.7k)

**This is the single biggest thing a line-count estimate gets wrong.** That ~9.6k lines of Rust
doesn't disappear — it reappears as a `coremltools` conversion pipeline, per architecture
family, in Python, which this repo has **no precedent for** (there is no Python beyond the
`scripts/` gates, and the whole design assumes weights arrive as caller-provisioned local
paths). Every new model family needs a conversion recipe, a numerical-parity check against the
reference implementation, and a re-export whenever the architecture changes.

---

## 2. Path A — MLX on iOS

MLX itself supports iOS; `mlx-swift` runs LLMs on iPhone today. `mlx-llm` is Rust over
`mlx-rs`, so in principle the engine compiles for `aarch64-apple-ios` unchanged.

**The blocker is the build, and I checked it rather than guessing.** The pinned fork
(`pmetal-mlx-sys` @ `932beb4`) has no iOS story at all:

- `build.rs` is `#[cfg(target_os = "macos")]`-gated in three places, including the entire
  deployment-target setup.
- It drives MLX's C++ via `cmake::Config` with **no iOS toolchain arguments** — no
  `CMAKE_SYSTEM_NAME=iOS`, no `IPHONEOS_DEPLOYMENT_TARGET`.
- It caches `mlx.metallib` into `~/.cache/pmetal/lib/`. **That is meaningless in an iOS app
  sandbox** — the metallib has to be bundled into the `.app`.
- It applies **three `required = true` patches** to MLX core; all three would need to keep
  applying on an iOS-capable base.
- `.cargo/config.toml`'s `MACOSX_DEPLOYMENT_TARGET = "26.2"` NAX-kernel setup is macOS-specific
  and does not carry over.

So Path A is not "flip a target triple". It is **fork-the-fork build engineering** with a
permanent maintenance tax — you'd own iOS support in a fork of a fork of mlx-rs.

**Effort: ~4–8 weeks** for one engineer to a running iPhone build, dominated by build/toolchain
work, plus ongoing fork upkeep.

**What you get:** the full existing engine — every model family, vision, GGUF ingest,
speculative decode, the lot. **What you don't:** the ANE. MLX is Metal-only, so you're on the
GPU, which on a phone means worse battery and thermals than the ANE path, and you're competing
with the UI for the GPU.

**Hard ceiling regardless of path:** iPhone enforces a flat per-app memory budget that does not
scale with installed RAM. Reported as **~6 GB** even on a 12 GB iPhone 17 Pro Max and even with
the increased-memory-limit entitlement — *treat that specific figure as single-source and
measure it yourself before planning against it* (see Sources). The conclusion is insensitive to
the exact number: at anywhere from 4 to 6 GB the practical target is a **~3–4B model at 4-bit**,
with real care over KV cache and buffer sizing (one report recovered headroom only by cutting a
GPU buffer cache from 512 MB to 64 MB).

---

## 3. Path B — a `coreml-llm` engine crate

The architecturally clean answer, and the only one that reaches the ANE.

**New Rust, roughly 4–6k lines** (against `mlx-llm`'s 22k, because the model layer is gone):

| Piece | Est. |
|---|---|
| `provider.rs` — the `TextLlm` impl | 800–1200 |
| CoreML FFI (`MLModel`, `MLMultiArray`, `MLState`, compute-unit selection) via `objc2` | 800–1200 |
| decode loop — prefill + stateful step | 600–900 |
| CPU sampler over the logits buffer (temp/top-p/top-k, seeded, constraint masking) | 400–600 |
| config/load/metadata, error mapping, tests | 900–1400 |

**Plus the conversion pipeline, which is where the time actually goes.** Per architecture
family: `torch.export` → `coremltools` with `minimum_deployment_target=iOS18`, a stateful
`slice_update` KV cache, 4-bit palettization, and model chunking for anything large. Apple's own
Llama 3.1 CoreML work is the right comparison for how non-trivial this is.

**Effort: ~8–14 weeks** for one engineer to a first shipping text-only provider covering **one**
architecture family, including conversion, the iOS bundle, and CI. Then **+2–4 weeks per
additional family**. That per-family tail is the ongoing cost you're signing up for, and it's the
main structural difference from today, where a new family is a Rust file that any backend loads.

**The technical catch worth knowing up front:** the ANE is a fixed-shape accelerator designed for
convolutions. Autoregressive decode — memory-bandwidth-bound, variable sequence length — is the
one workload it handles badly, and CoreML adds meaningful per-op overhead on small operations.
The emerging state of the art is **disaggregated**: ANE for prefill (compute-bound, big fixed
matmuls) and GPU for decode. Worth noting that the registry/composition model accommodates
exactly that — it's two providers, or one provider holding both handles.

---

## 4. Path C — Apple Foundation Models — cheap, but wrong shape

Superficially the cheapest option: iOS 26 ships a ~3B on-device model with guided generation and
tool calling, and the adapter is maybe 500–800 lines.

**The disqualifier is the weights premise, not conformance.** An FM provider cannot load
caller-provisioned weights at all — there is one Apple-supplied model and no `WeightsSource::Dir`
to point at. That contradicts the premise of the entire epic-13657 self-fetch design, in which
inference receives every model component as a local path the consumer provisioned. It also means
no model choice, no quantization control, and no vision/video path of your choosing.

*(Correction to an earlier draft of this memo: I asserted FM would fail
`check_seed_determinism`. It would not. `GenerationOptions` exposes `sampling: .greedy` for
deterministic output and `SamplingMode.random(top:seed:)` for seeded sampling, which satisfies
both legs of the check — same seed ⇒ same output, different seed ⇒ different output. The
conformance objection doesn't hold; the weights objection is independent and stronger.)*

Recommend against as a main path. If wanted, scope it deliberately as a convenience provider
outside the model-loading registry, where the `WeightsSource` contract doesn't apply.

---

## 5. Repo-side costs common to A and B

Cheap to fix, easy to discover late:

1. **`ModelFormat` is a closed enum** — `{Gguf, Safetensors}`, with `detect_format` switching on
   bytes/layout (`core-llm/src/prepare.rs`). A `.mlpackage`/`.mlmodelc` source means editing a
   *tensor-free contract crate*, not just adding a backend. Decide deliberately: does conversion
   live **outside** the repo (a pre-compiled `.mlmodelc` arriving via `WeightsSource::Dir`, which
   is consistent with the self-fetch boundary — my recommendation), or does it become a third
   `SnapshotPreparer`?
2. **A `runtime-ios` bundle** — bump `EXPECTED_MEMBER_COUNT` (currently 90), add an ordered
   catalog surface test, add an iOS lane to `scripts/ci/select_lanes.py`.
3. **CI has no precedent for this.** All three existing bundles are desktop targets that
   `cargo test` natively. On-device/simulator testing is new infrastructure, not a line item —
   budget it separately.

---

## 6. Media and audio — explicitly out of scope

Not because `mlx-gen` is 304k lines and `candle-gen` is 238k, but for a structural reason: the
ratio is **worse** than those numbers suggest, not proportional to them. `sceneworks-gen-core`
has a far wider contract surface than `TextLlm`'s three methods, and a large multi-provider
diffusion graph gets **none** of the "compiled graph replaces the model code" saving that makes
the LLM port cheap — every provider needs its own conversion, its own scheduler, its own
parity check. Treat image/video/audio on iPhone as a separate program, not a phase of this one.

---

## 7. Recommendation

1. **Decide the driver first** — "on iPhone" or "on the ANE". Everything downstream turns on it.
2. If the answer is *ship something on iPhone soon*: **Path A**, eyes open about owning an
   iOS-capable mlx-sys fork.
3. If the answer is *battery/thermals matter, this is a product surface*: **Path B**, budget the
   conversion pipeline as its own workstream with its own owner, and keep conversion out of this
   repo behind `WeightsSource::Dir`.
4. **Path A and Path B are not mutually exclusive** — the contract makes them two providers. A
   plausible endgame is MLX for decode and CoreML/ANE for prefill in one bundle.
5. **Path C only** as a deliberate non-conformant extra, never as the main path.

**Bottom line:** a text-LLM CoreML variant is a real but bounded project — **one engineer,
roughly one quarter**, for a first shipping architecture family — and it is *not* a
reimplementation and *not* a fork. The tensor-free contract layer is the reason, and it is the
part of this codebase that most obviously pays off here.

---

## Sources

Repo-local findings (contract shape, line counts, `mlx-sys` `build.rs` gating, conformance
check set, `EXPECTED_MEMBER_COUNT`) were read directly from this workspace and the pinned
`~/.cargo/git/checkouts/mlx-rs-.../932beb4/mlx-sys` checkout. External claims:

- CoreML stateful models / `MLState` KV cache, `minimum_deployment_target=iOS18`, the ~1.6×
  Mistral-7B speedup from stateful vs. non-stateful KV cache —
  [coremltools Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html),
  [WWDC24: Deploy ML and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/)
- ANE fixed-shape constraint, low-bit palettization vs. block-wise quantization, CoreML per-op
  overhead — [Apple ML Research: On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama),
  [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)
- Disaggregated NPU-prefill / GPU-decode direction —
  [SqueezeBits: Disaggregated Inference on Apple Silicon](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- **iPhone ~6 GB per-app ceiling (single-source — verify before planning against it)** and the
  GPU-buffer-cache/jetsam anecdote —
  [How Fast Are On-Device LLMs on iPhone 17 Pro and iPad Pro?](https://rickytakkar.com/blog_russet_mlx_benchmark.html)
- MLX-on-iOS precedent —
  [Running an LLM on iPhone with MLX Swift (awni)](https://gist.github.com/awni/fe4f96c21ead68e60191190cbc1c129b),
  [WWDC25: Explore LLMs on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- Foundation Models `GenerationOptions` sampling/seed —
  [WWDC25: Deep dive into the Foundation Models framework](https://developer.apple.com/videos/play/wwdc2025/301/),
  [Exploring the Foundation Models framework](https://www.createwithswift.com/exploring-the-foundation-models-framework/)
