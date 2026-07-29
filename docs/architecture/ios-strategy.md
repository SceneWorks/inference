# iOS strategy — MLX on iOS, and why not CoreML

**Status: recommended, pending sign-off.** This records the decision, the options considered, and
the evidence for each. **The recommendation is Lane A — build MLX for iOS** — and it is *not*
contingent on a comparison spike: the deciding input is a product requirement (on-device image
generation), not a benchmark. §6 is the argument; §7 is the decision.

Date: 2026-07-29. Basis: read of `core-llm`, `mlx-llm`, `candle-llm`, `gen-core`, the media
provider crates, the bundles, the conformance testkit, and the pinned `mlx-sys` fork's `build.rs`.

Implementation plan: [iOS project spec](ios-project-spec.md).

---

## 0. The question, and what settles it

"CoreML-compatible" collapses two different asks that differ by roughly an order of magnitude:

| Ask | Door | Cost |
|---|---|---|
| **Run on iPhone at all** | MLX already runs on iOS (Metal). Reuse `mlx-llm` nearly whole. | weeks |
| **Run on the Neural Engine** (battery, thermals, leave the GPU free) | CoreML is the only door. New backend + an ahead-of-time conversion pipeline. | months |

An earlier draft proposed a spike to measure ANE energy and decide between them. **That gate is
no longer the deciding factor.** On-device image generation is a launch requirement, and §6 shows
that requirement is structurally incompatible with a CoreML LLM lane — for reasons that hold
regardless of what an ANE benchmark returns. The decision is therefore made on architecture, and
the remaining spike (project spec §2) is a *feasibility* check on the chosen lane, not a
comparison.

**Not a reimplementation, and not a fork of this repo.** Either lane is an **additional provider
crate plus one bundle inside this workspace**, composed through the existing registry exactly as
`candle-llm` and `mlx-llm` are today. No parallel library, no divergent history, one lockfile.
The contract boundary is thin and genuinely tensor-free — the thing that usually kills ports like
this. It holds here.

---

## 1. What the architecture buys you

`core_llm::TextLlm` is **three methods**:

```rust
fn descriptor(&self) -> &TextLlmDescriptor;
fn validate(&self, req: &TextLlmRequest) -> Result<()>;
fn generate(&self, req: &TextLlmRequest, on_event: &mut dyn FnMut(StreamEvent)) -> Result<TextLlmOutput>;
```

That is the entire surface a new engine must satisfy. `candle-llm` already proves "add a second
backend behind this contract" is a repeated, solved exercise here.

### Reused verbatim — ~6.6k lines, zero port cost

All of `crates/contracts/core-llm/` is tensor-free by design:

- `template.rs` (1058) — Jinja chat templates
- `constraint.rs` (712) — the `JsonState` incremental-validity machine
- `tool.rs` (516) — tool-call segmentation/parsing
- `prepare.rs` (477), `schedule.rs` (339), `prefix.rs` (315), `thinking.rs` (312),
  `message.rs` (302), `speculative.rs` (283), `stop.rs` (250), `paging.rs` (208),
  `detok.rs` (205), `tokenizer.rs` (200), `capabilities.rs`, `registry.rs`
- The `core-llm-testkit` conformance suite — the definition of done, already written

Tokenization, chat templating, stop matching, incremental detokenization, thinking/tool
segmentation, capability gating, scheduling and prefix policy: **none of it is rewritten**, on
either lane.

### What a CoreML lane would have had to replace

Recorded because it sizes the rejected option, not because it is planned work:

- **Reimplemented:** `decode/` (~1.7k), `primitives/sampler.rs` (455),
  `primitives/kv_cache.rs` + `paged_kv_cache.rs` (~1.2k) — all MLX-`Array`-typed.
- **Deleted, but relocated:** `models/*` (~5k), `primitives/{rope, attention, gated_delta}`
  (~1.9k), `gguf/{dequant, iq_grids, convert}` (~2.7k). The compiled `.mlpackage` absorbs these,
  but that ~9.6k of Rust reappears as a `coremltools` conversion pipeline, per architecture
  family, in Python — which this repo has **no precedent for**. That relocation is the single
  biggest thing a line-count estimate of a CoreML port gets wrong.

---

## 2. Lane A — MLX on iOS **(recommended)**

MLX supports iOS; `mlx-swift` runs LLMs on iPhone today. `mlx-llm` is Rust over `mlx-rs`, so the
engine's own code compiles for `aarch64-apple-ios` essentially unchanged.

**The blocker is the build, and it was checked rather than assumed.** The pinned fork
(`pmetal-mlx-sys` @ `932beb4`) has no iOS story at all:

- `build.rs` is `#[cfg(target_os = "macos")]`-gated in three places, including the entire
  deployment-target setup. **Note the subtlety:** build scripts compile for the *host*, so on a
  macOS host those branches run even when cross-compiling to iOS. They do not skip the iOS
  build — they **mis-configure** it (setting `MACOSX_DEPLOYMENT_TARGET`, resolving the macOS
  clang runtime, caching the metallib to `$HOME`). The fix is target-aware branching on
  `CARGO_CFG_TARGET_OS` / the `TARGET` env var, not un-gating.
- It drives MLX's C++ via `cmake::Config` with **no iOS toolchain arguments** — no
  `CMAKE_SYSTEM_NAME=iOS`, no `IPHONEOS_DEPLOYMENT_TARGET`.
- It caches `mlx.metallib` into `~/.cache/pmetal/lib/`. **That is meaningless in an iOS app
  sandbox** — the metallib must be bundled into the `.app` (the patched resolver's
  `$PMETAL_METALLIB_PATH` / `set_metallib_path()` is the seam).
- It applies **three `required = true` patches** to MLX core; all three must keep applying on an
  iOS-capable base.
- `.cargo/config.toml`'s `MACOSX_DEPLOYMENT_TARGET = "26.2"` NAX-kernel setup is macOS-specific
  and does not carry over.

So Lane A is not "flip a target triple". It is **fork-the-fork build engineering** with a
maintenance tax — the repo would own iOS support in a fork of a fork of `mlx-rs`. Mitigation:
keep the delta additive and attempt upstreaming to `pmetal-mlx-rs` early; a merged iOS target
erases most of that tax.

**Effort: ~4–8 weeks** to a running iPhone build, dominated by build/toolchain work.

**What it gets:** the full existing engine — every model family, vision, GGUF ingest, speculative
decode, paged KV — *and* a viable path to on-device image generation (§6).

**What it does not:** the ANE. MLX is Metal-only, so decode runs on the GPU, competing with the
UI and costing more battery than an ANE path would. This is a real and accepted cost; see §7.3
for how it could be recovered later.

**Zero contract changes.** Lane A registers `mlx_llm::text_registry()` and
`mlx_llm::snapshot_preparer_registry()` exactly as `runtime-macos` does. No new `ModelFormat`
variant, no widened `backend` tag, no catalog carve-out. That is a genuine and underrated
advantage over Lane B, which needs all three (§5).

---

## 3. Lane B — a `coreml-llm` engine crate **(considered, not recommended)**

The only door to the ANE, and architecturally clean in isolation.

**New Rust, roughly 4–6k lines** (against `mlx-llm`'s 22k, because the model layer is gone):
`provider.rs` (800–1200), CoreML FFI via `objc2` (800–1200), prefill/stateful-step decode loop
(600–900), CPU sampler over the logits buffer (400–600), config/load/errors/tests (900–1400).

**Plus the conversion pipeline, which is where the time actually goes:** per architecture family,
`torch.export` → `coremltools` with `minimum_deployment_target=iOS18`, a stateful `slice_update`
KV cache, 4-bit palettization, and chunking for large graphs. Apple's own Llama 3.1 CoreML work
is the right comparison for how non-trivial that is.

**Effort: ~8–14 weeks** to a first shipping *text-only* provider covering **one** architecture
family, then **+2–4 weeks per additional family**. That per-family tail is the structural
difference from today, where a new family is a Rust file every backend loads.

**Three further catches, independent of the §6 argument:**

1. The ANE is a fixed-shape accelerator designed for convolutions. Autoregressive decode —
   memory-bandwidth-bound, variable sequence length — is the one workload it handles badly, and
   CoreML adds meaningful per-op overhead on small operations. The ANE win is therefore *assumed,
   not established*.
2. Fixed input shapes mean **context length is baked into the converted graph**. A longer context
   is a re-conversion, not a config change.
3. The graph must emit **full float logits** (no fused argmax/sampling), because
   `check_seed_determinism`, JSON-constraint masking, and speculative decode all need host access
   to the distribution. At Qwen3's ~151.9k vocab that is ~300 KB crossing ANE→CPU *every token* —
   a plausible enough bottleneck that it would need measuring before committing.

---

## 4. Lane C — Apple Foundation Models — cheap, but wrong shape

Superficially cheapest: iOS 26 ships a ~3B on-device model with guided generation and tool
calling, and the adapter is maybe 500–800 lines.

**The disqualifier is the weights premise, not conformance.** An FM provider cannot load
caller-provisioned weights at all — there is one Apple-supplied model and no `WeightsSource::Dir`
to point at. That contradicts the premise of the entire epic-13657 self-fetch design, in which
inference receives every model component as a local path the consumer provisioned. It also means
no model choice, no quantization control, and no image generation.

*(An earlier draft asserted FM would fail `check_seed_determinism`. It would not.
`GenerationOptions` exposes `sampling: .greedy` and `SamplingMode.random(top:seed:)`, satisfying
both legs of the check. The conformance objection does not hold; the weights objection is
independent and stronger.)*

Not a main path. If wanted, scope it deliberately as a convenience provider outside the
model-loading registry, where the `WeightsSource` contract does not apply.

---

## 5. Repo-side costs, by lane

| Cost | Lane A | Lane B |
|---|---|---|
| `runtime-ios` bundle, `EXPECTED_MEMBER_COUNT` bump, ordered surface test, CI lane | yes | yes |
| On-device/simulator CI (no precedent — all three existing bundles are natively `cargo test`-able desktop targets) | yes | yes |
| Swift/host FFI layer (`&mut dyn FnMut(StreamEvent)` does not cross FFI) | yes | yes |
| **`ModelFormat::CoreML` variant** in the tensor-free contract crate | no | **yes** |
| **Widened `TextLlmDescriptor.backend` doc** beyond `"mlx" \| "candle"` | no | **yes** |
| **A second `runtime-catalog` backend carve-out** to mix CoreML text with MLX media | no | **yes** |
| Per-architecture Python conversion pipeline, maintained per model release | no | **yes** |

On the `ModelFormat` row: a snapshot preparer is **mandatory** —
`runtime-catalog/src/lib.rs:358` fails validation with `"runtime has no snapshot preparer"` on an
empty registry. `can_prepare` is a free `fn(&PrepareSpec) -> bool` and can sniff a `.mlmodelc`
without touching the closed enum, but `PrepareReport.input_format` is a **required
`ModelFormat`** field. A CoreML preparer must therefore either report a `.mlmodelc` as
`Safetensors` — the kind of silent dishonesty this codebase rejects everywhere else — or the
enum gains an additive `CoreML` variant. Lane A sidesteps this entirely by reusing
`mlx_llm::snapshot_preparer_registry()`.

---

## 6. Image generation — the deciding argument

On-device image generation is a launch requirement: a small image-only generator, and a unified
model that generates both text and images. **Both already exist as crates in this workspace**,
which reframes the question from "can we build this?" to "can we build *what we have*?".

| Need | Crate | Size |
|---|---|---|
| Unified AR LLM + image | `mlx-gen-sensenova` — SenseNova-U1 (NEO-Unify), *"a unified AR LLM + flow-matching image generator"* | 9.6k lines |
| Small image-only | `mlx-gen-sana` — SANA (NVlabs) 0.6B/1.6B + DC-AE deep-compression decoder | 6.4k lines |
| Alternative | `mlx-gen-z-image` — Z-Image-Turbo, few-step but larger | 15.1k lines |

Note that SANA's text conditioning **is Gemma-2-2B-it**, reused from `mlx-gen-pid`'s
`CaptionEncoder` (epic 8485 / sc-8488). "A Gemma that generates images" is already the shipping
architecture here, not a new direction. Rough budget at 4-bit: Gemma-2-2B encoder ~1.4 GB + SANA
DiT ~0.35 GB + DC-AE decoder ≈ **~2 GB**, comfortably under the per-app cap with headroom to
co-resident an LLM.

### Why this rules out a CoreML LLM lane

Three structural reasons, each verified against the source:

1. **`mlx-gen-sensenova` depends on `mlx-llm` directly.** Its dual-path AR runtime and denoise
   loop consume `ContiguousKvCache`, `sample`, and `Rope` from the LLM engine (sc-7159) rather
   than hand-rolled copies. **Those are precisely the primitives a CoreML lane deletes.** A
   CoreML LLM does not merely fail to help the unified model — it removes what the unified model
   is built on.

2. **`gen-core` has 19 public traits to `core-llm`'s one**, and several are *tensor-manipulation*
   traits: `LatentOps`, `Sampler<L: LatentOps>`, `SamplerPolicy`, `GuidanceOps`, `ModelSampling`.
   Diffusion does real latent arithmetic **between** every model call. `TextLlm` gets away with
   handing back an opaque logits buffer the host samples on CPU; a media backend cannot. On
   CoreML you would either read back full latents every step (slow) or compile the sampler into
   the graph (destroying the pluggable-sampler architecture). **This is why image generation is
   categorically harder to port than text, not merely bigger** — and why the "compiled graph
   replaces the model code" saving that makes a CoreML LLM cheap does not apply here at all.

3. **A mixed-backend bundle needs a new architectural carve-out.** `runtime-catalog` enforces one
   tensor backend across the media, LLM, and preparer registries. The audio lane is the *single*
   sanctioned exception (sc-12901), and it is a deliberate, documented decision with its own
   backend field and preparer registry. CoreML-text-plus-MLX-images would need a second such
   exception — an ADR-level change, not an implementation detail.

### And the media crates are already portable

`grep -rn 'target_os|cfg(unix)|cfg(windows)'` across `mlx-gen/src`, `mlx-gen-sana`,
`mlx-gen-pid`, and `mlx-gen-sensenova` returns **zero** platform gates. The media Rust is
platform-neutral; the only iOS blocker is `mlx-sys`'s `build.rs` — which is *already* Lane A's
first phase. **The image-generation track's build risk collapses into Lane A's**, and is
additional-cost-free on top of it.

---

## 7. Decision

### 7.1 Recommendation: Lane A

**Build MLX for iOS.** Reasons, in order of weight:

1. **It is the only lane that satisfies the image-generation requirement** (§6). Lanes B and C
   cannot, for structural reasons that no benchmark changes.
2. **It reuses what exists.** `mlx-llm` (all ten architectures, vision, GGUF, speculative decode),
   `mlx-gen-sana`, `mlx-gen-sensenova`, and the whole `mlx-gen` core come along once one
   `build.rs` supports iOS.
3. **It requires zero contract changes** (§5) — no `ModelFormat` variant, no widened backend tag,
   no catalog carve-out.
4. **It is 5–8 weeks cheaper on text alone**, before the image track is counted.

**Accepted costs**, recorded so they are not rediscovered as surprises:

- **No ANE.** Decode runs on the GPU: more battery, more heat, and contention with the UI.
- **A fork of a fork.** The repo owns iOS support in `pmetal-mlx-sys` until it can be upstreamed.
- **A threading contract.** `mlx-llm` engines are neither `Send` nor `Sync`, and MLX's shared
  Metal device is not thread-safe. On macOS that reads as a test detail; on iOS it is host-app
  correctness and must be specified, not left implicit.

### 7.2 What would reopen this

Record the trigger, so the decision is revisitable on evidence rather than by re-litigation:

- The image-generation requirement is dropped or deferred indefinitely, **and** measured GPU
  battery/thermal cost on device proves unacceptable for the product; or
- `mlx-sys` proves genuinely un-buildable for `aarch64-apple-ios` (project spec §2 tests this
  first, precisely because it is the one assumption Lane A rests on). In that case Lane B becomes
  the text fallback and the image track needs re-planning from scratch.

### 7.3 Not rejected — deferred

The ANE is worth revisiting **after** v1 ships, in the form the state of the art actually favours:
**disaggregated** inference — ANE for prefill (compute-bound, big fixed matmuls) and GPU for
decode (which the ANE handles badly anyway). The registry/composition model accommodates this
natively: it is two providers, or one provider holding both handles, added without disturbing
anything above the engine. Choosing Lane A now does not foreclose it.

---

## Sources

Repo-local findings (contract shape, line counts, `mlx-sys` `build.rs` gating, conformance check
set, `runtime-catalog` validation, media-crate platform gates, `EXPECTED_MEMBER_COUNT`) were read
directly from this workspace and the pinned
`~/.cargo/git/checkouts/mlx-rs-.../932beb4/mlx-sys` checkout. External claims:

- CoreML stateful models / `MLState` KV cache, `minimum_deployment_target=iOS18`, the ~1.6×
  Mistral-7B speedup from stateful vs. non-stateful KV cache —
  [coremltools Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html),
  [WWDC24: Deploy ML and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/)
- ANE fixed-shape constraint, low-bit palettization vs. block-wise quantization, CoreML per-op
  overhead — [Apple ML Research: On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama),
  [CoreML-LLM](https://github.com/john-rocky/CoreML-LLM)
- Disaggregated NPU-prefill / GPU-decode direction (§7.3) —
  [SqueezeBits: Disaggregated Inference on Apple Silicon](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- **iPhone ~6 GB per-app ceiling (single-source — verify on device before planning against it)**
  and the GPU-buffer-cache/jetsam anecdote —
  [How Fast Are On-Device LLMs on iPhone 17 Pro and iPad Pro?](https://rickytakkar.com/blog_russet_mlx_benchmark.html)
- MLX-on-iOS precedent —
  [Running an LLM on iPhone with MLX Swift (awni)](https://gist.github.com/awni/fe4f96c21ead68e60191190cbc1c129b),
  [WWDC25: Explore LLMs on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- Foundation Models `GenerationOptions` sampling/seed —
  [WWDC25: Deep dive into the Foundation Models framework](https://developer.apple.com/videos/play/wwdc2025/301/),
  [Exploring the Foundation Models framework](https://www.createwithswift.com/exploring-the-foundation-models-framework/)
