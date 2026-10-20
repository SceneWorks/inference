# sc-26324 Candle native MOSS story review

VERDICT: CHANGES_REQUIRED

HEAD: `13d0cf343a4b5e44812ccc305ec4be45e39f4e39`

Reviewed delta: `51993b11be56d9d4dab5149a004f7c50b6b7c916..13d0cf343a4b5e44812ccc305ec4be45e39f4e39`, including shared commit `ef6990c4a7994aedf085d616d3302743c31c1b83`. The checkout was clean at entry and exit. This is the single independent story review, against `backend-review-brief.md`, the three Candle acceptance bullets in `backend-briefs.md`, and the approved sc-26324 description in `tracker-approved-implementation.json`. The current approved implementation scope supersedes the historical feasibility-only memory entry.

Confidence: **high** in the concrete finding and inspected source behavior; **medium** in overall numerical/backend correctness pending current-head CUDA compilation and the separately authorized sc-26325 real-weight campaign. No GPU, real weights, build target, source, Git, GitHub, or Shortcut mutation occurred. This report is the only written artifact.

## Concrete findings

#### [F-276] Preserve supplied FP32 conditioning on the full-FP32 DMD route

Priority: **P2 / major acceptance gap**. Confidence: **high**.

Location: `crates/media/candle-gen/candle-gen-moss-image/src/text_encoder.rs:109-113` (tight primary range). Caller: `src/lib.rs` positive advanced-conditioning branch; downstream conversion: `src/transformer.rs:603`.

Scenario: call `generate_moss_advanced` on CUDA with `acceleration = full_fp32`, supplied positive FP32 embeddings containing values not exactly representable as BF16, and supplied noise. `supplied()` selects dtype only from `device.is_cpu()`, so every CUDA/Metal row is narrowed to BF16 before the full-FP32 transformer receives it. The transformer later casts that already-rounded row back to the FP32 projection dtype. For example, the valid input `1.0010000467300415f32` becomes `1.0`; the original information cannot be recovered. Distinct valid FP32 conditioning inputs can therefore execute as identical conditioning on GPU, while CPU preserves their distinction. This violates the full-FP32 advanced conditioning semantics and backend equivalence in AC2.

Source authority: the retained frozen author pipeline (SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`) preserves caller embedding dtype in `prepare`, at `frozen_author_pipeline.py:383-389`: it moves embeddings to the device without a dtype cast. `_TextProjection.forward` then converts directly to `self.proj.weight.dtype`. For the published full-FP32 transformer that dtype is FP32. BF16 text-encoder outputs for string prompts are a separate, intentional policy; they do not justify quantizing explicitly supplied FP32 embeddings. The native `Transformer::forward` already contains the appropriate projection-owned conversion at line 603.

Fix direction: retain supplied rows as F32 through compaction/upload and let the selected transformer's text projection own the eventual cast. Alternatively, carry the resolved transformer conditioning dtype explicitly into preparation and preserve F32 for FullFp32. Keep the receipt hash bound to the caller values and mask.

Regression without GPU: make the production conditioning dtype decision directly testable independently of Device construction, if retaining a policy helper. Exercise the non-CPU/full-FP32 policy with a value such as 1.001f32, a noncontiguous mask, and the tiny FP32 author projection/graph on `Device::Cpu`; assert the compacted values remain exact and output matches the unrounded author expectation. Also show that an intentionally BF16-rounded input produces a different result. This tests the GPU route's policy on CPU without invoking CUDA. If the simpler fix removes the device-dependent cast entirely, test the same shared preparation/projection path with the precision-discriminating values and ensure there is no remaining device-dependent narrowing. The current supplied-embedding test only calls `Device::Cpu`, which takes the preserving branch even before the fix; the current direct full-FP32 transformer fixtures bypass `supplied()` altogether.

The coordinator explicitly assigned F-276. Per the read-only review boundary, no registry allocation was written here; the sole fix batch should add the required `CODE_REVIEW_*.md` record and append the allocation row, citing sc-26324 in code/tests as required by CONTRIBUTING.md.

No other concrete defect was established in this pass. There are no speculative/style findings.

## Acceptance coverage

| Acceptance | Review outcome |
| --- | --- |
| AC1: native exact-component DiT/text/VAE, ordinary sampling and correct optional distillation without Python production execution | Implemented in the inspected source: frozen manifest authentication, headless Qwen3.5, new MOSS transformer, exact small decode-only VAE, ordinary schedule and both DMD policies. Mathematical and structural fixture coverage exists. F-276 affects the advanced full-FP32 input route described under AC2. No real-weight execution claim is made. |
| AC2: full shared controls and advanced API with backend-equivalent semantics and exact optional precision behavior | **Gap: F-276.** Dimensions, expanded count/batch, masks, positive/negative conditioning, supplied FP32 noise, CFG 0/1/nonunit, unconditioned skips, SLG intervals, latent output, DMD prefix/Q recurrence and default-OFF validation are implemented. GPU preparation loses supplied FP32 conditioning precision on full-FP32 DMD. |
| AC3: catalog/load identity, memory/admission/device, progress/cancel/cleanup/provenance/cache, focused tests and CUDA compilation | Source integration and focused tests cover these interfaces; no additional defect established. CUDA compile validation remains an explicit current-head CI gate, so the complete AC is pending that gate rather than claimed met by source review. Real-weight acceptance belongs to sc-26325 and is not a missing local review gate. |

## Reviewed evidence and boundaries

- Read the full story delta and surrounding implementation relevant to loaders, transformer math, sampling, advanced conditioning, memory admission, Qwen callbacks, VAE decode, catalogs, licenses and fixture generation. CodeGraph was unavailable; no root CODEGRAPH.md was present in the reviewed inference checkout.
- Confirmed all **41** Rust frozen asset entries exactly match the combined approved `native-assets.expected.json` and `full-fp32-assets.expected.json` identities, sizes and digests. Independently hashed the retained author source and matched its required digest.
- Inspected same-owned-buffer authentication against both known expected SHA and retained pin for tensor/config/tokenizer/header parsing, prepared-child fail-closed behavior, confined indexed shard-set validation, and explicit Base/SFT/shared/DMD source selection. Optional assets do not select acceleration.
- Compared the MOSS graph with the retained author implementation: image/text sequence order, headwise RMSNorm, FP32 norm/RoPE calculations, source situ and FFN split, time features, shared/head modulation indexing, ordinary CFG/SLG/skips and custom Q orientation/fixed-noise recurrence. Mixed residual projections retain FP32 results, while source-mandated BF16 casts and FP32 time/modulation islands follow the source. Full-FP32 weights receive no second LoRA.
- Inspected selected-mode memory accounting, authenticated largest owned-shard transient, FP32 text upper bound, extra adapters/islands, request batch/context/branches, bounded own-weight cache credit and independent generation-time budget/geometry recheck. Old renderers are released before new-mode conditioning. Unimplemented optimized/tiling/streaming behavior is not represented as measured capability.
- Inspected final receipt publication, clearing before attempts, error cleanup and synchronization, typed cancellation at evaluated Qwen/DiT block boundaries and phase/image/step/final boundaries, fresh per-prompt Qwen state, and immutable OFF-to-ON-to-OFF reconstruction. Existing Qwen users retain a no-op check callback without added synchronization.
- Inspected explicit registry/catalog/feature forwarding, runtime catalog composition, footprint/contract registration, exact-bundle license disclosure, source-supported preview refusal, and the ignored terminal entry point. No terminal campaign was run or treated as a pass.
- Reviewer executed only bounded read-only checks: exact HEAD/clean status, `git diff --check` (pass), review-ID lookup, manifest equality/hash verification, and a stdlib IEEE-754 rounding demonstration. No redundant builds/tests were launched against the implementer's active target. Supplied implementer evidence reports core **1199 passed / 2 existing real-weight ignored**, new package **15 unit + 1 registry passed / 1 terminal ignored**, CPU independent FP32/BF16 and DMD fixtures executed, and Qwen **48 passed**; these counts were provided by the coordinator and were not independently rerun here.

Next authorized action: one batched fix for F-276, affected self-verification and required current-head CI. No routine second independent story review is requested.

## Sole remediation batch — sc-26324

The coordinator allocated F-276 to this review; the allocation is recorded in
`docs/code-review-finding-allocations.tsv`. This document preserves the original
independent review and its reviewed HEAD. The following is implementer remediation
evidence, not a second independent review.

The supplied-conditioning loader now preserves caller FP32 values through mask
compaction and device upload for every device. The selected transformer's existing
weight-dtype conversion is the sole conditioning conversion before text projection.
String-prompt Qwen dtype policy, receipt hashing, and public memory/admission APIs
are unchanged. This matches frozen author `prepare` and `_TextProjection.forward`.

The CPU-only regression
`supplied_fp32_conditioning_preserves_precision_until_selected_projection_sc26324`
uses deliberately non-BF16-representable values, a noncontiguous mask, and the
production ingestion and projection paths. It checks exact compacted caller values
and input receipt hash, compares the full-FP32 projection against a scalar affine
reference using the author fixture weights, distinguishes prematurely rounded
projection and complete-graph outputs, and verifies that ordinary BF16 and mixed
DMD BF16 text projections still perform the author's conversion. No GPU or real
weights are used.

Remediation status: **addressed and self-verified** in the sole authorized fix
batch; the original independent verdict above is preserved as review history.
Actual affected checks after the fix:

- Focused precision regression: passed on CPU.
- `cargo test --locked -p candle-gen-moss-image`: 16 unit and 1 registry integration
  passed; 1 real-weight terminal test remains explicitly ignored.
- `cargo test --locked -p candle-gen-catalog --lib`: 45 passed.
- Strict all-target Clippy and `RUSTDOCFLAGS='-D warnings'` rustdoc for
  `candle-gen-moss-image`, `candle-gen-catalog`, and `candle-llm`: passed.
- `cargo check --locked -p runtime-cpu --tests`: passed.
- Workspace gate: 108 members passed. Clock-assertion ratchet: 281 passed.
  Append-only review registry: F-276 high-water passed against the feature target.
  Integration-test consolidation, formatting, and diff whitespace checks passed.

The earlier full gen-core (1199 passed / 2 existing ignored) and Qwen model
(48 passed) evidence is retained; those unchanged components were not rerun for
this conditioning-only fix. All 49 earlier logs plus the fix logs are retained
under the epic's external `implementation/candle-native/` evidence directory.
CUDA compilation and real-weight validation remain the previously declared
downstream gates. No GPU, CUDA, Metal, or real-weight execution occurred locally.
