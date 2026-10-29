# Documentation

Documentation for the SceneWorks inference workspace.

## Start here

- **[Getting Started](guide/getting-started.md)** — depend on a runtime bundle, build its
  validated registries, and load/run media generators, LLMs, and snapshot preparers. The
  consumer *how-to*.
- **[Model Catalog Reference](reference/model-catalog.md)** — every shipped provider id, per
  platform, with the MLX/Candle deltas. Built from the committed exact-surface tests.
- **[Mochi 1 tier strategy](reference/mochi-1-tier-strategy.md)** — the accepted quantization
  packaging decision for the native Mochi 1 port (pre-quantized per-tier artifacts, default q4).
- **[LTX-2.5 access + component evidence baseline (sc-18756)](reference/sc-18756-ltx-2-5-evidence-baseline.md)** —
  epic 18755 Phase 0: measured HF access state, the reference-impl pin, the full 2.3→2.5 component
  inventory (14 safetensors files, header-only Range reads, authenticated capture), and the
  pre-change `dump-engine-capabilities` baseline every later LTX-2.5 story cites instead of
  re-deriving.
- **[MiniMax-H3 withheld upstream components](reference/minimax-h3-withheld-upstream-components.md)**
  — what `H3-Context-IR`, `H3-Regenerate-2K`, sparse-attention inference and the `<d>` dialogue
  markers cost this port, what the crates do instead, and what would change if upstream publishes.

## Architecture

- **[Inference Rearchitecture Rationale](architecture/inference-rearchitecture.md)** — why
  the repositories were consolidated, why provider discovery is explicit, the alternatives
  considered, the accepted tradeoffs, and the invariants future changes must preserve. The
  *why* behind everything above.
- **[Audio Backend Strategy](architecture/audio-backend-strategy.md)** — why audio generation
  is Candle-native on every platform (no ONNX/third backend), and how the runtime catalog's
  dedicated audio section carries a `candle` audio lane inside the mlx macOS bundle.

## Licensing

The [`licensing/`](licensing/) directory holds the primary-source evidence behind the model-weight
licence surface — the canonical text URL, the verbatim upstream identifier, and a quoted operative
clause for every term assigned to a licence family.

- **[Licence family evidence pack (sc-16662)](licensing/sc-16662-licence-family-evidence.md)** —
  the licence families, quote-checkable, with an explicit unresolved list. **Draft, unsigned.** It
  records facts, not legal conclusions.
- **[Media checkpoint census (sc-16665)](licensing/sc-16665-media-checkpoint-census.md)** — which
  upstream checkpoints every registered media provider actually loads, read from the code. Records
  component identity only: it assigns no families and asserts no licences, and marks every component
  whose upstream the repository does not state as UNDETERMINED.
- **[Qwen Research License evidence (sc-24108)](licensing/sc-24108-qwen-research-licence-evidence.md)** —
  the primary-source read behind the `qwen-research` family and the `qwen_image_2_1` component row
  (Qwen-Image 2.1). **Draft, unsigned.**
- **[FLUX Kommunity licence evidence (sc-26234)](licensing/sc-26234-flux-kommunity-licence-evidence.md)** —
  the primary-source read behind the `flux-kommunity-v1-0` family and the `flux3_image_klein`
  component row (FLUX 3 Image Klein, epic sc-26233), including the LFS-sha256 identity of the
  bundled text encoder with `Qwen/Qwen3-VL-4B-Instruct`.
- **[Qwen-Image 2.1 Turbo licence evidence (sc-25900)](licensing/sc-25900-qwen-image-2-1-turbo-licence-evidence.md)** —
  the primary-source read behind the `qwen_image_2_1_turbo` component row (same `qwen-research`
  family; the `LICENSE` is byte-identical to the base repository's). **Draft, unsigned.**
- **[Iris-3B licence evidence (sc-25679)](licensing/sc-25679-iris-licence-evidence.md)** — the
  primary-source read behind the `iris_3b` and `qwen3_vl_4b_instruct` component rows (Iris-3B,
  Apache-2.0). **Draft, unsigned.**
- **[Tongyi Qianwen License evidence (sc-22989)](licensing/sc-22989-tongyi-qianwen-licence-evidence.md)** —
  the primary-source read behind the `tongyi-qianwen` family and YuE2's `yue2_qwen_tiktoken`
  component row (`qwen.tiktoken` is byte-identical to Qwen-7B's). **Draft, unsigned.**
- **[Checkpoint licence evidence (sc-16665)](licensing/sc-16665-checkpoint-licence-evidence.md)** —
  the primary-source licence read for each of those checkpoints, and the sign-off document behind
  `license::components`. **Draft, unsigned.** Its
  [known holes](licensing/sc-16665-checkpoint-licence-evidence.md#known-holes--the-rows-sc-16665-deliberately-did-not-write)
  section is the authoritative list of checkpoints that were left without a row rather than given a
  guessed one.

## Native H3 ConvRot INT8 checkpoints (sc-26683)

The H3 projection loaders distinguish symmetric ConvRot INT8 from MLX affine Q8:
`weight` is I8 `[out, in]`, `weight_scale` is **F32 `[out, 1]`**, and a bounded U8
`.comfy_quant` JSON tensor declares `int8_tensorwise`, `convrot: true`, and
`convrot_groupsize: 64` or `256`. These are **rotation** groups, not scale groups.
Partial rotation tails, unknown fields/formats, contradictory affine operands,
nonpositive/nonfinite scales, and missing descriptors fail before execution.

Tiny HTTP ranges of both `Minimax-h3_Singularity_ref2va_v2_int8.safetensors` and
`Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors` at
[`WarmBloodAban/Minimax-h3_Singularity@a56850c870825d6a8ae501a595c4dbbbc77bbc7e`](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/tree/a56850c870825d6a8ae501a595c4dbbbc77bbc7e)
established 1,035 tensors, a 129,920-byte header, and 250 descriptors in a
17,950-byte descriptor span in **each** file: 50 group-64 AdaLN projections and
200 group-256 attention/MLP projections. All 250 pairs are I8 codes and F32 row
scales. No checkpoint weight payload or full-model inference was used for this evidence.

The executable authority is
[`comfy-kitchen@f8b7e97a1418d2e529bd5405fbf86312be198ae6`](https://github.com/Comfy-Org/comfy-kitchen/tree/f8b7e97a1418d2e529bd5405fbf86312be198ae6),
with exporter
[`comfy-model-tools@d6797787e6bdb1a1fb0094d588a26f8e71a1c757`](https://github.com/Comfy-Org/comfy-model-tools/tree/d6797787e6bdb1a1fb0094d588a26f8e71a1c757).
The separate
[`starsFriday W4A8 loader@ad1d4e072bd2ed5eec1d30d07ec390f327a3af52`](https://github.com/starsFriday/ComfyUI-W4A8-Loader/tree/ad1d4e072bd2ed5eec1d30d07ec390f327a3af52)
is not authority for this INT8 dialect. Shared regular Hadamard rotation is
followed by dynamic per-token quantization: store the unrounded F32 scale,
round the divisor and quotient through the activation dtype, nearest-even
integer rounding, and clamp activation codes to `[-128, 127]`. The projection
accumulates integer products in I32, applies both row scales in F32, adds bias,
and rounds the output once. Weights remain compact; no full dense weight set is
created. Candle reuses `Int8Linear` / one `Int8Context` / cuBLASLt, retaining the
original byte Tensor storage rather than copying it during staging. MLX uses a
shared signed-integer Metal projection through MLX's existing `MetalKernel`
compiler seam and the shared adapter base. Two immutable rotation matrices are
shared per load, not duplicated per layer. CPU Candle refuses native execution.

[`scripts/h3_convrot_int8_oracle.py`](../scripts/h3_convrot_int8_oracle.py) regenerates
the six small tensor-valued cases in
[`tests/fixtures/h3_convrot_int8.json`](../tests/fixtures/h3_convrot_int8.json).
It **executes** the pinned upstream Hadamard helpers on CPU Torch; its activation
quantizer/epilogue are an explicit arithmetic port of the separately hashed
pinned CUDA source, not a claim to have executed that CUDA source on CPU.
Cases cover both groups, multiple rotation blocks, five token rows, source
F32/BF16/F16 rounding, zero rows, halfway values and a large outlier. F32 native
output tolerance is `2e-5 * max(1, abs(expected))`; BF16/F16 outputs require exact
agreement with the committed oracle. The host test checks rotation and every
activation code/scale; native tests load actual synthetic I8 safetensors through
the production H3 projection dispatcher, check selected execution, then compare
all output values. Neither an ignored test nor a CUDA compile is a device pass.

Explicit device commands (exclusive coordinator-reserved device required):

```sh
cargo test --locked -p mlx-gen-minimax-h3 --test integration convrot_int8::native_projections_match_pinned_kitchen -- --ignored --exact --nocapture
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration convrot_int8::native_projections_match_pinned_kitchen -- --ignored --exact --nocapture
```

The existing manual `windows-cuda` CI lane runs the latter command. CUDA
typecheck and device execution remain distinct evidence. SceneWorks catalog
admission and its consumer pin are paired changes, not implied by this inference
implementation. Asymmetric W4A8 remains a separate slice; table-driven AdaLN is below.

The manual lane's broad CUDA package-test step sets `IRIS_DIST_SHARE_DEVICE=1`:
Iris' real multi-process fixtures explicitly share the lane's one visible card,
rather than requesting absent `cuda:1`/`cuda:2`/`cuda:3` devices and failing before
rendezvous. Their separate missing-device test still exercises both the explicit
refusal and opt-in-sharing paths. This does not skip scenarios or relax numeric
assertions.

## Native H3 centered-curve AdaLN checkpoints (sc-26684)

Both H3 backends consume the supplied `adaln_t_table` and reduced projections
directly. The shared `gen-core::pruned_adaln` contract validates the entire
canonical topology before execution: one F32 `[grid >= 2, rank]` table, all
`transformer_blocks.{i}.adaln_proj.linear.{weight,bias}`, and final
`norm_out.linear.{weight,bias}`. Reduced block projections emit `18 * hidden`
columns; the final projection emits `2 * hidden`. All consume the table's rank.
A missing table, folded bias, mismatched width, nonfinite operand, or simultaneous
full timestep MLP is refused, never replaced with stock weights or another basis.
The source partition and complete block count remain unchanged.

The arithmetic authority is the executable pruning, transformer, and packing code
at [WanGP `47b06ff920c16b4b7b36de068367a5de0e0968e9`](https://github.com/deepbeepmeep/Wan2GP/tree/47b06ff920c16b4b7b36de068367a5de0e0968e9/models/minimax_h3).
Its centered SVD stores `U*S` in the table, `W*Vh.T` in the reduced projections,
and `b + W*mean` in the folded bias. Inference does **not** reconstruct `Vh`,
rederive a basis, or apply a second SiLU. Coordinates interpolate at
`clamp(t, 0, 1) * (grid - 1)` with `t = 1 - sigma`, without a factor of 1000.
Text and target-video rows use the video timestep; conditioned-video rows use
`max(video_t, .999)`, target audio uses `audio_t`, and conditioned audio uses 1.

The frozen pruned INT8 artifact at
[`a56850c870825d6a8ae501a595c4dbbbc77bbc7e`](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/tree/a56850c870825d6a8ae501a595c4dbbbc77bbc7e)
has 50 blocks, a **stored F32 `[1025,8]` table**, 50 BF16 `[96768,8]`
projections plus final BF16 `[10752,8]`, and folded F16 biases of the
corresponding output widths. Stored precision is identified and hashed per operand
before lossless F32 compute widening, matching the pinned runtime's `_lock_dtype`.
The engine retains one widened compute operand set, not both source and compute
copies, and never recasts weights/biases per timestep. Resident-byte accessors report
the actual F32 allocations (174,602,272 B including the table at this geometry),
not the narrower checkpoint bytes. The pruner's F32 output is not misreported as
the artifact's stored precision. The 200 body ConvRot
group-256 projections remain native INT8, not dense AdaLN or group-64 substitutes.

The shared Candle loader now takes `Weights::from_files_native_int8(files, device)`:
there is no load-time compute-dtype argument. It preserves **all** floating source
dtypes, including unrelated F32 I/O heads, and materializes original signed INT8
codes as exact U8 two's-complement storage with F32 scales. H3 projections choose
their compute dtype at the projection construction boundary. Consumers migrating from the former three-argument
API must remove that argument, not add a pre-load cast or reread pruning operands.

`PruningIdentity` binds the table, **every** reduced projection and folded bias,
their canonical keys, shapes and stored dtypes. Same-shaped operands from a
different centered basis are incompatible; content identity participates in
modulation-cache reuse. It is not a filename, allowlist, or table-only identity.
The existing provider/import platform remains the only route; SceneWorks import
mapping/admission and its exact inference pin require their paired consumer change.

[`scripts/h3_pruned_adaln_oracle.py`](../scripts/h3_pruned_adaln_oracle.py) executes
hashed pinned definitions on CPU with deterministic synthetic data. It generates
[`tests/fixtures/h3_pruned_adaln.json`](../tests/fixtures/h3_pruned_adaln.json),
not the downloaded source table: 10 off-grid/endpoint/clamped/conditioned timestep
cases, mixed stored precision, two complete tiny block outputs and final
video/audio outputs. The F32-coordinate absolute bound is `2e-6`, reduced-projection
bound `2e-5`, Candle block/output bound `2e-5`, and MLX block/output bound `2e-3`
(its existing Metal F32 body-matmul precision). These are elementwise gates, not
cosine-only or hash-only checks. Explicit native tests run in the existing hosted
Metal and manual CUDA CI jobs, with private HF environments. Ignored tests or
compile-only checks are not device evidence:

```sh
cargo test --locked -p mlx-gen-minimax-h3 --test integration pruned_adaln::native_pruned_adaln_matches_pinned_reference -- --ignored --exact --nocapture
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration pruned_adaln::native_pruned_adaln_matches_pinned_reference -- --ignored --exact --nocapture
```

Regeneration takes an offline `--reference-dir` containing the three pinned text
files and verifies their SHA256 hashes before execution. Local CPU checks must
use disposable `HF_HOME`, `HF_HUB_CACHE`, and `HUGGINGFACE_HUB_CACHE`, plus OS
denial of production model roots and all networking. No real-weight downloads,
full-checkpoint execution, memory campaign, or permanent consumer pin is part of S5.

## Migration records

The [`migration/`](migration/) directory records how this repository was assembled from the
former `core-llm`, `mlx-llm`, `candle-llm`, `mlx-gen`, and `candle-gen` histories — source
SHAs, filtered-history commit maps, tree-equivalence checks, and per-phase checkpoints. See
[`migration/README.md`](migration/README.md) for the index.

## Crate-level docs

Each layer has a README next to its source:

| Layer | Crates |
| --- | --- |
| Contracts | [`gen-core`](../crates/contracts/gen-core/README.md), [`gen-core-testkit`](../crates/contracts/gen-core-testkit/README.md), [`gen-core-dist`](../crates/contracts/gen-core-dist/README.md), [`gen-core-telemetry`](../crates/contracts/gen-core-telemetry/README.md), [`core-llm`](../crates/contracts/core-llm/README.md) |
| Bundles | [`runtime-catalog`](../crates/bundles/runtime-catalog/README.md), [`runtime-macos`](../crates/bundles/runtime-macos/README.md), [`runtime-cuda`](../crates/bundles/runtime-cuda/README.md), [`runtime-cpu`](../crates/bundles/runtime-cpu/README.md) |
| LLM engines | [`mlx-llm`](../crates/llm/mlx-llm/README.md), [`candle-llm`](../crates/llm/candle-llm/README.md) |
| Media engines | [`mlx-gen`](../crates/media/mlx-gen/README.md) (+ [`mlx-gen-catalog`](../crates/media/mlx-gen/mlx-gen-catalog/README.md)), [`candle-gen`](../crates/media/candle-gen/README.md) (+ [`candle-gen-catalog`](../crates/media/candle-gen/candle-gen-catalog/README.md)) |

For MLX media internals, see
[`mlx-gen/ARCHITECTURE.md`](../crates/media/mlx-gen/ARCHITECTURE.md) and
[`mlx-gen/docs/MODEL_ARCHITECTURE.md`](../crates/media/mlx-gen/docs/MODEL_ARCHITECTURE.md).

## Release

- [`release/README.md`](../release/README.md) — the immutable, calendar-versioned release
  train (`runtime-YYYY.MM.patch` tags), release gates, and bundle contents.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`SECURITY.md`](../SECURITY.md) — contribution
  boundaries and security reporting.
