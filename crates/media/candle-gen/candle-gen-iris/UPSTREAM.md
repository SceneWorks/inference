# candle-gen-iris — upstream coverage

The coverage table, pinned revisions, fixtures and tolerances for Iris-3B live in the MLX twin's
[`UPSTREAM.md`](../../mlx-gen/mlx-gen-iris/UPSTREAM.md), which carries a Candle column; this crate's
parity tests read the same committed fixtures. This file records only what is specific to the
Candle crate.

## Memory-strategy contract (sc-25690)

Upstream ships no memory policy; this is the repo's shared ladder (`gen_core::MemoryProviderContract`)
declared for what each Candle route actually runs. `src/memory_strategy.rs` owns all three contracts,
and `register_providers` registers each with its weights-free fixture and behavior seam. `iris_3b` is a
registry generator; depth (a provider-specific API) and restoration (a transform) are registered as
composed routes.

| | `iris_3b` | `iris_3b_depth` | `iris_3b_restore` |
|---|---|---|---|
| Memory mode key | `text_to_image` | `depth` | `restoration` |
| Resident | implemented | implemented | implemented (one backbone forward per export tile, always) |
| StagedResidency | implemented (text encoder → encode → drop → backbone) | not applicable (no text encoder) | not applicable (no text encoder) |
| BoundedDecode | not applicable (pixel space) | not applicable | not applicable |
| BoundedAttention | implemented (64 Mi score elements) | missing | missing |
| BoundedTransformerResidency | missing | missing | missing |
| Witnessed load surfaces | {bf16, q8, q4} × {resident, sequential} × eager (sc-26266) | bf16 × {resident, sequential} × eager | same as depth |
| Resident bytes (materialized snapshot) | text tower (`model.language_model.*` + untied `lm_head`, at the production device's tower dtype; attention/MLP projections as GGML blocks at a quantized tier) + backbone (projections at the compute dtype — GGML blocks over the padded input at a quantized tier — norm gains / modulation tables / text position table at f32) + adapter overlays | `empty_prompt.safetensors` (f32) + backbone and reducer | `empty_prompt.safetensors` (f32) + backbone |

Differences from the MLX twin, both deliberate:

- **BoundedAttention is implemented here and missing on MLX.** Candle's attention materializes the
  `[rows, heads, queries, keys]` scores tensor, so bounding it is a real lever; MLX's fused kernel
  never builds it. The budget is the shared 64 Mi operating point, a derived candidate the epic's
  terminal campaign has yet to measure.
- **No deferred load surface.** No Iris Candle loader has a block-streaming shape, so a
  `DeferredMaterialization` spec has no contract and is refused at load.

Staged residency and bounded attention are per-request decisions on generation: the request scope
writes `GenerationMemory`, and `generate` honours it. A `Sequential` load stages every request.
Depth and restoration requests carry no memory block, so those routes are resident only; that is
also why their bounded-attention rung is missing rather than not applicable.

Every byte figure is read from safetensors headers, including the adapter stack
(`adapters::adapter_footprint`), which is declared as up to two typed components:
`iris_3b.adapters.forward_residuals` (LoRA factors, structured LoKr factors, low-rank LoHa) and
`iris_3b.adapters.materialized_deltas` (a LoHa at or above its break-even rank, a LoKr whose
factors are not plain matrices). Each loader prices its contract before reading a weight and
propagates the error, so a spec with no contract fails `load`.

Two request-side rules follow from the contract:

- `validate` refuses the memory controls this route has no mechanism for
  (`GenerationMemory::tile_vae_decode`, `stream_transformer_blocks`, `chunk_attention` without a
  size).
- The request scope refuses a request whose `prompt_batch.len() × count` exceeds the admitted
  `MemoryGeometry::batch`: a prompt batch denoises its rows together, so each prompt is a batch row.

### What bounded attention does to the image

Chunking computes the same attention to GEMM-order f32 rounding. On the FP32 path a bounded render
equals the unbounded one to within one RGB8 level. Under the bf16 policy (and the q8 / q4 tiers, which compute under it)
every projection re-rounds to bf16, so the render is not bit-identical. Declared bound, held on the
miniature snapshot on Candle CPU over three seeds and three binding budgets: no channel moves by
more than **8 RGB8 levels** and the mean absolute change is at most **1.0 level** (observed on arm64
macOS: 2 levels and 0.20). This bounds the miniature; the release weights on CUDA are unmeasured.

Verification: `memory_strategy` unit tests (declaration, admission gates, header pricing held to the
loader's recorded dtypes and installed residuals, registry conformance, load refusals), `nn` unit
tests (no kernel launch exceeds its budget), and `tests/memory_strategy.rs` (a staged selection on a
warm generator loads the two components in staged order and renders the identical image; an admitted
budget reaches the kernels; the bf16 bound above).

## Quantized tiers (sc-26266)

`iris_3b` loads at bf16 and at the `q8` / `q4` tiers (`LoadSpec::quantize`, descriptor
`supported_quants: [Q8, Q4]`), quantized **at load** from the same dense release — nothing
converted is shipped or written, the MLX twin's convention. The tier semantics are the MLX twin's
(see *Quantized tiers* in its `UPSTREAM.md`); only the storage differs:

* **Backbone** (`load_backbone_quantized`, `nn::ProjectionLayout`): every projection at least one
  64-wide group wide is held as GGML `Q8_0` / `Q4_0` (32-wide blocks, 8.5 / 4.5 bits per weight —
  the cost of MLX's group-64 affine scales and biases), quantized on the host straight from the
  checkpoint's f32 values and placed on the device. The 6826-wide SwiGLU down projections are
  zero-padded to 6848 (exact) and packed; an adapted load keeps them dense, as on MLX. The 3-, 12-
  and 16-wide pixel and pooling heads, the norms, the modulation biases and the text position table
  stay dense.
* **Text tower** (`IrisTextEncoder::load_quantized`): every attention/MLP projection of the
  Qwen3-VL language tower is quantized by the shared decoder (`candle_llm` `QuantSpec::q8` /
  `QuantSpec::q4` — `Q8_0`, and `Q4_K` where its 256-wide super-block divides, `Q4_0` otherwise, both
  4.5 bits per weight). The shards are staged on the host and installed on the device tensor by
  tensor, so the dense tower never occupies the device beside the packed one. The tied embedding
  and the norms stay in the tower dtype. A tower whose projection widths are not multiples of the
  32-wide block (the released miniature's 48-wide MLP) has no quantized tier — a typed refusal.
* **No activation is quantized.** Both components dequantize the resident blocks to the activation
  dtype per forward and multiply densely (the shared `DequantDense` forward, sc-7702; on the tower
  `QuantSpec::with_dequantized_forward`). candle's `QMatMul` would quantize the activation to
  per-32 `q8_1` on CUDA, where one outlier channel flushes its block — and the tower's hidden
  states are the conditioning itself.
* A quantized tier runs under the bf16 compute policy only (`Precision::Fp32` + a tier and NVFP4
  are typed refusals, shared by `load` and every contract factory).

Header-priced resident weights of the release (the contract's own pricing, before candle's
512-element CUDA row padding per packed tensor): backbone 5.57 / 2.96 / 1.57 GiB, text tower 7.49 /
4.32 / 2.63 GiB, together 13.06 / 7.28 / 4.20 GiB for bf16 / q8 / q4 (0.558 x and 0.322 x of bf16)
— the MLX twin's figures. Verification: `memory_strategy` unit tests (quantized pricing recomputed
from the headers, adapted pricing, tier handshake, witnessed surfaces), `tests/quant_tiers.rs` on
the miniature (Candle CPU, every tier: measured resident bytes equal to the contract, image /
conditioning distance to the dense tier, the production load at both residency policies) and the
`#[ignore]`d `tests/quant_real_weights.rs` (measured resident ratios on the release; 512² bf16 /
q8 / q4 renders from one seed against the MLX twin's declared tolerance) on the CUDA lane.

## The folded tape (training, sc-25690)

Candle's autograd tape keeps the output of **every** primitive alive until the loss drops: each op
node holds its operands whether its backward needs them or not. The fused inference kernels
(`rms_norm`, `softmax_last_dim`, a grouped convolution) have no backward, so training runs the same
math as chains of composable ops — and each chain's every link is retained. `tape::fold1` /
`tape::fold3` put such a chain on the tape as one node: the value is computed off the tape, and the
node's backward replays the chain for that node alone (fresh leaves at the operands, the chain, a
backward seeded with the node's cotangent).

| folded chain | where | the plain chain kept | the node keeps |
|---|---|---|---|
| SDPA core (scores, ·scale, +mask, softmax's −max / exp / ÷sum), whichever kernel the request scope selects — the i32-overflow guard or the bounded-attention rung, chunks included | `nn::attention` | 5–6 × `[B, heads, N, N]` | its output |
| composable RMSNorm (x², x/rms, ·gain) | `nn::RmsNorm::forward` | 3 activations (f32) | its output |
| a frozen ConvNeXt block of the restoration critic (depthwise 7×7 as 49 shifted slices, norm, MLP) | `train::critic::ConvNextTrunk::features` | ~125 feature maps | its output |
| the critic head's BlurPool sum | `train::critic::depthwise` | 31 feature maps | its output |

**Gradients.** The replay runs the chain's own ops in the chain's own order, and each operand's
contributions land where the plain graph put them (an operand the chain reads twice is passed twice,
in the order the plain backward reaches its two readers; an operand the chain never reads gets a
zero gradient). The gradients are the plain chains' **bit for bit** — asserted on the Candle CPU
device for generation (f32 and bf16 policy, with and without activation checkpointing), the critic's
generator term and head step, the chunked SDPA kernel with grouped-query k / v and a per-row mask,
and a tracked attention under a bounded score budget (`tests/train_tape_memory.rs`); the CUDA
memory-probe lane runs the same comparison on the device at bf16, critic included
(`folds_are_bit_identical_on_the_build_device`).

**Cost.** One extra evaluation of each folded chain per backward, and one copy of its output: a
training step is 20–30 % slower where nothing spilled before (Iris-3B on CUDA, batch 1: 512²
generation 3.0 s → 3.6 s, 1024² fully checkpointed 6.9 s → 9.0 s). Inside an activation-checkpointed
region a folded chain is evaluated three times per step: the region's tape-free forward, its replay
on the tape, and the node's own replay during that replay's backward.

**Also.** An activation-checkpointed step's tape-free backbone holds the tracked backbone's
compute-dtype casts, detached (`Checkpoint::loaded`), instead of casting the f32 masters again (half
a backbone resident for the whole step).

## Activation checkpointing of the depth and restoration trainers (sc-25690)

`TrainingConfig::gradient_checkpointing` selects `full` checkpointing on the Candle depth and
restoration trainers: the generation trainer's regions (one per trunk block and per PiT pixel
block, `train::model::grads_under_ac`) around the widened depth backbone, and around the generator
in the restoration generator step (the LPIPS and critic terms sit downstream of the regions and
stay on the plain tape; the critic step needs no generator gradient). Default off; a checkpointed
run is the plain run to the bounds of the generation trainer's (f32: gradients ≤ 1e-6 of the tensor
peak for depth, ≤ 1e-5 for restoration, whose adversarial term is O(10²) on the miniature and
reaches the text embedder as a difference of large partials — 3.8e-6 measured, 1.6e-8 with the L2
term alone; bf16: the same loss bit for bit and the declared noise bound of the generation row).
A checkpointed backward drops the loss graph as soon as its plain backward has run
(`backward_segmented` owns the loss): everything downstream of the last regions — the backbone's
head, and here the LPIPS and critic tapes — is off the tape before the first region is replayed
rather than resident under every replay.
Upstream ships neither task with checkpointing; it changes memory and compute only, so it is no
part of a run's identity. The MLX depth and restoration trainers take the same option (MLX's
checkpoint transform per region; see the section of the same name in the MLX crate's `UPSTREAM.md`).
