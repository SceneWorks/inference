# Native memory admission

Both native providers estimate load and request memory before allocating checkpoint tensors or
vision/model workspaces. These are conservative admission estimates, not measured peaks or a
claim that the architectural context window fits on every device. Architectural context limits
remain unchanged.

`SCENEWORKS_LLM_AVAILABLE_MEMORY_BYTES` optionally supplies an operational byte budget. It caps
fresh available capacity; it cannot enlarge it. Invalid or non-Unicode values fail closed. Missing
capacity also fails closed, even when a budget was supplied. CPU capacity comes from current host
availability (Linux MemAvailable, macOS reclaimable/free vm_stat pages, Windows FreePhysicalMemory).
Qwen3.8 and Prism/Bonsai are accelerator-only and Candle rejects those snapshots on CPU before
checkpoint inventory or tensor allocation; this host-capacity behavior remains available to other
model families.
MLX uses that unified-memory capacity. Candle CUDA applies the operational cap to VRAM and reads
current free memory from the loaded CUDA device's own context; host RAM is never substituted for
VRAM, and a launch-time GPU snapshot never substitutes for current post-load free VRAM. CUDA load
admission checks host staging independently against current host capacity. Request estimates apply
to *additional* request memory against current free capacity, so loaded weights are not charged
twice.

Load upper bounds use checkpoint files without evaluating tensor payloads. Dense Candle CUDA's
pinned loader reads one safetensors shard into a host buffer, copies its
tensors directly to CUDA at their stored dtype, then drops that buffer before reading the next
shard. Its host bound is therefore the largest shard rather than two complete checkpoint copies.
Qwen3.8 is BF16 and same-dtype model construction shares the loaded storage. CUDA device admission
reserves the complete payload plus 25 percent temporary space, covering construction-time casts
such as the F32 vision tower. Candle external projectors reserve four times their stored payload.

MLX safetensors follow a different allocation path. In the pinned mlx-rs dependency (MLX
0.32.0), upstream `mlx/io/safetensors.cpp` creates lazy Load arrays from headers.
`mlx/backend/common/load.cpp` allocates the output buffer and `pread`s directly into it — owned
Metal shared buffers, not a mapping, so source pages are never reclaimable by the OS while an array
holds them. `mlx/ops.cpp::astype` returns the same array when its dtype already matches. Rust
`Weights` and model clones share array handles. Thus BF16 language weights and vision weights do
not need separate full host and GPU copies.

### MLX safetensors allocation order (sc-24446)

Read from `mlx-llm` (`Weights::from_dir`, `Weights::materialize_groups`, `CausalLm::build_lazy` /
`LayerPlan::load`, `Qwen35Model::build_lazy` / `build_ffn`, `Gemma4Mm::from_weights`,
`Projection::load`, `SwitchLinear::{load,stack}`) and MLX 0.32.0 (`affine_quantize`,
`fast::Quantize::eval_gpu`, the Metal allocator and its buffer cache):

1. Construction is lazy. Every projection, cast, split and quantization is an unevaluated graph
   node over the loaded source arrays; nothing is allocated yet.
2. The provider then **materializes the decoder group by group** (`Weights::materialize_groups`;
   one group per decoder or MTP layer, plus the arrays outside the layer stack —
   `CausalLm::param_groups` / `Qwen35Model::param_groups`). For each group it reads the group's
   pending safetensors loads on the CPU stream (`eval_pending_loads`, so no GPU op waits on the
   disk — sc-24245), checks every source it read for a coherent GPU view (sc-22414), evaluates the
   group's conversions on the GPU, drops the `Weights` map's handles on those sources and clears
   MLX's buffer cache. A source consumed into a different array (a quantized projection, a BF16
   cast, a `1 + w` norm, a stacked expert bank) is returned to the system before the next group
   reads its own; a source the model keeps as stored (a BF16 matrix) stays as the model's array.
   Every accessed source is still read and verified before the load returns, and no array is
   left lazy, so the first request neither reads a weight nor derives one inside its command
   stream. A source no group consumed (`Materialized::leftover`) would stay resident outside the
   priced bound, so the provider **refuses the load** (`materialize_decoder`); every
   architecture's fixture loads through the provider path to pin each enumeration
   (`every_architecture_loads_through_the_provider_with_every_source_consumed`,
   `gemma4_loads_through_the_provider_with_every_source_consumed`, the MoE suite, and the Qwen3.5
   / Prism provider tests).

   **Pacing.** The Metal driver returns a released buffer to the system asynchronously — measured
   0.1–1.25 s after `clear_cache` — so a fast load could read group after group while earlier
   groups' sources still count against the process. Before each group reads, its
   `phys_footprint` is held (waiting at most 3 s) to the start value plus the arrays built, the
   previous group's consumed sources and a 128 MiB slack: at most one released group is ever
   outstanding (`a_group_waits_for_the_driver_to_return_the_previous_groups_sources`). MLX's own
   cache is empty after every group (`Materialized::cache_after_groups`, pinned 0).
3. Vision towers and Gemma 4's media embedders keep the earlier order: their constructors verify
   every source up front and keep it beside their (lazily derived) arrays.
4. A caller that keeps its own `Weights` map (`CausalLm::from_weights_with`,
   `Qwen35Model::from_weights_with`, the text encoders in `mlx-gen`) verifies every source up front
   and then evaluates every derived array group by group before the constructor returns — the
   same derived arrays, materialized at load rather than by the first forward; the map it owns
   keeps the sources until it drops them.

Greedy outputs are unchanged: the order only moves when each array is computed (the
`materializing_group_by_group_consumes_every_source_and_decodes_identically` suite decodes every
MoE family and layout, dense and at load-time Q8 / Q4, bit-identically to the caller-owned-map
constructor).

#### The fused Qwen3.6 expert bank (sc-24446)

Qwen3.6-35B-A3B ships each layer's routed experts as one fused `experts.gate_up_proj`
`[experts, 2·moe_inter, hidden]` (gate rows ‖ up rows). The loader used to cut it into gate and
up banks with a gathered `take` — a second copy of ~44 GB of BF16 expert weights. The MoE block
(`primitives::moe::SparseMoe::fused`) now keeps it fused: one `gather_mm` / `gather_qmm` over the
fused bank, its `[.., 2·moe_inter]` output split per token into gate and up. Quantizing the fused
bank produces exactly the quantized rows of the two halves (groups run along `hidden`), and the
gathered matmul over the fused bank is bit-identical to the two over the halves — pinned dense
(f32 and BF16) and at Q8 / Q4 on the unsorted decode and sorted prefill shapes, and at model level
against the per-expert layout of the same weights (which stacks exactly the old split banks),
MTP-on-MoE included. Per-expert checkpoints (the BF16 Qwen3.5 release, prepared quantized
snapshots, Qwen2-MoE, DeepSeek-V2) keep the separate gate and up banks.

### The bounds (`mlx-llm/src/load_memory.rs`)

**Materialize-at-load bound** (`materialized_bound`): `resident + max(window) +
max(consumed sources of a group) + 128 MiB` plus the host heap. While group `g` converts, memory
holds the arrays already built, `g`'s sources and transients, `g`'s outputs and — under pacing — the
previous group's consumed sources the driver has not yet returned, plus the pacing slack:

| Term | Charge | Why |
| --- | --- | --- |
| resident, quantized at load | `n·bits/8 + 4·n/group` + three pages | packed words + BF16 scales + biases (the source is released) |
| resident, per-expert | the stacked copy in its loaded form + a page | `SwitchLinear::stack` |
| resident, vector | `max(stored, 2n)` + a page | `1 + w` norms, `exp(A_log)` |
| resident, wider float | `2n` + a page | BF16 cast (the source is released) |
| resident, otherwise | stored + a page | BF16 matrices and stored quantized parts are the model's arrays |
| resident, media | stored + a page + every derived term below | vision / audio constructors keep their sources |
| window of a group | its consumed sources; a BF16 cast ahead of a quantize; each expert's loaded form ahead of its stack; `4n` norm intermediates; checksum widening | what group `g` reads and converts |

A decoder tensor's group is its `…layers.{i}.` prefix (decoder and MTP layers alike), or the
non-layer group. Load-time quantization never touches embeddings, the LM head, stored quantized
parts, or the matrices the decoders keep dense (MoE routers, the shared-expert gate, Qwen3.5's
`in_proj_a/b`).

**Earlier bound** (`derived_bytes`): `payload + derived`, the verify-everything-then-derive
order's peak — the stored payload once, plus per stored tensor of `n` elements (header only;
"language" = not under `model.visual.` / `vision_tower.`):

| Term | Charge | Why |
| --- | --- | --- |
| page rounding | 16 KiB per stored tensor and per derived array | Metal buffers round to 16 KiB pages |
| BF16 cast | `2n` for a language F16/F32/F64 tensor | `as_dtype(BF16)` allocates; BF16 is shared |
| vector intermediates | `4n` for a language tensor of rank ≤ 1 | `1 + w` norms, `exp(A_log)`, signs |
| vision patch transpose | `4n` for `*patch_embed.proj.weight` | contiguous channels-last copy |
| checksum widening | `4·bytes` when `bytes % 4 ≠ 0` | the load-boundary checksum widens a byte view to `u32` |
| gathered split | `2n` for a rank-3 `*.pos_embedding` | `take_axis` copies (Gemma 4 row/column tables) |
| load-time Q4/Q8 | `n·bits/8 + 4·n/group` for every language float matrix whose input width divides the group, except embeddings and the LM head | packed words + BF16 scales + biases |
| expert stacking | one more copy of each `*experts.{e}.*` tensor in its loaded form (quantized, BF16, or the stored packed part) | `SwitchLinear::stack` (sc-24440) concatenates the bank |

The materialize-at-load order never holds more than the earlier bound beyond the pacing slack
(each group's sources are a slice of the payload, its outputs and transients a slice of the
derived set — pinned on a fixture against MLX's active peak and the exact `phys_footprint` peak,
`the_materialized_bound_covers_the_provider_load_order_on_a_fixture`), so the earlier bound stays
an upper bound for every cell its probes verified.

Beside the arrays, a load builds host heap MLX does not see: the parsed tokenizer (measured at
12–17× the `tokenizer.json` bytes for Llama 3.2, Qwen3 and Gemma 2), the chat template, the lazy
graph's nodes and the Metal pipeline states its checksum and first forward compile. And the Metal
driver takes back the resources it returned while the process idled: 112 MiB for one tiny op,
129 MiB for the whole load of a 100 KB snapshot (exact kernel peak), 137–170 MiB for a fixture's
one-token request (`MLX_DRIVER_WAKE_BYTES` = 256 MiB). Every safetensors load is charged
`24 × tokenizer.json bytes + 64 MiB + 256 MiB` for them. The Metal device and MLX's kernel library
(~170 MB) are process-wide one-time costs and are not charged per load.

**What a load is charged**, by the probe evidence behind its **cell** — the snapshot's top-level
`model_type` (exactly: Mistral or dense Qwen2 are not Llama, though one decoder serves them), the
conversion (dense, load-time Q4, load-time Q8, stored-quantized) and the one floating dtype every
decoder matrix is stored in (`LoadCell`):

1. a cell in `MATERIALIZED_VERIFIED` (each backed by an exact-peak probe of the
   materialize-at-load order): `materialized_bound` + host heap;
2. a cell in `EARLIER_VERIFIED` (each backed by a covered probe of the earlier order): the earlier
   bound + host heap;
3. anything else: the two-copy bound **raised to the earlier bound when that is larger**, plus
   the host heap.

| Cell | Charged |
| --- | --- |
| `llama`, `qwen3`, `gemma2`, `gemma4_unified`, `qwen3_5_moe` × dense / load-time Q4 / load-time Q8, BF16-stored; `qwen3_5` × load-time Q4 / Q8 | `materialized_bound` + host (exact-peak probes, 2026-10-01, all covered) |
| `qwen3_5`, `qwen3_5_text`, `qwen3_vl`, `prism_hadamard_qwen35` unquantized (dense or stored-quantized, any stored dtype) — the cells `main` verified | exactly `main`'s charge (`payload + casts`) + host; the lower of that and the materialized bound where the exact probes cover the cell (`qwen3_5` dense BF16) |
| every other cell — stored-quantized snapshots of other families (incl. a prepared Qwen3.6 Q4: probed at 22.72 GB but its 9.4 MB manifest is not committed), Mistral, dense Qwen2, F16/F32 checkpoints of a probed family, a config without `model_type` | `max(2×, earlier)` + host |

The host term (`24 × tokenizer.json + 64 MiB + 256 MiB` driver wake) is an allocation `main` did not
price; the exact probes measured it (Qwen3.8-27B BF16 peaked 0.35 GB above `main`'s whole charge),
so a `main`-verified cell is never charged more than `main` charged for its arrays, and never
refused where `main` admitted it except inside that host term
(`main_verified_cells_are_never_charged_more_than_main_charged_them`, on the committed Qwen3.8
manifest and synthetic Qwen3-VL-32B / `qwen3_5_text` / stored-quantized / Prism headers).

On this 128 GiB host (~97 GB available idle) every probed cell is admitted — charges in the
measured-probes table below; the largest, Qwen3.6-35B-A3B BF16, at 72.7 GB.

Header-only bounds for the probe targets (arrays only, recomputed from the committed manifests;
host heap excluded; GB):

| Snapshot | Payload | BF16: earlier / materialized | Q4: earlier / materialized | Q8: earlier / materialized |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B (`995ad96e`) | 71.90 | 71.93 / 72.07 | 91.34 / 25.91 | 108.58 / 43.14 |
| Qwen3.8-27B (`1d4bf0f2`) | 55.56 | 55.60 / 55.73 | 69.54 / 21.69 | 81.93 / 34.07 |
| Gemma 4 unified enhancer (`791ef617`) | 23.92 | 23.96 / 24.08 | 30.12 / 9.39 | 35.60 / 14.84 |
| Qwen3-8B (`b968826d`) | 16.38 | 16.39 / 16.52 | 20.30 / 7.32 | 23.78 / 10.79 |

The Qwen3.6 earlier bounds already exclude the gate/up split copies the fused bank removed (they
were ≈ 116 / 135 / 153 GB with them).

For GGUF it counts the source mapping, retained affine words and both intermediate/final scales,
dense conversion arrays, and the largest per-tensor host conversion/reordering buffers. Projectors
are priced separately from their headers because MLX decodes them to F32. This remains conservative
when mapped source pages are reclaimed; it does not price compact language weights as dense 27B.

A load the bound refuses fails with `load admission: loading this model requires an estimated …
bytes of resident weights and load-time conversions but only … bytes are available` (both
backends, `core_llm::admit_load_memory`) — never the request-side "reduce prompt/media length or
max_new_tokens" remedy, which cannot help a load.

The pinned parent safetensors bound is 55,597,679,416 bytes of arrays (sc-24446 page rounding
included; 55,572,906,808 before) plus its 374,532,544-byte host heap
(`24 × 12,809,320` tokenizer bytes `+ 64 MiB`). GGUF language bounds are 17,061,529,952 (PQ2)
and 15,802,009,952 (PTQ1), plus 2,868,438,080 for BF16 projector or 2,566,539,200 for Q8 projector. Terminal admission adds
its separately named operational reserve to these bounds. The ignored
`pinned_header_only_load_admission_bounds` audit compares native estimates (optionally with a
`quantize` of `q4`/`q8`) against a JSON list of pinned paths and expected bounds without loading a
model.

### Measured probes (sc-24446)

2026-10-01, Apple M5 Max (Mac17,6), 128 GiB, one model at a time under the memory guard
(`HARD_GB=80`), earlier (verify-everything-then-derive) load order, `phys_footprint` sampled
in-process every 100 ms over a baseline taken after the Metal device and MLX's kernel library
initialized.

**One method for every cell** (`load_memory::tests::Measured::peak`): the load's share of the
peak is `max(footprint growth through the end of the load, first one-token request's peak
growth − that request's own working set)`, the working set measured by a second identical
request on the materialized model. (These probes recorded the working set as MLX's active peak,
not its footprint — which can only overstate the load's share.)

**Margin** (`Measured::margin`): 0.5 % of the peak for run-to-run variation, plus — for a sampled
probe — everything the sampler can have missed: the arrays the first request derived, which a
`clear_cache` between two 100 ms samples can hide (the 0.5 s guard under-read the Qwen3.8 Q8 load
by 5 GiB). So no sampled load-time probe can verify its cell: the margin it needs is as large as
its quantized arrays.

Every row is **recomputed from code on every test run**: its estimate from the committed
header-only manifest of the pinned snapshot (`crates/llm/testdata/load_admission/*.json` — config,
every tensor's name / dtype / shape, shard and tokenizer sizes; no weights), its peak and margin by
the rules above (`verified_cells_are_exactly_the_probed_and_covered_ones`). Estimates include the
256 MiB driver wake.

| Snapshot | Conversion | Estimate (GB) | Uniform peak (GB) | Required margin (GB) | Covered | Raw first-request growth / request MLX working set (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 2 2B-it | BF16 | 5.992 | 5.524 | 0.035 | yes | 9.052 / 3.528 |
| Gemma 2 2B-it | Q4 | 7.134 | 6.587 | 1.182 | no | 9.013 / 2.426 |
| Llama 3.2 1B Instruct | BF16 | 3.028 | 2.645 | 0.016 | yes | 2.680 / 0.035 |
| Llama 3.2 1B Instruct | Q4 | 3.578 | 3.024 | 0.568 | no | 3.065 / 0.041 |
| Llama 3.2 1B Instruct | Q8 | 4.064 | 3.685 | 1.057 | no | 3.723 / 0.038 |
| Qwen3-1.7B | BF16 | 4.058 | 3.612 | 0.025 | yes | 3.658 / 0.046 |
| Qwen3-1.7B | Q4 | 4.854 | 4.418 | 0.825 | no | 4.456 / 0.038 |
| Qwen3-8B | Q4 | 20.912 | 19.875 | 4.021 | no | 19.962 / 0.087 |
| Qwen3-8B | Q8 | 24.385 | 23.906 | 7.514 | no | 23.988 / 0.082 |
| Qwen3-8B | BF16 | 17.001 | 16.557 | 0.093 | yes | 16.610 / 0.053 |
| Gemma 4 unified (LTX-2.5 enhancer) | BF16 | 25.065 | 25.066 | 0.163 | **no — above its estimate** | 30.482 / 5.416 |
| Gemma 4 unified (LTX-2.5 enhancer) | Q4 | 31.231 | 29.925 | 6.354 | no | 34.391 / 4.466 |
| Qwen3.8-27B | Q4 | 70.185 | 69.477 | 14.326 | no | 69.713 / 0.236 |
| Qwen3.8-27B | Q8 | 82.573 | 80.640 | 26.770 | no | 80.870 / 0.230 |

#### Exact-peak re-probes of the current build (2026-10-01)

Every cell re-measured on the current build — group-by-group materialize-at-load with pacing,
BF16 GeGLU for LLM decode — one model at a time under the 80 GB guard (no cap hit), with the
kernel's **exact** footprint maxima (`ri_interval_max_phys_footprint`,
`tests/common/footprint.rs`) over a driver-settled baseline, so the rows need only the 0.5 %
variation margin. Every row is a `Measured { Order::Materialized, Sampling::Exact }` entry; all
21 are covered, and exactly their cells form `MATERIALIZED_VERIFIED`. "Request" is the exact
footprint growth of a second one-token request (driver wake, cache and host heap included).

| Snapshot | Conversion | Charged (GB) | Exact load peak (GB) | Slack beyond margin (GB) | Request (GB) | Main charged (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 3.2 1B | BF16 | 3.162 | 2.743 | 0.406 | 0.170 | 4.943 |
| Llama 3.2 1B | Q4 | 2.010 | 1.463 | 0.540 | 0.226 | 4.943 |
| Llama 3.2 1B | Q8 | 2.497 | 1.951 | 0.536 | 0.226 | 4.943 |
| Qwen3-1.7B | BF16 | 4.190 | 3.747 | 0.425 | 0.218 | 6.882 |
| Qwen3-1.7B | Q4 | 2.372 | 1.821 | 0.542 | 0.236 | 6.882 |
| Qwen3-1.7B | Q8 | 3.077 | 2.526 | 0.539 | 0.236 | 6.882 |
| Gemma 2 2B-it | BF16 | 6.124 | 5.636 | 0.459 | 0.188 | 10.457 |
| Gemma 2 2B-it | Q4 | 3.532 | 2.888 | 0.629 | 0.195 | 10.457 |
| Gemma 2 2B-it | Q8 | 4.544 | 3.900 | 0.624 | 0.195 | 10.457 |
| Qwen3-8B | BF16 | 17.132 | 16.694 | 0.355 | 0.225 | 32.763 |
| Qwen3-8B | Q4 | 7.928 | 7.090 | 0.802 | 0.269 | 32.763 |
| Qwen3-8B | Q8 | 11.401 | 10.563 | 0.785 | 0.254 | 32.763 |
| Gemma 4 unified enhancer | BF16 | 25.191 | 24.570 | 0.497 | 0.248 | 47.839 |
| Gemma 4 unified enhancer | Q4 | 10.500 | 9.086 | 1.368 | 0.295 | 47.839 |
| Gemma 4 unified enhancer | Q8 | 15.950 | 14.415 | 1.463 | 0.287 | 47.839 |
| Qwen3.8-27B | BF16 | 56.216 | 55.920 | 0.016 | 0.359 | 55.573 |
| Qwen3.8-27B | Q4 | 22.337 | 20.304 | 1.931 | 0.382 | 111.126 (refused) |
| Qwen3.8-27B | Q8 | 34.712 | 32.868 | 1.680 | 0.378 | 111.126 (refused) |
| Qwen3.6-35B-A3B | BF16 | 72.711 | 72.265 | 0.085 | 0.241 | 143.808 (refused) |
| Qwen3.6-35B-A3B | Q4 | 26.552 | 23.362 | 3.073 | 0.251 | 143.808 (refused) |
| Qwen3.6-35B-A3B | Q8 | 43.783 | 40.662 | 2.918 | 0.245 | 143.808 (refused) |

The tightest are the two large BF16 loads (Qwen3.8 — charged `main`'s charge plus the host heap,
the lower bound — by 16 MB, Qwen3.6 by 85 MB beyond the 0.5 % margin): BF16 keeps every source as the model's own array, so the bound is the payload plus a few
hundred MB and a sub-percent run-to-run variation is all the headroom there is to need.

### One-token request working set (sc-24446)

The second-request MLX working sets above were not priced by request admission: its estimate for
those one-token "Hi" requests was 6–9 MB on Llama 3.2 1B / Qwen3-1.7B / Qwen3-8B (measured
35–87 MB) and 14–31 MB on Gemma 2 / the Gemma 4 enhancer (measured 2.4–5.4 GB). The causes,
now priced on top of the decoder's own workspace (`provider.rs`), are pinned against the recorded
probes (`the_request_estimate_covers_the_recorded_one_token_working_sets` — MLX active peaks, the
only figure those probes kept; their Gemma rows ran the `f32` GeGLU) and against fixture
measurements taken **both** as the exact, driver-settled `phys_footprint` peak growth (MLX's
cache, the host heap and the driver's wake included) and as MLX's active peak, the estimate
covering the larger (`a_one_token_request_working_set_is_priced_under_either_activation_dtype`).
The exact footprint request working sets of the 2026-10-01 re-probes — 0.17–0.38 GB on every
BF16-GeGLU / SwiGLU path, Gemma included, and 2.67–6.41 GB on Gemma's `f32` GeGLU path — are
pinned too (`the_request_estimate_covers_the_exact_footprint_working_sets`). The terms:

- **Gemma promoted its weights every forward.** `mlx_rs::nn::gelu_approximate` builds its
  constants as `f32` arrays, so Gemma's GeGLU turned a BF16 input into `f32` and the residual
  stream, every later projection's input and the final hidden state stayed `f32`. Each dense matmul
  then materialized an `f32` copy of its BF16 weight — the LM head's alone is vocabulary × hidden ×
  4 bytes: 2.36 GB on Gemma 2, 4.03 GB on the 262K-token Gemma 4 — and each quantized matmul an
  `f32` copy of its scales and biases. That is now an **activation-dtype policy**
  (`mlx-llm/src/primitives/activation.rs`, the one place every tanh-GELU asks): LLM decode paths
  (`ActivationRole::LlmDecode` — every provider load, draft models, the LTX-2.5 prompt enhancer,
  StarVector's GPTBigCode / StarCoder2 decoders) return the GELU in its input's dtype, so the
  stream stays BF16 and promotes nothing; the LTX-2.5 text encoder (`LtxTextEncoder`, its
  real-weight goldens captured with `f32` activations) and vision towers / projectors
  (`VisionEncoder`) keep `f32`. For a role that keeps `f32`, request admission prices the
  promotion (`CausalLm::activations_promote`): the LM head's copy plus MLX's evaluation window (ten
  committed command buffers and the one encoding) × (the 50 MB per-buffer cap + the largest
  promoted projection); for every other decoder that term is zero.
- **MLX runtime terms the shared estimates do not see** (generic decoders; the Qwen3.5 hybrid's
  contract already prices its own): K/V held in whole 256-position blocks over every position the
  request can write — committed prompt and generation, an MTP verify width, and a prompt-lookup /
  draft overshoot crossing into a new block (`kv_block_padding_covers_committed_and_verify_positions`,
  `a_lookup_overshoot_into_a_new_block_charges_the_block`); every other layer's temporaries in
  flight across MLX's window, capped at 11 × (50 MB + the op that crosses each buffer — a long
  eager prefill's full `prompt × prompt` score block per head, or a `prompt × inter` MLP
  activation; `the_in_flight_window_prices_each_buffers_crossing_op`); a page of rounding per op
  output in it.
- **The driver's wake** (256 MiB, every request and every contract): after the process idles, the
  Metal driver's returned resources are taken back by the next evaluation — 137–170 MiB measured
  for a fixture's one-token request.

#### Activation-dtype parity gate (sc-24446)

A role switches to the activation dtype only behind a fixture gate
(`primitives::activation::parity`): the decoder greedy-decodes 16 tokens on the `f32` path, the
BF16 path is teacher-forced along them, and every step must hold `|Δlogit| ≤ 2⁻⁵ · max(|logit|,
1)` (eight BF16 ULPs at the logit's magnitude: one BF16 rounding of each GeGLU output, 2⁻⁹
relative, then a BF16 stream, over 2–4 layers and the head) with greedy tokens agreeing wherever
the `f32` top-2 margin exceeds twice that step's `|Δlogit|`. Measured worst `|Δlogit|` relative:
Gemma 2 1.06e-2, Gemma 4 1.08e-2 (0 flips each), GPTBigCode 8.7e-3 (1 flip, on a margin inside
the rule), StarCoder2 8.3e-3 (0 flips). A different activation (SiLU for GELU) lands at
7.7e-2–1.4e-1 and is caught; a near-identical one (the exact erf GELU) is not caught by the gate —
`gelu_tanh_is_the_tanh_approximation` pins the tanh formula directly instead.

Existing goldens: none moved past its tolerance; none was regenerated. The Gemma 4 decoder goldens
(`tests/gemma4_decoder.rs`, an `f32` NumPy oracle) load the fixture in the text encoder's role
(`f32` GeGLU: 5.8e-3 hidden / 3.0e-3 logits against the 2e-2 budget, ~3.4× headroom). The BF16
path measures 1.3e-2 / 1.1e-2 and is held to a **derived** budget, `BF16_ABS_TOL = 2e-2 + 2⁻⁵`
(triangle inequality: its distance to the oracle is at most the parity gate's bound on its
distance to the `f32` path plus the `f32` path's budget) — ~3.9× headroom, with the narrowest
mutation still 2.1× outside it (`the_reference_holds_under_either_activation_dtype`). `architecture_forward`'s Gemma 2 golden is
not numerically pinned on MLX (eager-attention drift; shape, finiteness and cache growth only).
The LTX-2.5 text encoder stays on `f32` until its real-weight goldens
(`ltx_2_5_te_connector_inputs`, `ltx_2_5_te_tier_quality`) are run with BF16.

Decode speed on real weights (128 greedy tokens after a short prompt, release build, idle GPU,
2026-10-01), `f32` GeGLU → BF16 GeGLU: Gemma 2 2B-it BF16 19.4 → 86.5 tok/s (4.5×), Q4 55.3 →
152.0 (2.7×); the Gemma 4 unified enhancer BF16 4.1 → 20.3 tok/s (5.0×), Q4 25.1 → 50.3 (2.0×).
The `f32` path's one-token request footprint was 3.81 GB (Gemma 2 BF16) / 2.67 GB (Q4) and 6.41 GB
(enhancer BF16) / 4.66 GB (Q4); the BF16 path's 0.19–0.30 GB. (Tiny-fixture figures from before
the re-probes: 1.4× BF16, 1.2× Q4.)

Request bounds include expanded visual tokens, eager attention scores/mask/softmax, full-length
K/V storage, recurrent state, projection/MLP/logit buffers, MTP cache/rollback and draft-width
buffers, RGB preprocessing, unmerged vision-patch attention and vision MLP workspaces. Candle prices
every workspace element at four bytes. Since sc-20671, MLX prices K/V, activations and logits at the
decoder's compute width (BF16) and eager attention scores at F32. Prism/Bonsai stays at four bytes
until its frozen allocator peaks are re-taken on the BF16 path; see
`docs/architecture/SC_20671_DENSE_BASELINE_MODEL_CONTRACT.md`. Checked
arithmetic rejects overflow. Visual geometry is computed before visual encoding. A low budget
therefore rejects a request inside the architectural token window without invoking the decoder.

The native composition fixtures exercise text, image and video routes for generic Qwen3-VL and
Qwen35 decoders, including reasoning, JSON, MTP and stops. Isolated subprocess tests verify that a
one-byte budget rejects both load and within-window visual requests, and malformed budgets fail.
The terminal campaign must still record actual accepted/rejected lengths and measured residency;
these deterministic guards do not replace that hardware evidence.

## Compressed KV cache (sc-20682)

A request opts into the compressed KV cache with `TextLlmRequest::kv_compression`
(`KvCompressionPolicy::Off` by default). MLX decides the request's cache before admission
(`plan_kv_cache` against `core_llm::KV_COMPRESSION_QUALIFICATIONS`), so the estimate prices the
cache the request actually runs on: the dense K/V term for every dense plan, and for a qualified
plan `core_llm::compressed_kv_cache_bytes` in its place. That size is derived from the
`KvCompressionFormat` (key/value code bits, group size, an f16 scale and zero per group), not
from a fixed ratio, at the cache's block-rounded capacity of prompt plus `max_new_tokens`:

* resident: per layer and KV head, packed K codes `⌈C/G⌉·G·D·bits_k/8`, K scale/zero
  `⌈C/G⌉·D·4`, V codes `C·⌈D·bits_v/8⌉` and V scale/zero `C·⌈D/G⌉·4`, plus one group of dense K
  and V residual rows at the compute width (`2·G·D·element_bytes`). `C` is the token count
  rounded up to the 256-position growth block and `G = 32`.
* transient: `min(layers, 11)` layers' packed arrays at full capacity plus one block (a block
  growth's pre-growth copy, or a group flush's output, beside the live arrays: one per in-flight
  MLX evaluator buffer), every layer's residual rows again (a step's rollback point), and every
  layer's fused-reader split-KV scratch (`Hq · 128 splits · (D + 2) · 4` bytes for the one-token
  decode dispatches a product generation issues; its prompt step attends through dense SDPA),
  plus one layer's dense prompt K/V (`P·Hkv·D·element_bytes·2`). The prompt step evaluates each
  layer's packed store and attention output before building the next layer; left as one lazy
  graph, every layer's dense prompt K/V stayed resident beside the packed store (measured 1.88x
  the estimate on a 28-layer, 8×128-KV-head decoder, above the same request run dense).

Every dense MLX estimate (a dense plan, an un-opted request, a dense re-admission) prices the
dense cache the same way: its block-rounded buffers plus `min(layers, 11)` layers' pre-growth
buffers, which a block growth holds beside their successors until the evaluator releases them
(a dense decode across growth blocks peaked 1.30x the unrounded K/V term without this).
`admission_estimates_cover_the_measured_peak_compressed_and_dense` holds the MLX allocator's
measured peak at or below the estimate for both cache kinds through prefill and a decode across
growth blocks.

For K8V8 the resident term is `2.25·D` bytes per layer, head and token against `4·D` for BF16
dense. A shallow decoder can price above dense once the transients are added; a qualified request
is still priced compressed, because that is the cache it runs on.

A compressed plan whose cache selection then refuses the fused reader runs dense from the start,
so the provider admits the request again at the dense estimate before any K/V exists. A compressed
generation that later transitions to dense (a reader fault over resident history) admits that
transition against fresh capacity before reconstructing: the larger of the reconstruction (every
layer's dense K/V at block capacity plus one layer's Float32 dequantization) and the dense cache
the rest of the generation grows to (block-rounded, with its window of growth copies). It is
admitted even when no history is resident to rebuild, because the rest of the generation still
runs dense. A refusal fails the generation with the same typed
`RequestResourceExhausted` rather than oversubscribing memory.

Every generation reports its cache on `TextLlmOutput::kv_cache` (`KvCacheReport`):
`format_version` (`KV_CACHE_FORMAT_VERSION`), `format` (`group-affine-k8v8`, or none when it ran
dense throughout), `fallback` (a stable `KvCacheFallbackReason::id`; none exactly when it ran
wholly compressed), `detail` (the backend's operation and reason words, never prompt content) and
`counters` (fused attention calls, dense fallback events, full-cache dequantizations, retained
compressed bytes). SceneWorks records this report on its prompt-refine job result and telemetry.
