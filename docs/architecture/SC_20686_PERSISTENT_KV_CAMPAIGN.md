# SC-20686 persistent K/V campaign transport

The campaign has two measurement lanes, selected by the safety policy's backend. The **Metal lane**
(`darwin-mlx` → `mlx-metal`) measures the MLX providers SceneWorks runs on Apple Silicon — the Mac
product path. The **CUDA lane** (`linux-cuda`/`windows-cuda` → `candle-cuda`) measures the Candle
providers. One sealed bundle is exactly one lane; see [Metal lane](#metal-mlx-lane) below.

The campaign adapter runs the one registered FLUX.2 Klein edit route and all five registered Wan
routes on both lanes (plus the MLX-only FLUX.2 Klein kv-edit route on the Metal lane). For every normal and cancellation arm it creates a private `events.jsonl` file, passes that
path to the product entrypoint with `--sc20686-events`, and seals the exact event transcript as a
separate bundle artifact. Provider stdout and stderr are retained only as diagnostics: progress
output, including carriage-return updates, is never parsed as campaign evidence.

Every child process runs inside its own adapter-owned `sealed-run` directory. The adapter passes an
absolute `--out` below that directory, so images and video frames cannot escape the run closure via
an entrypoint default. FLUX writes `media.png`; Wan writes frames below `media`. Before deleting the
private run directory, the publisher copies every normal-arm output into the final campaign bundle.
Each run has a canonical media manifest that preserves output kind, relative path, byte count, and
content hash; both that metadata hash and every media content hash are bound by the row and campaign
receipts. Cancellation arms must seal an `absent` manifest and are rejected if they leave partial
media behind. The event transcript and media output therefore share one isolated parent, while the
final evidence remains independently reproducible after that private directory is removed.

## Sealed provenance and snapshot layouts

Invoke the adapter with `--inference-revision <40-hex-commit>` for the initial
capture. It verifies the initial checkout, freezes that revision as provenance,
passes it as `--sc20686-source-ref`, and requires observer metadata to reproduce
it. A restart reads the sealed resolved inputs from `--resume-dir` and rechecks
the actual entrypoint binaries, model snapshots, route inputs, coverage, source
map, and adapter bytes; Git HEAD movement alone does not invalidate an otherwise
identical completed arm. The repository revision is never inferred from a model path.

Both single and matrix launch modes require `--safety-policy` and an absolute
external `--resume-dir`. The policy is strict schema version 1 with backend
`darwin-mlx`, `linux-cuda` or `windows-cuda` matching the execution host; it sets positive deadline, poll, termination grace, host reserve, child
footprint cap, stdout/stderr/event caps and, for the CUDA backends, the selected CUDA GPU UUID, GPU-free reserve,
and child GPU cap. A resume directory is bound to the lane it was captured for. Each validated normal/cancel arm is preserved in a sealed
unit with observer events, bounded logs, generated media or verified absence,
process samples, supervisor exit and cleanup, and exact campaign identity. A
later watchdog failure leaves the bundle incomplete; it cannot serve as the
product cancellation oracle or a terminal No-go.

On Windows, the supervisor starts the child suspended, assigns it to a native
kill-on-close Job Object, then resumes it. The Job owns descendants even after
the root exits; incompatible enclosing Jobs refuse the launch. Windows host
availability comes from `GlobalMemoryStatusEx`, while Job-member working sets
and the Job's peak private commit feed the child cap. CUDA sampling uses the
trusted System32 `nvidia-smi.exe` and the policy's exact GPU UUID. Unknown,
`N/A`, missing, or un-attributable process GPU memory fails closed; a Windows
run without a positive attributable GPU sample cannot publish success.

No static bound covers the whole-process peak of the route-specific loader,
transient graphs, and output coexistence, and none is required. Each arm is
admitted by its runtime guards: the policy must configure every guard (deadline,
sampling, termination grace, host reserve, child footprint cap, GPU UUID, GPU
reserve and child GPU cap), and immediately before spawn host and selected-GPU
free memory must cover the arm's estimate plus reserve (`estimate-plus-reserve-v1`).
On the Metal lane each arm's estimate is the product's own admission profile
(`product-admission-profile`): before every arm the adapter measures the admission budget
(host available minus the reserve) and runs the coordinate's MLX entrypoint with
`--sc20686-estimate --sc20686-budget-bytes <budget>` and its exact arguments, which resolves
the route like the run (product `LoadSpec`, geometry, tier, Lightning, references /
control clip) and prints, without loading weights or touching MLX, the max of the staged
phases: text encoder; VAE + conservative encode working set; DiT resident (the fit gate's
residency, adapters included) + VAE + its `72 B/token/dim` activation (VACE at the
documented +30% under-fit); DiT + VAE + the decode working set of exactly the tiling decision
the product's automatic planner (`auto_tiling_budgeted[_z16]`) makes with the budget left beside
the DiT and VAE (`decodeMode` single-pass / tiled / over-budget, the last priced single-pass),
and the run is pinned to that decision through `WAN_VAE_BUDGET_GIB` = the priced safe budget
(Wan, `mlx_gen_wan::admission_estimate`); for FLUX.2 Klein the Qwen3 encoder vs DiT +
VAE + the registered 1024² activation anchor scaled by the squared token ratio (target
plus references), doubled for true CFG, plus the KV route's cached reference K/V
(`mlx_gen_flux2::admission_estimate`). A later arm of the same coordinate uses the max of
that estimate and the completed arm's measured peak. An estimate above the cap falls back
to the cap (`child-footprint-cap-fallback`), as do the Candle lanes, whose entrypoints have no
estimate mode. The host and CUDA watchdogs then
terminate the owned tree on a cap or reserve breach. Every sealed unit's
`supervision.json` records that `admission` (`mode: runtime-guarded`, `rule`,
caps, reserves, optional static floors, each estimate's source and bytes, the
available bytes compared) with `wholeProcessPeakBoundBytes: null` and
its `wholeProcessPeakUnknownReason`; a resumed unit without it is rejected. A
pre-spawn refusal, watchdog abort, or failed child writes a sealed
`unaccepted.json` (`accepted: false`, outcome `refused`/`aborted`/`failed`) into
the retained `--resume-dir/failed/incomplete-*` diagnostics; it is never an
accepted arm. The fixed route geometry and normal/cancel schedule are never
shortened to fit.

Model identity has two independent fields: `model_snapshot_revision` is the immutable Hugging Face
revision, while `model_snapshot_sha256` hashes only the selected model/tier root. A selected root may
be either a component/tier directory with `config.json`, or a Diffusers pipeline root with
`model_index.json` and at least one component `config.json`. Nested tier roots such as
`<snapshot-revision>/q4` resolve the revision from the nearest two ancestors while hashing only the
`q4` contents. This preserves exact tier identity without confusing `q4` with a revision or widening
the hash to unrelated siblings.

## Product-equivalent residency

Residency is a sealed route axis, is passed to the entrypoint as `--sc20686-residency`, and is
applied to the real `LoadSpec` (plus request-scoped generation staging for FLUX.2 edit). The frozen
SceneWorks-equivalent strategies are:

| Product route | Strategy |
| --- | --- |
| `flux2_klein_9b_edit` | `sequential` |
| `flux2_klein_9b_kv_edit` (Metal lane only) | `sequential` |
| `wan2_2_ti2v_5b` | `sequential` |
| `wan2_2_t2v_14b` | `sequential` |
| `wan2_2_i2v_14b` | `sequential` |
| `wan_vace` | `resident` |
| `wan2_2_vace_fun_14b` | `sequential` |

The adapter, entrypoints, observer metadata, resolved-input manifest, row receipts, and reducer all
reject a different strategy rather than measuring a non-product residency shape. This includes the
Wan 14B ComfyUI-expert route: campaign residency is passed explicitly through its external-expert
loader, rather than falling back to that loader's ordinary resident default.

The Wan entrypoints are wired to the product-owned observer after each real route has bound its
snapshot-backed geometry. With observation off, the ownership hooks remain inactive and do not
allocate campaign evidence or retain cache ids. The adapter rejects a missing, non-JSONL, or
carriage-return-containing event transcript before reducing a campaign row.

Normal/cancellation pairs are inseparable decision evidence. The reducer independently requires the
cancel arm's product-owned cancellation identity, exactly one `cancelled` terminal, then metrics,
invalidation, and release in product order; each coordinate decision records that verification.

FLUX.2 Klein edit (`flux2_klein_9b_edit`) is an evidence-based no-go candidate for
persistent-reference-K/V productization on both lanes. Its `DoubleAttention` path projects
reference K/V for each denoise evaluation and joins it into dense attention;
there is no persistent reference K/V boundary or packed reader to promote. The observers record that
transient reference-slice work so a live campaign can establish the no-go without representing an
ordinary attention allocation as a promotable cache. The Mac product's other Klein edit route, `flux2_klein_9b_kv_edit`, *does* own a
persistent reference-K/V cache; it exists only on MLX and is measured by the Metal lane.

## Metal (MLX) lane

The Metal lane is the Mac product-path lane: it measures the MLX providers the SceneWorks worker
loads on Apple Silicon, through the same provider loaders, with the frozen product residency applied
to `LoadSpec::offload_policy` (`sequential` → `OffloadPolicy::Sequential` staged component/expert
residency, `resident` → `OffloadPolicy::Resident`). The CUDA lane continues to measure Candle and is
unchanged by it.

### Coverage and route map

The Metal lane runs the CUDA matrix unchanged — the same six routes at the same native coordinates
(resolution, frames, reference count, prompt, guidance, steps) with the same normal/cancel arms — plus
two lane extensions recorded in `sc20686_coverage_manifest.json` (`lane_extensions.mlx-metal`):
`flux2_klein_9b_kv_edit` at the two FLUX coordinates, and the A14B product default (Lightning on) as
a `-lightning` twin of each T2V/I2V coordinate at the forced guidance 1. A lane extension may add
coordinates to a shared route but never redefine one. Dropping any route blocks its family decision;
nothing is narrowed to fit.

| Route | MLX entrypoint | Cache kind on MLX | Residency |
| --- | --- | --- | --- |
| `flux2_klein_9b_edit` | `sc20686_flux2_edit` | recomputed reference slice | `sequential` |
| `flux2_klein_9b_kv_edit` | `sc20686_flux2_edit` | persistent reference K/V (`Flux2KvCache`) | `sequential` |
| `wan2_2_ti2v_5b` | `sc20686_wan` | persistent cross-K/V (`StepCache`) | `sequential` |
| `wan2_2_t2v_14b` | `sc20686_wan` | persistent cross-K/V per expert | `sequential` |
| `wan2_2_i2v_14b` | `sc20686_wan` | persistent cross-K/V per expert | `sequential` |
| `wan_vace` | `sc20686_wan` | recomputed text K/V | `resident` |
| `wan2_2_vace_fun_14b` | `sc20686_wan` | recomputed text K/V | `sequential` |

### Product load

The per-route load decisions live in the provider crates (`mlx_gen_wan::product_load`,
`mlx_gen_flux2::product_load`), which the SceneWorks worker calls for its own loads; both MLX
entrypoints build their `LoadSpec` through the same modules' `product_load_spec`, and the source map
records the decisions per route as `product_load` (and, on the A14B routes, `advanced_lightning`).
Only residency comes from the sealed axis above.

| Route | Snapshot (`SC20686_MLX_*` / `--flux-*-snapshot`) | Load quantization |
| --- | --- | --- |
| `flux2_klein_9b_edit`, `flux2_klein_9b_kv_edit` | the product's default `q4/` Klein tier root (`resolved_route` `flux2_klein_9b` / `flux2_klein_9b_kv`) | none: the tier is packed |
| `wan2_2_ti2v_5b`, `wan2_2_t2v_14b`, `wan2_2_i2v_14b` | the product's default `q4/` quant-matrix tier root | none: the tier's `config.json` is authoritative |
| `wan_vace` | the worker-assembled `wan_vace` snapshot: the dense **Wan2.1-VACE-1.3B** transformer plus the base-Wan 14B `q4/` tier's UMT5, z16 VAE and tokenizer | none: dense, projections held bf16 |
| `wan2_2_vace_fun_14b` | the worker-assembled dense VACE-Fun 14B high/low experts plus the same shared components | **Q4**, forced by the product unless the user picks |

The entrypoints and the adapter refuse any other tier (a packed tier whose `quantization.bits` is
not 4), and a VACE snapshot that is not the worker's assembled layout for that route (VACE-Fun also
needs `transformer_2/`). The Mac product's `wan_vace` is the 1.3B transformer, not the
Wan2.1-VACE-14B tree the Candle lane reads; the Wan entrypoint refuses any other transformer size.

Two product fixes (SC-20686, both change Wan-VACE output numerics) set what these routes measure:

* **Text context.** Each CFG branch's prompt embedding is zero-padded to `text_len` (512) and the
  transformer attends over all 512 tokens unmasked, as diffusers `WanVACEPipeline`
  (`_get_t5_prompt_embeds`: trim to the token count, then `torch.cat([u, u.new_zeros(512 - len,
  dim)])`) and base Wan build it (shared `mlx_gen_wan::pad_text_context`). The MLX VACE pipelines
  previously attended over the unpadded prompt (e.g. 13 and 126 tokens). Every recomputed text K/V
  slice is therefore 512 tokens on both branches.
* **Weight dtype.** The Wan2.1-VACE-1.3B checkpoint stores its 30 main blocks F32 beside BF16 VACE
  blocks; every attn/FFN/VACE projection and qk-norm weight is now cast to the bf16 compute dtype at
  load (diffusers casts every module outside `_keep_in_fp32_modules` under `torch_dtype`), while the
  reference's f32 set — embedders, modulation tables, the affine `norm2`, the output projection —
  stays f32. The 1.3B transformer's resident weights fall from 6.66 GiB to 4.06 GiB, and the
  preflight now prices VACE weights at those load dtypes rather than their stored width.

The two VACE snapshots are assembled exactly as the worker assembles them
(`mlx_gen_wan::convert::assemble_wan_vace_snapshot` / `assemble_wan_vace_fun_snapshot`, linked):
`transformer/` (and `transformer_2/`) from `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
(`linoyts/Wan2.2-VACE-Fun-14B-diffusers`), and `t5_encoder.safetensors`, `vae.safetensors`,
`tokenizer.json` from the first complete tier of the base-Wan 14B turnkey, `q4/` first (the tiers'
shared components differ as blobs). Each assembled root carries a `.snapshot-revision` holding the
VACE repository revision.

The A14B routes default to the Lightning distill (`advanced.lightning` unset means on): the
per-architecture `lightx2v/Wan2.2-Lightning` high/low LoRA pair at strength 1.0, forced to 4 steps at
guidance 1. Every Metal A14B coordinate states `--lightning on|off`: the shared coordinates are the
Lightning-off request, and the `-lightning` coordinates are the product default. They name the
Hugging Face hub (`--lightning-hub`, `SC20686_HF_HUB`) from which the entrypoint resolves the
`lightx2v/Wan2.2-Lightning` snapshot exactly as the product does (`refs/main`, else the pinned
revision; never the repository root), and the sealed pair (`--lora-high`/`--lora-low`,
`SC20686_WAN_{T2V,I2V}_LIGHTNING_{HIGH,LOW}`), which the entrypoint refuses unless it is
byte-identical to the pair the product loads from that snapshot.

Three source facts differ from the Candle lane and are recorded as cache kinds in the lane's source map
(`sc20686_source_map.json`, `lanes.mlx-metal`), not smoothed over:

* MLX Wan-VACE (`vace.rs`, `Attn::cross_attn`) projects the text K/V inside every main and VACE block
  on every CFG forward of every step. There is no persistent cross-K/V cache on this route, so its
  rows carry zero persistent bytes and the recomputed transient, like the FLUX edit route.
* MLX Wan stacks the CFG cond/uncond contexts on the batch axis of one cache. The source map's
  `cfg_kv_batch` rule fixes the exact batch: TI2V-5B is `2B` only when guidance > 1, the A14B
  experts always stack (`2B`), and FLUX.2 kv-edit and VACE are `B`. Each cache's exact `nbytes` must
  equal `2·kv_batch·H·Skv·D·dtype`; CUDA-lane events may not carry `kv_batch` at all.
* MLX FLUX.2 kv-edit extracts one reference-K/V slot per double and single layer on the first
  evaluation, with **one cache per CFG branch** (`Flux2KvCfgCaches`). The joint attention is unmasked,
  so reference K/V are prompt-dependent from the first double layer; the route previously shared one
  cache (the mflux fork's shape), so the negative extract overwrote the positive slots and every
  positive cached step attended over negative-branch reference K/V — a pre-existing wrong-output
  defect under guidance > 1, fixed with this lane and pinned by a CFG parity test (per-branch kv
  equals the non-kv forward exactly on every cached step). The extract step also materializes every
  slot together with its output, so a slot never keeps its layer's whole `[txt, target, ref]` K/V
  alive. The Candle lane has no kv-edit route and is unaffected.

### Observer and attribution method

The observer is `mlx_gen::sc20686` (`crates/media/mlx-gen/src/sc20686.rs`). It is inert unless the
entrypoint arms an output request on the rendering thread and the provider's `generate` enters
`observe_generation`; with nothing armed the provider calls `generate_impl` directly and every hook
returns before evaluating, allocating or resetting anything. It writes the same JSONL schema as the
Candle observers plus `"backend": "mlx-metal"` on every event and a `kv_batch` on each creation.

* **Persistent bytes** are the exact `nbytes` of the retained K/V arrays, registered when the product
  creates a cache (`register_cross_kv_set` in Wan's `build_cache`; `Flux2KvCache::apply` extract) and
  released when the product drops it (`StepCache`'s ownership guard; `Flux2KvCache`'s `Drop`, which
  frees the arrays before sampling the post-release remnant).
* **Transient peaks** come from MLX's `active`, `cache` and `peak` allocator counters. The observer
  owns every `reset_peak_memory` during a campaign and folds the counter into three nested
  high-waters before each reset: the run (`peak_bytes`, also folding every `active + cache`
  reservation), the current phase window, and each read window. A read window evaluates its inputs,
  resets the peak, runs the attention, evaluates the output, and records `high - before`. MLX is
  lazy, so these evaluation boundaries are what make a read measurable; they exist only in campaign
  mode and are a documented perturbation of the lazy schedule (hook tests prove the outputs stay
  bit-identical).
* **Per-phase attribution** is emitted as `phase-window` events (`encode`, `load`, `prepare-cache`,
  `denoise-step` with its step index, `post-denoise`, `decode`) driven by the provider's own progress
  stream and cache-build hooks. Each window carries its allocator before/after/high/reserved and the
  process `phys_footprint`/`phys_footprint_peak` from `proc_pid_rusage` (the ledger
  `/usr/bin/footprint` prints, read without a subprocess). The supervisor adds its
  own sampled `phys_footprint` of the owned process tree and admits each arm by `vm_stat` available
  memory -- free, speculative, purgeable, and file-backed (cached-file) pages, each component recorded in
  `admission.hostMemoryComponents` (runtime-guarded admission, no static peak bound).
* **Reuse/invalidation** events are exact: every read names its cache id, a rebuild releases the old
  id before the new creation, and the reducer's per-cache minimum reuse counts caches that were never
  read. A cached FLUX read window opens on the query and the fresh K/V, so the splice of the stored
  reference K/V is materialized inside it; a mid-denoise expert swap relabels its pending step window
  as `load`, so every `(window, index)` is unique.

**Campaign schedule vs product schedule.** Every number from the decision arms — the run and
phase-window peaks that feed the reducer's `peak_bytes`, and the durations — is measured on the
*campaign* schedule: the per-read evaluation windows cut the product's lazy graph at every cached
cross-attention, which can move both peak memory and time. They are attribution numbers, not the
product's own. The **schedule-control arm** (`--schedule-control`, Metal lane only) runs every
coordinate once more with `--sc20686-schedule-control`: cache creation/release and phase windows are
still recorded, but no read window evaluates or resets anything, so its run and phase peaks follow
the product's schedule. It publishes a separate sealed `sc-20686-schedule-control-v1` bundle whose
per-coordinate summaries (run peak, process peak, per-phase high-water and `phys_footprint_peak`) the
reducer recomputes from the transcripts over the complete Metal coverage; it is never decision
evidence and a decision bundle cannot be a control bundle.

The adapter and reducer require, for Metal rows only, `mlx-metal` on every observer event, a
`denoise-step` window (and `decode` for normal arms) before the terminal event with valid allocator
ordering and positive footprint, and — for persistent routes — the exact `kv_batch` byte identity.
A CUDA-lane transcript may not claim another backend. The v5 producer rows, v4 resolved inputs,
v6 bundles and v6 reducer decisions carry the lane; one bundle may not mix lanes.

### One-command Metal campaign

Build the two MLX entrypoints once, then run the sealed matrix (18 coordinates × normal/cancel):

```text
eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
cargo build --locked --release -p mlx-gen-wan --example sc20686_wan \
  -p mlx-gen-flux2 --example sc20686_flux2_edit
export SC20686_MLX_BIN_DIR=$PWD/target/release/examples
export SC20686_MLX_WAN_TI2V_5B_SNAPSHOT=… SC20686_MLX_WAN_T2V_14B_SNAPSHOT=… \
  SC20686_MLX_WAN_I2V_14B_SNAPSHOT=… SC20686_MLX_WAN_VACE_SNAPSHOT=… \
  SC20686_MLX_WAN_VACE_FUN_14B_SNAPSHOT=… SC20686_WAN_I2V_REFERENCE=… \
  SC20686_VACE_CONTROL_17_DIR=… SC20686_VACE_MASK_17_DIR=… SC20686_VACE_CONTROL_33_DIR=… \
  SC20686_VACE_MASK_33_DIR=… SC20686_VACE_REFERENCE=… SC20686_WAN_T2V_LIGHTNING_HIGH=… \
  SC20686_WAN_T2V_LIGHTNING_LOW=… SC20686_WAN_I2V_LIGHTNING_HIGH=… SC20686_WAN_I2V_LIGHTNING_LOW=… SC20686_HF_HUB=…
python3 scripts/sc20686_campaign_adapter.py --campaign --matrix \
  --inference-revision "$(git rev-parse HEAD)" \
  --safety-policy /abs/sc20686-darwin-mlx-policy.json \
  --resume-dir /abs/external/sc20686-mlx-resume \
  --wan-manifest scripts/sc20686_mlx_wan_campaign_manifest.example.json \
  --flux-entrypoint "$SC20686_MLX_BIN_DIR/sc20686_flux2_edit" \
  --flux-snapshot <flux2-klein-9b tier root> --flux-kv-snapshot <flux2-klein-9b-kv tier root> \
  --flux-reference /abs/ref.png --flux-reference2 /abs/ref2.png \
  --matrix-output /abs/evidence/sc20686-mlx-campaign
# Product-schedule control (same inputs, its own resume directory and output):
python3 scripts/sc20686_campaign_adapter.py --campaign --matrix --schedule-control … \
  --resume-dir /abs/external/sc20686-mlx-control-resume \
  --matrix-output /abs/evidence/sc20686-mlx-schedule-control
```

The MLX entrypoints are strict: unknown flags (including the Candle harness's `--single-only`) and
repeated flags are refused, and campaign mode requires every coordinate argument, so the Metal
coordinates seal an explicit `--seed 42`.

The policy is the strict `darwin-mlx` schema (`schemaVersion`, `backend`, `deadlineSeconds`,
`pollMillis`, `termGraceMillis`, `hostFreeReserveBytes`, `childFootprintCapBytes`, `stdoutCapBytes`,
`stderrCapBytes`, `eventCapBytes`); the caps must be chosen for the host that runs it, and every arm
is refused before spawn unless free memory covers its estimate (above) plus reserve. The run needs the Metal GPU for
its duration. Each snapshot is an immutable tier root (`<revision>/q4` or a revision directory),
except the two Metal VACE routes, which take the worker-assembled snapshot (`transformer/` beside
`t5_encoder.safetensors`, `vae.safetensors`, `tokenizer.json`; VACE-Fun adds `transformer_2/`)
carrying a `.snapshot-revision` file with the VACE repository's 40-hex revision. Its identity hash
follows the assembly's symlinked transformer directories.
