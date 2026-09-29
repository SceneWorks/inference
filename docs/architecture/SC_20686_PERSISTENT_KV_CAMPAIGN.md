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
free memory must cover cap plus reserve. The host and CUDA watchdogs then
terminate the owned tree on a cap or reserve breach. Every sealed unit's
`supervision.json` records that `admission` (`mode: runtime-guarded`, caps,
reserves, optional static floors) with `wholeProcessPeakBoundBytes: null` and
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
one lane extension recorded in `sc20686_coverage_manifest.json` (`lane_extensions.mlx-metal`):
`flux2_klein_9b_kv_edit` at the two FLUX coordinates. Dropping any route blocks its family decision;
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

Three source facts differ from the Candle lane and are recorded as cache kinds in the lane's source map
(`sc20686_source_map.json`, `lanes.mlx-metal`), not smoothed over:

* MLX Wan-VACE (`vace.rs`, `Attn::cross_attn`) projects the text K/V inside every main and VACE block
  on every CFG forward of every step. There is no persistent cross-K/V cache on this route, so its
  rows carry zero persistent bytes and the recomputed transient, like the FLUX edit route.
* MLX Wan stacks the CFG cond/uncond contexts on the batch axis of one cache, so each cache's
  `kv_batch` is 2 under CFG; its exact `nbytes` must equal `2·kv_batch·H·Skv·D·dtype`.
* MLX FLUX.2 kv-edit extracts one reference-K/V slot per double and single layer on the first
  evaluation. With CFG the negative pass re-extracts every slot: the transcript records that as a
  rebuild (release, then create), so the positive slots' zero reads are counted honestly.

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
  process `phys_footprint`/`phys_footprint_peak` from `/usr/bin/footprint`. The supervisor adds its
  own sampled `phys_footprint` of the owned process tree and admits each arm by `vm_stat` free memory
  (runtime-guarded admission, no static peak bound).
* **Reuse/invalidation** events are exact: every read names its cache id, a rebuild releases the old
  id before the new creation, and the reducer's per-cache minimum reuse counts caches that were never
  read.

The adapter and reducer require, for Metal rows only, `mlx-metal` on every observer event, a
`denoise-step` window (and `decode` for normal arms) before the terminal event with valid allocator
ordering and positive footprint, and — for persistent routes — the exact `kv_batch` byte identity.
A CUDA-lane transcript may not claim another backend. The v5 producer rows, v4 resolved inputs,
v6 bundles and v6 reducer decisions carry the lane; one bundle may not mix lanes.

### One-command Metal campaign

Build the two MLX entrypoints once, then run the sealed matrix (14 coordinates × normal/cancel):

```text
eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
cargo build --locked --release -p mlx-gen-wan --example sc20686_wan \
  -p mlx-gen-flux2 --example sc20686_flux2_edit
export SC20686_MLX_BIN_DIR=$PWD/target/release/examples
export SC20686_MLX_WAN_TI2V_5B_SNAPSHOT=… SC20686_MLX_WAN_T2V_14B_SNAPSHOT=… \
  SC20686_MLX_WAN_I2V_14B_SNAPSHOT=… SC20686_MLX_WAN_VACE_SNAPSHOT=… \
  SC20686_MLX_WAN_VACE_FUN_14B_SNAPSHOT=… SC20686_WAN_I2V_REFERENCE=… \
  SC20686_VACE_CONTROL_17_DIR=… SC20686_VACE_MASK_17_DIR=… SC20686_VACE_CONTROL_33_DIR=… \
  SC20686_VACE_MASK_33_DIR=… SC20686_VACE_REFERENCE=…
python3 scripts/sc20686_campaign_adapter.py --campaign --matrix \
  --inference-revision "$(git rev-parse HEAD)" \
  --safety-policy /abs/sc20686-darwin-mlx-policy.json \
  --resume-dir /abs/external/sc20686-mlx-resume \
  --wan-manifest scripts/sc20686_mlx_wan_campaign_manifest.example.json \
  --flux-entrypoint "$SC20686_MLX_BIN_DIR/sc20686_flux2_edit" \
  --flux-snapshot <flux2-klein-9b tier root> --flux-kv-snapshot <flux2-klein-9b-kv tier root> \
  --flux-reference /abs/ref.png --flux-reference2 /abs/ref2.png \
  --matrix-output /abs/evidence/sc20686-mlx-campaign
```

The policy is the strict `darwin-mlx` schema (`schemaVersion`, `backend`, `deadlineSeconds`,
`pollMillis`, `termGraceMillis`, `hostFreeReserveBytes`, `childFootprintCapBytes`, `stdoutCapBytes`,
`stderrCapBytes`, `eventCapBytes`); the caps must be chosen for the host that runs it, and every arm
is refused before spawn unless free memory covers cap plus reserve. The run needs the Metal GPU for
its duration. Each snapshot is an immutable tier root (`<revision>/q4` or a revision directory).
