# iOS device runbook

Reconciled 2026-10-26 against candidate `2697693a981f` and the sc-26270
ANCHOR campaign (two endpoint configurations per family, with two optimized
repeats). Source repairs and terminal device evidence remain in progress. These
are implemented harness commands, not a claim the epic
is on main or that all models render on the phone. See [strategy](architecture/ios-strategy.md),
[specification](architecture/ios-project-spec.md), and [remaining work](ios-epics.md).

## Current model and evidence status

Target: **iPhone 16 Pro Max, 8 GB, iOS 26.6.2**. No successful target-device image
has been recorded. Registration is not admission or demonstrated fit.

| Model set | Feature status at this snapshot | Target-phone status |
|---|---|---|
| SANA base/sprint | Registered; source-bound admission and packed Gemma embeddings merged in #1272 (`23c7e3953`) | Admitted loader refuses without applicable capture; 1024/tile192 attempt terminated, cause unconfirmed |
| Anima Turbo | Registered without training; Base/Aesthetic excluded | No demonstrated phone fit |
| Iris 3B | q8/q4 engine fixes merged #1275 (`2f3265c71`); iOS registration #1277 merged | No demonstrated phone fit; depth access is explicitly diagnostic |
| SDXL, Illustrious-XL v1/v2, RealVisXL, RealVisXL Lightning | Low-memory engine fixes merged #1274 (`a54ad6c3a`); iOS registration #1278 merged | No demonstrated phone fit |
| Supra2-IMG | Provider delivered separately; iOS registration sc-26269 is explicitly user-deferred and non-blocking for this general delivery | No iOS registration/fit claim here |
| Z-Image Turbo (`z_image_turbo` text / img2img / LoRA+LoKr; `z_image_turbo_control` pose/canny/depth) | Registered without training at its stored q4/q8/bf16 tiers (sc-26433); every quantum bounded; host-derived reachable sizes below | No demonstrated phone fit. Michael's 8 GB **Mac** q4 run is Mac evidence, not device evidence |

The legacy S3 image check drives SANA. The sc-26270 campaign protocol below now
selects every catalog provider/route/tier/geometry/mode/strategy explicitly. There
are still no `--iris`, `--sdxl`, `--anima` or `--supra2` shortcut flags. Z-Image Turbo
is a catalog model (sc-26433); `run_smoke.sh --zimage` is now only a shortcut for ONE
bounded capture of a Z-Image campaign case (`IOS_SMOKE_ZIMAGE_CASE`, default
`z_image_turbo/z_image_turbo/q4/1024/text/4/s_t512_o64_c67108864_w1_dit`) through the same capture permit — there is
no unadmitted Z-Image loader.

## Model execution policy: ordinary CI versus explicit proofs

The canonical rule lives in `AGENTS.md`: normal PR and `main` CI compiles/lints
all applicable test targets but **never executes learned model/module/layer
forward or backward, generation or training**, even on tiny synthetic weights.
Raw matmul/conv/SDPA/norm/allocator kernels, metadata/schema/math/tokenizers,
mocks and cheap constructor/load-only checks remain ordinary. Model source may
land on `main`; ignored model tests retain their complete bodies/assertions and
are registered in `scripts/ci/model-inference-proofs.json`.

`model-inference-proofs.yml` is a **separate, workflow-dispatch-only** execution
path. Nothing automatically runs a model because a PR merges, a pin changes, or
an epic reaches its end. The workflow takes only `backend` and `request` inputs;
request JSON contains a registered selection and an explicit existing-supervisor
resource policy. There are no hidden RAM defaults, free Cargo arguments, or
unfiltered parameter/measurement grids. Obtain execution authorization and use
approved caps/reserves/deadlines before dispatch; no dispatch is recorded here.

Synthetic H3 and MOSS validation also lives in the separate explicit
`real-weights.yml` workflow. The former ordinary-CI dispatch inputs map as follows:

| Former `ci.yml` input | `real-weights.yml` profile | Preserved execution route |
|---|---|---|
| `synthetic_w4a8=mlx` | `synthetic-w4a8-mlx` | Hosted macOS 15 precise projection plus self-hosted `rw-llm` Metal suite |
| `synthetic_w4a8=cuda` | `synthetic-w4a8-cuda` | Self-hosted Windows CUDA H3 suite, physical GPU0 UUID binding |
| `synthetic_w4a8=both` | `synthetic-w4a8-both` | All three H3 routes above |
| `lanes=moss-edit-cuda` | `moss-edit-cuda` | Exact four synthetic cases on the pinned physical GPU1 UUID/PCI identity |
| `lanes=moss-edit-metal` | `moss-edit-metal` | Exact prepared NAX Metal oracle with immutable source/runtime/binary receipts |

Dispatch against the intended source ref, for example:

```sh
gh workflow run real-weights.yml --repo SceneWorks/inference --ref <source-ref> -f profile=synthetic-w4a8-both
```

These profiles never run through `all`, schedule, PR, push, merge group or an
ordinary CI dependency. Every run/attempt keeps its own queue entry with
cancellation disabled. Existing exact test counts, named assertions, fixture and
source controls, backend markers, device restrictions and MOSS evidence artifacts
are preserved. Tests remain compiled and ignored in ordinary CI; the dedicated
MOSS CUDA runner verifies ignored membership before explicitly selecting those
same cases. `ci.yml` retains its ordinary build/lint/non-model manual lanes and
the read-only GPU1 diagnostic. An empty or obsolete model selector cannot schedule
model execution from that workflow.

For source-conditional SD35 cases, optional request `configuration` selects only a
name declared in that case's fixed `configurations` map; omission selects its explicit
`default_configuration`. One canonical case identity is retained. Each recipe binds
its source identity, exact set/UNSET values for the source-read SD35 controls,
per-recipe Cargo features, backend/device set, scope, and required source work.
Inherited recipe flags are not authority: declared null values explicitly remove
them. Asset paths and optional tolerance controls remain separate source controls;
recipes cannot set credentials, child selectors, device-authority flags, or arbitrary
environment/argv. Unknown names and incompatible execution lanes fail before work.

The three ordinary SD35 component-parity variants default to full F32/CPU encoder
and DiT work. The `sd35_bigg_pad_token_ab` source default instead covers only CPU
encoder-conditioning A/B; its CUDA recipe additionally requires both DiT forwards.
Do not interpret that encoder-only result as full parity. CPU recipes do not inherit
a CUDA build merely because another recipe supports CUDA. The plan, effective flag
record, case declaration, and summary identify the exact selected scope and work;
missing required work still rejects acceptance under the existing completion protocol.

Source-bound `configuration_kind: "feature-only"` recipes use empty controls and
`environment: {}`; they select only declared Cargo feature graphs, never request
provided environment or argv. If the ordinary source-default graph compiles the
case out or returns before learned work, `default_configuration: null`,
`backend: null`, and `expected_devices: []` explicitly mean **no runnable NN default**.
A named supported recipe is required. Exact, package, and `all` selectors refuse
that unresolved case rather than silently omitting it or claiming partial coverage.
The packed CPU reference-pair case retains its genuine CPU source-default recipe.

A recipe may separately declare `execution_lane` for its required compile/resource
host. For example, the Qwen2.1 miniature block-0 comparison retains its original
CUDA-or-Metal file gate but actually evaluates learned models on CPU. Its declared
CUDA/Metal build recipes therefore require the corresponding host while keeping
`backend` and the observed/expected NN device set **CPU only**. A GPU-compatible
build host or conservative physical lease is not a GPU-model PASS. Impossible
lane/feature/device combinations and undeclared lane overrides reject; absent
`execution_lane` preserves legacy routing.

`configuration_kind: "source-controls"` adds only source-declared fixed control
names and literal values/UNSET from each canonical recipe. Requests still select a
registered name; they cannot supply environment, Cargo arguments, or lane overrides.
The declared names must match every recipe's settings exactly. Source presence
semantics are preserved: an empty string or `"0"` is still SET, not a boolean false.
Source-default recipes explicitly UNSET inherited controls; optional stream, device,
precision, rung, and fixture branches run only as declared, not as an automatic grid.

Process/toolchain variables, credentials, child selectors, permits, physical CUDA
visibility/order, lease descriptors, and external asset-path/required-asset bindings
are not injectable recipe controls. A source logical-device selector such as
`CANDLE_LLM_DEVICE` cannot replace controller-approved visibility or observed NN
device evidence. Required external assets/gates remain separate; a fixed recipe
neither invents mandatory environment names nor retargets source cache fallbacks.

```sh
# Inspect fixed source-defined bindings; this does not run models.
python3.12 -I -c 'import json; print(json.dumps(json.load(open("scripts/ci/model-inference-proofs.json", encoding="utf-8")), indent=2))'
# Prepare request JSON with selection + the existing supervisor's complete policy schema.
# MODEL_PROOF_POLICY points to the independently approved resource policy, not a checkpoint.
jq --arg selection 'mlx-gen-sdxl/quantum_tests::pricing::additivity::optimized_bf16_img2img_preserves_full_request_rgb' \
  --slurpfile policy "$MODEL_PROOF_POLICY" '{selection:$selection,policy:$policy[0]}' \
  > /tmp/model-proof-request.json
# Exact completed source revision; this command is an example, not approval to execute.
gh workflow run model-inference-proofs.yml --repo SceneWorks/inference \
  --ref "$COMPLETED_SOURCE_REF" -f backend=mlx -f request="$(cat /tmp/model-proof-request.json)"
```

Before the new workflow exists on the default branch, dispatch through the **existing**
`real-weights.yml` workflow identity at the exact feature ref. GitHub does not make a
brand-new workflow dispatchable merely because its file is in a feature branch. The
bridge uses the same canonical runner, retains the existing 25-input schema, and does
not alter the frozen `ios-pricing-compile-only` profile or its JSON default:

```sh
jq --arg backend mlx --slurpfile request /tmp/model-proof-request.json \
  '{backend:$backend,request:$request[0]}' > /tmp/model-proof-binding.json
gh workflow run real-weights.yml --repo SceneWorks/inference \
  --ref "$COMPLETED_SOURCE_REF" -f profile=model-inference-proofs \
  -f ios_pricing_compile_binding="$(cat /tmp/model-proof-binding.json)"
```

Host platform/runner and actual test compute-device SET are recorded separately.
A CPU child on a Mac is **not** Metal model evidence. Each executed work scope
reports its actual selected device before context restoration; default/global
context is not unioned into used devices. Mixed-device comparisons must report
all actual scopes without creating a GPU stream or rerendering for telemetry.
The proof runner requires exact nonzero discovery, full-body completion, all
source-declared root/CPU-child/rank successful-summary cardinalities, preserved
expected-fault assertions, and actual in-scope device observations.
Missing assets/dependencies, skipped/ignored bodies and partial workers fail
proof acceptance; compile/source/discovery success is never a runtime PASS.

Evidence contains exact source/fixture-model identity, backend/configuration,
pass/body counts, existing computed outputs/metrics, bounded redacted logs and
duration, plus a workflow summary. Only regular, bounded, explicitly known files
from the owned workflow output tree can be uploaded. Checkpoints, HF caches,
credentials and user-reference files are never collected; no model is rerun to
create evidence. Fast tests that end between GPU sampling ticks record an
unknown GPU-memory peak, not a zero measurement or a capacity/admission proof.

Actual external inputs require successful, scope-bound `MODEL_INFERENCE_INPUT_USED`
receipts covering every source-declared required slot, API reader/role, and complete
selected collection. A source descriptor or empty environment summary is not actual
external input identity. Synthetic/source-defined fixtures retain their source binding;
unknown external slot coverage refuses before NN execution, rather than inventing paths
or mandatory environment. Optional external overrides bind the **actually consumed**
source branch; a committed/generated/in-memory default creates no external obligation.
A relative, checkout, or cache default path is not thereby source-defined.

A source-declared `requirement_bundles` list expresses alternatives of complete AND
requirements, in addition to common `requirements`. Each bundle has a source identity;
its optional `additional_observed_readers` lists exact reader/role pairs for original
optional or attempted reads, never a wildcard or a substitute for required work. Every
observation retains its scope and provenance grade. Positive mode-specific observations
must be explained by the same complete bundle: an ordinary-directory subset cannot hide
Hadamard inputs from an incomplete Prism bundle. A single GGUF collection may represent
its original embedded model/config/tokenizer bundle without inventing three receipts.
`input_requirement_bundles` reports observation compatibility, not a selected model type,
source predicate, or successful template parse. No request can select or override these
source-fixed bundles, reader roles, or branch predicates.

Conditional defaults use an explicit positive source-defined branch contract plus the
separately verified completed immutable body. `source_origin_basis` distinguishes that
derivation (`source-defined-branch-contract`) from an observed `actual-reader-tag`;
no default read event is fabricated. External cases may use such optional-slot defaults,
but every complete bundle must retain unconditional external input requirements; this
never makes the whole external case a source-defined/no-input default.
Logical source obligations that share one observable reader/role collection are retained
losslessly in source facts rather than converted into guessed distinct-file, shard, or
invocation counts. Such lower bounds do not independently attest every named component.

Input provenance remains graded: `validated-content` reuses an existing retained
consumption receipt; `same-read-buffer` digests only bytes the original reader already
naturally consumed; `successful-reader-binding` records the actual resolved input set
with `content_integrity: not_attested`. The last grade proves only labelled source-bound
case execution, never checkpoint-byte integrity, a verified named revision, or terminal
model evidence. Existing stronger checksum/pin requirements remain stronger. No eager
mapped-page scans, extra Tensor evaluation, duplicate payload reads, or path/mtime hashes
are introduced. Physical completion, artifact validation, and workflow summaries preserve
`acceptance_scope`, `input_grade`, and claim limits; `accepted: true` alone is not a stronger
input or checkpoint claim.

The source sharding case that actually uses CUDA ordinals 0 and 1 additionally requires
an explicit ordered request `cuda_device_uuids: [role0UUID, role1UUID]` and one full
`additional_cuda_policies` entry for role 1. The existing primary policy binds role 0.
Both UUIDs must be distinct and explicitly approved, with identical host/tree/log/deadline
settings and independently declared GPU caps/reserves. CVD follows **source role order**;
UUID sorting is lock order only. Both leases, admission baselines, fresh ordinal-to-UUID
recheck, bounded per-UUID live/owned-descendant monitoring, and every post-idle check must
complete before sealing. Failure on either device rejects the whole case; missing peak
samples remain unknown, never fabricated zeros. Other handles or streams are not extra
GPUs, and no idle-device discovery, physical-index guess, or source retarget is authorized.

Dispatches queue without cancelling running inference or replacing older pending
proofs. Per-run execution remains serialized on compatible physical resources.
Provisioned Mac launches additionally hold the shared cooperative POSIX lease
`/Users/Shared/SceneWorks/gpu-locks/apple-metal.lock` through the owned child tree;
CUDA uses the existing `%PUBLIC%/SceneWorks/gpu-locks/<physical-UUID>.lock` byte-lock
protocol for the explicitly approved UUID, never a different idle GPU. The Iris
sharing/refusal case preserves its original child's `CUDA_VISIBLE_DEVICES=0` reset.
That value names a CUDA ordinal under the effective `CUDA_DEVICE_ORDER`, **not an
NVML physical index**. While holding the approved UUID lease, the controller runs
the verified context-free driver metadata primitive in a fresh process with that
exact child-effective visibility/order and requires ordinal 0's UUID to match the
leased UUID. Missing, ambiguous, mismatched, or unverified mappings reject before
model launch; neither source controls nor the approved device are retargeted. This
metadata is resource-binding evidence, never learned-model device/runtime proof.
Runner labels and unique Actions concurrency keys are not device mutexes. Cooperating launchers in
both repositories/direct-local wrappers use the same namespace. Unwrapped user apps
or other local processes still require the existing physical census/operator
coordination; the lease does not claim universal exclusion. No GPU kill behavior is
added by this coordination.

Normal PR CI retains its separate superseded-run cancellation rule because it
contains no model execution. This policy does not waive epic/device acceptance
requirements or renew/retire runtime measurements when a pin changes.

## Explicit CPU fixture captures (not phone anchors)

The `real-weights.yml` **`ios-selected-captures` profile** runs only registered
ignored fixture proofs on `nax-macos-2`; it never joins a scheduled or `all`
profile. This is not the phone campaign below. Obtain the host-time authorization
before dispatching. These are examples, not a record of captures executed.

GitHub permits at most 25 dispatch inputs, so the existing JSON input is
profile-specific: **only this capture profile** reads its `selection` key. An
omitted/blank binding or absent selection key selects `iris`; a provided selection
must name `all`, a fixed family, or an exact `package/FQN` from
`scripts/ci/ios-selected-captures.json`. Empty, null and unknown provided selections
fail rather than selecting another family. Binding JSON travels through an
environment variable and quoted argv, then the existing runner decodes/validates
it before any capture—not a GitHub `fromJSON('')` startup expression or shell
code. Discovery, parent/CPU-child completion and exact selected counts remain
mandatory.

```sh
# Default iris selection; choose the approved source ref explicitly.
gh workflow run real-weights.yml --repo SceneWorks/inference --ref "$SOURCE_REF" \
  -f profile=ios-selected-captures
# One family (use {"selection":"all"} for every registered capture).
gh workflow run real-weights.yml --repo SceneWorks/inference --ref "$SOURCE_REF" \
  -f profile=ios-selected-captures -f ios_pricing_compile_binding='{"selection":"sdxl"}'
# One exact registered proof.
gh workflow run real-weights.yml --repo SceneWorks/inference --ref "$SOURCE_REF" \
  -f profile=ios-selected-captures \
  -f ios_pricing_compile_binding='{"selection":"mlx-gen-sdxl/quantum_tests::pricing::final_norm_guards_cover_norm_pool_and_projection"}'
```

The **`ios-pricing-compile-only` profile is unchanged**: its original default JSON,
required `source_sha`/`manifest_sha256`/`host` fields, phase handling and strict
unknown-key validation still apply. Do **not** add `selection` to a compile
binding. The former standalone `ios_capture_selection` input is replaced only for
the capture profile by the JSON selection above.

## Campaign protocol and command matrix (sc-26270)

**Mechanism only; no new device measurements are recorded here.** Diagnostic
endpoint captures use the source-owned capture permit and per-quantum guards;
they do not establish admitted execution. A primitive result, launch headroom,
tensor byte sum or soft MLX allocator limit does not establish model fit or a
validated physical-memory cap. The current empty artifact-bound evidence remains
fail-closed for admitted execution. Unsupported strategy/tier/task combinations remain
visible instead of being silently omitted. Malformed/missing snapshots are
`diagnostic_failure`, **not** next-up admission-refusal evidence.

Weights-free listing uses the actual `runtime-ios` catalog (including transforms
and composed task memory registrations), with the SDXL provider's route constant.
It creates no arrays and loads no model. A new registered provider/task becomes
visible automatically. Supra2 currently appears in the dependency record, not as
a fictitious supported provider; once registered it enumerates at 256².

```sh
export PATH="$HOME/.cargo/bin:$PATH"
# CPU-only catalog exporter; --list alone NEVER launches a device campaign.
python3 scripts/ios/campaign.py list > /tmp/ios-campaign-catalog.json
# This is the full, executable menu, not a handwritten campaign population.
python3 scripts/ios/campaign.py list --catalog /tmp/ios-campaign-catalog.json \
  --provider iris_3b --tier q8 --edge 768 --rung 3
```

| Axis | Values / source |
|---|---|
| Provider | Catalog generators SANA base/sprint, Anima Turbo, SDXL, Iris, Z-Image Turbo and its control route; registered restoration and composed depth tasks |
| SDXL route | `sdxl`, `illustrious_xl_v1`, `illustrious_xl_v2`, `realvisxl`, `realvisxl_lightning` (provider constant) |
| Tier | `q4`, `q8`, `bf16`; unsupported task/tier combinations stay explicit refusals |
| Geometry | Anchors at the 1024² reference geometry (future Supra2: 256²); 512²/768² are derived, never captured (one 512² validation case) |
| Mode | Descriptor conditioning kinds plus text, advertised prompt batch, SDXL `ip_adapter`, registered restoration/depth; no edit-to-text downgrade. A route whose control input is required (`z_image_turbo_control`) has no text mode: its `control` and `reference` cells both carry the control image (pose) |
| Strategy | Two anchors per family: rung 0 (resident) and the deepest implemented rung at its cheapest selection (optimized); every rung in between is derived |

Provision complete caller-owned snapshots into a distinct slot for every
route/tier. The slot name is not proof of stored precision. Iris generation needs
its Qwen3-VL encoder, depth/restoration their task exports; SDXL may need its
fp16-fix VAE. A `components.json` beside the snapshot maps component IDs to
confined relative directories within that slot, e.g.
`{"text_encoder":"qwen3_vl","vae":"fp16_fix_vae"}`. Nothing downloads or
silently resolves a sibling artifact. Existing source inventories, deferred-read
checks and admitted request binding are preserved.

```sh
# After the host app has been installed (no campaign is started by pushing):
scripts/ios/push_model.sh "$SNAPSHOT" --campaign-route iris_3b --tier q8 --device "$DEVICE"
# A q8/q4 slot must hold a PACKED artifact: a dense snapshot requested at q8/q4 is refused by
# admission (typed `artifact`), never quantized on the phone. Produce packed tiers on the host CPU
# (external RSS guard required; no downloads, Metal trapped), then push each tier into its slot:
cargo run -p mlx-gen-sana --example sana_prequant_cpu -- "$SANA_BF16" "$SANA_Q8" 8 "$TENSOR_BUDGET_MIB"
scripts/ios/push_model.sh "$SANA_Q8" --campaign-route sana_1600m --tier q8 --device "$DEVICE"
cargo run -p mlx-gen-iris --example iris_prequant_cpu -- generation "$IRIS_DENSE" "$IRIS_Q8" 8 "$TENSOR_BUDGET_MIB"
cargo run -p mlx-gen-iris --example iris_prequant_cpu -- text-encoder "$QWEN3_VL" "$IRIS_Q8/text_encoder" 8 "$TENSOR_BUDGET_MIB"
# Same mechanism for every route and tier in the actual listing:
scripts/ios/push_model.sh "$SDXL_SNAPSHOT" --campaign-route illustrious_xl_v2 --tier q4 --device "$DEVICE"
# SDXL `control` and `ip_adapter` cases need their component in the slot. --control/--ip-adapter
# place it beside the snapshot and map the `control` / `ip_adapter` roles in components.json onto
# the load spec's ControlNet / IP-Adapter sources (attached only for that case's own mode). A case
# without its component is a provisioning failure, never admission or refusal evidence.
scripts/ios/push_model.sh "$SDXL_SNAPSHOT" --campaign-route sdxl --tier q4 \
  --control "$CONTROLNET_TILE_SDXL" --ip-adapter "$IP_ADAPTER_SDXL" --device "$DEVICE"
# Z-Image Turbo: the published `SceneWorks/z-image-turbo-mlx` tiers are already stored packed
# (q4/, q8/) or dense (bf16/) — push each tier dir as-is; a tier slot holding another stored tier is
# a typed `artifact` refusal. Both routes load the `z_image_turbo` slot; the control route
# (`z_image_turbo_control`) also needs the dense Fun-Controlnet-Union checkpoint as the slot's
# `control` role (packed on the phone at load, one bounded quantum per block):
scripts/ios/push_model.sh "$ZIMAGE_MLX/q4" --campaign-route z_image_turbo --tier q4 \
  --control "$ZIMAGE_TURBO_FUN_CONTROLNET_UNION" --device "$DEVICE"
```

**Quantum pricing: safety on the host, accuracy on the device.** Every counted
quantum price is proven on the host CPU against MLX's stalled peak
(`mlx_gen::ios_quantum::measured_peak_bytes`). Formula ≥ measured is a HARD
failure: it is the safety bound, and host load only lowers that measurement.
Formula ≤ 1.5× measured is REPORT-ONLY: an over-ratio prints
`PRICING-RATIO-HIGH <label> <ratio>` and the test passes. The CPU's in-flight
window (about 11 single-op tasks) cannot observe the phone's wider retention (up
to ten in-flight command buffers of up to 20 ops / 40 MB each), and it swings with
host load. Low-end accuracy is therefore calibrated here, on the device: the
fully optimized anchor captures and the validation case of this campaign
calibrate the formulas. There are no per-proof allowances.

Campaign case keys are `provider/route/tier/edge/mode/rung/selection/steps`. The
selection token is the cell's memory selection from the provider's OWN declared
space (`MemoryProviderContract::selection_space`, read from its weights-free
contract surface at the campaign's load shape): `s` (optional phase eviction),
`t<edge>`/`o<overlap>` (decode tile), `c<elements>` (attention chunk), `w<n>`
(transformer window) and the window scope `dit`/`te`/`both`, `_`-joined; `base`
when the rung declares no parameter. `steps` is the denoise step count the case
runs (`0` = the provider's own default). A case is executed exactly as selected (a
token outside the artifact's declared space is a typed `strategy` refusal) and
recorded in the evidence key's execution levers. The listing (schema 2) names each
case's `purpose` (`resident`, `optimized`, `validation_steps`,
`validation_geometry`) and `repeats`, plus every route/tier the provider does not
serve (`unserved`, never silently dropped). Control-required routes
(`CONTROL_REQUIRED_ROUTES`: `z_image_turbo_control`) have one family per accepted
control kind (`control_pose`, `control_canny`, `control_depth`, `reference_pose`,
...) and no plain text mode.

### Anchors and derivation: the recorded memory-anchor rule (epic 16774)

**The campaign measures two extremes per family and derives everything in between.** This is
the rule SceneWorks already records for its desktop lanes — `docs/memory-strategy-contract.md`:
"the anchor + analytic derivation is the complete, self-sufficient admission basis", and "every
non-anchor rung and geometry is now derived analytically from the anchor … rather than measured"
(`crates/sceneworks-core/src/memory_anchor.rs`, image derivation law). It replaces the exact-match
evidence table that needed one measured row per model × route × tier × mode × size × rung ×
selection (18,450 cells). Michael: "measure the extreme cases, fully resident, and fully
optimized, and estimate in between."

**Anchor family** = `(device, base ARCHITECTURE digest, tier, variant)` — the recorded rule's
one anchor per (model, tier, lane, VARIANT/decoder), keyed on what actually shapes memory:

* **Architecture**, not route or checkpoint: `layout::architecture_digest` hashes the base
  components' safetensors HEADERS — tensor names, shapes, dtypes and so packing — and the memory
  load settings; never values, file names, config content, paths or the route. Checkpoints of one
  architecture share anchors; any header difference (a distilled U-Net, another VAE, other dense
  dtypes) is its own family automatically. The per-checkpoint artifact `layout_sha256` stays in
  the key: an anchor's own key is exact; any other request of the family is derived.
* **Variant**: actual source role/mechanism family plus its resolved sampling class and
  configured overlay identity. Unproved reference/init, control+reference, mask/edit and IP
  graphs do not borrow text/control-only evidence. Role-identical aliases may share a family;
  the existing raw `reference_count` remains aggregate image geometry (including Mask/Control).
  Typed mask/control subcounts partition that aggregate only in the validated source-known
  carrier set; true Reference roles are counted separately when the actual carrier exists.
  Exact raw counts remain binding/pricing inputs, not an arbitrary measured-count grid.
  Only a source-certified correlated additivity domain can permit further sharing. Legacy
  `derive` fixtures, logical scratch sums, and `IOS_NON_ADDITIVE_MODES` are not that proof;
  `Uncertified` is not capability completion. SDXL validates its actual sampler first, then
  applies the runtime's IP predicate to configured IP + Reference with no mask/control/acceleration.
  Its current earlier accel+Reference refusal remains a refusal, not an invented accepted INIT
  family. Reference strength overrides global strength; Control uses each conditioning scale,
  not the ignored top-level `control_scale`. <!-- NONADDITIVE -->

Per family at most two anchors (`evidence/ios_device_evidence.json`, schema 3):

| Anchor | Configuration | Captured | Evidence |
|---|---|---|---|
| `resident` | rung 0, its first selection | once | completed captures, **or** its typed pre-allocation `capture_refused` quantum (allowed on the 8 GB phone) |
| `optimized` | the deepest rung the provider implements, its **last (cheapest)** selection — every saving engaged | **2 repeats** | must complete; a refusal is a defect to fix, never evidence |

Both anchors sit at the **reference geometry 1024²** (Supra2: 256²) and run **2 denoise steps**:

* 1024² is the largest ladder size. The fully optimized configuration must fit at every ladder size
  (see the fit table below), so it completes here by requirement, and this is the tightest cell —
  where an accurate low end matters — so it is measured, not derived.
* Desktop activation measurements do not prove an iPhone process-footprint transfer law.
  Intermediate geometry and optimization claims require the actual source certificate,
  applicable process model and matching endpoint evidence described below.
* Two steps, not N: one denoise step's working set is released before the next, so a short run's
  peak is a long run's; two steps also cross a step-to-step release. The `validation_steps` case
  checks this on one model.

**Derivation (`mlx-gen/src/ios_anchor_law.rs` and catalog `ios_admission.rs`).**
The former fitted scratch-ratio/resident-subtraction law is disabled. Do not
subtract a native upper bound from a process measurement, borrow an unmeasured
phase's overhead, or transfer a receipt between different family keys.

Provider certificates bind the actual source inventory, request and execution
facts. A native-allocation certificate remains a native-allocation result; it is
not a process-footprint result. Mapping it to process growth additionally needs
the declared physical/process and HOST/layout premises with their actual scope.
Missing source terms remain unproved. The SDK and campaign default policy does
not approve these premises. The engineering-model proposal in sc-26270 activity
26569 remains a separate unresolved acceptance decision at this snapshot.

Exact measured keys and intermediate derivations are distinct evidence routes.
A typed resident refusal is permitted by epic acceptance and supplies no measured
resident peak. The terminal repair must support that contract without inventing
a resident observation. No completed phone endpoint or complete intermediate
admission claim exists yet.

**Self-check.** Applicable completed endpoint observations must satisfy the
certified consistency checks. Any underprediction fails; the retired desktop
recapture percentage is not an allowed underprediction tolerance. A refusal or
unexecuted body cannot serve as a numeric consistency pass.

**Low end: admission is the measurement.** At the optimized anchor's own key admission charges the
largest of its 2 repeats' growth (kernel lifetime `phys_footprint` peak − the capture's starting
footprint) **plus the margin = the repeats' measured spread (max − min), floored at 32 MiB** (an
engineering floor so identical repeats never claim a zero margin; ~0.5% of the historical 6142 MiB
launch headroom). No counted-formula scratch and no 512 MiB formula reserve is stacked on a
measured anchor: `demand + margin` must fit live headroom (`runtime-ios`'s incremental recheck,
unchanged: only growth realized since admission is credited). The admitted load and every render
then run inside that **measured envelope** (`ios_quantum::open_admitted_envelope`): while the
process stays within it, a quantum guard needs only the envelope's remaining bytes when that is
less than its formula price; past the envelope every guard is back on its formula (the runtime
safety net). Captures run on the formula guards alone.

**Per-formula DEVICE observations.** Captures record each quantum's priced scratch
and observed kernel-footprint growth, with exact versus masked attribution.
`campaign.py evidence` reports these diagnostics. A ratio between a measured
interval and a formula is not by itself a source proof or a general permission
to discount future allocations. Preserve the source/phase/window identities and
the admission envelope's actual policy when interpreting these rows.

**Per-phase capture.** Every quantum check and synchronization reads the kernel lifetime peak
(`task_vm_info` REV3); growth inside an opened quantum belongs to that quantum's phase (the
provider's `ios_quantum_phase(label)`), glue after a synchronization to the next quantum's. A
phase is `exact` when the lifetime peak rose inside it, else an upper bound (masked by an earlier
phase). An unattributed phase is priced at the whole-run peak.

**Campaign size.** `campaign.py list --snapshots <dir of the route/tier slots that will be pushed>`
(the runner's own catalog, never a hand list) groups the catalog into anchor FAMILIES by the
slots' architecture digests and plans <!-- COUNT --> Without `--snapshots` the listing is
weights-free and conservative: every route is its own family. The 20,000-case budget is gone by
construction (`--max-cases` is an optional cap).

Use the current catalog's family keys over the actual provisioned snapshots.
Header similarity alone does not authorize borrowing between text, reference,
control or init graphs. Retain architecture, tier, sampling-allocation class and
accepted mode distinctions, including the SANA/Sprint, SDXL-family, Anima, Iris
generation/restoration/depth and Z-Image variants. Share endpoint captures only
where the source-owned catalog establishes the same family.

**Validation cases (part of the campaign, not optional).** For the first optimized family in
catalog order: `validation_steps` — the optimized anchor at the provider's full step count,
captured beside the anchor (`evidence` prints full/short growth, whole run and per phase); and
`validation_geometry` — the optimized configuration at 512², derived DOWN from the 1024² anchor.
For the first q4 family deriving a `control` member: `validation_mode` — an SDXL ControlNet
request at 1024² (the optimized selection), derived from the family's TEXT anchor by the additive-
mode law. Both admitted cases run with `campaign.py validate` once the anchors are committed: each
row records the anchor law's derived demand (`admitted.demand_bytes`, basis `Derived`) beside the
measured growth, and `validate` prints the error.

#### The fully optimized configuration fits every served cell (host-derived prediction)

Michael: "for the phone, we want to make sure the fully optimized will fit". Per provider route ×
tier × mode × size the fully optimized configuration (the optimized anchor's selection) is priced
by the providers' counted iOS pricing over the real snapshot headers — every phase's resident
bytes plus its largest counted quantum, plus the 512 MiB formula reserve — against the 6142 MiB
entitled launch headroom. **Predictions, not device evidence**; regenerate with each crate's
`reachable_sizes` / `reachable_sizes_prediction` ignored test, and the catalog's own request shape
with `IOS_PRICING_SNAPSHOTS=<dir of route/tier snapshot dirs> cargo test -p mlx-gen-ios-catalog
--lib host_integration -- --ignored --nocapture` (54 cells: SANA, Z-Image, SDXL routes).

| Provider / mode | Tier | Before (worst of 512/768/1024, MiB) | After (512 / 768 / 1024) | What made it fit |
|---|---|---:|---|---|
| SANA base / Sprint, text + reference | q4 / q8 / bf16 | 3808 / 3946 / 5566 | ≤ 3808 / ≤ 3927 / 5566 | already fit (rung 3; SANA withholds rung 4) |
| Anima Turbo, text | all | 2903 | 2052 / 2070 / 2903 | already fit (rung 4 is implemented on the iOS load; the "planned" label was wrong) |
| Z-Image Turbo, text + reference | q4 / q8 / bf16 | fit at the cheapest selection | ≤ 4659 / ≤ 4660 / ≤ 4885 | already fit |
| Z-Image control (+ reference), each kind | q4 | fit | 2758 / 3435 / 4659 | — |
| Z-Image control (+ reference), each kind | **q8** | **6383 / 7061 / 8009 ✗** | 3392 / 4070 / 5018 | the deferred (rung-4) control load packs only its 2 resident refiner blocks; each streamed window packs its own block (was: all 15 packed and kept resident) |
| Z-Image control (+ reference), each kind | bf16 | — | 4208 / 4886 / 5834 | same |
| Iris 3B text / prompt batch 2 | **bf16** | **load 11979–12014 ✗** | 1696–1997 / 1890–2757 / 2450–3873 | (a) Qwen3-VL text-tower layer streaming (rung 4, scope `te`/`both`); (b) rung-4 backbone block streaming on a deferred load |
| Iris 3B prompt batch 2, 1024 | **q8 / q4** | **7576 / 6152 ✗** | 3868 / 3751 | (b) rung-4 backbone streaming ((c) the per-row pixel stage alone leaves the trunk block binding) |
| Iris restore / depth | **bf16** | **11944 / 11946 ✗** | 2018 / 1400–1955 | (b) streamed blocks + deferred skeletons |
| SDXL family, text | all | fit | ≤ 2210 (bf16) / ≤ 2341 / ≤ 2317 | rung 4 priced as the iOS load actually runs it (the U-Net resident only as its non-block remainder) |
| SDXL family, reference / mask | **bf16** | **6270 / 6718 ✗ (768 / 1024)** | 2076 / 2076 / 2428 | (b) rung 4 priced correctly on the iOS load |
| SDXL family, control | **bf16** | **8127–8469 ✗** | 4085 / 4360 / 4770 | (b) |
| SDXL family, ip_adapter | **bf16** | **7590–8097 ✗** | 3425 | (b) + (d) a staged IP request releases the CLIP-ViT-H encoder and Resampler once its tokens exist |
| SDXL / RealVisXL, ip_adapter 1024 | **q8** | **6261 ✗** | 3375 | (b) + (d) |
| Illustrious v1/v2 (cached q8 revision: no packed marker, rung 2 deepest), control / ip_adapter | q8 | 5987 / 6261 ✗ | 5972 / 5599 | load floors follow `Weights::materialize` batching with cast sources held (fixes a 44% cast-load under-count); (d) |

No cell is physically impossible. The largest indivisible tensor anywhere is Iris's bf16 Qwen3-VL
token-embedding table, 151936 × 2560 × 2 B = 742 MiB: with the 70 MiB that stays resident and the
512 MiB reserve, 1324 MiB. The tightest fully optimized cell is Illustrious q8 ControlNet on the
cached (marker-less) q8 revision, 5972 MiB (170 MiB margin); on a packed-marker q8 revision it
streams at rung 4 like RealVisXL (3748 MiB). Every rung-0 RESIDENT configuration above 512² (and
most at 512²) is predicted over the headroom: those anchors are expected to be typed pre-allocation
refusals, which is their evidence.

Z-Image Turbo host-derived reachable sizes (sc-26433), kept as the pre-anchor
record of the quality-first selection walk the campaign NO LONGER performs (it
captures the cheapest selection as the optimized anchor and derives the rest) —
**predictions, not device evidence**. Per mode, tier and edge: the peak of every phase (its resident bytes
plus its largest counted quantum's live input and scratch, `Backend::Metal`, plus
the 512 MiB reserve) against the 6142 MiB entitled launch headroom, under each
rung-4 selection in the capture ladder's quality-first order. Unstaged cells keep
every component loaded; staged (`s_`) cells drop the text encoder before denoise
and the DiT before decode; scope `dit` streams the DiT's blocks (text encoder
resident), `te` the encoder's layers (DiT resident), `both` both. The img2img and
control encodes run the bounded encode on the device (every full-resolution layer
tiled at 256 input pixels, `IOS_ENCODE_TILE_PX`); the untiled encode is what
refused every 768/1024 reference/control cell before it. The control route adds
its resident embedder and 2-block refiner, its 15 streamed main blocks (dense
read + packing at the tier) and its 17 live hints; its checkpoint bytes are
derived from the published architecture (not provisioned on the host).
`ZIMAGE_TIER_ROOT=<snapshot> cargo test -p mlx-gen-z-image --lib
reachable_sizes -- --ignored --nocapture` regenerates it from the snapshot's
headers alone. "First" is the ladder's first selection
(`t768_o64_c67108864_w1_dit`); "fits" is the first selection the ladder reaches
that fits.

| mode | tier | edge | first selection peak | first fitting selection | its peak | binding quantum |
|---|---|---:|---:|---|---:|---|
| text | q4 | 512 | 7722 | t448_o64_c67108864_w1_dit | 5554 | decode tiled upsample |
| text | q4 | 768 | 13282 | t320_o64_c67108864_w1_dit | 5872 | decode tiled upsample |
| text | q4 | 1024 | 10915 | t448_o64_c67108864_w1_both | 6101 | decode tiled upsample |
| text | q8 | 512 | 9988 | t448_o64_c67108864_w1_both | 4138 | decode tiled upsample |
| text | q8 | 768 | 15548 | t512_o64_c67108864_w1_both | 5642 | decode tiled upsample |
| text | q8 | 1024 | 13180 | t320_o64_c67108864_w1_both | 5971 | decode tiled resnet |
| text | bf16 | 512 | 14236 | t448_o64_c67108864_w1_both | 5138 | decode tiled upsample |
| text | bf16 | 768 | 19796 | t384_o64_c67108864_w1_both | 5777 | decode tiled upsample |
| text | bf16 | 1024 | 17429 | s_t512_o64_c67108864_w1_both | 5966 | decode tiled upsample |
| reference | q4 | 512 | 7724 | t448_o64_c67108864_w1_dit | 5556 | decode tiled upsample |
| reference | q4 | 768 | 13284 | t320_o64_c67108864_w1_dit | 5874 | decode tiled upsample |
| reference | q4 | 1024 | 10917 | t448_o64_c67108864_w1_both | 6103 | decode tiled upsample |
| reference | q8 | 512 | 9990 | t448_o64_c67108864_w1_both | 4140 | decode tiled upsample |
| reference | q8 | 768 | 15550 | t512_o64_c67108864_w1_both | 5644 | decode tiled upsample |
| reference | q8 | 1024 | 13183 | t320_o64_c67108864_w1_both | 5974 | decode tiled resnet |
| reference | bf16 | 512 | 14238 | t448_o64_c67108864_w1_both | 5140 | decode tiled upsample |
| reference | bf16 | 768 | 19798 | t384_o64_c67108864_w1_both | 5779 | decode tiled upsample |
| reference | bf16 | 1024 | 17431 | s_t512_o64_c67108864_w1_both | 5966 | decode tiled upsample |
| control (each kind) | q4 | 512 | 7941 | t448_o64_c67108864_w1_dit | 5773 | decode tiled upsample |
| control (each kind) | q4 | 768 | 13501 | t320_o64_c67108864_w1_dit | 6091 | decode tiled upsample |
| control (each kind) | q4 | 1024 | 11134 | t384_o64_c67108864_w1_both | 5919 | decode tiled upsample |
| control (each kind) | q8 | 512 | 10400 | t448_o64_c67108864_w1_both | 4550 | decode tiled upsample |
| control (each kind) | q8 | 768 | 15960 | t512_o64_c67108864_w1_both | 6054 | decode tiled upsample |
| control (each kind) | q8 | 1024 | 13593 | s_t512_o64_c67108864_w1_dit | 5964 | decode tiled upsample |
| control (each kind) | bf16 | 512 | 15011 | t448_o64_c67108864_w1_both | 5913 | decode tiled upsample |
| control (each kind) | bf16 | 768 | 20571 | t256_o64_c67108864_w1_both | 5969 | decode tiled resnet |
| control (each kind) | bf16 | 1024 | 18204 | s_t512_o64_c67108864_w1_both | 5991 | denoise control block |
| control_reference (each kind) | q4 | 512 | 7943 | t448_o64_c67108864_w1_dit | 5775 | decode tiled upsample |
| control_reference (each kind) | q4 | 768 | 13503 | t320_o64_c67108864_w1_dit | 6093 | decode tiled upsample |
| control_reference (each kind) | q4 | 1024 | 11136 | t384_o64_c67108864_w1_both | 5921 | decode tiled upsample |
| control_reference (each kind) | q8 | 512 | 10403 | t448_o64_c67108864_w1_both | 4553 | decode tiled upsample |
| control_reference (each kind) | q8 | 768 | 15963 | t512_o64_c67108864_w1_both | 6057 | decode tiled upsample |
| control_reference (each kind) | q8 | 1024 | 13596 | s_t512_o64_c67108864_w1_dit | 5964 | decode tiled upsample |
| control_reference (each kind) | bf16 | 512 | 15014 | t448_o64_c67108864_w1_both | 5916 | decode tiled upsample |
| control_reference (each kind) | bf16 | 768 | 20574 | t256_o64_c67108864_w1_both | 5971 | decode tiled resnet |
| control_reference (each kind) | bf16 | 1024 | 18207 | s_t512_o64_c67108864_w1_both | 5995 | denoise control block |

(MiB.) A selection that does not fit is expected to stop at its first
over-headroom quantum as `capture_refused` — that refusal, not a jetsam kill, is
the evidence — and the ladder then tries the next cheaper selection at the same
size. At q4 / 1024 / text the walk refuses the 7 `dit` and 7 `te` selections and
`t768`–`t512` `both`, then completes at `t448_o64_c67108864_w1_both` (6101 MiB);
the staged `s_t512_o64_c67108864_w1_dit` (Michael's Mac configuration, tile 512)
predicts 5963 MiB.

The following are **future session recipes, not instructions to run now**. Phone
availability alone does not authorize a full-model diagnostic. Use a fresh evidence
directory and save the checkout SHA, OS, entitlement, conversion/source receipts
and catalog JSON. Every selected case starts a fresh app process, before legacy
LLM checks or kernel smokes consume memory. The default is admitted, never
inherited `IOS_SMOKE_UNADMITTED`. Execute only after the relevant safety criteria
are independently met. The session builds/installs **once**, then launches one-shot
cases without `--terminate-existing`. `--timeout-seconds` (default 1800) is passed
to the device's cooperative `CancelFlag` deadline and controls host polling, not
an unrelated 600-second media loop. After a case returns, the app writes its
report and exits itself; the host verifies exit before advancing or finishing.
On deadline/cancellation failure it records a harness failure and stops further
launches if exit cannot be proved. It never hard-kills an active device render.
Timed-out host commands own a new process group, so cargo/xcodebuild/devicectl
children are terminated together rather than orphaned. Cleanup/exit verification
has a separate bounded grace window. Launch exit codes are retained.

`--memory-entitlement on|off` applies to the once-installed build. Every case
requires a fresh matching `launch-memory.txt`; missing copies are failures.
Each case/probe has its own host artifact directory alongside the JSONL file.
`--max-cases` (optional; the anchor listing is finite by construction) counts renders,
repeats and next-up probes; exhausted cases are explicit skips. Direct `run_smoke.sh --campaign-case` flags delegate to this same driver,
never to the legacy reinstall/terminate-existing path.

```sh
# The anchor campaign. List the families over the slots that will be pushed (header reads only):
python3 scripts/ios/campaign.py list --snapshots "$SLOTS" > /tmp/ios-campaign-catalog.json
# Step 1 — capture every resident, optimized (x2) and validation_steps case
# as a labelled bounded DEVICE CAPTURE (no evidence needed: every quantum of every registered model
# checks live OS headroom BEFORE constructing its graph). A resident `capture_refused` row is that
# anchor's evidence; an optimized anchor that does not complete is a DEFECT (exit 2). No size or
# selection ladder is walked and nothing runs that is not in the listing.
python3 scripts/ios/campaign.py capture --catalog /tmp/ios-campaign-catalog.json \
  --all-cases --device "$DEVICE" --output "$EVIDENCE/anchors.jsonl"
# One family only (e.g. to re-capture after a fix):
python3 scripts/ios/campaign.py capture --catalog /tmp/ios-campaign-catalog.json \
  --provider sana_1600m --tier q4 --mode text --device "$DEVICE" --output "$EVIDENCE/sana-q4.jsonl"
# Step 2 — candidate schema-3 ANCHOR rows for the single reviewed table, PRINTED for review (plus
# the full-steps / short-steps validation and any problem: < 3 completed optimized repeats, a
# refused optimized anchor). A human commits them to
# crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json.
python3 scripts/ios/campaign.py evidence --input "$EVIDENCE/anchors.jsonl"
# Step 3 — after the anchors are committed and the app rebuilt: the validation_geometry case runs
# ADMITTED, so its row records the anchor law's derived demand next to the measured growth.
python3 scripts/ios/campaign.py validate --catalog /tmp/ios-campaign-catalog.json \
  --all-cases --device "$DEVICE" --output "$EVIDENCE/validate.jsonl"
# Any configuration of the catalog's domain, ADMITTED (an anchor key admits at its measurement,
# anything else at its anchor-law derivation):
python3 scripts/ios/campaign.py run --catalog /tmp/ios-campaign-catalog.json \
  --case sana_1600m/sana_1600m/q4/768/text/3/s_t192_o48_c1048576/0 --device "$DEVICE" \
  --output "$EVIDENCE/sana-768.jsonl"
python3 scripts/ios/campaign.py table --input "$EVIDENCE/anchors.jsonl"
```

Each case emits `CAMPAIGN_JSON` and JSONL with elapsed seconds, sampled MLX
active/cache/active+cache peaks, MLX allocator peak, sampled Darwin
`TASK_VM_INFO.phys_footprint` peak (words 36–37), the kernel's exact lifetime
`ledger_phys_footprint_peak` (words 42–43; the full SDK `TASK_VM_INFO_COUNT` of 93 words
is requested, because only a REV3+ reply of at least 84 words carries the peak — a shorter reply
is recorded as an explicit `kernel_lifetime_peak_unavailable` reading, never a silent null), and minimum
`os_proc_available_memory()` headroom. Rows also carry the path-free artifact
layout identity, the full memory-relevant `evidence_key` (hardware, provider,
route, tier, layout, mode, geometry, references, execution levers, token window;
never prompt/seed/path), `evidence_class` (`admitted` or `capture`), any typed
`refused_quantum`, selected strategy parameters, the structured `anchor_selection`,
`steps`, the per-phase `phase_peaks` (`{phase: {peak_bytes, exact}}`, split from the
kernel lifetime peak through the quantum marks), an admitted case's `admitted`
price (`demand_bytes`, `margin_bytes`, `basis`: measured anchor or derived), evidence revision,
entitlement, refusal-only marker, per-launch nonce, compiled build SHA and iOS
version. The nonce prevents a previous report with the same case/SHA from passing
freshness checks. Render
filenames are case-specific and registered in `renders.txt`, including each
image of a real two-prompt batch. **RSS and MLX active+cache are not kernel
physical footprint.** Sampling is 20 ms and can miss transients; these are
observations, not an independent instantaneous working-set proof. Unavailable
readings are null, never fabricated zeros. Statuses are `successful` (admitted
image only), `admitted_refused`, `diagnostic_success` (a completed capture, never
production), `capture_refused` (a capture whose quantum refused before
construction), `diagnostic_failure`, or `skipped` (explicit host case budget). Missing/stale
reports, timeout, malformed output or a terminated process cannot prove admission
refusal. Missing physical-footprint/headroom telemetry cannot yield production
success. Noncompletion returns nonzero; a successful listing is not a render.
The table includes each row's purpose, active/cache/combined columns, probe markers,
any optimized-anchor DEFECT, and per anchor case its completed repeats' demand
(kernel lifetime peak − capture baseline) or its typed refusal. Refusals have typed
`refusal_kind`: `capacity`, `no_evidence`, `mode`, `strategy`, or `artifact`,
taken from runtime-ios's typed refusal, never parsed from a message. Only a
**capacity** refusal proves the next-up condition (an admitted capacity refusal,
or a capture's typed quantum refusal); missing footprint evidence, unsupported
modes/rungs, malformed sources and termination never prove it. Layout-profile
and request-binding errors are failures before admission, not capacity refusals.
Commit the actual measured anchors only after the terminal campaign, not this
harness change. The 4096 MiB budget remains provisional (admission still checks a
projected absolute footprint against it, so until sc-16780 measures the cap a
measured demand above it is a typed capacity refusal); 6142/3374 MiB launch
readings remain unvalidated diagnostics, and no cap/table is invented here.

### Bounded synthetic primitive characterization (sc-16780)

**Primitive characterization only: not a model render, not admission evidence
for any model, not a cap.** The reserved key `primitive/sana_block_quantum` is
not a catalog case and opens no model file or snapshot. One cold launch runs one
SANA-block quantum boundary: a synthetic f32 activation `[1,8,8,2240]` (256 px
latent) against a resident F16 1x1 convolution weight `[11200,1,1,2240]` shaped
like `conv_inverted`, so MLX materializes the F32-promoted weight copy inside
`ios_quantum::run` (live headroom recheck, eval, stream synchronize, zero-cache
drain). Before **any** allocation it refuses unless live
`os_proc_available_memory()` covers the requirement derived from the budget
module: `mlx_gen_sana::quantum::block_bytes(configured_block_quantum(sana_1600m, 1, 8, 8, 300, UNBOUNDED))`
+ the probe's resident activation/weight bytes + `PROCESS_RESERVE_BYTES` =
119.043 + 48.398 + 512 = **679.441 MiB** (712,445,952 bytes). Residents are
charged because they are allocated before the quantum's own live recheck of
floor + reserve. The floor is the counted SANA block (sc-26270): the promoted
f32 copy of the F16 inverted conv plus the largest of the self-attention,
cross-attention and score-plus-Mix-FFN phases at 64 tokens.

The row (`evidence_class: primitive_characterization`,
`model_admission_evidence: false`, `diagnostic: true`) records `before`,
`resident` (weights materialized) and `after` readings — available bytes, current
phys_footprint, exact kernel lifetime peak, MLX active/cache/peak — plus the
sampled extrema, entitlement, build SHA, iOS version and nonce.
`quantum_peak_over_baseline_bytes` is the new lifetime peak minus the pre-probe
footprint (null when an earlier excursion masks it), and
`unaccounted_by_mlx_peak_bytes` subtracts MLX's own peak: the device/driver
overhead the 512 MiB engineering reserve currently guesses. The host validates
the label, nonce and every reading, or rejects the row.

```sh
# ~5-10 min install (cargo ios build + xcodebuild), then one launch of seconds.
# Requires the phone unlocked; the increased-memory-limit entitlement is on.
SMOKE_BUNDLE_ID=com.michaeltrefry.SceneWorksSmoke DEVELOPMENT_TEAM=R2526H7YN9 \
  python3 scripts/ios/campaign.py primitive \
  --device 2EB6F380-7996-5CA7-9C12-39199D18199F --memory-entitlement on \
  --output "/tmp/ios-primitive-$(date +%Y%m%d-%H%M%S).jsonl"
```

## 1. Prerequisites and checkout

Use an Apple-silicon Mac, Xcode with iPhoneOS SDK and Metal toolchain, the
committed Rust toolchain, a paired/unlocked phone with Developer Mode enabled,
and an Apple Developer account signed into Xcode. XcodeGen 2.46 is installed on
the development Mac. Do not use a simulator for memory evidence.

```sh
# From the isolated checkout of feature/sc-16774-ios-runtime (or its docs story).
export PATH="$HOME/.cargo/bin:$PATH"
git rev-parse HEAD
rustc --version
xcrun -f metal
xcodegen --version
rustup target add aarch64-apple-ios
xcrun devicectl list devices

# Use the identifier returned above, not a device name or a guessed field number.
export DEVICE='<paired-device-identifier>'
export DEVELOPMENT_TEAM=R2526H7YN9
export SMOKE_BUNDLE_ID=com.michaeltrefry.SceneWorksSmoke
# Use a fresh evidence directory for this session.
export EVIDENCE="$HOME/ios-evidence/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EVIDENCE"
git rev-parse HEAD > "$EVIDENCE/source-sha.txt"

# Record BEFORE invoking the harness, including attempts that fail before launch.
# This wrapper deliberately returns normally so recovery is not skipped by set -e.
# Its saved exit-status file, NOT the wrapper's return, is the harness outcome.
run_attempt() {
  ATTEMPT=$1; EXPECTED_LIMIT=$2; shift 2
  ATTEMPT_STARTED=$(date +%s)
  printf '%s\n' "$ATTEMPT_STARTED" > "$EVIDENCE/$ATTEMPT.started-epoch"
  printf '%s\n' "$EXPECTED_LIMIT" > "$EVIDENCE/$ATTEMPT.expected-entitlement"
  if "$@" > "$EVIDENCE/$ATTEMPT.log" 2>&1; then
    ATTEMPT_STATUS=0
  else
    ATTEMPT_STATUS=$?
  fi
  printf '%s\n' "$ATTEMPT_STATUS" > "$EVIDENCE/$ATTEMPT.exit-status"
  printf 'Harness exit %s; log: %s\n' "$ATTEMPT_STATUS" "$EVIDENCE/$ATTEMPT.log"
}
```

Signing is automatic. Other developers supply their own team/bundle ID. Changing
the bundle ID or uninstalling the app loses access to the existing model
container and requires a re-push. Keep the phone unlocked through launch.
Michael permits phone use anytime; that is not a standing Mac GPU loan. Any
host real-weight/GPU measurement needs a separately agreed bounded window and
external RSS/physical-footprint guard. No such loan is active for this docs work.

## 2. Install and reproduce the entitlement A/B

Run without `--media` to install the host and measure launch headroom without
requesting SANA renders. The LLM load is refused by default when no applicable
footprint estimate exists; model absence may also yield skipped checks.

```sh
# Start with a clean diagnostic environment, not inherited unadmitted/soak knobs.
unset IOS_SMOKE_UNADMITTED IOS_SMOKE_SANA_CAPTURE IOS_SMOKE_IMAGE_ONLY IOS_SMOKE_ONLY
unset IOS_SMOKE_SOAK_SECS MLX_GEN_SANA_DECODE_TILE
for entitlement in on off; do
  expected=YES; [ "$entitlement" = on ] || expected=NO
  for n in 1 2 3; do
    run_attempt "launch-$entitlement-$n" "$expected" \
      scripts/ios/run_smoke.sh --device "$DEVICE" --memory-entitlement "$entitlement"
    # Stop the sequence, but keep the failed status/log and enter section 6 recovery.
    if [ "$ATTEMPT_STATUS" -ne 0 ]; then break 2; fi
    cp /tmp/ios-smoke-launch-memory.txt "$EVIDENCE/$ATTEMPT.launch-memory.txt"
    cp /tmp/ios-smoke-report.txt "$EVIDENCE/$ATTEMPT.report.txt"
  done
done
```

Check each command's exit status and fresh report before proceeding; on failure
use the recovery steps below, not a prior `/tmp` report. `run_smoke.sh` checks
staticlib/report freshness and verifies the launch record's entitlement tag.
The control entitlement retains extended virtual addressing and removes only
increased-memory-limit. Use `--memory-entitlement off`, not an invented
`SMOKE_ENTITLEMENTS` override.

Observed S3 launch readings: **6142 MiB on**, **3374 MiB off**, three identical
readings each. These are `os_proc_available_memory()` at launch, not a validated
safe working-set cap. The runtime's provisional budget constants remain
unchanged. Do not replace them with 6142 or derive a safe render budget from the
entitlement delta. MLX allocator limits are not a proven jetsam barrier.

## 3. Provision and inspect the actual SANA snapshot

Inference downloads nothing. Supply an already fetched/converted complete SANA
component tree (`transformer/`, `text_encoder/`, `vae/`, plus required configs
and tokenizers). Do not rely on a directory's q4 name. The failed target attempt
had a dense **1,179,648,000-byte embedding**. Packed q4/q8 Gemma embedding support
is implemented, but that does not retroactively change the pushed artifact.

```sh
export SNAPSHOT='/absolute/path/to/provisioned/sana-component-tree'
# Header-only profiler: no model construction or tensor payload evaluation.
cargo run --locked -p mlx-gen-ios-catalog --example snapshot_profile -- "$SNAPSHOT" \
  > "$EVIDENCE/sana-profile.txt"
scripts/ios/push_model.sh "$SNAPSHOT" sana --device "$DEVICE" \
  > "$EVIDENCE/push-sana.log" 2>&1
```

Inspect `layout_sha256`, all three component byte totals and embedding dtype and
shape. Keep the conversion source and component precision with the receipt.
The profile is layout identity, not a hash of weight values or proof of image
quality. The helper stages symlinks as real files and checks transferred sizes.
The installed app must exist first; the remote tree is `Documents/sana/`.

## 4. Admitted SANA check

```sh
run_attempt sana-admitted YES env IOS_SMOKE_ONLY=sana \
  scripts/ios/run_smoke.sh --media --device "$DEVICE" --memory-entitlement on
if [ "$ATTEMPT_STATUS" -eq 0 ]; then
  cp /tmp/ios-smoke-report.txt "$EVIDENCE/$ATTEMPT.report.txt"
fi
# If nonzero, retain the saved status/log and use section 6 before another attempt.
```

The regular SANA harness configurations are 1024/tile192, 1024/tile128 and
512/tile256. The 512 row is off-distribution for the 1024 checkpoint and is not
a quality benchmark. On the current feature, the artifact-bound admitted path
refuses before loading: historical observations cannot authorize this snapshot
on this phone. Inspect the named checks for refusal or skip even when the report
header says PASS. That header is not a successful admitted image result.

## 5. Explicit UNADMITTED evidence capture

This is a diagnostic experiment, **not production admission or a successful
production render**, even if a PNG is produced. It requires a real device and
explicit `--unadmitted`. The capture path selects only 256/tile128, not the
legacy untiled experiment; there is no arbitrary size flag for SANA.

**Mandatory BEFORE capture:** independently establish and record a conservative
**total process working-set bound for the actual snapshot and exact capture
configuration**. The record must identify the layout/component precisions,
source, phone/OS/entitlement, 256/tile128 execution configuration, derivation and
margin, and account for weight conversions, simultaneously resident tensors,
activations, allocator/cache and driver/process overhead. Compare that bound
conservatively with the available capacity for this attempt. The header-only
payload/profile and the harness's automated payload/cap/headroom preflight
**DO NOT satisfy this prerequisite**: they cannot bound these transients.
Neither launch headroom nor the provisional constant establishes such a bound.
**Do not execute capture if this independent bound is missing or unsupported.**
No numerical safe bound is supplied by this runbook, and none is asserted to
exist for the target snapshot. Merely creating a receipt file is not evidence.

Only after the operator has reviewed and recorded that independently supported
bound, set `CAPTURE_BOUND_RECORD` to its receipt and run:

```sh
# Leave unset unless the mandatory total-working-set prerequisite above is met.
if [ -n "${CAPTURE_BOUND_RECORD:-}" ] && [ -s "$CAPTURE_BOUND_RECORD" ]; then
  cp "$CAPTURE_BOUND_RECORD" "$EVIDENCE/capture-total-working-set-bound.txt"
  run_attempt sana-UNADMITTED-capture YES env \
    IOS_SMOKE_ONLY=sana IOS_SMOKE_SANA_CAPTURE=1 \
    scripts/ios/run_smoke.sh --media --unadmitted \
    --device "$DEVICE" --memory-entitlement on
else
  printf '%s\n' 'NOT RUN: independent total-working-set bound is missing.' >&2
fi
# If attempted, retain its saved exit status even on failure and enter section 6.
```

Keep the exit status, launch-memory record, progress trace, full report and
render manifest. Generated PNGs are pulled to `~/Desktop/sana-ios/` by default
(`ARTIFACT_DIR` overrides this). Copy the current session's artifacts into its
evidence directory. Do not set `IOS_SMOKE_IMAGE_ONLY`: it selects a different,
legacy diagnostic sequence beginning with untiled decode.

No completion of this capture or target image success is asserted here. The
previous 1024/tile192 target attempt ended before a final report; termination
cause remains **unconfirmed**, not proven jetsam.

## 6. Recover evidence from a failed/terminated run

A missing report can mean launch failure, crash, termination or slow execution.
A progress file proves how far execution got, not why it stopped. Retrieve the
matching timestamped device crash/jetsam diagnostic before assigning a cause.

Use the `ATTEMPT` recorded **before that invocation**, never the current time or
another attempt's receipt. A failure before launch or before progress truncation
can leave both device files from an older run. A local copy's creation/mtime is
not freshness evidence. Preserve the device-reported modification timestamps in
the metadata JSON below; if metadata is unavailable, keep files **unverified**.

```sh
# Retrieval failures must not abort the remaining recovery steps under set -e.
unset DEVICE_PROGRESS_MTIME DEVICE_LAUNCH_MTIME  # Never reuse a prior transcription.
if xcrun devicectl device info files --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
  --subdirectory Documents --no-recurse \
  --json-output "$EVIDENCE/$ATTEMPT.device-files.json" \
  > "$EVIDENCE/$ATTEMPT.device-files.log" 2>&1; then
  printf '%s\n' 'Device metadata saved; inspect each source modification timestamp.'
else
  printf '%s\n' 'Metadata unavailable: recovered files remain UNVERIFIED.' >&2
fi
for name in sana-progress.txt launch-memory.txt; do
  if xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
    --source "Documents/$name" \
    --destination "$EVIDENCE/$ATTEMPT.$name.unverified" \
    > "$EVIDENCE/$ATTEMPT.$name.copy.log" 2>&1; then
    printf '%s\n' "Copied $name as UNVERIFIED, not current-attempt evidence."
  else
    printf '%s\n' "Unavailable: $name; preserve the copy log." >&2
  fi
done
cat "$EVIDENCE/$ATTEMPT.exit-status"  # Original harness outcome, never overwritten.
```

Inspect the saved device metadata for **each exact source file**, and record its
modification time as Unix epoch seconds, retaining the original device timestamp
and timezone alongside that transcription. Do not substitute the copy timestamp.
The following classifier takes that device-derived value, not a guessed JSON
field (Xcode metadata schemas may differ). Unknown/missing timestamps, times at
or before the pre-invocation boundary, and mismatched entitlement are not current
evidence. The equality case is conservatively ambiguous at one-second precision.

```sh
classify_recovered() {
  file=$1; device_mtime=$2; kind=$3
  started=$(cat "$EVIDENCE/$ATTEMPT.started-epoch")
  expected=$(cat "$EVIDENCE/$ATTEMPT.expected-entitlement")
  case "$device_mtime" in
    ''|*[!0-9]*) printf '%s\n' UNVERIFIED; return ;;
  esac
  if [ ! -f "$file" ]; then printf '%s\n' MISSING; return; fi
  if [ "$device_mtime" -le "$started" ]; then
    printf '%s\n' STALE_OR_AMBIGUOUS; return
  fi
  if [ "$kind" = launch ]; then
    case "$expected" in YES|NO) ;; *) printf '%s\n' UNVERIFIED; return ;; esac
    if ! grep -q "increased-memory-limit = $expected\$" "$file"; then
      printf '%s\n' WRONG_ENTITLEMENT; return
    fi
  fi
  printf '%s\n' CURRENT
}
# Leave unknown until transcribed from the preserved DEVICE metadata, separately.
classify_recovered "$EVIDENCE/$ATTEMPT.sana-progress.txt.unverified" \
  "${DEVICE_PROGRESS_MTIME:-}" progress > "$EVIDENCE/$ATTEMPT.progress-freshness"
classify_recovered "$EVIDENCE/$ATTEMPT.launch-memory.txt.unverified" \
  "${DEVICE_LAUNCH_MTIME:-}" launch > "$EVIDENCE/$ATTEMPT.launch-freshness"
```

Only `CURRENT` files may be attributed to this attempt. Fresh progress alone
does not establish the entitlement: require a `CURRENT` launch record matching
YES/NO as well; otherwise keep build/entitlement attribution unverified. Store
all stale/mismatched files as historical evidence, not this run's measurements.
The script also attempts `/tmp/ios-sana-progress.txt` on failure; apply the same
attribution discipline. Preserve logs immediately: `/tmp/ios-smoke-report.txt`
and launch-memory filenames are reused. Retrieve crash/termination reports through Xcode Devices and Simulators
or the phone's Analytics Data, matching app, time, device and source receipt.
Do not label a timeout or a breadcrumb as a jetsam diagnosis.

Common failures: locked device (`FBSOpenApplicationErrorDomain`), missing Xcode
account for the signing team, changed bundle/container, incomplete snapshot,
stale staticlib/report, or a macOS metallib packaged for iOS. Fix the specific
failure; never bypass freshness/platform checks to get a green report.

## 7. Evidence ledger: Mac engineering, not phone fit

Existing authorized Mac runs establish engine behavior, not E4 target admission.
No GPU measurements were run for this documentation update.

| Evidence | Exact source/provenance | Result and limitation |
|---|---|---|
| Iris conditioning, fixed-input denoiser, renders, resident bytes | `1d64ac261df315000ef68a81d19db2ee4ac95455`, merged via #1275 `2f3265c71`; local logs `/tmp/sc16774-rerun3` | All passed; q8 backbone+tower 7.283 GiB, q4 4.200 GiB. Not process footprint or phone fit. |
| SDXL/Illustrious sequential CFG and all-three low-memory pipeline | Earlier `770fa87c28a9c5efcbe8e942c10d36e86dd79827` run, `/tmp/sc16774-rerun2` | Sequential CFG byte-identical; all-three MLX peak 3.2289 GiB, NOT process footprint (sampled suite footprint reached 12.736 GiB). Predates final fp16 accumulation fix; NOT a remeasured final pipeline claim. |
| Final VAE parity | `7b5684a1bdc9bf9ec5273351be9cb05febbc962e`, merged via #1274 `a54ad6c3a`; `/tmp/sc16774-vae-final/source-and-build.txt`, `metrics.json`, `full.log` | 26/26 comparisons passed: absolute pixel error max ≤3 and mean ≤0.25, both in 8-bit pixel units (equivalently normalized max ≤3/255 and mean ≤0.25/255). fp16 tile256 overflow fixed by narrowly widening convolution accumulation to f32 with f16 storage. |
| Final VAE memory | Same final source/run | Genuinely untiled 1024 f32 reference peak 7.69 GiB; sweep maximum MLX peak 10.48 GiB; sampled process footprint 8.971 GiB. VAE test, not full-pipeline or target-phone evidence. |

Historical SANA data on the 12 GB iPhone 17 Pro Max is unbound planning data:
its active+cache values are **not** kernel `phys_footprint`, and it is not the
8 GB target. Do not import its old fit/no-fit verdicts into the target table.

## 8. Terminal device campaign — not yet measured

sc-26270 owns the ANCHOR campaign (per served provider route × tier × mode: the
resident anchor once and the optimized anchor twice at 1024² with 2 steps, plus the
validation cases), feature review, pin and main integration. Retain every
registered mode; report explicit refusals, not missing cells. Add real rows here
only after captures, with this schema:

| Source SHA / artifact layout and component tiers | Phone / OS / entitlement | Provider / mode / size / strategy | Time | MLX peak | Kernel footprint | Min headroom | Completed / refused / unconfirmed termination | Image / diagnostic receipt |
|---|---|---|---|---|---|---|---|---|
| Not yet captured | iPhone 16 Pro Max / iOS 26.6.2 | Full required model/tier set remains outstanding | — | — | — | — | No target image success recorded | — |

A successful diagnostic capture does not populate an admitted-success row.
Do not invent measurements, infer safe caps from launch headroom, or silently
mark the explicitly deferred Supra2 iOS registration complete.
