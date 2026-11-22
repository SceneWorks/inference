# FLUX.1 base T2I comparison

The existing dispatch-only `model-inference-proofs.yml` accepts `flux1_pair=tier1`
or `tier2` with the exact selection `flux1-base-pair`. Each dispatch runs both
registered variants on one physical backend in one job, from the pre-migration
engine `2b09938e33370915d4f853389961709b8bbf8bf9` and the dispatched head.
Base calls the registry loader and public `generate`. Head compiles each plan
before loading its provider and calls `generate_planned` on that exact plan.
Each tier and backend has an independent receipt.

| Mode | Inputs and claim |
|---|---|
| Tier 1, migration batch | Deterministic synthetic BF16 weights from the immutable [snapshot header inventory](../../../scripts/ci/flux1_pair/headers.json) and [tensor geometry](../../../scripts/ci/flux1_pair/geometry.tsv). One full-width double and single FLUX block, one full-width CLIP and T5 block, full VAE. This compares the actual provider pipeline at reduced depth. |
| Tier 2, feature end | Complete 19/38-block Q4 snapshots. Dev uses `SceneWorks/flux1-dev-mlx@323fd12d79f78ad444e882e8d8e871914584f2b9/q4` (9,614,909,969 input bytes), schnell uses `SceneWorks/flux1-schnell-mlx@bba3ae01dfd94089f173c05edd4e1a4c551f2599/q4` (9,609,145,903 bytes). Input hashes must match the committed snapshot files before and after the pair. MLX uses its Q4 load option; Candle autodetects its packed snapshot. This witness covers the two plain T2I rows. |

Re-derive every retained tensor location and shape using the pinned HF cache
with `python3 scripts/ci/flux1_pair/fixture.py "$HUB"`. This reads safetensors
headers only; neither source weight payloads nor generated tensors enter Git.

**Frozen comparison.** One resident load per process; seed 104729, prompt “red
kite”, 256×256, count one, three Euler outer steps, variant-native scheduler,
dev guidance 3.5, schnell no guidance, no adapters, reference, memory rung
or decoder override. Preserve backend-native mixed precision, latent RNG and
schedule. The CUDA cuBLAS workspace setting is `:4096:8` for both arms. MLX uses
its unified-memory allocator and shipping macOS 26.2 NAX deployment floor; Candle uses its native CUDA allocator. No cross-backend
numerical or memory comparison is performed.

Both executables are built before the first render. Head preflight on both
variants checks registration, 256 packed tokens of width 64, the seed and the
full native schedule before any learned forward. The fixed setup-only numerical
bound is 2×10⁻⁷ against the analytic native schedule. The migration fidelity
bound is **zero**. Observe the current native sampler tensor at each of the three
Euler evaluations and its terminal tensor, actual compute device and state dtype,
the four f32 sigma bit patterns, ordered outer steps and final RGB bytes. Require
finite, changing states and one nonuniform complete RGB image. Compare each
same-backend pair by exact SHA-256 and publish both PNGs. Neither the observation
taps nor the Tier 1 depth substitutions change production trees; the receipt
identifies each temporary overlay, executable, toolchain, fixture and source tree.
An instrumented render is fidelity evidence. PERF throughput and memory acceptance
require the terminal performance sampling method.

**Bounded cost.** Per backend and mode: two builds, two pre-load plan inspections
and four render processes. Each learned process has a maximum deadline of 20
minutes. A CUDA claim budgets six such child allowances, one allowance for
claim acquisition/admission, and a final reserve of 30 seconds plus one
termination grace and one poll interval. Each child must still fit its original
deadline before that reserve or it is refused before launch. The job has the
existing six-hour timeout including build and setup.
Tier 1 generates about 2.00 GB for dev and 1.98 GB for schnell on CPU. Tier 2
reads provisioned files from the existing `MLX_GEN_MODELS_ROOT` or
`CANDLE_GEN_MODELS_ROOT`; missing files fail preflight. The existing supervisor
bounds child footprint and on CUDA GPU allocations. The existing physical-device
lease covers the comparison. The artifact keeps each attempted child’s
bounded, redacted stdout, stderr, supervisor outcome and available observer
events. The verifier checks each published log’s allowed name, size and digest.
Failure receipts retain the logs produced before refusal or cleanup. No automatic retry, per-PR campaign or model
execution in ordinary CI.

For Tier 1 on MLX, use the [secondary Mac planner policy](mlx-route-plan.json)
and the actual published branch with this controller:

```sh
request="$(jq -c '.request | .selection = "flux1-base-pair"' docs/evidence/sga-eng-1a/mlx-route-plan.json)"
gh workflow run model-inference-proofs.yml --repo SceneWorks/inference --ref "$PUBLISHED_BRANCH" \
  -f backend=mlx -f flux1_pair=tier1 -f request="$request"
```

For CUDA start with [the CUDA policy](cuda-route-plan.json), verify the selected
physical UUID and its free-memory/cap budget from the live pool, remove the
sample conformance configuration and select `flux1-base-pair`. MLX requires
`runner=nax`, routed via `rw-llm` with enforced `nax-macos-2` identity.
Only select Tier 2 for the feature-end real-weight campaign. A mismatch, skipped
body, timeout, asset or setup error leaves the comparison unaccepted and retains
the failure receipt where available. The two probe checkouts and generated
weights are private temporary job files.

## ENG-4b terminal real-weight comparison

Select `eng4b-tier2` and request `flux1-eng4b-pair` on `candle-cuda`.
Both arms use locked Cargo release builds and the planned route. The reference
is the frozen native-format revision `be45bdf746c5d4a12d4db76f9653b8686e7f33f8`.
Both arms include the shared native checkpoint loaders. The head is the exact
dispatched revision and tree. Both use the full Q4 Dev and Schnell
snapshots and the request settings above. The controller verifies the
published file hashes before and after the comparison. One instrumented
pair per variant checks bit-exact states and RGB8 before the performance
samples start.

The SGA PERF-1 and PERF-2 contract uses six fixed pairs per variant. Each
pair uses fresh processes and alternates which revision goes first. Each
measured process loads once, completes one identical warm-up, then times
one warm resident generation. Its end-to-end timer covers planned generation
through materialized host RGB. Its seconds per outer step come from the
synchronized interval from the first Euler evaluation through the terminal
state, divided by three. Evidence writes occur outside the timed intervals.
Every measured run completes the frozen plan and produces the fidelity
pair's RGB8. One physical CUDA claim covers the entire comparison.

The timing statistic is the median of the six paired head/base ratios.
The smallest and largest observed ratios bound the population median with
96.875% coverage for independent pairs. Each two-sided interval passes
when its upper bound is at most 1.02 and fails when its lower bound exceeds
1.02. Crossing the ceiling is inconclusive. A result at the fixed sample
budget requires a decision before any new run. Host memory uses the Windows
process peak working set. Device memory uses the CUDA allocator's continuous
live and driver-reserved pool high-water counters. These matched process
peaks cover load, warm-up and generation. The head maximum for each domain
must be at most 1.02 times the base maximum. External guard samples and the
driver-wide allocation bound remain diagnostics and safety controls.

The fixed budget is two builds, four model-free preflights, four fidelity
renders, and 24 measured processes, each with one warm-up and one timed
render. Set the child deadline to at most 360 seconds and keep the existing
one-second guard poll, 24 GiB child host cap and 18 GiB child GPU cap. At the
360-second limit the physical claim allows 198 minutes plus the existing
release reserve. The existing six-hour job timeout also covers builds and
asset verification. By default the workflow requires provisioned cache entries.
Set `"download_missing": true` in the `flux1-eng4b-pair` request to stage absent
files from the exact pinned Hub snapshots before verification.
The profile measures CUDA resident Q4 Dev and Schnell. FLUX.1 conditional
modes use `conditional-tier1` and its separate immutable asset bindings.
Native BFL and the ControlNet and PuLID injection routes retain their
own model proof cases. Their evidence applies to those cases' settings.

### ENG-4b conditional fidelity

Select `eng4b-conditional-tier1` on Candle CUDA and use the
`flux1-eng4b-conditional-pair` request. The controller builds the exact merged
ENG-4a baseline and dispatched head in private Git worktrees. Both arms compile
and use the same plan. The profile covers ControlNet pose, XLabs IP Dev and
PuLID fake CFG. It binds the historical shape-faithful BF16 fixture, source
image bytes and real overlay inputs from the accepted
[ControlNet](https://github.com/SceneWorks/inference/actions/runs/42462655901),
[XLabs IP](https://github.com/SceneWorks/inference/actions/runs/43558292122)
and [PuLID](https://github.com/SceneWorks/inference/actions/runs/42460488791)
receipts. The profile compares current outputs within each new pair with
bit-exact state, event and RGB8 requirements.

For each case, reuse its archived receipt's `request`, set `selection` to
`flux1-eng4b-conditional-pair` and use the existing 250 ms guard poll.
Each dispatch builds twice and runs two model-free preflights and two renders.
The child deadline is at most 20 minutes. At that limit, the claim budgets
100 minutes plus its normal release reserve. The 24 GiB host and 18 GiB device
caps apply. This comparison covers conditional fidelity on the fixed fixture.
The complete real-Q4 Dev and Schnell timing and memory comparison uses the
separate `eng4b-tier2` profile.

If the target Windows listener lacks these inputs, dispatch the existing
asset inventory once with `backend=candle-cuda`, `flux1_pair=none`,
`stage_pinned_assets=true` and this request for the affected listener.

```json
{"selection":"flux1-asset-inventory","runner":"cuda-windows-2","mode":"hash","cases":["control-pose","ip-dev","pulid-fake"]}
```

This setup stages only absent files from the union of the selected cases.
It checks the sealed SHA-256 before publishing each download and hashes all
selected installed files before reporting the cases ready. A digest mismatch
leaves the existing file in place and fails the setup with an inventory receipt.
