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

**Frozen comparison.** One resident load per process; seed 104729, prompt “red
kite”, 256×256, count one, three Euler outer steps, variant-native scheduler,
dev guidance 3.5, schnell no guidance, no adapters, reference, memory rung
or decoder override. Preserve backend-native mixed precision, latent RNG and
schedule. The CUDA cuBLAS workspace setting is `:4096:8` for both arms. MLX uses
its unified-memory allocator, Candle its native CUDA allocator. No cross-backend
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
minutes; the job has the existing six-hour timeout including build and setup.
Tier 1 generates about 2.00 GB for dev and 1.98 GB for schnell on CPU. Tier 2
reads provisioned files from the existing `MLX_GEN_MODELS_ROOT` or
`CANDLE_GEN_MODELS_ROOT`; missing files fail preflight. The existing supervisor
bounds child footprint and on CUDA GPU allocations. The existing physical-device
lease covers the comparison. No automatic retry, per-PR campaign or model
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
