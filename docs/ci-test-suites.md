# Automatic controls and separately dispatched backend suites

Ordinary PR, push, merge-group and tag CI compile/link the original selected backend
test targets, run strict lint and documentation checks, and execute only the exact
backend controls in `scripts/ci/safe_controls.json` through `audited-safe` mode.
Tensor-neutral contract packages retain their ordinary suites. Learned model,
module, layer, inference-kernel, generation, encode/decode and training/optimizer
execution belongs in separately dispatched validation, including CPU execution
with tiny or synthetic inputs.

The former open-ended routine partition was insufficient: an unregistered learned
test could run automatically. The small safe-control table is closed. Each entry
binds package, target kind/name, exact case, source, libtest protocol, platform and
source-review rationale. Every other backend target is **compiled only**, never
reported as a passed test. No Rust assertions, ignore attributes or test bodies
are changed by this separation.

## Execution boundary

`audited-safe` compiles the complete selected graph with
`--no-run --message-format=json`. Before listing any approved executable, it binds
its target/source to Cargo metadata and checks the package manifest for the
standard libtest harness. Unknown targets, including custom harnesses, are never
probed with a guessed `--list`. Approved cases must exist, be non-ignored and run
with `--exact`; the runner checks one passing case, zero failures/ignores and the
expected filtered count. Missing or renamed controls, wrong package binding,
custom harnesses and zero-case executions fail. Cargo working directory, manifest
environment and dynamic-loader paths are preserved.

`scripts/check_model_inference_policy.py` binds the exact safe runner and control
table hashes. Its ordinary workflow exception accepts only `audited-safe`;
`routine` and `exhaustive` are manual-only. Broad/direct backend `cargo test`
commands, including unknown listings, are rejected. Direct tensor-neutral
contract suites and the existing runtime-ios compile-fail interface doctests
remain ordinary checks. Python tooling uses a fresh venv with no user/system
site packages, unset Python path/home, pinned numpy 2.4.3, safetensors 0.8.0 and
PyYAML 6.0.3, and asserts that Torch, MLX, transformers, diffusers, accelerate and
tokenizers are absent before running tooling tests.

## Required baseline and exhaustive validation

Dispatch the existing workflow against the prepared candidate:

```sh
gh workflow run model-test-suites.yml --repo SceneWorks/inference --ref YOUR_REF \
  -f test_suite=baseline
gh workflow run model-test-suites.yml --repo SceneWorks/inference --ref YOUR_REF \
  -f test_suite=exhaustive
```

The baseline is required delivery evidence for this execution-policy migration.
It retains the original ordinary suite commands/assertions, with the original
routine partition semantics for the two CPU groups and full MLX graph:

| Existing hosted job | Baseline execution retained |
| --- | --- |
| Candle CPU, base | `--lib` over candle-llm, candle-quant-kernels, **candle-audio*** and runtime-cpu; separate candle-llm conformance; all named SA3 integration targets; runtime-cpu no-default LLM, audio and media profiles |
| Candle CPU, gen | `--lib --tests -p 'candle-gen*'`, retaining all integration targets |
| MLX Metal | Full original `--lib --tests` package graph, including runtime-ios, neutral packages and mlx-llm/campaign; LLM-only runtime-macos/runtime-ios; both device-capture surfaces; macOS audio; both Candle Metal reduction/reference-latent integration modules; iOS host media library with its original smoke exclusion |

`exhaustive` remains the default for both direct dispatch and `workflow_call`,
including the existing real-weights `model-exhaustive-tests` bridge. It executes
registered exact cases in `scripts/ci/test_suites.json`, including registered
ignored cases. That registry and bespoke real-weight/H3/MOSS acceptance routes
are unchanged. NAX and CUDA numerical execution, physical GPU1 binding, rank
sharing, compute-capability requirements and cache isolation remain dispatch-only
under their existing contracts. Baseline runs on hosted Ubuntu/macOS, avoiding
the developer Mac GPU.

The baseline proves the former default suites actually execute. It is not a
per-case benchmark, full real-model acceptance or proof of asset-gated cases.
Ignored, cfg-absent, missing-asset or self-skipped cases are omissions, never model
acceptance passes. Required real-weight/product acceptance remains separate.

## Evidence and bounds

Each hosted manual job always uploads logs, exact commands/features, candidate
SHA and tracked source/fixture tree identities, toolchain and device data,
selection and both registries. Evidence must be associated with its exact run and
candidate; failed or omitted execution remains visible. Synthetic fixtures are
identified by the tracked source that constructs them. External asset omissions
do not become passed acceptance merely because the suite exits successfully.

Automatic suite deadlines remain the existing 4500s CPU/3600s full MLX overrides
and 90-minute jobs. Safe control execution is bounded to 120s per exact case;
metadata/listing to 120s/60s and compilation to 2400s, all within the invocation
deadline. Routine manual targets retain 900s and exhaustive cases 600s bounds;
selected manual invocations allow 7200s and jobs retain 150/180-minute limits.
Timeouts terminate child process trees. Dependency-aware package selection,
compilation, strict lint, caches, failure propagation and gate job IDs remain.

CI_IMPACT: no new jobs or automatic learned computation. Execution moves from
ordinary checks to the existing dispatched jobs. A prior full hosted MLX job took
about 53 minutes, including about 28 minutes in its test step; this is historical
evidence, not a forecast. Added manual compilation/download and full baseline
runtime are unknown until the required candidate dispatch. No new benchmark or
calibration campaign is introduced. Local verification uses synthetic runner and
policy transcripts only, with no model execution.
