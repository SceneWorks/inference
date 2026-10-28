# Repository Guidelines

## Project Structure & Module Organization

This is the unified SceneWorks inference workspace. `crates/contracts/` holds
backend-neutral APIs and conformance testkits; `crates/llm/` contains MLX and
Candle LLM engines; `crates/kernels/` holds shared low-precision Candle kernels
(`candle-quant-kernels`: NVFP4 / FP8 / INT8 cuBLASLt paths, plus the fused decode
primitives in `fused_decode` and the NVFP4 decode GEMV in `nvfp4_gemv`) used by
candle-gen and candle-llm — new runtime-compiled
CUDA kernels go through the nvrtc compile-once seam (`nvrtc::KernelSource::compiled`),
never a private `compile_ptx` + `load_module`; `crates/media/` contains MLX and Candle media-provider families;
and `crates/bundles/` defines supported CPU, macOS, CUDA, and catalog
compositions. Keep implementation in a crate's `src/`, integration tests in
`tests/`, examples in `examples/`, and model/tokenizer assets beside the owning
provider. Architecture and migration evidence live under `docs/`; release
fixtures and tooling live in `release/` and `scripts/release/`.

## Build, Test, and Development Commands

- `./scripts/check-workspace.py` validates workspace membership and dependency
  boundaries; run it after manifest or crate-layout changes.
- `python3 scripts/check_clock_assertions.py --check-baseline .` must pass
  before every push: it is the ratchet on wall-clock-dependent test assertions
  (baseline `scripts/clock_assertions_baseline.txt`; `--write-baseline` only
  with a recorded justification), and CI's `workspace` job runs the same check.
- `scripts/git-hooks/pre-push` runs both checks automatically before each push
  (seconds-fast; no builds, no tests). Install it once per clone with
  `git config core.hooksPath scripts/git-hooks`; skip a single push with
  `git push --no-verify`.
- `python3 -m unittest discover -s scripts/tests -v` runs repository tooling
  tests.
- `cargo test --locked -p core-llm -p core-llm-testkit` runs a focused contract
  suite. Use `-p <crate>` for the smallest relevant package set.
- `cargo fmt -p sceneworks-gen-core -p sceneworks-gen-core-testkit --check` and
  `cargo clippy --locked -p <crate>` provide the baseline formatting and lint
  checks.

Do not rely on `cargo test --workspace --all-features`: it is not a supported
configuration. Validate the named platform bundle affected by the change and
follow the matching CI lane for feature flags such as `--features cuda`.

## Coding Style & Naming Conventions

Write idiomatic Rust, using `rustfmt` formatting (four-space indentation) and
`snake_case` modules/functions, `UpperCamelCase` types, and `SCREAMING_SNAKE_CASE`
constants. Preserve existing public crate names, provider IDs, serialized fields,
and weight keys unless making an explicit compatibility change. Keep contract
crates tensor-neutral. Compose media providers through family/platform catalogs;
do not introduce `inventory` registrations, global loaders, or force-link anchors.
Review authors and remediation work must also follow the repository-wide
[code-review finding reference convention](CONTRIBUTING.md#code-review-finding-references).

## Testing Guidelines

Add focused unit tests near behavior and integration/conformance tests under the
crate's `tests/` directory. Name Rust test files after the capability (for
example, `tests/conformance.rs`) and Python tests `test_*.py`.

### Model execution is manual-only (PR and main policy)

Normal pull-request and `main` CI **never executes model inference or training**.
This includes learned NN module/block/layer forward or backward, generation, and
training on tiny or synthetic fixtures: small weights do not make model execution
a kernel or metadata test. Model implementations and tests may land on `main`,
but their execution belongs to the explicit `model-inference-proofs.yml` workflow,
not to ordinary CI and not to an automatic campaign at each pin change.

- Keep all applicable test targets compiled and linted. Newly moved model tests
  use unconditional `#[ignore = "model inference; explicit model-proof workflow"]`.
  Existing unconditional ignores may retain their original prerequisite reasons
  verbatim; registration and exact late BODY/device protocols are still required.
  Do not replace model work with an early-return PASS or drop assertions/cases.
- Raw matmul/conv/SDPA/norm/allocator operations, schema/math/tokenizer/header
  checks, mocks, and cheap constructor/load-only checks remain ordinary tests.
  A learned module's `.forward` is not exempt merely because its implementation
  eventually calls one of those raw kernels. Heavy whole-model pack proofs are
  explicit manual workloads too.
- New or changed tests must be classified and registered in
  `scripts/ci/model-inference-proofs.json` when model work is reachable. Ordinary
  workflows must not override those ignores. The ordinary structural policy
  check (`scripts/check_model_inference_policy.py`) enforces bidirectional source BODY/
  registry coverage (plus new canonical tags) and rejects automatic model execution
  paths, including runnable learned-model Rustdoc seams.
- Manual proofs select exact registered cases/backends and explicit external
  resource guards. Evidence records the exact source/fixture-model identity,
  configuration, actual selected compute-device SET inside each work scope
  (separate from host/default metadata), nonzero completed bodies/pass counts,
  existing outputs/metrics/logs and duration. Missing assets, skipped bodies,
  absent device observations, failed or partial children are **not proof**.
- Preserve every test, assertion and supported model/tier/control surface.
  Source inspection, compilation, test discovery, or ordinary CI success is not
  a model-runtime PASS and does not waive epic/device acceptance criteria.
  Proofs run only on explicit dispatch with queue-preserving physical resource
  coordination; never cancel running inference to admit another dispatch.
- The source-guarded Candle Iris rank entry is a **parent-only NN helper**: retain
  its original unconditional ignore and exact same-executable child guard, register
  its exact nested-rank parent bindings, and never fabricate a standalone helper
  BODY. It is checked by compiled-ignore discovery but cannot be selected or accepted
  as its own proof. Nested-rank successful-libtest-summary counts describe normal
  test-process completion, not successful training: preserve original parent JSON,
  `RankLost`/`SplitWorld`, and intentional rank-loss assertions. Failed libtest
  summaries still reject the complete case; a pre-NN early return is not NN proof.

See `docs/ios-device-runbook.md` for the explicit proof and evidence commands.

## Commit & Pull Request Guidelines

Use concise, imperative Conventional Commit-style subjects, such as `ci: select
Git Bash on the Windows runner` or `docs: record completed inference release`.
Keep changes scoped by crate or platform. PRs should explain the affected
contracts/providers/bundles, link the issue when available, list validation run,
and include compatibility or migration evidence in `docs/migration/` when a
release boundary changes. Report security issues through the process in
`SECURITY.md`, not in public issues.
