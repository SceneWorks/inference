# Repository Guidelines

## Project Structure & Module Organization

This is the unified SceneWorks inference workspace. `crates/contracts/` holds
backend-neutral APIs and conformance testkits; `crates/llm/` contains MLX and
Candle LLM engines; `crates/media/` contains MLX and Candle media-provider
families; and `crates/bundles/` defines supported CPU, macOS, CUDA, and catalog
compositions. Keep implementation in a crate's `src/`, integration tests in
`tests/`, examples in `examples/`, and model/tokenizer assets beside the owning
provider. Architecture and migration evidence live under `docs/`; release
fixtures and tooling live in `release/` and `scripts/release/`.

## Build, Test, and Development Commands

- `./scripts/check-workspace.py` validates workspace membership and dependency
  boundaries; run it after manifest or crate-layout changes.
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

## Testing Guidelines

Add focused unit tests near behavior and integration/conformance tests under the
crate's `tests/` directory. Name Rust test files after the capability (for
example, `tests/conformance.rs`) and Python tests `test_*.py`. Avoid requiring
real weights or accelerator hardware for ordinary tests; those belong in the
separate ignored or platform CI gates.

## Commit & Pull Request Guidelines

Use concise, imperative Conventional Commit-style subjects, such as `ci: select
Git Bash on the Windows runner` or `docs: record completed inference release`.
Keep changes scoped by crate or platform. PRs should explain the affected
contracts/providers/bundles, link the issue when available, list validation run,
and include compatibility or migration evidence in `docs/migration/` when a
release boundary changes. Report security issues through the process in
`SECURITY.md`, not in public issues.
