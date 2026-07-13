# Contributing

The repository is in a staged history-preserving migration.

- Do not refactor files under `imports/` during Phase 1.
- Do not rename public crates, provider IDs, serialized fields, or weight keys as
  part of a path/history move.
- Keep contract crates tensor-neutral.
- Validate named platform bundles; `--workspace --all-features` is not a supported
  universal configuration.
- Record compatibility and tree-equivalence evidence under `docs/migration/`.

