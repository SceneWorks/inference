# Contributing

This repository is the canonical source for SceneWorks inference contracts,
engines, provider families, and named platform runtime bundles. Its assembled Git
history preserves the source repositories that preceded it.

- Do not rename public crates, provider IDs, serialized fields, or weight keys as
  incidental cleanup; treat those as compatibility changes.
- Keep contract crates tensor-neutral.
- Compose media providers through family and platform catalogs; do not add media
  `inventory` submissions, global loaders, or force-link anchors.
- Validate named platform bundles; `--workspace --all-features` is not a supported
  universal configuration.
- Release product-consumed changes through immutable `runtime-*` tags after the
  affected hosted and platform-owned gates pass.
- Record migration compatibility evidence and release-boundary decisions under
  `docs/migration/`.

See
[`docs/architecture/inference-rearchitecture.md`](docs/architecture/inference-rearchitecture.md)
for the rationale and accepted tradeoffs behind these boundaries.
