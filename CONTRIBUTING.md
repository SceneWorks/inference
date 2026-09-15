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

## Code-review finding references

Historical review documents under `crates/media/` allocated identifiers such as
`F-026` independently. A bare finding identifier can therefore name several
unrelated findings and is not a durable citation.

The repository convention is:

- Do not renumber or sweep legacy review documents and citations solely to repair
  this historical ambiguity. When touching a legacy bare citation, use Git history
  and its linked Shortcut story to resolve the original review cycle; do not infer
  its meaning from whichever in-tree document happens to contain the same number.
- Allocate identifiers for every new review cycle from one repository-wide,
  monotonically increasing sequence. Find the next id with:

  ```sh
  python3 scripts/check-review-findings.py --next
  ```

  The command returns `F-182` at adoption. A new review must use a
  `CODE_REVIEW_*.md` document with finding headings such as `#### [F-182]`, then
  append one `review<TAB>start<TAB>end<TAB>document` row to
  `docs/code-review-finding-allocations.tsv` in the same change. Never restart at
  `F-001` or maintain a backend-local sequence. Registry rows are permanent even
  if a finding is later withdrawn, so an assigned id can never be reused.
  Allocations remain provisional until merge: sync the latest `origin/main`,
  renumber if another review merged first, and run
  `python3 scripts/check-review-findings.py --base origin/main` immediately before
  merging. CI compares the registry to the PR base and rejects modified or removed
  rows, changed legacy ranges, gaps, duplicate allocations, and registry/document
  drift.
- Once a finding has a Shortcut story, use the globally unique `sc-NNNNN` story
  id as the durable citation in source, tests, configuration, commits, and
  remediation notes. The review id may remain as qualified context, for example
  `sc-12345 (F-182, 2026-08-01 review)`, but never as a bare `F-182` reference.
- Carrying a missing historical review document into this repository does not
  make its finding ids unique and is not a substitute for the story citation.

See
[`docs/architecture/inference-rearchitecture.md`](docs/architecture/inference-rearchitecture.md)
for the rationale and accepted tradeoffs behind these boundaries.
