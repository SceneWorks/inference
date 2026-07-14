# Runtime 2026.07.1 reconciliation checkpoint

Date: 2026-07-13

## Why this release exists

`runtime-2026.07.0` completed the repository consolidation and remains an
immutable, valid release. Before SceneWorks could cut over, however, the legacy
MLX and Candle generation repositories received additional product work. The
SceneWorks application was already pinned beyond the legacy heads used for
`runtime-2026.07.0`.

Moving the existing tag or dropping those product deltas would make the cutover
unreproducible. The follow-up therefore preserves `runtime-2026.07.0`, imports
the exact revisions consumed by SceneWorks, and publishes the reconciled tree as
`runtime-2026.07.1`.

## Exact source boundary

The final legacy product cutoffs are:

- `mlx-gen`: `c9faefc5d2e0650dd5679745089f2c03be51ba7a`
- `candle-gen`: `6709463b8b13e076dd6c499575d47188e8546c07`

Those are product cutoffs, not a promise to follow future legacy repository
activity. Their individual first-parent effects and canonical destination are
recorded in
[`post-import-reconciliation.toml`](post-import-reconciliation.toml).

## Canonical adaptation

The product deltas were reconciled in canonical inference commit
`592f373f2d37278b8799aa7e0482dabb9f5c68c1`. The replay preserves the runtime
behavior while retaining the architectural constraints established during the
consolidation:

- provider discovery remains an explicit composition concern;
- model registrations remain owned by provider crates and are collected by
  explicit family and platform catalogs;
- memory footprints flow through the provider registry rather than a global
  inventory;
- no force-link anchors or process-global loader registry were reintroduced;
- all packages continue to share the normalized workspace contracts, backend
  pins, and root lockfile.

This includes the post-cutover generation work for Krea, SeedVR2, LTX,
PID/early-stop behavior, progress contracts, adaptive memory policy, tiled VAE
paths, `text_style_gain`, per-component footprints, and SANA Sprint.

## Validation boundary

The reconciled revision passed the following local gates before release
publication:

- workspace structure, explicit-registry, and pinned-backend validation;
- all migration-script unit tests;
- contract test suites;
- strict all-target Clippy for contracts and every MLX and Candle generation
  package;
- the complete default MLX generation test matrix, including available
  real-checkpoint SeedVR2 coverage;
- the complete default Candle generation test matrix;
- formatting and staged-diff checks.

Release publication additionally requires the immutable candidate/final-tag
workflow in [`../../release/README.md`](../../release/README.md), including
deterministic bundle verification and hosted platform/real-weight gates at the
exact tagged revision.

## Consumer cutover

SceneWorks must consume `runtime-2026.07.1` by immutable tag through the named
runtime bundle. It must not retain direct dependencies on the retired legacy
repositories or recreate their global discovery behavior in application code.

Once the SceneWorks cutover is green, future inference changes land in this
repository and follow its release train. The legacy product cutoffs above remain
historical provenance only.
