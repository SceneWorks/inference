# Runtime 2026.07.2 reconciliation checkpoint

Date: 2026-07-14

## Why this release exists

`runtime-2026.07.1` reconciled the legacy MLX and Candle generation repositories
into the monorepo through the SceneWorks product cutoffs (`mlx-gen` `c9faefc`,
`candle-gen` `6709463`). Its checkpoint recorded those cutoffs as historical
provenance and stated that future inference work would land in this repository.

That expectation did not immediately hold. After the cutover, epic 10834
("MLX unified-memory fit-gate + sequential component residency") continued and
**completed in the legacy repositories on 2026-07-14**, one day after
`runtime-2026.07.1` was published:

- `mlx-gen`: `c9faefc` → `45428fa` (+18 commits — the sequential-residency fan-out)
- `candle-gen`: `6709463` → `ef84441` (+1 commit — a `gen-core` pin re-bump only)
- `sceneworks-worker`: capability-derived fit-gate selection (consumer repository,
  out of scope for `inference`)

The monorepo was therefore one epic behind its own declared cutover. This
follow-up imports that epic's MLX provider work, adapts it to the explicit-catalog
architecture, verifies it on real weights, and publishes the reconciled tree as
`runtime-2026.07.2`.

## Exact source boundary

Final legacy product cutoffs reconciled here:

- `mlx-gen`: `45428fa9727c569f3f3723c7343c96b0944f9007`
- `candle-gen`: `ef84441c82222df8f63701e761c0834c1699ceb0`

Legacy development on these repositories is **halted** pending the durable cutover
decision — whether the monorepo becomes the sole live development line. These
cutoffs are the reconciliation boundary, not a commitment to track further legacy
activity; while the halt holds there is none to track. The individual first-parent
effects and canonical destination are recorded in
[`post-cutover-reconciliation-epic-10834.toml`](post-cutover-reconciliation-epic-10834.toml).

## Canonical adaptation

The MLX residency fan-out was reconciled in canonical inference commit
`757dbe6f` (branch `claude/reconcile-sc10834-residency`). The replay preserves the
runtime behavior while retaining the architectural constraints established during
the consolidation:

- provider discovery remains explicit — the legacy `inventory::submit!` / force-link
  registration seam was **not** reintroduced (the workspace gate forbids `inventory`);
- residency body changes were 3-way merged onto the monorepo's explicit
  `mlx_gen::register_generators! { pub(crate) const … }` registration constants;
- `Capabilities.supports_sequential_offload` was fanned out to the wired families
  (flux, flux2, chroma, sd3, sana, boogu, bernini, anima, ideogram, kolors; scail2
  is a test-only staging gate) — the worker's capability-derived selection consumes
  this without an allowlist;
- the default `OffloadPolicy::Resident` path is byte-untouched (zero parity risk);
- the eight new `sequential_residency_real_weights.rs` suites were re-homed off the
  removed global `mlx_gen::load(...)` onto `provider_registry().load(...)`.

Two source deltas required **no** change under the monorepo's model:

- `candle-gen`'s post-cutoff commit is a `gen-core` SHA re-pin only; internal
  dependencies here are workspace paths, so it is a no-op.
- `gen-core`'s `OffloadPolicy` + `Capabilities.supports_sequential_offload` contract
  predates the cutoff (epic 10765 / sc-10821); the fan-out touches no contract file.

This also carries the one non-residency delta in the same commit range — the krea
render-time resolution lever (sc-11749, epic 8459).

## Validation boundary

Reconciled revision `757dbe6f` passed the following on a Mac M5 / 128 GB /
macOS 26.5.1 (NAX fast path live):

- workspace structure, explicit-registry, and pinned-backend validation
  (`scripts/check-workspace.py`);
- strict all-target Clippy for the complete MLX generation set + `runtime-macos`
  (`-D warnings`);
- the default MLX generation/parity/conformance matrix (515 test binaries, 0 failed)
  including the `runtime-macos` catalog exact-surface test;
- the LLM-only (`--no-default-features`) `runtime-macos` bundle;
- **real-weight Sequential↔Resident A/B**, closing the verification debt epic 10834
  left open as sc-11922 for the families whose snapshots are present:

  | family | text encoder dropped | Resident peak | Sequential peak | saved | output |
  |---|---|---|---|---|---|
  | `sana_1600m` (1024² @ 12) | Gemma | 21.485 GiB | 16.615 GiB | 4.870 GiB (22.7%) | byte-identical |
  | `anima_base` (768² @ 8) | Qwen3 | 9.579 GiB | 8.468 GiB | 1.110 GiB (11.6%) | byte-identical |

  Both A/B suites also assert repeat-job bounding (no component stays resident
  across generations). Remaining fan-out families were not A/B'd here for
  environmental snapshot reasons only (absent or stub HF-cache entries; kolors
  needs an assembled ChatGLM3-tokenizer snapshot) — the same carry-over class as the
  source epic's sc-11922, with the Resident path proven byte-untouched per file.

Release publication additionally requires the immutable candidate/final-tag workflow
in [`../../release/README.md`](../../release/README.md), including deterministic
bundle verification and hosted platform gates at the exact tagged revision.

## Consumer cutover

SceneWorks and ChatWorks consume `runtime-2026.07.2` by immutable tag through the
named runtime bundle. Whether future inference changes originate here rather than in
the legacy repositories is the open cutover decision this checkpoint documents but
does not itself resolve; legacy development is halted until it is made.

## Post-release resolution

The source-of-truth decision is resolved on canonical `main` by
[`PHASE_5_CONSUMER_CUTOVER.md`](PHASE_5_CONSUMER_CUTOVER.md). This release-tag checkpoint remains a
historical record of the decision state at tag time; the closeout records the final consumer,
authority, published-ref, and rollback outcomes without moving this immutable release tag.
