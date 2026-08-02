# Runtime release train

Inference releases use immutable calendar-versioned tags:

- release candidate: `runtime-YYYY.MM.patch-rc.N`
- final release: `runtime-YYYY.MM.patch`

The patch number increments for each runtime release in a calendar month. A final
tag may only be created from the exact revision of its passing release candidate;
the `runtime-*` tag is never moved or reused.

Build a dry-run release bundle from a clean checkout:

```sh
python3 scripts/release/build_release.py --tag runtime-2026.07.0-rc.0 --offline
python3 scripts/release/verify_release.py dist/release --offline
```

The bundle contains:

- a deterministic source archive rooted at `inference-<tag>/`;
- `runtime-manifest.json`, listing all workspace package versions, the complete
  lockfile identity, Rust toolchain, backend sources/revisions, and artifact
  hashes;
- an SPDX 2.3 JSON SBOM for the complete cross-platform lockfile graph;
- `<tag>.model-licenses.json`, the **model-weight-licence manifest** (sc-13332, `schema_version` 3
  since sc-16663) — the reviewed licence families, one row per loaded model artifact, and the
  provider→component mapping with each provider's derived term union;
- `SHA256SUMS` covering the source archive, manifest, SBOM, and model-licenses manifest.

## Model-weight licences (`release/model-weight-licenses.json`)

The SPDX SBOM covers the licence of every Cargo crate (the *source* axis). Model **weights** are a
separate axis cargo tooling never sees: each provider loads its own pinned checkpoints, whose
licences a consumer must be able to surface on its end-product licences page.

**This surface is disclosure-only.** It records what upstream licence texts *name* so a consumer can
show a user. Nothing in it blocks, gates, degrades or withholds anything, and nothing added to it
ever should. Whether a given use is permitted is the consumer's evaluation of these facts against its
own situation — its revenue, whether it redistributes weights, which agreements it has with its own
users — and inference has none of that information.

The schema has three fact sections (see `sceneworks-gen-core::license`):

- `families` — one entry per reviewed upstream licence text, with its typed terms. Every term is
  backed by a verbatim quote in `docs/licensing/sc-16662-licence-family-evidence.md`.
- `components` — one row per **loaded artifact**, not per provider. Each carries `source_url`,
  the verbatim `declared` identifier, the `family` it normalizes to, whether the upstream distributes
  it `gated`, and the `retrieved` date the declaration was read. An artifact loaded by several
  providers is one row that all of them point at.
- `providers` — the registry id, its component keys, and its **derived** term union. Derived, never
  hand-authored: a hand-typed "effective licence" is a second place to be wrong and can drift from
  the components it claims to summarize.

Provider crates own their component rows beside their pinned `HUB_REPO`/`HUB_REVISION`;
`candle-audio-catalog` aggregates them into the committed
`release/model-weight-licenses.json`. Two gates keep it honest:

- `candle-audio-catalog::every_shipped_provider_has_a_weight_license` — the ship-gate in the
  composition root: a provider that reaches the catalog without resolving to well-formed component
  rows fails the build, so **no audio provider can ship without its weight licence recorded**;
- `candle-audio-catalog::component_licenses_manifest_matches_committed_file` — the drift gate: the
  committed JSON must equal what the catalog produces (regenerate with
  `UPDATE_WEIGHT_LICENSES=1 cargo test -p candle-audio-catalog component_licenses_manifest_matches_committed_file`).

`build_release.py` copies the committed manifest into the bundle (and fails if it is absent or
structurally incomplete); `verify_release.py` re-checks the bundled copy.

**What schema 2 stored, and why it is gone.** The retired schema recorded a `commercial_use` boolean
— a legal *conclusion* depending on facts inference does not have. Several shipped checkpoints had no
correct value: every Stable Audio 3 registration carried `commercial_use: false`, yet the Stability
AI Community License does not prohibit commercial use at all; it names a revenue threshold and a
registration. A wrong boolean is worse than an absent field, because a join computed over it reads as
authoritative. sc-16663 migrated the audio lane onto the layers above and deleted the flag; the
Python validator rejects any row that still carries it.

The verification step checks the manifest/SBOM relationship graph, requires the
landed runtime provider bundles, and builds a small external Cargo consumer
against the contract crates extracted from the source archive. The external
consumer smoke remains deliberately contract-only.

## Release gates

Before a final tag is created:

1. Workspace, contracts, affected backend/platform, documentation, and
   supply-chain lanes pass for the candidate revision.
2. Required real-weight profiles pass against the revisions recorded in the
   real-weight fixture manifest.
3. The source bundle is rebuilt from the candidate revision and passes
   `verify_release.py` without `--allow-dirty` or `--skip-smoke`.
4. The uploaded artifact hashes match `SHA256SUMS`.
5. The final tag is created at the candidate revision; artifacts are rebuilt and
   attached without changing source.
