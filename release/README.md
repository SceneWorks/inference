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
- `<tag>.model-licenses.json`, the **model-weight-license manifest** (sc-13332) — one row per
  shipped audio provider recording the license of its pinned Hugging Face weight checkpoint
  (SPDX id, license name, source URL, attribution, and a `commercial_use` flag);
- `SHA256SUMS` covering the source archive, manifest, SBOM, and model-licenses manifest.

## Model-weight licenses (`release/model-weight-licenses.json`)

The SPDX SBOM covers the license of every Cargo crate (the *source* axis). Model **weights** are a
separate axis cargo tooling never sees: each provider pins its own Hugging Face checkpoint, whose
license (Apache-2.0 / MIT today; possibly CC-BY-NC or research-only for a model that lands later)
SceneWorks — a **non-commercial** product — must surface on its end-product licenses page.

The source of truth is a `gen_core::WeightLicense` recorded by each provider crate beside its pinned
`HUB_REPO`/`HUB_REVISION`; `candle-audio-catalog` aggregates every *registered* provider's license
into the committed `release/model-weight-licenses.json`. Two gates keep it honest:

- `candle-audio-catalog::every_shipped_provider_has_a_weight_license` — the ship-gate in the
  composition root: a provider that reaches the catalog without a recorded, well-formed license
  fails the build, so **no audio provider can ship without its weight license recorded**;
- `candle-audio-catalog::weight_licenses_manifest_matches_committed_file` — the drift gate: the
  committed JSON must equal what the catalog produces (regenerate with
  `UPDATE_WEIGHT_LICENSES=1 cargo test -p candle-audio-catalog weight_licenses_manifest_matches_committed_file`).

`build_release.py` copies the committed manifest into the bundle (and fails if it is absent or
incomplete); `verify_release.py` re-checks the bundled copy is present and complete.

**Restriction discipline:** `commercial_use = false` marks a non-commercial (CC-BY-NC),
research-only, or otherwise restricted checkpoint. Such weights are admissible for the
non-commercial product, but every non-commercial entry MUST carry a `restriction` note describing
the terms the product has to surface — the Rust `WeightLicense::is_well_formed` check and the
Python `validate_model_weight_licenses` check both reject a non-commercial entry with no restriction
note. All seven currently-shipped audio providers are permissive (MIT / Apache-2.0).

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
