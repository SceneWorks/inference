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
- `SHA256SUMS` covering the source archive, manifest, and SBOM.

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
