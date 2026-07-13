# Phase 1 History Import Checkpoint

> **Status:** Complete. The assembled history and its migration/history refs are
> published to the canonical private `SceneWorks/inference` repository.

## Completed

- Bootstrapped the provisional inference repository with Apache-2.0 licensing,
  Rust 1.96.0, migration guidance, and an empty/excluded root workspace.
- Installed and recorded `git-filter-repo` 2.47.0.
- Filtered the five source histories beneath temporary `imports/` namespaces.
- Merged the five baseline histories without changing imported files.
- Created annotated `migration-baseline/<repository>` tags.
- Preserved the four-commit-newer Candle tracking history as
  `history/candle-gen-tracking-main` without merging it into the compatibility
  baseline tree.
- Committed all `git-filter-repo` commit and ref maps under `docs/migration/`.
- Recorded source commits, filtered commits, and tree IDs in
  `phase-1-sources.toml`.
- Published `main`, every `migration-baseline/*` tag, and
  `history/candle-gen-tracking-main` to the canonical remote without rewriting
  any imported history.

## Equivalence result

Every recorded filtered baseline commit has a subtree exactly equal to the source
repository root tree:

| Repository | Source tree | Result |
|---|---|---|
| `core-llm` | `e115397e83414217c3384698318522f6f4e8593a` | Match |
| `mlx-llm` | `4e19df11b186c864879ed871b55e55eeb147335b` | Match |
| `candle-llm` | `40ec531d2d6735930227a1e84175e8c894726add` | Match |
| `mlx-gen` | `a8cf8de2713df6ff9a426ee40018fedfb64b3809` | Match |
| `candle-gen` | `b610bdf1565ceb810e7b938122473cca8b785b83` | Match |
| Newer Candle tracking commit | `26790038397b8c45c55a56dd612e52a97847c965` | Match and preserved separately |

## Publication outcome

The Phase 0 recommendation proposed public visibility because the source
repositories were public. The repository was instead created private, and the
migration deliberately preserves that actual visibility. Changing it is an
independent administrative decision and is not authorized by history import,
runtime release, or consumer cutover.

No package normalization, dependency conversion, or provider refactor belongs in
this phase.
