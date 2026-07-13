# Phase 1 History Import Checkpoint

> **Status:** Local import complete; GitHub repository creation and push blocked by
> invalid GitHub CLI authentication.

## Completed locally

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

## Remaining Phase 1 publication work

1. Reauthenticate GitHub CLI for an account allowed to create repositories in the
   SceneWorks organization.
2. Create public `SceneWorks/inference` without an auto-generated README/license so
   it accepts the assembled history.
3. Add the canonical `origin` remote.
4. Push `main`, migration/history tags, and verify the remote objects.
5. Add the remote URL to repository metadata and mark this checkpoint published.

No package normalization, dependency conversion, or provider refactor belongs in
this phase.

