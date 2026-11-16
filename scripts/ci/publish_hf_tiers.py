#!/usr/bin/env python3
"""Publish converted model tiers to a SceneWorks Hugging Face repository and verify them (SW-7104).

Each `--map LOCAL=PREFIX` uploads every file under the local tier directory LOCAL to PREFIX/ in
`--repo`, in one commit. A file whose remote copy already has the same size and digest is not
uploaded again, so a re-run after a partial failure only sends what is missing. Files the repo
holds under PREFIX/ that the local tier does not have are deleted in the same commit: PREFIX/ ends
up holding exactly the local tier.

`prompt/` is never published. MiniMax's licensed prompt files are read from the upstream snapshot
at run time (SW-7099 decision), so a tier directory that carries one has it excluded, by name.

After the commit the script reads the repository at the new revision with file metadata and checks
every published file against the local bytes: the LFS sha256 for an LFS file, the git blob sha1 for
a regular one, and the size for both. It writes a JSON report with the revision and, per file, the
path, size and sha256.

Authentication is the runner's own Hugging Face login (`HfApi().whoami()`), as the Chroma
publication lane uses. The token is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXCLUDED_TOP_LEVEL = {"prompt"}


def file_digests(path: Path) -> tuple[int, str, str]:
    """Size, sha256 and git blob sha1 of `path`, in one read."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return size, sha256.hexdigest(), blob.hexdigest()


def local_files(root: Path, prefix: str) -> dict[str, Path]:
    if not root.is_dir():
        raise SystemExit(f"tier directory {root} does not exist")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in EXCLUDED_TOP_LEVEL:
            print(f"excluded (licensed upstream prompt file, never published): {rel.as_posix()}")
            continue
        if ".cache" in rel.parts or any(part.startswith(".") for part in rel.parts):
            continue
        files[f"{prefix}/{rel.as_posix()}"] = path
    if not files:
        raise SystemExit(f"tier directory {root} holds no files")
    return files


def lfs_sha256(sibling) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def matches(sibling, size: int, sha256: str, blob_sha1: str) -> bool:
    if sibling is None or sibling.size != size:
        return False
    remote_sha = lfs_sha256(sibling)
    if remote_sha is not None:
        return remote_sha == sha256
    return getattr(sibling, "blob_id", None) == blob_sha1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--map", action="append", required=True, metavar="LOCAL=PREFIX")
    parser.add_argument("--message", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    identity = api.whoami()
    if not identity.get("name"):
        raise SystemExit("an authenticated Hugging Face identity is required for publication")
    print(f"publishing as {identity['name']}")

    prefixes: list[str] = []
    local: dict[str, Path] = {}
    for item in args.map:
        root, sep, prefix = item.partition("=")
        prefix = prefix.strip("/")
        if not sep or not prefix:
            raise SystemExit(f"--map {item!r} must be LOCAL=PREFIX")
        prefixes.append(prefix)
        local.update(local_files(Path(root), prefix))

    digests = {}
    for remote_path, path in local.items():
        digests[remote_path] = file_digests(path)
        size, sha, _ = digests[remote_path]
        print(f"local {remote_path} {size} B sha256 {sha}")

    # Public, matching the other SceneWorks re-hosts (SceneWorks/minimax-h3-mlx is public and
    # ungated). An existing repository keeps whatever visibility it already has.
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)

    def under_prefixes(path: str) -> bool:
        return any(path.startswith(f"{prefix}/") for prefix in prefixes)

    head = api.model_info(args.repo, files_metadata=True)
    remote = {s.rfilename: s for s in head.siblings or [] if under_prefixes(s.rfilename)}
    operations = [
        CommitOperationDelete(path_in_repo=path)
        for path in sorted(remote.keys() - local.keys())
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local[path]))
        for path in sorted(local)
        if not matches(remote.get(path), *digests[path])
    )
    if operations:
        print(f"committing {len(operations)} operation(s) to {args.repo}")
        info = api.create_commit(
            repo_id=args.repo,
            repo_type="model",
            operations=operations,
            commit_message=args.message,
        )
        revision = info.oid
    else:
        print(f"{args.repo} already holds every file; nothing to upload")
        revision = head.sha

    published = api.model_info(args.repo, revision=revision, files_metadata=True)
    siblings = {s.rfilename: s for s in published.siblings or []}
    failures = []
    for path in sorted(local):
        if not matches(siblings.get(path), *digests[path]):
            failures.append(path)
    extra = sorted(p for p in siblings if under_prefixes(p) and p not in local)
    if failures or extra:
        raise SystemExit(
            f"{args.repo}@{revision} failed verification: mismatched {failures}, unexpected {extra}"
        )

    report = {
        "repository": args.repo,
        "revision": revision,
        "private": bool(getattr(published, "private", False)),
        "publishedBy": identity["name"],
        "prefixBytes": {
            prefix: sum(digests[p][0] for p in local if p.startswith(f"{prefix}/"))
            for prefix in prefixes
        },
        "files": [
            {
                "path": path,
                "size": digests[path][0],
                "sha256": digests[path][1],
                "remoteCheck": "lfs-sha256" if lfs_sha256(siblings[path]) else "git-blob-sha1",
            }
            for path in sorted(local)
        ],
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
