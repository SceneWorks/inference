python3.12 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

api = HfApi()
identity = api.whoami()
if not identity.get("name"):
    raise SystemExit("authenticated Hugging Face identity is required for publication")

repository = os.environ["CHROMA_REPOSITORY"]
expected = os.environ["CHROMA_SOURCE_REVISION"]
expected_model = api.model_info(repository, revision=expected, files_metadata=True)
current = api.model_info(repository).sha
current_model = api.model_info(repository, revision=current, files_metadata=True)
packed_root = Path(os.environ["CHROMA_PACKED"])
additions = {}
for tier in ("q4", "q8"):
    tier_root = packed_root / tier
    for path in tier_root.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            additions[f"{tier}/{path.relative_to(tier_root).as_posix()}"] = path

def lfs_sha(sibling):
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None) or (
        lfs.get("sha256") if isinstance(lfs, dict) else None
    )

def remote_signature(model, prefix):
    return {
        sibling.rfilename: (
            sibling.size,
            lfs_sha(sibling),
            getattr(sibling, "blob_id", None),
        )
        for sibling in model.siblings
        if sibling.rfilename.startswith(prefix)
    }

if remote_signature(current_model, "bf16/") != remote_signature(expected_model, "bf16/"):
    raise SystemExit(f"{repository} bf16 tree moved from immutable source; refusing")

def published_payload_matches(model):
    siblings = {
        sibling.rfilename: sibling
        for sibling in model.siblings
        if sibling.rfilename.startswith(("q4/", "q8/"))
    }
    if siblings.keys() != additions.keys():
        return False
    for remote_path, local_path in additions.items():
        sibling = siblings[remote_path]
        if sibling.size != local_path.stat().st_size:
            return False
        remote_sha = lfs_sha(sibling)
        if remote_sha:
            digest = hashlib.sha256()
            with local_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != remote_sha:
                return False
        else:
            downloaded = Path(hf_hub_download(
                repo_id=repository,
                filename=remote_path,
                revision=model.sha,
            ))
            if downloaded.read_bytes() != local_path.read_bytes():
                return False
    return True

idempotent = current != expected
if idempotent:
    if not published_payload_matches(current_model):
        raise SystemExit(
            f"{repository} main moved to {current} but is not the verified SC-16462 payload"
        )
    commit_url = f"https://huggingface.co/{repository}/commit/{current}"
else:
    existing_paths = {
        sibling.rfilename
        for sibling in current_model.siblings
        if sibling.rfilename.startswith(("q4/", "q8/"))
    }
    operations = [
        CommitOperationDelete(path_in_repo=path)
        for path in sorted(existing_paths - additions.keys())
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=local_path)
        for path, local_path in sorted(additions.items())
    )
    info = api.create_commit(
        repo_id=repository,
        operations=operations,
        commit_message="sc-16462: publish complete packed q4/q8 tiers",
        parent_commit=current,
    )
    current = info.oid
    commit_url = str(info.commit_url)
    current_model = api.model_info(repository, revision=current, files_metadata=True)
    if not published_payload_matches(current_model):
        raise SystemExit(f"{repository}@{current} failed post-publish verification")

sizes = {
    tier: sum(
        sibling.size or 0
        for sibling in current_model.siblings
        if sibling.rfilename.startswith(f"{tier}/")
    )
    for tier in ("q4", "q8", "bf16")
}
report = {
    "repository": repository,
    "sourceRevision": expected,
    "resolvedRevision": current,
    "sizes": sizes,
    "commit": {"oid": current, "url": commit_url},
    "publishedBy": identity["name"],
    "idempotentVerification": idempotent,
}
Path(os.environ["CHROMA_REPORT"]).write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
