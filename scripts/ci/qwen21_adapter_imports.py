"""Materialize immutable adapter transfer assets; never create or publish a release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile


CACHE_SCHEMA = "qwen21-approved-transfer-cache-v1"
CACHE_RECEIPT = "adapter-cache-resolved.json"
APPROVED_MANIFEST_BINDING_SHA256 = (
    "f11cac12baf0254594c92aefc64e839c0eb39bc76df9a71f791fcd2c87b33d14"
)
APPROVED_SOURCE_ARTIFACT = {
    "repository": "SceneWorks/inference",
    "runId": 37392084691,
    "artifactId": 11383988900,
    "name": "qwen-image-2-1-mlx-evidence",
    "digest": "sha256:d5a17f4881645ae5bbe9f94c8e43058c5276e3851f3e9897434b9506c3dd6b21",
    "sizeInBytes": 503983926,
}


def validate_manifest(manifest):
    if manifest.get("repository") != "SceneWorks/inference":
        raise ValueError("adapter assets must come from SceneWorks/inference")
    if not isinstance(manifest.get("releaseId"), int) or manifest["releaseId"] <= 0:
        raise ValueError("a positive releaseId is required")
    entries = manifest.get("adapters", [])
    names, files = set(), set()
    if not entries:
        raise ValueError("adapter manifest is empty")
    for entry in entries:
        if not re.fullmatch(r"[a-z0-9_]+", entry.get("name", "")):
            raise ValueError("invalid adapter name")
        if entry["name"] in names or entry.get("file") in files:
            raise ValueError("duplicate adapter name or file")
        names.add(entry["name"])
        files.add(entry.get("file"))
        receipt = (manifest.get("purpose") == "DIAGNOSTIC_ONLY"
                   and entry.get("name") == "training_receipt"
                   and entry.get("kind") == "diagnostic_training_receipt")
        replay = (manifest.get("purpose") == "DIAGNOSTIC_ONLY"
                  and manifest.get("acceptanceEvidence") is False
                  and entry.get("name") in ("q4_edit_base", "q4_edit_mlx_lokr")
                  and entry.get("kind") == "q4_replay_png")
        suffix = "json" if receipt else "png" if replay else "safetensors"
        if not re.fullmatch(rf"[A-Za-z0-9_-]+\.{suffix}", entry.get("file", "")):
            raise ValueError("adapter file must be one safe safetensors basename")
        if entry.get("kind") not in ("lora", "lokr") and not (receipt or replay):
            raise ValueError("adapter kind must be lora or lokr")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError("adapter SHA-256 is required")
        if not isinstance(entry.get("assetId"), int) or entry["assetId"] <= 0:
            raise ValueError("a positive assetId is required")
        if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
            raise ValueError("expected byte length is required")
        if not re.fullmatch(rf"[A-Za-z0-9_-]+\.{suffix}", entry.get("assetName", "")):
            raise ValueError("expected release asset name is required")
    return entries


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_binding(manifest):
    return {
        "repository": manifest["repository"],
        "releaseId": manifest["releaseId"],
        "adapters": manifest["adapters"],
    }


def manifest_binding_sha256(manifest):
    body = json.dumps(
        manifest_binding(manifest), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def validate_approved_cache_manifest(manifest):
    validate_manifest(manifest)
    if manifest_binding_sha256(manifest) != APPROVED_MANIFEST_BINDING_SHA256:
        raise ValueError("transfer cache manifest is not the approved immutable manifest")


def require_cache_root(cache_root, *, must_exist):
    cache_root = Path(cache_root)
    if not cache_root.is_absolute():
        raise ValueError("transfer cache root must be absolute")
    if cache_root.is_symlink():
        raise ValueError("transfer cache root must not be a symlink")
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError("transfer cache root must be a directory")
    if must_exist and not cache_root.is_dir():
        raise ValueError("transfer cache root must already exist")
    return cache_root


def cache_directory(manifest, cache_root):
    return cache_root / "qwen21-transfer-cache" / str(manifest["releaseId"])


def verified_entry(path, entry, label):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or symlinked for {entry['name']}")
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"{label} byte length mismatch for {entry['name']}")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch for {entry['name']}")
    return path


def load_verified_cache(manifest, cache_root):
    """Return an all-or-nothing approved cache, or None when no cache exists."""
    validate_approved_cache_manifest(manifest)
    cache_root = require_cache_root(cache_root, must_exist=False)
    directory = cache_directory(manifest, cache_root)
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("transfer cache directory must be a real directory")
    parent = directory.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("transfer cache parent must be a real directory")
    receipt_path = directory / CACHE_RECEIPT
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("existing transfer cache is partial: approval receipt missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing transfer cache approval receipt is invalid") from error
    if set(receipt) != {"schema", "sourceArtifact", "manifest"}:
        raise ValueError("existing transfer cache approval receipt has unexpected fields")
    if receipt["schema"] != CACHE_SCHEMA:
        raise ValueError("existing transfer cache approval schema mismatch")
    if receipt["sourceArtifact"] != APPROVED_SOURCE_ARTIFACT:
        raise ValueError("existing transfer cache source artifact mismatch")
    if receipt["manifest"] != manifest_binding(manifest):
        raise ValueError("existing transfer cache manifest identity mismatch")
    expected_names = {CACHE_RECEIPT}
    verified = {}
    for entry in manifest["adapters"]:
        expected_names.add(entry["file"])
        verified[entry["file"]] = verified_entry(
            directory / entry["file"], entry, "transfer cache"
        )
    if {path.name for path in directory.iterdir()} != expected_names:
        raise ValueError("existing transfer cache contains unexpected or partial files")
    return verified


def source_cache_status(manifest, cache_root):
    return load_verified_cache(manifest, cache_root) is not None


def validate_source_artifact_metadata(metadata):
    """Bind staging to the exact, still-live Actions artifact selected for recovery."""
    expected = APPROVED_SOURCE_ARTIFACT
    actual = {
        "repository": "SceneWorks/inference",
        "runId": metadata.get("workflow_run", {}).get("id"),
        "artifactId": metadata.get("id"),
        "name": metadata.get("name"),
        "digest": metadata.get("digest"),
        "sizeInBytes": metadata.get("size_in_bytes"),
    }
    if actual != expected:
        raise ValueError("downloaded transfer source artifact identity mismatch")
    if metadata.get("expired") is not False:
        raise ValueError("downloaded transfer source artifact is expired")


def extract_source_archive(manifest, archive, destination, source_artifact_metadata):
    """Verify the complete artifact ZIP and extract only approved transfer inputs."""
    validate_approved_cache_manifest(manifest)
    validate_source_artifact_metadata(source_artifact_metadata)
    archive = Path(archive)
    destination = Path(destination)
    if not archive.is_absolute() or archive.is_symlink() or not archive.is_file():
        raise ValueError("downloaded transfer archive must be an absolute real file")
    if archive.stat().st_size != APPROVED_SOURCE_ARTIFACT["sizeInBytes"]:
        raise ValueError("downloaded transfer archive byte length mismatch")
    expected_digest = APPROVED_SOURCE_ARTIFACT["digest"].removeprefix("sha256:")
    if sha256(archive) != expected_digest:
        raise ValueError("downloaded transfer archive SHA-256 mismatch")
    if not destination.is_absolute() or destination.is_symlink() or destination.exists():
        raise ValueError("transfer archive destination must be a new absolute path")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("transfer archive destination parent must be a real directory")

    selected = {
        "imports/adapters/adapter-imports-resolved.json",
        *(f"imports/adapters/{entry['file']}" for entry in manifest["adapters"]),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("downloaded transfer archive contains duplicate paths")
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                        or stat.S_ISLNK(mode)):
                    raise ValueError("downloaded transfer archive contains an unsafe path")
            if source.testzip() is not None:
                raise ValueError("downloaded transfer archive CRC mismatch")
            by_name = {info.filename: info for info in infos}
            if not selected.issubset(by_name):
                raise ValueError("downloaded transfer archive is missing approved inputs")
            for name in sorted(selected):
                target = temporary / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(by_name[name]) as input_stream, target.open("xb") as output:
                    shutil.copyfileobj(input_stream, output)
                    output.flush()
                    os.fsync(output.fileno())
        temporary.rename(destination)
        fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "imports/adapters"


def fsync_file(path):
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_json_synced(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage_source_cache(manifest, source, cache_root, source_artifact_metadata):
    """Atomically publish exact artifact bytes into the task-owned persistent cache."""
    validate_approved_cache_manifest(manifest)
    entries = manifest["adapters"]
    validate_source_artifact_metadata(source_artifact_metadata)
    source = Path(source)
    cache_root = require_cache_root(cache_root, must_exist=True)
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise ValueError("downloaded transfer source must be an absolute real directory")
    source_receipt = source / "adapter-imports-resolved.json"
    if source_receipt.is_symlink() or not source_receipt.is_file():
        raise ValueError("downloaded transfer source receipt is missing")
    try:
        resolved = json.loads(source_receipt.read_text(encoding="utf-8"))
        source_binding = {key: resolved[key] for key in ("repository", "releaseId", "adapters")}
    except (KeyError, OSError, json.JSONDecodeError) as error:
        raise ValueError("downloaded transfer source receipt is invalid") from error
    if source_binding != manifest_binding(manifest):
        raise ValueError("downloaded transfer source manifest identity mismatch")
    expected_source_names = {"adapter-imports-resolved.json", *(entry["file"] for entry in entries)}
    if {path.name for path in source.iterdir()} != expected_source_names:
        raise ValueError("downloaded transfer source contains unexpected or partial files")
    for entry in entries:
        verified_entry(source / entry["file"], entry, "downloaded transfer source")

    existing = load_verified_cache(manifest, cache_root)
    directory = cache_directory(manifest, cache_root)
    if existing is not None:
        return directory

    parent = directory.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError("transfer cache parent must be a real directory")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("transfer cache parent must not be a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{manifest['releaseId']}-", dir=parent))
    try:
        for entry in entries:
            target = temporary / entry["file"]
            shutil.copyfile(source / entry["file"], target)
            verified_entry(target, entry, "staged transfer cache")
            fsync_file(target)
        write_json_synced(temporary / CACHE_RECEIPT, {
            "schema": CACHE_SCHEMA,
            "sourceArtifact": APPROVED_SOURCE_ARTIFACT,
            "manifest": manifest_binding(manifest),
        })
        fsync_directory(temporary)
        try:
            temporary.rename(directory)
        except FileExistsError:
            if load_verified_cache(manifest, cache_root) is None:
                raise ValueError("transfer cache appeared without a complete approved receipt")
        fsync_directory(parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if load_verified_cache(manifest, cache_root) is None:
        raise ValueError("atomic transfer cache publication did not complete")
    return directory


def atomic_verified_copy(source, destination, entry):
    if destination.is_symlink():
        raise ValueError(f"destination must not be a symlink for {entry['name']}")
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        with source.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        verified_entry(temporary, entry, "staged destination")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(manifest, destination, opener=urllib.request.urlopen, source_cache_root=None):
    entries = validate_manifest(manifest)
    destination = Path(destination)
    cached = None
    if source_cache_root is not None:
        cached = load_verified_cache(manifest, source_cache_root)
    destination.mkdir(parents=True, exist_ok=True)
    if cached is not None:
        for entry in entries:
            path = destination / entry["file"]
            if path.is_symlink():
                raise ValueError(f"destination must not be a symlink for {entry['name']}")
            if path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]:
                continue
            atomic_verified_copy(cached[entry["file"]], path, entry)
        return write_resolved(manifest, destination)

    headers = {"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    release_request = urllib.request.Request(
        f"https://api.github.com/repos/{manifest['repository']}/releases/{manifest['releaseId']}",
        headers=headers,
    )
    with opener(release_request) as response:
        release = json.load(response)
    assets = {asset["id"]: asset for asset in release["assets"]}
    for entry in entries:
        asset = assets.get(entry["assetId"])
        if asset is None or asset["name"] != entry["assetName"] or asset["size"] != entry["size"]:
            raise ValueError(f"release does not own the expected asset {entry['name']}")
    for entry in entries:
        path = destination / entry["file"]
        if path.is_symlink():
            raise ValueError(f"destination must not be a symlink for {entry['name']}")
        if path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]:
            continue
        request = urllib.request.Request(
            f"https://api.github.com/repos/{manifest['repository']}/releases/assets/{entry['assetId']}",
            headers={
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "Accept": "application/octet-stream",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        temporary = None
        try:
            with opener(request) as response, tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.", suffix=".download", dir=destination, delete=False
            ) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(response, stream)
                stream.flush()
                os.fsync(stream.fileno())
            verified_entry(temporary, entry, "downloaded asset")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return write_resolved(manifest, destination)


def write_resolved(manifest, destination):
    resolved = {**manifest, "directory": str(destination.resolve())}
    output = destination / "adapter-imports-resolved.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=".adapter-imports-resolved.",
        suffix=".json", dir=destination, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(resolved, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--stage-source", type=Path)
    mode.add_argument("--cache-status", action="store_true")
    mode.add_argument("--verify-source-artifact", action="store_true")
    mode.add_argument("--extract-source-archive", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--source-artifact-metadata", type=Path)
    parser.add_argument("--archive-destination", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.cache_status:
        if args.source_cache_root is None:
            parser.error("--cache-status requires --source-cache-root")
        print(f"hit={'true' if source_cache_status(manifest, args.source_cache_root) else 'false'}")
    elif args.verify_source_artifact:
        validate_approved_cache_manifest(manifest)
        if args.source_artifact_metadata is None:
            parser.error("--verify-source-artifact requires --source-artifact-metadata")
        metadata = json.loads(args.source_artifact_metadata.read_text(encoding="utf-8"))
        validate_source_artifact_metadata(metadata)
    elif args.extract_source_archive is not None:
        if args.source_artifact_metadata is None or args.archive_destination is None:
            parser.error(
                "--extract-source-archive requires --source-artifact-metadata "
                "and --archive-destination"
            )
        metadata = json.loads(args.source_artifact_metadata.read_text(encoding="utf-8"))
        print(extract_source_archive(
            manifest, args.extract_source_archive, args.archive_destination, metadata
        ))
    elif args.stage_source is not None:
        if args.source_cache_root is None:
            parser.error("--stage-source requires --source-cache-root")
        if args.source_artifact_metadata is None:
            parser.error("--stage-source requires --source-artifact-metadata")
        metadata = json.loads(args.source_artifact_metadata.read_text(encoding="utf-8"))
        print(stage_source_cache(
            manifest, args.stage_source, args.source_cache_root, metadata
        ))
    else:
        print(materialize(
            manifest, args.destination, source_cache_root=args.source_cache_root
        ))


if __name__ == "__main__":
    main()
