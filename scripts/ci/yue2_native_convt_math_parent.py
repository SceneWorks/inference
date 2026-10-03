#!/usr/bin/env python3
"""Fetch and authenticate the immutable stage-2 parent before any CUDA child."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import urlsplit
import zipfile


HERE = Path(__file__).parent / "yue2_bf16_tile_diagnostic"
ANCHOR = HERE / "native-convt-math-parent.json"
AUDIT = HERE / "native-convt-parent-audit.json"


class ArtifactRedirects(urllib.request.HTTPRedirectHandler):
    """Keep GitHub credentials on its HTTPS API origin only."""

    @staticmethod
    def origin(url: str) -> tuple[str, str, int]:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("artifact redirect requires an HTTPS URL without userinfo")
        return parsed.scheme.lower(), parsed.hostname.lower(), parsed.port or 443

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        original = self.origin(request.full_url)
        destination = self.origin(new_url)
        redirected = super().redirect_request(request, file_pointer, code, message, headers, new_url)
        if redirected is None:
            return None
        if destination != original:
            for collection in (redirected.headers, redirected.unredirected_hdrs):
                for key in tuple(collection):
                    if key.lower() in {"authorization", "proxy-authorization"}:
                        redirected.remove_header(key)
            if any(key.lower() in {"authorization", "proxy-authorization"}
                   for collection in (redirected.headers, redirected.unredirected_hdrs)
                   for key in collection):
                raise ValueError("artifact redirect retained a credential across origins")
        return redirected


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_parent_zip(archive: Path, destination: Path) -> dict:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    require(digest(AUDIT) == anchor["independentAuditSha256"], "parent independent audit changed")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    require(audit["diagnosticCollectionVerified"] is True and
            audit["rawCheckpointsPerDtype"] == 198 and
            audit["earliestBf16Stage"] == 2 and
            audit["waveform"]["originalWaveformHashParity"] is True and
            audit["historicalStage2"]["currentRecordsByteIdentical"] is True,
            "parent independent audit did not establish the frozen proof")
    require(digest(archive) == anchor["metricsZipSha256"], "parent metrics artifact ZIP changed")
    require(not destination.exists(), "parent extraction destination already exists")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            require(len(names) == len(set(names)), "parent ZIP has duplicate entries")
            required = {"data/report.json"} | {"data/" + row["file"]
                                                   for row in anchor["records"].values()}
            require(required <= set(names), "parent ZIP lacks a pinned stage-2 record")
            for name in sorted(required):
                require(not name.startswith("/") and ".." not in Path(name).parts and
                        not source.getinfo(name).is_dir(), "unsafe parent ZIP entry")
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(name) as stream, target.open("xb") as sink:
                    shutil.copyfileobj(stream, sink, length=8 * 1024 * 1024)
        report_path = destination / "data/report.json"
        require(digest(report_path) == anchor["reportSha256"], "parent report changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["schemaVersion"] == 5 and report["selector"] == "native_convt_columns" and
                report["engineSha"] == anchor["engineSha"] and
                report["waveformParity"] is True and
                report["earliestBf16Stage"] == 2 and
                report["nativeColumns"]["status"] == "collected" and
                report["historicalStage2"]["matchesHistorical"] is True,
                "parent report lost the verified stage-2 proof")
        for label, row in anchor["records"].items():
            path = destination / "data" / row["file"]
            require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
                    f"parent stage-2 bytes changed: {label}")
        return anchor
    except BaseException:
        shutil.rmtree(destination)
        raise


def verify_extracted_parent(destination: Path) -> dict:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    require(digest(AUDIT) == anchor["independentAuditSha256"], "parent audit changed")
    proof = json.loads((destination / "parent-proof.json").read_text(encoding="utf-8"))
    for key in ("runId", "runAttempt", "controlSha", "engineSha", "metricsArtifactId",
                "metricsZipSha256", "reportSha256", "sourceZipSha256", "independentAuditSha256"):
        require(proof.get(key) == anchor[key], f"parent extraction proof changed {key}")
    require(digest(destination / "data/report.json") == anchor["reportSha256"],
            "parent report changed after extraction")
    for label, row in anchor["records"].items():
        path = destination / "data" / row["file"]
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == row["bytes"] and digest(path) == row["sha256"],
                f"parent stage-2 raw bytes changed after extraction: {label}")
    expected = {"report.json"} | {row["file"] for row in anchor["records"].values()}
    actual = {path.name for path in (destination / "data").iterdir()}
    require(actual == expected and all(path.is_file() and not path.is_symlink()
                                       for path in (destination / "data").iterdir()),
            "parent extraction has unexpected or missing files")
    return anchor


def fetch_parent(archive: Path) -> None:
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GitHub artifact token absent")
    api = ("https://api.github.com/repos/SceneWorks/inference/actions/artifacts/"
           + str(anchor["metricsArtifactId"]))
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    opener = urllib.request.build_opener(ArtifactRedirects())
    with opener.open(urllib.request.Request(api, headers=headers), timeout=30) as response:
        metadata = json.load(response)
    require(metadata["id"] == anchor["metricsArtifactId"] and
            metadata["name"] == anchor["metricsArtifactName"] and
            metadata.get("expired") is False and
            metadata["workflow_run"]["id"] == anchor["runId"] and
            metadata["workflow_run"]["head_sha"] == anchor["controlSha"],
            "parent artifact provenance changed")
    require(not archive.exists(), "parent archive destination already exists")
    try:
        with opener.open(urllib.request.Request(api + "/zip", headers=headers), timeout=60) as source, archive.open("xb") as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
        require(digest(archive) == anchor["metricsZipSha256"], "parent artifact ZIP digest changed")
    except BaseException:
        archive.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="verify a supplied exact ZIP")
    args = parser.parse_args()
    if not args.offline:
        fetch_parent(args.archive)
    anchor = verify_parent_zip(args.archive, args.destination)
    (args.destination / "parent-proof.json").write_text(
        json.dumps({key: anchor[key] for key in ("runId", "runAttempt", "controlSha", "engineSha",
                                            "metricsArtifactId", "metricsZipSha256", "reportSha256",
                                            "sourceZipSha256", "independentAuditSha256")}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
