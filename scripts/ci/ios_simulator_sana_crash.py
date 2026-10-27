#!/usr/bin/env python3
"""Recover one exact simulator app crash from the claiming Mac; never run the app."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat


MAX_IPS_BYTES = 4 * 1024 * 1024
MAX_CANDIDATES = 128
UDID = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\Z")


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("crash recovery window requires an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def crash_time(value: str) -> datetime:
    for pattern in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(value, pattern).astimezone(timezone.utc)
        except ValueError:
            pass
    raise ValueError("unrecognized Apple crash timestamp")


def matching_ips(data: bytes, bundle: str, udid: str, start: datetime,
                 end: datetime) -> dict | None:
    header_bytes, separator, body_bytes = data.partition(b"\n")
    if not separator:
        return None
    header, body = json.loads(header_bytes), json.loads(body_bytes)
    if not isinstance(header, dict) or not isinstance(body, dict):
        return None
    if header.get("app_name") != "SceneWorksSmoke" or body.get("procName") != "SceneWorksSmoke":
        return None
    bundle_info = body.get("bundleInfo")
    identities = (header.get("bundleID"),
                  bundle_info.get("CFBundleIdentifier") if isinstance(bundle_info, dict) else None,
                  body.get("codeSigningID") if body.get("codeSigningID") == bundle else None)
    stated = [identity for identity in identities if identity]
    if not stated or any(identity != bundle for identity in stated):
        return None
    path = body.get("procPath", "")
    if not isinstance(path, str) or not path.endswith("/SceneWorksSmoke.app/SceneWorksSmoke"):
        return None
    # Apple may redact the device path. A visible different device is a mismatch; a
    # redacted path can identify the run, but cannot attest the simulator UUID.
    if "/CoreSimulator/Devices/" in path:
        if f"/{udid}/".casefold() not in path.casefold():
            return None
        device_binding = "path-confirmed"
    else:
        device_binding = "run-bundle-time-only; device-not-attested-by-crash"
    captured = crash_time(body.get("captureTime", ""))
    stamped = crash_time(header.get("timestamp", ""))
    if not start <= captured <= end or not start <= stamped <= end:
        return None
    pid = body.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    return {"pid": pid, "capture_time_utc": captured.isoformat(),
            "header_time_utc": stamped.isoformat(), "proc_path": path,
            "device_binding": device_binding,
            "termination": body.get("termination"), "exception": body.get("exception"),
            "fatal_dyld_error": body.get("fatalDyldError")}


def recover(root: Path, output: Path, run_id: str, attempt: str, udid: str,
            start: datetime, end: datetime) -> bool:
    if (not re.fullmatch(r"[1-9][0-9]*", run_id)
            or not re.fullmatch(r"[1-9][0-9]*", attempt) or not UDID.fullmatch(udid)
            or end <= start or (end - start).total_seconds() > 300):
        raise ValueError("exact run, attempt, owned simulator UUID, and <=5-minute window required")
    if output.exists() or not output.is_absolute():
        raise ValueError("fresh absolute recovery output required")
    output.mkdir(parents=True)
    bundle = f"com.michaeltrefry.SceneWorksSmokeSim{run_id}a{attempt}"
    candidates = sorted(root.glob("SceneWorksSmoke*.ips"))
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("too many simulator crash candidates to inspect safely")
    matches = []
    for path in candidates:
        try:
            meta = path.lstat()
            if not stat.S_ISREG(meta.st_mode) or meta.st_size > MAX_IPS_BYTES:
                continue
            with path.open("rb") as stream:
                data = stream.read(MAX_IPS_BYTES + 1)
            if len(data) > MAX_IPS_BYTES:
                continue
            details = matching_ips(data, bundle, udid, start, end)
            if details is not None:
                matches.append((data, details))
        except (OSError, ValueError, UnicodeDecodeError, TypeError):
            continue
    receipt = {"accepted": False, "claim": "diagnostic-only app crash; no image or phone-fit evidence",
               "run_id": run_id, "attempt": attempt, "bundle_id": bundle,
               "expected_owned_udid": udid,
               "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
               "matching_crash_count": len(matches)}
    if len(matches) == 1:
        data, details = matches[0]
        (output / "matching-crash.ips").write_bytes(data)
        receipt.update(status="app-crash-diagnostic-retained", crash=details,
                       crash_bytes=len(data), crash_sha256=hashlib.sha256(data).hexdigest())
    else:
        receipt["status"] = "no-match" if not matches else "ambiguous-matches"
    (output / "recovery.json").write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return len(matches) == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--owned-udid", required=True)
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("RUNNER_OS") != "macOS"
            or os.environ.get("RUNNER_ARCH") != "ARM64"
            or os.environ.get("RUNNER_NAME") != "nax-macos-2"):
        raise ValueError("crash recovery requires an explicit dispatch on nax-macos-2")
    found = recover(Path.home() / "Library/Logs/DiagnosticReports", args.output,
                    args.run_id, args.attempt, args.owned_udid,
                    utc(args.start_utc), utc(args.end_utc))
    print("exact simulator crash retained" if found else "no unique exact simulator crash found")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
