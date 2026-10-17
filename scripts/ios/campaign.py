#!/usr/bin/env python3
"""Catalog-driven, cold-process iOS evidence collection. Listing never launches a device.

`run` exercises ADMISSION against the reviewed evidence table. `capture` is the explicitly
labelled bounded DEVICE CAPTURE (no evidence needed; every quantum checks live headroom before
constructing its graph): it walks each selected model/route/tier/mode/rung up the edge ladder and
stops at the first case that is not a completed capture. A `capture_refused` row is the typed
"next size refused before jetsam" evidence. `evidence` turns completed capture rows into candidate
rows for crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json; it only
prints them. Admission changes only through a human-reviewed commit of that table."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("q4", "q8", "bf16")
EDGES = (256, 512, 768, 1024, 1280)
STATUSES = {"successful", "admitted_refused", "diagnostic_success", "diagnostic_failure", "capture_refused", "skipped"}
REFUSAL_KINDS = ("capacity", "no_evidence", "mode", "strategy", "artifact")
EVIDENCE_TABLE = "crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json"


def case_key(value):
    p = value.split("/")
    if len(p) != 6 or any(not s or not all(c.isascii() and (c.isalnum() or c in "_-") for c in s) for s in p):
        raise argparse.ArgumentTypeError("case must be provider/route/tier/edge/mode/rung")
    if p[2] not in TIERS or p[3] not in tuple(map(str, EDGES)) or p[5] not in tuple(map(str, range(5))):
        raise argparse.ArgumentTypeError("case outside finite tier/geometry/rung domain")
    return value


def positive(value):
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive integer") from exc
    if not 1 <= n <= 20000:
        raise argparse.ArgumentTypeError("budget must be 1..20000")
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "run", "table", "capture", "evidence"))
    p.add_argument("--catalog", type=Path)
    for name in ("provider", "route", "mode"):
        p.add_argument("--" + name)
    p.add_argument("--tier", choices=TIERS)
    p.add_argument("--edge", type=int, choices=EDGES)
    p.add_argument("--rung", type=int, choices=range(5))
    p.add_argument("--case", type=case_key)
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--next-edge", type=int, choices=EDGES, help="additional capacity-refusal-only probe")
    p.add_argument("--device")
    p.add_argument("--memory-entitlement", choices=("on", "off"), default="on")
    p.add_argument("--output", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--max-cases", type=positive, default=20000, help="counts renders AND probes")
    p.add_argument("--timeout-seconds", type=positive, default=1800, help="per-case cooperative device deadline and host polling budget")
    return p


def command(argv, timeout, env=None):
    """Own the entire host subprocess group, including cargo/xcodebuild/devicectl children."""
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Even if the group leader exited, an orphaned child may still own the group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        raise


def catalog(args):
    if args.catalog:
        data = json.loads(args.catalog.read_text(encoding="utf-8"))
    else:
        env = dict(os.environ, CARGO_BUILD_JOBS="4")
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
        result = command(["cargo", "run", "--locked", "--offline", "--manifest-path", "ios-host/smoke/Cargo.toml", "--features", "media", "--example", "campaign_catalog", "-j", "4"], args.timeout_seconds, env)
        if result.returncode:
            raise ValueError("catalog export failed: " + result.stderr)
        data = json.loads(result.stdout)
    if data.get("schema") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("empty/malformed catalog is not coverage")
    keys = [case_key(k) for k in data["cases"]]
    if len(keys) != len(set(keys)) or len(keys) > 20000:
        raise ValueError("duplicate cases or catalog exceeds finite campaign budget")
    return data


def selected(args, key):
    if args.case:
        return key == args.case
    fields = key.split("/")
    values = (args.provider, args.route, args.tier, args.edge, args.mode, args.rung)
    return all(v is None or str(v) == f for v, f in zip(values, fields))


def failure(key, reason):
    return {"schema": 1, "case": key, "status": "diagnostic_failure", "production_success": False,
            "reason": reason, "refusal_kind": None, "elapsed_seconds": None,
            "mlx_active_peak_bytes": None, "mlx_cache_peak_bytes": None, "mlx_peak_bytes": None,
            "mlx_active_cache_peak_bytes": None, "phys_footprint_peak_bytes": None,
            "kernel_lifetime_phys_footprint_peak_bytes": None, "min_headroom_mib": None,
            "launcher_exit_code": None, "refusal_only": False}


def parse_report(text, key):
    try:
        rows = [json.loads(line.removeprefix("CAMPAIGN_JSON ")) for line in text.splitlines() if line.startswith("CAMPAIGN_JSON ")]
    except (ValueError, TypeError):
        return failure(key, "malformed JSON case report")
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return failure(key, "missing/duplicate/malformed case report")
    row = rows[0]
    status = row.get("status")
    if row.get("schema") != 1 or row.get("case") != key or not isinstance(status, str) or status not in STATUSES:
        return failure(key, "stale or malformed case identity/status")
    success = status == "successful"
    if row.get("production_success") is not success or (row.get("diagnostic") and success) or (status in ("diagnostic_success", "capture_refused") and row.get("diagnostic") is not True):
        return failure(key, "inconsistent status/admission accounting")
    if status == "admitted_refused" and row.get("refusal_kind") not in REFUSAL_KINDS:
        return failure(key, "missing typed refusal kind")
    if status == "capture_refused" and (row.get("refusal_kind") != "capacity" or not isinstance(row.get("refused_quantum"), dict) or row.get("evidence_class") != "capture"):
        return failure(key, "capture refusal without its typed quantum record")
    if success:
        for name in ("elapsed_seconds", "mlx_active_peak_bytes", "mlx_cache_peak_bytes", "mlx_active_cache_peak_bytes", "mlx_peak_bytes", "phys_footprint_peak_bytes", "kernel_lifetime_phys_footprint_peak_bytes", "min_headroom_mib"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return failure(key, "production result missing finite device telemetry: " + name)
        if row["phys_footprint_peak_bytes"] == 0:
            return failure(key, "zero physical footprint is not a device measurement")
    return row


def capacity_refusal(row):
    artifact = row.get("artifact_identity")
    if not isinstance(artifact, dict):
        return False
    profile = artifact.get("layout_sha256", artifact.get("profile_sha256"))
    revision = row.get("evidence_revision")
    return (row.get("status") == "admitted_refused" and row.get("refusal_kind") == "capacity"
            and isinstance(profile, str) and len(profile) == 64 and all(c in "0123456789abcdef" for c in profile)
            and isinstance(revision, str) and bool(revision) and revision != "no-device-capture"
            and isinstance(row.get("selected_parameters"), dict))


def clean_environment():
    env = dict(os.environ)
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    env["CARGO_BUILD_JOBS"] = "4"
    for name in ("IOS_SMOKE_UNADMITTED", "IOS_SMOKE_SANA_CAPTURE", "IOS_CAMPAIGN_CASE", "IOS_CAMPAIGN_LIST", "IOS_CAMPAIGN_DIAGNOSTIC", "IOS_CAMPAIGN_REFUSAL_ONLY"):
        env.pop(name, None)
    return env


class DeviceSession:
    """Install once, then launch without terminate-existing; never hard-kill a device render."""
    def __init__(self, args):
        self.args = args
        self.bundle = os.environ.get("SMOKE_BUNDLE_ID", "com.idkplay.SceneWorksSmoke")
        self.blocked = False
        self.sha = command(["git", "rev-parse", "HEAD"], 10).stdout.strip()
        self.env = clean_environment()

    def device(self, tail, timeout=15):
        return command(["xcrun", "devicectl", *tail], timeout, self.env)

    def exited(self):
        with tempfile.TemporaryDirectory(prefix="ios-campaign-processes-") as tmp:
            path = Path(tmp) / "processes.json"
            result = self.device(["device", "info", "processes", "--device", self.args.device, "--json-output", str(path)])
            if result.returncode:
                raise ValueError("cannot verify device process exit: " + result.stderr)
            processes = json.loads(path.read_text(encoding="utf-8")).get("result", {}).get("runningProcesses")
            if not isinstance(processes, list):
                raise ValueError("unknown device process-list schema; refusing to advance")
            for process in processes:
                if not isinstance(process, dict):
                    raise ValueError("malformed device process record")
                if not any(isinstance(process.get(k), str) and process[k] for k in ("executable", "name", "bundleIdentifier")):
                    raise ValueError("unidentifiable device process; cannot prove app exited")
                identity = str(process.get("executable", "")) + str(process.get("name", "")) + str(process.get("bundleIdentifier", ""))
                if "SceneWorksSmoke" in identity or self.bundle in identity:
                    return False
            return True

    def install(self):
        if not self.exited():
            raise ValueError("smoke app already running; no reinstall/termination is allowed")
        result = command([str(ROOT / "scripts/ios/run_smoke.sh"), "--media", "--install-only", "--device", self.args.device, "--memory-entitlement", self.args.memory_entitlement], self.args.timeout_seconds, self.env)
        if result.returncode:
            raise ValueError(f"campaign install failed (exit {result.returncode}): {result.stderr}")

    def copy(self, remote, local):
        return self.device(["device", "copy", "from", "--device", self.args.device, "--domain-type", "appDataContainer", "--domain-identifier", self.bundle, "--source", "Documents/" + remote, "--destination", str(local)])

    def run_case(self, key, refusal_only=False):
        row = failure(key, "case did not produce a report")
        started = time.time()
        end = time.monotonic() + self.args.timeout_seconds
        artifact = self.args.output.parent / (self.args.output.stem + ".artifacts") / (key.replace("/", "-") + ("-probe" if refusal_only else ""))
        launcher_code = None
        run_id = uuid.uuid4().hex
        try:
            artifact.mkdir(parents=True, exist_ok=False)
            if not self.exited():
                self.blocked = True
                raise ValueError("prior app still running; cannot launch next case")
            variables = {"IOS_CAMPAIGN_CASE": key, "IOS_CAMPAIGN_DIAGNOSTIC": "1" if self.args.diagnostic else "0",
                         "IOS_CAMPAIGN_REFUSAL_ONLY": "1" if refusal_only else "0", "IOS_CAMPAIGN_TIMEOUT_SECONDS": str(self.args.timeout_seconds), "IOS_CAMPAIGN_BUILD_SHA": self.sha, "IOS_CAMPAIGN_RUN_ID": run_id}
            result = self.device(["device", "process", "launch", "--device", self.args.device, "--environment-variables", json.dumps(variables), self.bundle])
            launcher_code = result.returncode
            if launcher_code:
                raise ValueError("device launcher failed: " + result.stderr)
            report = artifact / "report.txt"
            while time.monotonic() < end:
                copied = self.copy("smoke-report.txt", report)
                if not copied.returncode and report.exists() and report.stat().st_mtime >= int(started):
                    row = parse_report(report.read_text(encoding="utf-8"), key)
                    break
                time.sleep(min(1, max(0, end - time.monotonic())))
            else:
                row = failure(key, "cooperative device deadline exceeded; no report, not a refusal proof")
                row["harness_terminated"] = True
            # The app writes its report only after generate has returned, and then exits itself.
            # If cancellation cannot finish, do NOT signal Metal or launch/reinstall another case.
            grace_end = time.monotonic() + 15
            while not self.exited():
                if time.monotonic() >= grace_end:
                    self.blocked = True
                    raise ValueError("device did not exit after cooperative deadline; campaign stopped without hard-killing render")
                time.sleep(1)
            launch = artifact / "launch-memory.txt"
            copied = self.copy("launch-memory.txt", launch)
            expected = "YES" if self.args.memory_entitlement == "on" else "NO"
            if copied.returncode or not launch.exists() or launch.stat().st_mtime < int(started) or not launch.read_text(encoding="utf-8").rstrip().endswith("increased-memory-limit = " + expected):
                raise ValueError("fresh launch-memory.txt entitlement verification is mandatory")
            if row.get("run_id") != run_id:
                raise ValueError("stale report from a different case launch")
            if row.get("build_sha") != self.sha:
                raise ValueError("installed build SHA does not match campaign checkout")
            if bool(row.get("diagnostic", False)) != self.args.diagnostic or bool(row.get("refusal_only", False)) != refusal_only:
                raise ValueError("device ignored selected diagnostic/refusal-only flags")
            manifest = artifact / "renders.txt"
            copied = self.copy("renders.txt", manifest)
            if not copied.returncode and manifest.exists():
                names = manifest.read_text(encoding="utf-8").splitlines()
                if row["status"] == "successful" and (not names or len(names) != row.get("expected_render_count") or len(set(names)) != len(names)):
                    raise ValueError("successful case render manifest count is missing/wrong")
                for name in names:
                    if not name or Path(name).name != name or not name.startswith("campaign-" + key.replace("/", "-") + "-") or not name.endswith(".png"):
                        raise ValueError("invalid render manifest path")
                    if self.copy(name, artifact / name).returncode:
                        raise ValueError("failed to pull case render " + name)
            elif row["status"] == "successful":
                raise ValueError("successful case has no render manifest")
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            row = failure(key, str(exc))
            # Exit verification is required on every exceptional path too.
            try:
                self.blocked = not self.exited()
            except (ValueError, OSError, subprocess.SubprocessError):
                self.blocked = True
            row["harness_terminated"] = True
        row.update(launcher_exit_code=launcher_code, refusal_only=refusal_only, device_exited=not self.blocked, host_artifact_directory=str(artifact))
        if refusal_only and not capacity_refusal(row):
            row = dict(row, status="diagnostic_failure", production_success=False, reason="next-up requires an artifact-bound CAPACITY refusal, not missing evidence/mode/strategy: " + row.get("reason", ""))
        return row


def invoke(session, key, refusal_only=False):
    return session.run_case(key, refusal_only=refusal_only)


def render(rows):
    lines = ["| Case | Probe | Status / refusal | Time s | Active MiB | Cache MiB | Combined MiB | MLX peak MiB | Sampled phys MiB | Kernel peak MiB | Min headroom MiB |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def metric(name, scale=1):
            v = r.get(name)
            return "unavailable" if v is None else f"{v / scale:.2f}"
        cols = [metric("elapsed_seconds")] + [metric(n, 1048576) for n in ("mlx_active_peak_bytes", "mlx_cache_peak_bytes", "mlx_active_cache_peak_bytes", "mlx_peak_bytes", "phys_footprint_peak_bytes", "kernel_lifetime_phys_footprint_peak_bytes")] + [metric("min_headroom_mib")]
        lines.append("| " + " | ".join([r["case"], "yes" if r.get("refusal_only") else "no", r["status"] + "/" + str(r.get("refusal_kind")), *cols]) + " |")
        quantum = r.get("refused_quantum")
        if isinstance(quantum, dict):
            lines.append(f"|   refused before construction: {quantum.get('label')} needs {quantum.get('scratch_bytes')} + reserve {quantum.get('reserve_bytes')} B, live {quantum.get('available_bytes')} B | | | | | | | | | | |")
    groups = {}
    for r in rows:
        fields = r["case"].split("/")
        group = "/".join(fields[:3] + fields[4:])
        if r["status"] == "successful" and r.get("production_success") is True and not r.get("refusal_only"):
            groups[group] = max(groups.get(group, 0), int(fields[3]))
    lines.append("\nLargest admitted completed edges (diagnostics excluded): " + json.dumps(groups, sort_keys=True))
    lines.append("Capacity-refused next-up probes: " + json.dumps([r["case"] for r in rows if r.get("refusal_only") and capacity_refusal(r)]))
    captured = {}
    for r in rows:
        fields = r["case"].split("/")
        group = "/".join(fields[:3] + fields[4:])
        if r["status"] == "diagnostic_success" and r.get("evidence_class") == "capture":
            captured.setdefault(group, {})["largest_completed_edge"] = max(captured.get(group, {}).get("largest_completed_edge", 0), int(fields[3]))
        elif r["status"] == "capture_refused":
            captured.setdefault(group, {})["refused_before_jetsam_edge"] = int(fields[3])
    lines.append("CAPTURE (diagnostic, not admission) ladders: " + json.dumps(captured, sort_keys=True))
    return "\n".join(lines)


def ladders(keys):
    """Selected keys grouped by everything but the edge, each group ascending by edge."""
    groups = {}
    for key in keys:
        fields = key.split("/")
        groups.setdefault(tuple(fields[:3] + fields[4:]), []).append(key)
    return [sorted(group, key=lambda k: int(k.split("/")[3])) for _, group in sorted(groups.items())]


def capture(args, keys, output):
    """Walk each ladder upward; stop a ladder at its first case that is not a completed capture.
    A quantum refusal there is the next-size-refused evidence; a failure stops it too. No retries,
    no cap change."""
    rows = []
    session = DeviceSession(args)
    session.install()
    executed = 0
    for ladder in ladders(keys):
        for key in ladder:
            if executed >= args.max_cases or session.blocked:
                row = dict(failure(key, "execution budget reached or prior device exit unverified"), status="skipped")
            else:
                executed += 1
                row = invoke(session, key)
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
            if row["status"] != "diagnostic_success":
                break
    return rows


def evidence_rows(rows):
    """Candidate evidence-table rows from COMPLETED captures with full device telemetry. Printed for
    review; never written to the table."""
    out = []
    for r in rows:
        key = r.get("evidence_key")
        peak = r.get("kernel_lifetime_phys_footprint_peak_bytes")
        if r.get("status") != "diagnostic_success" or r.get("evidence_class") != "capture" or not isinstance(key, dict):
            continue
        values = [peak, r.get("mlx_peak_bytes"), r.get("min_headroom_mib"), r.get("elapsed_seconds")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or not peak:
            continue
        out.append({"key": key, "footprint_bytes": int(peak), "mlx_peak_bytes": int(r["mlx_peak_bytes"]),
                    "min_headroom_mib": float(r["min_headroom_mib"]), "elapsed_seconds": float(r["elapsed_seconds"]),
                    "evidence_class": "device_capture", "run_id": str(r.get("run_id") or ""), "build_sha": str(r.get("build_sha") or "")})
    return out


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.action == "evidence":
        if not args.input:
            p.error("evidence requires --input (capture output)")
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
        print(f"# Candidate rows for {EVIDENCE_TABLE}. Review, then commit by hand; capture never changes admission.")
        print(json.dumps(evidence_rows(rows), indent=2))
        return 0
    if args.action == "table":
        if not args.input:
            p.error("table requires --input")
        print(render([json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]))
        return 0
    data = catalog(args)
    keys = [k for k in data["cases"] if selected(args, k)]
    if not keys:
        p.error("selection matches no actual catalog cases")
    if args.action == "list":
        print(json.dumps(dict(data, cases=keys), indent=2))
        return 0
    if not args.device or not args.output:
        p.error("run requires explicit --device and new --output")
    if not args.all_cases and not args.case and not any((args.provider, args.route, args.tier, args.edge, args.mode, args.rung is not None)):
        p.error("run requires selectors or --all-cases")
    if args.action == "capture":
        if args.next_edge is not None or args.edge is not None or args.case:
            p.error("capture walks the edge ladder itself; no --edge/--case/--next-edge")
        args.diagnostic = True
        with args.output.open("x", encoding="utf-8") as output:
            rows = capture(args, keys, output)
        print(render(rows))
        return 0 if rows and all(r["status"] in ("diagnostic_success", "capture_refused") for r in rows) else 2
    if args.next_edge is not None and args.diagnostic:
        p.error("next-up refusal proof must use admitted mode, not diagnostics")
    if args.next_edge is not None and any(args.next_edge <= int(key.split("/")[3]) for key in keys):
        p.error("next-edge must exceed every selected completed edge")
    rows = []
    with args.output.open("x", encoding="utf-8") as output:
        session = DeviceSession(args)
        session.install()
        executed = 0
        def collect(key, probe=False):
            nonlocal executed
            if executed >= args.max_cases or session.blocked:
                return dict(failure(key, "execution budget reached or prior device exit unverified"), status="skipped", refusal_only=probe)
            executed += 1
            return invoke(session, key, refusal_only=probe)
        def record(row):
            output.write(json.dumps(row) + "\n")
            output.flush()
            rows.append(row)
        for key in keys:
            row = collect(key)
            record(row)
            if args.next_edge is not None and row["status"] == "successful":
                fields = key.split("/")
                fields[3] = str(args.next_edge)
                record(collect("/".join(fields), True))
    print(render(rows))
    return 0 if all(r["status"] == "successful" and r.get("production_success") for r in rows) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"campaign: {error}", file=sys.stderr)
        sys.exit(2)
