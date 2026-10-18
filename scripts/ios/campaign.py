#!/usr/bin/env python3
"""Catalog-driven, cold-process iOS ANCHOR campaign. Listing never launches a device.

The campaign measures anchors only, per the recorded memory-anchor rule (two measured extremes per
family, every other rung / selection / geometry DERIVED by the anchor law in
crates/media/mlx-gen/mlx-gen-ios-catalog/src/anchor.rs). The listing (the runner's own catalog, never
a hand list) names, per provider route x tier x mode, the fully RESIDENT anchor (rung 0) and the
fully OPTIMIZED anchor (deepest implemented rung, cheapest selection, captured 2 times: its spread is
the low-end margin), both at the reference geometry with 2 denoise steps; plus one model's
full-step case (validation_steps) and its second-geometry case (validation_geometry).

`capture` runs the resident, optimized and validation_steps cases as labelled bounded DEVICE
CAPTUREs (no evidence needed; every quantum checks live headroom before constructing its graph). A
resident `capture_refused` row is that anchor's evidence; an optimized anchor that does not complete
is a DEFECT (exit 2). `evidence` turns capture rows into candidate schema-3 anchor rows for
crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json (it only prints them; a
human reviews and commits). `validate` then runs the validation_geometry case ADMITTED against the
committed anchors and prints the anchor law's prediction against the measurement. `run` exercises
admission for any case of the catalog's domain."""
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
# Reserved synthetic primitive key: never a catalog case, never admission evidence.
PRIMITIVE_KEY = "primitive/sana_block_quantum"
PRIMITIVE_CLASS = "primitive_characterization"
STEPS_MAX = 100
PURPOSES = ("resident", "optimized", "validation_steps", "validation_geometry", "validation_mode")
ADMITTED_VALIDATIONS = ("validation_geometry", "validation_mode")
CAPTURE_PURPOSES = ("resident", "optimized", "validation_steps")
OPTIMIZED_REPEATS = 2
# Mirrors evidence::ANCHOR_MARGIN_FLOOR_BYTES: the low-end margin's floor.
ANCHOR_MARGIN_FLOOR_BYTES = 32 << 20
STATUSES = {"successful", "admitted_refused", "diagnostic_success", "diagnostic_failure", "capture_refused", "skipped"}
REFUSAL_KINDS = ("capacity", "no_evidence", "mode", "strategy", "artifact")
EVIDENCE_TABLE = "crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json"


def case_key(value):
    p = value.split("/")
    if len(p) != 8 or any(not s or not all(c.isascii() and (c.isalnum() or c in "_-") for c in s) for s in p):
        raise argparse.ArgumentTypeError("case must be provider/route/tier/edge/mode/rung/selection/steps")
    if p[2] not in TIERS or p[3] not in tuple(map(str, EDGES)) or p[5] not in tuple(map(str, range(5))) \
            or not p[7].isdigit() or not 0 <= int(p[7]) <= STEPS_MAX:
        raise argparse.ArgumentTypeError("case outside finite tier/geometry/rung/steps domain")
    return value


def positive(value):
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive integer") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("expected positive integer")
    return n


def seconds(value):
    """The device's cooperative deadline domain (1..20000 s, mirrored by the app)."""
    n = positive(value)
    if n > 20000:
        raise argparse.ArgumentTypeError("deadline must be 1..20000 seconds")
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "run", "table", "capture", "evidence", "validate", "primitive"))
    p.add_argument("--catalog", type=Path)
    for name in ("provider", "route", "mode", "selection"):
        p.add_argument("--" + name)
    p.add_argument("--tier", choices=TIERS)
    p.add_argument("--edge", type=int, choices=EDGES)
    p.add_argument("--rung", type=int, choices=range(5))
    p.add_argument("--steps", type=int, choices=range(STEPS_MAX + 1))
    p.add_argument("--purpose", choices=PURPOSES)
    p.add_argument("--case", type=case_key)
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--next-edge", type=int, choices=EDGES, help="additional capacity-refusal-only probe")
    p.add_argument("--device")
    p.add_argument("--memory-entitlement", choices=("on", "off"), default="on")
    p.add_argument("--output", type=Path)
    p.add_argument("--snapshots", type=Path, help="host dir of <route>/<tier> slots (the ones pushed): group anchor families by architecture digest")
    p.add_argument("--input", type=Path)
    p.add_argument("--max-cases", type=positive, help="optional cap on launches (renders, repeats AND probes); the anchor listing is finite by construction")
    p.add_argument("--timeout-seconds", type=seconds, default=1800, help="per-case cooperative device deadline and host polling budget")
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
        if getattr(args, "snapshots", None):
            env["IOS_CAMPAIGN_SNAPSHOTS"] = str(args.snapshots.resolve())
        env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
        result = command(["cargo", "run", "--locked", "--offline", "--manifest-path", "ios-host/smoke/Cargo.toml", "--features", "media", "--example", "campaign_catalog", "-j", "4"], args.timeout_seconds, env)
        if result.returncode:
            raise ValueError("catalog export failed: " + result.stderr)
        data = json.loads(result.stdout)
    if data.get("schema") != 2 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("empty/malformed catalog is not coverage")
    for entry in data["cases"]:
        if not isinstance(entry, dict) or set(entry) != {"case", "purpose", "repeats"} \
                or entry["purpose"] not in PURPOSES or isinstance(entry["repeats"], bool) \
                or not isinstance(entry["repeats"], int) or entry["repeats"] < 1:
            raise ValueError("malformed catalog entry: " + json.dumps(entry))
        case_key(entry["case"])
        if entry["purpose"] == "optimized" and entry["repeats"] < OPTIMIZED_REPEATS:
            raise ValueError("an optimized anchor needs >= %d repeats: %s" % (OPTIMIZED_REPEATS, entry["case"]))
    keys = [e["case"] for e in data["cases"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate cases")
    return data


def selected(args, entry):
    """Whether a listing entry (or a bare key) matches the selectors."""
    key = entry["case"] if isinstance(entry, dict) else entry
    if getattr(args, "purpose", None) and (not isinstance(entry, dict) or entry["purpose"] != args.purpose):
        return False
    if args.case:
        return key == args.case
    fields = key.split("/")
    values = (args.provider, args.route, args.tier, args.edge, args.mode, args.rung, args.selection, getattr(args, "steps", None))
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


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_primitive(row, run_id):
    """Primitive rows are characterization only. Anything claiming more, or missing a reading
    a characterization needs, is rejected rather than reported."""
    def reject(reason):
        return dict(row, status="diagnostic_failure", production_success=False, reason="primitive row rejected: " + reason + "; " + str(row.get("reason")))
    if row.get("evidence_class") != PRIMITIVE_CLASS or row.get("model_admission_evidence") is not False:
        return reject("not labelled primitive characterization / claims model admission")
    if row.get("production_success") is not False or row.get("cap_validated", False) is not False:
        return reject("claims production success or a validated cap")
    if row.get("nonce") != run_id:
        return reject("nonce does not match this launch")
    if row.get("status") != "diagnostic_success":
        return row
    if row.get("allocation_attempted") is not True or row.get("refused_before_allocation") is not False:
        return reject("success without an attempted, unrefused allocation")
    if not isinstance(row.get("entitlement"), str) or not row["entitlement"]:
        return reject("entitlement marker missing")
    primitive = row.get("primitive")
    if not isinstance(primitive, dict) or not _count(primitive.get("required_headroom_bytes")):
        return reject("missing derived headroom requirement")
    for phase in ("before", "resident", "after"):
        reading = row.get(phase)
        if not isinstance(reading, dict):
            return reject("missing " + phase + " reading")
        for name in ("available_bytes", "footprint_bytes", "kernel_lifetime_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes"):
            if not _count(reading.get(name)):
                return reject(f"{phase}.{name} is not a finite device reading")
        if reading["footprint_bytes"] == 0:
            return reject("zero physical footprint is not a device measurement")
    if row["before"]["available_bytes"] < primitive["required_headroom_bytes"]:
        return reject("ran below the derived headroom requirement")
    return row


def render_primitive(row):
    def mib(value):
        return "unavailable" if not _count(value) else f"{value / 1048576:.3f}"
    lines = ["PRIMITIVE CHARACTERIZATION ONLY - not a model render, not admission evidence, not a cap",
             f"status: {row.get('status')} ({row.get('reason')})"]
    primitive = row.get("primitive") or {}
    lines.append("required headroom MiB: " + mib(primitive.get("required_headroom_bytes")) + " (floor " + mib(primitive.get("scratch_floor_bytes")) + " + reserve " + mib(primitive.get("engineering_reserve_bytes")) + ")")
    lines.append("| Phase | Available MiB | Footprint MiB | Kernel lifetime peak MiB | MLX active MiB | MLX cache MiB | MLX peak MiB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for phase in ("before", "resident", "after"):
        r = row.get(phase) or {}
        lines.append("| " + " | ".join([phase] + [mib(r.get(n)) for n in ("available_bytes", "footprint_bytes", "kernel_lifetime_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes")]) + " |")
    lines.append("quantum peak over baseline MiB: " + mib(row.get("quantum_peak_over_baseline_bytes")))
    unaccounted = row.get("unaccounted_by_mlx_peak_bytes")
    lines.append("unaccounted by MLX peak MiB: " + ("unavailable" if not isinstance(unaccounted, int) or isinstance(unaccounted, bool) else f"{unaccounted / 1048576:.3f}"))
    return "\n".join(lines)


def run_primitive(args):
    """One cold launch of the reserved primitive key through the same DeviceSession."""
    args.diagnostic = True
    with args.output.open("x", encoding="utf-8") as output:
        session = DeviceSession(args)
        session.install()
        row = invoke(session, PRIMITIVE_KEY)
        # The HOST's launch nonce, never the row's own claim.
        row = validate_primitive(row, getattr(session, "run_id", None))
        output.write(json.dumps(row) + "\n")
    print(render_primitive(row))
    return 0 if row.get("status") == "diagnostic_success" else 2


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
        self.run_id = run_id
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
    lines = ["| Case | Purpose | Probe | Status / refusal | Time s | Active MiB | Cache MiB | Combined MiB | MLX peak MiB | Sampled phys MiB | Kernel peak MiB | Min headroom MiB |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def metric(name, scale=1):
            v = r.get(name)
            return "unavailable" if v is None else f"{v / scale:.2f}"
        cols = [metric("elapsed_seconds")] + [metric(n, 1048576) for n in ("mlx_active_peak_bytes", "mlx_cache_peak_bytes", "mlx_active_cache_peak_bytes", "mlx_peak_bytes", "phys_footprint_peak_bytes", "kernel_lifetime_phys_footprint_peak_bytes")] + [metric("min_headroom_mib")]
        lines.append("| " + " | ".join([r["case"], str(r.get("purpose", "")), "yes" if r.get("refusal_only") else "no", r["status"] + "/" + str(r.get("refusal_kind")), *cols]) + " |")
        quantum = r.get("refused_quantum")
        if isinstance(quantum, dict):
            lines.append(f"|   refused before construction: {quantum.get('label')} needs {quantum.get('scratch_bytes')} + reserve {quantum.get('reserve_bytes')} B, live {quantum.get('available_bytes')} B | | | | | | | | | | | |")
        if r.get("anchor_defect"):
            lines.append(f"|   DEFECT: {r['anchor_defect']} | | | | | | | | | | | |")
    lines.append("\nCapacity-refused next-up probes: " + json.dumps([r["case"] for r in rows if r.get("refusal_only") and capacity_refusal(r)]))
    # Per anchor case: completed repeats (demand MiB = kernel lifetime peak - capture baseline), or
    # the typed refusal.
    anchors = {}
    for r in rows:
        if r.get("purpose") not in ("resident", "optimized"):
            continue
        entry = anchors.setdefault(r["case"], {"purpose": r["purpose"], "completed_demand_mib": [], "refused": None})
        capture = capture_of(r)
        if capture:
            entry["completed_demand_mib"].append(round((capture["footprint_bytes"] - capture["baseline_footprint_bytes"]) / 1048576, 1))
        elif r["status"] == "capture_refused":
            entry["refused"] = (r.get("refused_quantum") or {}).get("label")
    lines.append("ANCHOR captures (diagnostic, not admission): " + json.dumps(anchors, sort_keys=True))
    return "\n".join(lines)


def capture(args, entries, output):
    """Run each selected CAPTURE case (resident, optimized, validation_steps) its listed number of
    times, each a cold launch. A repeat stops at the first non-completed run (a refused or failed
    case is not retried). The fully optimized anchor must complete: anything else is recorded as a
    DEFECT. No size or selection ladder is walked: the anchor law derives everything in between."""
    rows = []
    session = DeviceSession(args)
    session.install()
    executed = 0

    def record(row):
        output.write(json.dumps(row) + "\n")
        output.flush()
        rows.append(row)

    for entry in entries:
        if entry["purpose"] not in CAPTURE_PURPOSES:
            continue
        for repeat in range(entry["repeats"]):
            key = entry["case"]
            if (args.max_cases is not None and executed >= args.max_cases) or session.blocked:
                record(dict(failure(key, "execution budget reached or prior device exit unverified"), status="skipped", purpose=entry["purpose"], repeat=repeat))
                break
            executed += 1
            row = dict(invoke(session, key), purpose=entry["purpose"], repeat=repeat)
            if entry["purpose"] in ("optimized", "validation_steps") and row["status"] != "diagnostic_success":
                row["anchor_defect"] = "the fully optimized configuration must complete on the phone; " + row["status"] + ": " + str(row.get("reason"))
            record(row)
            if row["status"] != "diagnostic_success":
                break
    return rows


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def capture_of(r):
    """A completed capture with full device telemetry, as a schema-3 `Capture`, else None."""
    peak = r.get("kernel_lifetime_phys_footprint_peak_bytes")
    baseline = r.get("baseline_phys_footprint_bytes")
    if r.get("status") != "diagnostic_success" or r.get("evidence_class") != "capture" or not isinstance(r.get("evidence_key"), dict):
        return None
    if not all(_finite(r.get(n)) for n in ("mlx_peak_bytes", "min_headroom_mib", "elapsed_seconds")) or not isinstance(peak, int) or isinstance(peak, bool):
        return None
    # The capture's starting footprint: required, nonzero and below the peak, so admission charges
    # only the render's own growth.
    if isinstance(baseline, bool) or not isinstance(baseline, int) or not 0 < baseline < peak:
        return None
    steps = r.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        return None
    phases = {}
    for name, value in (r.get("phase_peaks") or {}).items():
        if not isinstance(value, dict):
            continue
        bytes_, exact = value.get("peak_bytes"), value.get("exact")
        if isinstance(bytes_, int) and not isinstance(bytes_, bool) and baseline <= bytes_ <= peak and isinstance(exact, bool):
            phases[name] = {"peak_bytes": bytes_, "exact": exact}
    quanta = {}
    for label, q in (r.get("quanta") or {}).items():
        if isinstance(q, dict) and label and _count(q.get("priced_scratch_bytes")) and _count(q.get("observed_bytes")) \
                and isinstance(q.get("exact"), bool) and q["observed_bytes"] <= peak:
            quanta[label] = {"priced_scratch_bytes": q["priced_scratch_bytes"], "observed_bytes": q["observed_bytes"], "exact": q["exact"]}
    return {"run_id": str(r.get("run_id") or ""), "build_sha": str(r.get("build_sha") or ""), "steps": steps,
            "baseline_footprint_bytes": baseline, "footprint_bytes": peak, "phase_peaks": phases, "quanta": quanta,
            "mlx_peak_bytes": int(r["mlx_peak_bytes"]), "min_headroom_mib": float(r["min_headroom_mib"]),
            "elapsed_seconds": float(r["elapsed_seconds"])}


def anchor_rows(rows):
    """Candidate schema-3 ANCHOR rows from capture rows: per (evidence key, purpose) the completed
    repeats, or a resident anchor's typed pre-allocation refusal. Returns (anchors, problems).
    Printed for review; never written to the table."""
    groups = {}
    for r in rows:
        if r.get("purpose") in ("resident", "optimized") and isinstance(r.get("evidence_key"), dict):
            groups.setdefault((json.dumps(r["evidence_key"], sort_keys=True), r["purpose"]), []).append(r)
    anchors, problems = [], []
    for (key, purpose), group in sorted(groups.items()):
        selection = next((r.get("anchor_selection") for r in group if isinstance(r.get("anchor_selection"), dict)), None)
        if selection is None:
            problems.append(f"{purpose} {group[0]['case']}: no anchor selection recorded")
            continue
        completed = [c for c in map(capture_of, group) if c and c["run_id"] and c["build_sha"] and c["steps"] >= 1]
        refused = [r for r in group if r.get("status") == "capture_refused" and isinstance(r.get("refused_quantum"), dict)]
        base = {"key": json.loads(key), "anchor_kind": purpose, "selection": selection, "evidence_class": "device_capture"}
        if completed:
            if purpose == "optimized" and len(completed) < OPTIMIZED_REPEATS:
                problems.append(f"optimized {group[0]['case']}: {len(completed)} completed repeats, {OPTIMIZED_REPEATS} needed for the margin")
                continue
            anchors.append(dict(base, outcome={"completed": {"captures": completed}}))
        elif refused and purpose == "resident":
            q = refused[0]["refused_quantum"]
            anchors.append(dict(base, outcome={"refused": {"refused_quantum": {"label": str(q.get("label")), "scratch_bytes": int(q.get("scratch_bytes") or 0), "reserve_bytes": int(q.get("reserve_bytes") or 0), "available_bytes": q.get("available_bytes")}, "run_id": str(refused[0].get("run_id") or ""), "build_sha": str(refused[0].get("build_sha") or "")}}))
        else:
            problems.append(f"{purpose} {group[0]['case']}: no completed capture" + (" (DEFECT: the fully optimized anchor must fit)" if purpose == "optimized" else ""))
    return anchors, problems


def calibration_table(rows):
    """The per-formula DEVICE calibration across every completed capture: per quantum label its
    worst EXACT observed/priced invocation (priced scratch vs device-observed growth), how many
    captures observed it, and whether the formula over-prices the device beyond the low-end margin
    (there, admitted work's guard runs at the anchor-derived quantum peak instead of its formula)."""
    table = {}
    for r in rows:
        c = capture_of(r)
        if not c:
            continue
        for label, q in c["quanta"].items():
            entry = table.setdefault(label, {"priced_scratch_bytes": None, "observed_bytes": None, "exact_captures": 0, "masked_captures": 0})
            if not q["exact"]:
                entry["masked_captures"] += 1
                continue
            entry["exact_captures"] += 1
            p, o = q["priced_scratch_bytes"], q["observed_bytes"]
            if p and (entry["priced_scratch_bytes"] is None or o * entry["priced_scratch_bytes"] > entry["observed_bytes"] * p):
                entry["priced_scratch_bytes"], entry["observed_bytes"] = p, o
    for entry in table.values():
        p, o = entry["priced_scratch_bytes"], entry["observed_bytes"]
        entry["priced_over_observed"] = round(p / o, 4) if p and o else None
        entry["over_priced_beyond_margin"] = bool(p is not None and p - o > ANCHOR_MARGIN_FLOOR_BYTES)
    return dict(sorted(table.items()))


def steps_validation(rows):
    """The short-step claim, checked: each validation_steps (full steps) capture against the
    optimized anchor of the same case at the anchors' step count. Ratios of growth (whole run and per
    phase) — a full-step peak above the short-step one means short captures under-measure."""
    out = []
    anchors = {}
    for r in rows:
        c = capture_of(r)
        if r.get("purpose") == "optimized" and c:
            anchors.setdefault("/".join(r["case"].split("/")[:7]), []).append(c)
    for r in rows:
        c = capture_of(r)
        if r.get("purpose") != "validation_steps" or not c:
            continue
        short = anchors.get("/".join(r["case"].split("/")[:7]))
        if not short:
            out.append({"case": r["case"], "problem": "no completed optimized anchor of the same case"})
            continue
        full_demand = c["footprint_bytes"] - c["baseline_footprint_bytes"]
        short_demand = max(s["footprint_bytes"] - s["baseline_footprint_bytes"] for s in short)
        phases = {}
        for name, value in c["phase_peaks"].items():
            shorts = [s["phase_peaks"][name]["peak_bytes"] - s["baseline_footprint_bytes"] for s in short if name in s["phase_peaks"]]
            if shorts and max(shorts) > 0:
                phases[name] = round((value["peak_bytes"] - c["baseline_footprint_bytes"]) / max(shorts), 4)
        out.append({"case": r["case"], "full_steps_demand_bytes": full_demand, "short_steps_demand_bytes": short_demand,
                    "ratio": round(full_demand / short_demand, 4) if short_demand else None, "phase_ratios": phases})
    return out


def geometry_validation(rows):
    """The anchor law's prediction (the admitted demand) against the measured growth of each
    ADMITTED validation_geometry run."""
    out = []
    for r in rows:
        if r.get("purpose") not in ADMITTED_VALIDATIONS:
            continue
        admitted = r.get("admitted") if isinstance(r.get("admitted"), dict) else {}
        peak, baseline = r.get("kernel_lifetime_phys_footprint_peak_bytes"), r.get("baseline_phys_footprint_bytes")
        predicted = admitted.get("demand_bytes")
        measured = peak - baseline if isinstance(peak, int) and isinstance(baseline, int) and 0 < baseline < peak else None
        out.append({"case": r["case"], "purpose": r.get("purpose"), "status": r.get("status"), "basis": admitted.get("basis"),
                    "predicted_demand_bytes": predicted, "margin_bytes": admitted.get("margin_bytes"),
                    "measured_demand_bytes": measured,
                    "relative_error": round((predicted - measured) / measured, 4) if isinstance(predicted, int) and measured else None})
    return out


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.action == "evidence":
        if not args.input:
            p.error("evidence requires --input (capture output)")
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
        anchors, problems = anchor_rows(rows)
        print(f"# Candidate anchor rows for {EVIDENCE_TABLE} (schema 3). Review, then commit by hand; capture never changes admission.")
        print(json.dumps({"schema": 3, "anchors": anchors}, indent=2))
        print("# Short-step validation (full / short growth): " + json.dumps(steps_validation(rows)))
        print("# Per-formula DEVICE calibration (priced scratch vs observed growth): " + json.dumps(calibration_table(rows)))
        for problem in problems:
            print("# PROBLEM: " + problem)
        return 0 if not problems else 2
    if args.action == "table":
        if not args.input:
            p.error("table requires --input")
        print(render([json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]))
        return 0
    if args.action == "primitive":
        if not args.device or not args.output:
            p.error("primitive requires explicit --device and new --output")
        if any(v is not None for v in (args.case, args.provider, args.route, args.tier, args.edge, args.mode, args.rung, args.selection, args.next_edge, args.steps, args.purpose)) or args.all_cases:
            p.error("primitive takes no catalog selectors")
        return run_primitive(args)
    data = catalog(args)
    entries = [e for e in data["cases"] if selected(args, e)]
    keys = [e["case"] for e in entries]
    if not keys and not args.case:
        p.error("selection matches no actual catalog cases")
    if args.action == "list":
        print(json.dumps(dict(data, cases=entries), indent=2))
        return 0
    if not args.device or not args.output:
        p.error("run requires explicit --device and new --output")
    if not args.all_cases and not args.case and not any((args.provider, args.route, args.tier, args.edge, args.mode, args.selection, args.purpose, args.rung is not None, args.steps is not None)):
        p.error("run requires selectors or --all-cases")
    if args.action == "capture":
        if args.next_edge is not None or args.case:
            p.error("capture runs the listed anchor cases; no --case/--next-edge")
        args.diagnostic = True
        with args.output.open("x", encoding="utf-8") as output:
            rows = capture(args, entries, output)
        print(render(rows))
        ok = rows and all(r["status"] == "diagnostic_success" or (r["status"] == "capture_refused" and r.get("purpose") == "resident") for r in rows)
        return 0 if ok else 2
    if args.action == "validate":
        entries = [e for e in entries if e["purpose"] in ADMITTED_VALIDATIONS]
        if not entries:
            p.error("validate runs the listing's validation_geometry / validation_mode cases; none selected")
        args.diagnostic = False
        rows = []
        with args.output.open("x", encoding="utf-8") as output:
            session = DeviceSession(args)
            session.install()
            for entry in entries:
                row = dict(invoke(session, entry["case"]), purpose=entry["purpose"])
                output.write(json.dumps(row) + "\n")
                output.flush()
                rows.append(row)
        print(render(rows))
        report = geometry_validation(rows)
        print("Anchor-law prediction vs measurement: " + json.dumps(report))
        return 0 if all(r["status"] == "successful" and r["relative_error"] is not None for r in report) else 2
    if args.case and args.case not in keys:
        # Any configuration of the catalog's domain may be run ADMITTED (the anchor law derives it).
        case_key(args.case)
        keys = [args.case]
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
            if (args.max_cases is not None and executed >= args.max_cases) or session.blocked:
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
