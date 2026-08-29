#!/usr/bin/env python3
"""Campaign-only producer for SC-20686 sealed attribution rows."""
import argparse, hashlib, json, math, os, re, struct, subprocess, sys, tempfile, time
from pathlib import Path

REDUCER = Path(__file__).with_name("sc20686_cache_attribution.py")
PRODUCER = "sc20686-campaign-adapter-v1"
GEOMETRY = ("resolution", "reference_count", "frames", "prompt", "guidance", "layers",
            "heads", "head_dimension", "sq", "skv", "dtype", "mask", "rope")
WAN_ROUTES = ("wan2_2_ti2v_5b", "wan2_2_t2v_14b", "wan2_2_i2v_14b", "wan_vace", "wan2_2_vace_fun_14b")
FLUX_ROUTES = ("flux2_klein_9b_edit",)

def load_wan_manifest(path):
    """Load the product-owned five-route executable/snapshot mapping.

    A single binary/snapshot cannot substantiate all Wan variants: each route owns a distinct
    provider and model inventory. The manifest is therefore required for matrix execution.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Wan campaign manifest is missing")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Wan campaign manifest is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != set(WAN_ROUTES):
        raise ValueError("Wan campaign manifest must contain exactly the five registered routes")
    result = {}
    for route in WAN_ROUTES:
        entry = document[route]
        if not isinstance(entry, dict) or not isinstance(entry.get("entrypoint"), str) \
                or not isinstance(entry.get("snapshot"), str) or not isinstance(entry.get("args", []), list) \
                or any(not isinstance(arg, str) for arg in entry.get("args", [])):
            raise ValueError(f"Wan manifest entry {route} is malformed")
        protected = {"--sc20686-campaign", "--sc20686-events", "--snapshot", "--variant", "--sc20686-cancel"}
        if protected.intersection(entry.get("args", [])):
            raise ValueError(f"Wan manifest entry {route} attempts to override campaign identity flags")
        binary = Path(entry["entrypoint"]).expanduser().resolve()
        snapshot = Path(entry["snapshot"]).expanduser().resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Wan manifest entrypoint is not executable: {route}")
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            raise ValueError(f"Wan manifest snapshot is missing config.json: {route}")
        result[route] = (binary, snapshot, tuple(entry.get("args", [])))
    return result

def digest(data):
    return hashlib.sha256(data).hexdigest()

def snapshot_identity(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
    if not files: raise ValueError("snapshot inventory is empty")
    aggregate, total = hashlib.sha256(), 0
    for path in files:
        file_hash, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk); total += len(chunk); file_hash.update(chunk)
        aggregate.update(str(path.relative_to(root)).encode())
        aggregate.update(b"\0" + struct.pack("<Q", size) + b"\0" + file_hash.digest() + b"\n")
    return aggregate.hexdigest(), total

def canonical(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

def seal(row, filename):
    unsigned = dict(row); unsigned["raw_receipt_sha256"] = ""; unsigned["raw_receipt_sidecar_sha256"] = ""
    receipt_hash = digest(canonical(unsigned))
    sidecar = f"{receipt_hash}  {filename}\n".encode("utf-8")
    sealed = dict(unsigned); sealed["raw_receipt_sha256"] = receipt_hash; sealed["raw_receipt_sidecar_sha256"] = digest(sidecar)
    return sealed, sidecar

def atomic_pair(raw, row, sidecar):
    raw.parent.mkdir(parents=True, exist_ok=True); side = Path(f"{raw}.sha256")
    handles = [tempfile.NamedTemporaryFile(prefix=f".{p.name}.", dir=p.parent, delete=False) for p in (raw, side)]
    try:
        handles[0].write(canonical(row)); handles[0].close(); handles[1].write(sidecar); handles[1].close()
        os.replace(handles[0].name, raw); os.replace(handles[1].name, side)
    finally:
        for handle in handles:
            try: os.unlink(handle.name)
            except FileNotFoundError: pass

def geometry_from(events):
    metadata_events = [e for e in events if e.get("phase") == "metadata"]
    if len(metadata_events) != 1:
        raise ValueError("producer must emit exactly one post-bind metadata event")
    metadata = metadata_events[0]
    geometry = metadata.get("geometry") if metadata else None
    if not isinstance(geometry, dict) or any(k not in geometry for k in GEOMETRY):
        raise ValueError("producer metadata must contain all typed geometry axes")
    if not isinstance(geometry["resolution"], str) or not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", geometry["resolution"]):
        raise ValueError("producer resolution must be positive WxH")
    for key in ("reference_count", "frames", "layers", "heads", "head_dimension", "sq", "skv"):
        value = geometry[key]
        minimum = 0 if key == "reference_count" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"producer geometry {key} is invalid")
    if not isinstance(geometry["prompt"], str) or len(geometry["prompt"]) != 64 or any(c not in "0123456789abcdef" for c in geometry["prompt"]):
        raise ValueError("producer prompt identity is invalid")
    try: guidance = float(geometry["guidance"])
    except (TypeError, ValueError): raise ValueError("producer guidance is invalid") from None
    if not math.isfinite(guidance): raise ValueError("producer guidance is invalid")
    if any(not isinstance(geometry[key], str) or not geometry[key] for key in ("dtype", "mask", "rope")):
        raise ValueError("producer dtype/mask/RoPE identity is invalid")
    return {k: geometry[k] for k in GEOMETRY}

def fake_events():
    now = time.monotonic_ns()
    return [{"phase": p, "persistent_bytes": 600 * 1024**2 if p in ("cross-kv-created", "cross-kv-read") else 0,
             "transient_bytes": 32 * 1024**2 if p == "cross-kv-read" else 0,
             "peak_bytes": 10 * 1024**3, "at_ns": now + i * 100_000_000} for i, p in enumerate(("generation-start", "cross-kv-created", "cross-kv-read", "generation-end", "invalidated", "cancelled", "released"))]

def dispatch_campaign(runner):
    """Dispatch every registered Wan route exactly once per normal/cancel arm.

    `runner` is the product-owned callback; it must return that run's observer JSONL events.
    Keeping this seam injectable makes orchestration testable without weights or a device.
    """
    dispatched = []
    for variant in WAN_ROUTES:
        for arm in ("normal", "cancel"):
            events = runner(variant, arm)
            if not isinstance(events, list) or not events:
                raise ValueError(f"{variant}/{arm} produced no observer events")
            dispatched.append((variant, arm, events))
    return dispatched

def entrypoint_campaign_runner(entrypoint, snapshot, extra_args=()):
    """Build the real-route runner used by the matrix command.

    The executable owns model loading and observer production; this adapter only supplies the
    registered route and cancellation arm. Missing executables or model paths are hard failures.
    """
    entrypoint = Path(entrypoint)
    if not entrypoint.is_file() or not os.access(entrypoint, os.X_OK):
        raise ValueError("campaign entrypoint is not executable")
    snapshot = Path(snapshot).resolve()
    if not snapshot.is_dir():
        raise ValueError("campaign snapshot path is missing")
    def run(variant, arm):
        command = [str(entrypoint), "--sc20686-campaign", "--sc20686-events", "-",
                   "--snapshot", str(snapshot), "--variant", variant, *map(str, extra_args)]
        if arm == "cancel": command.append("--sc20686-cancel")
        child = subprocess.Popen(command, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_samples = []
        while child.poll() is None:
            try:
                rss_kib = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(child.pid)], text=True).strip() or "0")
                if rss_kib > 0: process_samples.append({"phase": "process-sample", "sample_kind": "process", "peak_bytes": rss_kib * 1024, "at_ns": time.time_ns()})
            except (OSError, subprocess.SubprocessError, ValueError): pass
            time.sleep(0.05)
        stdout, stderr = child.communicate()
        if child.returncode: raise ValueError(f"{variant}/{arm} entrypoint failed: {stderr.strip()}")
        events = [json.loads(line) for line in stdout.splitlines() if line.lstrip().startswith("{")]
        events.extend(process_samples)
        if not events: raise ValueError(f"{variant}/{arm} produced no observer events")
        return events
    return run

def publish_matrix_campaign(runner, row_builder, destination):
    """Run and atomically publish the complete Wan+FLUX normal/cancel matrix.

    The runner owns product execution and `row_builder` must derive a sealed row from each
    same-run event stream. Publication is all-or-nothing: the final directory must not already
    exist, and all artifacts are written under a sibling staging directory before one rename.
    """
    coordinates = [(route, arm) for route in WAN_ROUTES + FLUX_ROUTES for arm in ("normal", "cancel")]
    rows = []
    for variant, arm in coordinates:
        events = runner(variant, arm)
        if not isinstance(events, list) or not events:
            raise ValueError(f"{variant}/{arm} produced no observer events")
        row = row_builder(variant, arm, events)
        if not isinstance(row, dict): raise ValueError("row builder returned no receipt row")
        rows.append(row)
    keys = [(row.get("family"), row.get("variant"), row.get("coordinate_id"), row.get("arm")) for row in rows]
    if len(keys) != len(set(keys)) or len(rows) != len(coordinates):
        raise ValueError("campaign matrix has missing or duplicate coordinates")
    staging = Path(destination).with_name(f".{Path(destination).name}.staging-{os.getpid()}")
    final = Path(destination)
    if final.exists(): raise ValueError("campaign destination already exists")
    try:
        staging.mkdir(parents=False)
        # Validate the complete sealed set before any publication is made.
        import importlib.util
        reducer_spec = importlib.util.spec_from_file_location("sc20686_reducer", REDUCER)
        reducer = importlib.util.module_from_spec(reducer_spec); reducer_spec.loader.exec_module(reducer)
        sealed_rows = []
        row_sidecars = []
        for index, row in enumerate(rows):
            sealed, sidecar = seal(row, f"row-{index:02d}.json")
            reducer.verify_seal_bytes(sealed, sidecar, f"row-{index:02d}.json")
            sealed_rows.append(sealed)
            row_sidecars.append((f"row-{index:02d}.json", canonical(sealed), sidecar))
        decision = reducer.reduce(sealed_rows)
        result = {"schema": "sc-20686-cache-attribution-v2", "decision": decision, "rows": sealed_rows}
        raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        markdown = ("# SC-20686 campaign\n\n" + "\n".join(
            f"- {family}/{variant} coordinate={coordinate} arm={arm}" for family, variant, coordinate, arm in keys
        ) + "\n\n## Decisions\n\n" + json.dumps(decision["decisions"], indent=2, sort_keys=True) + "\n").encode()
        for name, payload in (("campaign.json", raw), ("campaign.md", markdown)):
            (staging / name).write_bytes(payload)
            (staging / f"{name}.sha256").write_text(f"{hashlib.sha256(payload).hexdigest()}  {name}\n", encoding="utf-8")
        for name, payload, sidecar in row_sidecars:
            (staging / name).write_bytes(payload)
            (staging / f"{name}.sha256").write_bytes(sidecar)
        os.replace(staging, final)
    except Exception:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        raise

def make_row(args, config, snapshot_hash, snapshot_bytes, events):
    if args.fake:
        raise ValueError("fake evidence is test-only and cannot produce a receipt")
    metadata = next((e for e in events if e.get("phase") == "metadata"), None)
    metrics = next((e for e in events if e.get("phase") == "metrics"), None)
    if (not metadata or not metrics or metadata.get("snapshot_sha256") != snapshot_hash
            or metadata.get("snapshot_bytes") != snapshot_bytes):
        raise ValueError("entrypoint must emit matching model identity metadata")
    source_ref = metadata.get("source_ref")
    if not isinstance(source_ref, str) or len(source_ref) != 40 or any(c not in "0123456789abcdef" for c in source_ref):
        raise ValueError("entrypoint must emit immutable 40-hex source ref")
    if metadata.get("variant") != args.variant:
        raise ValueError("producer variant does not match requested route")
    metric_keys = ("current_persistent_bytes", "current_read_transient_bytes", "candidate_persistent_bytes", "candidate_read_transient_bytes", "generation_duration_ms", "cache_read_duration_ms", "reused_requests")
    if any(not isinstance(metrics.get(k), (int, float)) or isinstance(metrics.get(k), bool)
           or not math.isfinite(metrics[k]) or metrics[k] < 0 for k in metric_keys):
        raise ValueError("entrypoint metrics are incomplete")
    phases = {e.get("phase") for e in events}
    required = {"generation-start", "cross-kv-created", "cross-kv-read", "invalidated", "released"}
    if args.cancel_campaign:
        required.add("cancelled")
    else:
        required.add("generation-end")
    if not required <= phases:
        raise ValueError("observer lifecycle/phase hooks are incomplete")
    singleton_phases = ("metadata", "generation-start", "invalidated", "released",
                        "cancelled" if args.cancel_campaign else "generation-end")
    phase_indices = {}
    for phase in singleton_phases:
        indices = [index for index, event in enumerate(events) if event.get("phase") == phase]
        if len(indices) != 1:
            raise ValueError(f"producer must emit exactly one {phase} event")
        phase_indices[phase] = indices[0]
    create_indices = [index for index, event in enumerate(events) if event.get("phase") == "cross-kv-created"]
    read_indices = [index for index, event in enumerate(events) if event.get("phase") == "cross-kv-read"]
    terminal = "cancelled" if args.cancel_campaign else "generation-end"
    if not (phase_indices["metadata"] < phase_indices["generation-start"]
            < min(create_indices) <= max(create_indices) < min(read_indices) <= max(read_indices)
            < phase_indices[terminal] < phase_indices["invalidated"] < phase_indices["released"]):
        raise ValueError("observer lifecycle events are out of product order")
    if metadata.get("real_weights") is not True or metadata.get("attention_kind") != "cross":
        raise ValueError("entrypoint must identify a real product cross-attention route")
    expected_full_generation = not args.cancel_campaign
    if metadata.get("full_generation") is not expected_full_generation:
        raise ValueError("entrypoint full-generation claim conflicts with campaign arm")
    cancelled = [i for i, e in enumerate(events) if e.get("phase") == "cancelled"]
    released = [i for i, e in enumerate(events) if e.get("phase") == "released"]
    if args.cancel_campaign and (metadata.get("cancellation_armed") is not True or not metadata.get("cancellation_arm_id")):
        raise ValueError("deliberate cancellation requires producer cancellation-arm identity")
    if not released or (args.cancel_campaign and (not cancelled or max(cancelled) > min(released))):
        raise ValueError("deliberate cancellation cleanup must precede release")
    if not args.cancel_campaign and cancelled:
        raise ValueError("normal generation cannot contain cancellation")
    if args.cancel_campaign and "generation-end" in phases:
        raise ValueError("cancel arm cannot claim successful generation completion")
    if not args.cancel_campaign and metrics["generation_duration_ms"] <= 0:
        raise ValueError("normal arm must record a positive product generation duration")
    reads = [event for event in events if event.get("phase") == "cross-kv-read"]
    if not reads or not any(isinstance(event.get("transient_bytes"), (int, float)) and event["transient_bytes"] > 0 for event in reads):
        raise ValueError("entrypoint must provide a measured dense reference read")
    creates = [event for event in events if event.get("phase") == "cross-kv-created"]
    if args.family == "flux2-klein":
        if metrics["current_persistent_bytes"] != 0 or any(event.get("persistent_bytes") != 0 for event in creates):
            raise ValueError("FLUX edit must report its non-persistent reference route honestly")
    elif metrics["current_persistent_bytes"] == 0:
        raise ValueError("persistent-cache routes cannot claim zero current persistence")
    geometry = geometry_from(events)
    if args.family == "flux2-klein" and geometry["reference_count"] < 1:
        raise ValueError("FLUX edit campaign requires at least one live reference image")
    coordinate_id = digest(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode())[:16]
    arm = "cancel" if args.cancel_campaign else "normal"
    allocator = [{"phase": e["phase"], "peak_bytes": e["peak_bytes"]} for e in events
                 if e.get("sample_kind") == "allocator" and "peak_bytes" in e]
    process = [{"phase": e["phase"], "peak_bytes": e["peak_bytes"]} for e in events
               if e.get("sample_kind") == "process" and "peak_bytes" in e]
    if (not allocator or not process
            or any(not isinstance(sample["peak_bytes"], (int, float)) or sample["peak_bytes"] <= 0
                   for sample in allocator + process)):
        raise ValueError("producer must provide distinct allocator and process samples")
    return {"producer": PRODUCER, "family": args.family, "variant": args.variant,
            "coordinate_id": coordinate_id, "arm": arm,
            "source_ref": source_ref,
            "model_snapshot_sha256": snapshot_hash, "model_snapshot_bytes": snapshot_bytes,
            "geometry": geometry,
            "lifecycle": {"created": sum(e["phase"] == "cross-kv-created" for e in events), "reused": sum(e["phase"] == "cross-kv-read" for e in events),
                           "invalidated": sum(e["phase"] == "invalidated" for e in events), "cancelled": sum(e["phase"] == "cancelled" for e in events), "released": sum(e["phase"] == "released" for e in events)},
            "allocator_samples": allocator, "process_samples": process,
            "raw_receipt_sha256": "", "raw_receipt_sidecar_sha256": "",
            "real_weights": metadata.get("real_weights") is True, "full_generation": metadata.get("full_generation") is True, "attention_kind": metadata.get("attention_kind"),
            "current_persistent_bytes": metrics["current_persistent_bytes"], "current_read_transient_bytes": metrics["current_read_transient_bytes"],
            "candidate_persistent_bytes": metrics["candidate_persistent_bytes"], "candidate_read_transient_bytes": metrics["candidate_read_transient_bytes"],
            "generation_duration_ms": metrics["generation_duration_ms"], "cache_read_duration_ms": metrics["cache_read_duration_ms"], "reused_requests": metrics["reused_requests"],
            "observer_events": events}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--campaign", action="store_true"); p.add_argument("--matrix", action="store_true"); p.add_argument("--family", choices=("flux2-klein", "wan"))
    p.add_argument("--snapshot", type=Path); p.add_argument("--output", type=Path); p.add_argument("--variant")
    p.add_argument("--wan-entrypoint", type=Path); p.add_argument("--wan-snapshot", type=Path); p.add_argument("--wan-manifest", type=Path); p.add_argument("--flux-entrypoint", type=Path); p.add_argument("--flux-snapshot", type=Path); p.add_argument("--flux-reference", type=Path); p.add_argument("--matrix-output", type=Path)
    p.add_argument("--entrypoint", type=Path); p.add_argument("--cancel-campaign", action="store_true"); p.add_argument("--fake", action="store_true"); args = p.parse_args()
    if not args.campaign: p.error("SC-20686 adapter requires explicit --campaign")
    if args.matrix:
        if not all((args.wan_manifest, args.flux_entrypoint, args.flux_snapshot, args.flux_reference, args.matrix_output)):
            p.error("matrix mode requires --wan-manifest, separate FLUX entrypoint/snapshot, --flux-reference, and --matrix-output")
        wan_manifest = load_wan_manifest(args.wan_manifest)
        flux_runner = entrypoint_campaign_runner(args.flux_entrypoint, args.flux_snapshot, ("--reference", args.flux_reference, "--single-only"))
        def runner(variant, arm):
            if variant in FLUX_ROUTES:
                return flux_runner(variant, arm)
            entrypoint, snapshot, route_args = wan_manifest[variant]
            return entrypoint_campaign_runner(entrypoint, snapshot, route_args)(variant, arm)
        def build_row(variant, arm, events):
            family = "flux2-klein" if variant in FLUX_ROUTES else "wan"
            snapshot = args.flux_snapshot if family == "flux2-klein" else wan_manifest[variant][1]
            snapshot_hash, snapshot_bytes = snapshot_identity(snapshot.resolve())
            row_args = argparse.Namespace(fake=False, family=family, variant=variant, cancel_campaign=arm == "cancel")
            return make_row(row_args, {}, snapshot_hash, snapshot_bytes, events)
        try:
            publish_matrix_campaign(runner, build_row, args.matrix_output)
            return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"SC-20686 matrix refused: {exc}", file=sys.stderr); return 1
    if not all((args.family, args.snapshot, args.output, args.variant)):
        p.error("single mode requires --family, --snapshot, --output, and --variant")
    root = args.snapshot.resolve(); config_path = root / "config.json"
    try:
        if not root.is_dir() or not config_path.is_file(): raise ValueError("snapshot must contain config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.fake:
            events = fake_events()
        elif args.entrypoint:
            command = [str(args.entrypoint), "--sc20686-campaign", "--sc20686-events", "-", "--snapshot", str(root), "--variant", args.variant]
            if args.cancel_campaign: command.append("--sc20686-cancel")
            if args.family == "flux2-klein":
                if not args.flux_reference:
                    raise ValueError("single FLUX campaign requires --flux-reference")
                command.extend(("--reference", str(args.flux_reference), "--single-only"))
            child = subprocess.Popen(command, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_samples = []
            while child.poll() is None:
                try:
                    rss_kib = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(child.pid)], text=True).strip() or "0")
                    if rss_kib > 0:
                        process_samples.append({"phase": "process-sample", "sample_kind": "process", "peak_bytes": rss_kib * 1024, "at_ns": time.time_ns()})
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                time.sleep(0.05)
            stdout, stderr = child.communicate()
            if child.returncode: raise ValueError(f"campaign entrypoint failed: {stderr.strip()}")
            events = [json.loads(line) for line in stdout.splitlines()
                      if line.lstrip().startswith("{")]
            events.extend(process_samples)
        else:
            events = fake_events() if args.fake else None
        if not isinstance(events, list): raise ValueError("real entrypoint must provide observer events")
        row = make_row(args, config, *snapshot_identity(root), events)
        raw = args.output.with_suffix(".raw.json"); sealed, sidecar = seal(row, raw.name); atomic_pair(raw, [sealed], sidecar)
        result = subprocess.run([sys.executable, str(REDUCER), str(raw), str(args.output), "--sidecar", str(raw) + ".sha256"], check=False, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout, end="")
        if result.returncode: print(result.stderr, file=sys.stderr, end="")
        return result.returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SC-20686 adapter refused: {exc}", file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
