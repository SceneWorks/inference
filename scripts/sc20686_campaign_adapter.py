#!/usr/bin/env python3
"""Campaign-only producer for SC-20686 sealed attribution rows."""
import argparse, hashlib, json, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

REDUCER = Path(__file__).with_name("sc20686_cache_attribution.py")
PRODUCER = "sc20686-campaign-adapter-v1"
GEOMETRY = ("resolution", "reference_count", "frames", "prompt", "guidance", "layers",
            "heads", "head_dimension", "sq", "skv", "dtype", "mask", "rope")

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
    metadata = next((e for e in events if e.get("phase") == "metadata"), None)
    geometry = metadata.get("geometry") if metadata else None
    if not isinstance(geometry, dict) or any(k not in geometry for k in GEOMETRY):
        raise ValueError("producer metadata must contain all typed geometry axes")
    return {k: geometry[k] for k in GEOMETRY}

def fake_events():
    now = time.monotonic_ns()
    return [{"phase": p, "persistent_bytes": 600 * 1024**2 if p in ("cross-kv-created", "cross-kv-read") else 0,
             "transient_bytes": 32 * 1024**2 if p == "cross-kv-read" else 0,
             "peak_bytes": 10 * 1024**3, "at_ns": now + i * 100_000_000} for i, p in enumerate(("generation-start", "cross-kv-created", "cross-kv-read", "generation-end", "invalidated", "cancelled", "released"))]

def make_row(args, config, snapshot_hash, snapshot_bytes, events):
    if args.fake:
        raise ValueError("fake evidence is test-only and cannot produce a receipt")
    metadata = next((e for e in events if e.get("phase") == "metadata"), None)
    metrics = next((e for e in events if e.get("phase") == "metrics"), None)
    if not metadata or not metrics or metadata.get("snapshot_sha256") != snapshot_hash:
        raise ValueError("entrypoint must emit matching model identity metadata")
    source_ref = metadata.get("source_ref")
    if not isinstance(source_ref, str) or len(source_ref) != 40 or any(c not in "0123456789abcdef" for c in source_ref):
        raise ValueError("entrypoint must emit immutable 40-hex source ref")
    if metadata.get("variant") != args.variant:
        raise ValueError("producer variant does not match requested route")
    metric_keys = ("current_persistent_bytes", "current_read_transient_bytes", "candidate_persistent_bytes", "candidate_read_transient_bytes", "generation_duration_ms", "cache_read_duration_ms", "reused_requests")
    if any(not isinstance(metrics.get(k), (int, float)) for k in metric_keys):
        raise ValueError("entrypoint metrics are incomplete")
    phases = {e.get("phase") for e in events}
    if not {"generation-start", "generation-end", "cross-kv-created", "cross-kv-read", "invalidated", "released"} <= phases:
        raise ValueError("observer lifecycle/phase hooks are incomplete")
    cancelled = [i for i, e in enumerate(events) if e.get("phase") == "cancelled"]
    released = [i for i, e in enumerate(events) if e.get("phase") == "released"]
    if args.cancel_campaign and (metadata.get("cancellation_armed") is not True or not metadata.get("cancellation_arm_id")):
        raise ValueError("deliberate cancellation requires producer cancellation-arm identity")
    if not released or (args.cancel_campaign and (not cancelled or max(cancelled) > min(released))):
        raise ValueError("deliberate cancellation cleanup must precede release")
    if not args.cancel_campaign and cancelled:
        raise ValueError("normal generation cannot contain cancellation")
    geometry = geometry_from(events)
    allocator = [{"phase": e["phase"], "peak_bytes": e["peak_bytes"]} for e in events
                 if e.get("sample_kind") == "allocator" and "peak_bytes" in e]
    process = [{"phase": e["phase"], "peak_bytes": e["peak_bytes"]} for e in events
               if e.get("sample_kind") == "process" and "peak_bytes" in e]
    if not allocator or not process:
        raise ValueError("producer must provide distinct allocator and process samples")
    return {"producer": PRODUCER, "family": args.family, "variant": args.variant,
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
    p = argparse.ArgumentParser(); p.add_argument("--campaign", action="store_true"); p.add_argument("--family", choices=("flux2-klein", "wan"), required=True)
    p.add_argument("--snapshot", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--variant", required=True)
    p.add_argument("--events", type=Path); p.add_argument("--entrypoint", type=Path); p.add_argument("--cancel-campaign", action="store_true"); p.add_argument("--fake", action="store_true"); args = p.parse_args()
    if not args.campaign: p.error("SC-20686 adapter requires explicit --campaign")
    root = args.snapshot.resolve(); config_path = root / "config.json"
    try:
        if not root.is_dir() or not config_path.is_file(): raise ValueError("snapshot must contain config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.fake:
            events = fake_events()
        elif args.entrypoint:
            child = subprocess.Popen([str(args.entrypoint), "--sc20686-campaign", "--sc20686-events", "-", "--snapshot", str(root)], text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            events = json.loads(args.events.read_text(encoding="utf-8")) if args.events else None
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
