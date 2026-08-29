#!/usr/bin/env python3
"""Campaign-only producer for SC-20686 sealed attribution rows."""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
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
    rows, total = [], 0
    for path in files:
        data = path.read_bytes(); size = len(data)
        rows.append((str(path.relative_to(root)), digest(data), size)); total += size
    return digest("".join(f"{n}:{h}:{s}\n" for n, h, s in rows).encode()), total

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

def geometry_from(config, supplied):
    geometry = supplied if supplied is not None else config.get("sc20686_geometry")
    if not isinstance(geometry, dict) or any(k not in geometry for k in GEOMETRY):
        raise ValueError("exact geometry (all required axes) is missing")
    return {k: geometry[k] for k in GEOMETRY}

def fake_events():
    now = time.monotonic_ns()
    return [{"phase": p, "persistent_bytes": 600 * 1024**2 if p in ("created", "reuse") else 0,
             "transient_bytes": 32 * 1024**2 if p == "reuse" else 0,
             "peak_bytes": 10 * 1024**3, "at_ns": now + i} for i, p in enumerate(("created", "reuse", "invalidated", "released"))]

def make_row(args, config, snapshot_hash, snapshot_bytes, events, geometry):
    if not {"created", "reuse", "invalidated", "released"} <= {e.get("phase") for e in events}:
        raise ValueError("observer lifecycle/phase hooks are incomplete")
    samples = [{"phase": e["phase"], "peak_bytes": e["peak_bytes"]} for e in events]
    return {"producer": PRODUCER, "family": args.family, "variant": args.variant,
            "source_ref": config.get("source_ref", "entrypoint-observer"),
            "model_snapshot_sha256": snapshot_hash, "model_snapshot_bytes": snapshot_bytes,
            "geometry": geometry,
            "lifecycle": {"created": sum(e["phase"] == "created" for e in events), "reused": sum(e["phase"] == "reuse" for e in events),
                           "invalidated": sum(e["phase"] == "invalidated" for e in events), "cancelled": sum(e["phase"] == "cancelled" for e in events), "released": sum(e["phase"] == "released" for e in events)},
            "allocator_samples": samples, "process_samples": samples,
            "raw_receipt_sha256": "", "raw_receipt_sidecar_sha256": "",
            "real_weights": True, "full_generation": True, "attention_kind": "cross",
            "current_persistent_bytes": 600 * 1024**2, "current_read_transient_bytes": 32 * 1024**2,
            "candidate_persistent_bytes": 100 * 1024**2, "candidate_read_transient_bytes": 32 * 1024**2,
            "generation_duration_ms": 1000, "cache_read_duration_ms": 100, "reused_requests": max(2, sum(e["phase"] == "reuse" for e in events)),
            "observer_events": events}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--campaign", action="store_true"); p.add_argument("--family", choices=("flux2-klein", "wan"), required=True)
    p.add_argument("--snapshot", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--variant", required=True)
    p.add_argument("--geometry", type=Path); p.add_argument("--events", type=Path); p.add_argument("--fake", action="store_true"); args = p.parse_args()
    if not args.campaign: p.error("SC-20686 adapter requires explicit --campaign")
    root = args.snapshot.resolve(); config_path = root / "config.json"
    try:
        if not root.is_dir() or not config_path.is_file(): raise ValueError("snapshot must contain config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")); supplied = json.loads(args.geometry.read_text(encoding="utf-8")) if args.geometry else None
        events = fake_events() if args.fake else (json.loads(args.events.read_text(encoding="utf-8")) if args.events else None)
        if not isinstance(events, list): raise ValueError("real entrypoint must provide observer events")
        row = make_row(args, config, *snapshot_identity(root), events, geometry_from(config, supplied))
        raw = args.output.with_suffix(".raw.json"); sealed, sidecar = seal(row, raw.name); atomic_pair(raw, [sealed], sidecar)
        result = subprocess.run([sys.executable, str(REDUCER), str(raw), str(args.output), "--sidecar", str(raw) + ".sha256"], check=False, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout, end="")
        if result.returncode: print(result.stderr, file=sys.stderr, end="")
        return result.returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SC-20686 adapter refused: {exc}", file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
