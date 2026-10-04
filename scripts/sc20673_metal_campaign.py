#!/usr/bin/env python3
"""Bounded SC-20673 upstream Metal reproduction harness."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, re, shlex, subprocess, sys, time
from pathlib import Path
SOURCE_COMMIT = "54989ee223611627592f7f9bd925e924658f1f22"
MAX_OUTPUT = 200_000
COMMANDS = (("parity", "python -m pytest veloxquant_mlx/tests/metal/test_scalar_attend.py veloxquant_mlx/tests/metal/test_rabitq_attend.py veloxquant_mlx/tests/metal/test_rabitq_encode.py veloxquant_mlx/tests/metal/test_rabitq_values.py veloxquant_mlx/tests/metal/test_rabitq_prefill.py veloxquant_mlx/tests/metal/test_kivi_quant.py veloxquant_mlx/tests/metal/test_turboquant_kernels.py veloxquant_mlx/tests/metal/test_rvq_quant_pack.py -q"), ("rabitq_decode_benchmark", "python scripts/metal_rabitq_attend_bench.py"), ("rabitq_encode_benchmark", "python scripts/metal_rabitq_encode_bench.py"), ("rabitq_prefill_benchmark", "python scripts/metal_rabitq_prefill_bench.py"))

def _derive_results(records: list[dict]) -> dict:
    """Parse named benchmark rows from this run; never hand-copy measurements."""
    text = {r["name"]: r.get("stdout_tail", "") for r in records}
    parity = re.findall(r"\|\s*(512|2048|8192|16384)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x", text.get("parity", ""))
    decode = re.findall(r"\|?\s*(512|2048|8192)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x", text.get("rabitq_decode_benchmark", ""))
    prefill = re.findall(r"\s*(256|1024)\s+(2048|8192)\s+\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)x", text.get("rabitq_prefill_benchmark", ""))
    if len(parity) >= 4:
        parity = parity[:4]
        rows = {n: (float(b), float(f), float(s)) for n, b, f, s in parity}
        results = {"scalar_group_affine": {"baseline": "MLX dequantize + SDPA", "timings_ms": {n: [b, f] for n, (b, f, _) in rows.items()}, "speedup": [s for _, _, s in rows.values()]}}
    else:
        results = {}
    if len(decode) == 3:
        rows = {n: (float(f), float(p), float(b), float(s)) for n, f, p, b, s in decode}
        results["rabitq_decode"] = {"baseline": "dequantize + MLX SDPA", "timings_ms": {n: [f, b] for n, (f, _, b, _) in rows.items()}, "speedup": [s for _, _, _, s in rows.values()]}
    if len(prefill) == 3:
        rows = {f"{q}x{kv}": (float(p), float(b), float(s)) for q, kv, p, _, b, s in prefill}
        results["rabitq_prefill"] = {"baseline": "dequantize + MLX SDPA", "timings_ms": {n: [p, b] for n, (p, b, _) in rows.items()}, "speedup": [s for _, _, s in rows.values()]}
    return results
def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--timeout", type=int, default=300); p.add_argument("--skip-benchmarks", action="store_true"); a = p.parse_args(); source = a.source.resolve()
    if not (source / ".git").exists(): p.error(f"not a git checkout: {source}")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, encoding="utf-8", errors="strict").strip()
    if commit != SOURCE_COMMIT: p.error(f"source HEAD {commit} is not frozen commit {SOURCE_COMMIT}")
    probe_path = a.output.with_suffix(".probe.json")
    probe = subprocess.run([sys.executable, str(Path(__file__).with_name("sc20673_frozen_probe.py")), "--source", str(source), "--output", str(probe_path)], text=True, encoding="utf-8", errors="strict", capture_output=True, timeout=a.timeout)
    if probe.returncode: p.error(f"fresh frozen probe failed: {probe.stderr[-MAX_OUTPUT:]}")
    probe_json = json.loads(probe_path.read_text(encoding="utf-8"))
    probe_path.unlink()
    records = []
    for name, command in (COMMANDS[:1] if a.skip_benchmarks else COMMANDS):
        started = time.monotonic()
        try:
            argv = shlex.split(command.replace("python ", f"{sys.executable} "))
            proc = subprocess.run(argv, cwd=source, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=a.timeout)
            elapsed = round(time.monotonic()-started, 3)
            records.append({"name": name, "command": command, "returncode": proc.returncode, "elapsed_s": elapsed, "stdout_tail": (proc.stdout+proc.stderr)[-MAX_OUTPUT:]})
        except subprocess.TimeoutExpired as exc:
            records.append({"name": name, "command": command, "returncode": 124, "elapsed_s": round(time.monotonic()-started, 3), "stdout_tail": str(exc.output or "")[-MAX_OUTPUT:], "timeout_s": a.timeout})
    dependency_hash = hashlib.sha256((source / "pyproject.toml").read_bytes()).hexdigest()
    receipt = {"schemaVersion": 3, "story": "SC-20673", "upstream": {"repository": "https://github.com/rajveer43/VeloxQuant-MLX", "tag": "v0.65.0", "commit": SOURCE_COMMIT}, "host": {"system": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "mlx": importlib.metadata.version("mlx"), "mlx_lm": importlib.metadata.version("mlx-lm"), "mlx_metal": importlib.metadata.version("mlx-metal"), "device": probe_json["deviceInfo"]}, "scope": {"sourceExternal": True, "noWeightsDownloaded": True, "maxOutputBytes": MAX_OUTPUT}, "provenance": {"inference_base": "3deb898c8dfa572e939ba9705adfe311dd6d43f0", "dependency_manifest": "pyproject.toml", "dependency_manifest_sha256": dependency_hash}, "probe": probe_json, "upstream_benchmarks": records, "product_eligibility": "pending independent SceneWorks integration; upstream benchmarks and isolated probes are not product evidence"}
    coverage_path = Path(__file__).parents[1] / "docs/architecture/receipts/sc-20673-coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        for obsolete in ("results", "timing", "timing_measurements"):
            coverage.pop(obsolete, None)
        coverage.get("provenance", {}).pop("run_receipt_sha256", None)
        derived = _derive_results(records)
        if not derived:
            p.error("failed to derive all named upstream benchmark results")
        coverage["upstream_results"] = derived
        coverage["observed_raw_rows"] = [line.strip() for record in records for line in record["stdout_tail"].splitlines() if "|" in line and line.strip()]
        coverage["probe"] = probe_json
        coverage["probe_results"] = {r["name"]: r["metrics"] for r in probe_json["probes"]}
        coverage["physical_bytes"] = {r["name"]: r["physical_bytes"] for r in probe_json["probes"]}
        coverage["provenance"]["host"] = receipt["host"]
        coverage["provenance"]["dependency_manifest_sha256"] = dependency_hash
        coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        coverage_bytes = coverage_path.read_bytes()
        coverage_path.with_suffix(coverage_path.suffix + ".sha256").write_text(hashlib.sha256(coverage_bytes).hexdigest() + "  " + coverage_path.name + "\n", encoding="utf-8")
        receipt["coverage"] = coverage
    encoded = (json.dumps(receipt, indent=2, sort_keys=True)+"\n").encode(); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_bytes(encoded); a.output.with_suffix(a.output.suffix + ".sha256").write_text(hashlib.sha256(encoded).hexdigest() + "  " + a.output.name + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), "sha256": hashlib.sha256(encoded).hexdigest(), "commands": len(records), "failed": sum(r["returncode"] != 0 for r in records)})); return 0 if all(r["returncode"] == 0 for r in records) else 1
if __name__ == "__main__": raise SystemExit(main())
