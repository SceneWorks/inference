#!/usr/bin/env python3
"""Fail-closed reducer for injected, real-weight SC-20686 generation receipts."""
import argparse, hashlib, json, sys
from pathlib import Path
MANIFEST=Path(__file__).with_name("sc20686_coverage_manifest.json")

FAMILIES=("flux2-klein","wan")
THRESHOLDS={"opportunity_bytes":512*1024**2,"opportunity_peak_pct":.05,"reused_requests":2,"saving_bytes":256*1024**2,"saving_peak_pct":.03,"runtime_only_pct":.05}
REQUIRED=("producer","family","variant","source_ref","model_snapshot_sha256","model_snapshot_bytes","geometry","lifecycle","allocator_samples","process_samples","raw_receipt_sha256","raw_receipt_sidecar_sha256","real_weights","full_generation","attention_kind","current_persistent_bytes","current_read_transient_bytes","candidate_persistent_bytes","candidate_read_transient_bytes","generation_duration_ms","cache_read_duration_ms","reused_requests")
GEOMETRY=("resolution","reference_count","frames","prompt","guidance","layers","heads","head_dimension","sq","skv","dtype","mask","rope")
LIFECYCLE=("created","reused","invalidated")

def fail(message): raise ValueError(message)
def digest(value,name):
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value): fail(f"invalid {name}")
def positive(value,name):
    if not isinstance(value,(int,float)) or value<0: fail(f"invalid {name}")
def validate(row):
    missing=set(REQUIRED)-row.keys()
    if missing: fail(f"missing fields: {sorted(missing)}")
    if row["producer"]!="sc20686-campaign-adapter-v1": fail("untrusted producer")
    if row["family"] not in FAMILIES or not isinstance(row["variant"],str) or not row["variant"]: fail("invalid family/variant")
    digest(row["model_snapshot_sha256"],"model snapshot hash"); digest(row["raw_receipt_sha256"],"raw receipt hash"); digest(row["raw_receipt_sidecar_sha256"],"receipt sidecar hash")
    if row["real_weights"] is not True or row["full_generation"] is not True: fail("real full-generation receipt required")
    if row["attention_kind"]!="cross": fail("self-attention is excluded")
    if not isinstance(row["geometry"],dict) or set(GEOMETRY)-row["geometry"].keys(): fail("incomplete geometry")
    if not isinstance(row["lifecycle"],dict) or set(LIFECYCLE)-row["lifecycle"].keys(): fail("incomplete lifecycle")
    if not isinstance(row["allocator_samples"],list) or not row["allocator_samples"] or not isinstance(row["process_samples"],list) or not row["process_samples"]: fail("missing allocator/process samples")
    for key in ("model_snapshot_bytes","current_persistent_bytes","current_read_transient_bytes","candidate_persistent_bytes","candidate_read_transient_bytes","generation_duration_ms","cache_read_duration_ms","reused_requests"): positive(row[key],key)
    if row["generation_duration_ms"]==0 or row["cache_read_duration_ms"]>row["generation_duration_ms"]: fail("invalid runtime duration")
    for sample in row["allocator_samples"]+row["process_samples"]: positive(sample.get("peak_bytes"),"sample peak_bytes")

def verify_seal(row, sidecar):
    unsigned=dict(row); unsigned["raw_receipt_sha256"]=""; unsigned["raw_receipt_sidecar_sha256"]=""
    expected=hashlib.sha256((json.dumps(unsigned,sort_keys=True,separators=(",", ":"))+"\n").encode("utf-8")).hexdigest()
    if expected != row["raw_receipt_sha256"]: fail("raw receipt checksum mismatch")
    data=sidecar.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["raw_receipt_sidecar_sha256"]: fail("sidecar checksum mismatch")
    fields=data.decode("utf-8").strip().split(None,1)
    if len(fields)!=2 or fields[0]!=row["raw_receipt_sha256"]: fail("sidecar receipt mismatch")

def reduce(rows):
    if not isinstance(rows,list) or not rows: fail("rows must be non-empty")
    try: manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): fail("checked-in coverage manifest unavailable")
    if manifest.get("schema")!="sc-20686-supported-coverage-v1": fail("invalid coverage manifest")
    for row in rows: validate(row)
    for row in rows:
        if row["variant"] not in manifest.get("families",{}).get(row["family"],[]): fail("variant absent from coverage manifest")
        if any(k not in row["geometry"] for k in manifest.get("required_geometry_axes",[])): fail("coverage geometry incomplete")
    keys=[(r["family"],r["variant"],json.dumps(r["geometry"],sort_keys=True)) for r in rows]
    if len(keys)!=len(set(keys)): fail("duplicate family/variant/geometry")
    decisions={}
    for family in FAMILIES:
        rs=[r for r in rows if r["family"]==family]
        variants={r["variant"] for r in rs}
        if not rs or len(variants)<1: decisions[family]={"decision":"blocked","reason":"required real-weight family coverage missing"}; continue
        peak=max(max(s["peak_bytes"] for s in r["process_samples"]) for r in rs)
        current=max(r["current_persistent_bytes"]+r["current_read_transient_bytes"] for r in rs)
        candidate=max(r["candidate_persistent_bytes"]+r["candidate_read_transient_bytes"] for r in rs)
        saving=current-candidate; saving_pct=saving/peak if peak else 0
        runtime=max(r["cache_read_duration_ms"]/r["generation_duration_ms"] for r in rs)
        reused=max(r["reused_requests"] for r in rs)
        opportunity=current>=THRESHOLDS["opportunity_bytes"] and current>=peak*THRESHOLDS["opportunity_peak_pct"] and reused>=THRESHOLDS["reused_requests"]
        eligible=opportunity and saving>=THRESHOLDS["saving_bytes"] and saving_pct>=THRESHOLDS["saving_peak_pct"]
        decisions[family]={"decision":"go" if eligible else "no-go","opportunity":opportunity,"current_whole_process_bytes":current,"candidate_whole_process_bytes":candidate,"net_saving_bytes":saving,"net_saving_peak_pct":saving_pct,"cache_read_runtime_fraction":runtime,"runtime_only_opportunity":runtime>=THRESHOLDS["runtime_only_pct"],"variants":sorted(variants),"self_attention_excluded":True}
    return {"schema":"sc-20686-cache-attribution-v2","thresholds":THRESHOLDS,"decisions":decisions,"rows":rows}
def main():
    p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("output",type=Path); p.add_argument("--sidecar",type=Path); a=p.parse_args()
    try:
        rows=json.loads(a.input.read_text(encoding="utf-8"))
        if a.sidecar:
            if not isinstance(rows,list) or len(rows)!=1: fail("sealed input must contain one row")
            verify_seal(rows[0],a.sidecar)
        result=reduce(rows)
    except (OSError,json.JSONDecodeError,ValueError) as e: print(f"SC-20686 invalid receipt: {e}",file=sys.stderr); return 1
    payload=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode(); a.output.write_bytes(payload); print(json.dumps({"sha256":hashlib.sha256(payload).hexdigest(),"output":str(a.output)})); return 0
if __name__=="__main__": raise SystemExit(main())
