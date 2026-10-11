"""Run the installed native SeedVR2 comparator serially after H3 measurements."""
import argparse
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location("acceptance_run",Path(__file__).with_name("run.py"))
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser()
    for name in ("binary","root","evidence"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu-uuid",required=True)
    args=parser.parse_args()
    report_path=args.evidence/"report.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if len(report["cases"])!=3 or any(len(case["variants"])!=3 or any(v["exit_code"] for v in case["variants"]) for case in report["cases"]):
        raise SystemExit("complete the H3 campaign before starting the serial comparator")
    assets={path.name:runner.sha(path) for path in sorted(args.root.glob("*.safetensors"))}
    for case in report["cases"]:
        folder=args.evidence/case["kind"]
        rgb=folder/"seedvr2.rgb"
        receipt=runner.execute(args,[args.binary,"--root",args.root,"--rgb24",folder/"source.rgb","--out",rgb],rgb)
        receipt["installed_asset_sha256"]=assets
        case["seedvr2"]=receipt
        if receipt["exit_code"]==0:
            output=folder/"seedvr2.mp4"
            runner.mux(rgb,folder/"source.mp4",output)
            receipt.update({"result":str(output),"result_sha256":runner.sha(output),"result_probe":runner.probe(output)})
        report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        if receipt["exit_code"]:
            raise SystemExit("native SeedVR2 failed; comparator readout remains incomplete")
    report["missing"]=[v for v in report["missing"] if v!="SeedVR2 side-by-side"]
    report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
