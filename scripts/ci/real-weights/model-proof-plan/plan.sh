"$REVIEWED_PYTHON" -I scripts/ci/model_inference_proofs.py --plan \
  --binding-env MODEL_PROOF_BINDING \
  --expected-sha "$GITHUB_SHA" > "$RUNNER_TEMP/model-proof-plan.json"
"$REVIEWED_PYTHON" -I -c 'import json,os; p=json.load(open(os.path.join(os.environ["RUNNER_TEMP"],"model-proof-plan.json"),encoding="utf-8")); matrix={"include":[{"runner":p["runner"],"asset_setup":p["asset_setup"]}]}; open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8").write("matrix="+json.dumps(matrix,separators=(",",":"))+"\nbackend="+p["backend"]+"\nrequest="+json.dumps(p["request"],separators=(",",":"))+"\n")'
