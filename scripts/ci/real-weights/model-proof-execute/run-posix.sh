if [[ "$RUNNER_OS" == macOS ]]; then PYTHON=python3.12; else PYTHON="$REVIEWED_PYTHON"; fi
"$PYTHON" -I scripts/ci/model_inference_proofs.py \
  --backend "$MODEL_PROOF_BACKEND" --request-env MODEL_PROOF_REQUEST \
  --expected-sha "$GITHUB_SHA" --output "$MODEL_PROOF_EVIDENCE"
