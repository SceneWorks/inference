# SW-7089 — AC1 parity vs diffusers and AC2 16 MP bounded-decode peak, MLX. Name-selected with
# `--exact` after the `--`, built with the registered `model-inference-proof` feature and run with
# MODEL_INFERENCE_CASE_FQN set; verify_model_inference_log.py then applies the proof runner's own
# check (one passing summary, one BODY, the actual device set), so a rename cannot go vacuously
# green. Two processes, so the parity run's buffers are reaped before the 16 MP peaks are measured.
set -o pipefail
logs="$RUNNER_TEMP/minimax-h3-image-gate-logs"
mkdir -p "$logs"
for name in image_vae_decode_matches_the_official_diffusers_vae image_vae_tiled_decode_lowers_the_16mp_peak; do
  fqn=real_weights::"$name"
  status=0
  MODEL_INFERENCE_CASE_FQN="$fqn" cargo test --locked --release -p mlx-gen-minimax-h3 \
    --features model-inference-proof --test integration \
    "$fqn" -- --exact --ignored --nocapture 2>&1 | tee "$logs/$name.log" >&2 || status=$?
  if ((status)) || ! python3.12 -I scripts/ci/verify_model_inference_log.py --package mlx-gen-minimax-h3 \
    --backend mlx --fqn "$fqn" --log "$logs/$name.log"; then
    echo "::error::'$name' failed or lacks its registered proof evidence (exit $status)" >&2
    exit 1
  fi
done
