# The image snapshot's DiT and fused heads, into the internal task-owned hub (never the app cache).
export HF_HUB_DISABLE_XET=1
image="$LANE_ROOT/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/492fd8c4eca666f1085f57e30738c1e0d86b3439"
for model in minimax-h3-image minimax-h3-image-dit; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$model" --snapshot "$image" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::$model materialization attempt $attempt failed"
    sleep 30
  done
done
echo "MINIMAX_H3_IMAGE_SNAPSHOT=$image" >> "$GITHUB_ENV"
