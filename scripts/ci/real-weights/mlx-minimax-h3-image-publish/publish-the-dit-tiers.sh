export HF_HUB_DISABLE_XET=1
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/ci/publish_hf_tiers.py \
  --repo SceneWorks/minimax-h3-image-mlx \
  --map "$PUBLISH_SCRATCH/dit/q8=q8" --map "$PUBLISH_SCRATCH/dit/q4=q4" \
  --message "SW-7104: MiniMax-H3-Image DiT q8/q4 tiers (inference $GITHUB_SHA)" \
  --report "$PUBLISH_REPORTS/minimax-h3-image-mlx.json"
rm -rf "$PUBLISH_SCRATCH/dit"
