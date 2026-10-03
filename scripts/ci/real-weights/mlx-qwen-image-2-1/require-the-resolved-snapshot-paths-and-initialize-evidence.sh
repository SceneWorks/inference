for v in MLX_GEN_MODELS_ROOT MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT; do
  if [[ -z "${!v}" ]]; then
    echo "$v is unset (repository variable MLX_GEN_MODELS_ROOT); this lane cannot resolve its snapshot paths" >&2
    exit 1
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path after resolution, got: ${!v}" >&2
    exit 1
  fi
done
evidence="$RUNNER_TEMP/qwen-image-2-1-mlx-evidence"
rm -rf "$evidence"
mkdir -p "$evidence"
echo "QWEN_IMAGE_2_1_RENDER_OUT=$evidence" >> "$GITHUB_ENV"
