if [[ -z "$QWEN_IMAGE_MLX_SNAPSHOT" ]]; then
  echo "Repository variable QWEN_IMAGE_MLX_SNAPSHOT is unset; this lane cannot resolve its snapshot path" >&2
  exit 1
fi
if [[ "$QWEN_IMAGE_MLX_SNAPSHOT" != /* ]]; then
  echo "QWEN_IMAGE_MLX_SNAPSHOT must be an absolute path, got: $QWEN_IMAGE_MLX_SNAPSHOT" >&2
  exit 1
fi
