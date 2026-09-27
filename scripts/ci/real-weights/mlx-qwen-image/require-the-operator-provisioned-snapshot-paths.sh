for v in QWEN_IMAGE_MLX_SNAPSHOT QWEN_IMAGE_EDIT_MLX_SNAPSHOT QWEN_CONTROL_UNION_SNAPSHOT MLX_GEN_MODELS_ROOT; do
  if [[ -z "${!v}" ]]; then
    echo "Repository variable $v is unset; this lane cannot resolve its snapshot path" >&2
    exit 1
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path, got: ${!v}" >&2
    exit 1
  fi
done
