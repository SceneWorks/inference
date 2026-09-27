for v in FLUX_DEV_DIR QWEN_IMAGE_MLX_SNAPSHOT PID_QWEN_SNAPSHOT PID_FLUX_SNAPSHOT PID_GEMMA_SNAPSHOT; do
  if [[ -z "${!v}" ]]; then
    echo "Repository variable behind $v is unset; this lane cannot resolve its snapshot path" >&2
    exit 1
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path, got: ${!v}" >&2
    exit 1
  fi
done
