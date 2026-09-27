for v in KREA_REALTIME_SNAPSHOT KREA_STYLE_LORA_SNAPSHOT KREA_DISTILL_LORA_SNAPSHOT; do
  if [[ -z "${!v}" ]]; then
    echo "Repository variable $v is unset; this lane cannot resolve its snapshot path" >&2
    exit 1
  fi
  if [[ "${!v}" != /* ]]; then
    echo "$v must be an absolute path, got: ${!v}" >&2
    exit 1
  fi
done
