GGUF=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b
BASELINE=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
# A preserved copy is bound when present. A missing one keeps the variable's runner-cache path, where
# provisioning downloads it; provisioning refuses to fetch into the preserved root itself.
if test -d "$GGUF"; then
  printf 'BONSAI_GGUF_SNAPSHOT=%s\n' "$GGUF" >> "$GITHUB_ENV"
else
  echo "::notice::SC-23935 preserved snapshot $GGUF is absent; BONSAI_GGUF_SNAPSHOT stays $BONSAI_GGUF_SNAPSHOT"
fi
if test -d "$BASELINE"; then
  printf 'BONSAI_BASELINE_SNAPSHOT=%s\n' "$BASELINE" >> "$GITHUB_ENV"
else
  echo "::notice::SC-23935 preserved snapshot $BASELINE is absent; BONSAI_BASELINE_SNAPSHOT stays $BONSAI_BASELINE_SNAPSHOT"
fi
