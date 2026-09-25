GGUF=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b
BASELINE=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
test -d "$GGUF"
test -d "$BASELINE"
printf 'BONSAI_GGUF_SNAPSHOT=%s\nBONSAI_BASELINE_SNAPSHOT=%s\n' "$GGUF" "$BASELINE" >> "$GITHUB_ENV"
