GGUF=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b
BASELINE=/Volumes/Models/Codex-models/sc-23935/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
# A preserved copy is bound only when it holds every file of its frozen publisher closure. A missing
# or partial one keeps the variable's runner-cache path, where provisioning downloads what is
# missing; provisioning never fetches into the preserved root itself. Hashes are verified later.
preserved_complete() {
  python3.12 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/release")
from qwen38_bonsai_assets import frozen_model
from verify_model_snapshot import load_model

model = load_model(Path("release/real-weight-models.toml"), sys.argv[1])
root = Path(sys.argv[2])
missing = [item["path"] for item in frozen_model(model)["files"] if not (root / item["path"]).is_file()]
if missing:
    print(f"{len(missing)} closure file(s) missing, first {missing[0]}")
    sys.exit(3)
PY
}
bind_preserved() {
  local key="$1" variable="$2" preserved="$3" status
  if ! test -d "$preserved"; then
    echo "::notice::SC-23935 preserved snapshot $preserved is absent; $variable stays ${!variable}"
    return 0
  fi
  if status="$(preserved_complete "$key" "$preserved")"; then
    printf '%s=%s\n' "$variable" "$preserved" >> "$GITHUB_ENV"
  elif [[ $? -eq 3 ]]; then
    echo "::notice::SC-23935 preserved snapshot $preserved is partial ($status); $variable stays ${!variable}"
  else
    echo "::error::could not check SC-23935 preserved snapshot $preserved" >&2
    return 1
  fi
}
bind_preserved bonsai-gguf BONSAI_GGUF_SNAPSHOT "$GGUF"
bind_preserved bonsai-qwen3vl-baseline BONSAI_BASELINE_SNAPSHOT "$BASELINE"
