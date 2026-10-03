python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# WHERE THE BYTES LAND (run 37121517158). On nax-macos-2 `~/.cache/huggingface/hub` resolves onto
# /Volumes/Models, a separate volume that had ~3.6 GB free, while `df $HOME` (what the shared
# headroom step prints) reported the 1 TiB Data volume -- so the fetch hit ENOSPC 75 s in with the
# headroom record looking healthy. Print the RESOLVED location of every path the fetch writes to.
# If the shared hub cannot hold this lane's ~66 GB, materialize into a cache-shaped hub under this
# job's own RUNNER_TEMP (wiped by the runner after the job) rather than touching the shared cache.
required_kib=$((70 * 1024 * 1024))
for p in "$HOME/.cache" "$HOME/.cache/huggingface" "$MLX_GEN_MODELS_ROOT" "$HOME/.cache/huggingface/xet" "${HF_HOME:-}" "${HF_XET_CACHE:-}" "${TMPDIR:-}" "$RUNNER_TEMP"; do
  [[ -n "$p" && -e "$p" ]] || continue
  real="$(cd "$p" 2>/dev/null && pwd -P)"
  echo "path: $p -> ${real:-?} (device $(stat -f %Sd "$p" 2>/dev/null || echo ?))"
  df -h "${real:-$p}" 2>/dev/null | tail -n 1
done
hub_real="$(cd "$MLX_GEN_MODELS_ROOT" 2>/dev/null && pwd -P || echo "$MLX_GEN_MODELS_ROOT")"
hub_free_kib="$(df -k "$hub_real" 2>/dev/null | awk 'NR == 2 { print $4 }')"
if [[ -z "$hub_free_kib" || "$hub_free_kib" -lt "$required_kib" ]]; then
  lane_hub="$RUNNER_TEMP/qwen-image-2-1-hub"
  echo "::warning::shared hub $MLX_GEN_MODELS_ROOT -> $hub_real has ${hub_free_kib:-unknown} KiB free (< 70 GiB); materializing into job-temp $lane_hub instead"
  mkdir -p "$lane_hub" "$RUNNER_TEMP/hf-xet"
  export HF_XET_CACHE="$RUNNER_TEMP/hf-xet"
  for v in MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT; do
    tail_path="${!v#"$MLX_GEN_MODELS_ROOT"/}"
    printf -v "$v" '%s' "$lane_hub/$tail_path"
    export "$v"
    echo "$v=${!v}" >> "$GITHUB_ENV"
  done
  MLX_GEN_MODELS_ROOT="$lane_hub"
  export MLX_GEN_MODELS_ROOT
  echo "MLX_GEN_MODELS_ROOT=$lane_hub" >> "$GITHUB_ENV"
  echo "HF_XET_CACHE=$HF_XET_CACHE" >> "$GITHUB_ENV"
  echo "job-temp hub $lane_hub (shared hub $hub_real had ${hub_free_kib:-?} KiB free)" > "$QWEN_IMAGE_2_1_RENDER_OUT/snapshot-location.txt"
fi
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen-image-2-1 --snapshot "$MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen-image-2-1-mlx-tiers --snapshot "$MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT"
# The optional third-party adapter (dispatch input `qwen_image_2_1_third_party_lora`). Pinned to a
# full commit sha and a .safetensors path, fetched THROUGH the persistent hub cache like the
# snapshots above, and exported only when it materialized. Empty input: nothing to do.
spec="${QWEN_IMAGE_2_1_THIRD_PARTY_LORA_SPEC:-}"
if [[ -n "$spec" ]]; then
  pattern='^([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)@([0-9a-f]{40}):([A-Za-z0-9._/-]+\.safetensors)(:(lora|lokr))?$'
  if [[ ! "$spec" =~ $pattern ]]; then
    echo "qwen_image_2_1_third_party_lora must be owner/repo@<40-hex sha>:<path>.safetensors[:lora|:lokr], got: $spec" >&2
    exit 1
  fi
  repo="${BASH_REMATCH[1]}"
  rev="${BASH_REMATCH[2]}"
  file="${BASH_REMATCH[3]}"
  kind="${BASH_REMATCH[5]:-lora}"
  if [[ "$file" == *..* || "$file" == /* ]]; then
    echo "qwen_image_2_1_third_party_lora: the file path must stay inside the repository, got: $file" >&2
    exit 1
  fi
  path="$(PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 -c 'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], cache_dir=sys.argv[4]))' "$repo" "$rev" "$file" "$MLX_GEN_MODELS_ROOT")"
  if [[ ! -f "$path" ]]; then
    echo "the third-party adapter did not materialize: $repo@$rev:$file -> '$path'" >&2
    exit 1
  fi
  echo "QWEN_IMAGE_2_1_THIRD_PARTY_LORA=$path" >> "$GITHUB_ENV"
  echo "QWEN_IMAGE_2_1_THIRD_PARTY_LORA_KIND=$kind" >> "$GITHUB_ENV"
  echo "$repo@$rev:$file ($kind)" > "$QWEN_IMAGE_2_1_RENDER_OUT/third-party-adapter.txt"
fi
