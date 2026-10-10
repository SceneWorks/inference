python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# The lane binds a task-owned persistent cache before this step. Record the actual filesystem.
hub_real="$(cd "$MLX_GEN_MODELS_ROOT" && pwd -P)"
df -h "$hub_real"
echo "persistent hub $MLX_GEN_MODELS_ROOT -> $hub_real" > "$QWEN_IMAGE_2_1_RENDER_OUT/snapshot-location.txt"
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
