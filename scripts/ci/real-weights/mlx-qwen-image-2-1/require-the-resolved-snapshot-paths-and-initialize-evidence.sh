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
# Task-owned persistent hub on the Data volume. This scope is only this dispatch profile;
# never mutate the shared app cache or rely on runner-temp weights surviving a reboot.
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub" "$lane_root/xet"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
for v in MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT; do
  tail_path="${!v#"$MLX_GEN_MODELS_ROOT"/}"
  if [[ "$tail_path" == "${!v}" || "$tail_path" == *..* ]]; then
    echo "snapshot must stay beneath resolved hub: $v=${!v}" >&2
    exit 1
  fi
  printf -v "$v" '%s' "$lane_hub/$tail_path"
  export "$v"
  echo "$v=${!v}" >> "$GITHUB_ENV"
done
export MLX_GEN_MODELS_ROOT="$lane_hub"
export HF_XET_CACHE="$HOME/sceneworks-rw-weights/xet"
echo "MLX_GEN_MODELS_ROOT=$MLX_GEN_MODELS_ROOT" >> "$GITHUB_ENV"
echo "HF_XET_CACHE=$HF_XET_CACHE" >> "$GITHUB_ENV"
evidence="$RUNNER_TEMP/qwen-image-2-1-mlx-evidence"
rm -rf "$evidence"
mkdir -p "$evidence"
echo "QWEN_IMAGE_2_1_RENDER_OUT=$evidence" >> "$GITHUB_ENV"
