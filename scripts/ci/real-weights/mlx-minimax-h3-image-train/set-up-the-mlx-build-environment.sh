target="$RUNNER_TEMP/minimax-h3-image-train-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
{
  echo "CARGO_TARGET_DIR=$target"
  echo "CARGO_PROFILE_RELEASE_STRIP=none"
  echo "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none"
} >> "$GITHUB_ENV"
rc=0
prebuilt="$(scripts/fetch-prebuilt-mlx.sh --build-type Release --github-env)" || rc=$?
if (( rc == 0 )); then
  echo "$prebuilt"
elif (( rc == 1 )); then
  echo "::warning title=prebuilt MLX unavailable::no prebuilt for the pinned mlx-rs rev; building libmlx from source"
  xcrun -sdk macosx metal --version
else
  exit "$rc"
fi
