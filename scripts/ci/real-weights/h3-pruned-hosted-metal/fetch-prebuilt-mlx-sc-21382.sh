set +e
scripts/fetch-prebuilt-mlx.sh --github-env
rc=$?
set -e
if [ "$rc" = 1 ]; then
  echo "::warning title=prebuilt MLX unavailable::no prebuilt for the pinned mlx-rs rev; building libmlx from source (see step log)"
elif [ "$rc" != 0 ]; then
  exit "$rc"
fi
