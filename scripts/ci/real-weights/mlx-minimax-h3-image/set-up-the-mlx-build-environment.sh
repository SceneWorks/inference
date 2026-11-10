# SW-7089 — the build environment the other per-run-building MLX lanes use (`mlx-supra2`,
# `mlx-minimax-h3-image-pe`): a run-scoped Cargo target with the release profile unstripped
# (rust-lang/rust#157750 — the box's shared target served E0463 for mis-stripped macOS proc-macro
# dylibs), and the fork's prebuilt libmlx (sc-21382), exported through GITHUB_ENV for the gates step
# (the test binary has no compiled-in metallib). Exit 1 from the fetch = nothing published for the
# pinned mlx-rs rev: libmlx then builds from source, which needs the Xcode Metal Toolchain.
target="$RUNNER_TEMP/minimax-h3-image-mlx-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
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
  if ! xcrun -sdk macosx metal --version; then
    echo "::error::$RUNNER_NAME has no Metal Toolchain and no prebuilt libmlx is published for the pinned mlx-rs rev"
    exit 1
  fi
else
  exit "$rc"
fi
