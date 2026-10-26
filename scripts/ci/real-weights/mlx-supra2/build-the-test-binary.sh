# A run-scoped Cargo target with the release profile unstripped (rust-lang/rust#157750 — the box's
# shared target served E0463 for mis-stripped macOS proc-macro dylibs); nothing shared is cleaned.
# The converter step builds into the same target.
target="$RUNNER_TEMP/supra2-mlx-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
{
  echo "CARGO_TARGET_DIR=$target"
  echo "CARGO_PROFILE_RELEASE_STRIP=none"
  echo "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none"
} >> "$GITHUB_ENV"
# Prebuilt libmlx (sc-21382): the fork's `prebuilt-<sha12>` Release cell for the mlx-rs rev pinned
# in Cargo.lock, exporting BOTH PMETAL_MLX_PREBUILT_DIR and PMETAL_METALLIB_PATH here and, through
# GITHUB_ENV, for the run step (the test binary has no compiled-in metallib). Exit 1 = nothing
# published for this rev: pmetal-mlx-sys then builds libmlx from source, which needs the Xcode
# Metal Toolchain component — checked up front so a box without it fails with the fix named
# instead of deep inside cmake. Any other exit is a corrupt/mislabeled asset and is fatal.
rc=0
prebuilt="$(scripts/fetch-prebuilt-mlx.sh --build-type Release --github-env)" || rc=$?
if (( rc == 0 )); then
  eval "$prebuilt"
  export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
  echo "$prebuilt" | tee "$SUPRA2_OUT/prebuilt-mlx.txt"
elif (( rc == 1 )); then
  echo "::warning title=prebuilt MLX unavailable::no prebuilt for the pinned mlx-rs rev; building libmlx from source"
  if ! xcrun -sdk macosx metal --version > "$SUPRA2_OUT/metal-version.txt" 2>&1; then
    cat "$SUPRA2_OUT/metal-version.txt"
    echo "::error::$RUNNER_NAME has no Metal Toolchain and no prebuilt libmlx is published for the pinned mlx-rs rev: install it on the runner (xcodebuild -downloadComponent MetalToolchain) or publish the fork's prebuilt-mlx release for the pinned rev"
    exit 1
  fi
else
  exit "$rc"
fi
cargo test --locked --release -p mlx-gen-supra2 --test integration --no-run
cargo test --locked --release -p mlx-gen-supra2 --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/supra2-mlx-test-binary"
test "$(wc -l < "$RUNNER_TEMP/supra2-mlx-test-binary")" -eq 1
cat "$RUNNER_TEMP/supra2-mlx-test-binary"
