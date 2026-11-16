# A run-scoped Cargo target with the release profile unstripped (rust-lang/rust#157750); nothing
# shared is cleaned. The converters build into the same target.
target="$RUNNER_TEMP/minimax-pe-mlx-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
{
  echo "CARGO_TARGET_DIR=$target"
  echo "CARGO_PROFILE_RELEASE_STRIP=none"
  echo "CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none"
} >> "$GITHUB_ENV"
# Prebuilt libmlx (sc-21382), exactly as `mlx-supra2`: exit 1 = nothing published for the pinned
# mlx-rs rev (build from source, which needs the Metal Toolchain); any other failure is fatal.
rc=0
prebuilt="$(scripts/fetch-prebuilt-mlx.sh --build-type Release --github-env)" || rc=$?
if (( rc == 0 )); then
  eval "$prebuilt"
  export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
  echo "$prebuilt" | tee "$MINIMAX_PE_OUT/prebuilt-mlx.txt"
elif (( rc == 1 )); then
  echo "::warning title=prebuilt MLX unavailable::no prebuilt for the pinned mlx-rs rev; building libmlx from source"
  if ! xcrun -sdk macosx metal --version > "$MINIMAX_PE_OUT/metal-version.txt" 2>&1; then
    cat "$MINIMAX_PE_OUT/metal-version.txt"
    echo "::error::$RUNNER_NAME has no Metal Toolchain and no prebuilt libmlx is published for the pinned mlx-rs rev"
    exit 1
  fi
else
  exit "$rc"
fi
cargo build --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16
cargo build --locked --release -p mlx-llm --example prepare_snapshot
cargo test --locked --release -p mlx-llm --test integration --no-run
cargo test --locked --release -p mlx-llm --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/minimax-pe-mlx-test-binary"
test "$(wc -l < "$RUNNER_TEMP/minimax-pe-mlx-test-binary")" -eq 1
cat "$RUNNER_TEMP/minimax-pe-mlx-test-binary"
