# A run-scoped Cargo target with the proc-macro/build-script override profile unstripped
# (rust-lang/rust#157750 — the box's shared target served E0463 for mis-stripped macOS proc-macro
# dylibs); nothing shared is cleaned. The plain build prints any compiler error as usual; the JSON
# pass (no-op once built) names the integration binary the guarded step runs.
target="$RUNNER_TEMP/flux3-klein-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
echo "CARGO_TARGET_DIR=$target" >> "$GITHUB_ENV"
cargo test --locked --release -p mlx-gen-flux3 --test integration --no-run
cargo test --locked --release -p mlx-gen-flux3 --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/flux3-klein-test-binary"
test -s "$RUNNER_TEMP/flux3-klein-test-binary"
cat "$RUNNER_TEMP/flux3-klein-test-binary"
