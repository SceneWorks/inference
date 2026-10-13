# A run-scoped Cargo target with the release profile unstripped (rust-lang/rust#157750 — the box's
# shared target served E0463 for mis-stripped macOS proc-macro dylibs); nothing shared is cleaned.
target="$RUNNER_TEMP/iris-memory-probe-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
echo "CARGO_TARGET_DIR=$target" >> "$GITHUB_ENV"
cargo test --locked --release -p mlx-gen-iris --test integration --no-run
cargo test --locked --release -p mlx-gen-iris --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/iris-memory-probe-test-binary"
test -s "$RUNNER_TEMP/iris-memory-probe-test-binary"
cat "$RUNNER_TEMP/iris-memory-probe-test-binary"
