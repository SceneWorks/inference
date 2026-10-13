# A run-scoped Cargo target (the box's shared target has served E0463 for mis-stripped macOS
# proc-macro dylibs, rust-lang/rust#157750): nothing shared is read or cleaned, and the release
# profile — proc-macros and build scripts included — stays unstripped.
target="$RUNNER_TEMP/iris-curricula-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
echo "CARGO_TARGET_DIR=$target" >> "$GITHUB_ENV"
# The plain build prints any compiler error as usual; the JSON pass (no-op once built) names the
# integration binary the guarded step runs.
cargo test --locked --release -p mlx-gen-iris --test integration --no-run
cargo test --locked --release -p mlx-gen-iris --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/iris-curricula-test-binary"
test -s "$RUNNER_TEMP/iris-curricula-test-binary"
cat "$RUNNER_TEMP/iris-curricula-test-binary"
