# A run-scoped Cargo target: the box's shared `../cargo-target` has served E0463 ("can't find crate
# for `rustversion`") to this epic's lanes on 2026-10-11 — a proc-macro dylib mis-stripped by Rust
# 1.96 / LLVM 22 on macOS (rust-lang/rust#157750). Proc-macros and build scripts build under the
# build-override profile, so leaving only THEM unstripped is the narrow form of the workaround (the
# test binary keeps the release profile's own settings); nothing shared is cleaned.
target="$RUNNER_TEMP/iris-restore-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
# The plain build prints any compiler error as usual; the JSON pass (no-op once built) names the
# integration binary the guarded step runs.
cargo test --locked --release -p mlx-gen-iris --test integration --no-run
cargo test --locked --release -p mlx-gen-iris --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])' > "$RUNNER_TEMP/iris-restore-test-binary"
test -s "$RUNNER_TEMP/iris-restore-test-binary"
cat "$RUNNER_TEMP/iris-restore-test-binary"
