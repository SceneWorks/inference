# A run-scoped Cargo target with the release profile unstripped (rust-lang/rust#157750 — the box's
# shared target served E0463 for mis-stripped macOS proc-macro dylibs); nothing shared is cleaned.
# Phase `device` needs only the crate's lib unit-test binary (`veda::device_gate::`), `ab` only the
# `integration` binary (`veda_ab::veda_ab_render_one_cell`); `all` builds both.
target="$RUNNER_TEMP/veda-mlx-cargo-target-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$target"
export CARGO_TARGET_DIR="$target"
export CARGO_PROFILE_RELEASE_STRIP=none
export CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_STRIP=none
echo "CARGO_TARGET_DIR=$target" >> "$GITHUB_ENV"
build() { # $1 cargo target selector, $2 output file, $3 extraction predicate
  cargo test --locked --release -p mlx-gen-minimax-h3 $1 --no-run
  cargo test --locked --release -p mlx-gen-minimax-h3 $1 --no-run --message-format=json \
    | VEDA_PICK="$3" python3.12 -c 'import json,os,sys
pick = os.environ["VEDA_PICK"]
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") != "compiler-artifact" or not m.get("executable") or not m["profile"]["test"]:
        continue
    t = m["target"]
    if (pick == "lib" and any(k.endswith("lib") for k in t["kind"])) or (pick == "integration" and t["name"] == "integration"):
        print(m["executable"])' > "$2"
  test "$(wc -l < "$2")" -eq 1
  cat "$2"
}
if [[ "$VEDA_MLX_PHASE" != "ab" ]]; then
  build --lib "$RUNNER_TEMP/veda-mlx-lib-test-binary" lib
fi
if [[ "$VEDA_MLX_PHASE" != "device" ]]; then
  build "--test integration" "$RUNNER_TEMP/veda-mlx-integration-test-binary" integration
fi
